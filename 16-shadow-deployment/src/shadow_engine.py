"""Último eslabón visible de la cadena de remediación por drift: 12 detecta
y dispara, 13 ingiere, 14 entrena el modelo sombra, 15 decide si se
promueve, 16 lo registra y lo corre en paralelo al modelo Champion (el de
producción) sin reemplazarlo -- inferencia dual, las dos predicciones
quedan logueadas para poder comparar antes de decidir un swap real (esa
decisión sigue siendo humana, fuera de este repo).

`active_shadow_model.pkl` es siempre el Challenger, nunca el Champion --
nomenclatura tal como la fija el Día 20 del plan. El Champion no tiene un
artefacto propio en ningún punto anterior de este portafolio (esta técnica
es la primera que necesita "el modelo que ya está sirviendo"), así que
`predict_dual` lo recibe como parámetro en vez de asumir dónde vive.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_DIR = "outputs/registry"
ACTIVE_MODEL_FILENAME = "active_shadow_model.pkl"
REGISTRY_MANIFEST_FILENAME = "registry_manifest.json"
LOG_TABLE_NAME = "dual_inference_logs"
ID_COLUMN = "client_id"


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _cargar_modelo(modelo: Any):
    """Acepta un modelo ya cargado (tiene `.predict_proba`, típico en tests
    -- no hace falta escribir un .pkl a disco solo para probar) o una ruta a
    un .pkl (el caso real de la CLI, que lee del filesystem)."""
    if hasattr(modelo, "predict_proba"):
        return modelo
    with open(modelo, "rb") as f:
        return pickle.load(f)


class ShadowDeploymentEngine:
    def __init__(self, registry_dir: str | Path = DEFAULT_REGISTRY_DIR):
        self.registry_dir = Path(registry_dir)

    def register_promoted_model(self, decision_path, models_dir, registry_dir=None) -> dict:
        """Lee `promotion_decision_<timestamp>.json`. En PROMOTED, copia
        `models_dir/<candidate_model>` a `registry_dir/active_shadow_model.pkl`
        y escribe `registry_manifest.json`. En REJECTED, no toca nada del
        modelo sombra activo -- solo lo deja registrado en el log."""
        if registry_dir is not None:
            self.registry_dir = Path(registry_dir)
        models_dir = Path(models_dir)

        decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))

        if decision.get("decision") != "PROMOTED":
            logger.info(
                "Decision de promocion '%s' (%s) -- el modelo sombra activo no cambia.",
                decision.get("decision"), decision.get("reason"),
            )
            return {"registered": False, "decision": decision.get("decision")}

        candidato = models_dir / decision["candidate_model"]
        if not candidato.exists():
            raise FileNotFoundError(
                f"'{decision_path}' dice candidate_model='{decision['candidate_model']}' "
                f"pero no encuentro '{candidato}'"
            )

        self.registry_dir.mkdir(parents=True, exist_ok=True)
        destino = self.registry_dir / ACTIVE_MODEL_FILENAME
        shutil.copy2(candidato, destino)

        manifest = {
            "active_version": decision["candidate_model"],
            "activated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "source_decision": Path(decision_path).name,
        }
        manifest_path = self.registry_dir / REGISTRY_MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "Modelo sombra activado: %s -> %s (manifiesto: %s)",
            decision["candidate_model"], destino, manifest_path,
        )
        return {"registered": True, "active_model_path": str(destino), "manifest_path": str(manifest_path)}

    def predict_dual(self, champion_model: Any, input_df: pd.DataFrame) -> pd.DataFrame:
        """Predicción de probabilidad de default con Champion y (si existe)
        Challenger, fila por fila. Sin Challenger activo, `pred_challenger`
        y `diff_abs` quedan en `None` -- sigue respondiendo con Champion
        solo, nunca lanza una excepción por la ausencia del sombra."""
        champion = _cargar_modelo(champion_model)

        challenger_path = self.registry_dir / ACTIVE_MODEL_FILENAME
        challenger = _cargar_modelo(challenger_path) if challenger_path.exists() else None
        if challenger is None:
            logger.info("Sin modelo sombra activo en '%s' -- respondiendo solo con Champion.", self.registry_dir)

        X = input_df.drop(columns=[ID_COLUMN], errors="ignore")

        pred_champion = champion.predict_proba(X)[:, 1]
        pred_challenger = challenger.predict_proba(X)[:, 1] if challenger is not None else None

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        resultado = pd.DataFrame({
            ID_COLUMN: input_df[ID_COLUMN].values if ID_COLUMN in input_df.columns
            else range(len(input_df)),
            "pred_champion": pred_champion,
            "pred_challenger": pred_challenger if pred_challenger is not None else pd.NA,
            "timestamp": timestamp,
        })
        resultado["diff_abs"] = (
            (resultado["pred_champion"] - resultado["pred_challenger"]).abs()
            if pred_challenger is not None else pd.NA
        )
        return resultado[[ID_COLUMN, "pred_champion", "pred_challenger", "diff_abs", "timestamp"]]

    def log_dual_predictions(self, predictions_df: pd.DataFrame, db_path) -> int:
        """Log auditable, no un estado que se upsertea: cada llamada agrega
        filas nuevas a `dual_inference_logs`, aunque el mismo client_id ya
        haya sido puntuado antes -- a diferencia de `credit_features`
        (técnica 13), acá el punto es preservar el historial completo de
        cada corrida de inferencia, no solo el valor más reciente."""
        db_path = Path(db_path)
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect(str(db_path))
        try:
            con.register("_staging", predictions_df)
            existe = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [LOG_TABLE_NAME],
            ).fetchone()[0] > 0
            if not existe:
                con.execute(f"CREATE TABLE {LOG_TABLE_NAME} AS SELECT * FROM _staging WHERE 1=0")
            con.execute(f"INSERT INTO {LOG_TABLE_NAME} SELECT * FROM _staging")
        finally:
            con.unregister("_staging")
            con.close()

        return len(predictions_df)
