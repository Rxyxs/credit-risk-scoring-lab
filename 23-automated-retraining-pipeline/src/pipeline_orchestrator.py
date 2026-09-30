"""Cierra el circuito: `22-automated-retraining-trigger` decide que hace
falta reentrenar y deja escrita esa decision; esta tecnica la lee, entrena
un nuevo candidato Challenger sobre la version mas fresca del Feature
Store (`13-feature-store-duckdb`), y registra que el disparador fue
atendido. No promueve nada por si sola -- el nuevo `shadow_model_*.pkl`
queda exactamente en la misma forma que deja
`14-shadow-model-training/src/train_shadow.py`, listo para que
`15-model-promotion` lo evalue de nuevo como si fuera cualquier otro
candidato. Eso es lo que hace esto un circuito *cerrado* y no una rama
aparte: el reentrenamiento automatico entra al mismo pipeline de
promocion que ya existia, no inventa uno paralelo.

Dos cosas separadas deliberadamente:
- `check_pending_triggers` solo lee -- nunca entrena nada, y nunca marca
  nada como procesado por si sola. Es seguro llamarla en un loop de
  polling sin efectos secundarios.
- `run_retraining_flow` es el unico metodo que muta estado: entrena,
  guarda artefactos, escribe en DuckDB, y recien ahi marca el disparador
  pendiente como atendido -- si algo falla antes de terminar, el
  disparador sigue pendiente para el proximo intento en vez de perderse.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
import uuid
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_STORE_TABLE = "credit_features"
ID_COLUMN = "client_id"
# Mismo orden de preferencia que `14-shadow-model-training/src/train_shadow.py`
# -- este pipeline reentrena sobre la misma tabla, asi que reconocer el
# target de otra forma solo generaria incompatibilidades silenciosas.
DEFAULT_TARGET_CANDIDATES = ("default_flag", "default", "target", "label")
MIN_ROWS_TO_TRAIN = 100

# Mas regularizacion que el LogisticRegression por defecto de sklearn
# (C=1.0) y que el de la tecnica 14: un reentrenamiento automatico
# responde a una señal de degradacion o de deriva sobre una ventana de
# datos recientemente distinta, y ese es exactamente el escenario donde
# sobreajustar a lo mas reciente sale caro. No es una busqueda de
# hiperparametros -- es un ajuste defensivo, documentado, en la unica
# direccion que tiene sentido para este caso de uso.
CHALLENGER_C_DEFAULT = 0.5

TABLA_EVENTOS = "model_lifecycle_events"
TIPO_EVENTO_EJECUTADO = "RETRAINING_EXECUTED"


class EmptyFeatureStoreError(RuntimeError):
    """El Feature Store no existe todavia, no tiene la tabla esperada, no
    llega al minimo de filas, o no tiene ninguna columna de target
    reconocible -- todos los motivos por los que `run_orchestrator.py`
    aborta con codigo 0 en vez de fallar."""


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"


class RetrainingPipelineOrchestrator:
    def __init__(self, state_path=None, challenger_c: float = CHALLENGER_C_DEFAULT,
                 random_state: int = 42):
        self.state_path = Path(state_path) if state_path is not None else None
        self.challenger_c = challenger_c
        self.random_state = random_state
        self.pending_trigger_path: Path | None = None
        self.metrics: dict | None = None
        self._procesados = self._cargar_procesados()

    def _cargar_procesados(self) -> set[str]:
        if self.state_path is not None and self.state_path.exists():
            try:
                return set(json.loads(self.state_path.read_text()))
            except json.JSONDecodeError:
                return set()
        return set()

    def _guardar_procesados(self) -> None:
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(sorted(self._procesados), indent=2))

    def check_pending_triggers(self, trigger_dir) -> bool:
        """`True` si hay al menos un `retraining_trigger_<timestamp>.json`
        con `trigger_activated == True` en `trigger_dir` que no este ya en
        el estado de procesados. Solo lee -- no muta nada, ni siquiera el
        estado local. El mas antiguo sin procesar queda en
        `self.pending_trigger_path` para que `run_retraining_flow` sepa
        cual marcar al terminar."""
        self.pending_trigger_path = None
        directorio = Path(trigger_dir)
        if not directorio.is_dir():
            return False

        for candidato in sorted(directorio.glob("retraining_trigger_*.json")):
            if candidato.name in self._procesados:
                continue
            try:
                payload = json.loads(candidato.read_text())
            except json.JSONDecodeError:
                logger.warning("'%s' no es JSON valido -- lo ignoro.", candidato)
                continue
            if payload.get("trigger_activated") is True:
                self.pending_trigger_path = candidato
                return True

        return False

    def _fetch_training_data(self, feature_store_path) -> tuple[pd.DataFrame, pd.Series]:
        es_en_memoria = str(feature_store_path) == ":memory:"
        if not es_en_memoria and not Path(feature_store_path).exists():
            raise EmptyFeatureStoreError(f"el feature store '{feature_store_path}' no existe")

        try:
            con = duckdb.connect(str(feature_store_path), read_only=not es_en_memoria)
        except duckdb.Error as exc:
            raise EmptyFeatureStoreError(
                f"no pude abrir el feature store '{feature_store_path}': {exc}") from exc

        try:
            existe = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                [FEATURE_STORE_TABLE],
            ).fetchone()[0] > 0
            if not existe:
                raise EmptyFeatureStoreError(
                    f"la tabla '{FEATURE_STORE_TABLE}' no existe en '{feature_store_path}'")
            df = con.execute(f"SELECT * FROM {FEATURE_STORE_TABLE}").fetchdf()
        finally:
            con.close()

        if len(df) < MIN_ROWS_TO_TRAIN:
            raise EmptyFeatureStoreError(
                f"'{FEATURE_STORE_TABLE}' tiene {len(df)} fila(s), menos de las "
                f"{MIN_ROWS_TO_TRAIN} minimas para reentrenar")

        columna_target = next((c for c in DEFAULT_TARGET_CANDIDATES if c in df.columns), None)
        if columna_target is None:
            raise EmptyFeatureStoreError(
                f"ninguna columna de target reconocible ({', '.join(DEFAULT_TARGET_CANDIDATES)}) "
                f"en '{FEATURE_STORE_TABLE}' -- columnas disponibles: {list(df.columns)}")

        y = df[columna_target]
        X = df.drop(columns=[c for c in (ID_COLUMN, columna_target) if c in df.columns])
        return X, y

    def _train_challenger(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        estratificar = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=estratificar)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=self.challenger_c, max_iter=1000,
                                        random_state=self.random_state)),
        ])
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_proba)) if y_test.nunique() > 1 else None

        self.metrics = {
            "roc_auc": roc_auc,
            "n_samples": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "challenger_c": self.challenger_c,
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        return pipeline

    def run_retraining_flow(self, feature_store_path, models_output_dir,
                            reports_output_dir, db_path) -> dict:
        """Extrae -> entrena -> evalua -> guarda `.pkl`/`.json` -> registra
        `RETRAINING_EXECUTED` -> marca el disparador pendiente como
        procesado, en ese orden. Si `check_pending_triggers` no se llamo
        antes (o no encontro nada), igual entrena: este metodo no impone
        la condicion de disparo, eso es responsabilidad de quien llama
        (ver `run_orchestrator.py`)."""
        X, y = self._fetch_training_data(feature_store_path)
        pipeline = self._train_challenger(X, y)

        models_output_dir = Path(models_output_dir)
        reports_output_dir = Path(reports_output_dir)
        models_output_dir.mkdir(parents=True, exist_ok=True)
        reports_output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = _timestamp()
        # Mismo prefijo que `14-shadow-model-training`: un candidato nacido
        # de un reentrenamiento automatico tiene que ser indistinguible,
        # por nombre de archivo, de uno entrenado a mano -- es lo que
        # permite que `15-model-promotion/run_promotion.py` lo recoja sin
        # cambios.
        model_path = models_output_dir / f"shadow_model_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        metrics_path = reports_output_dir / f"shadow_metrics_{timestamp}.json"
        metrics_path.write_text(json.dumps(self.metrics, indent=2))

        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        event_id = str(uuid.uuid4())
        promoted_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        con = duckdb.connect(str(db_path))
        try:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLA_EVENTOS} (
                    event_id VARCHAR,
                    event_type VARCHAR,
                    previous_champion VARCHAR,
                    new_champion VARCHAR,
                    promoted_at VARCHAR
                )
            """)
            con.execute(
                f"INSERT INTO {TABLA_EVENTOS} VALUES (?, ?, ?, ?, ?)",
                [event_id, TIPO_EVENTO_EJECUTADO, None, None, promoted_at],
            )
        finally:
            con.close()

        if self.pending_trigger_path is not None:
            self._procesados.add(self.pending_trigger_path.name)
            self._guardar_procesados()

        logger.info(
            "Challenger reentrenado: ROC-AUC=%s, n=%d -> %s (evento %s)",
            f"{self.metrics['roc_auc']:.4f}" if self.metrics["roc_auc"] is not None else "n/d",
            self.metrics["n_samples"], model_path, event_id,
        )

        return {
            "event_id": event_id,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "metrics": dict(self.metrics),
        }
