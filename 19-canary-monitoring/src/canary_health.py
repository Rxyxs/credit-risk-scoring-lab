"""El último guardia de la cadena de canary release: 18 enruta una fracción
del tráfico al Challenger, 19 mira esas predicciones recientes y, si algo
se ve mal (nulos/infinitos, un salto grande frente al Champion, o un pico
de rechazos), fuerza el rollback a 0% sin esperar a que alguien lo note.

No importa `CanaryRouter` de `18-canary-deployment/src/canary_router.py`
directamente: el nombre de esa carpeta ("18-canary-deployment") no es un
identificador de paquete Python válido (empieza con un dígito), y cada
técnica de este portafolio instala su propio `requirements.txt` en su
propio venv -- ninguna técnica anterior depende del paquete `src` de otra
por import directo, todas se integran por archivos (JSON, CSV, .pkl,
DuckDB) en disco. Reimplemento acá la escritura de `canary_config.json`
con el mismo esquema exacto que `trigger_rollback` de la técnica 18, no
como atajo sino porque es la forma consistente con el resto del portafolio.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANARY = "CANARY"
CHAMPION = "CHAMPION"
HIGH_RISK_THRESHOLD = 0.85
ROUTING_LOG_TABLE = "canary_routing_log"


class InsufficientCanaryTrafficError(RuntimeError):
    """No hay predicciones CANARY recientes para evaluar -- la base/tabla no
    existe, o `predictions_df` está vacío o no tiene ninguna fila CANARY."""


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _force_rollback(config_path, reason: str) -> dict:
    """Mismo esquema exacto que `CanaryRouter.trigger_rollback` (técnica 18)
    -- escribe al mismo `canary_config.json` que esa técnica lee, así que
    interopera de verdad aunque no comparta código Python."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "canary_percentage": 0,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rollback": True,
        "rollback_reason": reason,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.warning("ROLLBACK AUTOMATICO canario: trafico forzado a 0%% -- motivo: %s", reason)
    return config


class CanaryHealthMonitor:
    def __init__(self):
        self.metrics: dict | None = None
        self.decision: dict | None = None

    def fetch_recent_predictions(self, db_path, table_name: str = ROUTING_LOG_TABLE) -> pd.DataFrame:
        """Lee `canary_routing_log` (la técnica 18) de solo lectura.
        `read_only=True` se salta para ':memory:' -- mismo motivo que en las
        técnicas 14/17: DuckDB no permite abrir una base en memoria en modo
        solo lectura."""
        es_en_memoria = str(db_path) == ":memory:"
        try:
            conn = duckdb.connect(str(db_path), read_only=not es_en_memoria)
        except duckdb.Error as exc:
            raise InsufficientCanaryTrafficError(f"no pude abrir '{db_path}': {exc}") from exc

        try:
            existe = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table_name],
            ).fetchone()[0] > 0
            if not existe:
                raise InsufficientCanaryTrafficError(f"la tabla '{table_name}' no existe en '{db_path}'")
            return conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        finally:
            conn.close()

    def compute_health_metrics(self, predictions_df: pd.DataFrame) -> dict:
        """Todo se mide sobre la cohorte CANARY -- `mean_score_diff` además
        necesita la cohorte CHAMPION del mismo lote para comparar."""
        if predictions_df is None or len(predictions_df) == 0:
            raise InsufficientCanaryTrafficError("predictions_df esta vacio")

        canary_rows = predictions_df[predictions_df["assigned_model"] == CANARY]
        if len(canary_rows) == 0:
            raise InsufficientCanaryTrafficError("no hay ninguna fila con assigned_model == 'CANARY'")

        pred_canary = pd.to_numeric(canary_rows["prediction"], errors="coerce")
        es_invalida = pred_canary.isna() | np.isinf(pred_canary)
        null_rate = float(es_invalida.mean())

        pred_canary_validas = pred_canary[~es_invalida]

        champion_rows = predictions_df[predictions_df["assigned_model"] == CHAMPION]
        mean_score_diff = None
        if len(champion_rows) > 0 and len(pred_canary_validas) > 0:
            pred_champion = pd.to_numeric(champion_rows["prediction"], errors="coerce")
            pred_champion_validas = pred_champion[pred_champion.notna() & np.isfinite(pred_champion)]
            if len(pred_champion_validas) > 0:
                mean_score_diff = float(abs(pred_canary_validas.mean() - pred_champion_validas.mean()))

        # Sobre predicciones validas: un score invalido no es "alto riesgo",
        # ya cuenta contra null_rate -- contarlo dos veces distorsionaria la
        # proporcion sin agregar informacion nueva.
        high_risk_proportion = (
            float((pred_canary_validas > HIGH_RISK_THRESHOLD).mean()) if len(pred_canary_validas) > 0 else 0.0
        )

        self.metrics = {
            "null_rate": null_rate,
            "mean_score_diff": mean_score_diff,
            "high_risk_proportion": high_risk_proportion,
            "n_canary": int(len(canary_rows)),
            "n_champion": int(len(champion_rows)),
        }
        return dict(self.metrics)

    def evaluate_and_guard(self, predictions_df: pd.DataFrame, canary_config_path,
                            max_null_rate: float = 0.01, max_score_diff: float = 0.15,
                            max_high_risk_rate: float = 0.50) -> dict:
        metrics = self.compute_health_metrics(predictions_df)

        violaciones = []
        if metrics["null_rate"] > max_null_rate:
            violaciones.append(("null_rate", metrics["null_rate"], max_null_rate))
        if metrics["mean_score_diff"] is not None and metrics["mean_score_diff"] > max_score_diff:
            violaciones.append(("mean_score_diff", metrics["mean_score_diff"], max_score_diff))
        if metrics["high_risk_proportion"] > max_high_risk_rate:
            violaciones.append(("high_risk_proportion", metrics["high_risk_proportion"], max_high_risk_rate))

        if violaciones:
            reason = "; ".join(f"{nombre}={valor:.4f} supera el limite {limite:.4f}"
                                for nombre, valor, limite in violaciones)
            _force_rollback(canary_config_path, reason)
            self.decision = {
                "status": "ROLLBACK_TRIGGERED",
                "violated_threshold": ", ".join(v[0] for v in violaciones),
                "reason": reason,
            }
        else:
            self.decision = {"status": "HEALTHY", "violated_threshold": None, "reason": None}

        return dict(self.decision)

    def generate_health_report(self, output_dir) -> dict:
        if self.metrics is None or self.decision is None:
            raise RuntimeError("no hay evaluacion todavia -- llama evaluate_and_guard() primero")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            **self.metrics,
            **self.decision,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        report_path = output_dir / f"canary_health_{_timestamp()}.json"
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Salud canaria: %s -> %s", payload["status"], report_path)
        return {"report_path": str(report_path), "report": payload}
