"""Remediación automática ante drift: decide y ejecuta una acción según el
semáforo agregado de `generate_drift_report` (`src/monitoring/drift.py`).

No entreno nada ni reentreno un modelo acá -- eso queda fuera del alcance de
este repo. Lo que hago es empaquetar lo que un pipeline de orquestación
externo necesitaría para decidir un reentrenamiento: un snapshot de los
datos que dispararon la alerta y un manifiesto JSON con el motivo, las
features afectadas y dónde quedó el snapshot. En rojo (PSI > 0.25) eso se
escribe a disco; en amarillo, solo queda una recomendación en el log; en
verde, no se toca el filesystem para nada.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

from src.monitoring.drift import overall_status

logger = logging.getLogger(__name__)

TRIGGER_REASON_CRITICAL_DRIFT = "CRITICAL_DRIFT_PSI"
SUGGESTED_ACTION_RETRAIN = "RETRAIN_SHADOW_MODEL"
RECOMMENDATION_SCHEDULE_COLLECTION = "SCHEDULE_DATA_COLLECTION"
ACTION_NONE = "NO_ACTION_REQUIRED"


def _validar_drift_report(drift_report) -> None:
    if not isinstance(drift_report, dict) or "features_criticas" not in drift_report \
            or "features_en_alerta" not in drift_report:
        raise ValueError(
            "drift_report invalido: esperaba el dict que devuelve "
            "generate_drift_report (con 'features_criticas' y 'features_en_alerta')"
        )


def _timestamp() -> str:
    # Sin ":" -- valido tanto en Windows como en Linux; distinto de los
    # timestamps ISO de drift.py, que solo van adentro de un JSON, nunca a un nombre de archivo.
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class DriftRemediationManager:
    """Evalúa un reporte de drift ya generado y, si corresponde, dispara la
    remediación. `current_model_version` se fija una vez al construir el
    manager (el deployment que está sirviendo ahora), no en cada llamada.
    """

    def __init__(self, current_model_version: str = "unknown"):
        self.current_model_version = current_model_version

    def evaluate_and_trigger(self, drift_report: dict, current_dataset: pd.DataFrame,
                              output_dir) -> dict:
        """Devuelve un dict con el resultado de la evaluación: siempre trae
        `status` y `action`; en rojo también `snapshot_path` y `manifest_path`.
        """
        _validar_drift_report(drift_report)
        if current_dataset is None:
            raise ValueError("current_dataset no puede ser None")

        status = overall_status(drift_report)

        if status == "red":
            return self._trigger_critical(drift_report, current_dataset, Path(output_dir))
        if status == "yellow":
            return self._recommend_data_collection(drift_report)
        return self._no_action(drift_report)

    def _trigger_critical(self, drift_report: dict, current_dataset: pd.DataFrame,
                           output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _timestamp()

        snapshot_path = output_dir / f"retrain_data_{timestamp}.csv"
        current_dataset.to_csv(snapshot_path, index=False)

        manifest = {
            "trigger_reason": TRIGGER_REASON_CRITICAL_DRIFT,
            "features_affected": list(drift_report["features_criticas"]),
            "current_model_version": self.current_model_version,
            "snapshot_path": str(snapshot_path),
            "suggested_action": SUGGESTED_ACTION_RETRAIN,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        manifest_path = output_dir / f"retrain_manifest_{timestamp}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.warning(
            "DRIFT CRITICO (PSI > 0.25) en %s -- modelo actual: %s -- snapshot: %s -- manifiesto: %s",
            manifest["features_affected"], self.current_model_version, snapshot_path, manifest_path,
        )

        return {
            "status": "red",
            "action": SUGGESTED_ACTION_RETRAIN,
            "snapshot_path": str(snapshot_path),
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }

    def _recommend_data_collection(self, drift_report: dict) -> dict:
        logger.warning(
            "Drift moderado (PSI entre 0.10 y 0.25) en %s -- recomendacion: %s",
            drift_report["features_en_alerta"], RECOMMENDATION_SCHEDULE_COLLECTION,
        )
        return {"status": "yellow", "action": RECOMMENDATION_SCHEDULE_COLLECTION}

    def _no_action(self, drift_report: dict) -> dict:
        logger.info("Sin drift relevante (PSI < 0.10) -- %s", ACTION_NONE)
        return {"status": "green", "action": ACTION_NONE}
