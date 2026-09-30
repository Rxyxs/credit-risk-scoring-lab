"""El ultimo eslabon visible antes de que el ciclo de vida de un modelo
vuelva a empezar: `21-post-cutover-telemetry` mide si el Champion sigue
sano una vez que su cartera madura; esta tecnica lee esa medicion y
decide -- con umbrales explicitos, no con intuicion -- si hace falta
disparar un reentrenamiento. No entrena nada: deja escrita la decision
para que otro proceso (fuera de este repo, como ya senalo
`15-model-promotion` sobre la suya) actue sobre ella.

Dos condiciones independientes, evaluadas por separado:
- desempeno realizado por debajo del piso (`realized_roc_auc < min_realized_auc`),
  solo cuando el reporte tiene etiquetas maduras (`status == "EVALUATED"`)
  y el AUC esta definido -- un reporte `INSUFFICIENT_MATURITY` nunca
  dispara por esta via, precisamente para no confundir "todavia no hay
  suficiente verdad de campo" con "el modelo se degrado".
- deriva de prediccion por encima del techo (`prediction_drift.psi >
  max_psi`) -- esta si se evalua aunque la maduracion sea insuficiente,
  porque el PSI no necesita verdad de campo: un Champion cuyos propios
  scores ya se corrieron de su baseline es una señal valida incluso el
  primer dia despues del cutover.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

MIN_REALIZED_AUC_DEFAULT = 0.72
MAX_PSI_DEFAULT = 0.20

# Mismo nombre y mismo esquema de columnas que la tabla que crea
# `20-full-promotion-cutover/src/cutover_manager.py` -- si `--db-path`
# apunta al mismo archivo, este evento se suma al mismo ledger de
# auditoria, no a uno paralelo.
TABLA_EVENTOS = "model_lifecycle_events"
TIPO_EVENTO = "RETRAINING_TRIGGERED"


class TelemetryReportError(RuntimeError):
    """El reporte de telemetria no existe, no es JSON valido, o no tiene
    la forma minima esperada -- todos los motivos por los que
    `run_trigger_check.py` aborta limpio en vez de fallar con un
    traceback."""


class RetrainingTriggerManager:
    def __init__(self):
        self.last_evaluation: dict | None = None

    def evaluate_trigger_conditions(self, telemetry_path,
                                     min_realized_auc: float = MIN_REALIZED_AUC_DEFAULT,
                                     max_psi: float = MAX_PSI_DEFAULT) -> dict:
        path = Path(telemetry_path)
        if not path.exists():
            raise TelemetryReportError(f"no existe el reporte de telemetria '{path}'")

        try:
            contenido = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TelemetryReportError(f"no pude leer '{path}': {exc}") from exc

        try:
            reporte = json.loads(contenido)
        except json.JSONDecodeError as exc:
            raise TelemetryReportError(f"'{path}' no es JSON valido: {exc}") from exc

        if not isinstance(reporte, dict):
            raise TelemetryReportError(f"'{path}' no tiene la forma esperada (objeto JSON)")

        performance = reporte.get("realized_performance") or {}
        drift = reporte.get("prediction_drift")

        razones: list[str] = []
        auc_evaluado = None
        psi_evaluado = None

        if performance.get("status") == "EVALUATED":
            auc = performance.get("realized_roc_auc")
            if auc is not None:
                auc_evaluado = float(auc)
                if auc_evaluado < min_realized_auc:
                    razones.append(
                        f"realized_roc_auc={auc_evaluado:.4f} por debajo del minimo "
                        f"{min_realized_auc:.4f}"
                    )

        if isinstance(drift, dict):
            psi = drift.get("psi")
            if psi is not None:
                psi_evaluado = float(psi)
                if psi_evaluado > max_psi:
                    razones.append(
                        f"prediction_drift.psi={psi_evaluado:.4f} por encima del maximo "
                        f"{max_psi:.4f}"
                    )

        resultado = {
            "trigger_activated": len(razones) > 0,
            "reasons": razones,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "telemetry_report_path": str(path),
            "telemetry_status": performance.get("status"),
            "realized_roc_auc": auc_evaluado,
            "psi": psi_evaluado,
            "min_realized_auc": min_realized_auc,
            "max_psi": max_psi,
        }
        self.last_evaluation = resultado
        return dict(resultado)

    def dispatch_retraining_event(self, output_dir, payload: dict, db_path) -> dict:
        """Escribe el manifiesto de disparo y suma una fila al ledger de
        eventos -- se llama solo cuando `payload["trigger_activated"]` es
        `True`; `run_trigger_check.py` es quien decide eso, este metodo
        solo ejecuta el registro."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"
        manifest_path = output_dir / f"retraining_trigger_{timestamp}.json"
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        event_id = str(uuid.uuid4())
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
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
                [event_id, TIPO_EVENTO, None, None, payload.get("evaluated_at", timestamp)],
            )
        finally:
            con.close()

        logger.warning("Reentrenamiento disparado (%s): %s -> %s",
                        event_id, "; ".join(payload.get("reasons", [])), manifest_path)

        return {"manifest_path": str(manifest_path), "event_id": event_id}
