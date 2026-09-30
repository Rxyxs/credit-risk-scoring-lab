"""Revisa si hay un disparador de reentrenamiento activo (Dia 26) y, si lo
hay, ejecuta el circuito cerrado completo: extrae la version mas fresca
del Feature Store, entrena un nuevo candidato Challenger, evalua sus
metricas, guarda `.pkl`/`.json`, y registra `RETRAINING_EXECUTED` en la
base DuckDB central.

Sin disparadores activos pendientes: log informativo, codigo de salida 0
-- el sistema no necesita reentrenar ahora, eso no es una falla. Con un
disparador activo pero el Feature Store vacio o inaccesible: tambien
codigo 0, con el motivo en el log -- el disparador queda sin marcar como
procesado, listo para reintentar en la proxima corrida.

    python run_orchestrator.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline_orchestrator import EmptyFeatureStoreError, RetrainingPipelineOrchestrator

BASE = Path(__file__).resolve().parent

DEFAULT_TRIGGER_DIR = "../22-automated-retraining-trigger/outputs/manifests"
DEFAULT_DB_PATH = "../20-full-promotion-cutover/outputs/lab_lifecycle.duckdb"
DEFAULT_FEATURE_STORE_PATH = "../13-feature-store-duckdb/outputs/offline_store.duckdb"
DEFAULT_MODELS_OUTPUT_DIR = "outputs/models"
DEFAULT_REPORTS_OUTPUT_DIR = "outputs/reports"
DEFAULT_STATE_PATH = "outputs/state/processed_triggers.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_orchestrator")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trigger-dir", default=DEFAULT_TRIGGER_DIR,
        help=f"Carpeta con retraining_trigger_<timestamp>.json (default: {DEFAULT_TRIGGER_DIR}, "
             "donde los deja 22-automated-retraining-trigger/run_trigger_check.py).")
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Base DuckDB central con model_lifecycle_events (default: {DEFAULT_DB_PATH}).")
    parser.add_argument("--feature-store-path", default=DEFAULT_FEATURE_STORE_PATH)
    parser.add_argument("--models-output-dir", default=DEFAULT_MODELS_OUTPUT_DIR)
    parser.add_argument("--reports-output-dir", default=DEFAULT_REPORTS_OUTPUT_DIR)
    parser.add_argument("--state-path", default=DEFAULT_STATE_PATH)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    orchestrator = RetrainingPipelineOrchestrator(state_path=BASE / args.state_path)

    if not orchestrator.check_pending_triggers(args.trigger_dir):
        logger.info("No hay disparadores activos pendientes -- el sistema no requiere "
                     "reentrenamiento en este momento.")
        return 0

    logger.info("Disparador activo detectado: %s -- iniciando flujo de reentrenamiento.",
                orchestrator.pending_trigger_path)

    try:
        resultado = orchestrator.run_retraining_flow(
            feature_store_path=args.feature_store_path,
            models_output_dir=BASE / args.models_output_dir,
            reports_output_dir=BASE / args.reports_output_dir,
            db_path=args.db_path,
        )
    except EmptyFeatureStoreError as exc:
        logger.info("No se pudo ejecutar el flujo de reentrenamiento: %s -- abortando sin "
                     "error (el disparador sigue pendiente para el proximo intento).", exc)
        return 0

    auc = resultado["metrics"]["roc_auc"]
    logger.info(
        "Nuevo candidato Challenger generado: ROC-AUC=%s, n=%d -> %s (%s)",
        f"{auc:.4f}" if auc is not None else "n/d",
        resultado["metrics"]["n_samples"], resultado["model_path"], resultado["metrics_path"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
