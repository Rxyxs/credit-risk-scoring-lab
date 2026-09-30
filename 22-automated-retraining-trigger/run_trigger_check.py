"""Lee el reporte de telemetria post-cutover mas reciente y decide si hay
que disparar un reentrenamiento -- desempeno realizado por debajo del
piso, o deriva de prediccion por encima del techo.

No activar el disparador no es lo mismo que no encontrar nada que
evaluar: sin reporte, o con un reporte que no es JSON valido, el script
aborta limpio con codigo 0 y un log explicando por que (la telemetria
puede no haber corrido todavia). Activar el disparador tambien termina en
codigo 0 -- es una decision de negocio valida, exactamente igual que
`15-model-promotion/run_promotion.py` trata un `REJECTED`.

    python run_trigger_check.py
    python run_trigger_check.py --min-realized-auc 0.75 --max-psi 0.15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.retraining_trigger import (
    MAX_PSI_DEFAULT, MIN_REALIZED_AUC_DEFAULT, RetrainingTriggerManager, TelemetryReportError,
)

BASE = Path(__file__).resolve().parent

DEFAULT_TELEMETRY_DIR = "../21-post-cutover-telemetry/outputs/reports"
DEFAULT_DB_PATH = "../20-full-promotion-cutover/outputs/lab_lifecycle.duckdb"
DEFAULT_MANIFEST_DIR = "outputs/manifests"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_trigger_check")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--telemetry-report-path", default=None,
        help="Ruta a un post_cutover_telemetry_<timestamp>.json puntual. Por defecto, el "
             f"mas reciente en {DEFAULT_TELEMETRY_DIR} (donde lo deja "
             "21-post-cutover-telemetry/run_telemetry.py).")
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Base DuckDB central con model_lifecycle_events (default: {DEFAULT_DB_PATH} "
             "-- la misma que crea 20-full-promotion-cutover).")
    parser.add_argument("--min-realized-auc", type=float, default=MIN_REALIZED_AUC_DEFAULT)
    parser.add_argument("--max-psi", type=float, default=MAX_PSI_DEFAULT)
    parser.add_argument("--output-dir", default=DEFAULT_MANIFEST_DIR)
    return parser.parse_args(argv)


def find_latest_telemetry_report(directory) -> Path | None:
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidatos = sorted(directory.glob("post_cutover_telemetry_*.json"))
    return candidatos[-1] if candidatos else None


def main(argv=None) -> int:
    args = parse_args(argv)
    manager = RetrainingTriggerManager()

    telemetry_path = args.telemetry_report_path
    if telemetry_path is None:
        latest = find_latest_telemetry_report(DEFAULT_TELEMETRY_DIR)
        telemetry_path = latest if latest is not None else (
            Path(DEFAULT_TELEMETRY_DIR) / "post_cutover_telemetry_missing.json")

    try:
        resultado = manager.evaluate_trigger_conditions(
            telemetry_path, min_realized_auc=args.min_realized_auc, max_psi=args.max_psi)
    except TelemetryReportError as exc:
        logger.info("No se pudo evaluar el disparador de reentrenamiento: %s -- "
                     "abortando sin error.", exc)
        return 0

    if not resultado["trigger_activated"]:
        logger.info(
            "Champion dentro de los parametros de salud (status=%s, auc=%s, psi=%s) -- "
            "no se dispara reentrenamiento.",
            resultado["telemetry_status"], resultado["realized_roc_auc"], resultado["psi"],
        )
        return 0

    despacho = manager.dispatch_retraining_event(
        BASE / args.output_dir, resultado, args.db_path)
    logger.info("Reentrenamiento disparado: %s -> manifiesto %s",
                "; ".join(resultado["reasons"]), despacho["manifest_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
