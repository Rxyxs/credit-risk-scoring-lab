"""Reconstruye el expediente de linaje completo de un artefacto de modelo
-- `champion_model.pkl` o cualquier `shadow_model_<timestamp>.pkl` -- leyendo
los archivos que las tecnicas 12 a 23 ya dejaron en disco y la base DuckDB
central, sin importar codigo de ninguna de ellas.

    python run_lineage.py --model-filename champion_model.pkl
    python run_lineage.py --model-filename shadow_model_20260101T000000Z.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.lineage_tracker import ModelLineageTracker

BASE = Path(__file__).resolve().parent

DEFAULT_DB_PATH = "../20-full-promotion-cutover/outputs/lab_lifecycle.duckdb"
DEFAULT_OUTPUT_DIR = "outputs/manifests"
DEFAULT_REPORTS_DIRS = (
    "../12-drift-monitoring-psi-ks/outputs/snapshots",
    "../14-shadow-model-training/outputs/models",
    "../15-model-promotion/outputs/decisions",
    "../20-full-promotion-cutover/outputs/manifests",
    "../22-automated-retraining-trigger/outputs/manifests",
    "../23-automated-retraining-pipeline/outputs/reports",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_lineage")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-filename", required=True,
        help="Nombre del artefacto a trazar, ej. champion_model.pkl o shadow_model_<timestamp>.pkl.")
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Base DuckDB central con model_lifecycle_events (default: {DEFAULT_DB_PATH}).")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta donde escribir model_lineage_<nombre>_<timestamp>.json (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument(
        "--reports-dir", action="append", dest="reports_dirs", default=None,
        help="Carpeta adicional donde buscar manifiestos/metricas/decisiones. Repetible. "
             "Sin esto, se usan las carpetas reales de las tecnicas 12, 14, 15, 20, 22 y 23.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    reports_dirs = args.reports_dirs if args.reports_dirs else list(DEFAULT_REPORTS_DIRS)

    tracker = ModelLineageTracker()
    trazado = tracker.trace_model_lineage(args.model_filename, args.db_path, reports_dirs)

    manifest_path = tracker.generate_governance_manifest(args.model_filename, BASE / args.output_dir)

    print(json.dumps(trazado, indent=2, default=str))
    logger.info("Expediente de linaje (%s): %s -> %s",
                trazado["status"], args.model_filename, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
