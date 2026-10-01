"""Corre la auditoria completa del laboratorio (tecnicas 12 a 25: bases
DuckDB, registro de artefactos, telemetria, disparadores de
reentrenamiento) y escribe `lab_summary_report_<timestamp>.json` +
`LAB_SUMMARY.md` en `--output-dir`.

    python run_lab_summary.py
    python run_lab_summary.py --db-path otra/ruta/lab_lifecycle.duckdb --output-dir otra/salida
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.lab_summary_engine import DEFAULT_CHAMPION_DIR, LabSummaryEngine

BASE = Path(__file__).resolve().parent

DEFAULT_DB_PATH = "../20-full-promotion-cutover/outputs/lab_lifecycle.duckdb"
DEFAULT_OUTPUT_DIR = "outputs"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Base DuckDB central con model_lifecycle_events (default: {DEFAULT_DB_PATH}).")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta donde escribir el informe JSON y LAB_SUMMARY.md (default: {DEFAULT_OUTPUT_DIR}).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    engine = LabSummaryEngine()
    resultado = engine.generate_master_summary(args.db_path, BASE / args.output_dir)
    resumen = resultado["summary"]

    print(f"estado del laboratorio: {resumen['lab_status']}")
    print(f"ejecuciones totales registradas: {resumen['total_executions']}")
    champion = resumen["champion_in_service"]
    if champion:
        print(f"champion en servicio: {champion['path']} (pickle valido: {champion['valid_pickle']})")
    else:
        print(f"champion en servicio: ninguno detectado en {DEFAULT_CHAMPION_DIR}")
    telemetria = resumen["latest_telemetry"]
    if telemetria:
        print(f"ultima telemetria: ROC-AUC={telemetria['realized_roc_auc']}  PSI={telemetria['psi']}")
    disparador = resumen["retraining_trigger_status"]
    if disparador:
        print(f"disparador de reentrenamiento mas reciente: activo={disparador['trigger_activated']}")
    print(f"informe -> {resultado['report_path']}")
    print(f"resumen -> {resultado['markdown_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
