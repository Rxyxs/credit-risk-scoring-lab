"""Evalúa la salud de las predicciones CANARY recientes y, si algo viola un
umbral, fuerza el rollback automático de `18-canary-deployment` escribiendo
`canary_config.json` a 0%.

`--predictions-data` acepta un CSV (si termina en `.csv`) o un archivo
DuckDB (cualquier otra extensión, se lee la tabla `canary_routing_log` que
`18-canary-deployment/run_canary.py` deja ahí).

Sin ninguna predicción CANARY que evaluar (la base/tabla no existe todavía,
o el CSV/tabla no tiene ninguna fila CANARY): aborta con código 0 -- no es
un error de CI, es el estado normal antes de que el canary reciba tráfico.

    python run_health_check.py
    python run_health_check.py --predictions-data predicciones.csv
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.canary_health import CanaryHealthMonitor, InsufficientCanaryTrafficError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_DATA = "../18-canary-deployment/outputs/canary_predictions.duckdb"
DEFAULT_CONFIG_PATH = "../18-canary-deployment/outputs/canary_config.json"
DEFAULT_OUTPUT_DIR = "outputs/reports"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-data", default=DEFAULT_PREDICTIONS_DATA,
        help=f"CSV o base DuckDB con predicciones canarias recientes (default: {DEFAULT_PREDICTIONS_DATA}).",
    )
    parser.add_argument(
        "--config-path", default=DEFAULT_CONFIG_PATH,
        help=f"canary_config.json a actualizar si hace falta rollback (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-null-rate", type=float, default=0.01)
    parser.add_argument("--max-score-diff", type=float, default=0.15)
    parser.add_argument("--max-high-risk-rate", type=float, default=0.50)
    return parser.parse_args(argv)


def _cargar_predicciones(monitor: CanaryHealthMonitor, ruta: str) -> pd.DataFrame:
    if str(ruta).lower().endswith(".csv"):
        return pd.read_csv(ruta)
    return monitor.fetch_recent_predictions(ruta)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    monitor = CanaryHealthMonitor()

    try:
        predictions_df = _cargar_predicciones(monitor, args.predictions_data)
        decision = monitor.evaluate_and_guard(
            predictions_df, args.config_path,
            max_null_rate=args.max_null_rate, max_score_diff=args.max_score_diff,
            max_high_risk_rate=args.max_high_risk_rate,
        )
    except (FileNotFoundError, InsufficientCanaryTrafficError) as exc:
        logger.info("No hay suficiente trafico canario para evaluar: %s -- abortando sin error.", exc)
        return 0

    resultado = monitor.generate_health_report(args.output_dir)

    print(f"status: {decision['status']}")
    if decision["status"] == "ROLLBACK_TRIGGERED":
        print(f"umbral violado: {decision['violated_threshold']}")
        print(f"motivo: {decision['reason']}")
    print(f"reporte -> {resultado['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
