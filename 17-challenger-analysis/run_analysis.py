"""Lee `dual_inference_logs` (la técnica 16), mide la divergencia entre
Champion y Challenger, y escribe `shadow_analysis_<timestamp>.json`.

Sin tabla, sin base, o sin filas con Challenger comparado: aborta con
código 0 -- "todavía no hay suficiente evidencia dual" no es un error de
CI, es el estado normal antes de que 16 corra por primera vez.

    python run_analysis.py
    python run_analysis.py --db-path otra/ruta/dual_inference_logs.duckdb
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.analysis import InsufficientLogDataError, ShadowDivergenceAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "../16-shadow-deployment/outputs/dual_inference_logs.duckdb"
DEFAULT_OUTPUT_DIR = "outputs/reports"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Base DuckDB con dual_inference_logs (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta donde escribir shadow_analysis_*.json (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analyzer = ShadowDivergenceAnalyzer()

    try:
        df = analyzer.fetch_logs(args.db_path)
        metricas = analyzer.compute_divergence_metrics(df)
    except InsufficientLogDataError as exc:
        logger.info("No hay suficientes datos de inferencia dual: %s -- abortando sin error.", exc)
        return 0

    resultado = analyzer.generate_analysis_report(args.output_dir)

    print(f"n_predictions: {metricas['n_predictions']}")
    print(f"mean_absolute_difference: {metricas['mean_absolute_difference']:.4f}")
    print(f"max_absolute_difference: {metricas['max_absolute_difference']:.4f}")
    print(f"pearson_correlation: {metricas['pearson_correlation']:.4f}")
    print(f"ks_statistic: {metricas['ks_statistic']:.4f}  p_value: {metricas['p_value']:.4g}")
    print(f"distribution_shift_detected: {resultado['report']['distribution_shift_detected']}")
    print(f"reporte -> {resultado['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
