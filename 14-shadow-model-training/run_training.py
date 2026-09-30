"""Orquesta fetch -> train -> save contra el feature store offline que deja
`13-feature-store-duckdb/`. Aborta con código 0 (no es un error de CI, es
"no había suficiente motivo para reentrenar todavía") si la tabla no
existe, el feature store no existe aún, no llega al mínimo de filas, o no
tiene ninguna columna de target reconocible.

    python run_training.py
    python run_training.py --db-path otra/ruta/offline_store.duckdb
    python run_training.py --output-dir outputs/models --target-column default_flag
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.train_shadow import InsufficientDataError, ShadowModelTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "../13-feature-store-duckdb/outputs/offline_store.duckdb"
DEFAULT_OUTPUT_DIR = "outputs/models"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Feature store de donde leer credit_features (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta donde guardar el modelo y sus metricas (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--target-column", default=None,
        help="Nombre de la columna target; por defecto se busca 'default_flag' u otras candidatas.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    trainer = ShadowModelTrainer(target_column=args.target_column)

    try:
        X, y = trainer.fetch_data(args.db_path)
    except InsufficientDataError as exc:
        logger.info("Sin datos suficientes para reentrenar el modelo sombra: %s -- abortando sin error.", exc)
        return 0

    metricas = trainer.train_and_evaluate(X, y)
    resultado = trainer.save_model(args.output_dir)

    print(
        f"ROC-AUC: {metricas['roc_auc']:.4f}  "
        f"(n={metricas['n_samples']}, train={metricas['n_train']}, test={metricas['n_test']})"
    )
    print(f"modelo -> {resultado['model_path']}")
    print(f"metricas -> {resultado['metrics_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
