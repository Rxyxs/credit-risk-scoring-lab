"""Sincroniza el registro del modelo sombra (si hay una promoción nueva
desde el Día 19) y corre inferencia dual Champion vs. Challenger sobre
`--input-data`, dejando ambas predicciones logueadas en DuckDB.

`--champion-model` no está en la lista de flags del Día 20, pero
`predict_dual` lo necesita de verdad: ningún día anterior de este
portafolio produjo un artefacto de "modelo de producción" separado del
modelo sombra -- así que hay que decirle a esta CLI explícitamente dónde
está el Champion, no hay un default razonable que inventar.

    python run_shadow_serving.py --champion-model outputs/champion_demo.pkl --input-data clientes.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.shadow_engine import DEFAULT_REGISTRY_DIR, ShadowDeploymentEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DECISION_DIR = "../15-model-promotion/outputs/decisions/"
DEFAULT_MODELS_DIR = "../14-shadow-model-training/outputs/models/"
DEFAULT_DB_PATH = "outputs/dual_inference_logs.duckdb"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-dir", default=DEFAULT_DECISION_DIR,
        help=f"Carpeta con promotion_decision_*.json (default: {DEFAULT_DECISION_DIR}).",
    )
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR,
        help=f"Carpeta con shadow_model_*.pkl (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument("--input-data", required=True, help="CSV con client_id + features a puntuar.")
    parser.add_argument("--champion-model", required=True, help="Ruta al .pkl del modelo Champion (produccion).")
    parser.add_argument("--registry-dir", default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    return parser.parse_args(argv)


def find_latest_decision(decision_dir: Path) -> Path | None:
    if not decision_dir.is_dir():
        return None
    candidatos = sorted(decision_dir.glob("promotion_decision_*.json"))
    return candidatos[-1] if candidatos else None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    engine = ShadowDeploymentEngine(registry_dir=args.registry_dir)

    decision_path = find_latest_decision(Path(args.decision_dir))
    if decision_path is None:
        logger.info(
            "No hay ningun promotion_decision_*.json en '%s' -- sigo con el registro tal como esta.",
            args.decision_dir,
        )
    else:
        logger.info("Decision mas reciente: %s", decision_path)
        engine.register_promoted_model(decision_path, args.models_dir, args.registry_dir)

    input_df = pd.read_csv(args.input_data)
    predicciones = engine.predict_dual(args.champion_model, input_df)
    filas_logueadas = engine.log_dual_predictions(predicciones, args.db_path)

    print(predicciones.to_string(index=False))
    print(f"\n{filas_logueadas} predicciones logueadas -> {args.db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
