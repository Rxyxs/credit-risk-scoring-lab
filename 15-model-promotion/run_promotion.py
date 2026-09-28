"""Busca el `shadow_metrics_*.json` más reciente en `--metrics-dir` (por
defecto, donde `14-shadow-model-training/` deja sus modelos), lo evalúa con
`ModelPromoter` contra los umbrales mínimos, y escribe la decisión.

PROMOTED y REJECTED son los dos resultados de negocio válidos -- el exit
code es 0 en ambos casos. Solo un problema técnico real (no hay ningún
archivo de métricas que evaluar, o el que hay está roto) hace que no se
escriba ninguna decisión, y tampoco eso es un error de CI: es "todavía no
hay nada que decidir".

    python run_promotion.py
    python run_promotion.py --min-roc-auc 0.80 --min-samples 500
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.promotion_logic import (
    DEFAULT_MIN_ROC_AUC,
    DEFAULT_MIN_SAMPLES,
    ModelPromoter,
    PromotionEvaluationError,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_METRICS_DIR = "../14-shadow-model-training/outputs/models/"
DEFAULT_OUTPUT_DIR = "outputs/decisions"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir", default=DEFAULT_METRICS_DIR,
        help=f"Carpeta con shadow_metrics_*.json (default: {DEFAULT_METRICS_DIR}).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Carpeta donde escribir promotion_decision_*.json (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--min-roc-auc", type=float, default=DEFAULT_MIN_ROC_AUC)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    return parser.parse_args(argv)


def find_latest_metrics(metrics_dir: Path) -> Path | None:
    """El más reciente por nombre de archivo (timestamp lexicográfico),
    mismo criterio que `find_latest_manifest` en la técnica 13."""
    if not metrics_dir.is_dir():
        return None
    candidatos = sorted(metrics_dir.glob("shadow_metrics_*.json"))
    return candidatos[-1] if candidatos else None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metrics_dir = Path(args.metrics_dir)

    metrics_path = find_latest_metrics(metrics_dir)
    if metrics_path is None:
        logger.info("No hay ningun shadow_metrics_*.json en '%s' -- nada que evaluar.", metrics_dir)
        return 0

    logger.info("Metricas mas recientes: %s", metrics_path)

    promoter = ModelPromoter()
    try:
        promoter.evaluate_candidate(metrics_path, min_roc_auc=args.min_roc_auc, min_samples=args.min_samples)
    except PromotionEvaluationError as exc:
        logger.info("No pude evaluar '%s': %s -- abortando sin error.", metrics_path, exc)
        return 0

    resultado = promoter.generate_decision(args.output_dir)
    decision = resultado["decision"]

    print(f"decision: {decision['decision']} -- {decision['reason']}")
    print(f"candidato: {decision['candidate_model']}")
    print(f"decision -> {resultado['decision_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
