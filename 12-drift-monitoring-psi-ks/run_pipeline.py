"""Corre `generate_drift_report` sobre una base y un set de scoring
sintéticos (uno de ellos con drift a propósito en una feature), exporta
HTML+JSON, y opcionalmente bloquea con código de salida distinto de cero si
alguna feature quedó en rojo (PSI > 0.25).

    python run_pipeline.py                                    # reporta, no bloquea
    python run_pipeline.py --fail-on-red                       # bloquea si hay rojo
    python run_pipeline.py --output-dir otra/ruta               # default: outputs/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.drift import export_drift_html, export_drift_json, generate_drift_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_demo_dataframes(shift: float = 1.0, seed: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base (train) y scoring (OOT) sintéticos. `shift` mueve la media de
    `ingreso_mensual` en `scoring` en unidades de desvío estándar de la base
    -- `shift=1.0` (default) reproduce el drift severo real medido en el
    README; `shift=0.0` genera scoring de la misma distribución que la base,
    sin drift en ninguna feature (usado por los tests para el caso estable).
    """
    rng = np.random.default_rng(seed)

    baseline = pd.DataFrame({
        "ingreso_mensual": rng.normal(loc=800_000, scale=150_000, size=5_000),
        "dti": rng.beta(2, 5, size=5_000),
        "antiguedad_laboral_meses": rng.exponential(scale=36, size=5_000),
    })
    scoring = pd.DataFrame({
        "ingreso_mensual": rng.normal(loc=800_000 + shift * 150_000, scale=160_000, size=2_000),
        "dti": rng.beta(2, 5, size=2_000),
        "antiguedad_laboral_meses": rng.exponential(scale=36, size=2_000),
    })
    return baseline, scoring


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-red", action="store_true",
        help="Termina con codigo de salida 1 si alguna feature queda en rojo (PSI > 0.25).",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Carpeta donde escribir drift_report.html y drift_report.json (default: outputs/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, baseline_df: pd.DataFrame | None = None,
         scoring_df: pd.DataFrame | None = None) -> int:
    """Punto de entrada real de la CLI. `baseline_df`/`scoring_df` son un
    seam de inyección solo para tests -- la CLI real siempre usa
    `build_demo_dataframes()`.
    """
    args = _parse_args(argv)

    if baseline_df is None or scoring_df is None:
        baseline_df, scoring_df = build_demo_dataframes()

    reporte = generate_drift_report(baseline_df, scoring_df, list(baseline_df.columns))

    print(f"baseline n={reporte['n_baseline']}  scoring n={reporte['n_scoring']}\n")
    for feature, datos in reporte["features"].items():
        print(
            f"  {feature:28s} PSI={datos['psi']:.4f} ({datos['psi_status']})  "
            f"KS p-value={datos['ks_p_value']:.4g}  "
            f"drift_ks={datos['ks_drift_detected']}"
        )
    print(f"\nfeatures en alerta:  {reporte['features_en_alerta']}")
    print(f"features criticas:   {reporte['features_criticas']}")

    output_dir = Path(args.output_dir)
    json_path = export_drift_json(reporte, output_dir / "drift_report.json")
    html_path = export_drift_html(reporte, output_dir / "drift_report.html")
    print(f"\nreporte -> {json_path}")
    print(f"reporte -> {html_path}")

    if args.fail_on_red and reporte["features_criticas"]:
        logger.error(
            "Drift critico (PSI > 0.25) en: %s -- recalibracion requerida.",
            ", ".join(reporte["features_criticas"]),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
