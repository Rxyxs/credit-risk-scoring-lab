"""Corre `generate_drift_report` sobre una base y un set de scoring
sintéticos (uno de ellos con drift a propósito en una feature), exporta
HTML+JSON, y opcionalmente bloquea con código de salida distinto de cero si
alguna feature quedó en rojo (PSI > 0.25).

    python run_pipeline.py                                    # reporta, no bloquea
    python run_pipeline.py --fail-on-red                       # bloquea si hay rojo
    python run_pipeline.py --output-dir otra/ruta               # default: outputs/
    python run_pipeline.py --auto-retrain-trigger               # dispara DriftRemediationManager

`--fail-on-red` y `--auto-retrain-trigger` son independientes: el primero es
un gate de CI (falla el build si hay drift crítico), el segundo es un flujo
de orquestación (empaqueta snapshot+manifiesto para que algo afuera
reentrene). Con los dos juntos, el manifiesto se escribe igual y el build
igual falla -- uno no cancela al otro.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring.drift import export_drift_html, export_drift_json, generate_drift_report
from src.remediation.trigger import DriftRemediationManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ID_COLUMN = "client_id"
TARGET_COLUMN = "default_flag"
NON_FEATURE_COLUMNS = (ID_COLUMN, TARGET_COLUMN)


def _default_flag(rng: np.random.Generator, dti: np.ndarray, ingreso_mensual: np.ndarray) -> np.ndarray:
    """Target binario sintetico, correlacionado de verdad con dti (a favor)
    e ingreso_mensual (en contra) -- no ruido puro: `14-shadow-model-training/`
    necesita una columna de target con señal real para que su prueba de "el
    pipeline aprende algo, no solo corre" tenga sentido sobre estos mismos datos."""
    z = 3.0 * (dti - dti.mean()) / dti.std() - 1.5 * (ingreso_mensual - ingreso_mensual.mean()) / ingreso_mensual.std()
    p_default = 1 / (1 + np.exp(-z))
    return rng.binomial(1, p_default)


def build_demo_dataframes(shift: float = 1.0, seed: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base (train) y scoring (OOT) sintéticos. `shift` mueve la media de
    `ingreso_mensual` en `scoring` en unidades de desvío estándar de la base
    -- `shift=1.0` (default) reproduce el drift severo real medido en el
    README; `shift=0.0` genera scoring de la misma distribución que la base,
    sin drift en ninguna feature (usado por los tests para el caso estable).

    `client_id` y `default_flag` no son features de negocio para el drift:
    el primero identifica cada fila para que el snapshot que se manda a
    `13-feature-store-duckdb/` en rojo tenga con qué hacer upsert, el segundo
    es el target que `14-shadow-model-training/` entrena después -- por eso
    `main()` excluye a los dos de `generate_drift_report` explícitamente (un
    ID no tiene "drift", y un target monitoreado como si fuera una feature de
    entrada no tendría sentido de negocio, aunque el cálculo no fallaría).
    """
    rng = np.random.default_rng(seed)

    dti_base = rng.beta(2, 5, size=5_000)
    ingreso_base = rng.normal(loc=800_000, scale=150_000, size=5_000)
    baseline = pd.DataFrame({
        "client_id": [f"CLI-{i:06d}" for i in range(5_000)],
        "ingreso_mensual": ingreso_base,
        "dti": dti_base,
        "antiguedad_laboral_meses": rng.exponential(scale=36, size=5_000),
        "default_flag": _default_flag(rng, dti_base, ingreso_base),
    })

    dti_scoring = rng.beta(2, 5, size=2_000)
    ingreso_scoring = rng.normal(loc=800_000 + shift * 150_000, scale=160_000, size=2_000)
    scoring = pd.DataFrame({
        "client_id": [f"CLI-{i:06d}" for i in range(5_000, 7_000)],
        "ingreso_mensual": ingreso_scoring,
        "dti": dti_scoring,
        "antiguedad_laboral_meses": rng.exponential(scale=36, size=2_000),
        "default_flag": _default_flag(rng, dti_scoring, ingreso_scoring),
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
    parser.add_argument(
        "--auto-retrain-trigger", action="store_true",
        help=(
            "Evalua el reporte con DriftRemediationManager: en rojo escribe un snapshot "
            "y un manifiesto de reentrenamiento en <output-dir>/snapshots/."
        ),
    )
    parser.add_argument(
        "--model-version", default="unknown",
        help="Version del modelo actualmente en produccion (va en el manifiesto de reentrenamiento).",
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

    features = [c for c in baseline_df.columns if c not in NON_FEATURE_COLUMNS]
    reporte = generate_drift_report(baseline_df, scoring_df, features)

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

    if args.auto_retrain_trigger:
        manager = DriftRemediationManager(current_model_version=args.model_version)
        resultado = manager.evaluate_and_trigger(reporte, scoring_df, output_dir / "snapshots")
        print(f"\nremediacion -> status={resultado['status']}  action={resultado['action']}")
        if resultado["status"] == "red":
            print(f"snapshot -> {resultado['snapshot_path']}")
            print(f"manifiesto -> {resultado['manifest_path']}")

    if args.fail_on_red and reporte["features_criticas"]:
        logger.error(
            "Drift critico (PSI > 0.25) en: %s -- recalibracion requerida.",
            ", ".join(reporte["features_criticas"]),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
