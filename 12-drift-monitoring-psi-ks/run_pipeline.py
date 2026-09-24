"""Demo de una corrida real: genera una base (train) y un set de scoring
sintéticos con drift a propósito en una sola feature, y corre
`generate_drift_report` sobre las tres.

    python run_pipeline.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.monitoring.drift import generate_drift_report


def main() -> None:
    rng = np.random.default_rng(2024)

    baseline = pd.DataFrame({
        "ingreso_mensual": rng.normal(loc=800_000, scale=150_000, size=5_000),
        "dti": rng.beta(2, 5, size=5_000),
        "antiguedad_laboral_meses": rng.exponential(scale=36, size=5_000),
    })
    # Escenario: 6 meses después, el ingreso medio de la cartera se corrió
    # (inflación / mezcla de segmento), dti y antigüedad se mantienen estables.
    scoring = pd.DataFrame({
        "ingreso_mensual": rng.normal(loc=950_000, scale=160_000, size=2_000),
        "dti": rng.beta(2, 5, size=2_000),
        "antiguedad_laboral_meses": rng.exponential(scale=36, size=2_000),
    })

    reporte = generate_drift_report(baseline, scoring, list(baseline.columns))

    print(f"baseline n={reporte['n_baseline']}  scoring n={reporte['n_scoring']}\n")
    for feature, datos in reporte["features"].items():
        print(
            f"  {feature:28s} PSI={datos['psi']:.4f} ({datos['psi_status']})  "
            f"KS p-value={datos['ks_p_value']:.4g}  "
            f"drift_ks={datos['ks_drift_detected']}"
        )
    print(f"\nfeatures en alerta:  {reporte['features_en_alerta']}")
    print(f"features criticas:   {reporte['features_criticas']}")

    with open("drift_report.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print("\nreporte completo -> drift_report.json")


if __name__ == "__main__":
    main()
