"""Orquestador end-to-end de la tecnica 04 (scorecard bayesiano jerarquico).

Uso:
    python run_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

ETAPAS = [
    ("Simular solicitudes con 32 segmentos y efectos jerarquicos", "src.data_generator"),
    ("Split train/test, estandarizacion e indice de segmento", "src.preprocessing"),
    ("Gibbs + Polya-Gamma: pooling completo, sin pooling y parcial", "src.fit_models"),
    ("Politica de aprobacion con incertidumbre posterior", "src.decision"),
    ("Graficos de resultados", "src.visualization.plots"),
]


def run_step(descripcion: str, modulo: str):
    print(f"\n{'=' * 74}\n>> {descripcion}\n{'=' * 74}")
    r = subprocess.run([sys.executable, "-m", modulo], cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Etapa fallida: {modulo} (codigo {r.returncode})")


def resumen():
    fit = json.loads((REPORTS_DIR / "fit_results.json").read_text())
    dec = json.loads((REPORTS_DIR / "decision_results.json").read_text())

    print(f"\n{'=' * 74}\n>> RESUMEN\n{'=' * 74}")
    print(f"{'enfoque':<28} {'AUC':>7} {'Brier':>8} {'logloss':>8} {'RMSE efectos':>13}")
    for nombre in ("complete", "none", "partial"):
        m = fit[nombre]["metricas_test"]
        rmse = fit[nombre]["rmse_efectos_segmento"]["todos"]
        print(f"{nombre:<28} {m['auc']:>7.4f} {m['brier']:>8.4f} {m['log_loss']:>8.4f} "
              f"{rmse:>13.4f}")
    m = fit["benchmark_logistica_dummies"]["metricas_test"]
    print(f"{'benchmark logistica+dummies':<28} {m['auc']:>7.4f} {m['brier']:>8.4f} "
          f"{m['log_loss']:>8.4f} {'-':>13}")

    tp = fit["partial"]["tau_posterior"]
    print(f"\ntau (dispersion entre segmentos): posterior {tp['media']:.3f} "
          f"[{tp['q05']:.3f}, {tp['q95']:.3f}] vs verdad {tp['true']:.3f}")

    c = fit["partial"]["convergencia"]
    print(f"Convergencia (pooling parcial): R-hat max {c['rhat_max']:.4f}, "
          f"ESS min {c['ess_min']:.0f}")

    comp = dec["comparacion_a_igual_volumen"]["0.80"]
    print(f"\nDecision al 80% de aprobacion (media vs percentil 95 de la posterior)")
    print(f"  Tasa de default realizada: {comp['tasa_mala_media']:.2%} -> "
          f"{comp['tasa_mala_conservadora']:.2%} ({comp['delta_tasa_mala_pp']:+.2f} pp)")
    print(f"  Utilidad realizada       : {comp['delta_utilidad_pct']:+.2f}%")
    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "data/processed", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
