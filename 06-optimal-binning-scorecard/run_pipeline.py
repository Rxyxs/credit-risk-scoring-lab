"""Orquestador end-to-end de la tecnica 06 (binning optimo -> scorecard).

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
    ("Simular 24 cohortes mensuales con deterioro en las ultimas 6", "src.data_generator"),
    ("Binning optimo por DP vs equifrecuente vs arbol, y scorecard", "src.fit_scorecard"),
    ("Monitoreo de estabilidad PSI/CSI vintage por vintage", "src.monitor_run"),
    ("Graficos de resultados", "src.visualization.plots"),
]


def run_step(descripcion: str, modulo: str):
    print(f"\n{'=' * 74}\n>> {descripcion}\n{'=' * 74}")
    r = subprocess.run([sys.executable, "-m", modulo], cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Etapa fallida: {modulo} (codigo {r.returncode})")


def resumen():
    sc = json.loads((REPORTS_DIR / "scorecard_results.json").read_text())
    mon = json.loads((REPORTS_DIR / "monitoring_results.json").read_text())

    print(f"\n{'=' * 74}\n>> RESUMEN\n{'=' * 74}")
    print(f"{'metodo de binning':<16} {'IV total':>9} {'no monot.':>10} "
          f"{'AUC in-time':>12} {'AUC out-of-time':>16}")
    for metodo, r in sc["metodos"].items():
        print(f"{metodo:<16} {r['iv_total']:>9.4f} "
              f"{len(r['variables_no_monotonas']):>10} "
              f"{r['in_time']['auc']:>12.4f} {r['out_of_time']['auc']:>16.4f}")

    bandas = sc["bandas_de_riesgo"]
    peor, mejor = bandas[0], bandas[-1]
    print(f"\nBandas de riesgo (quintiles del puntaje, validacion in-time)")
    print(f"  Banda {peor['banda']}: {peor['tasa_mala']:.2%} de default | "
          f"Banda {mejor['banda']}: {mejor['tasa_mala']:.2%} "
          f"({peor['tasa_mala'] / mejor['tasa_mala']:.1f}x de separacion)")

    r = mon["resumen"]
    d = mon["degradacion"]
    print(f"\nMonitoreo de estabilidad")
    print(f"  Quiebre real de poblacion : vintage {mon['vintage_quiebre_real']}")
    print(f"  Primer vintage en alerta  : {r['primer_vintage_en_alerta']} "
          f"(PSI max {r['psi_max']:.4f})")
    print(f"  Variable mas inestable    : {r['variable_mas_inestable']} "
          f"(CSI {r['csi_maximo']:.4f})")
    print(f"  AUC estables -> deteriorados: {d['auc_vintages_estables']:.4f} -> "
          f"{d['auc_vintages_deteriorados']:.4f}")
    print(f"  Calibracion en el deterioro : PD predicha "
          f"{d['pd_predicha_media_deteriorados']:.2%} vs real "
          f"{d['tasa_mala_real_deteriorados']:.2%} "
          f"({d['sesgo_de_calibracion_deteriorados_pp']:+.2f} pp)")
    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "data/processed", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
