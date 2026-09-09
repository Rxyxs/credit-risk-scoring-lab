"""Orquestador end-to-end de la tecnica 10 (PD through-the-cycle vs
point-in-time, modelo de un factor de Vasicek/ASRF).

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
    ("Simular 30 anos de cohortes, 5 grados, un factor sistematico compartido",
     "src.data_generator"),
    ("Recuperar rho y el ciclo; capital PIT vs TTC; verificacion Monte Carlo",
     "src.experiment"),
    ("Graficos de resultados", "src.visualization.plots"),
]


def run_step(descripcion: str, modulo: str):
    print(f"\n{'=' * 78}\n>> {descripcion}\n{'=' * 78}")
    r = subprocess.run([sys.executable, "-m", modulo], cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Etapa fallida: {modulo} (codigo {r.returncode})")


def resumen():
    res = json.loads((REPORTS_DIR / "experiment_results.json").read_text())

    print(f"\n{'=' * 78}\n>> RESUMEN\n{'=' * 78}")
    print("Correlacion de activos recuperada por grado (metodo de momentos)")
    print(f"  {'grado':<8} {'rho verdadero':>14} {'rho estimado':>13} {'error':>8}")
    for f in res["correlacion_por_grado"]:
        err = f["rho_hat_mom"] - f["rho_true"]
        print(f"  {f['grado']:<8} {f['rho_true']:>14.4f} {f['rho_hat_mom']:>13.4f} "
              f"{err:>+8.4f}")

    rc = res["recuperacion_ciclo"]
    print(f"\nCiclo economico recuperado desde tasas de default agregadas")
    print(f"  Correlacion con el ciclo verdadero: {rc['correlacion']:.4f}")

    c = res["capital"]
    print(f"\nDensidad de RWA, PIT vs TTC")
    print(f"  TTC: estable en {c['densidad_rwa_ttc_promedio_pct']:.2f}% (por construccion)")
    print(f"  PIT: entre {c['densidad_rwa_pit_min_pct']:.2f}% y "
          f"{c['densidad_rwa_pit_max_pct']:.2f}% "
          f"({c['swing_pit_pp']:.1f} puntos porcentuales de oscilacion)")

    granularidad = res["sesgo_granularidad"]
    chico = [f for f in granularidad if f["tamano"] == "chica"]
    err_mom = sum(abs(f["rho_hat_mom"] - f["rho_true"]) for f in chico) / len(chico)
    err_asrf = sum(abs(f["rho_hat_asrf_limite"] - f["rho_true"]) for f in chico) / len(chico)
    print(f"\nSesgo de granularidad, cartera chica (error absoluto medio en rho)")
    print(f"  Metodo de momentos : {err_mom:.4f}")
    print(f"  Limite ASRF        : {err_asrf:.4f}")
    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
