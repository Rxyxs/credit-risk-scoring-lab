"""Orquestador end-to-end de la tecnica 09 (reject inference con el
contrafactual conocido).

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
    ("Simular MAR y MNAR con el desenlace de TODOS, incluidos rechazados",
     "src.data_generator"),
    ("Ocho metodos de reject inference, evaluados contra la verdad",
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
    rho_v, rho_e = res["rho_verdadero"], res["rho_estimado_probit_bivariado"]

    print(f"\n{'=' * 78}\n>> RESUMEN\n{'=' * 78}")
    print("Recuperacion de rho (correlacion entre seleccion y desenlace)")
    print(f"  {'regimen':<8} {'verdadero':>10} {'sin instrumento':>17} "
          f"{'con instrumento':>17}")
    for regimen in rho_v:
        r = rho_e[regimen]
        print(f"  {regimen.upper():<8} {rho_v[regimen]:>10.3f} "
              f"{r['sin_instrumento']:>17.3f} {r['con_instrumento']:>17.3f}")

    tabla = {(f["regimen"], f["metodo"]): f for f in res["resultados"]}
    mnar_naive = tabla[("mnar", "aprobados_solo")]
    mnar_biv = tabla[("mnar", "probit_bivariado")]
    mnar_oraculo = tabla[("mnar", "oraculo")]

    print("\nError de nivel en MNAR (PD estimada de la cartera - PD real)")
    print(f"  Solo aprobados       : {mnar_naive['error_nivel_pp']:+.2f} pp")
    print(f"  Probit bivariado     : {mnar_biv['error_nivel_pp']:+.2f} pp "
          f"(con instrumento)")
    print(f"  Oraculo (cota)       : {mnar_oraculo['error_nivel_pp']:+.2f} pp")

    print("\nSensibilidad de parcelling: el factor no se puede validar sin la")
    print("verdad, y aca se ve por que")
    for f in res["sensibilidad_parcelling_mnar"]:
        print(f"  factor {f['factor']:.1f} -> error de nivel {f['error_nivel_pp']:+.2f} pp")

    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "data/processed", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
