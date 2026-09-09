"""Orquestador end-to-end de la tecnica 11 (scoring federado entre bancos).

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
    ("Simular 6 bancos en los escenarios IID y no-IID", "src.data_generator"),
    ("Solo local vs FedAvg vs oraculo centralizado, en ambos escenarios",
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
    r = res["resumen_nacional"]["no_iid"]

    print(f"\n{'=' * 78}\n>> RESUMEN\n{'=' * 78}")
    print("Escenario no-IID, desempeno sobre el test nacional")
    print(f"  {'politica':<24} {'AUC':>7} {'Brier':>7} {'Sesgo calib. (pp)':>18}")
    for nombre, clave in [("Solo local (promedio)", "solo_local_promedio"),
                          ("Federado (FedAvg)", "federado"),
                          ("Oraculo centralizado", "oraculo_centralizado")]:
        m = r[clave]
        print(f"  {nombre:<24} {m['auc']:>7.4f} {m['brier']:>7.4f} "
              f"{m['sesgo_calibracion_pp']:>+18.2f}")

    peor = max(r["solo_local_por_banco"].items(),
              key=lambda kv: abs(kv[1]["sesgo_calibracion_pp"]))
    print(f"\nEl banco con el peor sesgo local aplicado nacionalmente: {peor[0]}")
    print(f"  PD media predicha  : {peor[1]['pd_media_predicha']:.2%}")
    print(f"  Sesgo de calibracion: {peor[1]['sesgo_calibracion_pp']:+.2f} pp")
    print(f"  (su AUC nacional sigue viendose razonable: {peor[1]['auc']:.4f} -- "
          f"el problema no se nota en el ranking)")

    barrido = res["barrido_epocas_locales"]
    mejor = min(barrido, key=lambda f: abs(f["nacional_sesgo_calibracion_pp"]))
    peor_e = max(barrido, key=lambda f: abs(f["nacional_sesgo_calibracion_pp"]))
    print(f"\nBarrido de epocas locales (mismo computo total, distinta frecuencia "
          f"de sincronizacion)")
    print(f"  Mejor calibracion  : E={mejor['local_epochs']} "
          f"(sesgo {mejor['nacional_sesgo_calibracion_pp']:+.3f} pp)")
    print(f"  Peor calibracion   : E={peor_e['local_epochs']} "
          f"(sesgo {peor_e['nacional_sesgo_calibracion_pp']:+.3f} pp)")

    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
