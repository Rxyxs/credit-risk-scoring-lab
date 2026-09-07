"""Orquestador end-to-end de la tecnica 08 (scoring con privacidad diferencial).

Uso:
    python run_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

ETAPAS = [
    ("Simular cartera chica, holdout de no miembros y canarios", "src.data_generator"),
    ("DP-SGD sobre una grilla de presupuestos, con ataques en cada corrida",
     "src.experiment"),
    ("Graficos de resultados", "src.visualization.plots"),
]


def run_step(descripcion: str, modulo: str):
    print(f"\n{'=' * 74}\n>> {descripcion}\n{'=' * 74}")
    r = subprocess.run([sys.executable, "-m", modulo], cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Etapa fallida: {modulo} (codigo {r.returncode})")


def resumen():
    res = json.loads((REPORTS_DIR / "experiment_results.json").read_text())
    cfg, corridas = res["configuracion"], res["corridas"]
    base = corridas[0]

    print(f"\n{'=' * 74}\n>> RESUMEN\n{'=' * 74}")
    print(f"Entrenamiento con {cfg['n_train']:,} solicitudes "
          f"({cfg['n_canarios']} canarios inyectados), {cfg['pasos']:,} pasos de "
          f"DP-SGD, delta = {cfg['delta']}")
    print(f"Cada fila promedia {base['n_repeticiones']} corridas independientes\n")

    print(f"{'escenario':<16} {'sigma':>7} {'epsilon':>8} {'AUC':>8} "
          f"{'canarios pp':>12} {'t':>7} {'fuga detectable':>16}")
    for c in corridas:
        eps = "inf" if np.isinf(c["epsilon_real"]) else f"{c['epsilon_real']:.2f}"
        print(f"{c['escenario']:<16} {c['sigma']:>7.2f} {eps:>8} "
              f"{c['utilidad_auc']:>8.4f} {c['canario_exposicion_pp']:>12.2f} "
              f"{c['canario_t']:>7.1f} {'SI' if c['fuga_detectable'] else 'no':>16}")

    # El presupuesto que interesa es el MAS LAXO que ya no deja fuga
    # detectable: cualquier epsilon menor tambien la elimina, pero paga mas
    # utilidad por la misma proteccion.
    sin_fuga = [c for c in corridas
                if not c["fuga_detectable"] and np.isfinite(c["epsilon_real"])]
    if sin_fuga:
        elegido = max(sin_fuga, key=lambda c: c["epsilon_real"])
        costo = 100 * (1 - elegido["utilidad_auc"] / base["utilidad_auc"])
        print(f"\nEl presupuesto mas laxo que ya no deja fuga detectable es "
              f"epsilon = {elegido['epsilon_real']:.2f}, "
              f"y cuesta {costo:.1f}% del AUC "
              f"({base['utilidad_auc']:.4f} -> {elegido['utilidad_auc']:.4f})")
    print(f"\nAtaque de membresia: AUC entre "
          f"{min(c['ataque_auc'] for c in corridas):.4f} y "
          f"{max(c['ataque_auc'] for c in corridas):.4f} en todos los escenarios")
    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "data/processed", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
