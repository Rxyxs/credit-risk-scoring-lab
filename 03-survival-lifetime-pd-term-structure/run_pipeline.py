"""Orquestador end-to-end de la tecnica 03 (supervivencia -> PD lifetime).

Cada etapa se invoca como el mismo comando que se documenta para correrla
sola, para que "correr todo" y "correr un paso" no puedan divergir.

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
    ("Simular cartera con historia mensual (default / prepago / censura)", "src.data_generator"),
    ("Split train/test + expansion a formato persona-periodo", "src.preprocessing"),
    ("Cox desde cero: Efron vs Breslow, verdad de terreno y test PH", "src.cox_model"),
    ("Hazard en tiempo discreto: PH vs efecto variable en el tiempo", "src.discrete_model"),
    ("Estructura temporal de PD, bandas y provision IFRS 9", "src.term_structure"),
    ("Graficos de resultados", "src.visualization.plots"),
]


def run_step(descripcion: str, modulo: str):
    print(f"\n{'=' * 74}\n>> {descripcion}\n{'=' * 74}")
    result = subprocess.run([sys.executable, "-m", modulo], cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Etapa fallida: {modulo} (codigo {result.returncode})")


def resumen():
    cox = json.loads((REPORTS_DIR / "cox_results.json").read_text())
    dis = json.loads((REPORTS_DIR / "discrete_results.json").read_text())
    ts = json.loads((REPORTS_DIR / "term_structure_results.json").read_text())

    print(f"\n{'=' * 74}\n>> RESUMEN\n{'=' * 74}")
    print("Discriminacion fuera de muestra (cartera de test)")
    c = cox["validacion_test"]
    print(f"  Cox (Efron)            C-index {c['c_index']:.4f} | AUC 12m {c['auc']:.4f} | "
          f"KS {c['ks_12m']:.4f}")
    for nombre, m in dis["validacion_test"].items():
        print(f"  Discreto [{nombre}]{'':7}C-index {m['c_index']:.4f} | AUC 12m {m['auc']:.4f} | "
              f"KS {m['ks_12m']:.4f}")

    print("\nRecuperacion de los coeficientes verdaderos")
    r = cox["recuperacion_de_verdad"]
    print(f"  Error absoluto medio   Efron {r['error_abs_medio_efron']:.4f} | "
          f"Breslow {r['error_abs_medio_breslow']:.4f}")
    print(f"  Atenuacion de Breslow  {r['atenuacion_media_breslow_pct']:.2f}% hacia cero")

    violan = [t["feature"] for t in cox["test_ph"] if t["viola_ph"]]
    print(f"  Covariables que violan PH: {', '.join(violan) if violan else 'ninguna'}")

    print("\nEstructura temporal")
    print(f"  Peak del hazard condicional : mes {ts['mes_peak_hazard_condicional_suavizado']}")
    print(f"  PD cartera 12m -> 36m       : {ts['pd_cartera_12m']:.2%} -> {ts['pd_cartera_36m']:.2%} "
          f"({ts['ratio_lifetime_sobre_12m']:.2f}x)")
    print(f"  Riesgo posterior al mes 12  : {ts['pct_riesgo_despues_de_12m']:.1f}% del total")

    p = ts["provision_ifrs9"]
    print(f"\nProvision (LGD {p['lgd_supuesta']:.0%}, EAD = monto originado)")
    print(f"  ECL todo a 12 meses    : CLP {p['ecl_solo_12m_clp']:,.0f}")
    print(f"  ECL con staging IFRS 9 : CLP {p['ecl_con_staging_ifrs9_clp']:,.0f} "
          f"(+{p['uplift_pct']:.1f}%)")
    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "data/processed", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
