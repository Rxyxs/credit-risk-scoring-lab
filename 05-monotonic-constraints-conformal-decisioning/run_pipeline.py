"""Orquestador end-to-end de la tecnica 05 (monotonia + prediccion conforme).

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
    ("Simular cartera (riesgo monotono + edad en U + interacciones)", "src.data_generator"),
    ("Split en train / calibracion / test", "src.preprocessing"),
    ("Entrenar logistica, GBM libre y GBM monotono + auditoria", "src.models"),
    ("Calibracion conforme y decision en tres vias", "src.conformal_run"),
    ("Stress: que pasa con la garantia si cambia la poblacion", "src.shift_stress"),
    ("Graficos de resultados", "src.visualization.plots"),
]


def run_step(descripcion: str, modulo: str):
    print(f"\n{'=' * 74}\n>> {descripcion}\n{'=' * 74}")
    r = subprocess.run([sys.executable, "-m", modulo], cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Etapa fallida: {modulo} (codigo {r.returncode})")


def resumen():
    modelos = json.loads((REPORTS_DIR / "model_results.json").read_text())
    conf = json.loads((REPORTS_DIR / "conformal_results.json").read_text())
    shift = json.loads((REPORTS_DIR / "shift_results.json").read_text())

    print(f"\n{'=' * 74}\n>> RESUMEN\n{'=' * 74}")
    print(f"{'modelo':<14} {'AUC':>7} {'KS':>7} {'Brier':>8} {'% casos con violacion':>22}")
    for nombre in ("logistica", "gbm_libre", "gbm_monotono"):
        m = modelos[nombre]["metricas_test"]
        a = modelos[nombre]["auditoria_monotonia"]
        print(f"{nombre:<14} {m['auc']:>7.4f} {m['ks']:>7.4f} {m['brier']:>8.4f} "
              f"{a['pct_casos_con_violacion_max']:>21.2%}")

    c = modelos["costo_de_la_restriccion"]
    print(f"\nCosto de la restriccion de monotonia: {c['delta_auc_pct']:+.2f}% de AUC "
          f"({c['delta_gini_pp']:+.2f} pp de Gini)")

    cob = [r for r in conf["cobertura"]
           if r["variante"] == "mondrian" and abs(r["alpha"] - conf["alpha_operativo"]) < 1e-9][0]
    marg = [r for r in conf["cobertura"]
            if r["variante"] == "marginal" and abs(r["alpha"] - conf["alpha_operativo"]) < 1e-9][0]
    print(f"\nCobertura conforme al alpha {conf['alpha_operativo']} "
          f"(objetivo {cob['cobertura_objetivo']:.2f})")
    print(f"  Mondrian : global {cob['cobertura_global']:.4f} | "
          f"clase 0 {cob['cobertura_clase_0']:.4f} | clase 1 {cob['cobertura_clase_1']:.4f}")
    print(f"  Marginal : global {marg['cobertura_global']:.4f} | "
          f"clase 0 {marg['cobertura_clase_0']:.4f} | clase 1 {marg['cobertura_clase_1']:.4f}")

    print("\nDecision en tres vias, al mismo volumen de revision manual")
    for p in conf["politicas_al_alpha_operativo"]:
        print(f"  {p['politica']:<22} aprueba {p['pct_aprobado']:.1%} | "
              f"revisa {p['pct_revision_manual']:.1%} | "
              f"error automatico {p['tasa_error_decisiones_automaticas']:.2%} | "
              f"utilidad CLP {p['utilidad_clp']:,.0f}")

    cs = shift["cobertura_deterioro_calibracion_vieja"]
    cr = shift["cobertura_deterioro_recalibrado"]
    print(f"\nBajo deterioro macro (default {shift['tasa_default_normal']:.2%} -> "
          f"{shift['tasa_default_deterioro']:.2%})")
    print(f"  Cobertura clase 0 con calibracion vieja : {cs['cobertura_clase_0']:.4f} "
          f"(objetivo {cs['cobertura_objetivo']:.2f})")
    print(f"  Cobertura clase 0 tras recalibrar       : {cr['cobertura_clase_0']:.4f}")
    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "data/processed", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
