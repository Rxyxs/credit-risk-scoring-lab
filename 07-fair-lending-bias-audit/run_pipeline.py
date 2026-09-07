"""Orquestador end-to-end de la tecnica 07 (auditoria de trato justo).

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
    ("Simular cartera: genero fuera del proceso generador, proxy sectorial",
     "src.data_generator"),
    ("Auditoria: proxies, brechas con IC, y cuatro escenarios de mitigacion",
     "src.audit_run"),
    ("Graficos de resultados", "src.visualization.plots"),
]


def run_step(descripcion: str, modulo: str):
    print(f"\n{'=' * 74}\n>> {descripcion}\n{'=' * 74}")
    r = subprocess.run([sys.executable, "-m", modulo], cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Etapa fallida: {modulo} (codigo {r.returncode})")


def resumen():
    res = json.loads((REPORTS_DIR / "audit_results.json").read_text())

    print(f"\n{'=' * 74}\n>> RESUMEN\n{'=' * 74}")
    px = res["proxies"]
    print(f"Atributo protegido: {res['atributo_protegido']} "
          f"(grupo {res['grupo_protegido']})")
    print(f"El modelo nunca lo usa, y aun asi puede reconstruirlo desde sus "
          f"features con AUC {px['reconstruccion']['auc_reconstruccion_grupo']:.4f}")
    print(f"Variables marcadas como proxy: {', '.join(px['features_sospechosas']) or 'ninguna'} "
          f"(razon proxy maxima {px['razon_proxy_maxima']:.1f}x)")

    d = res["descomposicion_de_la_brecha"]
    print(f"\nBrecha de PD predicha: {d['brecha_bruta'] * 100:+.3f} pp bruta -> "
          f"{d['brecha_condicional'] * 100:+.3f} pp entre perfiles equivalentes "
          f"({d['pct_explicado_por_factores_legitimos']:.1f}% explicado por "
          f"factores legitimos)")

    print(f"\nEscenarios, todos al {res['tasa_aprobacion_comparada']:.0%} de aprobacion")
    print(f"{'escenario':<18} {'AUC':>7} {'ratio 4/5':>10} {'paridad':>9} "
          f"{'utilidad MM CLP':>16}")
    for nombre, r in res["escenarios"].items():
        e, eco = r["equidad"], r["economia"]
        print(f"{nombre:<18} {r['auc']:>7.4f} {e['ratio_impacto_adverso']:>10.4f} "
              f"{e['paridad_demografica_pp']:>8.2f}p {eco['utilidad_clp'] / 1e6:>16,.1f}")

    ic = {r["metrica"]: r for r in res["intervalos_bootstrap"]}
    p = ic["paridad_demografica_pp"]
    print(f"\nParidad demografica del modelo base: {p['estimacion']:.2f} pp "
          f"[{p['ic_inferior']:.2f}, {p['ic_superior']:.2f}] al 95%")
    print(f"\nReportes en {REPORTS_DIR}")


def main():
    for d in ["data/raw", "data/processed", "outputs/plots", "outputs/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
    for descripcion, modulo in ETAPAS:
        run_step(descripcion, modulo)
    resumen()


if __name__ == "__main__":
    main()
