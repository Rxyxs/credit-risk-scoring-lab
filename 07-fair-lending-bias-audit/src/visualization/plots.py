"""Graficos de la auditoria de trato justo."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import GRUPO_PROTEGIDO, GRUPO_REFERENCIA
from src.fairness_metrics import UMBRAL_REGLA_CUATRO_QUINTOS, umbral_por_tasa_de_aprobacion

BASE = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE / "outputs" / "reports"
PLOTS_DIR = BASE / "outputs" / "plots"

AZUL = "#1f4e79"
NARANJA = "#e07b39"
VERDE = "#1b7837"
ROJO = "#c0392b"
GRIS = "#7a7a7a"
COLOR_GRUPO = {GRUPO_PROTEGIDO: "#7b52ab", GRUPO_REFERENCIA: "#1f4e79"}


def _save(fig, nombre: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}")


def plot_proxies():
    tabla = pd.read_csv(REPORTS_DIR / "fuerza_de_proxy.csv")
    res = json.loads((REPORTS_DIR / "audit_results.json").read_text())
    auc_recon = res["proxies"]["reconstruccion"]["auc_reconstruccion_grupo"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    colores = [ROJO if s else AZUL for s in tabla["es_proxy_sospechoso"]]
    ax1.scatter(tabla["senal_riesgo"], tabla["senal_grupo"], s=90, color=colores)
    lim = max(tabla["senal_grupo"].max(), tabla["senal_riesgo"].max()) * 1.2
    ax1.plot([0, lim], [0, lim], color=GRIS, ls="--", lw=1,
             label="Igual senal de grupo que de riesgo")
    for _, f in tabla.iterrows():
        ax1.annotate(f["feature"], (f["senal_riesgo"], f["senal_grupo"]),
                     fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax1.set_xlabel("Senal de riesgo (AUC de predecir default − 0.5)")
    ax1.set_ylabel("Senal de grupo (AUC de predecir el genero − 0.5)")
    ax1.set_title("Cada variable: .cuanto aporta al riesgo\ny cuanto delata al grupo?")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25)

    orden = tabla.sort_values("razon_proxy")
    colores2 = [ROJO if s else AZUL for s in orden["es_proxy_sospechoso"]]
    ax2.barh(orden["feature"], orden["razon_proxy"], color=colores2)
    ax2.axvline(1.0, color=GRIS, ls="--", lw=1.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Razon proxy = senal de grupo / senal de riesgo (escala log)")
    ax2.set_title(f"El modelo nunca ve el genero,\npero puede reconstruirlo con AUC {auc_recon:.3f}")
    ax2.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    _save(fig, "deteccion_de_proxies.png")


def plot_escenarios():
    res = json.loads((REPORTS_DIR / "audit_results.json").read_text())
    esc = res["escenarios"]
    nombres = list(esc)
    x = np.arange(len(nombres))

    ratio = [esc[n]["equidad"]["ratio_impacto_adverso"] for n in nombres]
    paridad = [esc[n]["equidad"]["paridad_demografica_pp"] for n in nombres]
    oportunidad = [esc[n]["equidad"]["igualdad_oportunidad_pp"] for n in nombres]
    utilidad = [esc[n]["economia"]["utilidad_clp"] / 1e6 for n in nombres]
    auc = [esc[n]["auc"] for n in nombres]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13.5, 9))

    ax1.bar(x, ratio, color=[VERDE if r >= UMBRAL_REGLA_CUATRO_QUINTOS else ROJO for r in ratio])
    ax1.axhline(UMBRAL_REGLA_CUATRO_QUINTOS, color=ROJO, ls="--", lw=1.4,
                label="Regla de los 4/5 (0.80)")
    ax1.axhline(1.0, color=GRIS, ls=":", lw=1.2, label="Paridad perfecta")
    ax1.set_ylim(0.75, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=15, ha="right")
    ax1.set_ylabel("Ratio de impacto adverso")
    ax1.set_title("Impacto adverso por escenario")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25, axis="y")

    ax2.bar(x - 0.2, paridad, 0.38, color=AZUL, label="Paridad demografica")
    ax2.bar(x + 0.2, oportunidad, 0.38, color=NARANJA, label="Igualdad de oportunidad")
    ax2.axhline(0, color=GRIS, lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres, rotation=15, ha="right")
    ax2.set_ylabel("Brecha (puntos porcentuales)")
    ax2.set_title("Brechas del grupo protegido respecto del de referencia")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    ax3.bar(x, utilidad, color=AZUL)
    ax3.set_xticks(x)
    ax3.set_xticklabels(nombres, rotation=15, ha="right")
    ax3.set_ylabel("Utilidad realizada (millones CLP)")
    ax3.set_title("Lo que cuesta cada mitigacion")
    ax3.grid(alpha=0.25, axis="y")

    ax4.bar(x, auc, color=GRIS)
    ax4.set_ylim(0.68, 0.73)
    ax4.set_xticks(x)
    ax4.set_xticklabels(nombres, rotation=15, ha="right")
    ax4.set_ylabel("AUC en test")
    ax4.set_title("Poder predictivo (casi no se mueve)")
    ax4.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, "escenarios_de_mitigacion.png")


def plot_descomposicion():
    res = json.loads((REPORTS_DIR / "audit_results.json").read_text())
    d = res["descomposicion_de_la_brecha"]
    estratos = pd.read_csv(REPORTS_DIR / "brecha_condicional.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    valores = [d["brecha_bruta"] * 100, d["brecha_condicional"] * 100]
    barras = ax1.bar(["Brecha bruta", "Brecha entre\nperfiles equivalentes"], valores,
                     color=[NARANJA, AZUL], width=0.55)
    for b, v in zip(barras, valores):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:+.3f} pp",
                 ha="center", fontsize=10)
    ax1.axhline(0, color=GRIS, lw=1)
    ax1.set_ylabel("Diferencia de PD predicha (pp)")
    ax1.set_title(f"{d['pct_explicado_por_factores_legitimos']:.1f}% de la brecha se explica\n"
                  f"por renta y carga financiera")
    ax1.grid(alpha=0.25, axis="y")

    ax2.hist(estratos["brecha"] * 100, bins=18, color=AZUL, alpha=0.8)
    ax2.axvline(0, color=GRIS, lw=1.4)
    ax2.axvline(d["brecha_condicional"] * 100, color=ROJO, ls="--", lw=1.6,
                label="Promedio ponderado")
    ax2.set_xlabel("Brecha de PD dentro del estrato (pp)")
    ax2.set_ylabel("Estratos")
    ax2.set_title(f"Brecha dentro de cada estrato de riesgo\n({d['n_estratos']} estratos comparables)")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, "descomposicion_de_la_brecha.png")


def plot_frontera():
    fr = pd.read_csv(REPORTS_DIR / "frontera_equidad_utilidad.csv")
    fig, ax1 = plt.subplots(figsize=(9, 5.4))

    ax1.plot(fr["tasa_aprobacion"] * 100, fr["ratio_impacto_adverso"], marker="o",
             color=AZUL, lw=2, label="Ratio de impacto adverso")
    ax1.axhline(UMBRAL_REGLA_CUATRO_QUINTOS, color=ROJO, ls="--", lw=1.4,
                label="Regla de los 4/5")
    ax1.set_xlabel("Tasa de aprobacion de la politica (%)")
    ax1.set_ylabel("Ratio de impacto adverso", color=AZUL)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(fr["tasa_aprobacion"] * 100, fr["utilidad_clp"] / 1e6, marker="s",
             color=NARANJA, lw=2, label="Utilidad (MM CLP)")
    ax2.set_ylabel("Utilidad realizada (millones CLP)", color=NARANJA)

    lineas = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    etiquetas = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lineas, etiquetas, frameon=False, fontsize=9, loc="lower right")
    ax1.set_title("Cuanto mas se aprueba, mas parejo queda el reparto\n"
                  "(y donde eso deja de convenir)")
    fig.tight_layout()
    _save(fig, "frontera_equidad_utilidad.png")


def plot_calibracion_y_distribucion():
    pred = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    umbral = umbral_por_tasa_de_aprobacion(pred["pd_base"].to_numpy(), 0.80)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for grupo, g in pred.groupby("genero"):
        deciles = pd.qcut(g["pd_base"], 10, labels=False, duplicates="drop")
        t = g.groupby(deciles).agg(pred=("pd_base", "mean"), obs=("default_12m", "mean"))
        ax1.plot(t["pred"], t["obs"], marker="o", color=COLOR_GRUPO[grupo], lw=1.8,
                 label=f"Grupo {grupo} (n={len(g):,})")
    lim = pred["pd_base"].quantile(0.995)
    ax1.plot([0, lim], [0, lim], color=GRIS, ls="--", lw=1, label="Calibracion perfecta")
    ax1.set_xlabel("PD media predicha (decil)")
    ax1.set_ylabel("Tasa de default observada")
    ax1.set_title("Calibracion por grupo\nun modelo puede estar bien calibrado y aun asi repartir distinto")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    for grupo, g in pred.groupby("genero"):
        ax2.hist(g["pd_base"], bins=45, alpha=0.55, density=True,
                 color=COLOR_GRUPO[grupo], label=f"Grupo {grupo}")
    ax2.axvline(umbral, color=ROJO, ls="--", lw=1.6,
                label=f"Umbral al 80% de aprobacion ({umbral:.3f})")
    ax2.set_xlabel("PD predicha por el modelo base")
    ax2.set_ylabel("Densidad")
    ax2.set_title("Distribucion de la PD por grupo")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "calibracion_y_distribucion.png")


def plot_intervalos():
    ic = pd.read_csv(REPORTS_DIR / "intervalos_bootstrap.csv")
    ic = ic[ic["metrica"] != "ratio_impacto_adverso"]
    y = np.arange(len(ic))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.hlines(y, ic["ic_inferior"], ic["ic_superior"], color=AZUL, lw=3)
    ax.scatter(ic["estimacion"], y, color=NARANJA, zorder=3, s=60)
    ax.axvline(0, color=GRIS, ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(ic["metrica"])
    ax.set_xlabel("Brecha en puntos porcentuales (IC 95% por bootstrap)")
    ax.set_title("Una brecha sin barra de error no es un hallazgo")
    ax.grid(alpha=0.25, axis="x")
    _save(fig, "intervalos_bootstrap.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_proxies()
    plot_escenarios()
    plot_descomposicion()
    plot_frontera()
    plot_calibracion_y_distribucion()
    plot_intervalos()


if __name__ == "__main__":
    main()
