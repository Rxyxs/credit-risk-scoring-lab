"""Graficos de la cartera real (Lending Club), desde los reportes de `src.study`."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.lab import TRIGGER_MAX_PSI, TRIGGER_MIN_AUC

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "outputs" / "reports"
PLOTS_DIR = BASE / "outputs" / "plots"

AZUL = "#1f4e79"
NARANJA = "#e07b39"
VERDE = "#1b7837"
ROJO = "#c0392b"
GRIS = "#7a7a7a"


def _save(fig, nombre: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}")


def _results() -> dict:
    return json.loads((REPORTS_DIR / "lending_club_results.json").read_text(encoding="utf-8"))


def plot_modelos(r: dict):
    b, m = r["binning_06"], r["models_01_05"]
    filas = [
        ("Scorecard, binning por deciles", b["equifrecuente"]["test_2015"]["auc"], GRIS),
        ("Scorecard, DP libre", b["dp_libre"]["test_2015"]["auc"], GRIS),
        ("Scorecard, DP monotono", b["dp_monotono"]["test_2015"]["auc"], AZUL),
        ("Scorecard, arbol", b["arbol"]["test_2015"]["auc"], GRIS),
        ("Gradient boosting libre", m["gbm_libre"]["test_2015"]["auc"], NARANJA),
        ("Gradient boosting monotono", m["gbm_monotono"]["test_2015"]["auc"], VERDE),
        ("Sub-grado de Lending Club", m["lending_club_sub_grade"]["test_2015"]["auc"], ROJO),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(filas))
    ax.barh(y, [f[1] for f in filas], color=[f[2] for f in filas])
    for i, (_, v, _) in enumerate(filas):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([f[0] for f in filas])
    ax.set_xlim(0.6, 0.7)
    ax.set_xlabel("AUC en los creditos de 2015 (entrenado con 2007-2013)")
    ax.set_title("Con datos reales, los cuatro binnings empatan, el boosting le gana al scorecard\n"
                 "y la restriccion monotona no cuesta nada; nadie alcanza al sub-grado del prestamista",
                 fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    _save(fig, "comparacion_modelos.png")


def plot_momento_del_default(r: dict):
    t = r["timing_03"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for clave, estilo, etiqueta in [("last_payment", dict(lw=2.4), "Fecha = ultimo pago"),
                                    ("ninety_days", dict(lw=1.4, ls="--"), "Fecha = 90 dias de mora")]:
        curva = t[clave]["curve"]
        ax.plot(np.arange(len(curva)), curva, color=AZUL, label=etiqueta, **estilo)
    ax.axvline(12, color=GRIS, lw=1)
    ax.text(12.4, 0.01, "mes 12", color=GRIS, fontsize=8)
    share = t["last_payment"]["km_share_after_12m"]
    ax.set_title(f"El {share:.0%} del riesgo de vida de un credito a 36 meses llega despues del mes 12\n"
                 "(la cartera simulada de la tecnica 03: 50,5%)", fontsize=10)
    ax.set_xlabel("Meses desde el desembolso")
    ax.set_ylabel("PD acumulada (Kaplan-Meier, prepagos censurados)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    _save(fig, "momento_del_default.png")


def plot_cosechas(r: dict):
    serie = pd.read_csv(REPORTS_DIR / "vintage_portfolio.csv")
    serie = serie[serie["n"] >= 100]
    rho = pd.DataFrame(r["vintages_10"]["by_grade"])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(serie))
    ax1.plot(x, 100 * serie["rate"], color=ROJO, lw=2, label="Tasa de default de la cosecha (%)")
    ax1.plot(x, 100 * serie["grade_mix_a_b_share"] / 4, color=GRIS, lw=1.4, ls=":",
             label="Participacion de grados A y B (% / 4)")
    ax1.set_ylabel("%")
    ax2 = ax1.twinx()
    ax2.plot(x, serie["unemployment_first_18m"], color=AZUL, lw=1.6, ls="--",
             label="Desempleo EE.UU., primeros 18 meses (%, eje der.)")
    ax2.set_ylabel("Desempleo (%)")
    paso = 4
    ax1.set_xticks(x[::paso])
    ax1.set_xticklabels(serie["quarter"].iloc[::paso], rotation=45, fontsize=8)
    lineas = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lineas, [l.get_label() for l in lineas], frameon=False, fontsize=8, loc="upper right")
    ax1.set_title("Las cosechas de 2007-2008 cayeron en la recesion; despues, la tasa sigue\n"
                  "a quien le presto Lending Club, no al desempleo", fontsize=10)
    ax1.grid(alpha=0.25)

    g = np.arange(len(rho))
    ancho = 0.2
    ax3.bar(g - 1.5 * ancho, rho["rho_mom_with_n"], ancho, color=AZUL, label="Momentos con N")
    ax3.bar(g - 0.5 * ancho, rho["rho_mom_without_n"], ancho, color=AZUL, alpha=0.45, label="Momentos sin N")
    ax3.bar(g + 0.5 * ancho, rho["rho_mom_with_n_detrended"], ancho, color=VERDE, label="Con N, sin tendencia")
    ax3.bar(g + 1.5 * ancho, rho["rho_basel_retail"], ancho, color=GRIS, label="Basilea, minorista (PD anual)")
    ax3.set_xticks(g)
    ax3.set_xticklabels(rho["grade"])
    ax3.set_ylabel("Correlacion de activos rho")
    ax3.set_title("rho por grado, desde 33-34 cosechas trimestrales", fontsize=10)
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    _save(fig, "cosechas_y_correlacion.png")


def plot_estabilidad(r: dict):
    q = pd.DataFrame(r["stability_12_22"]["quarters"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(q))
    etiquetas = q["quarter"].tolist()

    ax = axes[0]
    ax.plot(x, q["auc"], "o-", color=AZUL)
    ax.axhline(TRIGGER_MIN_AUC, color=ROJO, ls="--", lw=1)
    ax.text(0, TRIGGER_MIN_AUC + 0.002, f"gatillo de la tecnica 22: AUC < {TRIGGER_MIN_AUC}", color=ROJO, fontsize=8)
    ax.set_ylim(0.6, 0.74)
    ax.set_title("AUC realizado por trimestre", fontsize=10)

    ax = axes[1]
    ax.plot(x, q["psi_score"], "o-", color=AZUL, label="PSI del puntaje")
    ax.plot(x, q["max_psi"], "s-", color=NARANJA, label="PSI maximo (puntaje y 6 variables)")
    ax.axhline(TRIGGER_MAX_PSI, color=ROJO, ls="--", lw=1)
    ax.text(0, TRIGGER_MAX_PSI - 0.015, f"gatillo: PSI > {TRIGGER_MAX_PSI}", color=ROJO, fontsize=8)
    ax.set_ylim(0, 0.25)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("PSI contra la poblacion 2007-2013", fontsize=10)

    ax = axes[2]
    ax.plot(x, 100 * q["observed_rate"], "o-", color=ROJO, label="Default observado")
    ax.plot(x, 100 * q["mean_pd"], "s-", color=AZUL, label="PD promedio del scorecard")
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Lo que si cambio: la tasa de default", fontsize=10)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, rotation=45, fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("El gatillo por AUC se dispara todos los trimestres, el de PSI ninguno, "
                 "y ninguno ve que la PD quedo corta", y=1.03, fontsize=11)
    fig.tight_layout()
    _save(fig, "estabilidad_trimestral.png")


def main():
    r = _results()
    plot_modelos(r)
    plot_momento_del_default(r)
    plot_cosechas(r)
    plot_estabilidad(r)


if __name__ == "__main__":
    main()
