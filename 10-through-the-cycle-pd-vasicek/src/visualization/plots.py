"""Graficos del experimento Vasicek/ASRF: PIT vs TTC."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import ANOS_RECESION, GRADOS

BASE = Path(__file__).resolve().parents[2]
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


def _sombrear_recesiones(ax):
    for anio in sorted(ANOS_RECESION):
        ax.axvspan(anio - 0.5, anio + 0.5, color=ROJO, alpha=0.08)


def plot_correlacion_por_grado():
    tabla = pd.read_csv(REPORTS_DIR / "correlacion_estimada.csv")
    x = np.arange(len(GRADOS))
    ancho = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.bar(x - ancho, tabla["rho_true"], ancho, color=GRIS, label="rho verdadero (simulador)")
    ax.bar(x, tabla["rho_hat_mom"], ancho, color=AZUL, label="rho estimado (metodo de momentos)")
    ax.bar(x + ancho, tabla["rho_basel_en_pd_hat"], ancho, color=VERDE,
           label="Formula de Basilea, evaluada en la PD estimada")
    ax.set_xticks(x)
    ax.set_xticklabels(GRADOS)
    ax.set_ylabel("Correlacion de activos (rho)")
    ax.set_title("Recuperando la correlacion de activos desde la serie de\n"
                 "tasas de default -- sin haber observado nunca el ciclo")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "correlacion_por_grado.png")


def plot_recuperacion_ciclo():
    ciclo = pd.read_csv(REPORTS_DIR / "ciclo_recuperado.csv")
    res = json.loads((REPORTS_DIR / "experiment_results.json").read_text())["recuperacion_ciclo"]

    fig, ax = plt.subplots(figsize=(11, 5))
    _sombrear_recesiones(ax)
    ax.plot(ciclo["anio"], ciclo["z_verdadero"], color=GRIS, lw=2.4, label="Z verdadero (nunca observado)")
    ax.plot(ciclo["anio"], ciclo["z_recuperado"], color=AZUL, lw=1.8, ls="--",
            label="Z recuperado (promedio entre los 5 grados)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Factor sistematico Z (ciclo economico)")
    ax.set_title(f"El ciclo economico reconstruido solo desde tasas de default agregadas\n"
                 f"correlacion con el verdadero: {res['correlacion']:.3f} "
                 f"(sombreado: anos de recesion marcados en el simulador)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "recuperacion_ciclo.png")


def plot_sesgo_granularidad():
    tabla = pd.read_csv(REPORTS_DIR / "sesgo_granularidad.csv")
    chica = tabla[tabla["tamano"] == "chica"].set_index("grado").loc[GRADOS]
    grande = tabla[tabla["tamano"] == "grande"].set_index("grado").loc[GRADOS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.arange(len(GRADOS))
    ancho = 0.26
    for ax, datos, titulo in [(ax1, chica, "Cartera CHICA (200 deudores/cohorte)"),
                              (ax2, grande, "Cartera GRANDE (20.000 deudores/cohorte)")]:
        ax.bar(x - ancho, datos["rho_true"], ancho, color=GRIS, label="rho verdadero")
        ax.bar(x, datos["rho_hat_mom"], ancho, color=AZUL,
               label="Metodo de momentos\n(exacto para cualquier N)")
        ax.bar(x + ancho, datos["rho_hat_asrf_limite"], ancho, color=ROJO,
               label="Limite ASRF\n(exacto solo si N->infinito)")
        ax.set_xticks(x)
        ax.set_xticklabels(GRADOS)
        ax.set_ylabel("rho estimado")
        ax.set_title(titulo, fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("El limite ASRF confunde ruido idiosincratico con riesgo sistematico\n"
                 "cuando la cartera no es lo bastante granular", y=1.03)
    fig.tight_layout()
    _save(fig, "sesgo_granularidad.png")


def plot_capital_pit_vs_ttc():
    capital = pd.read_csv(REPORTS_DIR / "capital_pit_vs_ttc.csv")

    fig, ax = plt.subplots(figsize=(11, 5.4))
    _sombrear_recesiones(ax)
    ax.plot(capital["anio"], capital["densidad_rwa_ttc_pct"], color=GRIS, lw=2.2,
            label="Densidad de RWA con PD through-the-cycle (estable por construccion)")
    ax.plot(capital["anio"], capital["densidad_rwa_pit_pct"], color=NARANJA, lw=2,
            label="Densidad de RWA con PD point-in-time (recalibrada cada ano)")
    ax.set_xlabel("Ano")
    ax.set_ylabel("RWA como % de la exposicion")
    ax.set_title("Prociclicidad del capital regulatorio:\n"
                 "usar la PD del ciclo en vez de la PD promedio dispara el capital justo en las recesiones")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, "capital_pit_vs_ttc.png")


def plot_verificacion_vasicek():
    muestras = pd.read_csv(REPORTS_DIR / "monte_carlo_muestras.csv")
    teorico = pd.read_csv(REPORTS_DIR / "monte_carlo_teorico.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax1.hist(muestras["tasa_default_mc"], bins=60, density=True, color=AZUL, alpha=0.55,
             label="Monte Carlo (5.000 carteras simuladas)")
    ax1.plot(teorico["x"], teorico["densidad_cerrada"], color=ROJO, lw=2.2,
             label="Densidad cerrada de Vasicek")
    ax1.set_xlabel("Tasa de default de la cartera")
    ax1.set_ylabel("Densidad")
    ax1.set_title("Grado BB: formula cerrada contra\nsimulacion independiente")
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.grid(alpha=0.25)

    emp = np.sort(muestras["tasa_default_mc"].to_numpy())
    cdf_emp = np.arange(1, len(emp) + 1) / len(emp)
    ax2.plot(emp, cdf_emp, color=AZUL, lw=2.4, label="CDF empirica (Monte Carlo)")
    ax2.plot(teorico["x"], teorico["cdf_cerrada"], color=ROJO, lw=1.6, ls="--",
             label="CDF cerrada de Vasicek")
    ax2.set_xlabel("Tasa de default de la cartera")
    ax2.set_ylabel("Probabilidad acumulada")
    ax2.set_title("Misma verificacion, en CDF")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "verificacion_vasicek.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_correlacion_por_grado()
    plot_recuperacion_ciclo()
    plot_sesgo_granularidad()
    plot_capital_pit_vs_ttc()
    plot_verificacion_vasicek()


if __name__ == "__main__":
    main()
