"""Graficos del experimento de reject inference."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE / "outputs" / "reports"
PLOTS_DIR = BASE / "outputs" / "plots"

AZUL = "#1f4e79"
NARANJA = "#e07b39"
VERDE = "#1b7837"
ROJO = "#c0392b"
GRIS = "#7a7a7a"

ORDEN_METODOS = [
    "aprobados_solo", "ipw", "parcelling", "augmentacion_em",
    "heckman_sin_instrumento", "probit_bivariado_sin_instrumento",
    "heckman_2etapas", "probit_bivariado", "oraculo",
]
ETIQUETAS = {
    "aprobados_solo": "Solo\naprobados", "ipw": "IPW", "parcelling": "Parcelling",
    "augmentacion_em": "Aumentacion\nEM",
    "heckman_sin_instrumento": "Heckman\nsin instrumento",
    "probit_bivariado_sin_instrumento": "Probit biv.\nsin instrumento",
    "heckman_2etapas": "Heckman\ncon instrumento",
    "probit_bivariado": "Probit biv.\ncon instrumento",
    "oraculo": "Oraculo",
}


def _save(fig, nombre: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}")


def plot_recuperacion_rho():
    res = json.loads((REPORTS_DIR / "experiment_results.json").read_text())
    rho_v = res["rho_verdadero"]
    rho_e = res["rho_estimado_probit_bivariado"]
    regimenes = list(rho_v)

    x = np.arange(len(regimenes))
    ancho = 0.28
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(x - ancho, [rho_v[r] for r in regimenes], ancho, color=GRIS,
           label="rho verdadero")
    ax.bar(x, [rho_e[r]["sin_instrumento"] for r in regimenes], ancho, color=ROJO,
           label="Probit bivariado, sin instrumento")
    ax.bar(x + ancho, [rho_e[r]["con_instrumento"] for r in regimenes], ancho,
           color=VERDE, label="Probit bivariado, con instrumento")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.upper() for r in regimenes])
    ax.set_ylabel("rho (correlacion entre los errores)")
    ax.set_title("Sin variable de exclusion, rho se estima mal en cualquier\n"
                 "direccion: falso positivo en MAR, falso negativo en MNAR")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "recuperacion_de_rho.png")


def plot_comparacion_metodos():
    tabla = pd.read_csv(REPORTS_DIR / "comparacion_metodos.csv")
    orden = [m for m in ORDEN_METODOS if m in tabla["metodo"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, regimen, color in [(ax1, "mar", AZUL), (ax2, "mnar", NARANJA)]:
        sub = tabla[tabla["regimen"] == regimen].set_index("metodo").loc[orden]
        colores = [VERDE if m == "oraculo" else
                  (ROJO if "sin_instrumento" in m else color) for m in orden]
        ax.bar(range(len(orden)), sub["error_abs_medio_coef"], color=colores)
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels([ETIQUETAS[m] for m in orden], fontsize=7.5, rotation=0)
        ax.set_ylabel("Error absoluto medio de los coeficientes")
        ax.set_title(f"Regimen {regimen.upper()}")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Que tan cerca de la verdad queda cada metodo\n"
                 "(rojo = version sin variable de exclusion, verde = cota del oraculo)", y=1.04)
    fig.tight_layout()
    _save(fig, "comparacion_metodos.png")


def plot_sesgo_de_nivel():
    tabla = pd.read_csv(REPORTS_DIR / "comparacion_metodos.csv")
    orden = [m for m in ORDEN_METODOS if m in tabla["metodo"].unique()]
    mnar = tabla[tabla["regimen"] == "mnar"].set_index("metodo").loc[orden]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    colores = [VERDE if m == "oraculo" else
              (ROJO if "sin_instrumento" in m else AZUL) for m in orden]
    ax.bar(range(len(orden)), mnar["error_nivel_pp"], color=colores)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([ETIQUETAS[m] for m in orden], fontsize=8)
    ax.set_ylabel("Error de nivel: PD estimada − PD real de la poblacion (pp)")
    ax.set_title("Regimen MNAR: el numero con el que se aprovisiona,\n"
                 "no solo el orden de riesgo")
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "sesgo_de_nivel_mnar.png")


def plot_sensibilidad_parcelling():
    sens = pd.read_csv(REPORTS_DIR / "sensibilidad_parcelling.csv")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(sens["factor"], sens["error_nivel_pp"], marker="o", color=AZUL, lw=2)
    ax.axhline(0, color=GRIS, ls="--", lw=1.2)
    ax.fill_between(sens["factor"], -2, 2, color=VERDE, alpha=0.12,
                    label="Zona de error de nivel chico (±2 pp)")
    ax.set_xlabel("Factor de castigo de parcelling")
    ax.set_ylabel("Error de nivel (PD estimada − real, pp)")
    ax.set_title("Parcelling depende de un numero que nadie puede validar\n"
                 "en produccion (aca se conoce solo porque los datos son simulados)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "sensibilidad_parcelling.png")


def plot_auc_por_poblacion():
    tabla = pd.read_csv(REPORTS_DIR / "comparacion_metodos.csv")
    orden = [m for m in ORDEN_METODOS if m in tabla["metodo"].unique()]
    mnar = tabla[tabla["regimen"] == "mnar"].set_index("metodo").loc[orden]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(orden))
    ancho = 0.38
    ax.bar(x - ancho / 2, mnar["auc_aprobados"], ancho, color=AZUL,
           label="AUC en aprobados (la region comoda)")
    ax.bar(x + ancho / 2, mnar["auc_rechazados"], ancho, color=NARANJA,
           label="AUC en rechazados (la region que importa)")
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETAS[m] for m in orden], fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.55, 0.85)
    ax.set_title("Regimen MNAR: todos discriminan bien entre los aprobados;\n"
                 "la brecha real esta en como les va con los rechazados")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "auc_aprobados_vs_rechazados.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_recuperacion_rho()
    plot_comparacion_metodos()
    plot_sesgo_de_nivel()
    plot_sensibilidad_parcelling()
    plot_auc_por_poblacion()


if __name__ == "__main__":
    main()
