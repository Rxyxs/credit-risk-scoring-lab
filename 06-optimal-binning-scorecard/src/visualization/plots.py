"""Graficos del scorecard: binning, tarjeta, bandas y monitoreo."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.binning import OptimalBinner, binning_arbol, binning_equifrecuente
from src.data_generator import VINTAGE_QUIEBRE
from src.monitoring import UMBRAL_ALERTA, UMBRAL_CRITICO

BASE = Path(__file__).resolve().parents[2]
PROC_DIR = BASE / "data" / "processed"
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


def plot_woe_por_metodo():
    train = pd.read_csv(PROC_DIR / "train.csv")
    y = train["default_12m"].to_numpy(float)
    variables = ["dti", "utilizacion_lineas", "edad"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, var in zip(axes, variables):
        x = train[var].to_numpy(float)
        resultados = {
            "DP monotono": (OptimalBinner().fit_numerica(x, y, var), AZUL, "o"),
            "Equifrecuente": (binning_equifrecuente(x, y, 6, var), GRIS, "s"),
            "Arbol": (binning_arbol(x, y, 6, nombre=var), NARANJA, "^"),
        }
        for etiqueta, (res, color, marca) in resultados.items():
            woe = res.tabla["woe"].to_numpy()
            sufijo = "" if res.monotono else "  (no monotono)"
            ax.plot(np.arange(len(woe)), woe, marker=marca, color=color, lw=2,
                    label=f"{etiqueta} · IV {res.iv:.3f}{sufijo}")
        ax.axhline(0, color=GRIS, lw=1, ls="--")
        ax.set_xlabel("Bin (de menor a mayor valor de la variable)")
        ax.set_ylabel("WOE")
        ax.set_title(var)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Como queda el WOE segun quien elija los cortes", y=1.02)
    fig.tight_layout()
    _save(fig, "woe_por_metodo.png")


def plot_comparacion_metodos():
    res = json.loads((REPORTS_DIR / "scorecard_results.json").read_text())
    metodos = list(res["metodos"])
    iv = [res["metodos"][m]["iv_total"] for m in metodos]
    auc_in = [res["metodos"][m]["in_time"]["auc"] for m in metodos]
    auc_out = [res["metodos"][m]["out_of_time"]["auc"] for m in metodos]
    no_mono = [len(res["metodos"][m]["variables_no_monotonas"]) for m in metodos]

    x = np.arange(len(metodos))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

    ax1.bar(x, iv, color=AZUL, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metodos, rotation=20, ha="right")
    ax1.set_ylabel("IV total (suma de variables)")
    ax1.set_title("Poder predictivo del binning")
    ax1.grid(alpha=0.25, axis="y")

    ax2.bar(x - 0.2, auc_in, 0.38, color=AZUL, label="Validacion in-time")
    ax2.bar(x + 0.2, auc_out, 0.38, color=NARANJA, label="Out-of-time (deterioro)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metodos, rotation=20, ha="right")
    ax2.set_ylim(0.70, 0.80)
    ax2.set_ylabel("AUC del scorecard")
    ax2.set_title("Desempeno del scorecard resultante")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    colores = [VERDE if n == 0 else ROJO for n in no_mono]
    ax3.bar(x, no_mono, color=colores, width=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metodos, rotation=20, ha="right")
    ax3.set_ylabel("Variables con WOE no monotono")
    ax3.set_title("Lo que hay que defender en el comite")
    ax3.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, "comparacion_metodos.png")


def plot_tarjeta():
    tarjeta = pd.read_csv(REPORTS_DIR / "tarjeta_scorecard.csv")
    iv = pd.read_csv(REPORTS_DIR / "iv_por_variable_dp_monotono.csv")
    top = iv.head(6)["variable"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, var in zip(axes.ravel(), top):
        sub = tarjeta[tarjeta["variable"] == var]
        colores = [VERDE if p >= sub["puntos"].mean() else ROJO for p in sub["puntos"]]
        ax.barh(np.arange(len(sub)), sub["puntos"], color=colores)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["etiqueta"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Puntos")
        fila_iv = iv[iv["variable"] == var].iloc[0]
        ax.set_title(f"{var} · IV {fila_iv['iv']:.3f} ({fila_iv['lectura']})", fontsize=10)
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("Tarjeta de puntos: cuanto suma o resta cada tramo (PDO 20)", y=1.0)
    fig.tight_layout()
    _save(fig, "tarjeta_puntos.png")


def plot_bandas_y_distribucion():
    bandas = pd.read_csv(REPORTS_DIR / "bandas_de_riesgo.csv")
    s_in = pd.read_csv(REPORTS_DIR / "scores_in_time.csv")
    s_out = pd.read_csv(REPORTS_DIR / "scores_out_of_time.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colores = {"A": VERDE, "B": "#7fbf7b", "C": "#f0a202", "D": NARANJA, "E": ROJO}
    ax1.bar(bandas["banda"], bandas["tasa_mala"] * 100,
            color=[colores[b] for b in bandas["banda"]])
    for _, f in bandas.iterrows():
        ax1.text(f["banda"], f["tasa_mala"] * 100 + 0.6,
                 f"{f['tasa_mala']:.1%}\n{int(f['score_min'])}-{int(f['score_max'])}",
                 ha="center", fontsize=8)
    ax1.set_ylabel("Tasa de default observada (%)")
    ax1.set_xlabel("Banda de riesgo (quintiles del puntaje)")
    ax1.set_title("Separacion por banda, validacion in-time")
    ax1.grid(alpha=0.25, axis="y")

    ax2.hist(s_in["score"], bins=45, alpha=0.65, color=AZUL, density=True,
             label=f"In-time (media {s_in['score'].mean():.0f})")
    ax2.hist(s_out["score"], bins=45, alpha=0.65, color=NARANJA, density=True,
             label=f"Out-of-time (media {s_out['score'].mean():.0f})")
    ax2.set_xlabel("Puntaje del scorecard")
    ax2.set_ylabel("Densidad")
    ax2.set_title("El deterioro de la poblacion corre la distribucion del puntaje")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "bandas_y_distribucion.png")


def plot_monitoreo():
    tabla = pd.read_csv(REPORTS_DIR / "monitoreo_por_vintage.csv")
    cols_csi = [c for c in tabla.columns if c.startswith("csi_")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    ax1.plot(tabla["vintage_idx"], tabla["psi_score"], marker="o", color=AZUL, lw=2,
             label="PSI del puntaje")
    ax1.axhline(UMBRAL_ALERTA, color=NARANJA, ls="--", lw=1.4, label="Alerta (0.10)")
    ax1.axhline(UMBRAL_CRITICO, color=ROJO, ls="--", lw=1.4, label="Critico (0.25)")
    ax1.axvline(VINTAGE_QUIEBRE - 0.5, color=GRIS, ls=":", lw=1.6)
    ax1.text(VINTAGE_QUIEBRE - 0.4, tabla["psi_score"].max() * 0.85,
             "quiebre real\nde poblacion", fontsize=8, color=GRIS)
    ax1b = ax1.twinx()
    ax1b.plot(tabla["vintage_idx"], tabla["tasa_mala"] * 100, color=VERDE, lw=1.6,
              ls="-.", label="Tasa de default (%)")
    ax1b.set_ylabel("Tasa de default observada (%)", color=VERDE)
    ax1.set_xlabel("Vintage")
    ax1.set_ylabel("PSI")
    ax1.set_title("El PSI levanta el cambio de poblacion\nen la misma cohorte en que ocurre")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25)

    m = tabla[cols_csi].to_numpy().T
    im = ax2.imshow(m, aspect="auto", cmap="OrRd", vmin=0, vmax=min(m.max(), 0.6))
    ax2.set_yticks(np.arange(len(cols_csi)))
    ax2.set_yticklabels([c.replace("csi_", "") for c in cols_csi], fontsize=8)
    ax2.set_xticks(np.arange(len(tabla)))
    ax2.set_xticklabels(tabla["vintage_idx"], fontsize=7)
    ax2.set_xlabel("Vintage")
    ax2.set_title("CSI por variable: cual se movio\n(escala recortada en 0.6)")
    fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)

    fig.tight_layout()
    _save(fig, "psi_csi_por_vintage.png")


def plot_convergencia():
    conv = pd.read_csv(REPORTS_DIR / "convergencia_grilla.csv")
    arbol_iv = json.loads((REPORTS_DIR / "scorecard_results.json").read_text())

    fig, ax = plt.subplots(figsize=(8.2, 5))
    ax.plot(conv["n_prebins"], conv["iv"], marker="o", color=AZUL, lw=2,
            label="DP (optimo sobre la grilla)")
    ax.set_xscale("log")
    ax.set_xlabel("Prebins usados por la programacion dinamica (escala log)")
    ax.set_ylabel("IV alcanzado en `dti`")
    ax.set_title("La DP es exacta sobre su grilla:\nrefinar la grilla es lo que cierra la brecha")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _ = arbol_iv
    _save(fig, "convergencia_grilla.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_woe_por_metodo()
    plot_comparacion_metodos()
    plot_tarjeta()
    plot_bandas_y_distribucion()
    plot_monitoreo()
    plot_convergencia()


if __name__ == "__main__":
    main()
