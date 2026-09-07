"""Graficos del pipeline bayesiano jerarquico.

Se alimentan de los CSV/JSON que dejan las etapas anteriores. La unica
excepcion son las trazas, que necesitan las cadenas crudas y por eso
vuelven a muestrear un ajuste corto con la misma semilla.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import FEATURES
from src.diagnostics import effective_sample_size, split_rhat
from src.hierarchical_logit import HierarchicalLogit
from src.preprocessing import (
    SEGMENTOS, design_matrix, design_names, segment_index,
)

BASE = Path(__file__).resolve().parents[2]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"
PLOTS_DIR = BASE / "outputs" / "plots"

AZUL = "#1f4e79"
NARANJA = "#e07b39"
VERDE = "#1b7837"
GRIS = "#7a7a7a"


def _save(fig, nombre: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}")


# ----------------------------------------------------------------------
def plot_shrinkage():
    parcial = pd.read_csv(REPORTS_DIR / "efectos_segmento_partial.csv")
    sin_pool = pd.read_csv(REPORTS_DIR / "efectos_segmento_none.csv")
    df = parcial.merge(sin_pool, on="segmento", suffixes=("_parcial", "_sin"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.scatter(df["n_train_parcial"], df["efecto_post_centrado_sin"], s=45,
                color=NARANJA, label="Sin pooling")
    ax1.scatter(df["n_train_parcial"], df["efecto_post_centrado_parcial"], s=45,
                color=AZUL, label="Pooling parcial")
    for _, f in df.iterrows():
        ax1.plot([f["n_train_parcial"]] * 2,
                 [f["efecto_post_centrado_sin"], f["efecto_post_centrado_parcial"]],
                 color=GRIS, lw=0.8, alpha=0.6)
    ax1.axhline(0, color=GRIS, ls="--", lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("Casos del segmento en train (escala log)")
    ax1.set_ylabel("Efecto del segmento (log-odds)")
    ax1.set_title("Shrinkage: los segmentos con menos datos\nson los que mas se corren al promedio")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    lim = 1.15 * max(df["efecto_true_parcial"].abs().max(),
                     df["efecto_post_centrado_sin"].abs().max())
    ax2.plot([-lim, lim], [-lim, lim], color=GRIS, ls="--", lw=1)
    ax2.scatter(df["efecto_true_parcial"], df["efecto_post_centrado_sin"], s=45,
                color=NARANJA, label="Sin pooling")
    ax2.scatter(df["efecto_true_parcial"], df["efecto_post_centrado_parcial"], s=45,
                color=AZUL, label="Pooling parcial")
    ax2.set_xlabel("Efecto verdadero del simulador")
    ax2.set_ylabel("Efecto estimado")
    ax2.set_title("Recuperacion del efecto de segmento")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "shrinkage_por_segmento.png")


def plot_posterior_vs_verdad():
    res = json.loads((REPORTS_DIR / "fit_results.json").read_text())
    diag = pd.read_csv(REPORTS_DIR / "diagnostico_partial.csv")
    nombres = design_names()
    diag = diag.set_index("parametro").loc[nombres].reset_index()
    beta = res["partial"]["beta_posterior"]

    y = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.hlines(y, diag["q05"], diag["q95"], color=AZUL, lw=2.5, label="IC creible 90%")
    ax.scatter(diag["media"], y, color=AZUL, zorder=3, s=45, label="Media posterior")
    ax.scatter([beta[n]["true"] for n in nombres], y, color=NARANJA, marker="D",
               zorder=4, s=55, label="Valor verdadero")
    ax.axvline(0, color=GRIS, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(nombres)
    ax.set_xlabel("Coeficiente (log-odds)")
    ax.set_title("Posterior de los efectos globales contra la verdad de terreno\n"
                 "pooling parcial, 4 cadenas x 3.000 draws")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="x")
    _save(fig, "posterior_vs_verdad.png")


def plot_trazas():
    """Trazas de tau y de dos coeficientes, con R-hat y ESS calculados."""
    train = pd.read_csv(PROC_DIR / "train.csv")
    draws = HierarchicalLogit(pooling="partial").fit(
        design_matrix(train), train["default_12m"].to_numpy(float),
        segment_index(train), len(SEGMENTOS), design_names(), SEGMENTOS,
        n_draws=1500, n_warmup=500, n_chains=4, seed=42,
    )
    gt = json.loads((RAW_DIR / "ground_truth.json").read_text())

    series = [
        ("tau", draws.tau, gt["tau_true"]),
        ("moras_z", draws.beta[:, :, design_names().index("moras_z")],
         gt["beta_true"]["moras_z"]),
        ("dti_z", draws.beta[:, :, design_names().index("dti_z")],
         gt["beta_true"]["dti_z"]),
    ]

    fig, axes = plt.subplots(len(series), 1, figsize=(11, 8), sharex=True)
    for ax, (nombre, chains, verdad) in zip(axes, series):
        for c in range(chains.shape[0]):
            ax.plot(chains[c], lw=0.6, alpha=0.75)
        ax.axhline(verdad, color=NARANJA, ls="--", lw=1.8)
        ax.set_ylabel(nombre)
        ax.set_title(f"{nombre}: R-hat {split_rhat(chains):.4f} | "
                     f"ESS {effective_sample_size(chains):.0f} | "
                     f"verdad {verdad:.3f}", fontsize=10, loc="left")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Draw despues del warmup")
    fig.suptitle("Trazas del muestreador de Gibbs con aumentacion Polya-Gamma", y=0.995)
    fig.tight_layout()
    _save(fig, "trazas_mcmc.png")


def plot_incertidumbre_por_segmento():
    pred = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    train = pd.read_csv(PROC_DIR / "train.csv")
    tam = train["segmento"].value_counts()
    pred["n_train"] = pred["segmento"].map(tam).fillna(0).astype(int)
    g = pred.groupby("segmento").agg(
        n_train=("n_train", "first"),
        ancho_ic=("pd_q95", "mean"),
        pd_media=("pd_media", "mean"),
        sd=("pd_sd", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(g["n_train"], g["sd"], s=55, color=AZUL)
    for seg, f in g.iterrows():
        if f["n_train"] < 60:
            ax.annotate(seg.split("|")[0][:12], (f["n_train"], f["sd"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points", color=GRIS)
    ax.set_xscale("log")
    ax.set_xlabel("Casos del segmento en train (escala log)")
    ax.set_ylabel("Desviacion posterior media de la PD")
    ax.set_title("El modelo sabe donde no sabe:\nla incertidumbre de la PD crece "
                 "donde el segmento tiene poca historia")
    ax.grid(alpha=0.25)
    _save(fig, "incertidumbre_por_segmento.png")


def plot_frontera_decision():
    fr = pd.read_csv(REPORTS_DIR / "frontera_decision.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for nombre, color in [("media", AZUL), ("conservadora", NARANJA)]:
        sub = fr[fr["politica"] == nombre].sort_values("tasa_aprobacion")
        ax1.plot(sub["tasa_aprobacion"], sub["tasa_mala_realizada"], marker="o",
                 color=color, lw=2, label=f"Politica: {nombre}")
        ax2.plot(sub["tasa_aprobacion"], sub["utilidad_realizada_clp"] / 1e6, marker="o",
                 color=color, lw=2, label=f"Politica: {nombre}")

    ax1.set_xlabel("Tasa de aprobacion")
    ax1.set_ylabel("Tasa de default realizada entre aprobados")
    ax1.set_title("Riesgo realizado a igual volumen aprobado")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    ax2.set_xlabel("Tasa de aprobacion")
    ax2.set_ylabel("Utilidad realizada (millones CLP)")
    ax2.set_title("Utilidad realizada (margen 7%, LGD 45%)")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "frontera_decision.png")


def plot_calibracion():
    pred = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.plot([0, 0.75], [0, 0.75], color=GRIS, ls="--", lw=1, label="Calibracion perfecta")

    for col, nombre, color, marca in [
        ("pd_media_pooling_completo", "Pooling completo", VERDE, "s"),
        ("pd_media_sin_pooling", "Sin pooling", NARANJA, "^"),
        ("pd_media", "Pooling parcial", AZUL, "o"),
    ]:
        deciles = pd.qcut(pred[col], 10, labels=False, duplicates="drop")
        g = pred.groupby(deciles).agg(pred=(col, "mean"), obs=("default_12m", "mean"))
        ax.plot(g["pred"], g["obs"], marker=marca, color=color, lw=1.4, label=nombre)

    ax.set_xlabel("PD media posterior (decil)")
    ax.set_ylabel("Tasa de default observada")
    ax.set_title("Calibracion por decil en test\n2.700 solicitudes")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "calibracion_modelos.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_shrinkage()
    plot_posterior_vs_verdad()
    plot_incertidumbre_por_segmento()
    plot_frontera_decision()
    plot_calibracion()
    plot_trazas()


if __name__ == "__main__":
    main()
