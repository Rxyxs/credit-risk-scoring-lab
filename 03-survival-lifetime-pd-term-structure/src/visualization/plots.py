"""Graficos de resultados del pipeline de supervivencia.

Todos salen de los CSV/JSON que dejan las etapas anteriores: el modulo no
reajusta nada, solo dibuja lo que efectivamente se estimo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import kaplan_meier

BASE = Path(__file__).resolve().parents[2]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"
PLOTS_DIR = BASE / "outputs" / "plots"

COLORS = {
    "A": "#1b7837", "B": "#7fbf7b", "C": "#f0a202",
    "D": "#e07b39", "E": "#c0392b",
}
AZUL = "#1f4e79"
NARANJA = "#e07b39"


def _save(fig, nombre: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = PLOTS_DIR / nombre
    fig.savefig(ruta, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta.name}")


# ----------------------------------------------------------------------
def plot_km_por_banda():
    test = pd.read_csv(PROC_DIR / "test_loans.csv")
    curvas = np.load(REPORTS_DIR / "discrete_pd_curves_test.npy")
    test["pd_12m"] = curvas[:, 11]
    test["banda"] = pd.qcut(test["pd_12m"], 5, labels=list(COLORS))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for banda, g in test.groupby("banda", observed=True):
        km = kaplan_meier(g["duracion_meses"], g["evento_default"])
        ax.step(km["tiempo"], km["cum_default"], where="post",
                color=COLORS[banda], lw=2, label=f"Banda {banda} (n={len(g)})")
        lo = np.clip(km["cum_default"] - 1.96 * km["se_greenwood"], 0, 1)
        hi = np.clip(km["cum_default"] + 1.96 * km["se_greenwood"], 0, 1)
        ax.fill_between(km["tiempo"], lo, hi, step="post", color=COLORS[banda], alpha=0.15)

    ax.set_xlabel("Meses desde la originacion")
    ax.set_ylabel("PD acumulada observada (Kaplan-Meier)")
    ax.set_title("Curvas de default observadas por banda de riesgo\n"
                 "(banda IC 95% de Greenwood, cartera de test)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "km_por_banda.png")


def plot_estructura_temporal():
    tabla = pd.read_csv(REPORTS_DIR / "pd_term_structure_por_banda.csv")
    agregada = pd.read_csv(REPORTS_DIR / "pd_curva_agregada.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    hitos = [6, 12, 24, 36]
    for _, fila in tabla.iterrows():
        ax1.plot(hitos, [fila[f"pd_pred_{h}m"] for h in hitos], marker="o",
                 color=COLORS[fila["banda"]], lw=2, label=f"Banda {fila['banda']}")
    ax1.axvline(12, color="grey", ls="--", lw=1)
    ax1.text(12.4, ax1.get_ylim()[1] * 0.92, "corte 12m\n(Stage 1)", fontsize=8, color="grey")
    ax1.set_xlabel("Horizonte (meses)")
    ax1.set_ylabel("PD acumulada predicha")
    ax1.set_title("Estructura temporal de PD por banda")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.bar(agregada["mes"], agregada["pd_marginal"], color=AZUL, alpha=0.75,
            label="PD marginal (cartera)")
    ax2b = ax2.twinx()
    ax2b.plot(agregada["mes"], agregada["pd_acumulada"], color=NARANJA, lw=2.2,
              label="PD acumulada")
    ax2.set_xlabel("Mes desde la originacion")
    ax2.set_ylabel("PD marginal del mes")
    ax2b.set_ylabel("PD acumulada")
    ax2.set_title("Seasoning: cuando llega el riesgo")
    lineas = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    etiquetas = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2.legend(lineas, etiquetas, frameon=False, fontsize=8)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "pd_term_structure.png")


def plot_coeficientes_vs_verdad():
    df = pd.read_csv(REPORTS_DIR / "cox_coefficients_vs_truth.csv")
    df = df.dropna(subset=["beta_true"]).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    err = 1.96 * df["se_efron"]
    ax.errorbar(df["coef_efron"], y, xerr=err, fmt="o", color=AZUL,
                capsize=3, label="Cox (Efron), IC 95%")
    ax.scatter(df["beta_true"], y, marker="D", color=NARANJA, zorder=3,
               label="Valor verdadero del simulador")
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Coeficiente (log-hazard)")
    ax.set_title("Recuperacion de los coeficientes verdaderos\n"
                 "Cox implementado desde cero, cartera de train")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="x")
    _save(fig, "coeficientes_vs_verdad.png")


def plot_ph_test():
    ph = pd.read_csv(REPORTS_DIR / "cox_ph_test.csv")
    ph = ph.sort_values("p_value")
    colores = [NARANJA if v else AZUL for v in ph["viola_ph"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(ph["feature"], -np.log10(ph["p_value"]), color=colores)
    ax.axvline(-np.log10(0.01), color="grey", ls="--", lw=1.2)
    ax.text(-np.log10(0.01) + 0.05, 0.1, "p = 0.01", fontsize=8, color="grey")
    ax.set_xlabel("-log10(p) del test de Schoenfeld")
    ax.set_title("Supuesto de hazards proporcionales por covariable\n"
                 "(naranjo = el efecto cambia con el tiempo)")
    ax.grid(alpha=0.25, axis="x")
    _save(fig, "ph_test_schoenfeld.png")


def plot_calibracion():
    cox = pd.read_csv(REPORTS_DIR / "cox_calibration_12m.csv")
    dis = pd.read_csv(REPORTS_DIR / "discrete_calibration_12m.csv")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    lim = max(cox["pd_observada_km"].max(), dis["pd_observada_km"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1, label="Calibracion perfecta")
    ax.scatter(cox["pd_predicha"], cox["pd_observada_km"], s=55, color=AZUL,
               label="Cox (Efron)")
    ax.scatter(dis["pd_predicha"], dis["pd_observada_km"], s=55, color=NARANJA,
               marker="^", label="Hazard discreto")
    ax.set_xlabel("PD a 12 meses predicha (media del decil)")
    ax.set_ylabel("PD a 12 meses observada (Kaplan-Meier)")
    ax.set_title("Calibracion por decil de riesgo\ncartera de test")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "calibracion_12m.png")


def plot_baseline_hazard():
    base = pd.read_csv(REPORTS_DIR / "discrete_baseline_hazard.csv")
    cox = pd.read_csv(REPORTS_DIR / "cox_baseline_hazard.csv")
    gt = json.loads((BASE / "data" / "raw" / "ground_truth.json").read_text())
    a = gt["baseline"]
    meses = base["mes"].to_numpy(dtype=float)
    verdadero = np.exp(a["a0"] + a["a1"] * np.log(meses) + a["a2"] * np.log(meses) ** 2)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(meses, base["hazard_mensual_tvc"], color=AZUL, lw=1.2, alpha=0.55,
            label="Hazard discreto (dummy por mes)")
    ax.plot(meses, base["hazard_mensual_tvc"].rolling(3, center=True, min_periods=1).mean(),
            color=AZUL, lw=2.4, label="Hazard discreto (suavizado 3m)")
    ax.plot(cox["mes"], cox["hazard_base"], color="#7a7a7a", lw=1.2, ls=":",
            label="Hazard base de Cox (Breslow)")
    ax.plot(meses, verdadero, color=NARANJA, lw=2.4, ls="--",
            label="Hazard verdadero del simulador")
    ax.set_xlabel("Mes desde la originacion")
    ax.set_ylabel("Hazard mensual condicional")
    ax.set_title("Forma del riesgo en el tiempo: estimado vs verdad de terreno")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, "hazard_base_vs_verdad.png")


def plot_ecl_ifrs9():
    res = json.loads((REPORTS_DIR / "term_structure_results.json").read_text())
    p = res["provision_ifrs9"]
    tabla = pd.DataFrame(res["bandas"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    valores = [p["ecl_solo_12m_clp"] / 1e6, p["ecl_con_staging_ifrs9_clp"] / 1e6]
    barras = ax1.bar(["Todo a 12 meses", "Staging IFRS 9\n(12m + lifetime)"], valores,
                     color=[AZUL, NARANJA], width=0.55)
    for b, v in zip(barras, valores):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.01, f"CLP {v:,.0f}M",
                 ha="center", fontsize=9)
    ax1.set_ylabel("Perdida esperada (millones CLP)")
    ax1.set_title(f"Provision de la cartera de test\n"
                  f"+{p['uplift_pct']:.1f}% al reconocer la vida completa "
                  f"del {p['pct_cartera_stage_2']:.1%} en Stage 2")
    ax1.grid(alpha=0.25, axis="y")

    ancho = 0.38
    x = np.arange(len(tabla))
    ax2.bar(x - ancho / 2, tabla["pd_pred_12m"], ancho, color=AZUL, label="PD 12 meses")
    ax2.bar(x + ancho / 2, tabla["pd_pred_36m"], ancho, color=NARANJA, label="PD lifetime (36m)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Banda {b}" for b in tabla["banda"]])
    ax2.set_ylabel("PD acumulada")
    ax2.set_title("Cuanto crece la PD al pasar de 12 meses a vida completa")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, "ecl_ifrs9.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_km_por_banda()
    plot_estructura_temporal()
    plot_coeficientes_vs_verdad()
    plot_ph_test()
    plot_calibracion()
    plot_baseline_hazard()
    plot_ecl_ifrs9()


if __name__ == "__main__":
    main()
