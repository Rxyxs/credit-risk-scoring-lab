"""Graficos del experimento de privacidad diferencial."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.accountant import calibrar_sigma, epsilon_del_entrenamiento
from src.attacks import ataque_de_membresia
from src.data_generator import FEATURES, generar_canarios
from src.dp_sgd import DPLogisticRegression

BASE = Path(__file__).resolve().parents[2]
RAW_DIR = BASE / "data" / "raw"
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


def _tabla() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "privacidad_vs_utilidad.csv")


def plot_privacidad_vs_utilidad():
    t = _tabla()
    priv = t[t["escenario"] != "sin_privacidad"].sort_values("epsilon_real")
    base = t[t["escenario"] == "sin_privacidad"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax1.errorbar(priv["epsilon_real"], priv["utilidad_auc"],
                 yerr=priv["utilidad_auc_sd"], marker="o", color=AZUL, lw=2,
                 capsize=4, label="DP-SGD")
    ax1.axhline(base["utilidad_auc"], color=NARANJA, ls="--", lw=1.8,
                label=f"Sin privacidad ({base['utilidad_auc']:.4f})")
    ax1.axhline(0.5, color=GRIS, ls=":", lw=1.4, label="Azar")
    ax1.set_xscale("log")
    ax1.set_xlabel("Presupuesto de privacidad epsilon (escala log)")
    ax1.set_ylabel("AUC en test")
    ax1.set_title("Lo que cuesta la privacidad\n(promedio de 10 corridas, +- 1 desviacion)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.errorbar(priv["epsilon_real"], priv["canario_exposicion_pp"],
                 yerr=priv["canario_exposicion_pp_sd"], marker="o", color=AZUL,
                 lw=2, capsize=4, label="Exposicion de canarios")
    ax2.axhline(base["canario_exposicion_pp"], color=NARANJA, ls="--", lw=1.8,
                label=f"Sin privacidad ({base['canario_exposicion_pp']:.2f} pp)")
    ax2.axhline(0, color=GRIS, lw=1.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Presupuesto de privacidad epsilon (escala log)")
    ax2.set_ylabel("PD de canarios vistos − no vistos (pp)")
    ax2.set_title("Lo que compra la privacidad\n(memorizacion de registros inyectados)")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "privacidad_vs_utilidad.png")


def plot_detectabilidad():
    t = _tabla().sort_values("epsilon_real")
    etiquetas = [e.replace("_", " ") for e in t["escenario"]]
    x = np.arange(len(t))
    colores = [ROJO if d else VERDE for d in t["fuga_detectable"]]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x, t["canario_t"], color=colores)
    ax.axhline(2.0, color=GRIS, ls="--", lw=1.4, label="Umbral de deteccion (|t| = 2)")
    ax.axhline(-2.0, color=GRIS, ls="--", lw=1.4)
    ax.set_yscale("symlog", linthresh=2)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, rotation=15, ha="right")
    ax.set_ylabel("t de la exposicion de canarios (escala symlog)")
    ax.set_title("La pregunta no es cuanta fuga hay, sino si se puede distinguir de cero\n"
                 "rojo = fuga detectable entre corridas")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "detectabilidad_de_la_fuga.png")


def plot_calibracion_de_ruido():
    q_actual = json.loads(
        (REPORTS_DIR / "experiment_results.json").read_text()
    )["configuracion"]["q"]
    epsilons = np.array([0.25, 0.5, 1, 2, 4, 8, 16])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for pasos, color in [(500, VERDE), (2500, AZUL), (10_000, NARANJA)]:
        sigmas = [calibrar_sigma(q_actual, pasos, float(e)) for e in epsilons]
        ax1.plot(epsilons, sigmas, marker="o", color=color, lw=2,
                 label=f"{pasos:,} pasos")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epsilon objetivo (escala log)")
    ax1.set_ylabel("Ruido necesario sigma (escala log)")
    ax1.set_title("Mas pasos de entrenamiento exigen mas ruido\npara el mismo presupuesto")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25)

    pasos_grilla = np.array([100, 250, 500, 1000, 2500, 5000, 10_000])
    for sigma, color in [(2.0, VERDE), (5.0, AZUL), (12.0, NARANJA)]:
        eps = [epsilon_del_entrenamiento(q_actual, sigma, int(p))["epsilon"]
               for p in pasos_grilla]
        ax2.plot(pasos_grilla, eps, marker="s", color=color, lw=2,
                 label=f"sigma = {sigma}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Pasos de entrenamiento (escala log)")
    ax2.set_ylabel("Epsilon gastado (escala log)")
    ax2.set_title("La composicion es acumulativa:\ncada paso gasta presupuesto")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "calibracion_de_ruido.png")


def plot_curvas_y_coeficientes():
    perdidas = pd.read_csv(REPORTS_DIR / "curvas_de_perdida.csv")
    coefs = pd.read_csv(REPORTS_DIR / "coeficientes.csv", index_col=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    colores = plt.cm.viridis(np.linspace(0, 0.9, len(perdidas.columns)))
    for col, color in zip(perdidas.columns, colores):
        ax1.plot(perdidas[col].rolling(25, min_periods=1).mean(), lw=1.6,
                 color=color, label=col.replace("_", " "))
    ax1.set_xlabel("Paso de DP-SGD")
    ax1.set_ylabel("Perdida de entrenamiento (media movil de 25)")
    ax1.set_title("El ruido no solo frena el aprendizaje:\ntambien impide que se estabilice")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.25)

    y = np.arange(len(coefs))
    ancho = 0.8 / len(coefs.columns)
    for i, (col, color) in enumerate(zip(coefs.columns, colores)):
        ax2.barh(y + i * ancho - 0.4, coefs[col], ancho, color=color,
                 label=col.replace("_", " "))
    ax2.set_yticks(y)
    ax2.set_yticklabels(coefs.index, fontsize=8)
    ax2.axvline(0, color=GRIS, lw=1)
    ax2.set_xlabel("Coeficiente")
    ax2.set_title("Con epsilon chico los coeficientes\ndejan de parecerse al modelo real")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    _save(fig, "entrenamiento_y_coeficientes.png")


def plot_ataque():
    """Distribucion de perdidas de miembros vs no miembros, con y sin DP."""
    train = pd.read_csv(RAW_DIR / "train.csv")
    holdout = pd.read_csv(RAW_DIR / "holdout.csv")
    canarios = pd.read_csv(RAW_DIR / "canarios.csv")
    sombra = generar_canarios(len(canarios), np.random.default_rng(42 + 777))

    mu = train[FEATURES].mean()
    sd = train[FEATURES].std(ddof=0).replace(0, 1.0)
    def X(df):
        return ((df[FEATURES] - mu) / sd).to_numpy(float)

    cfg = json.loads((REPORTS_DIR / "experiment_results.json").read_text())["configuracion"]
    escenarios = {
        "Sin privacidad": 0.0,
        "epsilon = 1": calibrar_sigma(cfg["q"], cfg["pasos"], 1.0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for ax, (nombre, sigma) in zip(axes[:2], escenarios.items()):
        modelo = DPLogisticRegression(clip=cfg["clip"], sigma=sigma, q=cfg["q"],
                                      pasos=cfg["pasos"], lr=cfg["lr"], seed=42)
        modelo.fit(X(train), train["default_12m"].to_numpy(float))
        p_in = modelo.perdida_por_ejemplo(X(train), train["default_12m"].to_numpy(float))
        p_out = modelo.perdida_por_ejemplo(X(holdout), holdout["default_12m"].to_numpy(float))
        res = ataque_de_membresia(modelo, X(train), train["default_12m"].to_numpy(float),
                                  X(holdout), holdout["default_12m"].to_numpy(float))

        ax.hist(p_in, bins=45, alpha=0.6, density=True, color=AZUL, label="Miembros (train)")
        ax.hist(p_out, bins=45, alpha=0.6, density=True, color=NARANJA,
                label="No miembros (holdout)")
        ax.set_xlabel("Perdida por ejemplo")
        ax.set_ylabel("Densidad")
        ax.set_title(f"{nombre}\nAUC del ataque {res['auc_ataque']:.4f}")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)

        if nombre == "Sin privacidad":
            pd_can = modelo.predict_proba(X(canarios))[:, 1]
            pd_som = modelo.predict_proba(X(sombra))[:, 1]
            axes[2].hist(pd_som, bins=20, alpha=0.65, color=GRIS, density=True,
                         label="Canarios no vistos")
            axes[2].hist(pd_can, bins=20, alpha=0.65, color=ROJO, density=True,
                         label="Canarios en el entrenamiento")

    axes[2].set_xlabel("PD asignada por el modelo sin privacidad")
    axes[2].set_ylabel("Densidad")
    axes[2].set_title("Memorizacion visible:\nel modelo trata distinto a los que vio")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "ataque_de_membresia.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_privacidad_vs_utilidad()
    plot_detectabilidad()
    plot_calibracion_de_ruido()
    plot_curvas_y_coeficientes()
    plot_ataque()


if __name__ == "__main__":
    main()
