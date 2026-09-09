"""Graficos del experimento de scoring federado."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import BANCOS

BASE = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE / "outputs" / "reports"
PLOTS_DIR = BASE / "outputs" / "plots"

AZUL = "#1f4e79"
NARANJA = "#e07b39"
VERDE = "#1b7837"
ROJO = "#c0392b"
GRIS = "#7a7a7a"

NOMBRES_CORTOS = {b: b.replace("Banco_", "").replace("_", " ") for b in BANCOS}


def _save(fig, nombre: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {nombre}")


def plot_calibracion_por_banco():
    res = json.loads((REPORTS_DIR / "experiment_results.json").read_text())
    r = res["resumen_nacional"]["no_iid"]
    bancos = list(BANCOS)
    sesgos = [r["solo_local_por_banco"][b]["sesgo_calibracion_pp"] for b in bancos]
    colores = [ROJO if abs(s) > 5 else AZUL for s in sesgos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax1.barh([NOMBRES_CORTOS[b] for b in bancos], sesgos, color=colores)
    ax1.axvline(0, color="black", lw=1)
    ax1.set_xlabel("Sesgo de calibracion nacional (pp): PD predicha − PD real del pais")
    ax1.set_title("Modelo entrenado SOLO con la cartera propia,\naplicado a la poblacion nacional")
    ax1.grid(alpha=0.25, axis="x")

    politicas = ["solo_local_promedio", "federado", "oraculo_centralizado"]
    etiquetas = ["Solo local\n(promedio)", "Federado\n(FedAvg)", "Oraculo\ncentralizado"]
    sesgos_pol = [r[p]["sesgo_calibracion_pp"] for p in politicas]
    colores_pol = [NARANJA, AZUL, VERDE]
    ax2.bar(etiquetas, sesgos_pol, color=colores_pol)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_ylabel("Sesgo de calibracion nacional (pp)")
    ax2.set_title("Promedio entre bancos, por politica\n(escenario no-IID)")
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("El ranking (AUC) casi no se mueve; el NIVEL de PD es donde\n"
                 "un modelo entrenado solo localmente se rompe", y=1.03)
    fig.tight_layout()
    _save(fig, "calibracion_por_banco.png")


def plot_comparacion_politicas():
    res = json.loads((REPORTS_DIR / "experiment_results.json").read_text())
    escenarios = ["iid", "no_iid"]
    politicas = ["solo_local_promedio", "federado", "oraculo_centralizado"]
    etiquetas_pol = ["Solo local", "Federado (FedAvg)", "Oraculo centralizado"]
    colores = [NARANJA, AZUL, VERDE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(escenarios))
    ancho = 0.26
    for i, (pol, etq, color) in enumerate(zip(politicas, etiquetas_pol, colores)):
        aucs = [res["resumen_nacional"][e][pol]["auc"] for e in escenarios]
        ax1.bar(x + (i - 1) * ancho, aucs, ancho, color=color, label=etq)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["IID", "No-IID"])
    ax1.set_ylabel("AUC sobre el test nacional")
    ax1.set_ylim(0.65, 0.82)
    ax1.set_title("Discriminacion (orden)")
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.grid(alpha=0.25, axis="y")

    for i, (pol, etq, color) in enumerate(zip(politicas, etiquetas_pol, colores)):
        brier = [res["resumen_nacional"][e][pol]["brier"] for e in escenarios]
        ax2.bar(x + (i - 1) * ancho, brier, ancho, color=color, label=etq)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["IID", "No-IID"])
    ax2.set_ylabel("Brier score sobre el test nacional (menor es mejor)")
    ax2.set_title("Calidad probabilistica total")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("En datos IID las tres politicas casi no se distinguen;\n"
                 "la heterogeneidad es lo que separa federado de solo-local", y=1.03)
    fig.tight_layout()
    _save(fig, "comparacion_politicas.png")


def plot_similitud_updates():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, escenario, titulo in [(axes[0], "iid", "IID"), (axes[1], "no_iid", "No-IID")]:
        sim = pd.read_csv(REPORTS_DIR / f"similitud_updates_{escenario}.csv", index_col=0)
        etiquetas = [NOMBRES_CORTOS[b] for b in sim.index]
        im = ax.imshow(sim.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(etiquetas)))
        ax.set_yticklabels(etiquetas, fontsize=8)
        ax.set_title(f"Escenario {titulo}", fontsize=10)
        for i in range(len(etiquetas)):
            for j in range(len(etiquetas)):
                ax.text(j, i, f"{sim.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(axes[1].images[0], ax=axes, fraction=0.025, pad=0.02,
                label="Similitud coseno del update de la ronda 1")
    fig.suptitle("Lo que un servidor honesto-pero-curioso ve en la primera ronda,\n"
                 "sin haber recibido nunca los datos crudos de nadie", y=1.02)
    _save(fig, "similitud_updates.png")


def plot_barrido_epocas():
    barrido = pd.read_csv(REPORTS_DIR / "barrido_epocas_locales.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax1.plot(barrido["local_epochs"], barrido["nacional_auc"], marker="o", color=AZUL, lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("Epocas locales por ronda (escala log)")
    ax1.set_ylabel("AUC sobre el test nacional")
    ax1.set_title("El orden casi no se resiente")
    ax1.grid(alpha=0.25)

    ax2.plot(barrido["local_epochs"], barrido["nacional_sesgo_calibracion_pp"],
             marker="o", color=ROJO, lw=2)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("Epocas locales por ronda (escala log)")
    ax2.set_ylabel("Sesgo de calibracion nacional (pp)")
    ax2.set_title("El nivel de PD si: mas computo local\npor ronda aleja al modelo antes de corregir")
    ax2.grid(alpha=0.25)

    fig.suptitle("El dial de comunicacion de FedAvg (mismo computo total, distinta\n"
                 "frecuencia de sincronizacion), escenario no-IID", y=1.03)
    fig.tight_layout()
    _save(fig, "barrido_epocas_locales.png")


def plot_historia_convergencia():
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for escenario, color, etiqueta in [("iid", AZUL, "IID"), ("no_iid", NARANJA, "No-IID")]:
        hist = pd.read_csv(REPORTS_DIR / f"historia_fedavg_{escenario}.csv")
        ax.plot(hist["ronda"], hist["deriva_media_clientes"], color=color, lw=2, label=etiqueta)
    ax.set_xlabel("Ronda de comunicacion")
    ax.set_ylabel("Distancia media entre el modelo global\ny los modelos locales antes de agregar")
    ax.set_title("Cuanto se alejan los bancos del modelo global\nen cada ronda, antes de que FedAvg los corrija")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, "historia_convergencia.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_calibracion_por_banco()
    plot_comparacion_politicas()
    plot_similitud_updates()
    plot_barrido_epocas()
    plot_historia_convergencia()


if __name__ == "__main__":
    main()
