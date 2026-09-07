"""Graficos del pipeline de restricciones monotonas + prediccion conforme."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import entrenar
from src.monotonicity_audit import curva_respuesta
from src.preprocessing import matriz

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


def plot_auditoria():
    libre = pd.read_csv(REPORTS_DIR / "auditoria_monotonia_gbm_libre.csv")
    mono = pd.read_csv(REPORTS_DIR / "auditoria_monotonia_gbm_monotono.csv")
    df = libre.merge(mono, on="feature", suffixes=("_libre", "_mono"))
    y = np.arange(len(df))
    ancho = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.barh(y - ancho / 2, df["pct_casos_con_violacion_libre"] * 100, ancho,
             color=NARANJA, label="GBM sin restricciones")
    ax1.barh(y + ancho / 2, df["pct_casos_con_violacion_mono"] * 100, ancho,
             color=AZUL, label="GBM monotono")
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["feature"])
    ax1.set_xlabel("% de solicitantes con al menos una violacion")
    ax1.set_title("Auditoria contrafactual de monotonia\n(mover una variable, dejar el resto fijo)")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.25, axis="x")

    ax2.barh(y - ancho / 2, df["violacion_maxima_pd_libre"] * 100, ancho, color=NARANJA)
    ax2.barh(y + ancho / 2, df["violacion_maxima_pd_mono"] * 100, ancho, color=AZUL)
    ax2.set_yticks(y)
    ax2.set_yticklabels(df["feature"])
    ax2.set_xlabel("Mayor caida de PD en la direccion equivocada (puntos porcentuales)")
    ax2.set_title("Magnitud de la peor violacion")
    ax2.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    _save(fig, "auditoria_monotonia.png")


def plot_curvas_respuesta():
    train = pd.read_csv(PROC_DIR / "train.csv")
    test = pd.read_csv(PROC_DIR / "test.csv")
    modelos = entrenar(train)
    X = matriz(test)

    features = ["dti", "utilizacion_lineas", "edad"]
    titulos = {
        "dti": "Carga financiera (restringida: creciente)",
        "utilizacion_lineas": "Utilizacion de lineas (restringida: creciente)",
        "edad": "Edad (sin restringir: el efecto real es en U)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, feature in zip(axes, features):
        for nombre, color, estilo in [
            ("gbm_libre", NARANJA, "-"),
            ("gbm_monotono", AZUL, "-"),
        ]:
            curva = curva_respuesta(modelos[nombre], X, feature)
            ax.plot(curva["valor"], curva["pd_media"], color=color, ls=estilo, lw=2.2,
                    label=nombre.replace("_", " "))
        ax.set_xlabel(feature)
        ax.set_ylabel("PD media predicha")
        ax.set_title(titulos[feature], fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Respuesta del modelo a cada variable, con el resto fijo", y=1.02)
    fig.tight_layout()
    _save(fig, "curvas_respuesta.png")


def plot_cobertura():
    cob = pd.read_csv(REPORTS_DIR / "cobertura_conforme.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, variante, titulo in [
        (ax1, "mondrian", "Conformal Mondrian (por clase)"),
        (ax2, "marginal", "Conformal marginal (agrupado)"),
    ]:
        sub = cob[cob["variante"] == variante].sort_values("alpha")
        ax.plot(sub["cobertura_objetivo"], sub["cobertura_objetivo"], color=GRIS,
                ls="--", lw=1.2, label="Objetivo 1 - alpha")
        ax.plot(sub["cobertura_objetivo"], sub["cobertura_global"], marker="o",
                color=AZUL, lw=2, label="Cobertura global")
        ax.plot(sub["cobertura_objetivo"], sub["cobertura_clase_0"], marker="s",
                color=VERDE, lw=2, label="Clase 0 (paga)")
        ax.plot(sub["cobertura_objetivo"], sub["cobertura_clase_1"], marker="^",
                color=ROJO, lw=2, label="Clase 1 (default)")
        ax.set_xlabel("Cobertura objetivo (1 - alpha)")
        ax.set_title(titulo)
        ax.grid(alpha=0.25)
    ax1.set_ylabel("Cobertura empirica en test")
    ax1.legend(frameon=False, fontsize=9)
    fig.suptitle("La cobertura marginal se cumple globalmente sacrificando la clase minoritaria",
                 y=1.01)
    fig.tight_layout()
    _save(fig, "cobertura_conforme.png")


def plot_politicas():
    pol = pd.read_csv(REPORTS_DIR / "comparacion_politicas.csv")
    barrido = pd.read_csv(REPORTS_DIR / "barrido_alpha.csv").sort_values("alpha")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(pol))
    ax1.bar(x - 0.2, pol["tasa_error_decisiones_automaticas"] * 100, 0.38,
            color=AZUL, label="Error de las decisiones automaticas")
    ax1.bar(x + 0.2, pol["tasa_mala_entre_aprobados"] * 100, 0.38,
            color=NARANJA, label="Tasa mala entre aprobados")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Conformal\n(alpha 0.10)", "Banda de score\n(mismo volumen)"])
    ax1.set_ylabel("%")
    ax1.set_title("Dos formas de decidir a quien mandar a revision manual\n"
                  "comparadas al mismo volumen de revision")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25, axis="y")

    ax2.plot(barrido["pct_revision_manual"] * 100,
             barrido["tasa_error_decisiones_automaticas"] * 100,
             marker="o", color=AZUL, lw=2)
    for _, f in barrido.iterrows():
        ax2.annotate(f"alpha {f['alpha']:g}",
                     (f["pct_revision_manual"] * 100,
                      f["tasa_error_decisiones_automaticas"] * 100),
                     fontsize=8, xytext=(5, 4), textcoords="offset points", color=GRIS)
    ax2.set_xlabel("% enviado a revision manual")
    ax2.set_ylabel("% de error en las decisiones automaticas")
    ax2.set_title("El dial operativo: cuanta revision manual\nse compra por cuanta seguridad")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "politicas_decision.png")


def plot_shift():
    res = json.loads((REPORTS_DIR / "shift_results.json").read_text())
    escenarios = [
        ("Poblacion normal", res["cobertura_normal"]),
        ("Deterioro,\ncalibracion vieja", res["cobertura_deterioro_calibracion_vieja"]),
        ("Deterioro,\nrecalibrado", res["cobertura_deterioro_recalibrado"]),
    ]
    x = np.arange(len(escenarios))
    ancho = 0.27

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x - ancho, [c["cobertura_global"] for _, c in escenarios], ancho,
            color=AZUL, label="Global")
    ax1.bar(x, [c["cobertura_clase_0"] for _, c in escenarios], ancho,
            color=VERDE, label="Clase 0 (paga)")
    ax1.bar(x + ancho, [c["cobertura_clase_1"] for _, c in escenarios], ancho,
            color=ROJO, label="Clase 1 (default)")
    ax1.axhline(res["cobertura_normal"]["cobertura_objetivo"], color=GRIS, ls="--", lw=1.4)
    ax1.text(2.35, res["cobertura_normal"]["cobertura_objetivo"] + 0.005, "objetivo",
             fontsize=8, color=GRIS, ha="right")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n for n, _ in escenarios])
    ax1.set_ylim(0.5, 1.02)
    ax1.set_ylabel("Cobertura empirica")
    ax1.set_title("La garantia depende de la intercambiabilidad:\nsi la poblacion cambia, se rompe")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25, axis="y")

    pol = pd.DataFrame(res["politicas"])
    ax2.bar(np.arange(len(pol)) - 0.2, pol["pct_aprobado"] * 100, 0.38,
            color=AZUL, label="% aprobado automaticamente")
    ax2.bar(np.arange(len(pol)) + 0.2, pol["tasa_error_decisiones_automaticas"] * 100, 0.38,
            color=NARANJA, label="% de error automatico")
    ax2.set_xticks(np.arange(len(pol)))
    ax2.set_xticklabels([n for n, _ in escenarios])
    ax2.set_ylabel("%")
    ax2.set_title("Consecuencia operativa del shift")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, "shift_cobertura.png")


def plot_mapa_decision():
    df = pd.read_csv(REPORTS_DIR / "conformal_test_sets.csv")
    colores = {"solo_bueno": VERDE, "solo_malo": ROJO, "ambos": NARANJA, "vacio": GRIS}
    etiquetas = {
        "solo_bueno": "{bueno} -> aprobar",
        "solo_malo": "{malo} -> rechazar",
        "ambos": "{bueno, malo} -> revisar",
        "vacio": "{} -> revisar (caso raro)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for tipo, color in colores.items():
        sub = df[df["tipo_conjunto"] == tipo]
        if sub.empty:
            continue
        ax1.scatter(sub["p_value_bueno"], sub["p_value_malo"], s=8, alpha=0.45,
                    color=color, label=f"{etiquetas[tipo]} ({len(sub)})")
    ax1.axvline(0.10, color=GRIS, ls="--", lw=1)
    ax1.axhline(0.10, color=GRIS, ls="--", lw=1)
    ax1.set_xlabel("p-value conforme de 'paga'")
    ax1.set_ylabel("p-value conforme de 'default'")
    ax1.set_title("Mapa de decision al alpha 0.10\n(las lineas son el umbral)")
    ax1.legend(frameon=False, fontsize=8, markerscale=2)
    ax1.grid(alpha=0.2)

    for tipo, color in colores.items():
        sub = df[df["tipo_conjunto"] == tipo]
        if sub.empty:
            continue
        ax2.hist(sub["pd_pred"], bins=40, alpha=0.6, color=color, label=etiquetas[tipo])
    ax2.set_xlabel("PD predicha por el GBM monotono")
    ax2.set_ylabel("Solicitudes")
    ax2.set_title("Donde cae cada tipo de decision en la escala de PD")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    _save(fig, "mapa_decision_conforme.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generando graficos:")
    plot_auditoria()
    plot_curvas_respuesta()
    plot_cobertura()
    plot_politicas()
    plot_shift()
    plot_mapa_decision()


if __name__ == "__main__":
    main()
