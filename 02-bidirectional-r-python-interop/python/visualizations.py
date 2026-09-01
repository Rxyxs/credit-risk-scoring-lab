"""Gráficos del módulo Python: mapa de calor de correlación de features, curva ROC
y curva KS. Todo se guarda como PNG estático -- sin dashboards ni visualizadores
externos, tal como se pidió."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "output" / "figures"


def plot_correlation_heatmap(X: pd.DataFrame, path: Path) -> None:
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Mapa de calor de correlación -- features de credit scoring", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_test: pd.Series, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        ax.plot(fpr, tpr, label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.5)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC -- modelos de credit scoring")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ks_curve(y_test: pd.Series, y_score: np.ndarray, model_name: str, path: Path) -> None:
    order = np.argsort(y_score)
    y_sorted = y_test.to_numpy()[order]
    score_sorted = y_score[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    ks_idx = np.argmax(np.abs(cum_bad - cum_good))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(score_sorted, cum_good, label="% acumulado buenos pagadores", linewidth=2)
    ax.plot(score_sorted, cum_bad, label="% acumulado default", linewidth=2)
    ax.axvline(score_sorted[ks_idx], color="gray", linestyle="--", alpha=0.7)
    ax.annotate(
        f"KS = {abs(cum_bad[ks_idx] - cum_good[ks_idx]):.3f}",
        xy=(score_sorted[ks_idx], (cum_bad[ks_idx] + cum_good[ks_idx]) / 2),
        xytext=(10, 0), textcoords="offset points",
    )
    ax.set_xlabel("Score de probabilidad de default")
    ax.set_ylabel("Proporción acumulada")
    ax.set_title(f"Curva KS -- {model_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mlp_activation_comparison(comparison_df: pd.DataFrame, path: Path) -> None:
    """Barras agrupadas AUC/KS/Gini para el MLP entrenado con ReLU, GELU y Swish --
    misma arquitectura y protocolo de entrenamiento, solo cambia la activación."""
    metrics = ["AUC_ROC", "KS", "Gini"]
    x = np.arange(len(comparison_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("MLP-", "") for m in comparison_df["modelo"]])
    ax.set_ylabel("Valor de métrica")
    ax.set_title("MLP (PyTorch) -- comparación de activaciones ReLU vs GELU vs Swish")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_three_model_comparison(all_results: list[dict], path: Path) -> None:
    """Compara LogReg, XGBoost y el mejor MLP (misma métrica AUC/KS/Gini) -- la
    triada completa de enfoques de modelado sobre el mismo dataset/split."""
    df = pd.DataFrame(all_results)
    metrics = ["AUC_ROC", "KS", "Gini"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["modelo"])
    ax.set_ylabel("Valor de métrica")
    ax.set_title("Credit scoring -- LogReg vs XGBoost vs MLP (mejor activación)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from python.credit_scoring import run_credit_scoring_pipeline

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = run_credit_scoring_pipeline()

    plot_correlation_heatmap(output["X_test"], FIGURES_DIR / "credit_correlation_heatmap.png")
    plot_roc_curves(output["y_test"], output["scores"], FIGURES_DIR / "credit_roc_curves.png")
    plot_ks_curve(output["y_test"], output["scores"]["xgb"], "XGBoost", FIGURES_DIR / "credit_ks_curve.png")

    print("Graficos guardados en", FIGURES_DIR)
