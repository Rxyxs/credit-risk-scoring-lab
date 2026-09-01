"""Explicabilidad del modelo XGBoost via SHAP -- qué features empujan la predicción
de default hacia arriba o hacia abajo, y en qué magnitud, por cliente."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import shap

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "output" / "figures"


def compute_shap_values(xgb_model, X_test):
    explainer = shap.TreeExplainer(xgb_model)
    return explainer(X_test)


def plot_shap_summary(shap_values, path: Path) -> None:
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, show=False)
    plt.title("Importancia SHAP -- modelo XGBoost de credit scoring")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    from python.credit_scoring import run_credit_scoring_pipeline

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = run_credit_scoring_pipeline()

    shap_values = compute_shap_values(output["models"]["xgb"], output["X_test"])
    plot_shap_summary(shap_values, FIGURES_DIR / "credit_shap_summary.png")

    print("SHAP summary guardado en", FIGURES_DIR / "credit_shap_summary.png")
