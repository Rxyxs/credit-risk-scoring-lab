"""Orquestador del lado Python: genera datos -> limpia -> entrena credit scoring ->
grafica -> SHAP. Ejecutar desde la raíz del repo con: python -m python.orchestrator
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from python.credit_scoring import run_credit_scoring_pipeline
from python.credit_scoring_mlp import run_mlp_activation_comparison
from python.data_cleaning import clean_credit_data, generate_raw_credit_data
from python.explainability import compute_shap_values, plot_shap_summary
from python.metrics_store import persist_model_comparison
from python.visualizations import (
    plot_correlation_heatmap,
    plot_ks_curve,
    plot_mlp_activation_comparison,
    plot_roc_curves,
    plot_three_model_comparison,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "output" / "figures"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1/4 Generando y limpiando datos de credito ===")
    raw = generate_raw_credit_data()
    raw.to_csv(DATA_DIR / "raw_credit_applications.csv", index=False)
    clean, report = clean_credit_data(raw)
    clean.to_csv(DATA_DIR / "clean_credit_applications.csv", index=False)
    print(f"  {len(raw)} filas crudas -> {len(clean)} filas limpias")
    print(f"  Duplicados eliminados: {report['duplicados_eliminados']}")
    print(f"  Tasa de default: {clean['default'].mean():.2%}")

    print("\n=== 2/4 Entrenando y evaluando modelos de credit scoring ===")
    output = run_credit_scoring_pipeline()
    for r in output["results"]:
        print(f"  {r['modelo']}: AUC={r['AUC_ROC']}  KS={r['KS']}  Gini={r['Gini']}")

    print("\n=== 3/4 Generando graficos (heatmap, ROC, KS) ===")
    plot_correlation_heatmap(output["X_test"], FIGURES_DIR / "credit_correlation_heatmap.png")
    plot_roc_curves(output["y_test"], output["scores"], FIGURES_DIR / "credit_roc_curves.png")
    plot_ks_curve(output["y_test"], output["scores"]["xgb"], "XGBoost", FIGURES_DIR / "credit_ks_curve.png")

    print("\n=== 4/6 Calculando SHAP ===")
    shap_values = compute_shap_values(output["models"]["xgb"], output["X_test"])
    plot_shap_summary(shap_values, FIGURES_DIR / "credit_shap_summary.png")

    print("\n=== 5/6 Entrenando MLP (PyTorch) -- comparacion ReLU/GELU/Swish ===")
    mlp_output = run_mlp_activation_comparison()
    for r in mlp_output["results"]:
        print(f"  {r['modelo']}: AUC={r['AUC_ROC']}  KS={r['KS']}  Gini={r['Gini']}")
    print(f"  Mejor activacion: {mlp_output['best_activation']}")
    plot_mlp_activation_comparison(
        mlp_output["comparison_df"], FIGURES_DIR / "credit_mlp_activation_comparison.png"
    )

    print("\n=== 6/6 Persistiendo metricas comparativas (DuckDB) ===")
    all_results = output["results"] + mlp_output["results"]
    all_results_slim = [
        {"modelo": r["modelo"], "AUC_ROC": r["AUC_ROC"], "KS": r["KS"], "Gini": r["Gini"]}
        for r in all_results
    ]
    best = mlp_output["best_activation"]
    mlp_preds_df = pd.DataFrame(
        {
            "activacion": best,
            "default_real": mlp_output["y_test"].astype(int),
            "pd_mlp": mlp_output["scores"][best],
        }
    )
    persist_model_comparison(all_results_slim, mlp_preds_df)
    plot_three_model_comparison(all_results_slim, FIGURES_DIR / "credit_three_model_comparison.png")

    print("\nPipeline Python completo.")


if __name__ == "__main__":
    main()
