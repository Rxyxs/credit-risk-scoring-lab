"""Persistencia local de métricas y predicciones comparativas de los tres enfoques
de credit scoring (Regresión Logística, XGBoost, MLP PyTorch) en un archivo DuckDB
embebido -- sin servidor, un solo .duckdb en output/, consultable con SQL para
auditoría o para alimentar el paso combinado en R."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "output" / "credit_risk_metrics.duckdb"


def persist_model_comparison(
    all_results: list[dict],
    mlp_predictions: pd.DataFrame,
    db_path: Path = DB_PATH,
) -> None:
    """Escribe dos tablas:
    - model_metrics: AUC/KS/Gini de cada modelo (LogReg, XGBoost, MLP-ReLU/GELU/Swish)
    - mlp_predictions: probabilidad de default predicha por el MLP en el set de test,
      para trazabilidad por cliente igual que credit_risk_scores.csv.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_results)

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE OR REPLACE TABLE model_metrics AS SELECT * FROM metrics_df")
        con.execute("CREATE OR REPLACE TABLE mlp_predictions AS SELECT * FROM mlp_predictions")
    finally:
        con.close()


def read_model_metrics(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SELECT * FROM model_metrics ORDER BY AUC_ROC DESC").fetchdf()
    finally:
        con.close()


if __name__ == "__main__":
    from python.credit_scoring import run_credit_scoring_pipeline
    from python.credit_scoring_mlp import run_mlp_activation_comparison

    cs_output = run_credit_scoring_pipeline()
    mlp_output = run_mlp_activation_comparison()

    all_results = cs_output["results"] + mlp_output["results"]
    all_results = [
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

    persist_model_comparison(all_results, mlp_preds_df)
    print(f"Métricas persistidas en {DB_PATH}")
    print(read_model_metrics())
