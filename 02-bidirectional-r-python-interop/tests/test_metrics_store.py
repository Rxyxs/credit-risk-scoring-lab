"""Tests unitarios para la persistencia de métricas comparativas en DuckDB."""

from __future__ import annotations

import pandas as pd
import pytest

from python.metrics_store import persist_model_comparison, read_model_metrics


@pytest.fixture()
def tmp_db_path(tmp_path):
    return tmp_path / "test_metrics.duckdb"


def test_persist_and_read_model_metrics_roundtrip(tmp_db_path):
    all_results = [
        {"modelo": "Regresion Logistica", "AUC_ROC": 0.75, "KS": 0.42, "Gini": 0.50},
        {"modelo": "XGBoost", "AUC_ROC": 0.73, "KS": 0.40, "Gini": 0.46},
        {"modelo": "MLP-ReLU", "AUC_ROC": 0.71, "KS": 0.38, "Gini": 0.42},
        {"modelo": "MLP-GELU", "AUC_ROC": 0.72, "KS": 0.39, "Gini": 0.44},
        {"modelo": "MLP-Swish", "AUC_ROC": 0.70, "KS": 0.37, "Gini": 0.40},
    ]
    predictions = pd.DataFrame(
        {"activacion": "GELU", "default_real": [0, 1, 0], "pd_mlp": [0.1, 0.8, 0.2]}
    )

    persist_model_comparison(all_results, predictions, db_path=tmp_db_path)
    assert tmp_db_path.exists()

    metrics_df = read_model_metrics(db_path=tmp_db_path)
    assert len(metrics_df) == 5
    assert set(metrics_df["modelo"]) == {r["modelo"] for r in all_results}
    # Ordenado descendente por AUC_ROC.
    assert metrics_df.iloc[0]["modelo"] == "Regresion Logistica"


def test_persist_overwrites_existing_tables(tmp_db_path):
    first_results = [{"modelo": "A", "AUC_ROC": 0.5, "KS": 0.1, "Gini": 0.0}]
    second_results = [{"modelo": "B", "AUC_ROC": 0.9, "KS": 0.5, "Gini": 0.8}]
    preds = pd.DataFrame({"activacion": "ReLU", "default_real": [0], "pd_mlp": [0.3]})

    persist_model_comparison(first_results, preds, db_path=tmp_db_path)
    persist_model_comparison(second_results, preds, db_path=tmp_db_path)

    metrics_df = read_model_metrics(db_path=tmp_db_path)
    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["modelo"] == "B"
