import numpy as np
import pandas as pd

from src.cleaning import run_cleaning_pipeline
from src.data_generator import generate_applicants
from src.ml_models import build_feature_matrix, compute_ks, evaluate, run_training_pipeline


def test_compute_ks_perfect_separation_is_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    assert compute_ks(y_true, y_score) == 1.0


def test_compute_ks_no_separation_is_near_zero():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 2000)
    y_score = rng.random(2000)  # score aleatorio, sin relacion con el target
    assert compute_ks(y_true, y_score) < 0.15


def test_build_feature_matrix_one_hot_encodes_categoricals():
    df = generate_applicants(n=500, seed=1)
    X = build_feature_matrix(df)
    assert "tipo_contrato_informal" in X.columns
    assert not any(X.dtypes == object)


def test_evaluate_returns_expected_keys():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    metrics = evaluate(y_true, y_proba)
    for key in ["auc", "gini", "ks_statistic", "f1_at_0.5", "f1_best_threshold", "best_threshold"]:
        assert key in metrics


def test_run_training_pipeline_end_to_end(tmp_path):
    raw = generate_applicants(n=3000, seed=5)
    cleaned = run_cleaning_pipeline(raw)

    report = run_training_pipeline(cleaned, tmp_path / "models")

    assert report["best_model"] in report["metrics"]
    assert (tmp_path / "models" / "best_ml_model.joblib").exists()
    assert (tmp_path / "reports" / "ml_model_comparison.json").exists()
    for name, m in report["metrics"].items():
        assert 0.0 <= m["test"]["auc"] <= 1.0
