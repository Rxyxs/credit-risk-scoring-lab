import json

import pandas as pd

from src.metrics_store import _connect, persist_deep_learning, persist_ml_challengers, persist_scorecard


def _write_reports(reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "scorecard_validation_metrics.json").write_text(json.dumps({
        "auc": 0.75, "gini": 0.5, "ks_statistic": 0.4, "psi_test_vs_train": 0.01,
    }))

    (reports_dir / "ml_model_comparison.json").write_text(json.dumps({
        "best_model": "xgboost",
        "metrics": {
            "xgboost": {
                "cv_auc_mean": 0.8, "cv_auc_std": 0.01,
                "test": {"auc": 0.81, "gini": 0.62, "ks_statistic": 0.45, "f1_best_threshold": 0.5},
            }
        },
    }))
    pd.DataFrame({"applicant_id": [1, 2], "default_12m": [0, 1], "pd_ml": [0.1, 0.9]}).to_csv(
        reports_dir / "ml_test_scores.csv", index=False
    )

    (reports_dir / "dl_model_comparison.json").write_text(json.dumps({
        "best_activation": "swish",
        "metrics": {
            "swish": {"auc": 0.79, "gini": 0.58, "ks_statistic": 0.42, "f1_best_threshold": 0.5},
        },
    }))
    pd.DataFrame({"applicant_id": [1, 2], "default_12m": [0, 1], "pd_dl": [0.15, 0.85]}).to_csv(
        reports_dir / "dl_test_scores.csv", index=False
    )


def test_persist_all_writes_rows_for_each_approach(tmp_path, monkeypatch):
    import src.metrics_store as metrics_store

    db_path = tmp_path / "metrics.duckdb"
    monkeypatch.setattr(metrics_store, "DB_PATH", db_path)

    reports_dir = tmp_path / "reports"
    _write_reports(reports_dir)

    con = metrics_store._connect()
    persist_scorecard(reports_dir, con)
    persist_ml_challengers(reports_dir, con)
    persist_deep_learning(reports_dir, con)

    runs = con.execute("SELECT approach, model_name, auc FROM model_runs ORDER BY approach").fetchdf()
    preds = con.execute("SELECT approach FROM predictions").fetchdf()
    con.close()

    assert db_path.exists()
    assert set(runs["approach"]) == {"baseline_interpretable", "ensemble_tree_boosting", "deep_learning"}
    assert len(runs) == 3
    assert set(preds["approach"]) == {"ensemble_tree_boosting", "deep_learning"}
