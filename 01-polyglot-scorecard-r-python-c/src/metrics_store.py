"""Persistencia de metricas/predicciones de los 3 enfoques (scorecard R,
ML challenger, deep learning) en DuckDB local, para poder consultar
comparaciones historicas entre corridas del pipeline sin re-parsear JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DB_PATH = BASE / "data" / "processed" / "metrics.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    run_ts TIMESTAMP DEFAULT current_timestamp,
    approach VARCHAR,      -- 'baseline_interpretable' | 'ensemble_tree_boosting' | 'deep_learning'
    model_name VARCHAR,
    auc DOUBLE,
    gini DOUBLE,
    ks_statistic DOUBLE,
    f1_best_threshold DOUBLE,
    extra JSON
);

CREATE TABLE IF NOT EXISTS predictions (
    run_ts TIMESTAMP DEFAULT current_timestamp,
    approach VARCHAR,
    applicant_id BIGINT,
    y_true INTEGER,
    y_proba DOUBLE
);
"""


def _connect() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute(SCHEMA)
    return con


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def persist_scorecard(reports_dir: Path, con: duckdb.DuckDBPyConnection | None = None, run_ts: datetime | None = None):
    own_con = con is None
    con = con or _connect()
    run_ts = run_ts or _now()
    metrics = json.loads((reports_dir / "scorecard_validation_metrics.json").read_text())
    con.execute(
        "INSERT INTO model_runs (run_ts, approach, model_name, auc, gini, ks_statistic, f1_best_threshold, extra) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_ts, "baseline_interpretable", "scorecard_r_woe_logit",
            metrics["auc"], metrics["gini"], metrics["ks_statistic"], None,
            json.dumps({"psi_test_vs_train": metrics.get("psi_test_vs_train")}),
        ],
    )
    if own_con:
        con.close()


def persist_ml_challengers(reports_dir: Path, con: duckdb.DuckDBPyConnection | None = None, run_ts: datetime | None = None):
    own_con = con is None
    con = con or _connect()
    run_ts = run_ts or _now()
    report = json.loads((reports_dir / "ml_model_comparison.json").read_text())
    for name, m in report["metrics"].items():
        t = m["test"]
        con.execute(
            "INSERT INTO model_runs (run_ts, approach, model_name, auc, gini, ks_statistic, f1_best_threshold, extra) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_ts, "ensemble_tree_boosting", name,
                t["auc"], t["gini"], t["ks_statistic"], t["f1_best_threshold"],
                json.dumps({"cv_auc_mean": m["cv_auc_mean"], "cv_auc_std": m["cv_auc_std"]}),
            ],
        )
    scores = pd.read_csv(reports_dir / "ml_test_scores.csv")
    con.register("scores_tmp", scores)
    con.execute(
        "INSERT INTO predictions (run_ts, approach, applicant_id, y_true, y_proba) "
        "SELECT ?, 'ensemble_tree_boosting', applicant_id, default_12m, pd_ml FROM scores_tmp",
        [run_ts],
    )
    if own_con:
        con.close()


def persist_deep_learning(reports_dir: Path, con: duckdb.DuckDBPyConnection | None = None, run_ts: datetime | None = None):
    own_con = con is None
    con = con or _connect()
    run_ts = run_ts or _now()
    report = json.loads((reports_dir / "dl_model_comparison.json").read_text())
    for activation, m in report["metrics"].items():
        con.execute(
            "INSERT INTO model_runs (run_ts, approach, model_name, auc, gini, ks_statistic, f1_best_threshold, extra) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_ts, "deep_learning", f"mlp_{activation}_focal_loss",
                m["auc"], m["gini"], m["ks_statistic"], m["f1_best_threshold"],
                json.dumps({"loss_function": "focal_loss", "activation": activation}),
            ],
        )
    scores = pd.read_csv(reports_dir / "dl_test_scores.csv")
    con.register("scores_tmp", scores)
    con.execute(
        "INSERT INTO predictions (run_ts, approach, applicant_id, y_true, y_proba) "
        "SELECT ?, 'deep_learning', applicant_id, default_12m, pd_dl FROM scores_tmp",
        [run_ts],
    )
    if own_con:
        con.close()


def persist_all(reports_dir: Path) -> pd.DataFrame:
    con = _connect()
    run_ts = _now()
    persist_scorecard(reports_dir, con, run_ts)
    persist_ml_challengers(reports_dir, con, run_ts)
    persist_deep_learning(reports_dir, con, run_ts)
    latest = con.execute(
        "SELECT approach, model_name, auc, gini, ks_statistic, f1_best_threshold "
        "FROM model_runs WHERE run_ts = (SELECT max(run_ts) FROM model_runs) "
        "ORDER BY auc DESC"
    ).fetchdf()
    con.close()
    return latest


if __name__ == "__main__":
    reports_dir = BASE / "outputs" / "reports"
    latest = persist_all(reports_dir)
    print(latest.to_string(index=False))
