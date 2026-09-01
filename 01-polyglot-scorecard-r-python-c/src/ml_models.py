"""Modelos ML modernos (challengers) para probabilidad de default, sobre
las features crudas/derivadas (NO las WOE) -- el contraste deliberado con
el scorecard estadistico de R, que si usa WOE. Comparados en el MISMO
holdout de test (columna `split`, generada una sola vez en cleaning.py).

Validacion: StratifiedKFold (no TimeSeriesSplit -- esto es un problema de
clasificacion transversal sobre solicitantes, no una serie de tiempo, asi
que estratificar por la clase minoritaria es la eleccion metodologicamente
correcta aqui).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    build_feature_matrix,
)

BASE = Path(__file__).resolve().parents[1]


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Estadistico KS: maxima separacion entre las CDFs empiricas de
    buenos y malos sobre el score predicho (misma definicion que en R)."""
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    cum_bad = np.cumsum(y_sorted) / max(n_bad, 1)
    cum_good = np.cumsum(1 - y_sorted) / max(n_good, 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    f1_at_best = float(f1s[best_idx])
    f1_at_05 = float(f1_score(y_true, (y_proba >= 0.5).astype(int)))

    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks_statistic": compute_ks(y_true, y_proba),
        "f1_at_0.5": f1_at_05,
        "f1_best_threshold": f1_at_best,
        "best_threshold": best_threshold,
    }


def build_candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
            random_state=RANDOM_STATE, scale_pos_weight=None,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, verbosity=-1, class_weight="balanced",
        ),
    }


def run_training_pipeline(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train_full = build_feature_matrix(train_df)
    X_test = build_feature_matrix(test_df).reindex(columns=X_train_full.columns, fill_value=0)
    y_train = train_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_full), columns=X_train_full.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_models = {}
    for name, model in build_candidate_models().items():
        X_tr = X_train_scaled if name == "logistic_regression" else X_train_full
        X_te = X_test_scaled if name == "logistic_regression" else X_test

        cv_auc = cross_val_score(model, X_tr, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_tr, y_train)
        y_proba_test = model.predict_proba(X_te)[:, 1]

        fitted_models[name] = model
        results[name] = {
            "cv_auc_mean": float(cv_auc.mean()),
            "cv_auc_std": float(cv_auc.std()),
            "test": evaluate(y_test, y_proba_test),
        }

    best_name = max(results, key=lambda k: results[k]["test"]["auc"])
    best_model = fitted_models[best_name]

    joblib.dump(best_model, out_dir / "best_ml_model.joblib")
    joblib.dump(list(X_train_full.columns), out_dir / "ml_feature_columns.joblib")
    if best_name == "logistic_regression":
        joblib.dump(scaler, out_dir / "ml_scaler.joblib")

    best_X_test = X_test_scaled if best_name == "logistic_regression" else X_test
    scores_df = test_df[["applicant_id", TARGET]].copy()
    scores_df["pd_ml"] = best_model.predict_proba(best_X_test)[:, 1]
    scores_df.to_csv(reports_dir / "ml_test_scores.csv", index=False)

    explain_shap(best_name, best_model, X_test if best_name != "logistic_regression" else X_test_scaled, reports_dir)

    report = {"best_model": best_name, "n_train": len(train_df), "n_test": len(test_df), "metrics": results}
    with open(reports_dir / "ml_model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def explain_shap(model_name: str, model, X_test: pd.DataFrame, reports_dir: Path, sample_size: int = 1000):
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    try:
        if model_name in ("xgboost", "lightgbm", "random_forest"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
    except Exception as exc:  # pragma: no cover - explicabilidad es best-effort
        print(f"SHAP fallo para {model_name}: {exc}")
        return

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance.to_csv(reports_dir / "shap_importance.csv", index=False)

    np.save(reports_dir / "shap_values_sample.npy", shap_values)
    X_sample.to_csv(reports_dir / "shap_sample_features.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv(BASE / "data" / "processed" / "applicants_clean.csv")
    out_dir = BASE / "outputs" / "models"
    report = run_training_pipeline(df, out_dir)
    print(json.dumps(report, indent=2))
