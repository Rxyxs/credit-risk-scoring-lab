"""Entrena y evalúa modelos de credit scoring (Regresión Logística como baseline
interpretable -- preferido regulatoriamente en la industria -- y XGBoost como
benchmark de desempeño), con las métricas estándar de riesgo crediticio: AUC-ROC,
estadístico KS, coeficiente de Gini."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from python.feature_engineering import FEATURE_COLUMNS, add_financial_ratios

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "output" / "models"
TABLES_DIR = ROOT_DIR / "output" / "tables"


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Estadístico Kolmogorov-Smirnov -- la métrica clásica de discriminación en
    credit scoring: la máxima separación entre las distribuciones acumuladas de
    score de buenos (no-default) y malos (default) pagadores."""
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]
    cum_bad = np.cumsum(y_true_sorted) / y_true_sorted.sum()
    cum_good = np.cumsum(1 - y_true_sorted) / (1 - y_true_sorted).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2*AUC - 1, la transformación estándar de industria del AUC-ROC."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def prepare_dataset(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    featured = add_financial_ratios(clean_df)
    X = featured[FEATURE_COLUMNS].astype(float)
    y = featured["default"].astype(int)
    return X, y


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(X_train_scaled, y_train)

    # scale_pos_weight compensa el desbalance de clases (~12% default) -- sin esto
    # XGBoost queda mal calibrado en el umbral 0.5 y casi no predice la clase
    # positiva (se detecto al inspeccionar la matriz de confusion, no fue un ajuste
    # preventivo generico).
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    xgb.fit(X_train, y_train)

    return {"logit": logit, "xgb": xgb, "scaler": scaler}


def evaluate_model(name: str, y_true: pd.Series, y_score: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_score)
    ks = ks_statistic(y_true.to_numpy(), y_score)
    gini = gini_coefficient(y_true.to_numpy(), y_score)
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "modelo": name,
        "AUC_ROC": round(auc, 4),
        "KS": round(ks, 4),
        "Gini": round(gini, 4),
        "matriz_confusion": cm.tolist(),
        "reporte_clasificacion": classification_report(y_true, y_pred, output_dict=True),
    }


def run_credit_scoring_pipeline() -> dict:
    clean_df = pd.read_csv(DATA_DIR / "clean_credit_applications.csv")
    X, y = prepare_dataset(clean_df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, clean_df.index, test_size=0.25, random_state=42, stratify=y
    )

    models = train_models(X_train, y_train)

    X_test_scaled = models["scaler"].transform(X_test)
    scores = {
        "logit": models["logit"].predict_proba(X_test_scaled)[:, 1],
        "xgb": models["xgb"].predict_proba(X_test)[:, 1],
    }

    results = [evaluate_model(name, y_test, s) for name, s in scores.items()]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(models["logit"], MODELS_DIR / "logit_model.joblib")
    joblib.dump(models["xgb"], MODELS_DIR / "xgb_model.joblib")
    joblib.dump(models["scaler"], MODELS_DIR / "scaler.joblib")

    # Scores de riesgo por cliente del set de test -- este CSV es el "contrato" de
    # datos que R lee en el paso de analisis combinado (bridge/).
    risk_scores_df = pd.DataFrame(
        {
            "customer_id": clean_df.loc[idx_test, "customer_id"].values,
            "default_real": y_test.values,
            "pd_logit": scores["logit"],
            "pd_xgb": scores["xgb"],
            "dti": X_test["dti"].values,
            "score_buro_externo": X_test["score_buro_externo"].values,
            "monto_solicitado_clp": clean_df.loc[idx_test, "monto_solicitado_clp"].values,
        }
    )
    risk_scores_df.to_csv(TABLES_DIR / "credit_risk_scores.csv", index=False)

    summary_df = pd.DataFrame(
        [{"modelo": r["modelo"], "AUC_ROC": r["AUC_ROC"], "KS": r["KS"], "Gini": r["Gini"]} for r in results]
    )
    summary_df.to_csv(TABLES_DIR / "credit_scoring_metrics.csv", index=False)

    return {
        "models": models,
        "X_test": X_test,
        "y_test": y_test,
        "scores": scores,
        "results": results,
        "risk_scores_df": risk_scores_df,
    }


if __name__ == "__main__":
    output = run_credit_scoring_pipeline()
    for r in output["results"]:
        print(f"\n=== {r['modelo']} ===")
        print(f"AUC-ROC: {r['AUC_ROC']}  KS: {r['KS']}  Gini: {r['Gini']}")
        print(f"Matriz de confusion: {r['matriz_confusion']}")
