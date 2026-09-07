"""Entrenamiento y comparacion de los modelos de scoring.

Tres modelos sobre exactamente las mismas features:

- `logistica`   : baseline interpretable, lineal y monotona por
                  construccion (el signo del coeficiente es la direccion).
- `gbm_libre`   : gradient boosting sin restricciones. El favorito por
                  desempeno, y el que hay que auditar.
- `gbm_monotono`: el mismo gradient boosting con restricciones de
                  monotonia en las 7 features donde el dominio las
                  respalda, y libre en `edad`, cuyo efecto real es en U.

Lo que se mide no es solo cual gana en AUC, sino cuanto AUC cuesta la
restriccion y que se compra con ese costo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generator import FEATURES, MONOTONIC_CST
from src.monotonicity_audit import auditar, resumen_auditoria
from src.preprocessing import etiqueta, matriz

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

RANDOM_STATE = 42
PARAMS_GBM = dict(
    max_iter=300,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=RANDOM_STATE,
)


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    orden = np.argsort(score)
    y = np.asarray(y_true)[orden]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def metricas(y_true: np.ndarray, pd_pred: np.ndarray) -> dict:
    auc = float(roc_auc_score(y_true, pd_pred))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, pd_pred),
        "brier": float(brier_score_loss(y_true, pd_pred)),
        "log_loss": float(log_loss(y_true, pd_pred)),
    }


def vector_restricciones() -> list[int]:
    """Restricciones en el orden exacto de las columnas del modelo."""
    return [MONOTONIC_CST[f] for f in FEATURES]


def construir_modelos() -> dict:
    return {
        "logistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
        ),
        "gbm_libre": HistGradientBoostingClassifier(**PARAMS_GBM),
        "gbm_monotono": HistGradientBoostingClassifier(
            monotonic_cst=vector_restricciones(), **PARAMS_GBM
        ),
    }


def entrenar(train: pd.DataFrame) -> dict:
    X, y = matriz(train), etiqueta(train)
    modelos = construir_modelos()
    for modelo in modelos.values():
        modelo.fit(X, y)
    return modelos


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PROC_DIR / "train.csv")
    test = pd.read_csv(PROC_DIR / "test.csv")

    modelos = entrenar(train)
    X_test, y_test = matriz(test), etiqueta(test)

    resultados, auditorias = {}, {}
    for nombre, modelo in modelos.items():
        pred = modelo.predict_proba(X_test)[:, 1]
        tabla = auditar(modelo, X_test)
        auditorias[nombre] = tabla
        resultados[nombre] = {
            "metricas_test": metricas(y_test, pred),
            "auditoria_monotonia": resumen_auditoria(tabla),
        }
        tabla.to_csv(REPORTS_DIR / f"auditoria_monotonia_{nombre}.csv", index=False)

    auc_libre = resultados["gbm_libre"]["metricas_test"]["auc"]
    auc_mono = resultados["gbm_monotono"]["metricas_test"]["auc"]
    resultados["costo_de_la_restriccion"] = {
        "delta_auc": auc_mono - auc_libre,
        "delta_auc_pct": 100.0 * (auc_mono / auc_libre - 1.0),
        "delta_gini_pp": 100.0 * (
            resultados["gbm_monotono"]["metricas_test"]["gini"]
            - resultados["gbm_libre"]["metricas_test"]["gini"]
        ),
    }

    # Predicciones para las etapas siguientes (conformal y decision).
    pred_test = pd.DataFrame({"applicant_id": test["applicant_id"],
                              "default_12m": y_test,
                              "monto_credito": test["monto_credito"]})
    for nombre, modelo in modelos.items():
        pred_test[f"pd_{nombre}"] = modelo.predict_proba(X_test)[:, 1]
    pred_test.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    (REPORTS_DIR / "model_results.json").write_text(json.dumps(resultados, indent=2))

    print(f"{'modelo':<14} {'AUC':>7} {'KS':>7} {'Brier':>8} {'logloss':>8} "
          f"{'% casos con violacion':>22}")
    for nombre in modelos:
        m = resultados[nombre]["metricas_test"]
        a = resultados[nombre]["auditoria_monotonia"]
        print(f"{nombre:<14} {m['auc']:>7.4f} {m['ks']:>7.4f} {m['brier']:>8.4f} "
              f"{m['log_loss']:>8.4f} {a['pct_casos_con_violacion_max']:>21.2%}")

    print("\nAuditoria de monotonia del GBM libre (por feature)")
    print(auditorias["gbm_libre"].round(4).to_string(index=False))

    c = resultados["costo_de_la_restriccion"]
    print(f"\nCosto de imponer monotonia: AUC {auc_libre:.4f} -> {auc_mono:.4f} "
          f"({c['delta_auc_pct']:+.2f}%, {c['delta_gini_pp']:+.2f} pp de Gini)")


if __name__ == "__main__":
    main()
