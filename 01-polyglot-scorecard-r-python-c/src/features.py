"""Definicion de features y construccion de la matriz de entrada compartida
por los modelos ML (challenger, en ml_models.py) y por el servicio de
scoring en produccion (api.py, reject_inference.py). Una sola fuente de
verdad para que "las mismas columnas en el mismo orden" no dependa de que
dos archivos se mantengan sincronizados a mano.
"""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "edad", "renta_liquida", "antiguedad_laboral_meses",
    "n_productos_activos", "deuda_total", "dti", "n_morosidad_reportes",
]
CATEGORICAL_FEATURES = ["tipo_contrato", "region"]
TARGET = "default_12m"
RANDOM_STATE = 42


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        X = pd.concat([X, dummies], axis=1)
    return X
