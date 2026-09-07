"""Deteccion de proxies: que variables del modelo cargan pertenencia al
grupo protegido sin pagar con poder predictivo.

"Fairness through unawareness" -- no meter el atributo protegido en el
modelo -- es lo primero que hace todo el mundo y casi nunca alcanza. Si el
sector laboral esta segregado por genero, el modelo puede reconstruir el
genero sin haberlo visto nunca.

Este modulo lo mide en dos niveles:

- **Reconstruccion global**: se entrena un modelo para predecir el atributo
  protegido *usando solo las features del scorecard*. Su AUC es cuanta
  informacion de grupo hay disponible en el conjunto de variables. Un AUC
  de 0.5 significa que el modelo no puede saber; 0.75 significa que si.
- **Por variable**: para cada feature, cuanto ayuda a predecir el grupo
  (AUC de grupo) contra cuanto ayuda a predecir el default (AUC de riesgo).
  El cociente entre ambas separa a las variables que aportan riesgo de las
  que principalmente aportan pertenencia -- que son las candidatas a
  revisar en una politica de credito.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

RANDOM_STATE = 42


def _auc_binaria(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return np.nan
    auc = roc_auc_score(y, np.asarray(score, dtype=float))
    return float(max(auc, 1 - auc))     # la direccion no importa aca


def poder_de_reconstruccion(X: np.ndarray, es_protegido: np.ndarray,
                            cv: int = 4, seed: int = RANDOM_STATE) -> dict:
    """AUC de predecir el atributo protegido con las features del modelo."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(es_protegido, dtype=int)
    modelo = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.07, random_state=seed
    )
    pred = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, pred))
    return {
        "auc_reconstruccion_grupo": auc,
        "interpretacion": (
            "el grupo es practicamente inferible desde las features"
            if auc >= 0.70 else
            "hay senal de grupo, pero parcial" if auc >= 0.60 else
            "las features casi no permiten inferir el grupo"
        ),
        "n": int(y.size),
        "prevalencia_grupo": float(y.mean()),
    }


def fuerza_de_proxy(df: pd.DataFrame, features: list[str], columna_grupo: str,
                    grupo_protegido: str, columna_y: str = "default_12m") -> pd.DataFrame:
    """Por variable: senal de grupo contra senal de riesgo.

    Para variables categoricas se usa la tasa del grupo por categoria como
    score (equivalente a un modelo de una sola variable), que es la forma
    honesta de compararlas con las numericas en la misma escala.
    """
    es_protegido = (df[columna_grupo] == grupo_protegido).astype(int).to_numpy()
    y = df[columna_y].to_numpy(int)

    filas = []
    for f in features:
        col = df[f]
        if col.dtype == object or str(col.dtype).startswith("category"):
            score_grupo = col.map(df.groupby(f)[columna_grupo]
                                  .apply(lambda s: (s == grupo_protegido).mean())).to_numpy()
            score_riesgo = col.map(df.groupby(f)[columna_y].mean()).to_numpy()
        else:
            score_grupo = col.to_numpy(float)
            score_riesgo = col.to_numpy(float)

        auc_grupo = _auc_binaria(es_protegido, score_grupo)
        auc_riesgo = _auc_binaria(y, score_riesgo)
        filas.append({
            "feature": f,
            "auc_predice_grupo": auc_grupo,
            "auc_predice_default": auc_riesgo,
            "senal_grupo": auc_grupo - 0.5,
            "senal_riesgo": auc_riesgo - 0.5,
            "razon_proxy": (auc_grupo - 0.5) / max(auc_riesgo - 0.5, 1e-9),
        })

    tabla = pd.DataFrame(filas).sort_values("razon_proxy", ascending=False)
    tabla["es_proxy_sospechoso"] = (
        (tabla["senal_grupo"] > 0.05) & (tabla["razon_proxy"] > 1.0)
    )
    return tabla.reset_index(drop=True)


def resumen_proxies(tabla: pd.DataFrame) -> dict:
    sospechosas = tabla.loc[tabla["es_proxy_sospechoso"], "feature"].tolist()
    peor = tabla.iloc[0]
    return {
        "features_sospechosas": sospechosas,
        "feature_con_mayor_razon_proxy": str(peor["feature"]),
        "razon_proxy_maxima": float(peor["razon_proxy"]),
        "auc_grupo_maxima": float(tabla["auc_predice_grupo"].max()),
    }
