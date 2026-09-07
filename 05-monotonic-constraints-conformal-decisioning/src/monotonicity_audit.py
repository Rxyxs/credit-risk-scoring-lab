"""Auditoria de monotonia por contrafactual.

Un supervisor no pregunta si el modelo tiene buen AUC, pregunta si puede
explicar sus decisiones. Y hay una pregunta que un modelo de riesgo no
puede permitirse contestar mal: *si al mismo solicitante le sube la carga
financiera, .el modelo le sube o le baja el riesgo?* Un gradient boosting
sin restricciones puede perfectamente bajarlo en algun tramo -- no por
error de codigo, sino porque ajusto ruido local -- y ese caso, mostrado en
una mesa de revision, hunde la aprobacion del modelo.

Este modulo lo mide en vez de suponerlo. Para cada feature con direccion
esperada, toma a cada solicitante de test, le mueve **solo** esa variable
por una grilla creciente dejando todo lo demas fijo, y verifica que la PD
predicha se mueva en la direccion correcta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generator import FEATURES, MONOTONIC_CST

N_PUNTOS_GRILLA = 12
TOL = 1e-9


def grilla_de_valores(x: np.ndarray, n_puntos: int = N_PUNTOS_GRILLA) -> np.ndarray:
    """Grilla creciente entre los percentiles 5 y 95 de la feature."""
    lo, hi = np.quantile(x, [0.05, 0.95])
    if np.isclose(lo, hi):
        return np.array([lo])
    return np.linspace(lo, hi, n_puntos)


def violaciones_por_feature(modelo, X: np.ndarray, feature: str,
                            direccion: int, n_puntos: int = N_PUNTOS_GRILLA,
                            muestra: int | None = 2000,
                            seed: int = 0) -> dict:
    """Cuenta violaciones de monotonia de `feature` sobre los casos de `X`.

    Devuelve tanto la fraccion de solicitantes con al menos una violacion
    como la magnitud maxima observada, porque no es lo mismo un desvio de
    1e-4 que uno de 3 puntos de PD.
    """
    if direccion == 0:
        raise ValueError(f"{feature} no tiene direccion esperada: no se audita")

    j = FEATURES.index(feature)
    Xa = np.asarray(X, dtype=float)
    if muestra is not None and len(Xa) > muestra:
        rng = np.random.default_rng(seed)
        Xa = Xa[rng.choice(len(Xa), muestra, replace=False)]

    valores = grilla_de_valores(Xa[:, j], n_puntos)
    curvas = np.empty((len(Xa), valores.size))
    for k, v in enumerate(valores):
        Xk = Xa.copy()
        Xk[:, j] = v
        curvas[:, k] = modelo.predict_proba(Xk)[:, 1]

    diffs = np.diff(curvas, axis=1) * direccion      # deben ser >= 0
    malos = diffs < -TOL
    return {
        "feature": feature,
        "direccion_esperada": int(direccion),
        "n_auditados": int(len(Xa)),
        "pct_casos_con_violacion": float(malos.any(axis=1).mean()),
        "pct_pasos_violados": float(malos.mean()),
        "violacion_maxima_pd": float(np.maximum(-diffs.min(), 0.0)),
        "efecto_medio_total": float(np.mean(curvas[:, -1] - curvas[:, 0]) * direccion),
    }


def auditar(modelo, X: np.ndarray, restricciones: dict[str, int] | None = None,
            **kwargs) -> pd.DataFrame:
    """Audita todas las features con direccion esperada declarada."""
    restricciones = restricciones or MONOTONIC_CST
    filas = [
        violaciones_por_feature(modelo, X, feature, direccion, **kwargs)
        for feature, direccion in restricciones.items()
        if direccion != 0
    ]
    return pd.DataFrame(filas)


def curva_respuesta(modelo, X: np.ndarray, feature: str,
                    n_puntos: int = 40, muestra: int = 1500,
                    seed: int = 0) -> pd.DataFrame:
    """Respuesta promedio del modelo a una feature (dependencia parcial)."""
    j = FEATURES.index(feature)
    Xa = np.asarray(X, dtype=float)
    if len(Xa) > muestra:
        rng = np.random.default_rng(seed)
        Xa = Xa[rng.choice(len(Xa), muestra, replace=False)]

    valores = grilla_de_valores(Xa[:, j], n_puntos)
    medias = []
    for v in valores:
        Xk = Xa.copy()
        Xk[:, j] = v
        medias.append(float(modelo.predict_proba(Xk)[:, 1].mean()))
    return pd.DataFrame({"valor": valores, "pd_media": medias})


def resumen_auditoria(tabla: pd.DataFrame) -> dict:
    return {
        "features_auditadas": int(len(tabla)),
        "features_con_alguna_violacion": int((tabla["pct_casos_con_violacion"] > 0).sum()),
        "pct_casos_con_violacion_max": float(tabla["pct_casos_con_violacion"].max()),
        "violacion_maxima_pd": float(tabla["violacion_maxima_pd"].max()),
        "auditoria_limpia": bool((tabla["pct_casos_con_violacion"] == 0).all()),
    }
