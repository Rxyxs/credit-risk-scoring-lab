"""Del binning a una tarjeta de puntos que alguien pueda leer.

Un scorecard tradicional no entrega una probabilidad: entrega **puntos**.
Esa es su virtud operativa -- un ejecutivo puede sumar a mano por que un
cliente quedo en 612 y no en 680 -- y depende de una transformacion que se
elige y se documenta, no de una libreria:

    factor = PDO / ln(2)
    offset = puntaje_base - factor * ln(odds_base)
    puntos(variable, bin) = -(beta_var * WOE_bin + alpha / n_vars) * factor
                            + offset / n_vars

con PDO = "points to double the odds". Con PDO 20, 600 puntos base y odds
base 50:1, cada 20 puntos adicionales el cliente es la mitad de riesgoso, y
la suma de los puntos de todos sus bins es su score.

El modelo debajo es una regresion logistica sobre las variables ya
transformadas a WOE: lineal en WOE, no lineal en la variable original, que
es exactamente el punto de haber hecho el binning bien.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.binning import ResultadoBinning, aplicar_binning

PDO = 20.0
PUNTAJE_BASE = 600.0
ODDS_BASE = 50.0


@dataclass
class Scorecard:
    binnings: dict[str, ResultadoBinning]
    coeficientes: dict[str, float]
    intercepto: float
    tarjeta: pd.DataFrame
    pdo: float = PDO
    puntaje_base: float = PUNTAJE_BASE
    odds_base: float = ODDS_BASE

    @property
    def variables(self) -> list[str]:
        return list(self.binnings)

    # ------------------------------------------------------------------
    def matriz_woe(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            aplicar_binning(self.binnings[v], df[v].to_numpy()) for v in self.variables
        ])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        eta = self.matriz_woe(df) @ np.array(
            [self.coeficientes[v] for v in self.variables]
        ) + self.intercepto
        return 1.0 / (1.0 + np.exp(-eta))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Puntaje del scorecard: suma de los puntos de cada bin."""
        factor = self.pdo / np.log(2.0)
        offset = self.puntaje_base - factor * np.log(self.odds_base)
        p = len(self.variables)
        woe = self.matriz_woe(df)
        coefs = np.array([self.coeficientes[v] for v in self.variables])
        puntos = -(woe * coefs + self.intercepto / p) * factor + offset / p
        return puntos.sum(axis=1)


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
        "brier": float(np.mean((pd_pred - np.asarray(y_true)) ** 2)),
    }


def construir_tarjeta(binnings: dict[str, ResultadoBinning], coeficientes: dict[str, float],
                      intercepto: float, pdo: float = PDO,
                      puntaje_base: float = PUNTAJE_BASE,
                      odds_base: float = ODDS_BASE) -> pd.DataFrame:
    """La tarjeta imprimible: una fila por variable y bin, con sus puntos."""
    factor = pdo / np.log(2.0)
    offset = puntaje_base - factor * np.log(odds_base)
    p = len(binnings)

    filas = []
    for var, res in binnings.items():
        beta = coeficientes[var]
        for _, fila in res.tabla.iterrows():
            puntos = -(beta * fila["woe"] + intercepto / p) * factor + offset / p
            filas.append({
                "variable": var,
                "bin": int(fila["bin"]),
                "etiqueta": fila["etiqueta"],
                "n": int(fila["n"]),
                "pct_poblacion": float(fila["pct_poblacion"]),
                "tasa_mala": float(fila["tasa_mala"]),
                "woe": float(fila["woe"]),
                "coef": float(beta),
                "puntos": float(puntos),
            })
    tarjeta = pd.DataFrame(filas)
    tarjeta["puntos"] = tarjeta["puntos"].round(1)
    return tarjeta


def ajustar_scorecard(train: pd.DataFrame, y: np.ndarray,
                      binnings: dict[str, ResultadoBinning]) -> Scorecard:
    """Regresion logistica sobre las variables transformadas a WOE."""
    variables = list(binnings)
    X = np.column_stack([
        aplicar_binning(binnings[v], train[v].to_numpy()) for v in variables
    ])
    modelo = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-10)
    modelo.fit(X, y)

    coeficientes = dict(zip(variables, modelo.coef_.ravel().tolist()))
    intercepto = float(modelo.intercept_[0])
    tarjeta = construir_tarjeta(binnings, coeficientes, intercepto)
    return Scorecard(binnings=binnings, coeficientes=coeficientes,
                     intercepto=intercepto, tarjeta=tarjeta)


def bandas_de_riesgo(score: np.ndarray, y: np.ndarray,
                     cortes: list[float] | None = None,
                     score_referencia: np.ndarray | None = None) -> pd.DataFrame:
    """Agrupa el puntaje en bandas comerciales y mide la tasa mala real.

    Sin cortes explicitos, las bandas se definen por los quintiles de una
    poblacion de referencia (por defecto, la misma). Fijar cortes redondos
    "bonitos" sobre una escala que depende del intercepto y del numero de
    variables es como se termina con el 77% de la cartera en una sola banda.
    """
    etiquetas = ["E", "D", "C", "B", "A"]
    if cortes is None:
        referencia = score if score_referencia is None else score_referencia
        cortes = list(np.quantile(referencia, [0.20, 0.40, 0.60, 0.80]))
    bandas = np.digitize(score, cortes)
    df = pd.DataFrame({"banda": [etiquetas[b] for b in bandas], "y": y, "score": score})
    tabla = df.groupby("banda").agg(
        n=("y", "size"),
        score_min=("score", "min"),
        score_max=("score", "max"),
        tasa_mala=("y", "mean"),
    ).reindex(etiquetas).dropna()
    tabla["pct_poblacion"] = tabla["n"] / tabla["n"].sum()
    return tabla.reset_index()
