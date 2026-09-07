"""Modelo de hazard en tiempo discreto sobre datos persona-periodo.

Cox trata el tiempo como continuo y paga el precio de los empates. Pero la
cartera *no* esta observada en tiempo continuo: un credito cae en default
en el mes 7, no en el instante 7.31. El modelo de hazard discreto asume
exactamente eso -- una regresion logistica sobre filas credito-mes donde
la etiqueta es "cayo en default este mes" -- y ahi los empates dejan de
ser una aproximacion incomoda para pasar a ser la estructura real del dato.

Ventaja concreta para este proyecto: como el efecto del tiempo entra con
un coeficiente por mes (dummies), el modelo puede reproducir la joroba de
seasoning, y una interaccion feature x tiempo permite modelar un efecto NO
proporcional en vez de solo detectarlo.

Dos especificaciones se comparan:
- `ph`   : efectos constantes en el tiempo (equivalente discreto de Cox).
- `tvc`  : agrega interaccion `informal x log(mes)` (time-varying coefficient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

MAX_MES_DEFAULT = 36


class DiscreteTimeHazard:
    """Regresion logistica sobre filas credito-mes con baseline no parametrico.

    Parameters
    ----------
    features : lista de columnas de riesgo (constantes dentro del credito).
    tvc_features : columnas que ademas entran interactuadas con log(mes).
    max_mes : ultimo mes modelado; el baseline usa una dummy por mes.
    """

    def __init__(self, features: list[str], tvc_features: list[str] | None = None,
                 max_mes: int = MAX_MES_DEFAULT):
        self.features = list(features)
        self.tvc_features = list(tvc_features or [])
        self.max_mes = int(max_mes)
        # tol apretado a proposito: con el default de sklearn (1e-4) lbfgs
        # corta antes de llegar al maximo, y el test de razon de verosimilitud
        # entre dos especificaciones anidadas llega a salir negativo -- algo
        # imposible en teoria y que solo delata al optimizador.
        self.model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=5000, tol=1e-10,
            fit_intercept=False,
        )

    # ------------------------------------------------------------------
    @property
    def design_names(self) -> list[str]:
        names = [f"mes_{m}" for m in range(1, self.max_mes + 1)]
        names += list(self.features)
        names += [f"{f}_x_logmes" for f in self.tvc_features]
        return names

    def build_design(self, df: pd.DataFrame) -> np.ndarray:
        """Dummies de mes (baseline libre) + features + interacciones."""
        mes = df["mes"].to_numpy(dtype=int)
        if mes.min() < 1 or mes.max() > self.max_mes:
            raise ValueError(f"mes fuera de rango 1..{self.max_mes}")
        n = len(df)
        D = np.zeros((n, self.max_mes))
        D[np.arange(n), mes - 1] = 1.0

        Xf = df[self.features].to_numpy(dtype=float)
        blocks = [D, Xf]
        if self.tvc_features:
            logm = np.log(mes.astype(float))[:, None]
            blocks.append(df[self.tvc_features].to_numpy(dtype=float) * logm)
        return np.hstack(blocks)

    # ------------------------------------------------------------------
    def fit(self, df_pp: pd.DataFrame) -> "DiscreteTimeHazard":
        X = self.build_design(df_pp)
        y = df_pp["default_mes"].to_numpy(dtype=int)
        self.model.fit(X, y)
        self.coef_ = self.model.coef_.ravel()
        self.n_obs_ = len(y)
        p1 = np.clip(self.model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        self.loglik_ = float(np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
        return self

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.design_names, "coef": self.coef_})

    def baseline_monthly_hazard(self) -> np.ndarray:
        """Hazard mensual del credito 'promedio' (features en 0)."""
        intercepts = self.coef_[: self.max_mes]
        return 1.0 / (1.0 + np.exp(-intercepts))

    # ------------------------------------------------------------------
    def predict_hazard_curve(self, df_loans: pd.DataFrame) -> np.ndarray:
        """Matriz (n_creditos, max_mes) de hazards mensuales condicionales."""
        n = len(df_loans)
        meses = np.arange(1, self.max_mes + 1)
        rep = np.repeat(np.arange(n), self.max_mes)
        grid = df_loans.iloc[rep][self.features].reset_index(drop=True)
        grid["mes"] = np.tile(meses, n)
        h = self.model.predict_proba(self.build_design(grid))[:, 1]
        return h.reshape(n, self.max_mes)

    def predict_cumulative_default(self, df_loans: pd.DataFrame,
                                   times: np.ndarray | None = None) -> np.ndarray:
        """PD acumulada F(t) = 1 - prod_{s<=t}(1 - h_s)."""
        h = self.predict_hazard_curve(df_loans)
        surv = np.cumprod(1.0 - h, axis=1)
        cum = 1.0 - surv
        if times is None:
            return cum
        idx = np.asarray(times, dtype=int) - 1
        return cum[:, idx]

    def predict_pd_at(self, df_loans: pd.DataFrame, mes: int = 12) -> np.ndarray:
        return self.predict_cumulative_default(df_loans, times=np.array([mes])).ravel()


def likelihood_ratio_test(model_restringido: DiscreteTimeHazard,
                          model_completo: DiscreteTimeHazard) -> dict:
    """Test LR entre dos especificaciones anidadas (PH vs time-varying)."""
    df = len(model_completo.coef_) - len(model_restringido.coef_)
    if df <= 0:
        raise ValueError("el modelo completo debe tener mas parametros")
    stat = 2.0 * (model_completo.loglik_ - model_restringido.loglik_)
    return {
        "lr_stat": float(stat),
        "df": int(df),
        "p_value": float(stats.chi2.sf(stat, df)),
        "loglik_restringido": model_restringido.loglik_,
        "loglik_completo": model_completo.loglik_,
    }
