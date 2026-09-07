"""Modelo de riesgos proporcionales de Cox implementado desde cero.

Por que desde cero y no `lifelines`: el punto del proyecto es la mecanica
de la verosimilitud parcial, y en particular como se tratan los *empates*.
La cartera esta observada en meses, asi que cientos de creditos comparten
exactamente el mismo tiempo de evento -- el caso donde la aproximacion de
Breslow y la de Efron dejan de coincidir y la eleccion pasa a tener efecto
medible sobre los coeficientes. Teniendo las dos implementadas y una
verdad de terreno simulada, esa diferencia se puede cuantificar en vez de
citarla de un libro.

Contenido:
- verosimilitud parcial (Breslow y Efron) con gradiente y hessiano
  analiticos, optimizada por Newton-Raphson;
- errores estandar desde el hessiano observado;
- hazard base acumulado por el estimador de Breslow;
- residuos de Schoenfeld escalados y test del supuesto PH.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class CoxPHFitResult:
    coef: np.ndarray
    se: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    ties: str
    feature_names: list[str] = field(default_factory=list)

    @property
    def z(self) -> np.ndarray:
        return self.coef / self.se

    @property
    def p_values(self) -> np.ndarray:
        return 2.0 * stats.norm.sf(np.abs(self.z))

    @property
    def hazard_ratio(self) -> np.ndarray:
        return np.exp(self.coef)

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        q = stats.norm.ppf(1.0 - alpha / 2.0)
        return np.column_stack([self.coef - q * self.se, self.coef + q * self.se])

    def summary(self) -> "list[dict]":
        ci = self.conf_int()
        names = self.feature_names or [f"x{i}" for i in range(self.coef.size)]
        return [
            {
                "feature": names[i],
                "coef": float(self.coef[i]),
                "se": float(self.se[i]),
                "z": float(self.z[i]),
                "p_value": float(self.p_values[i]),
                "hazard_ratio": float(self.hazard_ratio[i]),
                "ci_lower": float(ci[i, 0]),
                "ci_upper": float(ci[i, 1]),
            }
            for i in range(self.coef.size)
        ]


class CoxPH:
    """Cox proporcional con empates por Breslow o Efron.

    Parameters
    ----------
    ties : {"efron", "breslow"}
        Aproximacion de la verosimilitud parcial ante tiempos de evento
        repetidos. Con datos mensuales los empates son masivos y Breslow
        sesga los coeficientes hacia cero.
    ridge : float
        Penalizacion L2 sobre los coeficientes (0 = sin penalizar).
    """

    def __init__(self, ties: str = "efron", ridge: float = 0.0,
                 max_iter: int = 100, tol: float = 1e-9):
        if ties not in {"efron", "breslow"}:
            raise ValueError("ties debe ser 'efron' o 'breslow'")
        self.ties = ties
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    # ------------------------------------------------------------------
    # Verosimilitud parcial
    # ------------------------------------------------------------------
    def _prepare(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray):
        X = np.asarray(X, dtype=float)
        duration = np.asarray(duration, dtype=float)
        event = np.asarray(event, dtype=int)
        if X.ndim != 2:
            raise ValueError("X debe ser 2-D (n_creditos, n_features)")
        if not (len(X) == len(duration) == len(event)):
            raise ValueError("X, duration y event deben tener el mismo largo")
        if event.sum() == 0:
            raise ValueError("no hay eventos observados: la verosimilitud parcial es vacia")

        order = np.argsort(duration, kind="mergesort")
        return X[order], duration[order], event[order]

    def _loglik_grad_hess(self, beta, X, duration, event):
        """Devuelve (loglik, gradiente, hessiano) de la verosimilitud parcial.

        Con los datos ordenados por tiempo ascendente, el conjunto en
        riesgo en t_k es un sufijo del arreglo, asi que S0/S1/S2 salen de
        sumas acumuladas por la cola en vez de un doble loop.
        """
        n, p = X.shape
        eta = X @ beta
        eta -= eta.max()                      # estabilidad numerica
        w = np.exp(eta)

        wx = w[:, None] * X
        wxx = (wx[:, :, None] * X[:, None, :]).reshape(n, p * p)

        # Sumas por sufijo: suf[i] = sum_{j >= i}
        suf_w = np.cumsum(w[::-1])[::-1]
        suf_wx = np.cumsum(wx[::-1], axis=0)[::-1]
        suf_wxx = np.cumsum(wxx[::-1], axis=0)[::-1]

        event_times = np.unique(duration[event == 1])
        # primer indice del conjunto en riesgo para cada tiempo de evento
        starts = np.searchsorted(duration, event_times, side="left")

        loglik = 0.0
        grad = np.zeros(p)
        hess = np.zeros((p, p))

        for t, start in zip(event_times, starts):
            in_t = (duration == t) & (event == 1)
            d = int(in_t.sum())
            Xd = X[in_t]
            wd = w[in_t]

            S0 = suf_w[start]
            S1 = suf_wx[start]
            S2 = suf_wxx[start].reshape(p, p)

            loglik += float(eta[in_t].sum())
            grad += Xd.sum(axis=0)

            if self.ties == "breslow" or d == 1:
                loglik -= d * np.log(S0)
                z = S1 / S0
                grad -= d * z
                hess -= d * (S2 / S0 - np.outer(z, z))
            else:
                S0d = float(wd.sum())
                wxd = wd[:, None] * Xd
                S1d = wxd.sum(axis=0)
                S2d = (wxd[:, :, None] * Xd[:, None, :]).sum(axis=0)
                for l in range(d):
                    f = l / d
                    den = S0 - f * S0d
                    num1 = S1 - f * S1d
                    num2 = S2 - f * S2d
                    z = num1 / den
                    loglik -= np.log(den)
                    grad -= z
                    hess -= num2 / den - np.outer(z, z)

        if self.ridge > 0:
            loglik -= 0.5 * self.ridge * float(beta @ beta)
            grad -= self.ridge * beta
            hess -= self.ridge * np.eye(p)

        return loglik, grad, hess

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------
    def fit(self, X, duration, event, feature_names: list[str] | None = None) -> CoxPHFitResult:
        Xs, ds, es = self._prepare(X, duration, event)
        self.X_, self.duration_, self.event_ = Xs, ds, es
        n, p = Xs.shape

        beta = np.zeros(p)
        loglik, grad, hess = self._loglik_grad_hess(beta, Xs, ds, es)
        converged = False
        it = 0

        for it in range(1, self.max_iter + 1):
            try:
                step = np.linalg.solve(-hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(-hess, grad, rcond=None)[0]

            # Newton amortiguado: si el paso empeora la verosimilitud, se
            # reduce a la mitad (protege contra separacion casi perfecta).
            alpha = 1.0
            for _ in range(20):
                cand = beta + alpha * step
                ll_c, g_c, h_c = self._loglik_grad_hess(cand, Xs, ds, es)
                if ll_c >= loglik - 1e-12:
                    break
                alpha *= 0.5
            delta = abs(ll_c - loglik)
            beta, loglik, grad, hess = cand, ll_c, g_c, h_c
            if delta < self.tol:
                converged = True
                break

        cov = np.linalg.inv(-hess)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        self.coef_ = beta
        self.cov_ = cov
        self.loglik_ = loglik
        self.feature_names_ = list(feature_names) if feature_names else []
        self._fit_baseline()

        self.result_ = CoxPHFitResult(
            coef=beta, se=se, loglik=loglik, n_iter=it,
            converged=converged, ties=self.ties, feature_names=self.feature_names_,
        )
        return self.result_

    # ------------------------------------------------------------------
    # Hazard base y prediccion
    # ------------------------------------------------------------------
    def _fit_baseline(self):
        """Estimador de Breslow del hazard base acumulado."""
        w = np.exp(self.X_ @ self.coef_)
        suf_w = np.cumsum(w[::-1])[::-1]
        times = np.unique(self.duration_[self.event_ == 1])
        starts = np.searchsorted(self.duration_, times, side="left")
        d = np.array([int(((self.duration_ == t) & (self.event_ == 1)).sum()) for t in times])
        h0 = d / suf_w[starts]
        self.baseline_times_ = times
        self.baseline_hazard_ = h0
        self.baseline_cumhazard_ = np.cumsum(h0)

    def predict_log_partial_hazard(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_

    def baseline_cumhazard_at(self, times) -> np.ndarray:
        """H0(t) evaluado en `times` (escalon constante entre eventos)."""
        times = np.atleast_1d(np.asarray(times, dtype=float))
        idx = np.searchsorted(self.baseline_times_, times, side="right") - 1
        out = np.where(idx >= 0, self.baseline_cumhazard_[np.clip(idx, 0, None)], 0.0)
        return out

    def predict_survival(self, X, times) -> np.ndarray:
        """S(t | x) = exp(-H0(t) * exp(x'beta)) -> matriz (n_creditos, n_times)."""
        H0 = self.baseline_cumhazard_at(times)
        risk = np.exp(self.predict_log_partial_hazard(X))
        return np.exp(-np.outer(risk, H0))

    def predict_cumulative_default(self, X, times) -> np.ndarray:
        """PD acumulada F(t | x) = 1 - S(t | x)."""
        return 1.0 - self.predict_survival(X, times)

    # ------------------------------------------------------------------
    # Diagnostico del supuesto de proporcionalidad
    # ------------------------------------------------------------------
    def schoenfeld_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        """Residuos de Schoenfeld escalados (Grambsch-Therneau).

        Devuelve (tiempos de evento, residuos escalados) con una fila por
        evento observado. Bajo el supuesto PH no deben tener tendencia
        contra el tiempo.
        """
        X, dur, ev = self.X_, self.duration_, self.event_
        n, p = X.shape
        w = np.exp(X @ self.coef_)
        suf_w = np.cumsum(w[::-1])[::-1]
        suf_wx = np.cumsum((w[:, None] * X)[::-1], axis=0)[::-1]

        idx_ev = np.where(ev == 1)[0]
        starts = np.searchsorted(dur, dur[idx_ev], side="left")
        xbar = suf_wx[starts] / suf_w[starts][:, None]
        raw = X[idx_ev] - xbar

        d = idx_ev.size
        scaled = raw @ (d * self.cov_) + self.coef_
        return dur[idx_ev], scaled

    def ph_test(self, feature_names: list[str] | None = None) -> list[dict]:
        """Test del supuesto PH por covariable.

        Correlaciona el residuo de Schoenfeld escalado con el rango del
        tiempo de evento (misma idea que `cox.zph` de R). Un p-value bajo
        dice que el efecto de esa covariable cambia con el tiempo, o sea
        que el hazard no es proporcional.
        """
        times, scaled = self.schoenfeld_residuals()
        g = stats.rankdata(times)
        names = feature_names or self.feature_names_ or [f"x{i}" for i in range(scaled.shape[1])]
        out = []
        for j, name in enumerate(names):
            rho, pval = stats.pearsonr(g, scaled[:, j])
            out.append({
                "feature": name,
                "rho_schoenfeld_vs_tiempo": float(rho),
                "p_value": float(pval),
                "viola_ph": bool(pval < 0.01),
            })
        return out
