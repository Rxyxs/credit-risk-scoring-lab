"""Regresion logistica jerarquica estimada con Gibbs + Polya-Gamma.

Modelo:

    y_i ~ Bernoulli(sigmoid(x_i' beta + b_{j(i)}))
    b_j ~ Normal(0, tau^2)          efecto del segmento j
    beta ~ Normal(0, sigma_beta^2)  prior debil sobre los efectos globales
    tau^2 ~ InvGamma(a0, b0)        cuanta variacion entre segmentos se admite

Los tres enfoques clasicos salen del *mismo* muestreador cambiando una sola
cosa, tau^2:

- `complete` : tau^2 = 0. No hay efectos de segmento. Un solo modelo para
               toda la cartera; ignora que los segmentos difieren.
- `none`     : tau^2 fijo y enorme. Cada segmento tiene su intercepto
               libre, estimado solo con sus propios datos; un segmento con
               40 casos queda a merced del ruido.
- `partial`  : tau^2 se estima desde los datos. Cada segmento se corre
               hacia el promedio global en proporcion a cuan poca
               informacion aporta -- shrinkage, no por regla ad hoc sino
               como consecuencia del modelo.

Todo el muestreador es conjugado gracias a la aumentacion Polya-Gamma: no
hay pasos de Metropolis, no hay tasa de aceptacion que ajustar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.polya_gamma import sample_polya_gamma

TAU2_NO_POOLING = 100.0        # "sin pooling": prior practicamente plana
TAU2_COMPLETE_POOLING = 1e-10  # "pooling completo": efectos clavados en 0


@dataclass
class PosteriorDraws:
    """Cadenas crudas del muestreador: (n_chains, n_draws, ...)."""
    beta: np.ndarray
    b_segmento: np.ndarray
    tau: np.ndarray
    pooling: str
    feature_names: list[str]
    segment_names: list[str]
    n_warmup: int
    segundos: float

    @property
    def n_chains(self) -> int:
        return self.beta.shape[0]

    @property
    def n_draws(self) -> int:
        return self.beta.shape[1]

    def flat(self, nombre: str) -> np.ndarray:
        """Apila las cadenas: (n_chains * n_draws, ...)."""
        arr = getattr(self, nombre)
        return arr.reshape(-1, *arr.shape[2:])


class HierarchicalLogit:
    """Muestreador de Gibbs con aumentacion Polya-Gamma.

    Parameters
    ----------
    pooling : {"partial", "none", "complete"}
    sigma_beta : desviacion de la prior normal sobre los efectos globales.
    a0, b0 : parametros de la prior InvGamma sobre tau^2 (solo `partial`).
    n_terms : terminos conservados en la suma de la Polya-Gamma.
    """

    def __init__(self, pooling: str = "partial", sigma_beta: float = 5.0,
                 a0: float = 2.0, b0: float = 0.5, n_terms: int = 60):
        if pooling not in {"partial", "none", "complete"}:
            raise ValueError("pooling debe ser 'partial', 'none' o 'complete'")
        self.pooling = pooling
        self.sigma_beta = float(sigma_beta)
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.n_terms = int(n_terms)

    # ------------------------------------------------------------------
    def _una_cadena(self, X, y, seg_idx, n_seg, n_draws, n_warmup, seed):
        rng = np.random.default_rng(seed)
        n, p = X.shape
        kappa = y - 0.5
        prior_prec = np.eye(p) / self.sigma_beta**2

        # Inicializacion dispersa entre cadenas: si todas parten del mismo
        # punto, R-hat no puede detectar falta de mezcla.
        beta = rng.normal(0.0, 0.5, size=p)
        b = rng.normal(0.0, 0.3, size=n_seg)
        if self.pooling == "complete":
            b = np.zeros(n_seg)
            tau2 = TAU2_COMPLETE_POOLING
        elif self.pooling == "none":
            tau2 = TAU2_NO_POOLING
        else:
            tau2 = float(rng.uniform(0.1, 1.0))

        total = n_warmup + n_draws
        beta_out = np.empty((n_draws, p))
        b_out = np.empty((n_draws, n_seg))
        tau_out = np.empty(n_draws)

        for it in range(total):
            # (1) variables auxiliares Polya-Gamma
            psi = X @ beta + b[seg_idx]
            omega = sample_polya_gamma(psi, rng, self.n_terms)

            # (2) efectos globales beta | omega, b  -- gaussiano conjugado
            resid = kappa - omega * b[seg_idx]
            XtO = X.T * omega
            prec = XtO @ X + prior_prec
            L = np.linalg.cholesky(prec)
            rhs = X.T @ resid
            mu = np.linalg.solve(prec, rhs)
            beta = mu + np.linalg.solve(L.T, rng.standard_normal(p))

            # (3) efectos de segmento b_j | omega, beta, tau^2
            if self.pooling != "complete":
                xb = X @ beta
                sum_omega = np.bincount(seg_idx, weights=omega, minlength=n_seg)
                sum_res = np.bincount(seg_idx, weights=kappa - omega * xb, minlength=n_seg)
                prec_b = sum_omega + 1.0 / tau2
                mu_b = sum_res / prec_b
                b = mu_b + rng.standard_normal(n_seg) / np.sqrt(prec_b)

            # (4) hiperparametro tau^2 | b  -- InvGamma conjugada
            if self.pooling == "partial":
                a_post = self.a0 + n_seg / 2.0
                b_post = self.b0 + 0.5 * float(b @ b)
                tau2 = b_post / rng.gamma(a_post)

            if it >= n_warmup:
                k = it - n_warmup
                beta_out[k] = beta
                b_out[k] = b
                tau_out[k] = np.sqrt(tau2)

        return beta_out, b_out, tau_out

    # ------------------------------------------------------------------
    def fit(self, X, y, seg_idx, n_seg: int, feature_names: list[str],
            segment_names: list[str], n_draws: int = 1000, n_warmup: int = 500,
            n_chains: int = 4, seed: int = 42) -> PosteriorDraws:
        import time

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        seg_idx = np.asarray(seg_idx, dtype=int)
        if seg_idx.max(initial=-1) >= n_seg or seg_idx.min(initial=0) < 0:
            raise ValueError("seg_idx fuera del rango [0, n_seg)")
        if X.shape[0] != y.size or y.size != seg_idx.size:
            raise ValueError("X, y y seg_idx deben tener el mismo largo")

        t0 = time.time()
        betas, bs, taus = [], [], []
        for c in range(n_chains):
            bo, bb, tt = self._una_cadena(
                X, y, seg_idx, n_seg, n_draws, n_warmup, seed + 1000 * c
            )
            betas.append(bo)
            bs.append(bb)
            taus.append(tt)

        return PosteriorDraws(
            beta=np.stack(betas), b_segmento=np.stack(bs), tau=np.stack(taus),
            pooling=self.pooling, feature_names=list(feature_names),
            segment_names=list(segment_names), n_warmup=n_warmup,
            segundos=time.time() - t0,
        )


# ----------------------------------------------------------------------
def posterior_pd(draws: PosteriorDraws, X: np.ndarray, seg_idx: np.ndarray,
                 thin: int = 10) -> np.ndarray:
    """PD posterior por solicitante: matriz (n_solicitantes, n_draws_thin).

    Se adelgaza la cadena porque lo que se necesita despues son cuantiles,
    y guardar cada draw para miles de solicitantes no aporta precision
    proporcional a la memoria que cuesta.
    """
    beta = draws.flat("beta")[::thin]
    b = draws.flat("b_segmento")[::thin]
    eta = X @ beta.T + b[:, seg_idx].T
    return 1.0 / (1.0 + np.exp(-eta))


def resumen_pd(pd_draws: np.ndarray, q: tuple[float, float] = (0.05, 0.95)) -> dict:
    """Media posterior y un intervalo creible por solicitante."""
    lo, hi = np.quantile(pd_draws, q, axis=1)
    return {
        "pd_media": pd_draws.mean(axis=1),
        "pd_mediana": np.median(pd_draws, axis=1),
        "pd_q_lo": lo,
        "pd_q_hi": hi,
        "pd_sd": pd_draws.std(axis=1),
    }
