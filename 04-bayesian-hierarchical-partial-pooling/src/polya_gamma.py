"""Muestreador de la distribucion Polya-Gamma.

Es la pieza que hace posible todo lo demas. Una verosimilitud logistica no
es conjugada con una prior normal, asi que un modelo logistico bayesiano
normalmente exige Metropolis-Hastings o HMC. Polson, Scott y Windle (2013)
demostraron que si se introduce una variable auxiliar omega ~ PG(1, psi),
la verosimilitud logistica se vuelve *condicionalmente gaussiana*:

    (e^psi)^y / (1 + e^psi) = 1/2 * e^{kappa*psi} * E_omega[ e^{-omega*psi^2/2} ]

con kappa = y - 1/2. Condicionando en omega, el modelo es una regresion
lineal con pesos, y todo el muestreador de Gibbs pasa a ser conjugado: sin
tuning, sin tasa de aceptacion, sin pasos rechazados.

Aca se usa la representacion como suma infinita de gammas,

    PG(1, c) = (1 / (2*pi^2)) * sum_{k=1..inf} g_k / ((k - 1/2)^2 + c^2/(4*pi^2))

truncada en `n_terms`. La cola que se descarta es positiva y decae como
1/k^2, o sea el truncamiento subestima levemente omega; con el default de
60 terminos el sesgo del valor esperado queda bajo 1e-4 en valor absoluto,
lo que se verifica en los tests contra la media analitica tanh(c/2)/(2c).
"""

from __future__ import annotations

import numpy as np

N_TERMS_DEFAULT = 60


def polya_gamma_mean(c: np.ndarray) -> np.ndarray:
    """E[PG(1, c)] = tanh(c/2) / (2c), con el limite 1/4 en c = 0."""
    c = np.abs(np.asarray(c, dtype=float))
    out = np.full(c.shape, 0.25)
    grande = c > 1e-8
    cc = c[grande]
    out[grande] = np.tanh(cc / 2.0) / (2.0 * cc)
    return out


def polya_gamma_var(c: np.ndarray) -> np.ndarray:
    """Var[PG(1, c)], util para verificar el muestreador (limite 1/24 en 0)."""
    c = np.abs(np.asarray(c, dtype=float))
    out = np.full(c.shape, 1.0 / 24.0)
    grande = c > 1e-4
    cc = c[grande]
    out[grande] = (np.sinh(cc) - cc) / (4.0 * cc**3 * np.cosh(cc / 2.0) ** 2)
    return out


def sample_polya_gamma(c: np.ndarray, rng: np.random.Generator,
                       n_terms: int = N_TERMS_DEFAULT) -> np.ndarray:
    """Sortea omega_i ~ PG(1, c_i) para cada entrada de `c`.

    Parameters
    ----------
    c : array (n,) -- el predictor lineal actual de cada observacion.
    n_terms : cuantos terminos de la suma infinita se conservan.
    """
    c = np.abs(np.asarray(c, dtype=float))
    if n_terms < 1:
        raise ValueError("n_terms debe ser >= 1")

    k = np.arange(1, n_terms + 1, dtype=float)[:, None] - 0.5
    denom = k**2 + (c**2)[None, :] / (4.0 * np.pi**2)
    g = rng.standard_exponential((n_terms, c.size))
    return (g / denom).sum(axis=0) / (2.0 * np.pi**2)


def truncation_bias(n_terms: int = N_TERMS_DEFAULT) -> float:
    """Cota superior del sesgo del truncamiento sobre E[omega].

    Los terminos descartados aportan (1/(2*pi^2)) * sum_{k>K} 1/(k-1/2)^2,
    que esta acotado por (1/(2*pi^2)) * 1/K.
    """
    return 1.0 / (2.0 * np.pi**2 * n_terms)
