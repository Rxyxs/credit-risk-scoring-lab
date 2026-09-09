"""El modelo de un factor de Vasicek/ASRF, implementado desde cero.

Es el modelo detras de todo el capital regulatorio de Basilea II/III bajo
el enfoque IRB. La idea: el default de cada deudor depende de un factor
sistematico comun Z (el ciclo economico) y de un factor idiosincratico
propio,

    X_i = sqrt(rho) * Z + sqrt(1 - rho) * eps_i,      Z, eps_i ~ N(0,1) iid
    default_i  <=>  X_i < Phi^-1(PD_ttc)

`rho` es la correlacion de activos: cuanto del riesgo de cada deudor es
compartido con el resto de la cartera. `PD_ttc` (through-the-cycle) es la
probabilidad de default incondicional, promediada sobre el ciclo completo.

Condicionando en un valor de Z, el default queda:

    PD(Z) = Phi( (Phi^-1(PD_ttc) - sqrt(rho)*Z) / sqrt(1 - rho) )

que es la PD **point-in-time**: sube en las recesiones (Z bajo) y baja en
los booms (Z alto). Esa es la distincion central del proyecto.

En el limite de una cartera infinitamente granular (ASRF, "Asymptotic
Single Risk Factor"), el riesgo idiosincratico se diversifica por completo
y la unica fuente de incertidumbre que queda es Z. Bajo ese limite, la
distribucion de la tasa de default de la cartera tiene CDF cerrada:

    F(x) = Phi( (sqrt(1-rho)*Phi^-1(x) - Phi^-1(PD_ttc)) / sqrt(rho) )

y su cuantil (el percentil 99.9% que exige Basilea) tiene inversa cerrada:

    Q(alpha) = Phi( (Phi^-1(PD_ttc) + sqrt(rho)*Phi^-1(alpha)) / sqrt(1-rho) )

Esa Q(0.999) menos la PD es, literalmente, la formula de capital de
Basilea IRB (con el ajuste de plazo y el LGD multiplicando).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

CONFIANZA_BASILEA = 0.999


def _validar_pd_rho(pd, rho):
    pd = np.asarray(pd, dtype=float)
    if np.any((pd <= 0) | (pd >= 1)):
        raise ValueError("pd debe estar en (0, 1)")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho debe estar en (0, 1)")
    return pd


def conditional_pd(pd_ttc, rho: float, z) -> np.ndarray:
    """PD point-in-time condicional al valor `z` del factor sistematico."""
    pd_ttc = _validar_pd_rho(pd_ttc, rho)
    z = np.asarray(z, dtype=float)
    return norm.cdf((norm.ppf(pd_ttc) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))


def implied_z(pd_ttc, rho: float, pd_observada) -> np.ndarray:
    """Inversa de `conditional_pd`: el factor sistematico que explica una
    tasa de default observada, dado (PD_ttc, rho) conocidos o estimados.

    Es la forma directa de reconstruir "que tan malo estuvo el ciclo" a
    partir de un numero agregado (la tasa de default de una cohorte) sin
    haber observado nunca el factor macro en si.
    """
    pd_ttc = _validar_pd_rho(pd_ttc, rho)
    pd_observada = np.clip(np.asarray(pd_observada, dtype=float), 1e-9, 1 - 1e-9)
    return (norm.ppf(pd_ttc) - np.sqrt(1.0 - rho) * norm.ppf(pd_observada)) / np.sqrt(rho)


def vasicek_loss_cdf(x, pd_ttc, rho: float) -> np.ndarray:
    """CDF de la tasa de default de una cartera infinitamente granular."""
    pd_ttc = _validar_pd_rho(pd_ttc, rho)
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1 - 1e-12)
    return norm.cdf((np.sqrt(1.0 - rho) * norm.ppf(x) - norm.ppf(pd_ttc)) / np.sqrt(rho))


def vasicek_loss_pdf(x, pd_ttc, rho: float) -> np.ndarray:
    """Densidad de la misma distribucion (formula cerrada de Vasicek 2002)."""
    pd_ttc = _validar_pd_rho(pd_ttc, rho)
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1 - 1e-12)
    a = np.sqrt((1.0 - rho) / rho)
    b = norm.ppf(pd_ttc)
    g = norm.ppf(x)
    return a * np.exp(-0.5 * (a * g - b) ** 2 + 0.5 * g**2)


def vasicek_loss_quantile(alpha, pd_ttc, rho: float) -> np.ndarray:
    """Cuantil (VaR) de la tasa de default: la inversa cerrada de la CDF."""
    pd_ttc = _validar_pd_rho(pd_ttc, rho)
    alpha = np.asarray(alpha, dtype=float)
    if np.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("alpha debe estar en (0, 1)")
    return norm.cdf((norm.ppf(pd_ttc) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1.0 - rho))


def basel_asset_correlation(pd_ttc) -> np.ndarray:
    """Correlacion de activos regulatoria (exposiciones corporate/soberano/banco).

    Formula de Basilea II/III (CRE31): decrece con la PD entre 0.24 (deudores
    de alta calidad, cuyo riesgo es mas sistemico) y 0.12 (deudores de baja
    calidad, cuyo riesgo es mas idiosincratico).
    """
    pd_ttc = np.asarray(pd_ttc, dtype=float)
    peso = (1.0 - np.exp(-50.0 * pd_ttc)) / (1.0 - np.exp(-50.0))
    return 0.12 * peso + 0.24 * (1.0 - peso)


def basel_maturity_adjustment(pd_ttc) -> np.ndarray:
    """Termino b(PD) del ajuste de plazo (CRE32)."""
    pd_ttc = np.asarray(pd_ttc, dtype=float)
    return (0.11852 - 0.05478 * np.log(pd_ttc)) ** 2


def basel_capital_requirement(pd_ttc, lgd: float, rho: float | None = None,
                              maturity: float = 2.5,
                              confianza: float = CONFIANZA_BASILEA) -> np.ndarray:
    """Requerimiento de capital IRB de Basilea (K), como fraccion de la EAD.

        K = LGD * [ Phi( (Phi^-1(PD) + sqrt(rho)*Phi^-1(0.999)) / sqrt(1-rho) ) - PD ]
            * (1 + (M - 2.5) * b(PD)) / (1 - 1.5 * b(PD))

    `rho` por defecto usa la formula regulatoria `basel_asset_correlation`;
    se puede pasar un rho distinto (por ejemplo, el estimado empiricamente)
    para comparar cuanto cambia el capital si se usa la correlacion real de
    la cartera en vez de la que asume el regulador.
    """
    pd_ttc = _validar_pd_rho(pd_ttc, rho if rho is not None else 0.15)
    if rho is None:
        rho = basel_asset_correlation(pd_ttc)
    else:
        rho = np.full_like(pd_ttc, rho, dtype=float) if np.ndim(rho) == 0 else np.asarray(rho)

    cuantil = norm.cdf((norm.ppf(pd_ttc) + np.sqrt(rho) * norm.ppf(confianza))
                       / np.sqrt(1.0 - rho))
    b = basel_maturity_adjustment(pd_ttc)
    ajuste_plazo = (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)
    return lgd * (cuantil - pd_ttc) * ajuste_plazo


def rwa(pd_ttc, lgd: float, ead, rho: float | None = None, maturity: float = 2.5) -> np.ndarray:
    """Activos ponderados por riesgo: RWA = K * 12.5 * EAD."""
    k = basel_capital_requirement(pd_ttc, lgd, rho=rho, maturity=maturity)
    return k * 12.5 * np.asarray(ead, dtype=float)
