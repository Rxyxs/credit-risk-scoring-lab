"""Estimar la correlacion de activos desde la serie de tasas de default de
una cartera -- sin haber observado nunca el factor sistematico.

Dos estimadores, con supuestos distintos sobre el tamano de la cartera, y
el proyecto los enfrenta a proposito:

1. **Metodo de momentos (exacto para cualquier N, si se le da N).** Bajo
   el modelo de un factor, la tasa de default de una cohorte de N
   deudores tiene varianza

       Var(D) = Var(p(Z)) + E[p(Z)(1 - p(Z))] / N
              = (Phi_2(g,g;rho) - PD^2) + (PD - Phi_2(g,g;rho)) / N

   El primer termino es el riesgo sistematico; el segundo es el ruido
   binomial de una cohorte finita, que se va a cero solo cuando N crece.
   Una version anterior de este modulo invertia solo el primer termino y
   lo llamaba "exacto para cualquier N": eso es el limite de cartera
   infinita, y con 200 deudores por cohorte sobreestima rho en los grados
   donde el ruido binomial pesa (ver `sesgo_de_granularidad`). Ahora la
   inversion incluye el segundo termino cuando se le pasa el tamano de
   las cohortes. Dada la varianza observada, se invierte por busqueda de
   raiz.

2. **Limite ASRF (exacto solo cuando N -> infinito).** La aproximacion
   "Asymptotic Single Risk Factor" en la que se apoya todo el capital
   IRB de Basilea supone que el riesgo idiosincratico se diversifica por
   completo, y que la tasa de default observada es directamente una
   funcion determinista del factor sistematico. Bajo ese supuesto,

       Phi^-1(D_t) = (Phi^-1(PD) - sqrt(rho)*Z_t) / sqrt(1-rho)

   y como Z_t ~ N(0,1), su varianza tiene que ser 1, lo que da una formula
   cerrada sin ninguna cuadratura:

       Var(Phi^-1(D)) = rho / (1 - rho)   =>   rho = Var / (1 + Var)

   Es elegante y rapida, pero **le atribuye TODO el movimiento de D_t al
   factor sistematico**, incluido el ruido idiosincratico que en una
   cartera chica no se termina de diversificar. El proyecto muestra
   exactamente eso: con carteras grandes los dos estimadores coinciden;
   con carteras chicas, el limite ASRF sobreestima rho porque confunde
   ruido idiosincratico con riesgo sistematico -- la misma critica que
   motiva el "ajuste de granularidad" que Basilea nunca termino de
   implementar en el marco estandar.

La formula del metodo de momentos necesita la CDF normal bivariada, que se
implementa aca de la misma forma que en la tecnica 09 (cuadratura de
Gauss-Legendre sobre la correlacion) -- cada tecnica de este repositorio
es autocontenida.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.stats import norm


def _phi2_densidad(a, b, r):
    un_menos = 1.0 - r**2
    return np.exp(-(a**2 - 2 * r * a * b + b**2) / (2 * un_menos)) / (
        2 * np.pi * np.sqrt(un_menos)
    )


def bivariate_normal_cdf(a: float, b: float, rho: float, n_nodos: int = 48) -> float:
    """Phi_2(a, b; rho) por cuadratura de Gauss-Legendre sobre la correlacion."""
    if not -1.0 < rho < 1.0:
        raise ValueError("rho debe estar en (-1, 1)")
    base = norm.cdf(a) * norm.cdf(b)
    if abs(rho) < 1e-12:
        return float(base)
    nodos, pesos = np.polynomial.legendre.leggauss(n_nodos)
    r = 0.5 * rho * (nodos + 1.0)
    integral = float(np.sum(pesos * _phi2_densidad(a, b, r))) * 0.5 * rho
    return float(np.clip(base + integral, 1e-15, 1.0 - 1e-15))


def _inverso_medio(n_por_cohorte) -> float:
    """Promedio de 1/N entre cohortes (0 si no se da N: cartera infinita)."""
    if n_por_cohorte is None:
        return 0.0
    n = np.asarray(n_por_cohorte, dtype=float)
    if np.any(n < 1):
        raise ValueError("cada cohorte necesita al menos un deudor")
    return float(np.mean(1.0 / n))


def varianza_teorica_tasa_default(pd_ttc: float, rho: float, n_por_cohorte=None) -> float:
    """Var(D) bajo el modelo de un factor.

    Sin `n_por_cohorte` devuelve el limite de cartera infinita,
    Phi_2(g, g; rho) - PD^2. Con el tamano de cada cohorte (un numero o uno
    por cohorte) suma el ruido binomial (PD - Phi_2) * promedio(1/N), que
    es lo que dispersa la tasa observada aunque no haya factor sistematico.
    """
    g = norm.ppf(pd_ttc)
    phi2 = bivariate_normal_cdf(g, g, rho)
    return (phi2 - pd_ttc**2) + (pd_ttc - phi2) * _inverso_medio(n_por_cohorte)


def mom_asset_correlation(pd_ttc: float, varianza_observada: float,
                          rho_min: float = 1e-4, rho_max: float = 0.999,
                          *, n_por_cohorte=None) -> float:
    """Invierte `varianza_teorica_tasa_default` para obtener rho.

    Es estrictamente creciente en rho (mas correlacion sistematica siempre
    dispersa mas la tasa de default entre cohortes), asi que la inversion
    es un problema de una sola raiz bien definida. Sin `n_por_cohorte`
    supone cohortes infinitas, y con cohortes chicas atribuye el ruido
    binomial a rho.
    """
    if varianza_observada <= 0:
        raise ValueError("la varianza observada debe ser positiva")
    inv_n = _inverso_medio(n_por_cohorte)
    g = norm.ppf(pd_ttc)

    def f(rho):
        phi2 = bivariate_normal_cdf(g, g, rho)
        return (phi2 - pd_ttc**2) + (pd_ttc - phi2) * inv_n - varianza_observada

    v_min, v_max = f(rho_min), f(rho_max)
    if v_min > 0:
        return rho_min           # varianza observada demasiado chica: rho ~ 0
    if v_max < 0:
        return rho_max           # varianza observada demasiado grande: rho ~ 1
    return float(optimize.brentq(f, rho_min, rho_max, xtol=1e-8))


def asrf_limit_correlation(tasas_default: np.ndarray) -> float:
    """Rho bajo el supuesto de cartera infinitamente granular (ASRF).

    Trata la tasa de default observada como si fuera exactamente la PD
    condicional del factor sistematico, sin ruido idiosincratico. Valido
    solo en el limite N -> infinito; en carteras chicas sobreestima rho.
    """
    d = np.clip(np.asarray(tasas_default, dtype=float), 1e-6, 1 - 1e-6)
    var_probit = float(np.var(norm.ppf(d), ddof=1))
    return var_probit / (1.0 + var_probit)
