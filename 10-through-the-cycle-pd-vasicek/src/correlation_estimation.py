"""Estimar la correlacion de activos desde la serie de tasas de default de
una cartera -- sin haber observado nunca el factor sistematico.

Dos estimadores, con supuestos distintos sobre el tamano de la cartera, y
el proyecto los enfrenta a proposito:

1. **Metodo de momentos (exacto para cualquier N).** Bajo el modelo de un
   factor, la varianza de la tasa de default observada entre cohortes
   tiene una expresion exacta que **no depende de cuantos deudores tenga
   cada cohorte** -- Var(D) converge a `Phi_2(g,g;rho) - PD^2` conforme N
   crece, pero la formula sale de la probabilidad conjunta de que dos
   deudores cualesquiera caigan juntos, que es correcta para toda N.
   Dada la varianza observada, se invierte esa ecuacion por busqueda de
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


def varianza_teorica_tasa_default(pd_ttc: float, rho: float) -> float:
    """Var(D) exacta bajo el modelo de un factor: Phi_2(g, g; rho) - PD^2."""
    g = norm.ppf(pd_ttc)
    return bivariate_normal_cdf(g, g, rho) - pd_ttc**2


def mom_asset_correlation(pd_ttc: float, varianza_observada: float,
                          rho_min: float = 1e-4, rho_max: float = 0.999) -> float:
    """Invierte `varianza_teorica_tasa_default` para obtener rho.

    Es estrictamente creciente en rho (mas correlacion sistematica siempre
    dispersa mas la tasa de default entre cohortes), asi que la inversion
    es un problema de una sola raiz bien definida.
    """
    if varianza_observada <= 0:
        raise ValueError("la varianza observada debe ser positiva")

    def f(rho):
        return varianza_teorica_tasa_default(pd_ttc, rho) - varianza_observada

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
