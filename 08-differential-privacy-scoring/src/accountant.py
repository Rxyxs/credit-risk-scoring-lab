"""Contabilidad de privacidad: RDP del gaussiano submuestreado y conversion
a (epsilon, delta), implementada desde cero.

Sin un contador de privacidad, "el modelo es privado" es una afirmacion sin
unidades. La contabilidad es la parte del sistema que convierte tres
decisiones de ingenieria -- cuanto ruido, cuantos pasos, que tan grande el
lote -- en un numero con significado formal: epsilon, la cota de cuanto
puede cambiar la salida del entrenamiento cuando entra o sale **una sola
persona** de la base.

Se usa Renyi Differential Privacy porque componer T pasos de entrenamiento
bajo RDP es una suma, mientras que componer en (epsilon, delta) directo es
notoriamente flojo.

Para el mecanismo gaussiano submuestreado con tasa de muestreo `q` y ruido
`sigma`, y orden entero alpha >= 2, se usa la cota estandar:

    RDP_alpha = 1/(alpha-1) * log( sum_k C(alpha,k) (1-q)^(alpha-k) q^k
                                   exp(k(k-1) / (2 sigma^2)) )

Casos limite que sirven de test: con q = 1 (sin submuestreo) la expresion
colapsa al gaussiano puro, RDP_alpha = alpha / (2 sigma^2), que es el
resultado analitico conocido.

La conversion a (epsilon, delta) usa la forma clasica
`epsilon = RDP(alpha) + log(1/delta)/(alpha-1)`, minimizada sobre alpha.
Es una cota superior algo conservadora frente a las conversiones mas
recientes -- lo que significa que el epsilon reportado aca nunca subestima
la perdida de privacidad real.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

ORDENES_DEFAULT = tuple(range(2, 129))
DELTA_DEFAULT = 1e-5


def _log_binomial(n: int, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rdp_gaussiano_submuestreado(q: float, sigma: float, alpha: int) -> float:
    """RDP de un paso del gaussiano submuestreado, en el orden `alpha`."""
    if not 0 < q <= 1:
        raise ValueError("la tasa de muestreo q debe estar en (0, 1]")
    if sigma <= 0:
        raise ValueError("sigma debe ser positivo")
    if alpha < 2 or int(alpha) != alpha:
        raise ValueError("alpha debe ser un entero >= 2")

    alpha = int(alpha)
    if q == 1.0:
        return float(alpha / (2.0 * sigma**2))

    k = np.arange(alpha + 1)
    log_terminos = (
        _log_binomial(alpha, k)
        + (alpha - k) * np.log1p(-q)
        + k * np.log(q)
        + (k * (k - 1)) / (2.0 * sigma**2)
    )
    return float(logsumexp(log_terminos) / (alpha - 1))


def rdp_total(q: float, sigma: float, pasos: int,
              ordenes=ORDENES_DEFAULT) -> np.ndarray:
    """RDP acumulado de `pasos` iteraciones: bajo RDP, componer es sumar."""
    if pasos < 0:
        raise ValueError("el numero de pasos no puede ser negativo")
    return np.array([pasos * rdp_gaussiano_submuestreado(q, sigma, a) for a in ordenes])


def epsilon_desde_rdp(rdp: np.ndarray, ordenes=ORDENES_DEFAULT,
                      delta: float = DELTA_DEFAULT) -> tuple[float, int]:
    """Convierte una curva RDP a (epsilon, delta), eligiendo el mejor orden."""
    if not 0 < delta < 1:
        raise ValueError("delta debe estar en (0, 1)")
    ordenes = np.asarray(ordenes, dtype=float)
    eps = np.asarray(rdp, dtype=float) + np.log(1.0 / delta) / (ordenes - 1.0)
    i = int(np.argmin(eps))
    return float(eps[i]), int(ordenes[i])


def epsilon_del_entrenamiento(q: float, sigma: float, pasos: int,
                              delta: float = DELTA_DEFAULT,
                              ordenes=ORDENES_DEFAULT) -> dict:
    """Presupuesto de privacidad gastado por una corrida de DP-SGD."""
    curva = rdp_total(q, sigma, pasos, ordenes)
    eps, orden = epsilon_desde_rdp(curva, ordenes, delta)
    return {
        "epsilon": eps,
        "delta": float(delta),
        "orden_optimo": orden,
        "q": float(q),
        "sigma": float(sigma),
        "pasos": int(pasos),
    }


def calibrar_sigma(q: float, pasos: int, epsilon_objetivo: float,
                   delta: float = DELTA_DEFAULT,
                   sigma_min: float = 0.3, sigma_max: float = 200.0,
                   tol: float = 1e-4) -> float:
    """Menor sigma que respeta el presupuesto pedido.

    Epsilon es decreciente en sigma, asi que una busqueda binaria basta.
    Se busca el sigma **mas chico** que cumple, porque mas ruido del
    necesario es utilidad regalada sin ganancia de privacidad.
    """
    if epsilon_objetivo <= 0:
        raise ValueError("el epsilon objetivo debe ser positivo")

    def eps(s: float) -> float:
        return epsilon_del_entrenamiento(q, s, pasos, delta)["epsilon"]

    if eps(sigma_max) > epsilon_objetivo:
        raise ValueError(
            f"ni con sigma={sigma_max} se alcanza epsilon={epsilon_objetivo}: "
            "hay que bajar los pasos o el tamano de lote"
        )
    lo, hi = sigma_min, sigma_max
    if eps(lo) <= epsilon_objetivo:
        return lo
    while hi - lo > tol:
        medio = 0.5 * (lo + hi)
        if eps(medio) > epsilon_objetivo:
            lo = medio
        else:
            hi = medio
    return float(hi)


def interpretar_epsilon(epsilon: float) -> str:
    """Lectura practica del presupuesto (convencion, no teorema)."""
    if epsilon <= 1.0:
        return "garantia fuerte"
    if epsilon <= 3.0:
        return "garantia razonable en la practica"
    if epsilon <= 10.0:
        return "garantia debil, aun asi acotada"
    return "garantia nominal: el epsilon ya casi no restringe"
