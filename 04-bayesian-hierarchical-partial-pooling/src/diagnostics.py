"""Diagnostico de convergencia MCMC, implementado desde cero.

Un muestreador que corre sin errores no es un muestreador que converge.
Estas son las dos verificaciones minimas antes de creerle un solo numero a
la posterior:

- **R-hat dividido**: compara la varianza entre cadenas contra la varianza
  dentro de cada cadena, partiendo ademas cada cadena en dos para detectar
  tendencias internas. Si las cadenas exploran la misma distribucion,
  R-hat baja hacia 1; por encima de 1.01 hay que sospechar.
- **Tamano de muestra efectivo (ESS)**: cuantos draws independientes
  equivalen a los draws correlacionados que se tienen. Se calcula con la
  secuencia positiva inicial de Geyer sobre la autocorrelacion estimada.

Ambos se calculan sobre las cadenas crudas, sin librerias de diagnostico,
para que el criterio de aceptacion sea explicito y no heredado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _split_chains(x: np.ndarray) -> np.ndarray:
    """(C, D) -> (2C, D//2): cada cadena partida por la mitad."""
    n_chains, n_draws = x.shape
    mitad = n_draws // 2
    return np.concatenate([x[:, :mitad], x[:, mitad: 2 * mitad]], axis=0)


def split_rhat(x: np.ndarray) -> float:
    """R-hat dividido de Gelman-Rubin para un parametro escalar."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x debe ser (n_chains, n_draws)")
    if x.shape[1] < 4:
        raise ValueError("se necesitan al menos 4 draws por cadena")

    s = _split_chains(x)
    m, n = s.shape
    medias = s.mean(axis=1)
    varianzas = s.var(axis=1, ddof=1)

    W = varianzas.mean()
    if W <= 0:
        return 1.0
    B = n * medias.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def _autocov_fft(chain: np.ndarray) -> np.ndarray:
    """Autocovarianza de una cadena via FFT (mas rapida que el doble loop)."""
    n = chain.size
    x = chain - chain.mean()
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(f * np.conjugate(f), n=m)[:n]
    return acov / n


def effective_sample_size(x: np.ndarray) -> float:
    """ESS con la secuencia positiva inicial de Geyer."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x debe ser (n_chains, n_draws)")
    s = _split_chains(x)
    m, n = s.shape
    if n < 4:
        raise ValueError("se necesitan al menos 8 draws por cadena")

    acov = np.array([_autocov_fft(c) for c in s])
    W = s.var(axis=1, ddof=1).mean()
    if W <= 0:
        return float(m * n)

    var_hat = (n - 1) / n * W + n * s.mean(axis=1).var(ddof=1) / n
    rho = 1.0 - (W - acov.mean(axis=0)) / var_hat
    rho[0] = 1.0

    # Suma de pares de Geyer: se corta cuando un par se vuelve negativo.
    suma_pares = 0.0
    t = 1
    while t + 1 < n:
        par = rho[t] + rho[t + 1]
        if par < 0:
            break
        suma_pares += par
        t += 2

    tau = 1.0 + 2.0 * suma_pares
    return float(m * n / max(tau, 1e-12))


def resumen_parametro(x: np.ndarray, nombre: str) -> dict:
    """Resumen posterior + diagnostico para un parametro escalar."""
    plano = np.asarray(x, dtype=float).ravel()
    ess = effective_sample_size(x)
    return {
        "parametro": nombre,
        "media": float(plano.mean()),
        "sd": float(plano.std(ddof=1)),
        "q05": float(np.quantile(plano, 0.05)),
        "q50": float(np.quantile(plano, 0.50)),
        "q95": float(np.quantile(plano, 0.95)),
        "r_hat": split_rhat(x),
        "ess": ess,
        "mcse": float(plano.std(ddof=1) / np.sqrt(max(ess, 1e-12))),
    }


def tabla_diagnostico(draws, incluir_segmentos: bool = True) -> pd.DataFrame:
    """Tabla de diagnostico para todos los parametros de un ajuste."""
    filas = [
        resumen_parametro(draws.beta[:, :, j], nombre)
        for j, nombre in enumerate(draws.feature_names)
    ]
    if draws.pooling == "partial":
        filas.append(resumen_parametro(draws.tau, "tau"))
    if incluir_segmentos and draws.pooling != "complete":
        filas += [
            resumen_parametro(draws.b_segmento[:, :, j], f"b[{nombre}]")
            for j, nombre in enumerate(draws.segment_names)
        ]
    return pd.DataFrame(filas)


def tabla_diagnostico_efecto_total(draws) -> pd.DataFrame:
    """Diagnostico sobre el efecto TOTAL por segmento: intercepto + b_j.

    Sin una prior que ancle los efectos de segmento, el intercepto global y
    los b_j solo estan identificados por su suma: el muestreador puede
    subir el intercepto y bajar todos los b_j sin cambiar la verosimilitud,
    y las cadenas se pasean por esa cresta. R-hat sobre los parametros
    crudos delata ese paseo -- correctamente, porque cada parametro por
    separado efectivamente no converge.

    La suma, en cambio, si esta identificada, y es lo que realmente entra
    en la prediccion. Diagnosticarla por separado distingue "el modelo no
    sirve" de "esta parametrizacion no es identificable".
    """
    if draws.pooling == "complete":
        raise ValueError("el pooling completo no tiene efectos de segmento")
    intercepto = draws.beta[:, :, 0]
    return pd.DataFrame([
        resumen_parametro(intercepto + draws.b_segmento[:, :, j], f"alpha[{nombre}]")
        for j, nombre in enumerate(draws.segment_names)
    ])


def convergio(tabla: pd.DataFrame, rhat_max: float = 1.01, ess_min: float = 400.0) -> dict:
    """Criterio de aceptacion explicito sobre la tabla de diagnostico."""
    malos_rhat = tabla.loc[tabla["r_hat"] > rhat_max, "parametro"].tolist()
    malos_ess = tabla.loc[tabla["ess"] < ess_min, "parametro"].tolist()
    return {
        "rhat_max": float(tabla["r_hat"].max()),
        "ess_min": float(tabla["ess"].min()),
        "parametros_con_rhat_alto": malos_rhat,
        "parametros_con_ess_bajo": malos_ess,
        "convergio": not malos_rhat and not malos_ess,
    }
