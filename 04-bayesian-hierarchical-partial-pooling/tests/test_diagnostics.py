"""Tests de R-hat y ESS.

Un diagnostico de convergencia que siempre dice "todo bien" es peor que no
tenerlo, asi que estos tests construyen los dos casos: cadenas que si
mezclan y cadenas que no.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import (
    convergio, effective_sample_size, resumen_parametro, split_rhat,
)


def _ar1(n_chains, n_draws, rho, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_chains, n_draws))
    for c in range(n_chains):
        e = rng.normal(0, np.sqrt(1 - rho**2), n_draws)
        for t in range(1, n_draws):
            x[c, t] = rho * x[c, t - 1] + e[t]
    return x


def test_rhat_cerca_de_uno_con_cadenas_iid():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 2000))
    assert split_rhat(x) == pytest.approx(1.0, abs=0.01)


def test_rhat_detecta_cadenas_con_medias_distintas():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 1000))
    x[0] += 3.0                       # una cadena se quedo en otra region
    assert split_rhat(x) > 1.5


def test_rhat_detecta_deriva_dentro_de_una_cadena():
    """Por eso se parte cada cadena en dos: una tendencia interna se ve."""
    n = 2000
    deriva = np.linspace(0, 6, n)
    x = np.tile(deriva, (4, 1)) + np.random.default_rng(3).normal(size=(4, n))
    assert split_rhat(x) > 1.2


def test_ess_cercano_a_n_con_draws_independientes():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4, 2000))
    ess = effective_sample_size(x)
    assert 0.85 * x.size < ess <= 1.2 * x.size


def test_ess_cae_con_autocorrelacion():
    x_baja = _ar1(4, 4000, rho=0.2, seed=5)
    x_alta = _ar1(4, 4000, rho=0.9, seed=5)
    ess_baja = effective_sample_size(x_baja)
    ess_alta = effective_sample_size(x_alta)
    assert ess_alta < ess_baja
    # para AR(1), ESS teorico ~ N * (1-rho)/(1+rho) = N/19 con rho=0.9
    assert ess_alta < 0.20 * x_alta.size


def test_resumen_parametro_tiene_cuantiles_ordenados():
    rng = np.random.default_rng(6)
    x = rng.normal(2.0, 0.5, size=(4, 1000))
    r = resumen_parametro(x, "theta")
    assert r["q05"] < r["q50"] < r["q95"]
    assert r["media"] == pytest.approx(2.0, abs=0.05)
    assert r["mcse"] < r["sd"]


def test_convergio_marca_los_parametros_problematicos():
    tabla = pd.DataFrame([
        {"parametro": "a", "r_hat": 1.001, "ess": 1500.0},
        {"parametro": "b", "r_hat": 1.400, "ess": 12.0},
    ])
    estado = convergio(tabla)
    assert estado["convergio"] is False
    assert estado["parametros_con_rhat_alto"] == ["b"]
    assert estado["parametros_con_ess_bajo"] == ["b"]

    ok = convergio(tabla.iloc[[0]])
    assert ok["convergio"] is True


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        split_rhat(np.zeros(10))
    with pytest.raises(ValueError):
        split_rhat(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        effective_sample_size(np.zeros((4, 4)))
