"""Tests del muestreador Polya-Gamma.

Es la pieza mas facil de tener "casi bien": una constante mal puesta en la
suma infinita produce muestras que igual se ven razonables, y el Gibbs
converge a la posterior equivocada sin avisar. Por eso se compara contra
los momentos analiticos de la distribucion.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.polya_gamma import (
    polya_gamma_mean, polya_gamma_var, sample_polya_gamma, truncation_bias,
)


def test_media_analitica_en_cero_y_continuidad():
    assert polya_gamma_mean(np.array([0.0]))[0] == pytest.approx(0.25)
    cerca = polya_gamma_mean(np.array([1e-6]))[0]
    assert cerca == pytest.approx(0.25, abs=1e-9)
    assert polya_gamma_var(np.array([0.0]))[0] == pytest.approx(1.0 / 24.0)


def test_media_muestral_coincide_con_la_analitica():
    rng = np.random.default_rng(0)
    for c in (0.0, 0.5, 2.0, 5.0):
        muestras = sample_polya_gamma(np.full(40_000, c), rng, n_terms=200)
        esperada = float(polya_gamma_mean(np.array([c]))[0])
        error_mc = 3.0 * muestras.std(ddof=1) / np.sqrt(muestras.size)
        assert abs(muestras.mean() - esperada) < error_mc + 2e-4


def test_varianza_muestral_coincide_con_la_analitica():
    rng = np.random.default_rng(1)
    for c in (0.5, 3.0):
        muestras = sample_polya_gamma(np.full(60_000, c), rng, n_terms=200)
        esperada = float(polya_gamma_var(np.array([c]))[0])
        assert muestras.var(ddof=1) == pytest.approx(esperada, rel=0.05)


def test_el_truncamiento_solo_puede_subestimar():
    """Los terminos descartados son positivos: truncar baja la media."""
    rng = np.random.default_rng(2)
    c = np.zeros(60_000)
    corto = sample_polya_gamma(c, rng, n_terms=5).mean()
    largo = sample_polya_gamma(c, rng, n_terms=400).mean()
    assert corto < largo
    assert abs(corto - 0.25) <= truncation_bias(5) * 1.5
    assert abs(largo - 0.25) < 0.002


def test_cota_del_sesgo_es_decreciente_y_chica_en_el_default():
    assert truncation_bias(10) > truncation_bias(60) > truncation_bias(200)
    assert truncation_bias(60) < 1e-3


def test_muestras_positivas_y_forma_correcta():
    rng = np.random.default_rng(3)
    c = np.linspace(-4, 4, 500)
    m = sample_polya_gamma(c, rng)
    assert m.shape == c.shape
    assert np.all(m > 0)
    # la distribucion depende de |c|, no de su signo
    izq = sample_polya_gamma(np.full(20_000, -2.0), np.random.default_rng(4)).mean()
    der = sample_polya_gamma(np.full(20_000, 2.0), np.random.default_rng(4)).mean()
    assert izq == pytest.approx(der, rel=1e-12)


def test_n_terms_invalido_falla():
    with pytest.raises(ValueError):
        sample_polya_gamma(np.zeros(5), np.random.default_rng(0), n_terms=0)
