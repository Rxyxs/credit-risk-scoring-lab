"""Tests de los estimadores de correlacion de activos.

El hallazgo central del proyecto (el limite ASRF sobreestima rho en
carteras chicas, el metodo de momentos no) se fija aca directamente: si un
cambio futuro hace que el estimador ASRF deje de sobreestimar con N
chico, o que el metodo de momentos empiece a fallar, la suite tiene que
notarlo.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from src.correlation_estimation import (
    asrf_limit_correlation, bivariate_normal_cdf, mom_asset_correlation,
    varianza_teorica_tasa_default,
)


def test_cdf_bivariada_contra_scipy():
    for rho in (-0.8, -0.3, 0.0, 0.25, 0.6, 0.9):
        for a, b in [(-2.0, -1.0), (0.0, 0.0), (1.5, 0.7)]:
            got = bivariate_normal_cdf(a, b, rho)
            exp = multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([a, b])
            assert got == pytest.approx(exp, abs=1e-8)


def test_cdf_bivariada_rechaza_rho_invalido():
    with pytest.raises(ValueError):
        bivariate_normal_cdf(0.0, 0.0, 1.0)


def test_varianza_teorica_crece_con_rho():
    pd_ttc = 0.02
    valores = [varianza_teorica_tasa_default(pd_ttc, r) for r in (0.01, 0.1, 0.3, 0.6, 0.9)]
    assert valores == sorted(valores)


def test_varianza_teorica_es_cero_sin_correlacion():
    """Sin factor sistematico, la tasa de default de una cartera infinita
    no varia entre cohortes: Var(D) -> 0 cuando rho -> 0."""
    assert varianza_teorica_tasa_default(0.02, 1e-4) == pytest.approx(0.0, abs=1e-4)


def _simular_serie(pd_ttc, rho, n_cartera, t_anios, seed):
    rng = np.random.default_rng(seed)
    umbral = norm.ppf(pd_ttc)
    tasas = np.empty(t_anios)
    for t in range(t_anios):
        z = rng.normal()
        eps = rng.normal(size=n_cartera)
        x = np.sqrt(rho) * z + np.sqrt(1 - rho) * eps
        tasas[t] = np.mean(x < umbral)
    return tasas


def test_metodo_de_momentos_recupera_rho_con_cartera_grande():
    tasas = _simular_serie(pd_ttc=0.02, rho=0.15, n_cartera=20_000, t_anios=500, seed=1)
    rho_hat = mom_asset_correlation(0.02, float(tasas.var(ddof=1)))
    assert rho_hat == pytest.approx(0.15, abs=0.05)


def test_metodo_de_momentos_tambien_funciona_con_cartera_chica():
    """A diferencia del limite ASRF, el metodo de momentos es exacto para
    cualquier N -- tiene que recuperar rho igual de bien con 100 deudores
    por cohorte que con 20.000."""
    tasas = _simular_serie(pd_ttc=0.02, rho=0.15, n_cartera=100, t_anios=800, seed=2)
    rho_hat = mom_asset_correlation(0.02, float(tasas.var(ddof=1)))
    assert rho_hat == pytest.approx(0.15, abs=0.06)


def test_limite_asrf_sobreestima_con_cartera_chica():
    """El hallazgo central: con pocos deudores por cohorte, el limite ASRF
    confunde ruido idiosincratico con riesgo sistematico y sobreestima rho
    de forma sustancial -- mientras que a N grande, coincide con la verdad."""
    pd_ttc, rho_true = 0.02, 0.15
    tasas_chica = _simular_serie(pd_ttc, rho_true, n_cartera=100, t_anios=500, seed=3)
    tasas_grande = _simular_serie(pd_ttc, rho_true, n_cartera=50_000, t_anios=500, seed=3)

    rho_asrf_chica = asrf_limit_correlation(tasas_chica)
    rho_asrf_grande = asrf_limit_correlation(tasas_grande)

    assert rho_asrf_chica > rho_true + 0.15      # sobreestimacion clara
    assert rho_asrf_grande == pytest.approx(rho_true, abs=0.04)


def test_metodo_de_momentos_le_gana_al_limite_asrf_en_cartera_chica():
    pd_ttc, rho_true = 0.02, 0.15
    tasas = _simular_serie(pd_ttc, rho_true, n_cartera=150, t_anios=500, seed=4)

    rho_mom = mom_asset_correlation(pd_ttc, float(tasas.var(ddof=1)))
    rho_asrf = asrf_limit_correlation(tasas)

    assert abs(rho_mom - rho_true) < abs(rho_asrf - rho_true)


def test_varianza_observada_no_positiva_falla():
    with pytest.raises(ValueError):
        mom_asset_correlation(0.02, 0.0)
    with pytest.raises(ValueError):
        mom_asset_correlation(0.02, -0.001)


def test_mom_devuelve_los_bordes_cuando_la_varianza_es_extrema():
    """Una varianza observada mas alta de lo que cualquier rho < 0.999
    puede producir tiene que devolver el borde, no reventar."""
    rho_borde_alto = mom_asset_correlation(0.02, varianza_observada=0.9)
    assert rho_borde_alto == pytest.approx(0.999, abs=1e-6)

    rho_borde_bajo = mom_asset_correlation(0.02, varianza_observada=1e-12)
    assert rho_borde_bajo == pytest.approx(1e-4, abs=1e-6)
