"""Tests del modelo de un factor de Vasicek/ASRF.

La verificacion central es la mas directa que se puede pedir: simular una
cartera muy granular por Monte Carlo y comparar su distribucion de
perdidas contra la formula cerrada. Si las dos no coinciden, la formula
esta mal -- no hay margen de interpretacion.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vasicek import (
    basel_asset_correlation, basel_capital_requirement, basel_maturity_adjustment,
    conditional_pd, implied_z, rwa, vasicek_loss_cdf, vasicek_loss_pdf,
    vasicek_loss_quantile,
)


# --------------------------------------------------------- PD condicional
def test_pd_condicional_promediada_sobre_el_ciclo_es_la_pd_ttc():
    """PD_ttc es la PD *incondicional*: el promedio de la PD condicional
    sobre la distribucion completa de Z, no su valor puntual en Z=0 (el
    condicionamiento es una transformacion no lineal de Z, asi que por
    Jensen conditional_pd(pd_ttc, rho, 0) != pd_ttc en general). Se
    verifica integrando por cuadratura de Gauss-Hermite."""
    nodos, pesos = np.polynomial.hermite.hermgauss(80)
    z = np.sqrt(2.0) * nodos
    peso_norm = pesos / np.sqrt(np.pi)
    for pd_ttc in (0.001, 0.01, 0.05, 0.2):
        for rho in (0.05, 0.15, 0.4):
            promedio = float(np.sum(peso_norm * conditional_pd(pd_ttc, rho, z)))
            assert promedio == pytest.approx(pd_ttc, abs=1e-6)


def test_pd_condicional_en_z_cero_no_es_la_pd_ttc_si_rho_es_alto():
    """El caso contrario al anterior, para que quede explicito: el valor
    puntual en Z=0 SI se aleja de PD_ttc cuando rho es grande."""
    pd_ttc = 0.05
    assert conditional_pd(pd_ttc, 0.5, 0.0) != pytest.approx(pd_ttc, abs=1e-6)


def test_pd_condicional_decrece_con_z():
    """Mejor estado del ciclo (Z mas alto) -> menos riesgo."""
    z = np.linspace(-3, 3, 25)
    pd_z = conditional_pd(0.03, 0.2, z)
    assert np.all(np.diff(pd_z) < 0)


def test_mas_correlacion_amplifica_la_sensibilidad_al_ciclo():
    """En una recesion (Z muy negativo), mas rho tiene que producir una PD
    condicional mas alta -- el factor sistematico pesa mas."""
    z_malo = -2.5
    pd_bajo_rho = conditional_pd(0.02, 0.05, z_malo)
    pd_alto_rho = conditional_pd(0.02, 0.40, z_malo)
    assert pd_alto_rho > pd_bajo_rho


def test_implied_z_es_la_inversa_exacta_de_conditional_pd():
    z = np.linspace(-3, 3, 15)
    pd_ttc, rho = 0.02, 0.18
    pd_z = conditional_pd(pd_ttc, rho, z)
    z_reconstruido = implied_z(pd_ttc, rho, pd_z)
    assert np.allclose(z, z_reconstruido, atol=1e-8)


def test_pd_y_rho_fuera_de_rango_fallan():
    with pytest.raises(ValueError):
        conditional_pd(0.0, 0.1, 0.0)
    with pytest.raises(ValueError):
        conditional_pd(1.0, 0.1, 0.0)
    with pytest.raises(ValueError):
        conditional_pd(0.02, 1.5, 0.0)
    with pytest.raises(ValueError):
        conditional_pd(0.02, -0.1, 0.0)


# --------------------------------------------------- distribucion cerrada
def test_cdf_y_cuantil_son_inversas():
    for alpha in (0.5, 0.9, 0.99, 0.999):
        q = vasicek_loss_quantile(alpha, 0.02, 0.15)
        assert vasicek_loss_cdf(q, 0.02, 0.15) == pytest.approx(alpha, abs=1e-6)


def test_cdf_es_monotona_creciente():
    x = np.linspace(0.001, 0.5, 60)
    f = vasicek_loss_cdf(x, 0.03, 0.2)
    assert np.all(np.diff(f) >= 0)


def test_cuantil_alto_es_mayor_que_la_pd_ttc():
    """El percentil 99.9% de la tasa de default tiene que superar a la PD
    promedio -- si no, no habria nada que capitalizar."""
    pd_ttc = 0.02
    q999 = vasicek_loss_quantile(0.999, pd_ttc, 0.15)
    assert q999 > pd_ttc


def test_pdf_integra_aproximadamente_uno():
    x = np.linspace(1e-4, 1 - 1e-4, 20000)
    densidad = vasicek_loss_pdf(x, 0.02, 0.15)
    integral = np.trapezoid(densidad, x)
    assert integral == pytest.approx(1.0, abs=0.01)


def test_formula_cerrada_contra_monte_carlo():
    """La verificacion central: una cartera muy granular simulada por
    Monte Carlo tiene que reproducir la distribucion cerrada de Vasicek."""
    rng = np.random.default_rng(7)
    pd_ttc, rho, n_cartera, n_replicas = 0.02, 0.15, 50_000, 3000

    from scipy.stats import norm

    z = rng.normal(size=n_replicas)
    umbral = norm.ppf(pd_ttc)
    tasas = np.empty(n_replicas)
    for r in range(n_replicas):
        eps = rng.normal(size=n_cartera)
        x = np.sqrt(rho) * z[r] + np.sqrt(1 - rho) * eps
        tasas[r] = np.mean(x < umbral)

    for alpha in (0.5, 0.75, 0.9, 0.95):
        cuantil_teorico = vasicek_loss_quantile(alpha, pd_ttc, rho)
        cuantil_empirico = np.quantile(tasas, alpha)
        assert cuantil_empirico == pytest.approx(cuantil_teorico, abs=0.01)

    assert np.mean(tasas) == pytest.approx(pd_ttc, abs=0.01)


# ------------------------------------------------------- formula de Basilea
def test_correlacion_de_basilea_esta_acotada_entre_012_y_024():
    assert basel_asset_correlation(0.0001) == pytest.approx(0.24, abs=0.005)
    assert basel_asset_correlation(0.999) == pytest.approx(0.12, abs=0.005)
    assert 0.12 <= basel_asset_correlation(0.05) <= 0.24


def test_correlacion_de_basilea_decrece_con_la_pd():
    pds = np.array([0.0001, 0.001, 0.01, 0.05, 0.1, 0.5])
    r = basel_asset_correlation(pds)
    assert np.all(np.diff(r) < 0)


def test_capital_crece_con_pd_y_es_lineal_en_lgd():
    pds = np.array([0.001, 0.01, 0.05, 0.1, 0.2])
    k = basel_capital_requirement(pds, 0.45)
    assert np.all(np.diff(k) > 0)

    k_lgd_bajo = basel_capital_requirement(0.02, 0.20)
    k_lgd_alto = basel_capital_requirement(0.02, 0.60)
    assert k_lgd_alto / k_lgd_bajo == pytest.approx(0.60 / 0.20, rel=1e-9)


def test_capital_es_siempre_positivo_y_menor_que_lgd():
    pds = np.array([0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 0.9])
    k = basel_capital_requirement(pds, 0.45)
    assert np.all(k > 0)
    assert np.all(k < 0.45)          # nunca puede superar el LGD


def test_ajuste_de_plazo_crece_con_la_madurez():
    k_corto = basel_capital_requirement(0.02, 0.45, maturity=1.0)
    k_largo = basel_capital_requirement(0.02, 0.45, maturity=5.0)
    assert k_largo > k_corto


def test_b_de_maturity_es_positivo_para_pd_tipicas():
    pds = np.array([0.0005, 0.005, 0.05, 0.2])
    assert np.all(basel_maturity_adjustment(pds) > 0)


def test_rwa_es_capital_por_12_5_por_ead():
    pd_ttc, lgd, ead = 0.02, 0.45, 1_000_000.0
    k = basel_capital_requirement(pd_ttc, lgd)
    assert rwa(pd_ttc, lgd, ead) == pytest.approx(k * 12.5 * ead)


def test_rho_explicito_reemplaza_a_la_formula_regulatoria():
    pd_ttc, lgd = 0.02, 0.45
    k_regulatorio = basel_capital_requirement(pd_ttc, lgd)
    k_rho_alto = basel_capital_requirement(pd_ttc, lgd, rho=0.5)
    assert k_rho_alto > k_regulatorio        # mas correlacion, mas capital
