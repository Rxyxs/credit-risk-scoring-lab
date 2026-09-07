"""Tests del contador de privacidad.

Un contador que reporta epsilon de menos es peor que no tener ninguno: da
una garantia falsa con aire de rigor. Por eso los tests anclan la
implementacion a resultados analiticos conocidos y a propiedades que
cualquier contabilidad correcta tiene que cumplir (monotonia en sigma, en
los pasos y en la tasa de muestreo).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.accountant import (
    calibrar_sigma, epsilon_del_entrenamiento, epsilon_desde_rdp,
    interpretar_epsilon, rdp_gaussiano_submuestreado, rdp_total,
)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("alpha", [2, 3, 8, 32, 64])
def test_sin_submuestreo_coincide_con_el_gaussiano_analitico(sigma, alpha):
    """Con q = 1 el resultado exacto es alpha / (2 sigma^2)."""
    assert rdp_gaussiano_submuestreado(1.0, sigma, alpha) == pytest.approx(
        alpha / (2 * sigma**2), rel=1e-12
    )


def test_el_submuestreo_amplifica_la_privacidad():
    """Muestrear menos gente por paso solo puede ayudar a la privacidad."""
    for alpha in (2, 8, 32):
        completo = rdp_gaussiano_submuestreado(1.0, 2.0, alpha)
        parcial = rdp_gaussiano_submuestreado(0.05, 2.0, alpha)
        assert parcial < completo


def test_mas_ruido_es_mas_privacidad():
    for alpha in (2, 16):
        valores = [rdp_gaussiano_submuestreado(0.1, s, alpha) for s in (1.0, 2.0, 4.0, 8.0)]
        assert valores == sorted(valores, reverse=True)


def test_componer_pasos_es_sumar_en_rdp():
    ordenes = (2, 4, 8)
    uno = rdp_total(0.1, 2.0, 1, ordenes)
    cien = rdp_total(0.1, 2.0, 100, ordenes)
    assert np.allclose(cien, 100 * uno)
    assert np.allclose(rdp_total(0.1, 2.0, 0, ordenes), 0.0)


def test_epsilon_crece_con_los_pasos_y_baja_con_el_ruido():
    e_corto = epsilon_del_entrenamiento(0.05, 2.0, 100)["epsilon"]
    e_largo = epsilon_del_entrenamiento(0.05, 2.0, 5000)["epsilon"]
    assert e_largo > e_corto

    e_poco_ruido = epsilon_del_entrenamiento(0.05, 1.0, 1000)["epsilon"]
    e_mucho_ruido = epsilon_del_entrenamiento(0.05, 8.0, 1000)["epsilon"]
    assert e_mucho_ruido < e_poco_ruido


def test_un_delta_mas_exigente_cuesta_epsilon():
    ordenes = np.arange(2, 64)
    rdp = rdp_total(0.05, 3.0, 500, ordenes)
    eps_laxo, _ = epsilon_desde_rdp(rdp, ordenes, delta=1e-3)
    eps_estricto, _ = epsilon_desde_rdp(rdp, ordenes, delta=1e-7)
    assert eps_estricto > eps_laxo


def test_la_calibracion_alcanza_el_presupuesto_pedido():
    q, pasos = 0.1, 1000
    for objetivo in (0.5, 1.0, 4.0):
        sigma = calibrar_sigma(q, pasos, objetivo)
        eps = epsilon_del_entrenamiento(q, sigma, pasos)["epsilon"]
        assert eps == pytest.approx(objetivo, rel=0.01)
        # y con un poco menos de ruido ya no alcanzaria
        assert epsilon_del_entrenamiento(q, sigma * 0.9, pasos)["epsilon"] > objetivo


def test_la_calibracion_avisa_cuando_el_presupuesto_es_inalcanzable():
    with pytest.raises(ValueError):
        calibrar_sigma(0.5, 500_000, 0.001, sigma_max=50.0)
    with pytest.raises(ValueError):
        calibrar_sigma(0.1, 100, -1.0)


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        rdp_gaussiano_submuestreado(0.0, 1.0, 2)
    with pytest.raises(ValueError):
        rdp_gaussiano_submuestreado(0.1, -1.0, 2)
    with pytest.raises(ValueError):
        rdp_gaussiano_submuestreado(0.1, 1.0, 1)
    with pytest.raises(ValueError):
        epsilon_desde_rdp(np.array([1.0]), (2,), delta=2.0)
    with pytest.raises(ValueError):
        rdp_total(0.1, 1.0, -5)


def test_lectura_del_epsilon():
    assert interpretar_epsilon(0.5) == "garantia fuerte"
    assert "razonable" in interpretar_epsilon(2.0)
    assert "debil" in interpretar_epsilon(8.0)
    assert "nominal" in interpretar_epsilon(50.0)
