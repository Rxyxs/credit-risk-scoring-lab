"""Tests de la CDF normal bivariada, sus derivadas y el probit bivariado.

Tres capas de verificacion, de la mas basica a la mas especifica del
proyecto:

1. La CDF bivariada contra `scipy.stats.multivariate_normal` (una
   implementacion independiente).
2. Las derivadas analiticas contra diferencias finitas de esa misma CDF.
3. El hallazgo central del proyecto: sin variable de exclusion, el probit
   bivariado recupera mal `rho` (y hasta con el signo equivocado); con
   ella, lo recupera bien. Si algun cambio futuro "arregla" el primer caso,
   ese test tiene que fallar -- es la prueba de que el problema es real y
   no un bug de esta implementacion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from src.selection_models import (
    BivariateProbitSelection, bivariate_normal_cdf, bivariate_normal_cdf_partials,
    fit_probit, heckman_dos_etapas, inverse_mills_ratio, predict_probit,
)


# ---------------------------------------------------------- CDF bivariada
@pytest.mark.parametrize("rho", [-0.9, -0.5, -0.2, 0.0, 0.3, 0.6, 0.85])
@pytest.mark.parametrize("a,b", [(-2.0, -1.0), (-0.5, 0.5), (0.0, 0.0),
                                 (1.2, -0.7), (2.5, 2.0)])
def test_cdf_bivariada_contra_scipy(rho, a, b):
    got = bivariate_normal_cdf([a], [b], rho)[0]
    exp = multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([a, b])
    assert got == pytest.approx(exp, abs=1e-8)


def test_cdf_bivariada_con_rho_cero_es_el_producto_de_marginales():
    a, b = np.array([0.3, -1.1, 2.0]), np.array([-0.4, 0.8, 1.5])
    got = bivariate_normal_cdf(a, b, 0.0)
    assert np.allclose(got, norm.cdf(a) * norm.cdf(b))


def test_cdf_bivariada_rechaza_rho_fuera_de_rango():
    with pytest.raises(ValueError):
        bivariate_normal_cdf([0.0], [0.0], 1.0)
    with pytest.raises(ValueError):
        bivariate_normal_cdf([0.0], [0.0], -1.5)


def test_derivadas_analiticas_contra_diferencias_finitas():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20), rng.normal(size=20)
    rho = 0.35
    da, db, dr = bivariate_normal_cdf_partials(a, b, rho)

    h = 1e-6
    da_fd = (bivariate_normal_cdf(a + h, b, rho) - bivariate_normal_cdf(a - h, b, rho)) / (2 * h)
    db_fd = (bivariate_normal_cdf(a, b + h, rho) - bivariate_normal_cdf(a, b - h, rho)) / (2 * h)
    dr_fd = (bivariate_normal_cdf(a, b, rho + h) - bivariate_normal_cdf(a, b, rho - h)) / (2 * h)

    assert np.allclose(da, da_fd, atol=1e-5)
    assert np.allclose(db, db_fd, atol=1e-5)
    assert np.allclose(dr, dr_fd, atol=1e-5)


# -------------------------------------------------------------- probit
def test_probit_recupera_coeficientes_conocidos():
    rng = np.random.default_rng(1)
    n = 20_000
    X = rng.normal(size=(n, 3))
    beta = np.array([-0.4, 0.9, -0.6, 0.35])
    y = (rng.random(n) < norm.cdf(beta[0] + X @ beta[1:])).astype(float)
    est = fit_probit(X, y)
    assert np.allclose(est, beta, atol=0.05)


def test_probit_con_pesos_iguales_es_igual_al_probit_normal():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.normal(size=(n, 2))
    y = (rng.random(n) < norm.cdf(0.2 + X @ [0.5, -0.3])).astype(float)
    sin_pesos = fit_probit(X, y)
    con_pesos = fit_probit(X, y, w=np.ones(n))
    assert np.allclose(sin_pesos, con_pesos, atol=1e-6)


def test_probit_pesos_negativos_o_de_largo_incorrecto_fallan():
    X = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        fit_probit(X, y, w=-np.ones(10))
    with pytest.raises(ValueError):
        fit_probit(X, y, w=np.ones(5))


def test_inverse_mills_ratio_positivo_y_negativo():
    z = np.array([-1.0, 0.0, 1.0, 2.0])
    imr_sel = inverse_mills_ratio(z, seleccionado=True)
    imr_no_sel = inverse_mills_ratio(z, seleccionado=False)
    assert np.all(imr_sel > 0)
    assert np.all(imr_no_sel < 0)


def test_predict_probit_devuelve_probabilidades_validas():
    coef = np.array([0.1, -0.5, 0.3])
    X = np.random.default_rng(3).normal(size=(200, 2))
    p = predict_probit(coef, X)
    assert np.all((p >= 0) & (p <= 1))


# ------------------------------------------- probit bivariado con seleccion
def _simular_seleccion(n, rho_true, seed, con_instrumento=True):
    """DGP identico al de data_generator: errores correlacionados via una
    Cholesky directa (mas simple que la parametrizacion gamma/delta, y
    exactamente equivalente para fijar `rho`)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    beta = np.array([-0.5, 0.8, -0.4])
    alpha = np.array([0.3, -0.6, 0.5])

    L = np.linalg.cholesky([[1.0, rho_true], [rho_true, 1.0]])
    e = rng.normal(size=(n, 2)) @ L.T
    z_y = beta[0] + X @ beta[1:]

    if con_instrumento:
        instrumento = rng.normal(size=n)
        z_s = alpha[0] + X @ alpha[1:] + 0.9 * instrumento
        X_sel = np.column_stack([X, instrumento])
    else:
        z_s = alpha[0] + X @ alpha[1:]
        X_sel = X

    y = ((z_y + e[:, 0]) > 0).astype(int)
    aprobado = ((z_s + e[:, 1]) > 0).astype(int)
    return X, X_sel, y, aprobado, beta


def test_probit_bivariado_recupera_rho_con_variable_de_exclusion():
    X, X_sel, y, aprobado, beta = _simular_seleccion(30_000, rho_true=-0.5, seed=10)
    m = BivariateProbitSelection(max_iter=1000).fit(X, X_sel, y, aprobado)
    assert m.rho_ == pytest.approx(-0.5, abs=0.08)
    assert np.allclose(m.coef_outcome_, beta, atol=0.08)


def test_probit_bivariado_falla_sin_variable_de_exclusion():
    """El hallazgo central: sin instrumento, rho queda mal estimado -- a
    veces con el signo equivocado. Si esto empieza a pasar, algo en el
    modelo cambio (o el problema de identificacion dejo de ser real), y
    hay que revisar el proyecto entero, no relajar este test."""
    X, X_sel, y, aprobado, beta = _simular_seleccion(30_000, rho_true=-0.5, seed=10,
                                                      con_instrumento=False)
    m = BivariateProbitSelection(max_iter=1000).fit(X, X, y, aprobado)
    # sin exclusion, el error de rho es grande: no cerca de -0.5
    assert abs(m.rho_ - (-0.5)) > 0.15


def test_con_rho_cero_ambas_versiones_dan_cerca_de_cero():
    """Cuando no hay seleccion sobre no observables, hasta la version sin
    instrumento deberia acercarse a rho=0 -- el problema de identificacion
    es sobre la MAGNITUD de rho, no sobre detectar su ausencia total."""
    X, X_sel, y, aprobado, _ = _simular_seleccion(30_000, rho_true=0.0, seed=11,
                                                   con_instrumento=True)
    m_con = BivariateProbitSelection(max_iter=1000).fit(X, X_sel, y, aprobado)
    assert abs(m_con.rho_) < 0.10


def test_probit_bivariado_valida_dimensiones_y_exige_aprobados():
    X = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        BivariateProbitSelection().fit(X, X, y, np.zeros(5))
    with pytest.raises(ValueError):
        BivariateProbitSelection().fit(X, X, y, np.zeros(10, dtype=int))


def test_heckman_dos_etapas_mejora_con_instrumento_en_seleccion_fuerte():
    """Heckman en dos etapas es una aproximacion (solo el IMR, no la MLE
    conjunta), asi que su error varia entre corridas; promediar varias
    semillas es lo que hace la comparacion robusta a esa variacion en vez
    de depender de que una corrida particular salga bien."""
    errores_con, errores_sin = [], []
    for seed in range(20, 25):
        X, X_sel, y, aprobado, beta = _simular_seleccion(20_000, rho_true=-0.5, seed=seed)
        res_con = heckman_dos_etapas(X, X_sel, y, aprobado)
        res_sin = heckman_dos_etapas(X, X, y, aprobado)
        errores_con.append(np.mean(np.abs(res_con["coef_outcome"] - beta)))
        errores_sin.append(np.mean(np.abs(res_sin["coef_outcome"] - beta)))
    assert np.mean(errores_con) < np.mean(errores_sin)
