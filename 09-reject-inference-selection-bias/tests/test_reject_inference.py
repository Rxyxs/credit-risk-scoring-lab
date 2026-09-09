"""Tests de los ocho metodos de reject inference y del simulador."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.data_generator import (
    ALPHA_SELECCION, BETA_TRUE, FEATURES, FEATURES_SELECCION, INSTRUMENTO,
    INTERCEPTO_SELECCION, INTERCEPTO_TRUE, generate_applicants, rho_verdadero,
)
from src.reject_inference import (
    METODOS, aprobados_solo, augmentacion_em, ipw, oraculo, parcelling,
)
from src.selection_models import predict_probit


# --------------------------------------------------------------- datos
def test_mar_tiene_rho_cero_y_mnar_no():
    assert rho_verdadero("mar") == pytest.approx(0.0)
    assert rho_verdadero("mnar") != 0.0


def test_el_simulador_es_reproducible():
    a = generate_applicants(n=2000, seed=5, regimen="mnar")
    b = generate_applicants(n=2000, seed=5, regimen="mnar")
    assert a.equals(b)


def test_instrumento_no_entra_al_desenlace_pero_si_a_la_seleccion():
    """Condicionando en las features de riesgo, la PD verdadera no depende
    del instrumento -- por construccion, ya que no esta en BETA_TRUE."""
    df = generate_applicants(n=20_000, seed=6, regimen="mnar")
    alto = df[df[INSTRUMENTO] > df[INSTRUMENTO].quantile(0.8)]
    bajo = df[df[INSTRUMENTO] < df[INSTRUMENTO].quantile(0.2)]
    assert abs(alto["pd_verdadera"].mean() - bajo["pd_verdadera"].mean()) < 0.01
    # pero la tasa de aprobacion si depende de el, y mucho
    assert alto["aprobado"].mean() > bajo["aprobado"].mean() + 0.15


def test_todas_las_solicitudes_tienen_desenlace_incluso_los_rechazados():
    df = generate_applicants(n=5000, seed=7, regimen="mnar")
    assert df["default_12m_real"].notna().all()
    rechazados = df[df["aprobado"] == 0]
    assert rechazados["default_12m_observado"].isna().all()
    assert (rechazados["default_12m_real"].isin([0, 1])).all()


def test_mnar_los_aprobados_tienen_mejor_informacion_blanda():
    df = generate_applicants(n=20_000, seed=8, regimen="mnar")
    aprobados = df[df["aprobado"] == 1]
    rechazados = df[df["aprobado"] == 0]
    assert aprobados["soft_info_no_observada"].mean() > \
           rechazados["soft_info_no_observada"].mean() + 0.3


def test_mar_aprobados_y_rechazados_no_difieren_en_soft_info():
    df = generate_applicants(n=20_000, seed=9, regimen="mar")
    aprobados = df[df["aprobado"] == 1]
    rechazados = df[df["aprobado"] == 0]
    assert abs(aprobados["soft_info_no_observada"].mean()
              - rechazados["soft_info_no_observada"].mean()) < 0.05


def test_features_seleccion_incluye_el_instrumento():
    assert FEATURES_SELECCION == FEATURES + [INSTRUMENTO]
    assert set(BETA_TRUE) == set(FEATURES)
    assert INSTRUMENTO not in BETA_TRUE


# ----------------------------------------------------------- metodos
def _cartera(n=6000, seed=1, regimen="mnar"):
    df = generate_applicants(n=n, seed=seed, regimen=regimen)
    X = df[FEATURES].to_numpy(float)
    X_sel = df[FEATURES_SELECCION].to_numpy(float)
    aprobado = df["aprobado"].to_numpy(int)
    y_obs = np.nan_to_num(df["default_12m_observado"].to_numpy(float))
    y_real = df["default_12m_real"].to_numpy(float)
    return df, X, X_sel, y_obs, aprobado, y_real


def test_aprobados_solo_entrena_solo_con_la_muestra_aprobada():
    _, X, _, y_obs, aprobado, _ = _cartera()
    res = aprobados_solo(X, y_obs, aprobado)
    assert res["detalle"]["n_entrenamiento"] == int(aprobado.sum())


def test_oraculo_usa_toda_la_poblacion():
    _, X, _, _, _, y_real = _cartera()
    res = oraculo(X, y_real)
    assert res["detalle"]["n_entrenamiento"] == len(X)


def test_ipw_da_mas_peso_a_quien_tenia_baja_probabilidad_de_aprobacion():
    _, X, X_sel, y_obs, aprobado, _ = _cartera(n=8000)
    res = ipw(X, X_sel, y_obs, aprobado)
    assert res["detalle"]["peso_medio"] >= 1.0
    assert res["detalle"]["peso_maximo"] >= res["detalle"]["peso_medio"]


def test_parcelling_imputa_mas_riesgo_que_la_tasa_de_los_aprobados():
    """Con factor > 1, la tasa imputada a los rechazados tiene que ser
    mayor que la observada entre los aprobados -- es literalmente lo que
    hace el factor de castigo."""
    _, X, _, y_obs, aprobado, _ = _cartera(n=8000, regimen="mnar")
    res = parcelling(X, y_obs, aprobado, factor=2.0)
    assert res["detalle"]["tasa_mala_imputada_media"] > \
           res["detalle"]["tasa_mala_aprobados"]


def test_parcelling_factor_uno_no_castiga():
    _, X, _, y_obs, aprobado, _ = _cartera(n=6000)
    res1 = parcelling(X, y_obs, aprobado, factor=1.0)
    res2 = parcelling(X, y_obs, aprobado, factor=3.0)
    assert res1["detalle"]["tasa_mala_imputada_media"] < \
           res2["detalle"]["tasa_mala_imputada_media"]


def test_augmentacion_em_converge():
    _, X, _, y_obs, aprobado, _ = _cartera(n=6000)
    res = augmentacion_em(X, y_obs, aprobado, max_iter=25, tol=1e-5)
    assert res["detalle"]["iteraciones"] <= 25
    ultima = res["detalle"]["historia"][-1]
    assert ultima["cambio_maximo"] < 1e-3


def test_todos_los_metodos_producen_coeficientes_del_largo_correcto():
    df, X, X_sel, y_obs, aprobado, y_real = _cartera(n=4000)
    p = len(FEATURES) + 1
    for nombre, funcion in METODOS.items():
        res = funcion(X, X_sel, y_obs, aprobado, y_real)
        assert res["coef"].shape == (p,), nombre
        assert np.all(np.isfinite(res["coef"])), nombre


def test_probit_bivariado_con_instrumento_recupera_mejor_que_sin_el():
    """El resultado central del proyecto, verificado sobre una cartera
    completa (no el DGP simplificado de test_selection_models)."""
    df, X, X_sel, y_obs, aprobado, y_real = _cartera(n=20_000, regimen="mnar")
    beta_true = np.array([INTERCEPTO_TRUE, *[BETA_TRUE[f] for f in FEATURES]])

    con = METODOS["probit_bivariado"](X, X_sel, y_obs, aprobado, y_real)
    sin = METODOS["probit_bivariado_sin_instrumento"](X, X_sel, y_obs, aprobado, y_real)

    err_con = float(np.mean(np.abs(con["coef"] - beta_true)))
    err_sin = float(np.mean(np.abs(sin["coef"] - beta_true)))
    assert err_con < err_sin
    assert abs(con["detalle"]["rho_estimado"] - rho_verdadero("mnar")) < \
           abs(sin["detalle"]["rho_estimado"] - rho_verdadero("mnar"))


def test_oraculo_dado_su_acceso_a_toda_la_verdad_esta_entre_los_mejores():
    df, X, X_sel, y_obs, aprobado, y_real = _cartera(n=15_000, regimen="mnar")
    beta_true = np.array([INTERCEPTO_TRUE, *[BETA_TRUE[f] for f in FEATURES]])

    errores = {}
    for nombre, funcion in METODOS.items():
        res = funcion(X, X_sel, y_obs, aprobado, y_real)
        errores[nombre] = float(np.mean(np.abs(res["coef"] - beta_true)))

    # el oraculo no tiene por que ser estrictamente el minimo (es una
    # sola muestra finita), pero si tiene que quedar en el grupo mejor
    mediana = np.median(list(errores.values()))
    assert errores["oraculo"] <= mediana
