"""Tests del simulador de bancos y del experimento de comparacion de
politicas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    BANCOS, DESVIO_CONCEPTO_NO_IID, FEATURES, generar_carteras,
)
from src.experiment import _splits, _Xy, correr_escenario, metricas


# --------------------------------------------------------------- datos
def test_reproducible():
    a = generar_carteras("no_iid", seed=1)
    b = generar_carteras("no_iid", seed=1)
    assert a.equals(b)


def test_seis_bancos_presentes():
    df = generar_carteras("iid", seed=2)
    assert set(df["banco"]) == set(BANCOS)
    assert len(df) == sum(p["n"] for p in BANCOS.values())


def test_escenario_invalido_falla():
    with pytest.raises(ValueError):
        generar_carteras("otro")


def test_iid_los_bancos_tienen_tasas_de_default_parecidas():
    """En IID, la unica diferencia entre bancos es el tamano de muestra:
    las tasas de default no deberian diferir mucho mas alla del ruido."""
    df = generar_carteras("iid", seed=3)
    tasas = df.groupby("banco")["default_12m"].mean()
    assert tasas.max() - tasas.min() < 0.06


def test_no_iid_los_bancos_difieren_marcadamente():
    """En no-IID, el rango de tasas de default entre bancos tiene que ser
    mucho mas ancho que en IID -- es la heterogeneidad que motiva todo
    el proyecto."""
    df = generar_carteras("no_iid", seed=4)
    tasas = df.groupby("banco")["default_12m"].mean()
    assert tasas.max() - tasas.min() > 0.25


def test_banco_microcredito_es_el_mas_riesgoso_en_no_iid():
    df = generar_carteras("no_iid", seed=5)
    tasas = df.groupby("banco")["default_12m"].mean()
    assert tasas.idxmax() == "Banco_Microcredito_Informal"


def test_estandarizacion_es_nacional_no_por_banco():
    """Si la estandarizacion fuera por banco, cada uno tendria media 0 en
    log_renta_z -- que destruiria la senal de que un banco sirve a gente
    de mas o menos ingresos que otro. Con estandarizacion nacional, las
    medias por banco tienen que diferir en no-IID."""
    df = generar_carteras("no_iid", seed=6)
    medias = df.groupby("banco")["log_renta_z"].mean()
    assert medias.max() - medias.min() > 0.3


def test_desvio_de_concepto_solo_aplica_en_no_iid():
    """El coeficiente de DTI implicito (pendiente de pd_verdadera respecto
    de dti, dentro de cada banco) tiene que variar entre bancos en no-IID
    y ser mucho mas parejo en IID."""
    def pendiente_dti_por_banco(df):
        pendientes = {}
        for b, g in df.groupby("banco"):
            # regresion simple univariada como proxy del coeficiente local
            x = g["dti"].to_numpy()
            y = np.log(np.clip(g["pd_verdadera"], 1e-6, 1 - 1e-6)
                      / (1 - np.clip(g["pd_verdadera"], 1e-6, 1 - 1e-6)))
            pendientes[b] = np.polyfit(x, y, 1)[0]
        return pd.Series(pendientes)

    iid = pendiente_dti_por_banco(generar_carteras("iid", seed=7))
    no_iid = pendiente_dti_por_banco(generar_carteras("no_iid", seed=7))
    assert (no_iid.max() - no_iid.min()) > 3 * (iid.max() - iid.min())


def test_desvio_de_concepto_tiene_una_entrada_por_banco():
    assert set(DESVIO_CONCEPTO_NO_IID) == set(BANCOS)


# ----------------------------------------------------------- metricas
def test_metricas_calibracion_calculada_a_mano():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.9, 0.5, 0.5])
    m = metricas(y, p)
    assert m["pd_media_predicha"] == pytest.approx(0.5)
    assert m["tasa_default_real"] == pytest.approx(0.5)
    assert m["sesgo_calibracion_pp"] == pytest.approx(0.0, abs=1e-9)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_metricas_detectan_sobreestimacion():
    y = np.array([0, 0, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    m = metricas(y, p)
    assert m["sesgo_calibracion_pp"] > 0      # predice mas riesgo del que hay


# ----------------------------------------------------------- experimento
def test_splits_son_disjuntos_y_cubren_todo():
    df = generar_carteras("no_iid", seed=8)
    trains, tests, pool_train, pool_test = _splits(df, seed=8)
    for b in BANCOS:
        ids_tr = set(trains[b]["applicant_id"])
        ids_te = set(tests[b]["applicant_id"])
        assert ids_tr & ids_te == set()
        assert len(ids_tr) + len(ids_te) == (df["banco"] == b).sum()
    assert len(pool_train) == sum(len(trains[b]) for b in BANCOS)
    assert len(pool_test) == sum(len(tests[b]) for b in BANCOS)


@pytest.fixture(scope="module")
def resultado_no_iid():
    return correr_escenario("no_iid", local_epochs=5)


def test_federado_calibra_mejor_que_solo_local_en_no_iid(resultado_no_iid):
    tabla = resultado_no_iid["tabla"]
    solo_local = tabla[tabla["politica"] == "solo_local"]
    federado = tabla[tabla["politica"] == "federado"]

    sesgo_local = solo_local["eval_nacional_sesgo_calibracion_pp"].abs().mean()
    sesgo_federado = federado["eval_nacional_sesgo_calibracion_pp"].abs().mean()
    assert sesgo_federado < sesgo_local


def test_banco_outlier_tiene_el_peor_sesgo_de_calibracion_local(resultado_no_iid):
    tabla = resultado_no_iid["tabla"]
    solo_local = tabla[tabla["politica"] == "solo_local"].set_index("banco")
    peor = solo_local["eval_nacional_sesgo_calibracion_pp"].abs().idxmax()
    assert peor == "Banco_Microcredito_Informal"


def test_similitud_de_updates_marca_al_outlier_en_no_iid(resultado_no_iid):
    sim = resultado_no_iid["similitud_updates_ronda1"]
    otros = [b for b in BANCOS if b != "Banco_Microcredito_Informal"]
    similitud_con_outlier = sim.loc["Banco_Microcredito_Informal", otros].mean()
    similitud_entre_otros = sim.loc[otros, otros].to_numpy()
    similitud_entre_otros = similitud_entre_otros[
        ~np.eye(len(otros), dtype=bool)
    ].mean()
    assert similitud_con_outlier < similitud_entre_otros


def test_iid_no_produce_un_outlier_de_similitud():
    r = correr_escenario("iid", local_epochs=5)
    sim = r["similitud_updates_ronda1"].to_numpy()
    fuera_diagonal = sim[~np.eye(len(sim), dtype=bool)]
    assert fuera_diagonal.min() > 0.9      # todos parecidos entre si
