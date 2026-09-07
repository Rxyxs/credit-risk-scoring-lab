"""Tests de la capa de negocio: bandas, estructura temporal y provision."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import apply_scaler, fit_scaler, load_loan_book, split_train_test
from src.term_structure import (
    LGD, UMBRAL_SICR, asignar_bandas, curva_agregada, provision_ifrs9, tabla_por_banda,
)


def _cartera_falsa(n=500, meses=36, seed=2):
    rng = np.random.default_rng(seed)
    pd_12 = np.clip(rng.beta(2, 12, n), 0.001, 0.9)
    # curva acumulada creciente y consistente con la PD a 12 meses
    forma = np.linspace(0.35, 1.0, meses) ** 0.8
    forma = forma / forma[11]
    curvas = np.clip(pd_12[:, None] * forma[None, :], 0, 0.999)
    test = pd.DataFrame({
        "loan_id": np.arange(n),
        "monto_credito": rng.uniform(500_000, 10_000_000, n),
        "duracion_meses": rng.integers(1, meses + 1, n),
        "evento_default": rng.binomial(1, 0.2, n),
    })
    return test, curvas


def test_bandas_ordenan_por_riesgo():
    _, curvas = _cartera_falsa()
    pd_12 = curvas[:, 11]
    bandas = asignar_bandas(pd_12)
    medias = pd.Series(pd_12).groupby(bandas).mean()
    assert list(medias.index) == ["A", "B", "C", "D", "E"]
    assert medias.is_monotonic_increasing


def test_tabla_por_banda_es_creciente_en_horizonte():
    test, curvas = _cartera_falsa()
    tabla = tabla_por_banda(test, curvas)
    assert len(tabla) == 5
    for _, fila in tabla.iterrows():
        assert fila["pd_pred_6m"] <= fila["pd_pred_12m"] <= fila["pd_pred_24m"] <= fila["pd_pred_36m"]
    assert tabla["pd_pred_12m"].is_monotonic_increasing
    assert (tabla["pct_riesgo_despues_de_12m"] > 0).all()


def test_curva_agregada_marginal_suma_la_acumulada():
    _, curvas = _cartera_falsa()
    ag = curva_agregada(curvas)
    assert len(ag) == curvas.shape[1]
    assert ag["pd_marginal"].sum() == pytest.approx(ag["pd_acumulada"].iloc[-1])
    assert (ag["pd_marginal"] >= -1e-12).all()


def test_provision_lifetime_es_mayor_que_solo_12m():
    test, curvas = _cartera_falsa()
    p = provision_ifrs9(test, curvas)
    assert p["ecl_con_staging_ifrs9_clp"] >= p["ecl_solo_12m_clp"]
    assert p["uplift_pct"] > 0
    assert p["n_stage_1"] + p["n_stage_2"] == len(test)
    assert p["lgd_supuesta"] == LGD


def test_umbral_sicr_manda_a_stage_2_a_la_cola_de_riesgo():
    test, curvas = _cartera_falsa()
    p = provision_ifrs9(test, curvas)
    pd_12 = curvas[:, 11]
    esperados = int((pd_12 > UMBRAL_SICR * pd_12.mean()).sum())
    assert p["n_stage_2"] == esperados
    assert p["pct_cartera_stage_2"] == pytest.approx(esperados / len(test))


def test_split_no_filtra_creditos_entre_train_y_test():
    df = load_loan_book()
    train, test = split_train_test(df)
    assert len(train) + len(test) == len(df)
    assert set(train["loan_id"]) & set(test["loan_id"]) == set()
    # la estandarizacion se ajusta solo en train
    scaler = fit_scaler(train)
    train_z = apply_scaler(train, scaler)
    assert train_z["dti_z"].mean() == pytest.approx(0.0, abs=1e-9)
    assert train_z["dti_z"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)
