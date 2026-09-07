"""Tests de las metricas con censura.

Los casos base estan calculados a mano: si Kaplan-Meier o el C-index se
comparan solo contra si mismos, un error sistematico pasa desapercibido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    auc_at_horizon, calibration_by_decile, calibration_summary,
    concordance_index, kaplan_meier, km_cum_default_at, ks_statistic,
    status_at_horizon,
)


def test_kaplan_meier_contra_calculo_a_mano():
    # 5 sujetos: eventos en t=2 y t=5, censura en t=3.
    dur = np.array([2, 3, 5, 5, 6])
    ev = np.array([1, 0, 1, 0, 0])
    km = kaplan_meier(dur, ev)

    # t=2: 5 en riesgo, 1 evento  -> S = 4/5 = 0.8
    # t=5: 3 en riesgo, 1 evento  -> S = 0.8 * 2/3 = 0.5333...
    assert km["tiempo"].tolist() == [2, 5]
    assert km["n_en_riesgo"].tolist() == [5, 3]
    assert np.allclose(km["survival"], [0.8, 0.8 * 2 / 3])
    assert np.allclose(km["cum_default"], [0.2, 1 - 0.8 * 2 / 3])


def test_kaplan_meier_evaluado_en_tiempos_arbitrarios():
    dur = np.array([2, 3, 5, 5, 6])
    ev = np.array([1, 0, 1, 0, 0])
    s = kaplan_meier(dur, ev, times=np.array([1, 2, 4, 10]))
    assert np.allclose(s["survival"], [1.0, 0.8, 0.8, 0.8 * 2 / 3])
    assert km_cum_default_at(dur, ev, 4) == pytest.approx(0.2)


def test_c_index_perfecto_y_aleatorio():
    dur = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ev = np.array([1, 1, 1, 1, 0])
    riesgo_perfecto = -dur                      # el que cae antes tiene mas riesgo
    assert concordance_index(dur, ev, riesgo_perfecto)["c_index"] == pytest.approx(1.0)
    assert concordance_index(dur, ev, dur)["c_index"] == pytest.approx(0.0)
    empatado = np.ones_like(dur)
    assert concordance_index(dur, ev, empatado)["c_index"] == pytest.approx(0.5)


def test_c_index_ignora_pares_no_comparables():
    # El sujeto censurado temprano (t=1) no forma pares comparables como
    # "el que cae antes", asi que no puede cambiar el resultado.
    dur = np.array([1.0, 2.0, 3.0])
    ev = np.array([0, 1, 1])
    res = concordance_index(dur, ev, np.array([99.0, 1.0, 0.0]))
    assert res["pares_comparables"] == 1
    assert res["c_index"] == pytest.approx(1.0)


def test_status_at_horizon_excluye_censurados_antes_del_corte():
    dur = np.array([5, 8, 12, 20, 30])
    ev = np.array([1, 0, 1, 0, 1])
    y, usable = status_at_horizon(dur, ev, horizonte=12)
    assert y.tolist() == [1, 0, 1, 0, 0]
    # el censurado en el mes 8 no tiene estado conocido a los 12 meses
    assert usable.tolist() == [True, False, True, True, True]


def test_auc_y_ks_sobre_estado_conocido():
    rng = np.random.default_rng(3)
    n = 2000
    riesgo = rng.uniform(0.01, 0.6, n)
    ev = (rng.random(n) < riesgo).astype(int)
    dur = np.where(ev == 1, rng.integers(1, 12, n), 36)
    res = auc_at_horizon(dur, ev, riesgo, horizonte=12)
    assert res["auc"] > 0.7
    assert res["gini"] == pytest.approx(2 * res["auc"] - 1)
    assert 0.0 <= res["brier"] <= 1.0
    assert 0.0 <= ks_statistic(dur, ev, riesgo, 12) <= 1.0


def test_calibracion_perfecta_da_error_bajo():
    rng = np.random.default_rng(5)
    n = 8000
    pd_true = rng.uniform(0.02, 0.5, n)
    cae = rng.random(n) < pd_true
    dur = np.where(cae, rng.integers(1, 13, n), 36)
    ev = cae.astype(int)
    cal = calibration_by_decile(dur, ev, pd_true, horizonte=12)
    resumen = calibration_summary(cal)
    assert len(cal) == 10
    assert resumen["error_absoluto_medio"] < 0.05
    assert isinstance(cal, pd.DataFrame)


def test_sin_pares_comparables_falla():
    with pytest.raises(ValueError):
        concordance_index(np.array([1.0, 2.0]), np.array([0, 0]), np.array([1.0, 2.0]))
