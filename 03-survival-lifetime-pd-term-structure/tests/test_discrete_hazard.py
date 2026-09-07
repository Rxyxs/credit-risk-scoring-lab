"""Tests del modelo de hazard en tiempo discreto y de la expansion
persona-periodo que lo alimenta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.discrete_hazard import DiscreteTimeHazard, likelihood_ratio_test
from src.preprocessing import to_person_period


def _cartera(n=4000, seed=13, meses=24):
    """Cartera simulada con hazard mensual conocido."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    informal = rng.binomial(1, 0.4, n).astype(float)

    duracion = np.full(n, meses, dtype=int)
    evento = np.zeros(n, dtype=int)
    activo = np.ones(n, dtype=bool)
    for t in range(1, meses + 1):
        idx = np.where(activo)[0]
        log_h = -3.3 + 0.7 * x[idx] + (1.2 * np.exp(-t / 6.0)) * informal[idx]
        cae = rng.random(idx.size) < (1 - np.exp(-np.exp(log_h)))
        duracion[idx[cae]] = t
        evento[idx[cae]] = 1
        activo[idx[cae]] = False

    return pd.DataFrame({
        "loan_id": np.arange(n),
        "x": x,
        "informal": informal,
        "duracion_meses": duracion,
        "evento_default": evento,
    })


def test_expansion_persona_periodo_a_mano():
    df = pd.DataFrame({
        "loan_id": [1, 2],
        "x": [0.5, -0.5],
        "duracion_meses": [3, 2],
        "evento_default": [1, 0],
    })
    pp = to_person_period(df, features=["x"])
    assert len(pp) == 5                                  # 3 + 2 filas credito-mes
    assert pp["mes"].tolist() == [1, 2, 3, 1, 2]
    # solo el ultimo mes del credito con evento lleva la etiqueta en 1
    assert pp["default_mes"].tolist() == [0, 0, 1, 0, 0]


def test_expansion_conserva_el_total_de_eventos():
    df = _cartera(n=500)
    pp = to_person_period(df, features=["x", "informal"])
    assert pp["default_mes"].sum() == df["evento_default"].sum()
    assert len(pp) == df["duracion_meses"].sum()


def test_design_matrix_tiene_una_dummy_por_mes():
    df = _cartera(n=300, meses=12)
    pp = to_person_period(df, features=["x", "informal"])
    m = DiscreteTimeHazard(["x", "informal"], max_mes=12)
    D = m.build_design(pp)
    assert D.shape == (len(pp), 12 + 2)
    assert np.allclose(D[:, :12].sum(axis=1), 1.0)       # exactamente un mes activo
    assert m.design_names[:2] == ["mes_1", "mes_2"]


def test_mes_fuera_de_rango_falla():
    m = DiscreteTimeHazard(["x"], max_mes=6)
    with pytest.raises(ValueError):
        m.build_design(pd.DataFrame({"mes": [7], "x": [0.0]}))


def test_recupera_efecto_y_curva_monotona():
    df = _cartera(n=6000, meses=24)
    pp = to_person_period(df, features=["x", "informal"])
    m = DiscreteTimeHazard(["x", "informal"], max_mes=24).fit(pp)

    coefs = dict(zip(m.design_names, m.coef_))
    assert coefs["x"] == pytest.approx(0.7, abs=0.12)

    cum = m.predict_cumulative_default(df.head(200))
    assert cum.shape == (200, 24)
    assert np.all(np.diff(cum, axis=1) >= -1e-12)        # PD acumulada no decrece
    assert np.all((cum >= 0) & (cum <= 1))
    assert np.all(m.predict_pd_at(df.head(200), 12) == cum[:, 11])


def test_lr_test_es_positivo_y_detecta_el_efecto_variable():
    df = _cartera(n=8000, meses=24)
    pp = to_person_period(df, features=["x", "informal"])
    ph = DiscreteTimeHazard(["x", "informal"], max_mes=24).fit(pp)
    tvc = DiscreteTimeHazard(["x", "informal"], tvc_features=["informal"], max_mes=24).fit(pp)

    lr = likelihood_ratio_test(ph, tvc)
    assert lr["df"] == 1
    assert lr["lr_stat"] > 0        # el modelo anidado nunca puede ajustar mejor
    assert lr["p_value"] < 0.01


def test_lr_test_rechaza_orden_invertido():
    df = _cartera(n=1000, meses=12)
    pp = to_person_period(df, features=["x", "informal"])
    ph = DiscreteTimeHazard(["x", "informal"], max_mes=12).fit(pp)
    tvc = DiscreteTimeHazard(["x", "informal"], tvc_features=["informal"], max_mes=12).fit(pp)
    with pytest.raises(ValueError):
        likelihood_ratio_test(tvc, ph)
