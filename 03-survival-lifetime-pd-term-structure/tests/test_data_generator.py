"""Tests del simulador de la cartera.

Un generador con un bug silencioso (etiqueta que no depende de las
features, censura mal aplicada) hace que todo lo que viene despues valide
contra nada. Estos tests fijan las propiedades que el resto del pipeline
asume."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_generator import (
    BETA_TRUE, GAMMA_INFORMAL_0, baseline_log_hazard, gamma_informal,
    generate_portfolio,
)


def test_estructura_y_reproducibilidad():
    df, gt = generate_portfolio(n=2000, seed=1)
    df2, _ = generate_portfolio(n=2000, seed=1)
    assert len(df) == 2000
    assert df.equals(df2)
    assert set(BETA_TRUE) <= set(gt["standardization_moments"])
    assert gt["seed"] == 1


def test_semillas_distintas_dan_carteras_distintas():
    df1, _ = generate_portfolio(n=1000, seed=1)
    df2, _ = generate_portfolio(n=1000, seed=2)
    assert not df1["duracion_meses"].equals(df2["duracion_meses"])


def test_censura_y_duracion_consistentes():
    df, _ = generate_portfolio(n=3000, seed=4)
    assert (df["duracion_meses"] >= 1).all()
    assert (df["duracion_meses"] <= df["ventana_obs_meses"]).all()
    assert (df["ventana_obs_meses"] == np.minimum(df["plazo_meses"], 36)).all()

    # la etiqueta de evento y el motivo de salida no pueden contradecirse
    assert (df.loc[df["evento_default"] == 1, "motivo_salida"] == "default").all()
    assert (df.loc[df["evento_default"] == 0, "motivo_salida"] != "default").all()

    # quien llega al fin de la ventana sale justo en ese mes
    fin = df["motivo_salida"] == "fin_ventana"
    assert (df.loc[fin, "duracion_meses"] == df.loc[fin, "ventana_obs_meses"]).all()


def test_tasa_de_default_en_rango_de_cartera_de_consumo():
    df, _ = generate_portfolio(n=6000, seed=8)
    assert 0.10 < df["evento_default"].mean() < 0.30


def test_el_riesgo_esta_realmente_en_las_features():
    """La senal debe ser aprendible: mas moras -> mas default observado."""
    df, _ = generate_portfolio(n=8000, seed=9)
    sin_moras = df.loc[df["n_moras_12m"] == 0, "evento_default"].mean()
    con_moras = df.loc[df["n_moras_12m"] >= 2, "evento_default"].mean()
    assert con_moras > sin_moras * 1.5

    alto_dti = df.loc[df["dti"] > df["dti"].quantile(0.8), "evento_default"].mean()
    bajo_dti = df.loc[df["dti"] < df["dti"].quantile(0.2), "evento_default"].mean()
    assert alto_dti > bajo_dti


def test_hazard_base_tiene_forma_de_joroba():
    t = np.arange(1, 37)
    h = np.exp(baseline_log_hazard(t))
    peak = int(t[np.argmax(h)])
    assert 5 <= peak <= 14              # seasoning: sube y luego baja
    assert h[-1] < h[peak - 1]
    assert h[0] < h[peak - 1]


def test_efecto_informal_decae_con_el_tiempo():
    g = gamma_informal(np.array([1, 12, 36]))
    assert g[0] == pytest.approx(GAMMA_INFORMAL_0 * np.exp(-0.1), rel=1e-9)
    assert g[0] > g[1] > g[2]
    assert g[2] < 0.1 * g[0]


def test_informales_caen_antes_que_formales():
    df, _ = generate_portfolio(n=8000, seed=10)
    d_inf = df.loc[(df["tipo_contrato"] == "informal") & (df["evento_default"] == 1),
                   "duracion_meses"].median()
    d_for = df.loc[(df["tipo_contrato"] == "formal") & (df["evento_default"] == 1),
                   "duracion_meses"].median()
    assert d_inf <= d_for
