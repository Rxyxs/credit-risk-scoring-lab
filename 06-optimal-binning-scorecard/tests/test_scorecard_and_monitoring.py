"""Tests de la tarjeta de puntos, del monitoreo y del simulador."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.binning import OptimalBinner
from src.data_generator import VINTAGE_QUIEBRE, generate_applicants
from src.monitoring import (
    UMBRAL_ALERTA, UMBRAL_CRITICO, csi_variable, monitoreo_por_vintage,
    psi_desde_conteos, psi_score, resumen_monitoreo, semaforo,
)
from src.scorecard import (
    ODDS_BASE, PDO, PUNTAJE_BASE, ajustar_scorecard, bandas_de_riesgo,
    construir_tarjeta, ks_statistic, metricas,
)


def _cartera(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.gamma(2.0, 0.12, n),
        "x2": rng.normal(0, 1, n),
    })
    eta = -2.0 + 4.0 * df["x1"] + 0.8 * df["x2"]
    df["default_12m"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return df


def _scorecard_de_prueba(df):
    y = df["default_12m"].to_numpy(float)
    binnings = {
        v: OptimalBinner(max_bins=5, min_eventos=20).fit_numerica(df[v].to_numpy(), y, v)
        for v in ("x1", "x2")
    }
    return ajustar_scorecard(df, y, binnings)


# ------------------------------------------------------------- scorecard
def test_la_escala_de_puntos_cumple_la_definicion_de_pdo():
    """Duplicar los odds tiene que mover el puntaje exactamente un PDO."""
    df = _cartera()
    sc = _scorecard_de_prueba(df)
    score = sc.score(df)
    proba = sc.predict_proba(df)
    odds = (1 - proba) / proba          # odds de ser bueno

    # dos clientes cualesquiera: la diferencia de puntaje tiene que ser
    # PDO * log2(razon de odds)
    i, j = np.argsort(score)[[10, -10]]
    esperado = PDO * np.log2(odds[j] / odds[i])
    assert score[j] - score[i] == pytest.approx(esperado, rel=1e-9)


def test_el_puntaje_base_corresponde_a_los_odds_base():
    df = _cartera(seed=2)
    sc = _scorecard_de_prueba(df)
    score = sc.score(df)
    proba = sc.predict_proba(df)
    odds = (1 - proba) / proba
    # invirtiendo la escala se recupera el puntaje: base + PDO*log2(odds/base)
    reconstruido = PUNTAJE_BASE + PDO * np.log2(odds / ODDS_BASE)
    assert np.allclose(score, reconstruido)


def test_mas_puntos_significa_menos_riesgo():
    """La relacion es monotona, no lineal: el puntaje es lineal en el
    log-odds y la PD es su sigmoide, asi que lo que tiene que dar -1 es la
    correlacion de rangos, no la de Pearson."""
    from scipy.stats import spearmanr

    df = _cartera(seed=3)
    sc = _scorecard_de_prueba(df)
    rho = spearmanr(sc.score(df), sc.predict_proba(df)).statistic
    assert rho == pytest.approx(-1.0, abs=1e-9)


def test_la_tarjeta_cubre_todos_los_bins_de_todas_las_variables():
    df = _cartera(seed=4)
    sc = _scorecard_de_prueba(df)
    esperado = sum(len(r.tabla) for r in sc.binnings.values())
    assert len(sc.tarjeta) == esperado
    assert set(sc.tarjeta["variable"]) == set(sc.binnings)
    assert sc.tarjeta["puntos"].notna().all()

    # reconstruir la tarjeta con los mismos insumos da lo mismo
    otra = construir_tarjeta(sc.binnings, sc.coeficientes, sc.intercepto)
    assert np.allclose(otra["puntos"], sc.tarjeta["puntos"])


def test_bandas_por_quintil_separan_el_riesgo():
    df = _cartera(seed=5)
    sc = _scorecard_de_prueba(df)
    score = sc.score(df)
    tabla = bandas_de_riesgo(score, df["default_12m"].to_numpy())
    assert len(tabla) == 5
    # El puntaje de un scorecard es discreto (suma de puntos de bins), asi
    # que con pocas variables los quintiles no pueden quedar exactos: con
    # dos variables de 5 bins hay a lo mas 25 puntajes distintos.
    assert np.allclose(tabla["pct_poblacion"], 0.2, atol=0.06)
    # E es la peor banda y A la mejor
    assert tabla.iloc[0]["tasa_mala"] > tabla.iloc[-1]["tasa_mala"]


def test_metricas_y_ks_coherentes():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    m = metricas(y, p)
    assert m["auc"] == pytest.approx(1.0)
    assert m["gini"] == pytest.approx(1.0)
    assert ks_statistic(y, p) == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(float(np.mean((p - y) ** 2)))


# ------------------------------------------------------------ monitoreo
def test_psi_cero_cuando_no_cambia_nada():
    conteos = np.array([100.0, 200.0, 300.0])
    assert psi_desde_conteos(conteos, conteos) == pytest.approx(0.0, abs=1e-12)
    assert psi_desde_conteos(conteos, conteos * 3) == pytest.approx(0.0, abs=1e-12)


def test_psi_calculado_a_mano():
    base = np.array([50.0, 50.0])
    nuevo = np.array([80.0, 20.0])
    esperado = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi_desde_conteos(base, nuevo) == pytest.approx(esperado)
    assert psi_desde_conteos(base, nuevo) > UMBRAL_CRITICO


def test_psi_crece_con_la_magnitud_del_desplazamiento():
    rng = np.random.default_rng(6)
    base = rng.normal(0, 1, 20_000)
    valores = [psi_score(base, base + d)[0] for d in (0.0, 0.1, 0.3, 0.8)]
    assert valores == sorted(valores)
    assert valores[0] < 0.01 < valores[-1]


def test_semaforo_usa_los_umbrales_declarados():
    assert semaforo(UMBRAL_ALERTA - 1e-6) == "estable"
    assert semaforo(UMBRAL_ALERTA) == "alerta"
    assert semaforo(UMBRAL_CRITICO) == "critico"


def test_psi_falla_con_poblacion_vacia_o_bins_distintos():
    with pytest.raises(ValueError):
        psi_desde_conteos(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        psi_desde_conteos(np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.0]))


def test_el_csi_senala_la_variable_que_efectivamente_se_movio():
    df = _cartera(seed=7)
    y = df["default_12m"].to_numpy(float)
    b1 = OptimalBinner(max_bins=5, min_eventos=20).fit_numerica(df["x1"].to_numpy(), y, "x1")
    b2 = OptimalBinner(max_bins=5, min_eventos=20).fit_numerica(df["x2"].to_numpy(), y, "x2")

    movido = df.copy()
    movido["x1"] = movido["x1"] * 1.8           # solo x1 se desplaza
    csi1 = csi_variable(b1, df["x1"], movido["x1"])
    csi2 = csi_variable(b2, df["x2"], movido["x2"])
    assert csi1 > UMBRAL_CRITICO
    assert csi2 == pytest.approx(0.0, abs=1e-12)


def test_monitoreo_por_vintage_devuelve_una_fila_por_cohorte():
    df = _cartera(seed=8)
    df["vintage_idx"] = np.repeat(np.arange(6), len(df) // 6)
    sc = _scorecard_de_prueba(df)
    tabla = monitoreo_por_vintage(sc, df, df)
    assert len(tabla) == 6
    assert {"psi_score", "estado_psi", "csi_x1", "csi_x2"} <= set(tabla.columns)
    # comparar la referencia consigo misma no puede dar alertas grandes
    assert tabla["psi_score"].max() < UMBRAL_ALERTA

    resumen = resumen_monitoreo(tabla)
    assert resumen["primer_vintage_critico"] is None
    assert resumen["vintages_estables"] == 6


# ------------------------------------------------------------- datos
def test_el_simulador_tiene_cohortes_y_un_quiebre_real():
    df = generate_applicants(n=6000, seed=5, n_vintages=24)
    assert df["vintage_idx"].nunique() == 24
    estables = df[df["vintage_idx"] < VINTAGE_QUIEBRE]
    deteriorados = df[df["vintage_idx"] >= VINTAGE_QUIEBRE]
    assert deteriorados["default_12m"].mean() > estables["default_12m"].mean() + 0.04
    assert deteriorados["utilizacion_lineas"].mean() > estables["utilizacion_lineas"].mean()
    assert (deteriorados["tipo_contrato"] == "informal").mean() > \
           (estables["tipo_contrato"] == "informal").mean()


def test_el_simulador_es_reproducible_y_tiene_senal():
    a = generate_applicants(n=3000, seed=9)
    b = generate_applicants(n=3000, seed=9)
    assert a.equals(b)
    alto = a.loc[a["dti"] > a["dti"].quantile(0.9), "default_12m"].mean()
    bajo = a.loc[a["dti"] < a["dti"].quantile(0.1), "default_12m"].mean()
    assert alto > bajo * 1.25
    con_moras = a.loc[a["n_moras_12m"] >= 2, "default_12m"].mean()
    sin_moras = a.loc[a["n_moras_12m"] == 0, "default_12m"].mean()
    assert con_moras > sin_moras * 1.5
