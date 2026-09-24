"""Tests unitarios de src/monitoring/drift.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    calculate_ks_drift,
    calculate_psi,
    classify_psi,
    generate_drift_report,
)


# ---------------------------------------------------------------- distribuciones idénticas

def test_identical_distributions_give_psi_approximately_zero():
    rng = np.random.default_rng(42)
    datos = rng.normal(size=5_000)

    assert calculate_psi(datos, datos.copy()) == pytest.approx(0.0, abs=1e-9)


def test_identical_distributions_give_ks_pvalue_approximately_one():
    rng = np.random.default_rng(42)
    datos = rng.normal(size=5_000)

    resultado = calculate_ks_drift(datos, datos.copy())

    assert resultado["statistic"] == pytest.approx(0.0, abs=1e-9)
    assert resultado["p_value"] == pytest.approx(1.0, abs=1e-9)
    assert resultado["drift_detected"] is False


def test_two_independent_samples_from_the_same_distribution_stay_stable():
    """No es solo el caso degenerado (mismo array): dos MUESTRAS distintas de
    la misma distribución tienen que quedar bien por debajo del umbral de
    alerta, no solo en cero exacto."""
    rng = np.random.default_rng(7)
    base = rng.normal(size=10_000)
    nuevo = rng.normal(size=10_000)

    psi = calculate_psi(base, nuevo)
    assert psi < 0.05
    assert classify_psi(psi) == "green"


# ---------------------------------------------------------------- distribución desplazada

def test_a_strongly_shifted_distribution_exceeds_the_critical_psi_threshold():
    rng = np.random.default_rng(3)
    base = rng.normal(loc=0.0, scale=1.0, size=5_000)
    corrida = rng.normal(loc=3.0, scale=1.0, size=5_000)  # +3 desvios: drift severo real

    psi = calculate_psi(base, corrida)

    assert psi > 0.25
    assert classify_psi(psi) == "red"


def test_a_strongly_shifted_distribution_is_also_flagged_by_ks():
    rng = np.random.default_rng(3)
    base = rng.normal(loc=0.0, scale=1.0, size=5_000)
    corrida = rng.normal(loc=3.0, scale=1.0, size=5_000)

    resultado = calculate_ks_drift(base, corrida)

    assert resultado["p_value"] < 0.001
    assert resultado["drift_detected"] is True


def test_a_moderate_shift_lands_in_the_yellow_band():
    """El umbral amarillo (0.10-0.25) tiene que ser alcanzable, no solo
    verde/rojo -- un corrimiento mas chico que el severo pero real."""
    rng = np.random.default_rng(11)
    base = rng.normal(loc=0.0, scale=1.0, size=8_000)
    corrida = rng.normal(loc=0.5, scale=1.0, size=8_000)

    psi = calculate_psi(base, corrida)
    assert 0.10 <= psi <= 0.25, f"psi={psi} no cayo en la banda amarilla esperada"
    assert classify_psi(psi) == "yellow"


def test_classify_psi_boundaries():
    assert classify_psi(0.0) == "green"
    assert classify_psi(0.099) == "green"
    assert classify_psi(0.10) == "yellow"
    assert classify_psi(0.25) == "yellow"
    assert classify_psi(0.250001) == "red"
    assert classify_psi(1.0) == "red"


# ---------------------------------------------------------------- manejo seguro de casos borde

def test_column_of_all_zeros_does_not_crash_and_reports_no_drift():
    base = np.zeros(200)
    nuevo = np.zeros(200)

    psi = calculate_psi(base, nuevo)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_nans_are_dropped_before_computing_psi():
    rng = np.random.default_rng(5)
    base = np.concatenate([rng.normal(size=500), [np.nan] * 50])
    nuevo = np.concatenate([rng.normal(size=500), [np.nan] * 50])

    # No debe lanzar, y el resultado no debe ser NaN.
    psi = calculate_psi(base, nuevo)
    assert np.isfinite(psi)


def test_all_nan_column_raises_a_clear_error():
    base = np.array([np.nan, np.nan, np.nan])
    nuevo = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="ningun valor finito"):
        calculate_psi(base, nuevo)


def test_few_data_points_do_not_crash_even_with_more_buckets_than_rows():
    """3 valores base con num_buckets=10: los cuantiles colapsan en cortes
    repetidos -- `np.unique` en `_bucket_edges` reduce los buckets
    efectivos en vez de fallar o dividir por un bucket vacio."""
    base = np.array([1.0, 2.0, 3.0])
    nuevo = np.array([1.5, 2.5, 3.5, 1.0])

    psi = calculate_psi(base, nuevo, num_buckets=10)
    assert np.isfinite(psi)


def test_a_single_unique_value_in_baseline_does_not_crash():
    """Caso extremo de 'columna constante': todos los cuantiles de `expected`
    colapsan en un solo corte, así que PSI por cuantiles no tiene con qué
    distinguir 7.0 de 9.0 -- da 0.0, no un error. Es una limitación real y
    esperada del método (comparte el mismo `searchsorted` que
    `06-optimal-binning-scorecard/src/monitoring.py::psi_score`, no algo
    introducido acá): 'manejo seguro' significa que no explota ni devuelve
    NaN/inf, no que resuelva un caso sin información para resolverlo."""
    base = np.full(300, 7.0)
    nuevo = np.concatenate([np.full(250, 7.0), np.full(50, 9.0)])

    psi = calculate_psi(base, nuevo)
    assert psi == pytest.approx(0.0)


def test_ks_drift_also_handles_nans_safely():
    base = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    nuevo = np.array([1.1, 2.1, 3.1, np.nan, np.nan])

    resultado = calculate_ks_drift(base, nuevo)
    assert np.isfinite(resultado["statistic"])
    assert np.isfinite(resultado["p_value"])


# ---------------------------------------------------------------- generate_drift_report

@pytest.fixture
def dataframes():
    rng = np.random.default_rng(99)
    baseline = pd.DataFrame({
        "estable": rng.normal(size=3_000),
        "corrida": rng.normal(size=3_000),
        "con_ceros": rng.integers(0, 2, size=3_000).astype(float),
    })
    scoring = pd.DataFrame({
        "estable": rng.normal(size=1_500),
        "corrida": rng.normal(loc=4.0, size=1_500),  # drift severo a propósito
        "con_ceros": rng.integers(0, 2, size=1_500).astype(float),
    })
    return baseline, scoring


def test_generate_drift_report_has_one_entry_per_feature(dataframes):
    baseline, scoring = dataframes
    reporte = generate_drift_report(baseline, scoring, ["estable", "corrida", "con_ceros"])

    assert set(reporte["features"].keys()) == {"estable", "corrida", "con_ceros"}
    for datos in reporte["features"].values():
        assert {"psi", "psi_status", "ks_statistic", "ks_p_value", "ks_drift_detected"} <= datos.keys()


def test_generate_drift_report_flags_the_shifted_feature_as_critical(dataframes):
    baseline, scoring = dataframes
    reporte = generate_drift_report(baseline, scoring, ["estable", "corrida", "con_ceros"])

    assert "corrida" in reporte["features_criticas"]
    assert reporte["features"]["corrida"]["psi_status"] == "red"
    assert reporte["features"]["corrida"]["ks_drift_detected"] is True


def test_generate_drift_report_does_not_flag_the_stable_feature(dataframes):
    baseline, scoring = dataframes
    reporte = generate_drift_report(baseline, scoring, ["estable", "corrida", "con_ceros"])

    assert "estable" not in reporte["features_en_alerta"]
    assert "estable" not in reporte["features_criticas"]
    assert reporte["features"]["estable"]["psi_status"] == "green"


def test_generate_drift_report_records_sample_sizes(dataframes):
    baseline, scoring = dataframes
    reporte = generate_drift_report(baseline, scoring, ["estable"])

    assert reporte["n_baseline"] == len(baseline)
    assert reporte["n_scoring"] == len(scoring)
