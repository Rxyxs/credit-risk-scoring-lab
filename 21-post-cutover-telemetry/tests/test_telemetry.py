"""Pruebas de PostCutoverTelemetryEngine.

`compute_realized_performance` se prueba contra valores conocidos de
antemano (una separacion perfecta con probabilidades limpias: 0.2/0.8),
no contra un dataset generico donde solo se podria verificar que "algun
numero salio". `compute_prediction_drift` se prueba tanto por deteccion
(una distribucion corrida de verdad produce un PSI alto, una igual
produce uno bajo) como por exactitud (una implementacion de referencia,
con un bucketing hecho a mano en Python puro en vez de vectorizado, debe
coincidir con la version rapida)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.telemetry_engine import (
    InsufficientTelemetryDataError, PostCutoverTelemetryEngine,
)


# ---------------------------------------------------------------------------
# compute_realized_performance
# ---------------------------------------------------------------------------

def _predicciones_y_verdad(client_ids, predicciones, etiquetas):
    predictions_df = pd.DataFrame({"client_id": client_ids, "prediction": predicciones})
    ground_truth_df = pd.DataFrame({"client_id": client_ids, "target": etiquetas})
    return predictions_df, ground_truth_df


def test_metricas_realizadas_contra_valores_conocidos_de_antemano():
    # Separacion perfecta con probabilidades limpias: AUC, Brier y log-loss
    # tienen forma cerrada, calculables a mano.
    predictions_df, ground_truth_df = _predicciones_y_verdad(
        ["c1", "c2", "c3", "c4"], [0.2, 0.2, 0.8, 0.8], [0, 0, 1, 1])

    engine = PostCutoverTelemetryEngine()
    resultado = engine.compute_realized_performance(predictions_df, ground_truth_df)

    assert resultado["realized_roc_auc"] == pytest.approx(1.0, abs=1e-9)
    assert resultado["brier_score"] == pytest.approx(0.04, abs=1e-9)
    assert resultado["log_loss"] == pytest.approx(-np.log(0.8), abs=1e-9)
    assert resultado["observed_default_rate"] == pytest.approx(0.5, abs=1e-9)
    assert resultado["predicted_default_rate"] == pytest.approx(0.5, abs=1e-9)
    assert resultado["n_evaluated"] == 4
    assert engine.performance == resultado


def test_auc_no_definida_con_una_sola_clase_pero_brier_si_se_calcula():
    predictions_df, ground_truth_df = _predicciones_y_verdad(
        ["c1", "c2", "c3"], [0.1, 0.3, 0.2], [0, 0, 0])

    resultado = PostCutoverTelemetryEngine().compute_realized_performance(
        predictions_df, ground_truth_df)

    assert resultado["realized_roc_auc"] is None
    assert resultado["auc_undefined_reason"] is not None
    assert resultado["brier_score"] == pytest.approx(np.mean([0.1**2, 0.3**2, 0.2**2]), abs=1e-9)


def test_columnas_faltantes_fallan_explicitamente():
    engine = PostCutoverTelemetryEngine()
    with pytest.raises(ValueError):
        engine.compute_realized_performance(
            pd.DataFrame({"client_id": ["c1"], "score": [0.5]}),
            pd.DataFrame({"client_id": ["c1"], "target": [0]}),
        )
    with pytest.raises(ValueError):
        engine.compute_realized_performance(
            pd.DataFrame({"client_id": ["c1"], "prediction": [0.5]}),
            pd.DataFrame({"client_id": ["c1"], "outcome": [0]}),
        )


def test_acepta_default_flag_como_alternativa_a_target():
    predictions_df = pd.DataFrame({"client_id": ["c1", "c2"], "prediction": [0.2, 0.8]})
    ground_truth_df = pd.DataFrame({"client_id": ["c1", "c2"], "default_flag": [0, 1]})

    resultado = PostCutoverTelemetryEngine().compute_realized_performance(
        predictions_df, ground_truth_df)
    assert resultado["n_evaluated"] == 2


def test_sin_client_id_en_comun_levanta_error_explicito():
    predictions_df = pd.DataFrame({"client_id": ["c1"], "prediction": [0.5]})
    ground_truth_df = pd.DataFrame({"client_id": ["c2"], "target": [1]})

    with pytest.raises(InsufficientTelemetryDataError):
        PostCutoverTelemetryEngine().compute_realized_performance(predictions_df, ground_truth_df)


# ---------------------------------------------------------------------------
# compute_prediction_drift (PSI)
# ---------------------------------------------------------------------------

def _psi_referencia(baseline: np.ndarray, recent: np.ndarray, num_buckets: int) -> float:
    """Misma definicion que `compute_prediction_drift`, pero con bucketing
    hecho a mano (un bisect por elemento) en vez de `np.searchsorted`
    vectorizado -- una implementacion deliberadamente distinta para
    verificar la rapida."""
    cortes = sorted(np.quantile(baseline, np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]))

    def bucket_de(x: float) -> int:
        b = 0
        for corte in cortes:
            if corte <= x:
                b += 1
            else:
                break
        return b

    conteo_b = [0] * num_buckets
    conteo_r = [0] * num_buckets
    for x in baseline:
        conteo_b[bucket_de(x)] += 1
    for x in recent:
        conteo_r[bucket_de(x)] += 1

    pct_b = [max(c / len(baseline), 1e-4) for c in conteo_b]
    pct_r = [max(c / len(recent), 1e-4) for c in conteo_r]
    return float(sum((pr - pb) * np.log(pr / pb) for pr, pb in zip(pct_r, pct_b)))


def test_psi_coincide_con_una_implementacion_de_referencia_no_vectorizada():
    rng = np.random.default_rng(11)
    baseline = rng.uniform(0, 1, size=317)
    recent = np.clip(rng.uniform(0, 1, size=241) + 0.2, 0, 1)

    rapido = PostCutoverTelemetryEngine().compute_prediction_drift(baseline, recent, num_buckets=8)
    referencia = _psi_referencia(baseline, recent, num_buckets=8)

    assert rapido["psi"] == pytest.approx(referencia, abs=1e-9)


def test_psi_detecta_una_distribucion_que_se_corrio_de_verdad():
    rng = np.random.default_rng(7)
    baseline = rng.uniform(0.0, 1.0, size=5000)
    recent_misma_distribucion = rng.uniform(0.0, 1.0, size=2000)
    recent_corrida = np.clip(rng.uniform(0.0, 1.0, size=2000) + 0.35, 0.0, 1.0)

    engine = PostCutoverTelemetryEngine()
    sin_corrimiento = engine.compute_prediction_drift(baseline, recent_misma_distribucion)
    con_corrimiento = engine.compute_prediction_drift(baseline, recent_corrida)

    assert sin_corrimiento["severity"] == "no_shift"
    assert sin_corrimiento["psi"] < 0.05

    assert con_corrimiento["severity"] == "significant_shift"
    assert con_corrimiento["psi"] > 0.25
    assert con_corrimiento["psi"] > sin_corrimiento["psi"]
    # El ultimo llamado es el que queda guardado en la instancia.
    assert engine.drift == con_corrimiento


def test_psi_con_baseline_insuficiente_falla_explicitamente():
    with pytest.raises(InsufficientTelemetryDataError):
        PostCutoverTelemetryEngine().compute_prediction_drift(
            baseline_preds=[0.1, 0.2, 0.3], recent_preds=[0.5], num_buckets=10)


# ---------------------------------------------------------------------------
# Maduracion insuficiente / reporte parcial
# ---------------------------------------------------------------------------

def test_nota_de_maduracion_insuficiente_no_calcula_metricas():
    engine = PostCutoverTelemetryEngine()
    nota = engine.note_insufficient_maturity(n_observed=12, min_required=50)

    assert nota["status"] == "INSUFFICIENT_MATURITY"
    assert nota["n_observed"] == 12
    assert nota["min_required"] == 50
    assert engine.performance == nota
    assert engine.drift is None


def test_reporte_parcial_por_maduracion_insuficiente_queda_escrito(tmp_path):
    engine = PostCutoverTelemetryEngine()
    engine.note_insufficient_maturity(n_observed=9, min_required=50)

    report_path = engine.generate_telemetry_report(tmp_path)
    assert report_path.exists()
    assert report_path.name.startswith("post_cutover_telemetry_")

    payload = json.loads(report_path.read_text())
    assert payload["realized_performance"]["status"] == "INSUFFICIENT_MATURITY"
    assert payload["prediction_drift"] is None
    assert "9" in payload["summary"]


def test_reporte_completo_combina_performance_y_drift(tmp_path):
    predictions_df, ground_truth_df = _predicciones_y_verdad(
        [f"c{i}" for i in range(60)],
        list(np.linspace(0.05, 0.95, 60)),
        [0] * 30 + [1] * 30,
    )
    engine = PostCutoverTelemetryEngine()
    engine.compute_realized_performance(predictions_df, ground_truth_df)
    engine.compute_prediction_drift(
        baseline_preds=np.linspace(0, 1, 200), recent_preds=np.linspace(0, 1, 100))

    report_path = engine.generate_telemetry_report(tmp_path)
    payload = json.loads(report_path.read_text())

    assert payload["realized_performance"]["status"] == "EVALUATED"
    assert payload["prediction_drift"]["psi"] == pytest.approx(engine.drift["psi"], abs=1e-9)
    assert "AUC" in payload["summary"]
    assert "PSI" in payload["summary"]


def test_generar_reporte_sin_nada_calculado_falla_explicitamente(tmp_path):
    with pytest.raises(RuntimeError):
        PostCutoverTelemetryEngine().generate_telemetry_report(tmp_path)
