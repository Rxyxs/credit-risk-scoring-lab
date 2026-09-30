"""Pruebas de RetrainingPipelineOrchestrator.

Cada prueba construye su propio Feature Store (una tabla `credit_features`
minima, con el mismo esquema que deja `13-feature-store-duckdb`) y su
propia carpeta de disparadores, para no depender de que las tecnicas 12,
13 o 22 hayan corrido antes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.pipeline_orchestrator import EmptyFeatureStoreError, RetrainingPipelineOrchestrator


def _feature_store_real(db_path, n=150, seed=3):
    """Una tabla `credit_features` con señal real (no ruido puro), del
    mismo tamaño y forma que la que deja `13-feature-store-duckdb` sobre
    un snapshot de drift real."""
    rng = np.random.default_rng(seed)
    dti = rng.beta(2, 5, size=n)
    ingreso = rng.normal(800_000, 150_000, size=n)
    z = 3.0 * (dti - dti.mean()) / dti.std() - 1.5 * (ingreso - ingreso.mean()) / ingreso.std()
    default_flag = rng.binomial(1, 1 / (1 + np.exp(-z)))

    df = pd.DataFrame({
        "client_id": [f"CLI-{i:06d}" for i in range(n)],
        "ingreso_mensual": ingreso,
        "dti": dti,
        "antiguedad_laboral_meses": rng.exponential(36, size=n),
        "default_flag": default_flag,
    })
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE credit_features AS SELECT * FROM df")
    con.close()


def _trigger(trigger_dir, nombre, activado: bool):
    trigger_dir.mkdir(parents=True, exist_ok=True)
    path = trigger_dir / nombre
    payload = {
        "trigger_activated": activado,
        "reasons": ["realized_roc_auc=0.6800 por debajo del minimo 0.7200"] if activado else [],
        "evaluated_at": "2026-01-01T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# check_pending_triggers
# ---------------------------------------------------------------------------

def test_sin_disparadores_no_hay_nada_pendiente(tmp_path):
    trigger_dir = tmp_path / "triggers"
    trigger_dir.mkdir()

    encontrado = RetrainingPipelineOrchestrator().check_pending_triggers(trigger_dir)

    assert encontrado is False


def test_disparador_sin_activar_no_cuenta_como_pendiente(tmp_path):
    trigger_dir = tmp_path / "triggers"
    _trigger(trigger_dir, "retraining_trigger_20260101T000000Z.json", activado=False)

    orquestador = RetrainingPipelineOrchestrator()
    encontrado = orquestador.check_pending_triggers(trigger_dir)

    assert encontrado is False
    assert orquestador.pending_trigger_path is None


def test_carpeta_de_disparadores_inexistente_no_rompe_nada(tmp_path):
    encontrado = RetrainingPipelineOrchestrator().check_pending_triggers(tmp_path / "no_existe")
    assert encontrado is False


def test_json_de_disparador_corrupto_se_ignora_y_sigue_buscando(tmp_path):
    trigger_dir = tmp_path / "triggers"
    trigger_dir.mkdir()
    (trigger_dir / "retraining_trigger_20260101T000000Z.json").write_text("{ no es json valido ]")
    activo = _trigger(trigger_dir, "retraining_trigger_20260102T000000Z.json", activado=True)

    orquestador = RetrainingPipelineOrchestrator()
    encontrado = orquestador.check_pending_triggers(trigger_dir)

    assert encontrado is True
    assert orquestador.pending_trigger_path == activo


# ---------------------------------------------------------------------------
# run_retraining_flow: el camino feliz
# ---------------------------------------------------------------------------

def test_disparador_activo_desencadena_reentrenamiento_genera_pkl_y_registra_evento(tmp_path):
    feature_store_path = tmp_path / "offline_store.duckdb"
    _feature_store_real(feature_store_path, n=150)

    trigger_dir = tmp_path / "triggers"
    _trigger(trigger_dir, "retraining_trigger_20260101T000000Z.json", activado=True)

    db_path = tmp_path / "lab_lifecycle.duckdb"
    orquestador = RetrainingPipelineOrchestrator(state_path=tmp_path / "state.json")

    assert orquestador.check_pending_triggers(trigger_dir) is True

    resultado = orquestador.run_retraining_flow(
        feature_store_path=feature_store_path,
        models_output_dir=tmp_path / "models",
        reports_output_dir=tmp_path / "reports",
        db_path=db_path,
    )

    model_path = tmp_path / "models" / Path(resultado["model_path"]).name
    metrics_path = tmp_path / "reports" / Path(resultado["metrics_path"]).name
    assert model_path.exists()
    assert metrics_path.exists()
    assert model_path.name.startswith("shadow_model_")
    assert metrics_path.name.startswith("shadow_metrics_")

    # El .pkl es un Pipeline entrenado y usable, no un placeholder.
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    assert hasattr(pipeline, "predict_proba")

    metricas_guardadas = json.loads(metrics_path.read_text())
    assert metricas_guardadas["roc_auc"] == pytest.approx(resultado["metrics"]["roc_auc"], abs=1e-9)
    assert metricas_guardadas["n_samples"] == 150

    con = duckdb.connect(str(db_path))
    try:
        fila = con.execute(
            "SELECT event_type, previous_champion, new_champion FROM model_lifecycle_events "
            "WHERE event_id = ?", [resultado["event_id"]],
        ).fetchdf()
    finally:
        con.close()
    assert len(fila) == 1
    assert fila.iloc[0]["event_type"] == "RETRAINING_EXECUTED"


def test_disparador_procesado_no_se_vuelve_a_detectar_en_la_siguiente_corrida(tmp_path):
    feature_store_path = tmp_path / "offline_store.duckdb"
    _feature_store_real(feature_store_path, n=150)
    trigger_dir = tmp_path / "triggers"
    _trigger(trigger_dir, "retraining_trigger_20260101T000000Z.json", activado=True)
    state_path = tmp_path / "state.json"

    primera = RetrainingPipelineOrchestrator(state_path=state_path)
    primera.check_pending_triggers(trigger_dir)
    primera.run_retraining_flow(
        feature_store_path=feature_store_path,
        models_output_dir=tmp_path / "models",
        reports_output_dir=tmp_path / "reports",
        db_path=tmp_path / "lab_lifecycle.duckdb",
    )

    # Una instancia nueva, releyendo el mismo estado persistido en disco.
    segunda = RetrainingPipelineOrchestrator(state_path=state_path)
    assert segunda.check_pending_triggers(trigger_dir) is False


# ---------------------------------------------------------------------------
# Regresion: directorio padre de la base DuckDB
# ---------------------------------------------------------------------------

def test_directorio_padre_de_duckdb_inexistente_se_crea_automaticamente(tmp_path):
    """`duckdb.connect` no crea su directorio padre por si solo -- la misma
    clase de bug ya encontrada y corregida en las tecnicas 20 y 22."""
    feature_store_path = tmp_path / "offline_store.duckdb"
    _feature_store_real(feature_store_path, n=150)
    trigger_dir = tmp_path / "triggers"
    _trigger(trigger_dir, "retraining_trigger_20260101T000000Z.json", activado=True)

    db_path = tmp_path / "no_existe_todavia" / "otro_nivel" / "lab_lifecycle.duckdb"
    assert not db_path.parent.exists()

    orquestador = RetrainingPipelineOrchestrator()
    orquestador.check_pending_triggers(trigger_dir)
    orquestador.run_retraining_flow(
        feature_store_path=feature_store_path,
        models_output_dir=tmp_path / "models",
        reports_output_dir=tmp_path / "reports",
        db_path=db_path,
    )

    assert db_path.exists()


# ---------------------------------------------------------------------------
# Feature Store vacio o inaccesible
# ---------------------------------------------------------------------------

def test_feature_store_inexistente_lanza_error_claro(tmp_path):
    with pytest.raises(EmptyFeatureStoreError):
        RetrainingPipelineOrchestrator().run_retraining_flow(
            feature_store_path=tmp_path / "no_existe.duckdb",
            models_output_dir=tmp_path / "models",
            reports_output_dir=tmp_path / "reports",
            db_path=tmp_path / "lab_lifecycle.duckdb",
        )


def test_feature_store_sin_tabla_credit_features_lanza_error_claro(tmp_path):
    db_path = tmp_path / "offline_store.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE otra_tabla (x INTEGER)")
    con.close()

    with pytest.raises(EmptyFeatureStoreError):
        RetrainingPipelineOrchestrator().run_retraining_flow(
            feature_store_path=db_path,
            models_output_dir=tmp_path / "models",
            reports_output_dir=tmp_path / "reports",
            db_path=tmp_path / "lab_lifecycle.duckdb",
        )


def test_feature_store_con_menos_del_minimo_de_filas_lanza_error_claro(tmp_path):
    db_path = tmp_path / "offline_store.duckdb"
    _feature_store_real(db_path, n=10)

    with pytest.raises(EmptyFeatureStoreError):
        RetrainingPipelineOrchestrator().run_retraining_flow(
            feature_store_path=db_path,
            models_output_dir=tmp_path / "models",
            reports_output_dir=tmp_path / "reports",
            db_path=tmp_path / "lab_lifecycle.duckdb",
        )


def test_feature_store_sin_columna_de_target_reconocible_lanza_error_claro(tmp_path):
    db_path = tmp_path / "offline_store.duckdb"
    df = pd.DataFrame({
        "client_id": [f"c{i}" for i in range(150)],
        "ingreso_mensual": np.random.default_rng(1).normal(size=150),
    })
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE credit_features AS SELECT * FROM df")
    con.close()

    with pytest.raises(EmptyFeatureStoreError):
        RetrainingPipelineOrchestrator().run_retraining_flow(
            feature_store_path=db_path,
            models_output_dir=tmp_path / "models",
            reports_output_dir=tmp_path / "reports",
            db_path=tmp_path / "lab_lifecycle.duckdb",
        )
