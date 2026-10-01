"""Pruebas de ModelLineageTracker.

Cada prueba construye su propia version en miniatura de la cadena
12->14->15->20 (reportes JSON + una tabla model_lifecycle_events) para no
depender de que esas tecnicas hayan corrido antes."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from src.lineage_tracker import ModelLineageTracker


def _escribir_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _construir_cadena_completa(tmp_path):
    """Una version minima, pero fiel, de lo que las tecnicas 12, 14, 15 y
    20 dejarian en disco para un unico modelo que fue entrenado,
    promovido, y puesto en produccion."""
    ts_trigger = "20260101T000000Z"
    ts_entrenamiento = "20260102T000000Z"
    ts_cutover = "20260103T000000Z"

    dirs = {
        "drift": tmp_path / "12",
        "metrics": tmp_path / "14",
        "decisions": tmp_path / "15",
        "registry": tmp_path / "16" / "registry",
        "cutover": tmp_path / "20",
    }
    shadow_name = f"shadow_model_{ts_entrenamiento}.pkl"

    _escribir_json(dirs["drift"] / f"retrain_manifest_{ts_trigger}.json", {
        "trigger_reason": "CRITICAL_DRIFT_PSI",
        "features_affected": ["ingreso_mensual"],
        "current_model_version": "unknown",
        "snapshot_path": str(dirs["drift"] / f"retrain_data_{ts_trigger}.csv"),
        "suggested_action": "RETRAIN_SHADOW_MODEL",
        "generated_at": "2026-01-01T00:00:00+00:00",
    })
    _escribir_json(dirs["metrics"] / f"shadow_metrics_{ts_entrenamiento}.json", {
        "roc_auc": 0.81, "n_samples": 2000, "n_train": 1600, "n_test": 400,
        "trained_at": "2026-01-02T00:00:00+00:00",
    })
    _escribir_json(dirs["decisions"] / f"promotion_decision_{ts_entrenamiento}.json", {
        "candidate_model": shadow_name, "decision": "PROMOTED",
        "reason": "Meets minimum ROC-AUC and sample size thresholds",
        "evaluated_at": "2026-01-02T00:05:00+00:00",
    })
    # Como en la cadena real: 16-shadow-deployment copia el candidato a un
    # nombre fijo (`active_shadow_model.pkl`), nunca al nombre original --
    # el cutover (tecnica 20) termina apuntando a ese nombre fijo, y el
    # nombre real solo sobrevive un salto mas atras, en
    # `registry_manifest.json`.
    active_shadow_path = dirs["registry"] / "active_shadow_model.pkl"
    _escribir_json(dirs["registry"] / "registry_manifest.json", {
        "active_version": shadow_name,
        "activated_at": "2026-01-02T12:00:00+00:00",
        "source_decision": f"promotion_decision_{ts_entrenamiento}.json",
    })
    _escribir_json(dirs["cutover"] / f"cutover_manifest_{ts_cutover}.json", {
        "event_id": "evt-1", "event_type": "FULL_CUTOVER",
        "previous_champion": None, "new_champion": "champion_model.pkl",
        "promoted_at": "2026-01-03T00:00:00+00:00",
        "champion_path": "models/champion/champion_model.pkl", "archived_path": None,
        "shadow_model_path": str(active_shadow_path),
        "canary_config_path": "canary_config.json", "db_path": "lab_lifecycle.duckdb",
    })

    db_path = tmp_path / "lab_lifecycle.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE model_lifecycle_events (
            event_id VARCHAR, event_type VARCHAR, previous_champion VARCHAR,
            new_champion VARCHAR, promoted_at VARCHAR
        )
    """)
    con.execute(
        "INSERT INTO model_lifecycle_events VALUES (?, ?, ?, ?, ?)",
        ["evt-1", "FULL_CUTOVER", None, "champion_model.pkl", "2026-01-03T00:00:00+00:00"],
    )
    con.close()

    reports_dirs = [dirs["drift"], dirs["metrics"], dirs["decisions"], dirs["cutover"]]
    return reports_dirs, db_path, shadow_name


# ---------------------------------------------------------------------------
# El camino feliz: un modelo promocionado a Champion
# ---------------------------------------------------------------------------

def test_linaje_de_champion_contiene_referencias_exactas(tmp_path):
    reports_dirs, db_path, shadow_name = _construir_cadena_completa(tmp_path)

    trazado = ModelLineageTracker().trace_model_lineage("champion_model.pkl", db_path, reports_dirs)

    assert trazado["status"] == "TRACED"
    assert trazado["resolved_origin"] == shadow_name
    assert trazado["training_metrics"]["roc_auc"] == 0.81
    assert trazado["training_metrics"]["n_samples"] == 2000
    assert trazado["promotion_decision"]["candidate_model"] == shadow_name
    assert trazado["promotion_decision"]["decision"] == "PROMOTED"
    assert trazado["drift_trigger"]["trigger_reason"] == "CRITICAL_DRIFT_PSI"
    assert trazado["data_origin"]["features_affected"] == ["ingreso_mensual"]

    fuentes = {ev.get("source", "duckdb") for ev in trazado["deployment_events"]}
    assert "cutover_manifest" in fuentes
    tipos = [ev.get("event_type") for ev in trazado["deployment_events"]]
    assert "FULL_CUTOVER" in tipos


def test_linaje_por_nombre_de_shadow_model_directo(tmp_path):
    reports_dirs, db_path, shadow_name = _construir_cadena_completa(tmp_path)

    trazado = ModelLineageTracker().trace_model_lineage(shadow_name, db_path, reports_dirs)

    assert trazado["status"] == "TRACED"
    assert trazado["resolved_origin"] is None  # ya era el nombre de origen, nada que resolver
    assert trazado["training_metrics"]["roc_auc"] == 0.81
    assert trazado["promotion_decision"]["decision"] == "PROMOTED"


def test_manifiesto_guardado_coincide_exactamente_con_el_trazado(tmp_path):
    reports_dirs, db_path, shadow_name = _construir_cadena_completa(tmp_path)
    tracker = ModelLineageTracker()
    trazado = tracker.trace_model_lineage(shadow_name, db_path, reports_dirs)

    manifest_path = tracker.generate_governance_manifest(shadow_name, tmp_path / "expedientes")

    assert manifest_path.exists()
    assert manifest_path.name.startswith(f"model_lineage_{Path(shadow_name).stem}_")
    assert json.loads(manifest_path.read_text()) == trazado


# ---------------------------------------------------------------------------
# Manejo seguro: directorios y conexiones que todavia no existen
# ---------------------------------------------------------------------------

def test_base_duckdb_inexistente_no_rompe_el_trazado(tmp_path):
    reports_dirs, _, shadow_name = _construir_cadena_completa(tmp_path)
    db_path = tmp_path / "no_existe_todavia" / "otro_nivel" / "lab_lifecycle.duckdb"
    assert not db_path.parent.exists()

    trazado = ModelLineageTracker().trace_model_lineage(shadow_name, db_path, reports_dirs)

    # El resto del linaje (entrenamiento, promocion) sigue resolviendose
    # via archivos -- solo los eventos de despliegue vienen vacios.
    assert trazado["training_metrics"]["roc_auc"] == 0.81
    assert trazado["deployment_events"] == []


def test_generate_governance_manifest_crea_el_directorio_padre_inexistente(tmp_path):
    reports_dirs, db_path, shadow_name = _construir_cadena_completa(tmp_path)
    tracker = ModelLineageTracker()
    tracker.trace_model_lineage(shadow_name, db_path, reports_dirs)

    output_dir = tmp_path / "no_existe" / "otro_nivel" / "expedientes"
    assert not output_dir.exists()

    manifest_path = tracker.generate_governance_manifest(shadow_name, output_dir)

    assert output_dir.exists()
    assert manifest_path.exists()


def test_generar_manifiesto_sin_trazado_previo_falla_explicitamente(tmp_path):
    with pytest.raises(RuntimeError):
        ModelLineageTracker().generate_governance_manifest("shadow_model_x.pkl", tmp_path)


def test_json_corrupto_entre_los_reportes_se_ignora_sin_romper_el_escaneo(tmp_path):
    reports_dirs, db_path, shadow_name = _construir_cadena_completa(tmp_path)
    (reports_dirs[2] / "promotion_decision_corrupto.json").write_text("{ esto no es json valido ]")

    trazado = ModelLineageTracker().trace_model_lineage(shadow_name, db_path, reports_dirs)

    assert trazado["promotion_decision"]["candidate_model"] == shadow_name


# ---------------------------------------------------------------------------
# UNTRACED: un modelo sin ningun historial
# ---------------------------------------------------------------------------

def test_modelo_sin_historial_retorna_untraced_sin_lanzar_excepcion(tmp_path):
    carpeta_vacia = tmp_path / "vacia"
    carpeta_vacia.mkdir()

    trazado = ModelLineageTracker().trace_model_lineage(
        "shadow_model_nunca_existio.pkl", tmp_path / "no_existe.duckdb", [carpeta_vacia])

    assert trazado["status"] == "UNTRACED"
    assert trazado["training_metrics"] is None
    assert trazado["promotion_decision"] is None
    assert trazado["drift_trigger"] is None
    assert trazado["data_origin"] is None
    assert trazado["deployment_events"] == []


def test_carpeta_de_reportes_inexistente_no_rompe_el_escaneo(tmp_path):
    trazado = ModelLineageTracker().trace_model_lineage(
        "shadow_model_x.pkl", tmp_path / "no_existe.duckdb", [tmp_path / "carpeta_que_no_existe"])
    assert trazado["status"] == "UNTRACED"
