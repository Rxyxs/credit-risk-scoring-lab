"""Pruebas de LabSummaryEngine y del endpoint /summary.

Cada prueba de `generate_master_summary` sobreescribe con monkeypatch
*todas* las rutas por defecto del motor (bases DuckDB, directorios de
registro, telemetria, disparadores) para quedar completamente aislada de
lo que haya o no en el resto del laboratorio en este checkout."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import dashboard_api
from src import lab_summary_engine as lse
from src.lab_summary_engine import LabSummaryEngine


def _desconectar_todas_las_bases_externas(monkeypatch, tmp_path):
    """Las cuatro bases que `generate_master_summary` consulta ademas del
    `db_path` principal, apuntadas a archivos que no existen -- el estado
    por defecto, limpio, para que cada prueba solo siembre lo que le
    importa."""
    for atributo in ("DEFAULT_FEATURE_STORE_DB", "DEFAULT_DUAL_INFERENCE_DB",
                     "DEFAULT_CANARY_DB", "DEFAULT_API_LOG_DB"):
        monkeypatch.setattr(lse, atributo, str(tmp_path / f"{atributo}.duckdb"))
    monkeypatch.setattr(lse, "DEFAULT_CHAMPION_DIR", str(tmp_path / "no_existe_champion"))
    monkeypatch.setattr(lse, "DEFAULT_SHADOW_DIRS", ())
    monkeypatch.setattr(lse, "DEFAULT_LINEAGE_DIR", str(tmp_path / "no_existe_lineage"))
    monkeypatch.setattr(lse, "DEFAULT_TELEMETRY_DIR", str(tmp_path / "no_existe_telemetry"))
    monkeypatch.setattr(lse, "DEFAULT_TRIGGER_DIR", str(tmp_path / "no_existe_trigger"))


def _base_vacia(path) -> Path:
    duckdb.connect(str(path)).close()
    return Path(path)


# ---------------------------------------------------------------------------
# audit_database_integrity
# ---------------------------------------------------------------------------

def test_audit_database_integrity_detecta_tablas_conteos_y_fechas(tmp_path):
    db_path = tmp_path / "lab_lifecycle.duckdb"
    eventos = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_type": ["FULL_CUTOVER", "RETRAINING_TRIGGERED"],
        "previous_champion": [None, None],
        "new_champion": ["champion_model.pkl", None],
        "promoted_at": ["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"],
    })
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE model_lifecycle_events AS SELECT * FROM eventos")
    con.close()

    resultado = LabSummaryEngine().audit_database_integrity(db_path)

    assert resultado["exists"] is True
    tabla = resultado["tables"]["model_lifecycle_events"]
    assert tabla["row_count"] == 2
    assert tabla["latest_event_at"] == "2026-01-02T00:00:00+00:00"


def test_audit_database_integrity_base_inexistente_no_lanza_excepcion(tmp_path):
    resultado = LabSummaryEngine().audit_database_integrity(tmp_path / "no_existe.duckdb")

    assert resultado["exists"] is False
    assert resultado["tables"] == {}


def test_audit_database_integrity_con_directorio_padre_inexistente_no_falla(tmp_path):
    db_path = tmp_path / "no_existe_todavia" / "otro_nivel" / "x.duckdb"

    resultado = LabSummaryEngine().audit_database_integrity(db_path)

    assert resultado["exists"] is False


def test_audit_database_integrity_tabla_sin_columna_de_timestamp_reconocible(tmp_path):
    db_path = tmp_path / "offline_store.duckdb"
    features = pd.DataFrame({"client_id": ["c1", "c2"], "dti": [0.1, 0.2]})
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE credit_features AS SELECT * FROM features")
    con.close()

    resultado = LabSummaryEngine().audit_database_integrity(db_path)

    assert resultado["tables"]["credit_features"]["row_count"] == 2
    assert resultado["tables"]["credit_features"]["latest_event_at"] is None


# ---------------------------------------------------------------------------
# audit_artifact_registry
# ---------------------------------------------------------------------------

def test_audit_artifact_registry_detecta_champion_candidatos_y_linaje_validos(tmp_path):
    registry_dir = tmp_path / "champion"
    registry_dir.mkdir()
    with open(registry_dir / "champion_model.pkl", "wb") as f:
        pickle.dump({"modelo": "stub"}, f)
    with open(registry_dir / "shadow_model_20260101T000000Z.pkl", "wb") as f:
        pickle.dump({"modelo": "candidato"}, f)
    (registry_dir / "model_lineage_champion_model_20260101T000000Z.json").write_text(
        json.dumps({"status": "TRACED"}))

    resultado = LabSummaryEngine().audit_artifact_registry(registry_dir)

    assert resultado["exists"] is True
    assert resultado["champion"]["valid_pickle"] is True
    assert len(resultado["shadow_candidates"]) == 1
    assert resultado["shadow_candidates"][0]["valid_pickle"] is True
    assert resultado["lineage_reports"][0]["valid_json"] is True
    assert resultado["lineage_reports"][0]["status"] == "TRACED"


def test_audit_artifact_registry_detecta_pickle_corrupto(tmp_path):
    registry_dir = tmp_path / "champion"
    registry_dir.mkdir()
    (registry_dir / "champion_model.pkl").write_bytes(b"esto no es un pickle valido")

    resultado = LabSummaryEngine().audit_artifact_registry(registry_dir)

    assert resultado["champion"]["valid_pickle"] is False


def test_audit_artifact_registry_directorio_inexistente(tmp_path):
    resultado = LabSummaryEngine().audit_artifact_registry(tmp_path / "no_existe")

    assert resultado["exists"] is False
    assert resultado["champion"] is None
    assert resultado["shadow_candidates"] == []
    assert resultado["lineage_reports"] == []


# ---------------------------------------------------------------------------
# generate_master_summary
# ---------------------------------------------------------------------------

def test_generate_master_summary_healthy_con_champion_valido(tmp_path, monkeypatch):
    db_path = tmp_path / "lab_lifecycle.duckdb"
    eventos = pd.DataFrame({
        "event_id": ["e1"], "event_type": ["FULL_CUTOVER"], "previous_champion": [None],
        "new_champion": ["champion_model.pkl"], "promoted_at": ["2026-01-01T00:00:00+00:00"],
    })
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE model_lifecycle_events AS SELECT * FROM eventos")
    con.close()

    _desconectar_todas_las_bases_externas(monkeypatch, tmp_path)
    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    with open(champion_dir / "champion_model.pkl", "wb") as f:
        pickle.dump({"modelo": "stub"}, f)
    monkeypatch.setattr(lse, "DEFAULT_CHAMPION_DIR", str(champion_dir))

    resultado = LabSummaryEngine().generate_master_summary(db_path, tmp_path / "outputs")
    resumen = resultado["summary"]

    assert resumen["lab_status"] == "HEALTHY"
    assert resumen["total_executions"] == 1
    assert resumen["champion_in_service"]["valid_pickle"] is True

    report_path = Path(resultado["report_path"])
    markdown_path = Path(resultado["markdown_path"])
    assert report_path.name.startswith("lab_summary_report_")
    assert markdown_path.name == "LAB_SUMMARY.md"
    assert json.loads(report_path.read_text()) == resumen
    assert "HEALTHY" in markdown_path.read_text()


def test_generate_master_summary_degraded_sin_champion_valido(tmp_path, monkeypatch):
    db_path = _base_vacia(tmp_path / "lab_lifecycle.duckdb")
    _desconectar_todas_las_bases_externas(monkeypatch, tmp_path)

    resultado = LabSummaryEngine().generate_master_summary(db_path, tmp_path / "outputs")

    assert resultado["summary"]["lab_status"] == "DEGRADED"
    assert resultado["summary"]["champion_in_service"] is None


def test_generate_master_summary_degraded_por_disparador_activo_sin_atender(tmp_path, monkeypatch):
    db_path = _base_vacia(tmp_path / "lab_lifecycle.duckdb")
    _desconectar_todas_las_bases_externas(monkeypatch, tmp_path)

    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    with open(champion_dir / "champion_model.pkl", "wb") as f:
        pickle.dump({"modelo": "stub"}, f)
    monkeypatch.setattr(lse, "DEFAULT_CHAMPION_DIR", str(champion_dir))

    trigger_dir = tmp_path / "triggers"
    trigger_dir.mkdir()
    (trigger_dir / "retraining_trigger_20260101T000000Z.json").write_text(json.dumps({
        "trigger_activated": True,
        "reasons": ["realized_roc_auc=0.6000 por debajo del minimo 0.7200"],
    }))
    monkeypatch.setattr(lse, "DEFAULT_TRIGGER_DIR", str(trigger_dir))

    resultado = LabSummaryEngine().generate_master_summary(db_path, tmp_path / "outputs")
    resumen = resultado["summary"]

    assert resumen["lab_status"] == "DEGRADED"
    assert resumen["retraining_trigger_status"]["trigger_activated"] is True


def test_generate_master_summary_crea_el_directorio_de_salida_inexistente(tmp_path, monkeypatch):
    db_path = _base_vacia(tmp_path / "lab_lifecycle.duckdb")
    _desconectar_todas_las_bases_externas(monkeypatch, tmp_path)

    outputs_dir = tmp_path / "no_existe" / "otro_nivel" / "outputs"
    assert not outputs_dir.exists()

    LabSummaryEngine().generate_master_summary(db_path, outputs_dir)

    assert outputs_dir.exists()
    assert (outputs_dir / "LAB_SUMMARY.md").exists()


# ---------------------------------------------------------------------------
# GET /summary (dashboard_api)
# ---------------------------------------------------------------------------

def test_endpoint_summary_responde_200_con_la_estructura_esperada(tmp_path, monkeypatch):
    db_path = _base_vacia(tmp_path / "lab_lifecycle.duckdb")
    _desconectar_todas_las_bases_externas(monkeypatch, tmp_path)
    monkeypatch.setattr(dashboard_api, "DB_PATH", str(db_path))
    monkeypatch.setattr(dashboard_api, "OUTPUTS_DIR", str(tmp_path / "outputs_api"))

    with TestClient(dashboard_api.app) as client:
        respuesta = client.get("/summary")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["lab_status"] in ("HEALTHY", "DEGRADED")
    assert "databases" in cuerpo
    assert "champion_in_service" in cuerpo
