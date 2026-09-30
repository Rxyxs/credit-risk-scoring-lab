"""Tests de src.ingestion.FeatureStoreManager y de run_ingestion.py."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from run_ingestion import find_latest_manifest, main
from src.ingestion import EXPECTED_SUGGESTED_ACTION, ID_COLUMN, TABLE_NAME, FeatureStoreManager


def _escribir_snapshot(tmp_path, nombre: str, filas: list[dict]) -> Path:
    ruta = tmp_path / nombre
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


def _escribir_manifiesto(tmp_path, nombre: str, snapshot_path, suggested_action: str,
                          model_version: str = "v1") -> Path:
    ruta = tmp_path / nombre
    ruta.write_text(json.dumps({
        "trigger_reason": "CRITICAL_DRIFT_PSI",
        "features_affected": ["ingreso_mensual"],
        "current_model_version": model_version,
        "snapshot_path": str(snapshot_path),
        "suggested_action": suggested_action,
        "generated_at": "2026-09-27T00:00:00+00:00",
    }, indent=2), encoding="utf-8")
    return ruta


# ------------------------------------------------------- creación de tabla

def test_ingest_crea_la_tabla_credit_features_en_memoria(tmp_path):
    snapshot = _escribir_snapshot(tmp_path, "snap.csv", [
        {"client_id": "C1", "ingreso_mensual": 1000.0, "dti": 0.2},
        {"client_id": "C2", "ingreso_mensual": 2000.0, "dti": 0.3},
    ])
    manifiesto = _escribir_manifiesto(tmp_path, "m.json", snapshot, EXPECTED_SUGGESTED_ACTION)

    with FeatureStoreManager(db_path=":memory:") as store:
        resultado = store.ingest_from_manifest(manifiesto)

        assert resultado["ingested"] is True
        assert resultado["rows"] == 2

        tablas = store.conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
        assert (TABLE_NAME,) in tablas

        filas = store.conn.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY client_id").fetchall()
        assert filas == [("C1", 1000.0, 0.2), ("C2", 2000.0, 0.3)]


def test_ingest_declara_client_id_como_primary_key(tmp_path):
    snapshot = _escribir_snapshot(tmp_path, "snap.csv", [{"client_id": "C1", "x": 1.0}])
    manifiesto = _escribir_manifiesto(tmp_path, "m.json", snapshot, EXPECTED_SUGGESTED_ACTION)

    with FeatureStoreManager(db_path=":memory:") as store:
        store.ingest_from_manifest(manifiesto)
        columnas = store.conn.execute(f"DESCRIBE {TABLE_NAME}").fetchall()
        fila_id = next(c for c in columnas if c[0] == ID_COLUMN)
        assert fila_id[3] == "PRI"  # columna 'key' de DESCRIBE: PRI = primary key


# --------------------------------------------------------- NO_ACTION_REQUIRED

def test_no_action_required_no_altera_la_base(tmp_path):
    snapshot = _escribir_snapshot(tmp_path, "snap.csv", [{"client_id": "C1", "x": 1.0}])
    manifiesto = _escribir_manifiesto(tmp_path, "m.json", snapshot, "NO_ACTION_REQUIRED")

    with FeatureStoreManager(db_path=":memory:") as store:
        resultado = store.ingest_from_manifest(manifiesto)

        assert resultado["ingested"] is False
        assert resultado["reason"] == "suggested_action_not_retrain"

        tablas = store.conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
        assert (TABLE_NAME,) not in tablas  # ni siquiera se creo la tabla


def test_schedule_data_collection_tampoco_ingiere(tmp_path):
    snapshot = _escribir_snapshot(tmp_path, "snap.csv", [{"client_id": "C1", "x": 1.0}])
    manifiesto = _escribir_manifiesto(tmp_path, "m.json", snapshot, "SCHEDULE_DATA_COLLECTION")

    with FeatureStoreManager(db_path=":memory:") as store:
        resultado = store.ingest_from_manifest(manifiesto)
        assert resultado["ingested"] is False


# ------------------------------------------------------------------ upsert

def test_upsert_suma_correctamente_filas_nuevas_y_actualiza_las_repetidas(tmp_path):
    snapshot_1 = _escribir_snapshot(tmp_path, "snap1.csv", [
        {"client_id": "C1", "ingreso_mensual": 1000.0},
        {"client_id": "C2", "ingreso_mensual": 2000.0},
    ])
    manifiesto_1 = _escribir_manifiesto(tmp_path, "m1.json", snapshot_1, EXPECTED_SUGGESTED_ACTION)

    snapshot_2 = _escribir_snapshot(tmp_path, "snap2.csv", [
        {"client_id": "C2", "ingreso_mensual": 9999.0},  # actualiza a C2
        {"client_id": "C3", "ingreso_mensual": 3000.0},  # cliente nuevo
    ])
    manifiesto_2 = _escribir_manifiesto(tmp_path, "m2.json", snapshot_2, EXPECTED_SUGGESTED_ACTION)

    with FeatureStoreManager(db_path=":memory:") as store:
        r1 = store.ingest_from_manifest(manifiesto_1)
        r2 = store.ingest_from_manifest(manifiesto_2)

        assert r1["rows"] == 2  # el conteo de la ingesta 1 es el tamaño de SU snapshot
        assert r2["rows"] == 2  # idem para la ingesta 2, aunque una de las dos sea un update

        total_filas = store.conn.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0]
        assert total_filas == 3  # C1, C2, C3 -- no 4: C2 se actualizo, no se duplico

        ingreso_c2 = store.conn.execute(
            f"SELECT ingreso_mensual FROM {TABLE_NAME} WHERE client_id = 'C2'"
        ).fetchone()[0]
        assert ingreso_c2 == 9999.0  # gano el valor mas reciente


# --------------------------------------------------------------------- errores

def test_manifiesto_sin_snapshot_path_lanza_error(tmp_path):
    ruta = tmp_path / "m.json"
    ruta.write_text(json.dumps({"suggested_action": EXPECTED_SUGGESTED_ACTION}), encoding="utf-8")

    with FeatureStoreManager(db_path=":memory:") as store:
        with pytest.raises(ValueError, match="snapshot_path"):
            store.ingest_from_manifest(ruta)


def test_snapshot_sin_client_id_lanza_error_claro(tmp_path):
    snapshot = _escribir_snapshot(tmp_path, "snap.csv", [{"ingreso_mensual": 1000.0}])
    manifiesto = _escribir_manifiesto(tmp_path, "m.json", snapshot, EXPECTED_SUGGESTED_ACTION)

    with FeatureStoreManager(db_path=":memory:") as store:
        with pytest.raises(ValueError, match="client_id"):
            store.ingest_from_manifest(manifiesto)


def test_resuelve_snapshot_relativo_a_la_carpeta_del_manifiesto(tmp_path):
    """El manifiesto guarda la ruta tal como la vio run_pipeline.py -- si
    ese path ya no existe desde el cwd actual, cae a buscar el archivo por
    nombre en la misma carpeta que el manifiesto."""
    snapshot = _escribir_snapshot(tmp_path, "snap.csv", [{"client_id": "C1", "x": 1.0}])
    manifiesto = tmp_path / "m.json"
    manifiesto.write_text(json.dumps({
        "suggested_action": EXPECTED_SUGGESTED_ACTION,
        "snapshot_path": "una/ruta/que/no/existe/snap.csv",  # ruta rota a proposito
        "current_model_version": "v1",
    }), encoding="utf-8")

    with FeatureStoreManager(db_path=":memory:") as store:
        resultado = store.ingest_from_manifest(manifiesto)
        assert resultado["ingested"] is True
        assert resultado["rows"] == 1


# -------------------------------------------------------------------------- log

def test_ingest_registra_filas_y_version_de_modelo_en_el_log(tmp_path, caplog):
    import logging
    snapshot = _escribir_snapshot(tmp_path, "snap.csv", [
        {"client_id": "C1", "x": 1.0}, {"client_id": "C2", "x": 2.0},
    ])
    manifiesto = _escribir_manifiesto(tmp_path, "m.json", snapshot, EXPECTED_SUGGESTED_ACTION,
                                       model_version="scorecard-v7")

    with caplog.at_level(logging.INFO):
        with FeatureStoreManager(db_path=":memory:") as store:
            store.ingest_from_manifest(manifiesto)

    assert any("2" in r.message and "scorecard-v7" in r.message for r in caplog.records)


# ------------------------------------------------------------------ run_ingestion.py

def test_find_latest_manifest_elige_el_de_timestamp_mas_alto(tmp_path):
    (tmp_path / "retrain_manifest_20260101T000000Z.json").write_text("{}")
    mas_reciente = tmp_path / "retrain_manifest_20260927T235959Z.json"
    mas_reciente.write_text("{}")
    (tmp_path / "retrain_manifest_20260615T120000Z.json").write_text("{}")

    assert find_latest_manifest(tmp_path) == mas_reciente


def test_find_latest_manifest_devuelve_none_si_no_hay_nada(tmp_path):
    assert find_latest_manifest(tmp_path / "no-existe") is None
    assert find_latest_manifest(tmp_path) is None


def test_cli_ingiere_el_manifiesto_mas_reciente_del_manifest_dir(tmp_path):
    snapshot = _escribir_snapshot(tmp_path, "retrain_data_20260927T000000Z.csv", [
        {"client_id": "C1", "ingreso_mensual": 1000.0},
    ])
    _escribir_manifiesto(tmp_path, "retrain_manifest_20260927T000000Z.json", snapshot,
                          EXPECTED_SUGGESTED_ACTION, model_version="v42")

    db_path = tmp_path / "store.duckdb"
    codigo = main(["--manifest-dir", str(tmp_path), "--db-path", str(db_path)])

    assert codigo == 0
    assert db_path.exists()
    con = duckdb.connect(str(db_path))
    assert con.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0] == 1
    con.close()


def test_cli_no_falla_si_no_hay_manifiestos(tmp_path):
    codigo = main(["--manifest-dir", str(tmp_path / "vacio"), "--db-path", str(tmp_path / "store.duckdb")])
    assert codigo == 0
