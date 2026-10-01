"""Pruebas de integracion sobre la API real, vía `TestClient` -- nada de
mockear FastAPI: cada prueba sube la aplicacion completa (lifespan
incluido) contra modelos y configuracion sembrados en `tmp_path`, para no
depender de que las tecnicas 16/18/20 hayan corrido de verdad antes.
"""

from __future__ import annotations

import json
import pickle

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import inference_api as api


def _entrenar_y_guardar(path, sesgo: float = 0.0):
    """Un Pipeline real, entrenado sobre un DataFrame con los mismos
    nombres de columnas que usa toda la cadena (`ingreso_mensual`, `dti`,
    `antiguedad_laboral_meses`) -- no un placeholder, para que
    `predict_proba` se ejercite de verdad contra el esquema real."""
    X = pd.DataFrame({
        "ingreso_mensual": [500_000, 900_000, 700_000, 1_200_000, 600_000, 1_000_000],
        "dti": [0.1, 0.5, 0.3, 0.6, 0.2, 0.55],
        "antiguedad_laboral_meses": [12, 36, 24, 48, 18, 40],
    })
    y = [0, 1, 0, 1, 0, 1]
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])
    modelo.fit(X, y)
    if sesgo:
        modelo.named_steps["clf"].intercept_ += sesgo
    with open(path, "wb") as f:
        pickle.dump(modelo, f)
    return path


@pytest.fixture
def cliente_con_champion(tmp_path, monkeypatch):
    champion_path = _entrenar_y_guardar(tmp_path / "champion.pkl")
    monkeypatch.setattr(api, "CHAMPION_MODEL_PATH", str(champion_path))
    monkeypatch.setattr(api, "CANARY_MODEL_PATH", None)
    monkeypatch.setattr(api, "CANARY_CONFIG_PATH", None)
    monkeypatch.setattr(api, "API_LOG_DB_PATH", str(tmp_path / "outputs" / "api_inference_log.duckdb"))
    with TestClient(api.app) as client:
        yield client, tmp_path


@pytest.fixture
def cliente_sin_modelo(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CHAMPION_MODEL_PATH", str(tmp_path / "no_existe.pkl"))
    monkeypatch.setattr(api, "CANARY_MODEL_PATH", None)
    monkeypatch.setattr(api, "CANARY_CONFIG_PATH", None)
    monkeypatch.setattr(api, "API_LOG_DB_PATH", str(tmp_path / "outputs" / "api_inference_log.duckdb"))
    with TestClient(api.app) as client:
        yield client


SOLICITUD_VALIDA = {
    "client_id": "CLI-000001",
    "ingreso_mensual": 650_000.0,
    "dti": 0.35,
    "antiguedad_laboral_meses": 20.0,
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_devuelve_200_cuando_el_champion_esta_cargado(cliente_con_champion):
    client, _ = cliente_con_champion
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "OK"
    assert cuerpo["champion_loaded"] is True
    assert cuerpo["canary_loaded"] is False
    assert "timestamp" in cuerpo


def test_health_devuelve_503_cuando_no_hay_champion_cargado(cliente_sin_modelo):
    respuesta = cliente_sin_modelo.get("/health")

    assert respuesta.status_code == 503
    assert respuesta.json()["status"] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# /predict -- camino feliz
# ---------------------------------------------------------------------------

def test_predict_con_solicitud_individual_valida_devuelve_la_estructura_esperada(cliente_con_champion):
    client, _ = cliente_con_champion
    respuesta = client.post("/predict", json=SOLICITUD_VALIDA)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["batch_size"] == 1
    prediccion = cuerpo["predictions"][0]
    assert prediccion["client_id"] == "CLI-000001"
    assert 0.0 <= prediccion["predicted_probability"] <= 1.0
    assert prediccion["risk_decision"] in ("APPROVED", "REJECTED")
    assert prediccion["model_used"] == "CHAMPION"
    assert prediccion["latency_ms"] >= 0.0


def test_predict_en_lote_procesa_multiples_solicitudes(cliente_con_champion):
    client, _ = cliente_con_champion
    lote = [
        {**SOLICITUD_VALIDA, "client_id": "CLI-000001"},
        {**SOLICITUD_VALIDA, "client_id": "CLI-000002", "dti": 0.6},
        {**SOLICITUD_VALIDA, "client_id": "CLI-000003", "dti": 0.1},
    ]
    respuesta = client.post("/predict", json=lote)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["batch_size"] == 3
    assert [p["client_id"] for p in cuerpo["predictions"]] == \
        ["CLI-000001", "CLI-000002", "CLI-000003"]


def test_predict_sin_champion_cargado_devuelve_503(cliente_sin_modelo):
    respuesta = cliente_sin_modelo.post("/predict", json=SOLICITUD_VALIDA)
    assert respuesta.status_code == 503


# ---------------------------------------------------------------------------
# /predict -- entradas invalidas -> 400, no 422 ni 500
# ---------------------------------------------------------------------------

def test_predict_con_string_en_campo_numerico_devuelve_400(cliente_con_champion):
    client, _ = cliente_con_champion
    invalida = {**SOLICITUD_VALIDA, "dti": "no-es-un-numero"}

    respuesta = client.post("/predict", json=invalida)

    assert respuesta.status_code == 400
    assert "detail" in respuesta.json()


def test_predict_con_campo_obligatorio_faltante_devuelve_400(cliente_con_champion):
    client, _ = cliente_con_champion
    incompleta = {"client_id": "CLI-000001", "ingreso_mensual": 500_000.0}
    # falta 'dti' y 'antiguedad_laboral_meses'

    respuesta = client.post("/predict", json=incompleta)

    assert respuesta.status_code == 400


def test_predict_con_lote_donde_un_elemento_es_invalido_devuelve_400(cliente_con_champion):
    client, _ = cliente_con_champion
    lote = [SOLICITUD_VALIDA, {**SOLICITUD_VALIDA, "client_id": "CLI-000002", "ingreso_mensual": "mucho"}]

    respuesta = client.post("/predict", json=lote)

    assert respuesta.status_code == 400


# ---------------------------------------------------------------------------
# Enrutamiento canario
# ---------------------------------------------------------------------------

def test_predict_enruta_a_canary_cuando_el_porcentaje_es_100(tmp_path, monkeypatch):
    champion_path = _entrenar_y_guardar(tmp_path / "champion.pkl")
    canary_path = _entrenar_y_guardar(tmp_path / "canary.pkl", sesgo=0.5)
    config_path = tmp_path / "canary_config.json"
    config_path.write_text(json.dumps({"canary_percentage": 100, "updated_at": "2026-01-01T00:00:00+00:00"}))

    monkeypatch.setattr(api, "CHAMPION_MODEL_PATH", str(champion_path))
    monkeypatch.setattr(api, "CANARY_MODEL_PATH", str(canary_path))
    monkeypatch.setattr(api, "CANARY_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(api, "API_LOG_DB_PATH", str(tmp_path / "outputs" / "api_inference_log.duckdb"))

    with TestClient(api.app) as client:
        respuesta = client.post("/predict", json=SOLICITUD_VALIDA)

    assert respuesta.status_code == 200
    assert respuesta.json()["predictions"][0]["model_used"] == "CANARY"


def test_predict_cae_a_champion_si_el_canary_no_cargo_aunque_el_porcentaje_sea_alto(tmp_path, monkeypatch):
    """Degradado seguro: `canary_percentage` en 100 no implica nada si el
    archivo del modelo Canary no existe o esta corrupto -- el 100% del
    trafico sigue respondiendo con el Champion."""
    champion_path = _entrenar_y_guardar(tmp_path / "champion.pkl")
    config_path = tmp_path / "canary_config.json"
    config_path.write_text(json.dumps({"canary_percentage": 100}))

    monkeypatch.setattr(api, "CHAMPION_MODEL_PATH", str(champion_path))
    monkeypatch.setattr(api, "CANARY_MODEL_PATH", str(tmp_path / "canary_no_existe.pkl"))
    monkeypatch.setattr(api, "CANARY_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(api, "API_LOG_DB_PATH", str(tmp_path / "outputs" / "api_inference_log.duckdb"))

    with TestClient(api.app) as client:
        respuesta = client.post("/predict", json=SOLICITUD_VALIDA)

    assert respuesta.status_code == 200
    assert respuesta.json()["predictions"][0]["model_used"] == "CHAMPION"


# ---------------------------------------------------------------------------
# Regresion: directorio padre de la base DuckDB del log de inferencias
# ---------------------------------------------------------------------------

def test_directorio_del_log_duckdb_se_crea_automaticamente(tmp_path, monkeypatch):
    champion_path = _entrenar_y_guardar(tmp_path / "champion.pkl")
    db_path = tmp_path / "no_existe_todavia" / "otro_nivel" / "api_inference_log.duckdb"

    monkeypatch.setattr(api, "CHAMPION_MODEL_PATH", str(champion_path))
    monkeypatch.setattr(api, "CANARY_MODEL_PATH", None)
    monkeypatch.setattr(api, "CANARY_CONFIG_PATH", None)
    monkeypatch.setattr(api, "API_LOG_DB_PATH", str(db_path))
    assert not db_path.parent.exists()

    with TestClient(api.app) as client:
        respuesta = client.post("/predict", json=SOLICITUD_VALIDA)

    assert respuesta.status_code == 200
    assert db_path.exists()

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        filas = con.execute(f"SELECT * FROM {api.LOG_TABLE_NAME}").fetchdf()
    finally:
        con.close()
    assert len(filas) == 1
    assert filas.iloc[0]["client_id"] == "CLI-000001"
