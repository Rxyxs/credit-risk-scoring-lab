import pytest
from fastapi.testclient import TestClient

try:
    from src.api import app
except FileNotFoundError:
    app = None

VALID_PAYLOAD = {
    "edad": 34, "region": "Metropolitana", "tipo_contrato": "formal",
    "renta_liquida": 750_000, "antiguedad_laboral_meses": 36,
    "n_productos_activos": 2, "deuda_total": 900_000, "dti": 1.2,
    "n_morosidad_reportes": 0,
}


@pytest.fixture(scope="module")
def client():
    if app is None:
        pytest.skip("score_engine.dll no compilada -- correr powershell -File c/build.ps1 primero")
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info_reports_both_models(client):
    body = client.get("/model-info").json()
    assert "champion" in body and "challenger" in body
    assert 0.0 <= body["champion"]["auc"] <= 1.0
    assert 0.0 <= body["challenger"]["auc"] <= 1.0


def test_score_champion(client):
    body = client.post("/score/champion", json=VALID_PAYLOAD).json()
    assert 0.0 <= body["pd_estimate"] <= 1.0
    assert body["latency_ms"] >= 0.0


def test_score_challenger(client):
    body = client.post("/score/challenger", json=VALID_PAYLOAD).json()
    assert 0.0 <= body["pd_estimate"] <= 1.0


def test_score_compare(client):
    body = client.post("/score/compare", json=VALID_PAYLOAD).json()
    assert set(body.keys()) == {"champion", "challenger", "pd_gap"}


def test_rejects_invalid_categorical(client):
    bad_payload = dict(VALID_PAYLOAD, region="No Existe")
    response = client.post("/score/champion", json=bad_payload)
    assert response.status_code == 422
