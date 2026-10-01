"""El unico punto de este laboratorio donde un cliente externo recibe una
respuesta en tiempo real, no un archivo que otra tecnica lee despues.
Todo lo anterior (12 a 24) se integra por archivos, en su propio tiempo;
esto tiene que responder en milisegundos y nunca caerse solo porque el
Champion todavia no se cargo.

Dos decisiones de "degradado seguro", no accidentes:
- Si no hay modelo Champion cargado, ni `/health` ni `/predict` fingen
  que todo esta bien -- ambos devuelven 503, con un cuerpo que dice por
  que, en vez de un 200 que miente o un 500 sin explicacion.
- Si hay un `canary_percentage` activo pero el modelo Canary no cargo (el
  archivo no existe, o esta corrupto), el enrutamiento no se cae: cae
  hacia el Champion para el 100% del trafico. Un Canary que no responde
  nunca debe significar que nadie recibe una prediccion.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Union

import duckdb
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ("ingreso_mensual", "dti", "antiguedad_laboral_meses")
RISK_THRESHOLD_DEFAULT = 0.5
CHAMPION = "CHAMPION"
CANARY = "CANARY"

# Nivel de modulo, no de funcion: las pruebas los sobreescriben con
# monkeypatch antes de que el lifespan cargue nada, para no depender de
# que las tecnicas 16/18/20 hayan corrido de verdad.
CHAMPION_MODEL_PATH = "../20-full-promotion-cutover/outputs/models/champion/champion_model.pkl"
CANARY_MODEL_PATH = "../16-shadow-deployment/outputs/registry/active_shadow_model.pkl"
CANARY_CONFIG_PATH = "../18-canary-deployment/outputs/canary_config.json"
API_LOG_DB_PATH = "outputs/api_inference_log.duckdb"
LOG_TABLE_NAME = "api_inference_log"


class CreditScoringRequest(BaseModel):
    client_id: str = Field(..., min_length=1)
    ingreso_mensual: float
    dti: float
    antiguedad_laboral_meses: float


class PredictionResult(BaseModel):
    client_id: str
    predicted_probability: float
    risk_decision: Literal["APPROVED", "REJECTED"]
    model_used: Literal["CHAMPION", "CANARY"]
    latency_ms: float


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
    batch_size: int


class HealthResponse(BaseModel):
    status: Literal["OK", "UNAVAILABLE"]
    timestamp: str
    champion_loaded: bool
    canary_loaded: bool
    canary_percentage: int


def _cargar_modelo(path) -> object | None:
    """`None` si el archivo no existe o no se pudo deserializar -- nunca
    lanza. Es la unica forma de que `/health` pueda reportar 503 en vez
    de que el proceso entero no levante."""
    if not path:
        return None
    ruta = Path(path)
    if not ruta.exists():
        return None
    try:
        with open(ruta, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        logger.warning("No pude cargar el modelo en '%s': %s", ruta, exc)
        return None


def _leer_canary_percentage(path) -> int:
    """`0` (sin canary) ante cualquier problema -- archivo faltante, JSON
    invalido, o el campo ausente. El router de la tecnica 18 ya trata
    `canary_percentage` como la unica fuente de verdad; este es el mismo
    criterio de lectura tolerante."""
    if not path:
        return 0
    ruta = Path(path)
    if not ruta.exists():
        return 0
    try:
        config = json.loads(ruta.read_text(encoding="utf-8"))
        return int(config.get("canary_percentage", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _determinar_ruta(client_id: str, canary_percentage: int) -> str:
    """Mismo hash MD5 determinista que `CanaryRouter.determine_route`
    (`18-canary-deployment/src/canary_router.py`): el mismo `client_id`
    cae siempre en la misma cohorte para un `canary_percentage` dado.
    Reimplementado aca, no importado -- ninguna tecnica de este
    portafolio depende del paquete `src` de otra (ver el docstring de
    `19-canary-monitoring/src/canary_health.py` sobre por que)."""
    hash_hex = hashlib.md5(str(client_id).encode()).hexdigest()
    bucket = int(hash_hex, 16) % 100
    return CANARY if bucket < canary_percentage else CHAMPION


def _log_prediccion(resultado: "PredictionResult", db_path) -> None:
    """Best-effort: un fallo al loguear nunca debe tumbar una prediccion
    que ya se calculo y ya se le va a responder al cliente."""
    try:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(db_path))
        try:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {LOG_TABLE_NAME} (
                    client_id VARCHAR,
                    predicted_probability DOUBLE,
                    risk_decision VARCHAR,
                    model_used VARCHAR,
                    latency_ms DOUBLE,
                    logged_at VARCHAR
                )
            """)
            con.execute(
                f"INSERT INTO {LOG_TABLE_NAME} VALUES (?, ?, ?, ?, ?, ?)",
                [resultado.client_id, resultado.predicted_probability, resultado.risk_decision,
                 resultado.model_used, resultado.latency_ms,
                 dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")],
            )
        finally:
            con.close()
    except duckdb.Error as exc:
        logger.warning("No pude loguear la prediccion de '%s' en DuckDB: %s", resultado.client_id, exc)


class ModelRegistry:
    def __init__(self):
        self.champion = None
        self.canary = None
        self.canary_percentage = 0
        self.loaded_at: str | None = None

    def load(self, champion_path, canary_model_path=None, canary_config_path=None) -> None:
        self.champion = _cargar_modelo(champion_path)
        self.canary = _cargar_modelo(canary_model_path) if canary_model_path else None
        self.canary_percentage = _leer_canary_percentage(canary_config_path)
        self.loaded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    @property
    def is_ready(self) -> bool:
        return self.champion is not None

    def predict_one(self, registro: CreditScoringRequest,
                     risk_threshold: float = RISK_THRESHOLD_DEFAULT) -> PredictionResult:
        inicio = time.perf_counter()

        # Degradado seguro: sin Canary cargado, el 100% del trafico va al
        # Champion sin importar que diga canary_percentage.
        ruta = _determinar_ruta(registro.client_id, self.canary_percentage) \
            if self.canary is not None else CHAMPION
        modelo = self.canary if ruta == CANARY else self.champion

        X = pd.DataFrame([{col: getattr(registro, col) for col in FEATURE_COLUMNS}])
        probabilidad = float(modelo.predict_proba(X)[:, 1][0])
        decision = "REJECTED" if probabilidad >= risk_threshold else "APPROVED"
        latencia_ms = (time.perf_counter() - inicio) * 1000.0

        return PredictionResult(
            client_id=registro.client_id,
            predicted_probability=probabilidad,
            risk_decision=decision,
            model_used=ruta,
            latency_ms=latencia_ms,
        )


registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load(CHAMPION_MODEL_PATH, CANARY_MODEL_PATH, CANARY_CONFIG_PATH)
    yield


app = FastAPI(title="Credit Scoring Inference API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def manejar_error_de_validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI devuelve 422 por defecto ante un error de validacion de
    Pydantic. El contrato de esta API es 400 Bad Request -- un dato de
    entrada invalido es un error del cliente, y 400 es el codigo que todo
    consumidor de esta API va a esperar para eso."""
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.get("/health")
async def health() -> JSONResponse:
    payload = HealthResponse(
        status="OK" if registry.is_ready else "UNAVAILABLE",
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        champion_loaded=registry.champion is not None,
        canary_loaded=registry.canary is not None,
        canary_percentage=registry.canary_percentage,
    )
    status_code = 200 if registry.is_ready else 503
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@app.post("/predict")
async def predict(payload: Union[List[CreditScoringRequest], CreditScoringRequest]) -> JSONResponse:
    if not registry.is_ready:
        return JSONResponse(
            status_code=503,
            content={"detail": "El modelo Champion no esta cargado todavia."},
        )

    registros = payload if isinstance(payload, list) else [payload]
    resultados = [registry.predict_one(r) for r in registros]
    for resultado in resultados:
        _log_prediccion(resultado, API_LOG_DB_PATH)

    respuesta = PredictResponse(predictions=resultados, batch_size=len(resultados))
    return JSONResponse(status_code=200, content=respuesta.model_dump())
