"""Servicio de scoring en produccion (FastAPI) que expone ambos modelos del
framework Champion/Challenger de este repo detras de la misma API:

- **Champion** (`/score/champion`): el scorecard estadistico de R (WOE +
  puntos PDO), evaluado por el motor compilado en C via `ChampionScorer`
  -- el hot-path de baja latencia que un sistema de originacion de
  creditos real correria en produccion.
- **Challenger** (`/score/challenger`): el mejor modelo ML candidato
  (`outputs/models/best_ml_model.joblib`, seleccionado en `ml_models.py`
  por AUC en el holdout de test) -- el modelo que compite contra el
  Champion para eventualmente reemplazarlo, siguiendo la metodologia
  Champion/Challenger estandar en gestion de modelos de riesgo bancario.

Los dos modelos se cargan una sola vez al iniciar el proceso (no por
request), que es precisamente lo que hace que el endpoint del Champion sea
de baja latencia: cada request solo hace binning + un lookup en el motor C,
no I/O de disco.

Levantar con: uvicorn src.api:app --reload
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.champion_scoring import ChampionScorer
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix

BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "outputs" / "models"
REPORTS_DIR = BASE / "outputs" / "reports"

REGIONES = [
    "Metropolitana", "Valparaiso", "Biobio", "Antofagasta",
    "Araucania", "Maule", "Los Lagos", "Coquimbo",
]


class ApplicantIn(BaseModel):
    edad: int = Field(ge=18, le=100)
    region: Literal[
        "Metropolitana", "Valparaiso", "Biobio", "Antofagasta",
        "Araucania", "Maule", "Los Lagos", "Coquimbo",
    ]
    tipo_contrato: Literal["formal", "informal"]
    renta_liquida: float = Field(gt=0)
    antiguedad_laboral_meses: int = Field(ge=0)
    n_productos_activos: int = Field(ge=0)
    deuda_total: float = Field(ge=0)
    dti: float = Field(ge=0)
    n_morosidad_reportes: int = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "edad": 34,
                "region": "Metropolitana",
                "tipo_contrato": "formal",
                "renta_liquida": 750000,
                "antiguedad_laboral_meses": 36,
                "n_productos_activos": 2,
                "deuda_total": 900000,
                "dti": 1.2,
                "n_morosidad_reportes": 0,
            }
        }
    }


class ChampionOut(BaseModel):
    model: str = "champion_scorecard_r"
    score: float
    pd_estimate: float
    latency_ms: float


class ChallengerOut(BaseModel):
    model: str
    pd_estimate: float
    latency_ms: float


class CombinedOut(BaseModel):
    champion: ChampionOut
    challenger: ChallengerOut
    pd_gap: float


def _load_challenger():
    model = joblib.load(MODELS_DIR / "best_ml_model.joblib")
    feature_columns = joblib.load(MODELS_DIR / "ml_feature_columns.joblib")
    scaler_path = MODELS_DIR / "ml_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    name = json.loads((REPORTS_DIR / "ml_model_comparison.json").read_text())["best_model"]
    return model, feature_columns, scaler, name


app = FastAPI(
    title="Chile Credit Risk Scoring Engine API",
    description=(
        "Servicio de scoring de bajo latencia para el motor de riesgo de "
        "credito de consumo. Sirve el modelo Champion (scorecard R, vía el "
        "motor C compilado) y el modelo Challenger (mejor candidato ML) "
        "detras de la misma API, siguiendo el patron Champion/Challenger de "
        "gestion de modelos de riesgo bancario."
    ),
    version="1.0.0",
)

champion_scorer = ChampionScorer()
challenger_model, challenger_feature_columns, challenger_scaler, challenger_name = _load_challenger()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    scorecard_metrics = json.loads((REPORTS_DIR / "scorecard_validation_metrics.json").read_text())
    ml_report = json.loads((REPORTS_DIR / "ml_model_comparison.json").read_text())
    return {
        "champion": {
            "name": "scorecard_r_woe_logit",
            "engine": "C (score_engine.dll via ctypes)",
            "features": champion_scorer.feature_order,
            "auc": scorecard_metrics["auc"],
            "gini": scorecard_metrics["gini"],
            "ks_statistic": scorecard_metrics["ks_statistic"],
        },
        "challenger": {
            "name": challenger_name,
            "engine": "scikit-learn (in-process)",
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "auc": ml_report["metrics"][challenger_name]["test"]["auc"],
            "gini": ml_report["metrics"][challenger_name]["test"]["gini"],
            "ks_statistic": ml_report["metrics"][challenger_name]["test"]["ks_statistic"],
        },
    }


def _score_champion(applicant: ApplicantIn) -> ChampionOut:
    features = applicant.model_dump()
    start = time.perf_counter()
    result = champion_scorer.score_one(features)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return ChampionOut(score=result["score"], pd_estimate=result["pd_estimate"], latency_ms=latency_ms)


def _score_challenger(applicant: ApplicantIn) -> ChallengerOut:
    start = time.perf_counter()
    row = pd.DataFrame([applicant.model_dump()])
    X = build_feature_matrix(row).reindex(columns=challenger_feature_columns, fill_value=0)
    if challenger_scaler is not None:
        X = pd.DataFrame(challenger_scaler.transform(X), columns=X.columns)
    pd_estimate = float(challenger_model.predict_proba(X)[:, 1][0])
    latency_ms = (time.perf_counter() - start) * 1000.0
    return ChallengerOut(model=challenger_name, pd_estimate=pd_estimate, latency_ms=latency_ms)


@app.post("/score/champion", response_model=ChampionOut)
def score_champion(applicant: ApplicantIn):
    return _score_champion(applicant)


@app.post("/score/challenger", response_model=ChallengerOut)
def score_challenger(applicant: ApplicantIn):
    return _score_challenger(applicant)


@app.post("/score/compare", response_model=CombinedOut)
def score_compare(applicant: ApplicantIn):
    champion = _score_champion(applicant)
    challenger = _score_challenger(applicant)
    return CombinedOut(
        champion=champion,
        challenger=challenger,
        pd_gap=float(np.abs(champion.pd_estimate - challenger.pd_estimate)),
    )
