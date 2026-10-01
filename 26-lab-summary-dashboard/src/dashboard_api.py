"""Version HTTP del mismo informe que `run_lab_summary.py` escribe a
disco -- un solo endpoint, pensado para que un dashboard externo (o
`curl`) pueda pedir el estado del laboratorio sin tener que leer un
archivo JSON del filesystem."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from src.lab_summary_engine import LabSummaryEngine

# Nivel de modulo, como en `25-api-inference-service`: las pruebas los
# sobreescriben con monkeypatch antes de pedir /summary, para no depender
# de que el resto del laboratorio haya corrido de verdad.
DB_PATH = "../20-full-promotion-cutover/outputs/lab_lifecycle.duckdb"
OUTPUTS_DIR = "outputs"

app = FastAPI(title="Lab Summary Dashboard")


@app.get("/summary")
async def summary() -> dict:
    engine = LabSummaryEngine()
    resultado = engine.generate_master_summary(DB_PATH, Path(__file__).resolve().parent.parent / OUTPUTS_DIR)
    return resultado["summary"]
