"""Estado de partida para probar el cutover, sin depender de que las
tecnicas 16/18/19 hayan corrido de verdad antes.

Estas pruebas construyen su propio Champion, su propio candidato sombra y
su propio reporte de salud canaria bajo un directorio temporal, con el
mismo esquema de archivo que usan las tecnicas reales -- ``canary_config``
usa el campo ``canary_percentage`` tal como lo escribe
``CanaryRouter.set_traffic_split`` en `18-canary-deployment/`, y
``canary_health_<timestamp>.json`` usa ``status`` tal como lo escribe
``CanaryHealthMonitor.generate_health_report`` en `19-canary-monitoring/`
-- para que `CutoverManager` se ejercite contra datos con la misma forma
que va a ver en produccion, aunque el Champion en si no exista todavia en
ningun punto anterior del pipeline (ver la nota en
`16-shadow-deployment/src/shadow_engine.py`).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Literal

EstadoCanario = Literal["HEALTHY", "ROLLBACK_TRIGGERED"]

METRICAS_POR_ESTADO = {
    "HEALTHY": {"auc_delta": 0.004, "calibration_bias_pp": 0.3, "latency_p99_ms": 42},
    "ROLLBACK_TRIGGERED": {"auc_delta": -0.021, "calibration_bias_pp": 3.8, "latency_p99_ms": 210},
}


def _modelo_de_prueba(nombre: str, version: str) -> dict:
    """Sustituto minimo de un modelo serializado: basta con que sea un
    objeto picklable con identidad propia para que archivar/copiar sea
    verificable byte a byte."""
    return {"model_name": nombre, "version": version, "coef": [0.41, -0.18, 0.07, 0.22]}


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sembrar_champion_y_sombra(models_dir: Path) -> dict:
    """Crea el Champion activo y el candidato sombra que el cutover va a
    promover, mas la configuracion canaria que debe volver a 0%."""
    models_dir = Path(models_dir)
    champion_dir = models_dir / "champion"
    shadow_dir = models_dir / "shadow"
    archive_dir = models_dir / "archive"
    for d in (champion_dir, shadow_dir, archive_dir):
        d.mkdir(parents=True, exist_ok=True)

    champion_path = champion_dir / "champion_model.pkl"
    with champion_path.open("wb") as f:
        pickle.dump(_modelo_de_prueba("scorecard_champion", "v3.2.0"), f)

    shadow_path = shadow_dir / "active_shadow_model.pkl"
    with shadow_path.open("wb") as f:
        pickle.dump(_modelo_de_prueba("scorecard_candidato", "v3.3.0-rc1"), f)

    canary_config_path = models_dir / "canary_config.json"
    canary_config_path.write_text(json.dumps(
        {"canary_percentage": 15, "updated_at": _timestamp()}, indent=2))

    return {
        "champion_dir": champion_dir,
        "shadow_dir": shadow_dir,
        "archive_dir": archive_dir,
        "champion_path": champion_path,
        "shadow_path": shadow_path,
        "canary_config_path": canary_config_path,
    }


def escribir_reporte_salud(output_dir: Path, estado: EstadoCanario = "HEALTHY") -> Path:
    """Escribe un ``canary_health_<timestamp>.json`` como lo haria la etapa
    de monitoreo canario, con el estado y las metricas que lo sustentan."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    reporte = {
        "status": estado,
        "candidate_version": "v3.3.0-rc1",
        "generated_at": timestamp,
        "metrics": METRICAS_POR_ESTADO[estado],
    }
    path = output_dir / f"canary_health_{timestamp}.json"
    path.write_text(json.dumps(reporte, indent=2))
    return path


def find_latest_health_report(directory: Path) -> Path | None:
    """El reporte ``canary_health_*.json`` mas reciente en ``directory``, o
    ``None`` si la carpeta no existe o esta vacia -- que es exactamente la
    condicion bajo la cual el cutover debe abortar."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidatos = sorted(directory.glob("canary_health_*.json"))
    return candidatos[-1] if candidatos else None
