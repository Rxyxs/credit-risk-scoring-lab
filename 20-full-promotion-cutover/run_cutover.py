"""Promueve el modelo sombra validado a Champion oficial (cutover
completo), archivando el Champion anterior y dejando el evento registrado
en DuckDB.

Como el resto de las tecnicas de este pipeline mlops (12-19), se integra
por archivos en disco con las tecnicas vecinas, no por import de su
codigo: el reporte de salud viene de `19-canary-monitoring/`, el modelo
sombra activo de `16-shadow-deployment/`, y `canary_config.json` es el
mismo archivo que lee y escribe `18-canary-deployment/`. Ninguna tecnica
anterior de este portafolio produjo nunca un `champion_model.pkl` propio
(ver la nota en `16-shadow-deployment/src/shadow_engine.py`) -- esta es la
primera, y de ahi en mas ese archivo es la fuente de verdad de "que modelo
esta sirviendo".

Aborta de forma limpia (codigo de salida 0, con un log explicativo) si la
cohorte canaria no esta `HEALTHY` o si no hay un modelo sombra activo que
promover -- un cutover no se ejecuta a ciegas, y "no hubo nada que
promover" no es un error del programa.

    python run_cutover.py
    python run_cutover.py --health-report-path ruta/canary_health_X.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.cutover_manager import CutoverManager
from src.fixtures import find_latest_health_report

BASE = Path(__file__).resolve().parent

# Estas dos no estan en la lista de flags, por el mismo motivo que
# `run_health_check.py` (tecnica 19) tampoco expone todo lo que lee: son
# rutas fijas hacia el archivo que otra tecnica de este mismo pipeline ya
# deja en un lugar conocido, no un parametro de negocio de este cutover.
DEFAULT_SHADOW_MODEL_PATH = "../16-shadow-deployment/outputs/registry/active_shadow_model.pkl"
DEFAULT_CANARY_CONFIG_PATH = "../18-canary-deployment/outputs/canary_config.json"

DEFAULT_HEALTH_REPORTS_DIR = "../19-canary-monitoring/outputs/reports"
DEFAULT_MODELS_DIR = "outputs/models"
DEFAULT_DB_PATH = "outputs/lab_lifecycle.duckdb"
DEFAULT_MANIFEST_DIR = "outputs/manifests"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_cutover")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--health-report-path", default=None,
        help="Ruta a un canary_health_<timestamp>.json puntual. Por defecto, el mas "
             f"reciente en {DEFAULT_HEALTH_REPORTS_DIR} (donde lo deja "
             "19-canary-monitoring/run_health_check.py).")
    parser.add_argument(
        "--models-dir", default=DEFAULT_MODELS_DIR,
        help=f"Directorio de los modelos que produce este cutover (default: {DEFAULT_MODELS_DIR}): "
             "<models-dir>/champion/champion_model.pkl y <models-dir>/archive/.")
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Base DuckDB donde vive model_lifecycle_events (default: {DEFAULT_DB_PATH}).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    models_dir = Path(args.models_dir)
    champion_dir = models_dir / "champion"
    archive_dir = models_dir / "archive"
    shadow_model_path = Path(DEFAULT_SHADOW_MODEL_PATH)
    canary_config_path = Path(DEFAULT_CANARY_CONFIG_PATH)

    health_report_path = args.health_report_path
    if health_report_path is None:
        latest = find_latest_health_report(DEFAULT_HEALTH_REPORTS_DIR)
        health_report_path = latest if latest is not None else (
            Path(DEFAULT_HEALTH_REPORTS_DIR) / "canary_health_missing.json")

    manager = CutoverManager()

    if not manager.verify_health_before_cutover(health_report_path):
        logger.info("Cutover abortado: la cohorte canaria no esta HEALTHY (o falta el "
                     "reporte). El Champion activo no fue modificado.")
        return 0

    if not shadow_model_path.exists():
        logger.info("Cutover abortado: no hay modelo sombra activo en %s. El Champion "
                     "activo no fue modificado.", shadow_model_path)
        return 0

    resultado = manager.execute_cutover(
        shadow_model_path=shadow_model_path,
        champion_dir=champion_dir,
        archive_dir=archive_dir,
        canary_config_path=canary_config_path,
        db_path=args.db_path,
    )
    manifest_path = manager.generate_cutover_manifest(BASE / DEFAULT_MANIFEST_DIR)

    logger.info("Cutover completo. Nuevo Champion: %s (anterior: %s). Manifiesto: %s",
                resultado["new_champion"], resultado["previous_champion"], manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
