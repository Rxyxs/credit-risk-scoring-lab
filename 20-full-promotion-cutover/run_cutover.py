"""CLI: promueve el modelo sombra validado a Champion oficial (cutover
completo), archivando el Champion anterior y dejando el evento registrado
en DuckDB.

Aborta de forma limpia (codigo de salida 0, con un log explicativo) si la
cohorte canaria no esta ``HEALTHY`` o si no hay un modelo sombra activo --
un cutover no se ejecuta a ciegas, y "no hubo nada que promover" no es un
error del programa.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.cutover_manager import CutoverManager
from src.fixtures import find_latest_health_report

BASE = Path(__file__).resolve().parent
CANARY_OUTPUTS_DIR_DEFAULT = BASE / "canary_monitoring_outputs"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_cutover")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cutover completo: promueve el modelo sombra validado a Champion.")
    parser.add_argument(
        "--health-report-path", default=None,
        help="Ruta a canary_health_<timestamp>.json. Por defecto, el mas reciente en "
             f"{CANARY_OUTPUTS_DIR_DEFAULT} -- esta tecnica es autocontenida y genera "
             "ese insumo ella misma (ver src/fixtures.py), en vez de depender de una "
             "carpeta '19-canary-monitoring/' que no existe en este laboratorio.")
    parser.add_argument(
        "--models-dir", default=str(BASE / "models"),
        help="Directorio raiz de modelos: <models-dir>/champion/, /shadow/, /archive/ "
             "y <models-dir>/canary_config.json.")
    parser.add_argument(
        "--db-path", default=str(BASE / "lab_lifecycle.duckdb"),
        help="Base de datos DuckDB central del laboratorio.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    models_dir = Path(args.models_dir)
    champion_dir = models_dir / "champion"
    shadow_dir = models_dir / "shadow"
    archive_dir = models_dir / "archive"
    canary_config_path = models_dir / "canary_config.json"
    shadow_model_path = shadow_dir / "active_shadow_model.pkl"

    health_report_path = args.health_report_path
    if health_report_path is None:
        latest = find_latest_health_report(CANARY_OUTPUTS_DIR_DEFAULT)
        health_report_path = latest if latest is not None else (
            CANARY_OUTPUTS_DIR_DEFAULT / "canary_health_missing.json")

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
    manifest_path = manager.generate_cutover_manifest(BASE / "outputs" / "manifests")

    logger.info("Cutover completo. Nuevo Champion: %s (anterior: %s). Manifiesto: %s",
                resultado["new_champion"], resultado["previous_champion"], manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
