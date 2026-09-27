"""Busca el manifiesto de reentrenamiento más reciente en `--manifest-dir`
(por defecto, la carpeta de snapshots que escribe
`12-drift-monitoring-psi-ks/src/remediation/trigger.py`) y lo ingiere en el
feature store offline con `FeatureStoreManager`.

    python run_ingestion.py
    python run_ingestion.py --manifest-dir otra/ruta/snapshots
    python run_ingestion.py --db-path outputs/offline_store.duckdb
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.ingestion import DEFAULT_DB_PATH, FeatureStoreManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_DIR = "../12-drift-monitoring-psi-ks/outputs/snapshots/"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", default=DEFAULT_MANIFEST_DIR,
        help=f"Carpeta con retrain_manifest_*.json (default: {DEFAULT_MANIFEST_DIR}).",
    )
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Ruta del archivo DuckDB del feature store (default: {DEFAULT_DB_PATH}).",
    )
    return parser.parse_args(argv)


def find_latest_manifest(manifest_dir: Path) -> Path | None:
    """El más reciente por nombre de archivo, no por mtime del filesystem --
    el timestamp en `retrain_manifest_<YYYYMMDDTHHMMSSZ>.json` ordena
    lexicográficamente igual que cronológicamente, y no depende de que el
    checkout/copia del directorio haya preservado las fechas originales."""
    if not manifest_dir.is_dir():
        return None
    manifiestos = sorted(manifest_dir.glob("retrain_manifest_*.json"))
    return manifiestos[-1] if manifiestos else None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_dir = Path(args.manifest_dir)

    manifest_path = find_latest_manifest(manifest_dir)
    if manifest_path is None:
        logger.info("No hay ningun retrain_manifest_*.json en '%s' -- nada que ingerir.", manifest_dir)
        return 0

    logger.info("Manifiesto mas reciente: %s", manifest_path)

    with FeatureStoreManager(db_path=args.db_path) as store:
        resultado = store.ingest_from_manifest(manifest_path)

    if resultado["ingested"]:
        print(
            f"ingeridas {resultado['rows']} filas en credit_features "
            f"(modelo: {resultado['model_version']}) -> {args.db_path}"
        )
    else:
        print(f"nada ingerido: {resultado['reason']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
