"""Feature store offline sobre DuckDB: ingiere los snapshots que
`12-drift-monitoring-psi-ks/src/remediation/trigger.py` deja en disco cuando
un reporte de drift sale en rojo, y los deja en una única tabla
(`credit_features`) con upsert por `client_id`.

No hay entrenamiento ni scoring acá -- el trabajo es puramente de
persistencia: convertir "un CSV suelto con un manifiesto al lado" en una
tabla consultable, con la fila más reciente por cliente ganando siempre
(un cliente que aparece en dos snapshots sucesivos no queda duplicado, y no
importa el orden en que lleguen los ingests -- el último ONE WINS por diseño
del upsert, no por casualidad de orden de inserción).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "outputs/offline_store.duckdb"
TABLE_NAME = "credit_features"
ID_COLUMN = "client_id"
EXPECTED_SUGGESTED_ACTION = "RETRAIN_SHADOW_MODEL"


class FeatureStoreManager:
    """Conexión a la base offline (`db_path`, por defecto un archivo DuckDB
    en `outputs/`) y el único método público: `ingest_from_manifest`."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "FeatureStoreManager":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def ingest_from_manifest(self, manifest_path) -> dict:
        """Lee `manifest_path`, y si `suggested_action` pide reentrenamiento,
        carga el snapshot que referencia y lo upsertea en `credit_features`.

        Devuelve un dict con `ingested` (bool) y, si ingirió, `rows` y
        `model_version` -- lo que necesita quien llama para loguear o decidir
        el próximo paso, sin tener que releer el manifiesto de nuevo.
        """
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        suggested_action = manifest.get("suggested_action")
        if suggested_action != EXPECTED_SUGGESTED_ACTION:
            logger.info(
                "Manifiesto %s con suggested_action=%r (no %r) -- abortando, nada que ingerir.",
                manifest_path.name, suggested_action, EXPECTED_SUGGESTED_ACTION,
            )
            return {"ingested": False, "reason": "suggested_action_not_retrain"}

        if "snapshot_path" not in manifest:
            raise ValueError(f"'{manifest_path}' no tiene 'snapshot_path'")

        snapshot_path = self._resolver_snapshot(manifest_path, manifest["snapshot_path"])
        dataset = pd.read_csv(snapshot_path)

        if ID_COLUMN not in dataset.columns:
            raise ValueError(
                f"'{snapshot_path}' no tiene columna '{ID_COLUMN}' -- ingest_from_manifest "
                "necesita un identificador de cliente por fila para poder hacer upsert."
            )

        filas = self._upsert(dataset)
        model_version = manifest.get("current_model_version", "unknown")

        logger.info(
            "Ingeridas %d filas en '%s' desde %s (modelo: %s)",
            filas, TABLE_NAME, snapshot_path.name, model_version,
        )

        return {
            "ingested": True,
            "rows": filas,
            "model_version": model_version,
            "snapshot_path": str(snapshot_path),
        }

    @staticmethod
    def _resolver_snapshot(manifest_path: Path, snapshot_path_en_manifiesto: str) -> Path:
        """El manifiesto guarda la ruta del snapshot tal como la vio
        `run_pipeline.py` al generarlo -- si quien ingiere corre desde otro
        directorio de trabajo, esa ruta relativa ya no resuelve. Antes de
        rendirse, intento la misma carpeta que el propio manifiesto (los dos
        siempre se escriben juntos, ver trigger.py)."""
        ruta = Path(snapshot_path_en_manifiesto)
        if ruta.exists():
            return ruta
        candidata = manifest_path.parent / ruta.name
        if candidata.exists():
            return candidata
        raise FileNotFoundError(
            f"no encontre el snapshot de '{manifest_path.name}' ni en '{ruta}' ni en '{candidata}'"
        )

    def _upsert(self, dataset: pd.DataFrame) -> int:
        self.conn.register("_staging", dataset)
        try:
            if not self._tabla_existe():
                self.conn.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM _staging WHERE 1=0")
                self.conn.execute(f'ALTER TABLE {TABLE_NAME} ADD PRIMARY KEY ("{ID_COLUMN}")')

            columnas = list(dataset.columns)
            columnas_no_id = [c for c in columnas if c != ID_COLUMN]
            lista_columnas = ", ".join(f'"{c}"' for c in columnas)
            set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in columnas_no_id)

            self.conn.execute(
                f'INSERT INTO {TABLE_NAME} ({lista_columnas}) '
                f'SELECT {lista_columnas} FROM _staging '
                f'ON CONFLICT ("{ID_COLUMN}") DO UPDATE SET {set_clause}'
            )
        finally:
            self.conn.unregister("_staging")

        return len(dataset)

    def _tabla_existe(self) -> bool:
        resultado = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [TABLE_NAME],
        ).fetchone()
        return resultado[0] > 0
