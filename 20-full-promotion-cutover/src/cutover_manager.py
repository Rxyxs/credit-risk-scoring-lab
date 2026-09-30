"""La ultima etapa del ciclo de vida de un modelo: convertir un candidato
validado en el Champion oficial, sin dejar un estado a medio camino.

``CutoverManager`` separa deliberadamente la verificacion de la ejecucion:
``verify_health_before_cutover`` es la unica puerta de entrada, y no toca
nada en disco ni en la base de datos. Solo si esa puerta devuelve ``True``
tiene sentido llamar a ``execute_cutover`` -- que si modifica estado, y lo
hace de forma que cada paso quede archivado o registrado antes de
sobrescribirse, para que un cutover fallido a mitad de camino nunca borre
sin dejar rastro lo que habia antes.

Como el resto de las tecnicas del pipeline mlops de este laboratorio
(12-19), esta se integra por archivos en disco, no por import directo de
paquetes de otras carpetas -- ``canary_config.json`` se reescribe con el
mismo esquema exacto (``canary_percentage``, ``updated_at``) que
``CanaryRouter.set_traffic_split`` (tecnica 18) y
``CanaryHealthMonitor._force_rollback`` (tecnica 19) ya usan, para que
`18-canary-deployment/run_canary.py` siga leyendo un archivo valido
despues del cutover sin que este modulo dependa de su codigo.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

TABLA_EVENTOS = "model_lifecycle_events"


class CutoverManager:
    def __init__(self):
        self._ultimo_cutover: Optional[dict] = None

    def verify_health_before_cutover(self, health_report_path) -> bool:
        """``True`` solo si el reporte existe y su ``status`` es
        ``HEALTHY``. Cualquier otra cosa -- ``ROLLBACK_TRIGGERED``, un
        status desconocido, o el archivo faltante -- aborta la conmutacion
        antes de que toque un solo archivo."""
        path = Path(health_report_path)
        if not path.exists():
            logger.warning("No se encontro el reporte de salud canaria en %s; "
                            "cutover abortado.", path)
            return False

        reporte = json.loads(path.read_text())
        status = reporte.get("status")
        if status != "HEALTHY":
            logger.warning("Estado canario '%s' (se esperaba HEALTHY) en %s; "
                            "cutover abortado.", status, path)
            return False
        return True

    def execute_cutover(self, shadow_model_path, champion_dir, archive_dir,
                        canary_config_path, db_path) -> dict:
        """Promueve el modelo sombra a Champion.

        Orden de las operaciones, para que una falla a mitad de camino deje
        el sistema en un estado reconstruible en vez de uno ambiguo:
        1. archivar el Champion anterior (si existia) -- nunca se pierde;
        2. copiar el sombra a la posicion de Champion;
        3. bajar el trafico canario a 0%;
        4. dejar un registro inmutable del evento en DuckDB.
        """
        shadow_model_path = Path(shadow_model_path)
        champion_dir = Path(champion_dir)
        archive_dir = Path(archive_dir)
        canary_config_path = Path(canary_config_path)

        champion_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

        champion_path = champion_dir / "champion_model.pkl"
        ahora = datetime.now(timezone.utc)
        # Microsegundos, no solo segundos: dos cutover en la misma prueba (o
        # en produccion, dos conmutaciones muy seguidas) no pueden compartir
        # nombre de archivo de archivo, o el segundo pisaria al primero.
        timestamp = ahora.strftime("%Y%m%dT%H%M%S%f") + "Z"

        previous_champion = None
        archived_path = None
        if champion_path.exists():
            previous_champion = f"champion_archive_{timestamp}.pkl"
            archived_path = archive_dir / previous_champion
            shutil.copy2(champion_path, archived_path)

        shutil.copy2(shadow_model_path, champion_path)
        new_champion = "champion_model.pkl"

        # Mismo esquema que `set_traffic_split`/`trigger_rollback` (tecnica
        # 18): reescritura completa, no merge -- si el archivo venia de un
        # rollback automatico (tecnica 19), sus campos `rollback` /
        # `rollback_reason` no deben sobrevivir: el 0% ahora es porque el
        # candidato es el Champion completo, no porque algo fallo.
        canary_config_path.parent.mkdir(parents=True, exist_ok=True)
        canary_config = {
            "canary_percentage": 0,
            "updated_at": ahora.isoformat(timespec="seconds"),
        }
        canary_config_path.write_text(json.dumps(canary_config, indent=2))

        event_id = str(uuid.uuid4())
        promoted_at = ahora.isoformat()

        con = duckdb.connect(str(db_path))
        try:
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLA_EVENTOS} (
                    event_id VARCHAR,
                    event_type VARCHAR,
                    previous_champion VARCHAR,
                    new_champion VARCHAR,
                    promoted_at VARCHAR
                )
            """)
            con.execute(
                f"INSERT INTO {TABLA_EVENTOS} VALUES (?, ?, ?, ?, ?)",
                [event_id, "FULL_CUTOVER", previous_champion, new_champion, promoted_at],
            )
        finally:
            con.close()

        resultado = {
            "event_id": event_id,
            "event_type": "FULL_CUTOVER",
            "previous_champion": previous_champion,
            "new_champion": new_champion,
            "promoted_at": promoted_at,
            "champion_path": str(champion_path),
            "archived_path": str(archived_path) if archived_path else None,
            "shadow_model_path": str(shadow_model_path),
            "canary_config_path": str(canary_config_path),
            "db_path": str(db_path),
        }
        self._ultimo_cutover = resultado
        return resultado

    def generate_cutover_manifest(self, output_dir) -> Path:
        """Guarda el detalle del ultimo cutover ejecutado en un
        ``cutover_manifest_<timestamp>.json``. Requiere haber llamado antes
        a ``execute_cutover`` en esta misma instancia."""
        if self._ultimo_cutover is None:
            raise RuntimeError(
                "generate_cutover_manifest() llamado sin un cutover previo "
                "ejecutado en esta instancia de CutoverManager.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self._ultimo_cutover["promoted_at"].replace(":", "").replace("-", "")
        manifest_path = output_dir / f"cutover_manifest_{timestamp}.json"
        manifest_path.write_text(json.dumps(self._ultimo_cutover, indent=2))
        return manifest_path
