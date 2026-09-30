"""Canary release para el modelo sombra promovido: en vez de un swap de
golpe (todo el tráfico a la vez) o el paralelo puro de la técnica 16 (dual
inference, el Champion sigue respondiendo siempre), acá una FRACCIÓN real
del tráfico se sirve con el Challenger -- y esa fracción se puede subir,
bajar, o llevar a 0 de emergencia (`trigger_rollback`) sin tocar código.

El enrutamiento es determinista por diseño: el mismo `client_id` cae
siempre en la misma cohorte para un `canary_percentage` dado (hash MD5 del
id, no un `random()` en cada request) -- necesario para que un cliente no
salte de Champion a Canary y de vuelta entre llamadas sucesivas, lo que
haría imposible atribuirle una experiencia consistente o medir el efecto
del canary sobre él.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "outputs/canary_config.json"
DEFAULT_LOG_DB_PATH = "outputs/canary_predictions.duckdb"
LOG_TABLE_NAME = "canary_routing_log"
ID_COLUMN = "client_id"
CANARY = "CANARY"
CHAMPION = "CHAMPION"


class InvalidCanaryPercentageError(ValueError):
    """`canary_percentage` fuera de [0, 100] -- nunca se acepta en silencio
    ni se recorta (clamp): un valor invalido es un error de configuracion
    real, no algo que "arreglar" adivinando la intencion."""


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _validar_porcentaje(canary_percentage: Any) -> None:
    if isinstance(canary_percentage, bool) or not isinstance(canary_percentage, (int, float)):
        raise InvalidCanaryPercentageError(f"canary_percentage debe ser numerico, recibi: {canary_percentage!r}")
    if not (0 <= canary_percentage <= 100):
        raise InvalidCanaryPercentageError(
            f"canary_percentage debe estar entre 0 y 100, recibi: {canary_percentage!r}"
        )


def _cargar_modelo(modelo: Any):
    """Acepta un modelo ya cargado (`.predict_proba`, para tests) o una
    ruta a un .pkl (el caso real de la CLI)."""
    if hasattr(modelo, "predict_proba"):
        return modelo
    with open(modelo, "rb") as f:
        return pickle.load(f)


class CanaryRouter:
    def set_traffic_split(self, canary_percentage: int, config_path=DEFAULT_CONFIG_PATH) -> dict:
        _validar_porcentaje(canary_percentage)

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = {"canary_percentage": canary_percentage, "updated_at": _timestamp()}
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        logger.info("Trafico canario actualizado a %d%% -> %s", canary_percentage, config_path)
        return config

    def determine_route(self, client_id: Any, canary_percentage: int) -> str:
        """MD5 de `client_id`, tomado como entero de base 16 y reducido
        mod 100 -- determinista y con distribucion aproximadamente uniforme
        en [0, 99] para cualquier familia razonable de ids (md5 no tiene
        sesgo conocido por posicion de los digitos hex)."""
        _validar_porcentaje(canary_percentage)
        hash_hex = hashlib.md5(str(client_id).encode()).hexdigest()
        bucket = int(hash_hex, 16) % 100
        return CANARY if bucket < canary_percentage else CHAMPION

    def route_and_predict(self, champion_model: Any, canary_model: Any, input_df: pd.DataFrame,
                           canary_percentage: int) -> pd.DataFrame:
        _validar_porcentaje(canary_percentage)
        champion = _cargar_modelo(champion_model)
        canary = _cargar_modelo(canary_model)

        rutas = input_df[ID_COLUMN].map(lambda cid: self.determine_route(cid, canary_percentage))
        X = input_df.drop(columns=[ID_COLUMN], errors="ignore")

        predicciones = pd.Series(index=input_df.index, dtype=float)
        es_canary = rutas == CANARY
        if es_canary.any():
            predicciones.loc[es_canary] = canary.predict_proba(X.loc[es_canary])[:, 1]
        if (~es_canary).any():
            predicciones.loc[~es_canary] = champion.predict_proba(X.loc[~es_canary])[:, 1]

        return pd.DataFrame({
            ID_COLUMN: input_df[ID_COLUMN].values,
            "assigned_model": rutas.values,
            "prediction": predicciones.values,
            "timestamp": _timestamp(),
        })

    def log_routing_decisions(self, predictions_df: pd.DataFrame, db_path=DEFAULT_LOG_DB_PATH) -> int:
        """Log auditable, no un estado que se upsertea -- mismo criterio que
        `log_dual_predictions` en la técnica 16: cada corrida agrega filas
        nuevas a `canary_routing_log`, aunque el mismo client_id ya haya
        sido puntuado antes. Es lo que `19-canary-monitoring/` lee despues
        para evaluar la salud de la cohorte CANARY reciente."""
        db_path = Path(db_path)
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect(str(db_path))
        try:
            con.register("_staging", predictions_df)
            existe = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [LOG_TABLE_NAME],
            ).fetchone()[0] > 0
            if not existe:
                con.execute(f"CREATE TABLE {LOG_TABLE_NAME} AS SELECT * FROM _staging WHERE 1=0")
            con.execute(f"INSERT INTO {LOG_TABLE_NAME} SELECT * FROM _staging")
        finally:
            con.unregister("_staging")
            con.close()

        return len(predictions_df)

    def trigger_rollback(self, config_path, reason: str) -> dict:
        """Emergencia: el trafico canario baja a 0% de inmediato, sin
        pasar por `set_traffic_split` -- distinto metodo a proposito, para
        que el motivo del rollback quede en el manifiesto y sea buscable
        por separado de una baja de porcentaje planificada."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "canary_percentage": 0,
            "updated_at": _timestamp(),
            "rollback": True,
            "rollback_reason": reason,
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        logger.warning("ROLLBACK canario: trafico forzado a 0%% -- motivo: %s", reason)
        return config
