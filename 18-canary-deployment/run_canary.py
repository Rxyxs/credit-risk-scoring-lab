"""Actualiza el porcentaje canario, fuerza un rollback de emergencia, o
corre inferencia enrutada sobre `--input-data` con el split actual.

`--champion-model`/`--canary-model` no están en la lista de flags del Día
22, pero `route_and_predict` los necesita de verdad -- ningún día anterior
de este portafolio produjo un artefacto de Champion separado (ver
`16-shadow-deployment/run_shadow_serving.py`, mismo motivo ahí).

    python run_canary.py --canary-percentage 20
    python run_canary.py --rollback --reason "p95 de latencia se disparo"
    python run_canary.py --champion-model champ.pkl --canary-model cnry.pkl --input-data clientes.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.canary_router import DEFAULT_CONFIG_PATH, DEFAULT_LOG_DB_PATH, CanaryRouter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FALLBACK_CANARY_PERCENTAGE = 10


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canary-percentage", type=int, default=None,
        help=f"Nuevo porcentaje canario (0-100). Sin esto, se lee de --config-path o se usa {FALLBACK_CANARY_PERCENTAGE}%%.",
    )
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rollback", action="store_true", help="Fuerza el trafico canario a 0%% de inmediato.")
    parser.add_argument("--reason", default="rollback manual via CLI", help="Motivo del rollback (con --rollback).")
    parser.add_argument("--input-data", default=None, help="CSV con client_id + features a puntuar.")
    parser.add_argument("--champion-model", default=None, help="Ruta al .pkl del modelo Champion.")
    parser.add_argument("--canary-model", default=None, help="Ruta al .pkl del modelo Canary (challenger).")
    parser.add_argument("--db-path", default=DEFAULT_LOG_DB_PATH, help="Donde loguear las predicciones enrutadas.")
    return parser.parse_args(argv)


def _porcentaje_actual(config_path: Path, default: int) -> int:
    if not config_path.exists():
        return default
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return int(config["canary_percentage"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    router = CanaryRouter()
    config_path = Path(args.config_path)

    if args.rollback:
        resultado = router.trigger_rollback(config_path, reason=args.reason)
        print(f"ROLLBACK ejecutado: canary_percentage={resultado['canary_percentage']} -- motivo: {args.reason}")
        return 0

    if args.canary_percentage is not None:
        router.set_traffic_split(args.canary_percentage, config_path)
        canary_percentage = args.canary_percentage
    else:
        canary_percentage = _porcentaje_actual(config_path, FALLBACK_CANARY_PERCENTAGE)

    print(f"canary_percentage activo: {canary_percentage}%")

    if args.input_data is None:
        return 0

    if args.champion_model is None or args.canary_model is None:
        logger.info("--input-data dado sin --champion-model/--canary-model -- no hay con que predecir, me detengo aqui.")
        return 0

    input_df = pd.read_csv(args.input_data)
    predicciones = router.route_and_predict(args.champion_model, args.canary_model, input_df, canary_percentage)
    filas_logueadas = router.log_routing_decisions(predicciones, args.db_path)

    print(predicciones.to_string(index=False))
    conteo = predicciones["assigned_model"].value_counts().to_dict()
    print(f"\nasignacion: {conteo}")
    print(f"{filas_logueadas} predicciones logueadas -> {args.db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
