"""Mide si el Champion que salio de `20-full-promotion-cutover/` sigue
prediciendo bien una vez que los prestamos que curso maduran, y si la
distribucion de sus propios scores se corrio respecto del baseline de
entrenamiento.

Ninguna tecnica anterior deja un lugar donde "prediccion del Champion en
produccion" y "resultado real observado, ya madurado" convivan para el
mismo cliente -- asi que esta es la primera en definir ese esquema:
`realized_predictions` (client_id, prediction, timestamp) y
`ground_truth_labels` (client_id, target, observed_at), ambas en su propia
base DuckDB (`--db-path`), poblada por lo que sea que alimente estas
tablas en produccion (fuera del alcance de este script).

Sin al menos `--min-observations` etiquetas de verdad de campo que crucen
con una prediccion (default 50): no hay error, hay un reporte parcial que
dice "la cartera todavia no maduro" y termina con codigo 0 -- exactamente
el estado normal en los primeros dias despues de un cutover.

    python run_telemetry.py
    python run_telemetry.py --baseline-preds-path baseline_scores.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

from src.telemetry_engine import (
    MIN_OBSERVATIONS_DEFAULT, InsufficientTelemetryDataError, PostCutoverTelemetryEngine,
)

BASE = Path(__file__).resolve().parent

DEFAULT_DB_PATH = "outputs/post_cutover_telemetry.duckdb"
DEFAULT_REPORT_DIR = "outputs/reports"
PREDICTIONS_TABLE = "realized_predictions"
GROUND_TRUTH_TABLE = "ground_truth_labels"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_telemetry")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", default=DEFAULT_DB_PATH,
        help=f"Base DuckDB con '{PREDICTIONS_TABLE}' y '{GROUND_TRUTH_TABLE}' (default: {DEFAULT_DB_PATH}).")
    parser.add_argument(
        "--baseline-preds-path", default=None,
        help="CSV opcional con una columna 'prediction': el baseline de entrenamiento "
             "contra el que se mide el PSI. Sin esto, no se calcula drift.")
    parser.add_argument(
        "--min-observations", type=int, default=MIN_OBSERVATIONS_DEFAULT,
        help=f"Minimo de etiquetas de verdad de campo cruzadas para evaluar desempeno "
             f"(default: {MIN_OBSERVATIONS_DEFAULT}).")
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def _tabla_o_vacio(db_path: str, nombre: str, columnas: list[str]) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame(columns=columnas)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        existe = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [nombre],
        ).fetchone()[0] > 0
        if not existe:
            return pd.DataFrame(columns=columnas)
        return con.execute(f"SELECT * FROM {nombre}").fetchdf()
    finally:
        con.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    engine = PostCutoverTelemetryEngine()

    predictions_df = _tabla_o_vacio(args.db_path, PREDICTIONS_TABLE, ["client_id", "prediction", "timestamp"])
    ground_truth_df = _tabla_o_vacio(args.db_path, GROUND_TRUTH_TABLE, ["client_id", "target", "observed_at"])

    cruzadas = predictions_df[["client_id"]].merge(ground_truth_df[["client_id"]], on="client_id", how="inner")
    n_observado = len(cruzadas)

    if n_observado < args.min_observations:
        engine.note_insufficient_maturity(n_observado, args.min_observations)
        logger.info(
            "Solo %d etiqueta(s) de verdad de campo cruzada(s) con una prediccion "
            "(se necesitan %d) -- reporte parcial, la cartera necesita mas tiempo "
            "de maduracion.", n_observado, args.min_observations,
        )
    else:
        engine.compute_realized_performance(predictions_df, ground_truth_df)

    if args.baseline_preds_path is not None:
        if predictions_df.empty:
            logger.info("Hay --baseline-preds-path pero no hay predicciones recientes "
                        "en '%s' -- no se calcula PSI.", args.db_path)
        else:
            baseline_preds = pd.read_csv(args.baseline_preds_path)["prediction"].to_numpy()
            try:
                engine.compute_prediction_drift(baseline_preds, predictions_df["prediction"].to_numpy())
            except InsufficientTelemetryDataError as exc:
                logger.info("No se pudo calcular el PSI: %s", exc)

    report_path = engine.generate_telemetry_report(BASE / args.output_dir)
    print(f"reporte -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
