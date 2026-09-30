"""Entrena un "modelo sombra" (shadow model) a partir de lo que
`13-feature-store-duckdb/` dejó en `credit_features` -- la respuesta al otro
lado de la cadena de remediación por drift: 12 detecta y dispara, 13 ingiere
el snapshot, 14 reentrena en la sombra (nunca reemplaza el modelo en
producción por sí solo; eso queda para una decisión humana/otro pipeline
fuera de este repo).

Ni el pipeline ni sus tests intentan maximizar el ROC-AUC -- son datos
sintéticos de punta a punta (ver `12-drift-monitoring-psi-ks/run_pipeline.py`
y `13-feature-store-duckdb/`). Lo que importa acá es que el mecanismo
(fetch -> train -> evaluate -> save) funcione de punta a punta sobre datos
reales del feature store, no que el modelo resultante sirva para nada.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TABLE_NAME = "credit_features"
ID_COLUMN = "client_id"
# En orden de preferencia -- se usa la primera que aparezca en la tabla.
DEFAULT_TARGET_CANDIDATES = ("default_flag", "default", "target", "label")
MIN_ROWS_TO_TRAIN = 100


class InsufficientDataError(RuntimeError):
    """El feature store no existe todavía, no tiene la tabla esperada, no
    llega al mínimo de filas, o no tiene ninguna columna de target
    reconocible -- todos los motivos por los que `run_training.py` aborta
    con código 0 en vez de fallar."""


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ShadowModelTrainer:
    """`fetch_data` -> `train_and_evaluate` -> `save_model`, en ese orden --
    las dos últimas leen/escriben el estado que deja la anterior
    (`self.pipeline`, `self.metrics`), no hay que pasarse nada a mano entre
    los tres pasos salvo `X`/`y`."""

    def __init__(self, target_column: str | None = None, random_state: int = 42):
        self.target_column = target_column
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.metrics: dict | None = None

    def fetch_data(self, db_path) -> tuple[pd.DataFrame, pd.Series]:
        """Conecta de solo lectura (nunca escribe el feature store) y
        devuelve `(X, y)`: `X` sin `client_id` ni la columna de target, `y`
        la columna de target sola.

        `read_only=True` se salta para ':memory:' -- DuckDB no permite abrir
        una base en memoria en modo solo lectura ("Cannot launch in-memory
        database in read-only mode"), y de todos modos una base en memoria
        siempre arranca vacia por conexion, sin nada que proteger de escritura.
        """
        es_en_memoria = str(db_path) == ":memory:"
        try:
            conn = duckdb.connect(str(db_path), read_only=not es_en_memoria)
        except duckdb.Error as exc:
            raise InsufficientDataError(f"no pude abrir el feature store en '{db_path}': {exc}") from exc

        try:
            existe = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [TABLE_NAME],
            ).fetchone()[0] > 0
            if not existe:
                raise InsufficientDataError(f"la tabla '{TABLE_NAME}' no existe en '{db_path}'")

            df = conn.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()
        finally:
            conn.close()

        if len(df) < MIN_ROWS_TO_TRAIN:
            raise InsufficientDataError(
                f"'{TABLE_NAME}' tiene {len(df)} fila(s), menos de las {MIN_ROWS_TO_TRAIN} minimas para entrenar"
            )

        target_col = self._resolver_target(df)
        y = df[target_col]
        X = df.drop(columns=[c for c in (ID_COLUMN, target_col) if c in df.columns])

        return X, y

    def _resolver_target(self, df: pd.DataFrame) -> str:
        if self.target_column is not None:
            if self.target_column not in df.columns:
                raise InsufficientDataError(
                    f"target_column='{self.target_column}' no esta en las columnas de '{TABLE_NAME}': "
                    f"{list(df.columns)}"
                )
            return self.target_column

        for candidata in DEFAULT_TARGET_CANDIDATES:
            if candidata in df.columns:
                return candidata

        raise InsufficientDataError(
            f"ninguna columna de target reconocible ({', '.join(DEFAULT_TARGET_CANDIDATES)}) "
            f"en '{TABLE_NAME}' -- columnas disponibles: {list(df.columns)}"
        )

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """80/20 train/test, `StandardScaler` + `LogisticRegression`, ROC-AUC
        sobre el 20% de test. Guarda el pipeline entrenado y las métricas en
        `self` para que `save_model` no necesite que se los vuelvan a pasar."""
        estratificar = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=estratificar,
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=self.random_state)),
        ])
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        self.pipeline = pipeline
        self.metrics = {
            "roc_auc": float(roc_auc),
            "n_samples": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        return dict(self.metrics)

    def save_model(self, output_dir) -> dict:
        """Escribe `shadow_model_<timestamp>.pkl` y `shadow_metrics_<timestamp>.json`
        (mismo timestamp para el par, igual que el manifiesto+snapshot del
        Día 16) en `output_dir`, y devuelve las dos rutas."""
        if self.pipeline is None or self.metrics is None:
            raise RuntimeError("no hay modelo entrenado todavia -- llama train_and_evaluate() primero")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _timestamp()

        model_path = output_dir / f"shadow_model_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.pipeline, f)

        metrics_path = output_dir / f"shadow_metrics_{timestamp}.json"
        metrics_path.write_text(json.dumps(self.metrics, indent=2), encoding="utf-8")

        logger.info(
            "Modelo sombra guardado: ROC-AUC=%.4f, n=%d -> %s",
            self.metrics["roc_auc"], self.metrics["n_samples"], model_path,
        )

        return {"model_path": str(model_path), "metrics_path": str(metrics_path)}
