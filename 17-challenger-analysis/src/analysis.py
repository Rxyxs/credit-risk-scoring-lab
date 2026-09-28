"""Cierra el loop de shadow deployment: 16 corre inferencia dual y la
loguea en `dual_inference_logs`, 17 mide qué tan distinto se está
comportando el Challenger frente al Champion -- no decide nada por sí solo
(esa decisión de swap sigue siendo humana), solo deja el análisis escrito
para que alguien la tome con datos en la mano.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TABLE_NAME = "dual_inference_logs"
MIN_VALID_ROWS = 2  # pearsonr exige al menos 2 puntos; menos que eso no es un analisis, es ruido
KS_ALPHA = 0.05


class InsufficientLogDataError(RuntimeError):
    """El feature store de logs no existe todavía, no tiene la tabla
    esperada, o no llega al mínimo de filas con `pred_challenger` no nulo --
    todos los motivos por los que `run_analysis.py` aborta con código 0 en
    vez de fallar."""


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ShadowDivergenceAnalyzer:
    """`fetch_logs` -> `compute_divergence_metrics` -> `generate_analysis_report`,
    en ese orden -- la última lee el estado que dejó la del medio en
    `self.metrics`, mismo patrón que las técnicas 14/15."""

    def __init__(self):
        self.metrics: dict | None = None

    def fetch_logs(self, db_path) -> pd.DataFrame:
        """Conecta de solo lectura (nunca escribe el log de inferencias) y
        devuelve la tabla `dual_inference_logs` completa, sin filtrar --
        el filtro de `pred_challenger` no nulo es trabajo de
        `compute_divergence_metrics`, no de acá.

        `read_only=True` se salta para ':memory:' -- igual que en
        `14-shadow-model-training/src/train_shadow.py`, DuckDB no permite
        abrir una base en memoria en modo solo lectura.
        """
        es_en_memoria = str(db_path) == ":memory:"
        try:
            conn = duckdb.connect(str(db_path), read_only=not es_en_memoria)
        except duckdb.Error as exc:
            raise InsufficientLogDataError(f"no pude abrir '{db_path}': {exc}") from exc

        try:
            existe = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [TABLE_NAME],
            ).fetchone()[0] > 0
            if not existe:
                raise InsufficientLogDataError(f"la tabla '{TABLE_NAME}' no existe en '{db_path}'")

            return conn.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()
        finally:
            conn.close()

    def compute_divergence_metrics(self, df: pd.DataFrame) -> dict:
        """Filtra a las filas con `pred_challenger` no nulo (las únicas
        donde de verdad hubo comparación dual) y calcula las métricas de
        divergencia sobre esas."""
        validas = df[df["pred_challenger"].notna()].copy()

        if len(validas) < MIN_VALID_ROWS:
            raise InsufficientLogDataError(
                f"solo {len(validas)} fila(s) con pred_challenger no nulo -- "
                f"hacen falta al menos {MIN_VALID_ROWS} para un analisis de divergencia"
            )

        champion = validas["pred_champion"].astype(float).to_numpy()
        challenger = validas["pred_challenger"].astype(float).to_numpy()
        diff_abs = validas["diff_abs"].astype(float).to_numpy()

        pearson_r, _ = stats.pearsonr(champion, challenger)
        ks_result = stats.ks_2samp(champion, challenger)

        self.metrics = {
            "mean_absolute_difference": float(diff_abs.mean()),
            "max_absolute_difference": float(diff_abs.max()),
            "pearson_correlation": float(pearson_r),
            "ks_statistic": float(ks_result.statistic),
            "p_value": float(ks_result.pvalue),
            "n_predictions": int(len(validas)),
        }
        return dict(self.metrics)

    def generate_analysis_report(self, output_dir) -> dict:
        if self.metrics is None:
            raise RuntimeError("no hay metricas todavia -- llama compute_divergence_metrics() primero")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        distribution_shift_detected = self.metrics["p_value"] < KS_ALPHA
        summary = self._resumen_textual(distribution_shift_detected)

        payload = {
            **self.metrics,
            "distribution_shift_detected": distribution_shift_detected,
            "summary": summary,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        report_path = output_dir / f"shadow_analysis_{_timestamp()}.json"
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Analisis de divergencia: %s -> %s", summary, report_path)
        return {"report_path": str(report_path), "report": payload}

    def _resumen_textual(self, shift_detected: bool) -> str:
        m = self.metrics
        base = (
            f"{m['n_predictions']} predicciones comparadas -- diferencia media absoluta "
            f"{m['mean_absolute_difference']:.4f} (maxima {m['max_absolute_difference']:.4f}), "
            f"correlacion de Pearson {m['pearson_correlation']:.4f}"
        )
        if shift_detected:
            return (
                base + f" -- KS detecto una diferencia de distribucion significativa "
                f"(p={m['p_value']:.4g} < {KS_ALPHA})"
            )
        return base + f" -- sin diferencia de distribucion significativa segun KS (p={m['p_value']:.4g})"
