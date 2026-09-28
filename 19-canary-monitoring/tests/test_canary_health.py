"""Tests de src.canary_health.CanaryHealthMonitor y de run_health_check.py."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from run_health_check import main
from src.canary_health import CanaryHealthMonitor, InsufficientCanaryTrafficError


def _predictions_df(canary_scores, champion_scores=None) -> pd.DataFrame:
    champion_scores = champion_scores if champion_scores is not None else [0.3] * len(canary_scores)
    filas = []
    for i, score in enumerate(canary_scores):
        filas.append({"client_id": f"CAN-{i:04d}", "assigned_model": "CANARY",
                      "prediction": score, "timestamp": "2026-09-28T00:00:00+00:00"})
    for i, score in enumerate(champion_scores):
        filas.append({"client_id": f"CHA-{i:04d}", "assigned_model": "CHAMPION",
                      "prediction": score, "timestamp": "2026-09-28T00:00:00+00:00"})
    return pd.DataFrame(filas)


def _escribir_log_en_duckdb(db_path, df: pd.DataFrame, tabla: str = "canary_routing_log") -> None:
    con = duckdb.connect(str(db_path))
    con.register("_staging", df)
    con.execute(f"CREATE TABLE {tabla} AS SELECT * FROM _staging")
    con.unregister("_staging")
    con.close()


# --------------------------------------------------------- compute_health_metrics

def test_null_rate_es_numericamente_exacto():
    df = _predictions_df(canary_scores=[0.1, 0.2, np.nan, np.inf, 0.5])  # 2 de 5 invalidas
    monitor = CanaryHealthMonitor()
    metricas = monitor.compute_health_metrics(df)

    assert metricas["null_rate"] == pytest.approx(0.4)
    assert metricas["n_canary"] == 5


def test_mean_score_diff_es_numericamente_exacto():
    df = _predictions_df(canary_scores=[0.5, 0.6, 0.7], champion_scores=[0.2, 0.2, 0.2])
    # media canary = 0.6, media champion = 0.2 -> diff = 0.4
    monitor = CanaryHealthMonitor()
    metricas = monitor.compute_health_metrics(df)

    assert metricas["mean_score_diff"] == pytest.approx(0.4)


def test_high_risk_proportion_es_numericamente_exacto():
    df = _predictions_df(canary_scores=[0.9, 0.95, 0.86, 0.2, 0.1])  # 3 de 5 > 0.85
    monitor = CanaryHealthMonitor()
    metricas = monitor.compute_health_metrics(df)

    assert metricas["high_risk_proportion"] == pytest.approx(0.6)


def test_sin_filas_champion_mean_score_diff_es_none():
    df = _predictions_df(canary_scores=[0.5, 0.6], champion_scores=[])
    monitor = CanaryHealthMonitor()
    metricas = monitor.compute_health_metrics(df)

    assert metricas["mean_score_diff"] is None


def test_predictions_df_vacio_lanza_insufficient_canary_traffic_error():
    monitor = CanaryHealthMonitor()
    with pytest.raises(InsufficientCanaryTrafficError):
        monitor.compute_health_metrics(pd.DataFrame())


def test_sin_ninguna_fila_canary_lanza_insufficient_canary_traffic_error():
    df = _predictions_df(canary_scores=[])  # solo champion
    monitor = CanaryHealthMonitor()
    with pytest.raises(InsufficientCanaryTrafficError):
        monitor.compute_health_metrics(df)


# ------------------------------------------------------------------ evaluate_and_guard

def test_cohorte_sana_da_status_healthy(tmp_path):
    df = _predictions_df(canary_scores=[0.3, 0.35, 0.4, 0.32], champion_scores=[0.3, 0.3, 0.3, 0.3])
    config_path = tmp_path / "canary_config.json"

    monitor = CanaryHealthMonitor()
    decision = monitor.evaluate_and_guard(df, config_path)

    assert decision["status"] == "HEALTHY"
    assert not config_path.exists()  # HEALTHY no toca la config canaria


def test_tasa_alta_de_nulos_dispara_rollback_y_pone_el_config_en_0(tmp_path):
    scores = [np.nan] * 20 + [0.3] * 5  # 80% invalidas, muy por encima del 1% default
    df = _predictions_df(canary_scores=scores)
    config_path = tmp_path / "canary_config.json"

    monitor = CanaryHealthMonitor()
    decision = monitor.evaluate_and_guard(df, config_path)

    assert decision["status"] == "ROLLBACK_TRIGGERED"
    assert decision["violated_threshold"] == "null_rate"

    contenido = json.loads(config_path.read_text(encoding="utf-8"))
    assert contenido["canary_percentage"] == 0
    assert contenido["rollback"] is True


def test_pico_de_alto_riesgo_dispara_rollback(tmp_path):
    scores = [0.9] * 6 + [0.2] * 4  # 60% > 0.85
    df = _predictions_df(canary_scores=scores)
    config_path = tmp_path / "canary_config.json"

    monitor = CanaryHealthMonitor()
    decision = monitor.evaluate_and_guard(df, config_path)

    assert decision["status"] == "ROLLBACK_TRIGGERED"
    assert "high_risk_proportion" in decision["violated_threshold"]
    assert json.loads(config_path.read_text(encoding="utf-8"))["canary_percentage"] == 0


def test_diferencia_grande_contra_champion_dispara_rollback(tmp_path):
    df = _predictions_df(canary_scores=[0.8, 0.85, 0.82], champion_scores=[0.1, 0.1, 0.1])  # diff = 0.72
    config_path = tmp_path / "canary_config.json"

    monitor = CanaryHealthMonitor()
    decision = monitor.evaluate_and_guard(df, config_path)

    assert decision["status"] == "ROLLBACK_TRIGGERED"
    assert "mean_score_diff" in decision["violated_threshold"]


def test_varias_violaciones_simultaneas_quedan_todas_reportadas(tmp_path):
    scores = [np.nan] * 10 + [0.95] * 10  # nulos Y alto riesgo, ambos por encima del limite
    df = _predictions_df(canary_scores=scores, champion_scores=[0.1] * 10)
    config_path = tmp_path / "canary_config.json"

    monitor = CanaryHealthMonitor()
    decision = monitor.evaluate_and_guard(df, config_path)

    assert decision["status"] == "ROLLBACK_TRIGGERED"
    assert "null_rate" in decision["violated_threshold"]
    assert "high_risk_proportion" in decision["violated_threshold"]


def test_umbrales_personalizados_se_respetan(tmp_path):
    df = _predictions_df(canary_scores=[0.3, 0.3, 0.3])  # sano con los defaults
    config_path = tmp_path / "canary_config.json"

    monitor = CanaryHealthMonitor()
    decision = monitor.evaluate_and_guard(df, config_path, max_high_risk_rate=0.0, max_score_diff=0.0)

    # con umbrales en 0, cualquier mean_score_diff > 0 dispara -- 0.3 vs 0.3 da diff=0, no dispara esa
    assert decision["status"] == "HEALTHY"


# -------------------------------------------------------------- generate_health_report

def test_generate_health_report_escribe_json_en_estado_healthy(tmp_path):
    df = _predictions_df(canary_scores=[0.3, 0.32, 0.31])
    monitor = CanaryHealthMonitor()
    monitor.evaluate_and_guard(df, tmp_path / "canary_config.json")
    resultado = monitor.generate_health_report(tmp_path / "reports")

    ruta = Path(resultado["report_path"])
    assert ruta.exists() and ruta.name.startswith("canary_health_")
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    assert contenido["status"] == "HEALTHY"
    for campo in ("null_rate", "mean_score_diff", "high_risk_proportion", "n_canary", "generated_at"):
        assert campo in contenido


def test_generate_health_report_escribe_json_en_estado_rollback_triggered(tmp_path):
    df = _predictions_df(canary_scores=[np.nan] * 10 + [0.3] * 2)
    monitor = CanaryHealthMonitor()
    monitor.evaluate_and_guard(df, tmp_path / "canary_config.json")
    resultado = monitor.generate_health_report(tmp_path / "reports")

    contenido = json.loads(Path(resultado["report_path"]).read_text(encoding="utf-8"))
    assert contenido["status"] == "ROLLBACK_TRIGGERED"
    assert contenido["violated_threshold"] == "null_rate"
    assert contenido["reason"] is not None


def test_generate_health_report_sin_evaluar_antes_lanza_error(tmp_path):
    monitor = CanaryHealthMonitor()
    with pytest.raises(RuntimeError, match="no hay evaluacion"):
        monitor.generate_health_report(tmp_path)


# -------------------------------------------------------------- fetch_recent_predictions

def test_fetch_recent_predictions_lee_la_tabla_real(tmp_path):
    db_path = tmp_path / "logs.duckdb"
    df_original = _predictions_df(canary_scores=[0.3, 0.4])
    _escribir_log_en_duckdb(db_path, df_original)

    monitor = CanaryHealthMonitor()
    df_leido = monitor.fetch_recent_predictions(db_path)

    assert len(df_leido) == len(df_original)


def test_fetch_recent_predictions_tabla_inexistente_lanza_insufficient_canary_traffic_error():
    monitor = CanaryHealthMonitor()
    with pytest.raises(InsufficientCanaryTrafficError, match="canary_routing_log"):
        monitor.fetch_recent_predictions(":memory:")


def test_fetch_recent_predictions_archivo_inexistente_no_crashea(tmp_path):
    monitor = CanaryHealthMonitor()
    with pytest.raises(InsufficientCanaryTrafficError):
        monitor.fetch_recent_predictions(tmp_path / "no" / "existe" / "logs.duckdb")


# ------------------------------------------------------------------- CLI: run_health_check.py

def test_cli_cohorte_sana_reporta_healthy(tmp_path):
    csv_path = tmp_path / "predicciones.csv"
    _predictions_df(canary_scores=[0.3, 0.32, 0.31]).to_csv(csv_path, index=False)
    config_path = tmp_path / "canary_config.json"
    output_dir = tmp_path / "reports"

    codigo = main([
        "--predictions-data", str(csv_path), "--config-path", str(config_path), "--output-dir", str(output_dir),
    ])

    assert codigo == 0
    reportes = list(output_dir.glob("canary_health_*.json"))
    assert len(reportes) == 1
    assert json.loads(reportes[0].read_text(encoding="utf-8"))["status"] == "HEALTHY"
    assert not config_path.exists()


def test_cli_cohorte_no_sana_dispara_rollback_via_cli(tmp_path):
    csv_path = tmp_path / "predicciones.csv"
    scores = [0.95] * 8 + [0.1] * 2  # 80% alto riesgo
    _predictions_df(canary_scores=scores).to_csv(csv_path, index=False)
    config_path = tmp_path / "canary_config.json"
    output_dir = tmp_path / "reports"

    codigo = main([
        "--predictions-data", str(csv_path), "--config-path", str(config_path), "--output-dir", str(output_dir),
    ])

    assert codigo == 0
    contenido_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert contenido_config["canary_percentage"] == 0

    reporte = json.loads(next(output_dir.glob("canary_health_*.json")).read_text(encoding="utf-8"))
    assert reporte["status"] == "ROLLBACK_TRIGGERED"


def test_cli_aborta_con_codigo_0_sin_excepcion_si_no_hay_datos(tmp_path, caplog):
    output_dir = tmp_path / "reports"

    with caplog.at_level(logging.INFO):
        codigo = main([
            "--predictions-data", str(tmp_path / "no-existe.csv"),
            "--config-path", str(tmp_path / "canary_config.json"),
            "--output-dir", str(output_dir),
        ])

    assert codigo == 0
    assert not output_dir.exists()
    assert any("abortando sin error" in r.message for r in caplog.records)


def test_cli_aborta_con_codigo_0_con_base_duckdb_sin_la_tabla(tmp_path):
    db_path = tmp_path / "vacia.duckdb"
    duckdb.connect(str(db_path)).close()  # crea el archivo, sin canary_routing_log
    output_dir = tmp_path / "reports"

    codigo = main([
        "--predictions-data", str(db_path),
        "--config-path", str(tmp_path / "canary_config.json"),
        "--output-dir", str(output_dir),
    ])

    assert codigo == 0
    assert not output_dir.exists()
