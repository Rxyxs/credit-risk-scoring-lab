"""Tests de src.analysis.ShadowDivergenceAnalyzer y de run_analysis.py."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from run_analysis import main
from src.analysis import InsufficientLogDataError, ShadowDivergenceAnalyzer


def _logs_df(pred_champion, pred_challenger, diff_abs=None) -> pd.DataFrame:
    pred_champion = np.asarray(pred_champion, dtype=float)
    pred_challenger_arr = np.array(pred_challenger, dtype=object)
    if diff_abs is None:
        diff_abs = [
            abs(c - h) if h is not None else None
            for c, h in zip(pred_champion, pred_challenger_arr)
        ]
    return pd.DataFrame({
        "client_id": [f"CLI-{i:04d}" for i in range(len(pred_champion))],
        "pred_champion": pred_champion,
        "pred_challenger": pred_challenger_arr,
        "diff_abs": diff_abs,
        "timestamp": "2026-09-28T00:00:00+00:00",
    })


def _escribir_logs_en_duckdb(db_path, df: pd.DataFrame) -> None:
    con = duckdb.connect(str(db_path))
    con.register("_staging", df)
    con.execute("CREATE TABLE dual_inference_logs AS SELECT * FROM _staging")
    con.unregister("_staging")
    con.close()


# ------------------------------------------------------- compute_divergence_metrics

def test_mean_and_max_absolute_difference_son_numericamente_exactos():
    df = _logs_df(
        pred_champion=[0.1, 0.3, 0.5, 0.7, 0.9],
        pred_challenger=[0.2, 0.5, 0.6, 0.9, 1.0],
    )
    # diff_abs = [0.1, 0.2, 0.1, 0.2, 0.1] -- calculado a mano, no por el propio codigo bajo prueba
    analyzer = ShadowDivergenceAnalyzer()
    metricas = analyzer.compute_divergence_metrics(df)

    assert metricas["mean_absolute_difference"] == pytest.approx(0.14)
    assert metricas["max_absolute_difference"] == pytest.approx(0.2)
    assert metricas["n_predictions"] == 5


def test_pearson_correlation_perfecta_positiva_da_exactamente_1():
    champion = [1.0, 2.0, 3.0, 4.0, 5.0]
    challenger = [2.0, 4.0, 6.0, 8.0, 10.0]  # challenger = 2*champion: correlacion lineal perfecta
    df = _logs_df(champion, challenger)

    analyzer = ShadowDivergenceAnalyzer()
    metricas = analyzer.compute_divergence_metrics(df)

    assert metricas["pearson_correlation"] == pytest.approx(1.0, abs=1e-9)


def test_pearson_correlation_perfecta_negativa_da_exactamente_menos_1():
    champion = [1.0, 2.0, 3.0, 4.0, 5.0]
    challenger = [5.0, 4.0, 3.0, 2.0, 1.0]
    df = _logs_df(champion, challenger)

    analyzer = ShadowDivergenceAnalyzer()
    metricas = analyzer.compute_divergence_metrics(df)

    assert metricas["pearson_correlation"] == pytest.approx(-1.0, abs=1e-9)


def test_pearson_correlation_coincide_con_scipy_calculado_aparte():
    """Ground truth independiente: scipy.stats.pearsonr sobre los mismos
    datos, no una copia del codigo bajo prueba -- si compute_divergence_metrics
    calculara mal, este test lo detectaria igual que el de arriba."""
    rng = np.random.default_rng(42)
    champion = rng.normal(0.5, 0.1, size=40)
    challenger = champion + rng.normal(0, 0.02, size=40)
    df = _logs_df(champion, challenger)

    analyzer = ShadowDivergenceAnalyzer()
    metricas = analyzer.compute_divergence_metrics(df)

    r_esperado, _ = stats.pearsonr(champion, challenger)
    assert metricas["pearson_correlation"] == pytest.approx(r_esperado)


def test_filtra_las_filas_sin_challenger_antes_de_calcular():
    df = _logs_df(
        pred_champion=[0.1, 0.2, 0.3, 0.4],
        pred_challenger=[0.15, None, 0.35, None],
    )
    analyzer = ShadowDivergenceAnalyzer()
    metricas = analyzer.compute_divergence_metrics(df)

    assert metricas["n_predictions"] == 2  # solo las 2 filas con challenger no nulo


# --------------------------------------------------------------------- prueba KS

def test_ks_detecta_distribuciones_claramente_distintas():
    rng = np.random.default_rng(1)
    champion = rng.uniform(0.05, 0.15, size=50)   # concentrado cerca de 0.10
    challenger = rng.uniform(0.85, 0.95, size=50)  # concentrado cerca de 0.90 -- totalmente separado
    df = _logs_df(champion, challenger)

    analyzer = ShadowDivergenceAnalyzer()
    metricas = analyzer.compute_divergence_metrics(df)

    assert metricas["p_value"] < 0.05
    assert metricas["ks_statistic"] > 0.9  # separacion casi total: el estadistico KS tiene que ser alto


def test_ks_no_detecta_diferencia_entre_distribuciones_identicas():
    valores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    df = _logs_df(pred_champion=valores, pred_challenger=valores)  # mismos valores exactos

    analyzer = ShadowDivergenceAnalyzer()
    metricas = analyzer.compute_divergence_metrics(df)

    assert metricas["ks_statistic"] == pytest.approx(0.0)
    assert metricas["p_value"] == pytest.approx(1.0)


def test_menos_del_minimo_de_filas_validas_lanza_insufficient_log_data_error():
    df = _logs_df(pred_champion=[0.5], pred_challenger=[0.6])  # 1 sola fila valida
    analyzer = ShadowDivergenceAnalyzer()

    with pytest.raises(InsufficientLogDataError):
        analyzer.compute_divergence_metrics(df)


def test_ninguna_fila_con_challenger_lanza_insufficient_log_data_error():
    df = _logs_df(pred_champion=[0.5, 0.6, 0.7], pred_challenger=[None, None, None])
    analyzer = ShadowDivergenceAnalyzer()

    with pytest.raises(InsufficientLogDataError):
        analyzer.compute_divergence_metrics(df)


# ------------------------------------------------------------ generate_analysis_report

def test_generate_analysis_report_escribe_el_json_con_todos_los_campos(tmp_path):
    rng = np.random.default_rng(1)
    champion = rng.uniform(0.05, 0.15, size=30)
    challenger = rng.uniform(0.85, 0.95, size=30)
    df = _logs_df(champion, challenger)

    analyzer = ShadowDivergenceAnalyzer()
    analyzer.compute_divergence_metrics(df)
    resultado = analyzer.generate_analysis_report(tmp_path)

    ruta = Path(resultado["report_path"])
    assert ruta.exists() and ruta.name.startswith("shadow_analysis_")

    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    for campo in ("mean_absolute_difference", "max_absolute_difference", "pearson_correlation",
                  "ks_statistic", "p_value", "n_predictions", "distribution_shift_detected",
                  "summary", "generated_at"):
        assert campo in contenido
    assert contenido["distribution_shift_detected"] is True  # distribuciones bien separadas


def test_generate_analysis_report_marca_shift_false_cuando_no_hay_diferencia(tmp_path):
    valores = list(np.linspace(0.1, 0.9, 20))
    df = _logs_df(pred_champion=valores, pred_challenger=valores)

    analyzer = ShadowDivergenceAnalyzer()
    analyzer.compute_divergence_metrics(df)
    resultado = analyzer.generate_analysis_report(tmp_path)

    assert resultado["report"]["distribution_shift_detected"] is False


def test_generate_analysis_report_sin_calcular_metricas_antes_lanza_error(tmp_path):
    analyzer = ShadowDivergenceAnalyzer()
    with pytest.raises(RuntimeError, match="no hay metricas"):
        analyzer.generate_analysis_report(tmp_path)


# --------------------------------------------------------------------- fetch_logs

def test_fetch_logs_lee_la_tabla_real_de_un_archivo_duckdb(tmp_path):
    db_path = tmp_path / "logs.duckdb"
    df_original = _logs_df(pred_champion=[0.1, 0.2, 0.3], pred_challenger=[0.15, 0.25, 0.35])
    _escribir_logs_en_duckdb(db_path, df_original)

    analyzer = ShadowDivergenceAnalyzer()
    df_leido = analyzer.fetch_logs(db_path)

    assert len(df_leido) == 3
    assert list(df_leido.columns) == list(df_original.columns)


def test_fetch_logs_tabla_inexistente_lanza_insufficient_log_data_error():
    """':memory:' -- cada conexion abre una base nueva y vacia (no
    comparten estado), la forma mas directa de simular 'todavia no corrio
    la tecnica 16' sin tocar el filesystem."""
    analyzer = ShadowDivergenceAnalyzer()
    with pytest.raises(InsufficientLogDataError, match="dual_inference_logs"):
        analyzer.fetch_logs(":memory:")


def test_fetch_logs_archivo_inexistente_lanza_insufficient_log_data_error_no_crashea(tmp_path):
    analyzer = ShadowDivergenceAnalyzer()
    with pytest.raises(InsufficientLogDataError):
        analyzer.fetch_logs(tmp_path / "no" / "existe" / "logs.duckdb")


# ------------------------------------------------------------------ CLI: run_analysis.py

def test_cli_analiza_de_punta_a_punta_con_una_base_real(tmp_path):
    db_path = tmp_path / "logs.duckdb"
    rng = np.random.default_rng(3)
    champion = rng.uniform(0.05, 0.15, size=25)
    challenger = rng.uniform(0.85, 0.95, size=25)
    _escribir_logs_en_duckdb(db_path, _logs_df(champion, challenger))

    output_dir = tmp_path / "reports"
    codigo = main(["--db-path", str(db_path), "--output-dir", str(output_dir)])

    assert codigo == 0
    reportes = list(output_dir.glob("shadow_analysis_*.json"))
    assert len(reportes) == 1
    contenido = json.loads(reportes[0].read_text(encoding="utf-8"))
    assert contenido["n_predictions"] == 25


def test_cli_aborta_con_codigo_0_sin_excepcion_si_la_base_no_existe(tmp_path, caplog):
    output_dir = tmp_path / "reports"

    with caplog.at_level(logging.INFO):
        codigo = main(["--db-path", str(tmp_path / "no-existe.duckdb"), "--output-dir", str(output_dir)])

    assert codigo == 0
    assert not output_dir.exists()
    assert any("abortando sin error" in r.message for r in caplog.records)


def test_cli_aborta_con_codigo_0_sin_filas_con_challenger(tmp_path):
    db_path = tmp_path / "logs.duckdb"
    _escribir_logs_en_duckdb(db_path, _logs_df(pred_champion=[0.5, 0.6], pred_challenger=[None, None]))
    output_dir = tmp_path / "reports"

    codigo = main(["--db-path", str(db_path), "--output-dir", str(output_dir)])

    assert codigo == 0
    assert not output_dir.exists()
