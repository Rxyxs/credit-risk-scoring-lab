"""Tests del exportador HTML/JSON (src.monitoring.drift) y de la CLI de
alertas de run_pipeline.py (--fail-on-red, --output-dir)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from run_pipeline import build_demo_dataframes, main
from src.monitoring.drift import export_drift_html, export_drift_json, generate_drift_report

FEATURES = ["estable", "corrida"]


@pytest.fixture
def reporte_con_drift():
    rng = np.random.default_rng(1)
    baseline = pd.DataFrame({
        "estable": rng.normal(size=2_000),
        "corrida": rng.normal(size=2_000),
    })
    scoring = pd.DataFrame({
        "estable": rng.normal(size=1_000),
        "corrida": rng.normal(loc=4.0, size=1_000),
    })
    return generate_drift_report(baseline, scoring, FEATURES)


# ---------------------------------------------------------------- export_drift_json

def test_export_drift_json_creates_a_valid_file(reporte_con_drift, tmp_path):
    ruta = export_drift_json(reporte_con_drift, tmp_path / "sub" / "drift_report.json")

    assert ruta.exists()
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    assert contenido["features"]["corrida"]["psi"] == pytest.approx(reporte_con_drift["features"]["corrida"]["psi"])
    assert contenido["features_criticas"] == ["corrida"]


def test_export_drift_json_adds_timestamp_and_overall_status(reporte_con_drift, tmp_path):
    ruta = export_drift_json(reporte_con_drift, tmp_path / "drift_report.json")
    contenido = json.loads(ruta.read_text(encoding="utf-8"))

    assert "generated_at" in contenido
    # ISO 8601 parseable, no solo una cadena cualquiera.
    from datetime import datetime
    datetime.fromisoformat(contenido["generated_at"])
    assert contenido["overall_status"] == "red"  # 'corrida' esta en features_criticas


def test_export_drift_json_reports_green_when_nothing_drifted(tmp_path):
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({"x": rng.normal(size=1_000)})
    scoring = pd.DataFrame({"x": rng.normal(size=1_000)})
    reporte = generate_drift_report(baseline, scoring, ["x"])

    ruta = export_drift_json(reporte, tmp_path / "drift_report.json")
    contenido = json.loads(ruta.read_text(encoding="utf-8"))

    assert contenido["overall_status"] == "green"


# ---------------------------------------------------------------- export_drift_html

def test_export_drift_html_creates_a_valid_file(reporte_con_drift, tmp_path):
    ruta = export_drift_html(reporte_con_drift, tmp_path / "sub" / "drift_report.html")

    assert ruta.exists()
    html = ruta.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "corrida" in html
    assert "estable" in html


def test_export_drift_html_uses_the_specified_badge_colors(reporte_con_drift, tmp_path):
    ruta = export_drift_html(reporte_con_drift, tmp_path / "drift_report.html")
    html = ruta.read_text(encoding="utf-8")

    assert "#dc3545" in html  # rojo: 'corrida' es critica
    assert "#28a745" in html  # verde: 'estable' no tiene drift
    assert "CRITICO" in html
    assert "ESTABLE" in html


def test_export_drift_html_has_no_external_dependencies():
    """Sin JS pesado ni CSS/fuentes externas -- tiene que poder abrirse sin red."""
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({"x": rng.normal(size=500)})
    scoring = pd.DataFrame({"x": rng.normal(size=500)})
    reporte = generate_drift_report(baseline, scoring, ["x"])

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        ruta = export_drift_html(reporte, Path(tmp) / "r.html")
        html = ruta.read_text(encoding="utf-8")

    assert "<script" not in html
    assert "http://" not in html and "https://" not in html


def test_html_magnitude_bar_is_wider_for_a_bigger_psi():
    """La barra de magnitud tiene que reflejar el PSI, no ser decorativa:
    una feature con PSI mas alto debe tener un ancho de barra mayor."""
    rng = np.random.default_rng(2)
    baseline = pd.DataFrame({
        "poco_drift": rng.normal(size=2_000),
        "mucho_drift": rng.normal(size=2_000),
    })
    scoring = pd.DataFrame({
        "poco_drift": rng.normal(loc=0.3, size=1_000),
        "mucho_drift": rng.normal(loc=3.0, size=1_000),
    })
    reporte = generate_drift_report(baseline, scoring, ["poco_drift", "mucho_drift"])
    assert reporte["features"]["mucho_drift"]["psi"] > reporte["features"]["poco_drift"]["psi"]

    import re
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        ruta = export_drift_html(reporte, Path(tmp) / "r.html")
        html = ruta.read_text(encoding="utf-8")

    anchos = [float(a) for a in re.findall(r"width:(\d+\.\d)%;height:100%", html)]
    assert len(anchos) == 2
    # El orden en la tabla sigue el orden de `features`: poco_drift, mucho_drift.
    assert anchos[1] > anchos[0]


# ---------------------------------------------------------------- CLI: --fail-on-red

def test_cli_fail_on_red_exits_nonzero_on_a_dataset_with_real_drift(tmp_path, caplog):
    baseline, scoring = build_demo_dataframes(shift=1.0)  # el drift severo real del README

    with caplog.at_level(logging.ERROR):
        codigo = main(["--fail-on-red", "--output-dir", str(tmp_path)],
                       baseline_df=baseline, scoring_df=scoring)

    assert codigo != 0
    assert any("Drift critico" in r.message for r in caplog.records)
    assert any("ingreso_mensual" in r.message for r in caplog.records)


def test_cli_without_fail_on_red_flag_exits_zero_even_with_drift(tmp_path):
    baseline, scoring = build_demo_dataframes(shift=1.0)

    codigo = main(["--output-dir", str(tmp_path)], baseline_df=baseline, scoring_df=scoring)

    assert codigo == 0


def test_cli_fail_on_red_exits_zero_on_stable_data(tmp_path, caplog):
    baseline, scoring = build_demo_dataframes(shift=0.0)  # sin drift en ninguna feature

    with caplog.at_level(logging.ERROR):
        codigo = main(["--fail-on-red", "--output-dir", str(tmp_path)],
                       baseline_df=baseline, scoring_df=scoring)

    assert codigo == 0
    assert not any("Drift critico" in r.message for r in caplog.records)


def test_cli_writes_html_and_json_into_the_given_output_dir(tmp_path):
    baseline, scoring = build_demo_dataframes(shift=1.0)

    main(["--output-dir", str(tmp_path)], baseline_df=baseline, scoring_df=scoring)

    assert (tmp_path / "drift_report.json").exists()
    assert (tmp_path / "drift_report.html").exists()


def test_cli_defaults_output_dir_to_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline, scoring = build_demo_dataframes(shift=0.0)

    main([], baseline_df=baseline, scoring_df=scoring)

    assert (tmp_path / "outputs" / "drift_report.json").exists()
    assert (tmp_path / "outputs" / "drift_report.html").exists()


def test_build_demo_dataframes_with_zero_shift_has_no_critical_feature():
    baseline, scoring = build_demo_dataframes(shift=0.0)
    reporte = generate_drift_report(baseline, scoring, list(baseline.columns))

    assert reporte["features_criticas"] == []
