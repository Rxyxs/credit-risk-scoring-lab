"""Tests de src.remediation.trigger.DriftRemediationManager y de la
integracion --auto-retrain-trigger en run_pipeline.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_pipeline import build_demo_dataframes, main
from src.monitoring.drift import generate_drift_report
from src.remediation.trigger import (
    ACTION_NONE,
    RECOMMENDATION_SCHEDULE_COLLECTION,
    SUGGESTED_ACTION_RETRAIN,
    TRIGGER_REASON_CRITICAL_DRIFT,
    DriftRemediationManager,
)

FEATURES = ["estable", "alerta", "critica"]


def _dataset(seed: int, corrida_loc: float, alerta_loc: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "estable": rng.normal(size=800),
        "alerta": rng.normal(loc=alerta_loc, size=800),
        "critica": rng.normal(loc=corrida_loc, size=800),
    })


@pytest.fixture
def baseline():
    return _dataset(seed=1, corrida_loc=0.0, alerta_loc=0.0)


@pytest.fixture
def reporte_red(baseline):
    scoring = _dataset(seed=2, corrida_loc=4.0, alerta_loc=0.0)  # 'critica' se va lejos: PSI > 0.25
    return generate_drift_report(baseline, scoring, FEATURES), scoring


@pytest.fixture
def reporte_yellow(baseline):
    # alerta_loc=0.25 mide PSI~0.137 con este seed -- comodo adentro de la banda
    # amarilla (0.10, 0.25]; verificado con calculate_psi antes de fijar el valor,
    # no a ojo (0.6 en un intento anterior daba PSI~0.53, ya en zona roja).
    scoring = _dataset(seed=3, corrida_loc=0.0, alerta_loc=0.25)
    return generate_drift_report(baseline, scoring, FEATURES), scoring


@pytest.fixture
def reporte_green(baseline):
    scoring = _dataset(seed=4, corrida_loc=0.0, alerta_loc=0.0)  # sin drift en ninguna feature
    return generate_drift_report(baseline, scoring, FEATURES), scoring


def _fuerza_status(reporte: dict, status: str) -> dict:
    """Fuerza el semaforo sin depender de que el generador aleatorio caiga
    exactamente en el bucket deseado -- los otros tests sí usan datos reales,
    este helper es solo para los casos donde lo unico que importa es el
    status en si (limites del semaforo, no el PSI concreto)."""
    reporte = dict(reporte)
    if status == "red":
        reporte["features_criticas"] = ["x"]
        reporte["features_en_alerta"] = []
    elif status == "yellow":
        reporte["features_criticas"] = []
        reporte["features_en_alerta"] = ["x"]
    else:
        reporte["features_criticas"] = []
        reporte["features_en_alerta"] = []
    return reporte


# --------------------------------------------------------------------- RED

def test_red_crea_snapshot_csv_y_manifiesto(reporte_red, tmp_path):
    reporte, scoring = reporte_red
    assert reporte["features_criticas"] == ["critica"]  # confirma que el fixture SI genero un rojo real

    manager = DriftRemediationManager(current_model_version="v3.2.1")
    resultado = manager.evaluate_and_trigger(reporte, scoring, tmp_path / "snapshots")

    assert resultado["status"] == "red"
    assert resultado["action"] == SUGGESTED_ACTION_RETRAIN

    snapshot_path = tmp_path / "snapshots"
    archivos = list(snapshot_path.iterdir())
    csvs = [f for f in archivos if f.suffix == ".csv"]
    jsons = [f for f in archivos if f.suffix == ".json"]
    assert len(csvs) == 1 and csvs[0].name.startswith("retrain_data_")
    assert len(jsons) == 1 and jsons[0].name.startswith("retrain_manifest_")

    # el snapshot tiene los datos reales de scoring, no un archivo vacio o truncado
    snapshot_leido = pd.read_csv(csvs[0])
    assert len(snapshot_leido) == len(scoring)
    assert list(snapshot_leido.columns) == FEATURES


def test_manifiesto_contiene_exactamente_los_campos_de_metadatos_pedidos(reporte_red, tmp_path):
    reporte, scoring = reporte_red
    manager = DriftRemediationManager(current_model_version="v3.2.1")
    resultado = manager.evaluate_and_trigger(reporte, scoring, tmp_path)

    manifiesto = json.loads(Path(resultado["manifest_path"]).read_text(encoding="utf-8"))

    assert manifiesto["trigger_reason"] == TRIGGER_REASON_CRITICAL_DRIFT
    assert manifiesto["features_affected"] == ["critica"]
    assert manifiesto["current_model_version"] == "v3.2.1"
    assert manifiesto["snapshot_path"] == resultado["snapshot_path"]
    assert manifiesto["suggested_action"] == SUGGESTED_ACTION_RETRAIN
    assert "generated_at" in manifiesto


def test_red_dispara_un_log_de_advertencia_critica(reporte_red, tmp_path, caplog):
    import logging
    reporte, scoring = reporte_red
    manager = DriftRemediationManager()

    with caplog.at_level(logging.WARNING):
        manager.evaluate_and_trigger(reporte, scoring, tmp_path)

    assert any("CRITICO" in r.message for r in caplog.records)
    assert any("critica" in r.message for r in caplog.records)


# ------------------------------------------------------------------ YELLOW

def test_yellow_no_crea_ningun_archivo(reporte_yellow, tmp_path):
    reporte, scoring = reporte_yellow
    assert reporte["features_en_alerta"] == ["alerta"]
    assert reporte["features_criticas"] == []

    manager = DriftRemediationManager()
    resultado = manager.evaluate_and_trigger(reporte, scoring, tmp_path / "snapshots")

    assert resultado["status"] == "yellow"
    assert resultado["action"] == RECOMMENDATION_SCHEDULE_COLLECTION
    assert not (tmp_path / "snapshots").exists()  # no se creo ni el directorio


def test_yellow_registra_la_recomendacion_en_el_log(reporte_yellow, tmp_path, caplog):
    import logging
    reporte, scoring = reporte_yellow
    manager = DriftRemediationManager()

    with caplog.at_level(logging.WARNING):
        manager.evaluate_and_trigger(reporte, scoring, tmp_path)

    assert any("SCHEDULE_DATA_COLLECTION" in r.message for r in caplog.records)


# ------------------------------------------------------------------- GREEN

def test_green_no_crea_ningun_archivo(reporte_green, tmp_path):
    reporte, scoring = reporte_green
    assert reporte["features_criticas"] == []
    assert reporte["features_en_alerta"] == []

    manager = DriftRemediationManager()
    resultado = manager.evaluate_and_trigger(reporte, scoring, tmp_path / "snapshots")

    assert resultado["status"] == "green"
    assert resultado["action"] == ACTION_NONE
    assert not (tmp_path / "snapshots").exists()


def test_green_registra_no_action_required(reporte_green, tmp_path, caplog):
    import logging
    reporte, scoring = reporte_green
    manager = DriftRemediationManager()

    with caplog.at_level(logging.INFO):
        manager.evaluate_and_trigger(reporte, scoring, tmp_path)

    assert any("NO_ACTION_REQUIRED" in r.message for r in caplog.records)


# --------------------------------------------------------- manejo seguro

def test_output_dir_inexistente_se_crea_solo(reporte_red, tmp_path):
    reporte, scoring = reporte_red
    destino = tmp_path / "no" / "existe" / "todavia"
    assert not destino.exists()

    manager = DriftRemediationManager()
    resultado = manager.evaluate_and_trigger(reporte, scoring, destino)

    assert destino.exists()
    assert resultado["status"] == "red"


def test_current_dataset_none_lanza_error_claro():
    reporte = _fuerza_status({"features_criticas": [], "features_en_alerta": []}, "red")
    manager = DriftRemediationManager()

    with pytest.raises(ValueError, match="current_dataset"):
        manager.evaluate_and_trigger(reporte, None, "no-importa")


def test_drift_report_malformado_lanza_error_claro(reporte_red):
    _, scoring = reporte_red
    manager = DriftRemediationManager()

    with pytest.raises(ValueError, match="drift_report"):
        manager.evaluate_and_trigger({"esto": "no es un drift_report"}, scoring, "no-importa")


def test_dataset_vacio_no_crashea_y_genera_un_snapshot_vacio(tmp_path):
    """Un DataFrame corrupto/vacio (0 filas, pero con las columnas esperadas)
    no tiene por que ser un error -- el snapshot se escribe igual, vacio."""
    reporte = _fuerza_status({"features_criticas": [], "features_en_alerta": []}, "red")
    dataset_vacio = pd.DataFrame(columns=FEATURES)
    manager = DriftRemediationManager()

    resultado = manager.evaluate_and_trigger(reporte, dataset_vacio, tmp_path)

    assert resultado["status"] == "red"
    snapshot = pd.read_csv(resultado["snapshot_path"])
    assert len(snapshot) == 0


# --------------------------------------------------------- CLI: --auto-retrain-trigger

def test_cli_auto_retrain_trigger_escribe_snapshot_y_manifiesto_en_rojo(tmp_path):
    baseline, scoring = build_demo_dataframes(shift=1.0)  # drift severo real

    codigo = main(
        ["--auto-retrain-trigger", "--model-version", "v9", "--output-dir", str(tmp_path)],
        baseline_df=baseline, scoring_df=scoring,
    )

    assert codigo == 0  # el trigger de remediacion NO es el gate de CI
    snapshots = list((tmp_path / "snapshots").glob("retrain_manifest_*.json"))
    assert len(snapshots) == 1
    manifiesto = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert manifiesto["current_model_version"] == "v9"
    assert manifiesto["trigger_reason"] == TRIGGER_REASON_CRITICAL_DRIFT


def test_cli_fail_on_red_y_auto_retrain_trigger_juntos_igual_falla_el_build(tmp_path):
    """--auto-retrain-trigger no anula --fail-on-red: el manifiesto se
    escribe Y el build sigue fallando -- son dos flujos independientes."""
    baseline, scoring = build_demo_dataframes(shift=1.0)

    codigo = main(
        ["--fail-on-red", "--auto-retrain-trigger", "--output-dir", str(tmp_path)],
        baseline_df=baseline, scoring_df=scoring,
    )

    assert codigo == 1
    assert list((tmp_path / "snapshots").glob("retrain_manifest_*.json"))


def test_cli_auto_retrain_trigger_no_crea_snapshots_sin_drift(tmp_path):
    baseline, scoring = build_demo_dataframes(shift=0.0)  # sin drift

    codigo = main(
        ["--auto-retrain-trigger", "--output-dir", str(tmp_path)],
        baseline_df=baseline, scoring_df=scoring,
    )

    assert codigo == 0
    assert not (tmp_path / "snapshots").exists()
