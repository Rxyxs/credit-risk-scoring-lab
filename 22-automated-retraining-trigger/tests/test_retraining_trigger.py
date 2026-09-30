"""Pruebas de RetrainingTriggerManager.

Cada prueba escribe su propio reporte de telemetria sintetico con la misma
forma exacta que produce `21-post-cutover-telemetry` (`realized_performance`
+ `prediction_drift`), para no depender de que esa tecnica haya corrido
antes."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from src.retraining_trigger import RetrainingTriggerManager, TelemetryReportError


def _reporte_telemetria(tmp_path, *, status="EVALUATED", realized_roc_auc=0.78,
                        psi=0.05, con_drift=True, nombre="post_cutover_telemetry_test.json"):
    performance = {"status": status}
    if status == "EVALUATED":
        performance.update({
            "realized_roc_auc": realized_roc_auc,
            "brier_score": 0.18,
            "log_loss": 0.45,
            "observed_default_rate": 0.40,
            "predicted_default_rate": 0.42,
            "n_evaluated": 80,
        })
    else:
        performance.update({
            "n_observed": 9, "min_required": 50,
            "reason": "solo 9 etiquetas observadas, se necesitan al menos 50",
        })

    drift = None
    if con_drift:
        drift = {
            "psi": psi, "num_buckets": 10, "severity": "no_shift" if psi < 0.10 else "significant_shift",
            "n_baseline": 1000, "n_recent": 120,
        }

    payload = {
        "realized_performance": performance,
        "prediction_drift": drift,
        "summary": "reporte de prueba",
        "generated_at": "2026-01-01T00:00:00+00:00",
    }
    path = tmp_path / nombre
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------
# evaluate_trigger_conditions
# ---------------------------------------------------------------------------

def test_telemetria_sana_no_dispara_reentrenamiento(tmp_path):
    reporte = _reporte_telemetria(tmp_path, realized_roc_auc=0.78, psi=0.05)

    resultado = RetrainingTriggerManager().evaluate_trigger_conditions(reporte)

    assert resultado["trigger_activated"] is False
    assert resultado["reasons"] == []
    assert resultado["telemetry_status"] == "EVALUATED"


def test_auc_realizado_bajo_dispara_reentrenamiento(tmp_path):
    reporte = _reporte_telemetria(tmp_path, realized_roc_auc=0.68, psi=0.05)

    resultado = RetrainingTriggerManager().evaluate_trigger_conditions(reporte, min_realized_auc=0.72)

    assert resultado["trigger_activated"] is True
    assert len(resultado["reasons"]) == 1
    assert "realized_roc_auc" in resultado["reasons"][0]
    assert "0.6800" in resultado["reasons"][0]


def test_psi_alto_dispara_reentrenamiento(tmp_path):
    reporte = _reporte_telemetria(tmp_path, realized_roc_auc=0.78, psi=0.28)

    resultado = RetrainingTriggerManager().evaluate_trigger_conditions(reporte, max_psi=0.20)

    assert resultado["trigger_activated"] is True
    assert len(resultado["reasons"]) == 1
    assert "psi" in resultado["reasons"][0]
    assert "0.2800" in resultado["reasons"][0]


def test_ambas_condiciones_mal_acumulan_dos_motivos(tmp_path):
    reporte = _reporte_telemetria(tmp_path, realized_roc_auc=0.68, psi=0.28)

    resultado = RetrainingTriggerManager().evaluate_trigger_conditions(reporte)

    assert resultado["trigger_activated"] is True
    assert len(resultado["reasons"]) == 2


def test_auc_no_definida_no_dispara_por_auc_aunque_este_evaluado(tmp_path):
    # status EVALUATED pero realized_roc_auc == None (una sola clase en el
    # lote, tal como lo deja 21-post-cutover-telemetry en ese caso).
    reporte = tmp_path / "post_cutover_telemetry_una_clase.json"
    payload = {
        "realized_performance": {
            "status": "EVALUATED", "realized_roc_auc": None,
            "auc_undefined_reason": "solo una clase observada",
            "brier_score": 0.05, "log_loss": 0.2,
            "observed_default_rate": 0.0, "predicted_default_rate": 0.03, "n_evaluated": 60,
        },
        "prediction_drift": {"psi": 0.05, "num_buckets": 10, "severity": "no_shift",
                              "n_baseline": 1000, "n_recent": 60},
        "summary": "x", "generated_at": "2026-01-01T00:00:00+00:00",
    }
    reporte.write_text(json.dumps(payload))

    resultado = RetrainingTriggerManager().evaluate_trigger_conditions(reporte)

    assert resultado["trigger_activated"] is False
    assert resultado["realized_roc_auc"] is None


def test_reporte_parcial_por_maduracion_insuficiente_no_dispara_falsos_positivos(tmp_path):
    reporte = _reporte_telemetria(tmp_path, status="INSUFFICIENT_MATURITY", con_drift=False)

    resultado = RetrainingTriggerManager().evaluate_trigger_conditions(reporte)

    assert resultado["trigger_activated"] is False
    assert resultado["reasons"] == []
    assert resultado["telemetry_status"] == "INSUFFICIENT_MATURITY"
    assert resultado["realized_roc_auc"] is None


def test_reporte_parcial_con_drift_alto_igual_dispara_por_psi(tmp_path):
    # Decision de diseño deliberada: el PSI no necesita verdad de campo
    # madurada, asi que un reporte INSUFFICIENT_MATURITY con un PSI real y
    # alto debe poder disparar -- solo la via del AUC queda bloqueada por
    # falta de maduracion, no la del drift.
    reporte = _reporte_telemetria(tmp_path, status="INSUFFICIENT_MATURITY", con_drift=True, psi=0.35)

    resultado = RetrainingTriggerManager().evaluate_trigger_conditions(reporte, max_psi=0.20)

    assert resultado["trigger_activated"] is True
    assert len(resultado["reasons"]) == 1
    assert "psi" in resultado["reasons"][0]


def test_archivo_inexistente_lanza_error_explicito(tmp_path):
    with pytest.raises(TelemetryReportError):
        RetrainingTriggerManager().evaluate_trigger_conditions(tmp_path / "no_existe.json")


def test_json_corrupto_lanza_error_explicito(tmp_path):
    corrupto = tmp_path / "post_cutover_telemetry_corrupto.json"
    corrupto.write_text("{ esto no es json valido ]")

    with pytest.raises(TelemetryReportError):
        RetrainingTriggerManager().evaluate_trigger_conditions(corrupto)


# ---------------------------------------------------------------------------
# dispatch_retraining_event
# ---------------------------------------------------------------------------

def test_dispatch_escribe_manifiesto_y_registra_el_evento_en_duckdb(tmp_path):
    reporte = _reporte_telemetria(tmp_path, realized_roc_auc=0.68, psi=0.05)
    manager = RetrainingTriggerManager()
    resultado = manager.evaluate_trigger_conditions(reporte, min_realized_auc=0.72)
    assert resultado["trigger_activated"] is True

    db_path = tmp_path / "lab_lifecycle.duckdb"
    despacho = manager.dispatch_retraining_event(tmp_path / "manifiestos", resultado, db_path)

    manifest_path = Path(despacho["manifest_path"])
    assert manifest_path.exists()
    assert manifest_path.name.startswith("retraining_trigger_")
    assert json.loads(manifest_path.read_text()) == resultado

    con = duckdb.connect(str(db_path))
    try:
        fila = con.execute(
            "SELECT event_id, event_type, previous_champion, new_champion, promoted_at "
            "FROM model_lifecycle_events WHERE event_id = ?",
            [despacho["event_id"]],
        ).fetchdf()
    finally:
        con.close()

    assert len(fila) == 1
    registro = fila.iloc[0]
    assert registro["event_type"] == "RETRAINING_TRIGGERED"
    assert registro["previous_champion"] is None
    assert registro["new_champion"] is None
    assert registro["promoted_at"] == resultado["evaluated_at"]


def test_dispatch_crea_el_directorio_padre_de_la_base_duckdb_si_falta(tmp_path):
    """`duckdb.connect` no crea el directorio padre por si solo -- falla
    con IOException si no existe. Regresion para un bug real encontrado
    corriendo el CLI de verdad contra la base 'central' por defecto
    (`../20-full-promotion-cutover/outputs/...`), que no existe hasta que
    esa tecnica corre al menos una vez."""
    reporte = _reporte_telemetria(tmp_path, realized_roc_auc=0.60, psi=0.05)
    manager = RetrainingTriggerManager()
    resultado = manager.evaluate_trigger_conditions(reporte)

    db_path = tmp_path / "no_existe_todavia" / "otro_nivel" / "lab_lifecycle.duckdb"
    assert not db_path.parent.exists()

    manager.dispatch_retraining_event(tmp_path / "manifiestos", resultado, db_path)

    assert db_path.exists()


def test_dispatch_se_suma_a_una_tabla_de_eventos_ya_existente(tmp_path):
    """Mismo esquema de columnas que `20-full-promotion-cutover`: si
    `--db-path` apunta al mismo archivo, un FULL_CUTOVER previo y un
    RETRAINING_TRIGGERED nuevo deben convivir en la misma tabla."""
    db_path = tmp_path / "lab_lifecycle.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE model_lifecycle_events (
            event_id VARCHAR, event_type VARCHAR, previous_champion VARCHAR,
            new_champion VARCHAR, promoted_at VARCHAR
        )
    """)
    con.execute(
        "INSERT INTO model_lifecycle_events VALUES (?, ?, ?, ?, ?)",
        ["evt-cutover-1", "FULL_CUTOVER", None, "champion_model.pkl", "2026-01-01T00:00:00+00:00"],
    )
    con.close()

    reporte = _reporte_telemetria(tmp_path, realized_roc_auc=0.60, psi=0.05)
    manager = RetrainingTriggerManager()
    resultado = manager.evaluate_trigger_conditions(reporte)
    manager.dispatch_retraining_event(tmp_path / "manifiestos", resultado, db_path)

    con = duckdb.connect(str(db_path))
    try:
        tipos = con.execute(
            "SELECT event_type FROM model_lifecycle_events ORDER BY event_type").fetchdf()["event_type"].tolist()
    finally:
        con.close()

    assert tipos == ["FULL_CUTOVER", "RETRAINING_TRIGGERED"]
