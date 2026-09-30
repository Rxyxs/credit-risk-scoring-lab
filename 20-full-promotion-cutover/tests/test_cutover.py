"""Pruebas de CutoverManager: la conmutacion completa de Champion.

Cada prueba construye su propio estado de laboratorio con
``src.fixtures`` (un Champion activo, un candidato sombra, un reporte de
salud canaria) en un directorio temporal, para que ninguna prueba dependa
de la ejecucion de otra ni dependa de ningun archivo dejado por una
ejecucion previa de run_cutover.py.
"""

from __future__ import annotations

import json
import pickle

import duckdb
import pytest

from src.cutover_manager import CutoverManager
from src.fixtures import escribir_reporte_salud, sembrar_champion_y_sombra


@pytest.fixture
def laboratorio(tmp_path):
    """Modelos + config canaria sembrados; el reporte de salud se agrega
    por cada prueba con el estado que corresponda."""
    models_dir = tmp_path / "models"
    estado = sembrar_champion_y_sombra(models_dir)
    estado["canary_outputs_dir"] = tmp_path / "canary_monitoring_outputs"
    estado["db_path"] = tmp_path / "lab_lifecycle.duckdb"
    return estado


def _contenido(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def test_estado_healthy_ejecuta_el_cutover_exitosamente(laboratorio):
    reporte = escribir_reporte_salud(laboratorio["canary_outputs_dir"], "HEALTHY")
    champion_original = _contenido(laboratorio["champion_path"])
    candidato = _contenido(laboratorio["shadow_path"])

    manager = CutoverManager()
    assert manager.verify_health_before_cutover(reporte) is True

    resultado = manager.execute_cutover(
        shadow_model_path=laboratorio["shadow_path"],
        champion_dir=laboratorio["champion_dir"],
        archive_dir=laboratorio["archive_dir"],
        canary_config_path=laboratorio["canary_config_path"],
        db_path=laboratorio["db_path"],
    )

    # El candidato es ahora el Champion.
    nuevo_champion_path = laboratorio["champion_dir"] / "champion_model.pkl"
    assert _contenido(nuevo_champion_path) == candidato

    # El Champion anterior quedo archivado, sin perderse.
    archivado_path = laboratorio["archive_dir"] / resultado["previous_champion"]
    assert archivado_path.exists()
    assert _contenido(archivado_path) == champion_original


def test_evento_de_cutover_queda_registrado_en_duckdb_con_los_valores_exactos(laboratorio):
    reporte = escribir_reporte_salud(laboratorio["canary_outputs_dir"], "HEALTHY")
    manager = CutoverManager()
    manager.verify_health_before_cutover(reporte)

    resultado = manager.execute_cutover(
        shadow_model_path=laboratorio["shadow_path"],
        champion_dir=laboratorio["champion_dir"],
        archive_dir=laboratorio["archive_dir"],
        canary_config_path=laboratorio["canary_config_path"],
        db_path=laboratorio["db_path"],
    )

    con = duckdb.connect(str(laboratorio["db_path"]))
    try:
        fila = con.execute(
            "SELECT event_id, event_type, previous_champion, new_champion, promoted_at "
            "FROM model_lifecycle_events WHERE event_id = ?",
            [resultado["event_id"]],
        ).fetchdf()
    finally:
        con.close()

    assert len(fila) == 1
    registro = fila.iloc[0]
    assert registro["event_type"] == "FULL_CUTOVER"
    assert registro["previous_champion"] == resultado["previous_champion"]
    assert registro["new_champion"] == "champion_model.pkl"
    assert registro["promoted_at"] == resultado["promoted_at"]


def test_rollback_triggered_aborta_sin_alterar_el_champion_activo(laboratorio):
    reporte = escribir_reporte_salud(laboratorio["canary_outputs_dir"], "ROLLBACK_TRIGGERED")
    champion_original = _contenido(laboratorio["champion_path"])

    manager = CutoverManager()
    assert manager.verify_health_before_cutover(reporte) is False

    # El script llamador debe abortar aqui; el Champion sigue intacto.
    assert _contenido(laboratorio["champion_path"]) == champion_original
    assert laboratorio["archive_dir"].exists()
    assert list(laboratorio["archive_dir"].iterdir()) == []


def test_reporte_de_salud_faltante_aborta_sin_alterar_el_champion_activo(laboratorio):
    reporte_inexistente = laboratorio["canary_outputs_dir"] / "canary_health_nunca_existio.json"
    champion_original = _contenido(laboratorio["champion_path"])

    manager = CutoverManager()
    assert manager.verify_health_before_cutover(reporte_inexistente) is False
    assert _contenido(laboratorio["champion_path"]) == champion_original


def test_config_canaria_vuelve_a_cero_por_ciento_tras_un_cutover_exitoso(laboratorio):
    config_previo = json.loads(laboratorio["canary_config_path"].read_text())
    assert config_previo["canary_percentage"] == 15

    reporte = escribir_reporte_salud(laboratorio["canary_outputs_dir"], "HEALTHY")
    manager = CutoverManager()
    manager.verify_health_before_cutover(reporte)
    manager.execute_cutover(
        shadow_model_path=laboratorio["shadow_path"],
        champion_dir=laboratorio["champion_dir"],
        archive_dir=laboratorio["archive_dir"],
        canary_config_path=laboratorio["canary_config_path"],
        db_path=laboratorio["db_path"],
    )

    config_posterior = json.loads(laboratorio["canary_config_path"].read_text())
    assert config_posterior["canary_percentage"] == 0
    # Reescritura completa, con el mismo esquema que `set_traffic_split`/
    # `trigger_rollback` (tecnica 18): un `rollback` de una corrida previa
    # no debe sobrevivir a un cutover exitoso.
    assert "rollback" not in config_posterior


def test_manifiesto_de_cutover_refleja_el_resultado_ejecutado(laboratorio, tmp_path):
    reporte = escribir_reporte_salud(laboratorio["canary_outputs_dir"], "HEALTHY")
    manager = CutoverManager()
    manager.verify_health_before_cutover(reporte)
    resultado = manager.execute_cutover(
        shadow_model_path=laboratorio["shadow_path"],
        champion_dir=laboratorio["champion_dir"],
        archive_dir=laboratorio["archive_dir"],
        canary_config_path=laboratorio["canary_config_path"],
        db_path=laboratorio["db_path"],
    )

    manifest_path = manager.generate_cutover_manifest(tmp_path / "outputs" / "manifests")
    assert manifest_path.exists()
    assert manifest_path.name.startswith("cutover_manifest_")

    manifiesto = json.loads(manifest_path.read_text())
    assert manifiesto == resultado


def test_generar_manifiesto_sin_cutover_previo_falla_explicitamente(tmp_path):
    manager = CutoverManager()
    with pytest.raises(RuntimeError):
        manager.generate_cutover_manifest(tmp_path / "outputs")


def test_cutover_sobrescribe_un_rollback_previo_sin_dejar_rastro(laboratorio):
    """Si el canary_config.json que dejo la tecnica 19 viene de un rollback
    automatico (`rollback: True`, `rollback_reason: ...`), un cutover
    exitoso posterior no debe arrastrar esos campos: el 0% ahora es porque
    el candidato paso a ser el Champion completo, no porque algo fallo."""
    laboratorio["canary_config_path"].write_text(json.dumps({
        "canary_percentage": 0,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "rollback": True,
        "rollback_reason": "null_rate=0.0500 supera el limite 0.0100",
    }, indent=2))

    reporte = escribir_reporte_salud(laboratorio["canary_outputs_dir"], "HEALTHY")
    manager = CutoverManager()
    manager.verify_health_before_cutover(reporte)
    manager.execute_cutover(
        shadow_model_path=laboratorio["shadow_path"],
        champion_dir=laboratorio["champion_dir"],
        archive_dir=laboratorio["archive_dir"],
        canary_config_path=laboratorio["canary_config_path"],
        db_path=laboratorio["db_path"],
    )

    config_posterior = json.loads(laboratorio["canary_config_path"].read_text())
    assert config_posterior == {
        "canary_percentage": 0,
        "updated_at": config_posterior["updated_at"],
    }


def test_verificacion_de_salud_ignora_campos_extra_del_reporte_real(laboratorio):
    """El reporte real de la tecnica 19 trae mas campos que solo `status`
    (null_rate, mean_score_diff, high_risk_proportion, n_canary,
    n_champion, violated_threshold, reason, generated_at) -- la puerta solo
    debe mirar `status`, sin que los demas campos le importen."""
    reporte_path = laboratorio["canary_outputs_dir"] / "canary_health_real.json"
    reporte_path.parent.mkdir(parents=True, exist_ok=True)
    reporte_path.write_text(json.dumps({
        "null_rate": 0.0021,
        "mean_score_diff": 0.031,
        "high_risk_proportion": 0.084,
        "n_canary": 412,
        "n_champion": 1988,
        "status": "HEALTHY",
        "violated_threshold": None,
        "reason": None,
        "generated_at": "2026-01-01T00:00:00+00:00",
    }, indent=2))

    manager = CutoverManager()
    assert manager.verify_health_before_cutover(reporte_path) is True


def test_cutover_repetido_archiva_cada_champion_anterior_por_separado(laboratorio):
    reporte = escribir_reporte_salud(laboratorio["canary_outputs_dir"], "HEALTHY")
    manager = CutoverManager()
    manager.verify_health_before_cutover(reporte)

    primer_resultado = manager.execute_cutover(
        shadow_model_path=laboratorio["shadow_path"],
        champion_dir=laboratorio["champion_dir"],
        archive_dir=laboratorio["archive_dir"],
        canary_config_path=laboratorio["canary_config_path"],
        db_path=laboratorio["db_path"],
    )

    # Un segundo candidato, un segundo cutover.
    segundo_candidato_path = laboratorio["shadow_dir"] / "active_shadow_model.pkl"
    with open(segundo_candidato_path, "wb") as f:
        pickle.dump({"model_name": "scorecard_candidato", "version": "v3.4.0-rc1"}, f)

    segundo_resultado = manager.execute_cutover(
        shadow_model_path=segundo_candidato_path,
        champion_dir=laboratorio["champion_dir"],
        archive_dir=laboratorio["archive_dir"],
        canary_config_path=laboratorio["canary_config_path"],
        db_path=laboratorio["db_path"],
    )

    archivados = list(laboratorio["archive_dir"].iterdir())
    assert len(archivados) == 2
    assert primer_resultado["event_id"] != segundo_resultado["event_id"]

    con = duckdb.connect(str(laboratorio["db_path"]))
    try:
        total = con.execute(
            "SELECT COUNT(*) FROM model_lifecycle_events").fetchone()[0]
    finally:
        con.close()
    assert total == 2
