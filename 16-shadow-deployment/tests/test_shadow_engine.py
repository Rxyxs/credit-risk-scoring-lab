"""Tests de src.shadow_engine.ShadowDeploymentEngine y de run_shadow_serving.py."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from run_shadow_serving import find_latest_decision, main
from src.shadow_engine import ACTIVE_MODEL_FILENAME, REGISTRY_MANIFEST_FILENAME, ShadowDeploymentEngine

FEATURES = ["ingreso_mensual", "dti"]


def _modelo_entrenado(seed: int = 0) -> LogisticRegression:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(50, 2))
    y = rng.integers(0, 2, size=50)
    return LogisticRegression().fit(X, y)


def _guardar_pkl(tmp_path, nombre: str, modelo) -> Path:
    ruta = tmp_path / nombre
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    return ruta


def _escribir_decision(tmp_path, nombre: str, decision: str, candidate_model: str = "shadow_model_a.pkl") -> Path:
    ruta = tmp_path / nombre
    ruta.write_text(json.dumps({
        "candidate_model": candidate_model,
        "decision": decision,
        "reason": "test",
        "evaluated_at": "2026-09-28T00:00:00+00:00",
    }, indent=2), encoding="utf-8")
    return ruta


def _clientes_de_prueba(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "client_id": [f"CLI-{i:04d}" for i in range(n)],
        "ingreso_mensual": rng.normal(size=n),
        "dti": rng.normal(size=n),
    })


# ------------------------------------------------------------- register_promoted_model

def test_promoted_copia_el_candidato_y_escribe_el_manifiesto(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _guardar_pkl(models_dir, "shadow_model_20260928T000000Z.pkl", _modelo_entrenado())

    decision_path = _escribir_decision(tmp_path, "d.json", "PROMOTED", "shadow_model_20260928T000000Z.pkl")
    registry_dir = tmp_path / "registry"

    engine = ShadowDeploymentEngine()
    resultado = engine.register_promoted_model(decision_path, models_dir, registry_dir)

    assert resultado["registered"] is True
    activo = registry_dir / ACTIVE_MODEL_FILENAME
    assert activo.exists()
    with open(activo, "rb") as f:
        pickle.load(f)  # el .pkl copiado sigue siendo deserializable

    manifiesto = json.loads((registry_dir / REGISTRY_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifiesto["active_version"] == "shadow_model_20260928T000000Z.pkl"
    assert "activated_at" in manifiesto
    assert manifiesto["source_decision"] == "d.json"


def test_rejected_no_toca_el_modelo_sombra_activo(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()

    # ya habia un modelo activo de un ciclo anterior
    activo = registry_dir / ACTIVE_MODEL_FILENAME
    activo.write_bytes(b"contenido-del-modelo-anterior")
    contenido_original = activo.read_bytes()

    decision_path = _escribir_decision(tmp_path, "d.json", "REJECTED")
    engine = ShadowDeploymentEngine()
    resultado = engine.register_promoted_model(decision_path, models_dir, registry_dir)

    assert resultado["registered"] is False
    assert activo.read_bytes() == contenido_original  # intacto, byte por byte


def test_promoted_con_candidato_faltante_lanza_error_claro(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()  # vacia: el candidato no esta
    decision_path = _escribir_decision(tmp_path, "d.json", "PROMOTED", "shadow_model_no_existe.pkl")

    engine = ShadowDeploymentEngine()
    with pytest.raises(FileNotFoundError, match="shadow_model_no_existe"):
        engine.register_promoted_model(decision_path, models_dir, tmp_path / "registry")


# --------------------------------------------------------------------- predict_dual

def test_predict_dual_calcula_ambas_probabilidades_y_diff_abs(tmp_path):
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    challenger = _modelo_entrenado(seed=1)
    with open(registry_dir / ACTIVE_MODEL_FILENAME, "wb") as f:
        pickle.dump(challenger, f)

    champion = _modelo_entrenado(seed=2)
    clientes = _clientes_de_prueba(5)

    engine = ShadowDeploymentEngine(registry_dir=registry_dir)
    resultado = engine.predict_dual(champion, clientes)

    assert list(resultado.columns) == ["client_id", "pred_champion", "pred_challenger", "diff_abs", "timestamp"]
    assert len(resultado) == 5
    assert list(resultado["client_id"]) == list(clientes["client_id"])

    X = clientes[FEATURES]
    esperado_champion = champion.predict_proba(X)[:, 1]
    esperado_challenger = challenger.predict_proba(X)[:, 1]
    np.testing.assert_allclose(resultado["pred_champion"].astype(float), esperado_champion)
    np.testing.assert_allclose(resultado["pred_challenger"].astype(float), esperado_challenger)
    np.testing.assert_allclose(
        resultado["diff_abs"].astype(float),
        np.abs(esperado_champion - esperado_challenger),
    )
    assert (resultado["pred_champion"].astype(float).between(0, 1)).all()


def test_predict_dual_sin_challenger_activo_responde_solo_con_champion_sin_lanzar_excepcion(tmp_path):
    registry_dir = tmp_path / "registry"  # nunca se crea -- sin challenger activo
    champion = _modelo_entrenado()
    clientes = _clientes_de_prueba(3)

    engine = ShadowDeploymentEngine(registry_dir=registry_dir)
    resultado = engine.predict_dual(champion, clientes)  # no debe lanzar

    assert len(resultado) == 3
    assert resultado["pred_challenger"].isna().all()
    assert resultado["diff_abs"].isna().all()
    assert not resultado["pred_champion"].isna().any()


def test_predict_dual_acepta_una_ruta_a_pkl_ademas_de_un_modelo_cargado(tmp_path):
    champion_path = _guardar_pkl(tmp_path, "champion.pkl", _modelo_entrenado())
    clientes = _clientes_de_prueba(2)

    engine = ShadowDeploymentEngine(registry_dir=tmp_path / "registry-vacio")
    resultado = engine.predict_dual(champion_path, clientes)

    assert len(resultado) == 2
    assert not resultado["pred_champion"].isna().any()


# --------------------------------------------------------------- log_dual_predictions

def test_log_dual_predictions_crea_la_tabla_y_agrega_filas(tmp_path):
    db_path = tmp_path / "logs.duckdb"
    champion = _modelo_entrenado()
    engine = ShadowDeploymentEngine(registry_dir=tmp_path / "registry-vacio")

    predicciones_1 = engine.predict_dual(champion, _clientes_de_prueba(3))
    n1 = engine.log_dual_predictions(predicciones_1, db_path)
    predicciones_2 = engine.predict_dual(champion, _clientes_de_prueba(2))
    n2 = engine.log_dual_predictions(predicciones_2, db_path)

    assert n1 == 3 and n2 == 2

    con = duckdb.connect(str(db_path))
    total = con.execute("SELECT count(*) FROM dual_inference_logs").fetchone()[0]
    con.close()
    assert total == 5  # append, no upsert -- las dos corridas coexisten


def test_log_dual_predictions_crea_el_directorio_padre_si_no_existe(tmp_path):
    """Regresion: duckdb.connect no crea directorios padre por su cuenta --
    esto crasheaba con IOException antes de que log_dual_predictions hiciera
    el mkdir explicito."""
    db_path = tmp_path / "no" / "existe" / "todavia" / "logs.duckdb"
    champion = _modelo_entrenado()
    engine = ShadowDeploymentEngine(registry_dir=tmp_path / "registry-vacio")

    engine.log_dual_predictions(engine.predict_dual(champion, _clientes_de_prueba(2)), db_path)

    assert db_path.exists()


# ------------------------------------------------------------ CLI: run_shadow_serving.py

def test_find_latest_decision_elige_el_mas_reciente(tmp_path):
    (tmp_path / "promotion_decision_20260101T000000Z.json").write_text("{}")
    mas_reciente = tmp_path / "promotion_decision_20260927T235959Z.json"
    mas_reciente.write_text("{}")

    assert find_latest_decision(tmp_path) == mas_reciente


def test_find_latest_decision_devuelve_none_si_no_hay_nada(tmp_path):
    assert find_latest_decision(tmp_path / "no-existe") is None


def test_cli_registra_predice_y_loguea_de_punta_a_punta(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _guardar_pkl(models_dir, "shadow_model_a.pkl", _modelo_entrenado(seed=5))

    decision_dir = tmp_path / "decisions"
    decision_dir.mkdir()
    _escribir_decision(decision_dir, "promotion_decision_20260928T000000Z.json", "PROMOTED", "shadow_model_a.pkl")

    champion_path = _guardar_pkl(tmp_path, "champion.pkl", _modelo_entrenado(seed=6))
    input_csv = tmp_path / "clientes.csv"
    _clientes_de_prueba(4).to_csv(input_csv, index=False)

    registry_dir = tmp_path / "registry"
    db_path = tmp_path / "logs.duckdb"

    codigo = main([
        "--decision-dir", str(decision_dir),
        "--models-dir", str(models_dir),
        "--input-data", str(input_csv),
        "--champion-model", str(champion_path),
        "--registry-dir", str(registry_dir),
        "--db-path", str(db_path),
    ])

    assert codigo == 0
    assert (registry_dir / ACTIVE_MODEL_FILENAME).exists()
    con = duckdb.connect(str(db_path))
    assert con.execute("SELECT count(*) FROM dual_inference_logs").fetchone()[0] == 4
    con.close()


def test_cli_sin_decision_nueva_sigue_prediciendo_con_lo_que_ya_hay_en_el_registro(tmp_path):
    """Sin promotion_decision_*.json nuevo, el registro no cambia -- pero la
    inferencia dual (y el log) igual corren con lo que ya estaba activo."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    with open(registry_dir / ACTIVE_MODEL_FILENAME, "wb") as f:
        pickle.dump(_modelo_entrenado(seed=7), f)

    champion_path = _guardar_pkl(tmp_path, "champion.pkl", _modelo_entrenado(seed=8))
    input_csv = tmp_path / "clientes.csv"
    _clientes_de_prueba(2).to_csv(input_csv, index=False)
    db_path = tmp_path / "logs.duckdb"

    codigo = main([
        "--decision-dir", str(tmp_path / "vacio"),
        "--models-dir", str(tmp_path / "models-vacio"),
        "--input-data", str(input_csv),
        "--champion-model", str(champion_path),
        "--registry-dir", str(registry_dir),
        "--db-path", str(db_path),
    ])

    assert codigo == 0
    con = duckdb.connect(str(db_path))
    fila = con.execute("SELECT pred_challenger FROM dual_inference_logs LIMIT 1").fetchone()
    con.close()
    assert fila[0] is not None  # el challenger que ya estaba activo SI participo
