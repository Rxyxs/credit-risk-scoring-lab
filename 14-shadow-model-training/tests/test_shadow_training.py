"""Tests de src.train_shadow.ShadowModelTrainer y de run_training.py."""
from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
import pytest

from run_training import main
from src.train_shadow import ID_COLUMN, MIN_ROWS_TO_TRAIN, InsufficientDataError, ShadowModelTrainer


def _tabla_ficticia(n: int, seed: int = 0) -> pd.DataFrame:
    """`default_flag` correlacionado de verdad con `dti`/`ingreso_mensual`
    (no ruido puro) -- si estuviera desconectado de las features, el ROC-AUC
    rondaria 0.5 por definicion y el test de "ROC-AUC valido" no distinguiria
    un pipeline roto de uno que aprendio algo."""
    rng = np.random.default_rng(seed)
    dti = rng.beta(2, 5, size=n)
    ingreso = rng.normal(loc=800_000, scale=150_000, size=n)
    z = 3.0 * (dti - dti.mean()) / dti.std() - 1.5 * (ingreso - ingreso.mean()) / ingreso.std()
    p_default = 1 / (1 + np.exp(-z))
    default_flag = rng.binomial(1, p_default)

    return pd.DataFrame({
        "client_id": [f"CLI-{i:06d}" for i in range(n)],
        "ingreso_mensual": ingreso,
        "dti": dti,
        "default_flag": default_flag,
    })


# ------------------------------------------------------------------ fetch_data

def test_fetch_data_separa_features_y_target_e_ignora_client_id(tmp_path):
    db_path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(db_path))
    df = _tabla_ficticia(200)
    con.register("_staging", df)
    con.execute("CREATE TABLE credit_features AS SELECT * FROM _staging")
    con.close()

    trainer = ShadowModelTrainer()
    X, y = trainer.fetch_data(db_path)

    assert ID_COLUMN not in X.columns
    assert "default_flag" not in X.columns
    assert list(X.columns) == ["ingreso_mensual", "dti"]
    assert y.name == "default_flag"
    assert len(X) == len(y) == 200


def test_fetch_data_usa_target_column_explicito_si_se_pasa(tmp_path):
    db_path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(db_path))
    df = _tabla_ficticia(200).rename(columns={"default_flag": "etiqueta_custom"})
    con.register("_staging", df)
    con.execute("CREATE TABLE credit_features AS SELECT * FROM _staging")
    con.close()

    trainer = ShadowModelTrainer(target_column="etiqueta_custom")
    X, y = trainer.fetch_data(db_path)

    assert "etiqueta_custom" not in X.columns
    assert y.name == "etiqueta_custom"


# ------------------------------------------------------------- train_and_evaluate

def test_train_and_evaluate_devuelve_un_roc_auc_valido():
    df = _tabla_ficticia(400)
    X = df.drop(columns=["client_id", "default_flag"])
    y = df["default_flag"]

    trainer = ShadowModelTrainer()
    metricas = trainer.train_and_evaluate(X, y)

    assert 0.0 <= metricas["roc_auc"] <= 1.0
    assert metricas["n_samples"] == 400
    assert metricas["n_train"] + metricas["n_test"] == 400
    assert metricas["n_test"] == pytest.approx(80, abs=1)  # 20% de 400
    assert "trained_at" in metricas


def test_train_and_evaluate_aprende_señal_real_no_ruido():
    """Con el target correlacionado a proposito (ver _tabla_ficticia), un
    pipeline que funciona de verdad tiene que superar 0.5 con margen --
    0.5 exacto significaria que no esta aprendiendo nada de las features."""
    df = _tabla_ficticia(1000, seed=3)
    X = df.drop(columns=["client_id", "default_flag"])
    y = df["default_flag"]

    trainer = ShadowModelTrainer()
    metricas = trainer.train_and_evaluate(X, y)

    assert metricas["roc_auc"] > 0.6


def test_save_model_escribe_pkl_y_json_con_el_mismo_timestamp(tmp_path):
    df = _tabla_ficticia(300)
    X = df.drop(columns=["client_id", "default_flag"])
    y = df["default_flag"]

    trainer = ShadowModelTrainer()
    trainer.train_and_evaluate(X, y)
    resultado = trainer.save_model(tmp_path)

    import pickle
    from pathlib import Path
    import json

    model_path = Path(resultado["model_path"])
    metrics_path = Path(resultado["metrics_path"])
    assert model_path.exists() and model_path.name.startswith("shadow_model_")
    assert metrics_path.exists() and metrics_path.name.startswith("shadow_metrics_")
    assert model_path.stem.replace("shadow_model_", "") == metrics_path.stem.replace("shadow_metrics_", "")

    with open(model_path, "rb") as f:
        pipeline_cargado = pickle.load(f)
    assert hasattr(pipeline_cargado, "predict_proba")

    metricas_leidas = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metricas_leidas["roc_auc"] == trainer.metrics["roc_auc"]
    assert "n_samples" in metricas_leidas and "trained_at" in metricas_leidas


def test_save_model_sin_entrenar_antes_lanza_error(tmp_path):
    trainer = ShadowModelTrainer()
    with pytest.raises(RuntimeError, match="no hay modelo entrenado"):
        trainer.save_model(tmp_path)


# ------------------------------------------------------- casos de aborto seguro

def test_fetch_data_tabla_inexistente_lanza_insufficient_data_error():
    """':memory:' como db_path: cada llamada a duckdb.connect(':memory:')
    abre una base nueva y vacia (no comparten estado entre si), asi que es
    la forma mas directa de simular "el feature store existe pero credit_features
    todavia no" sin tocar el filesystem para nada."""
    trainer = ShadowModelTrainer()
    with pytest.raises(InsufficientDataError, match="credit_features"):
        trainer.fetch_data(":memory:")


def test_fetch_data_pocas_filas_lanza_insufficient_data_error(tmp_path):
    db_path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(db_path))
    df = _tabla_ficticia(MIN_ROWS_TO_TRAIN - 1)  # justo por debajo del minimo
    con.register("_staging", df)
    con.execute("CREATE TABLE credit_features AS SELECT * FROM _staging")
    con.close()

    trainer = ShadowModelTrainer()
    with pytest.raises(InsufficientDataError, match=str(MIN_ROWS_TO_TRAIN)):
        trainer.fetch_data(db_path)


def test_fetch_data_sin_target_reconocible_lanza_insufficient_data_error(tmp_path):
    db_path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(db_path))
    df = _tabla_ficticia(200).drop(columns=["default_flag"])  # sin ninguna columna de target
    con.register("_staging", df)
    con.execute("CREATE TABLE credit_features AS SELECT * FROM _staging")
    con.close()

    trainer = ShadowModelTrainer()
    with pytest.raises(InsufficientDataError, match="target"):
        trainer.fetch_data(db_path)


def test_fetch_data_db_inexistente_lanza_insufficient_data_error_no_crashea(tmp_path):
    trainer = ShadowModelTrainer()
    with pytest.raises(InsufficientDataError):
        trainer.fetch_data(tmp_path / "no" / "existe" / "store.duckdb")


# ------------------------------------------------------------ CLI: run_training.py

def test_cli_entrena_y_guarda_cuando_hay_datos_suficientes(tmp_path):
    db_path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(db_path))
    df = _tabla_ficticia(300)
    con.register("_staging", df)
    con.execute("CREATE TABLE credit_features AS SELECT * FROM _staging")
    con.close()

    output_dir = tmp_path / "models"
    codigo = main(["--db-path", str(db_path), "--output-dir", str(output_dir)])

    assert codigo == 0
    assert list(output_dir.glob("shadow_model_*.pkl"))
    assert list(output_dir.glob("shadow_metrics_*.json"))


def test_cli_aborta_con_codigo_0_sin_excepcion_si_no_hay_tabla(tmp_path, caplog):
    db_path = tmp_path / "vacio.duckdb"
    duckdb.connect(str(db_path)).close()
    output_dir = tmp_path / "models"

    with caplog.at_level(logging.INFO):
        codigo = main(["--db-path", str(db_path), "--output-dir", str(output_dir)])

    assert codigo == 0
    assert not output_dir.exists()  # no se creo nada: no hubo entrenamiento
    assert any("abortando sin error" in r.message for r in caplog.records)


def test_cli_aborta_con_codigo_0_si_el_archivo_de_la_base_no_existe(tmp_path):
    codigo = main([
        "--db-path", str(tmp_path / "no" / "existe.duckdb"),
        "--output-dir", str(tmp_path / "models"),
    ])
    assert codigo == 0


def test_cli_aborta_con_codigo_0_con_pocas_filas(tmp_path):
    db_path = tmp_path / "store.duckdb"
    con = duckdb.connect(str(db_path))
    df = _tabla_ficticia(10)
    con.register("_staging", df)
    con.execute("CREATE TABLE credit_features AS SELECT * FROM _staging")
    con.close()

    codigo = main(["--db-path", str(db_path), "--output-dir", str(tmp_path / "models")])
    assert codigo == 0
