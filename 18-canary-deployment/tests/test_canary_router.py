"""Tests de src.canary_router.CanaryRouter y de run_canary.py."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from run_canary import main
from src.canary_router import CANARY, CHAMPION, LOG_TABLE_NAME, CanaryRouter, InvalidCanaryPercentageError

FEATURES = ["ingreso_mensual", "dti"]


def _modelo_constante(clase_constante: int) -> DummyClassifier:
    """Un clasificador que ignora por completo las features y siempre
    predice `clase_constante` -- `predict_proba` da 0.0 o 1.0 exactos, sin
    ambigüedad, para poder confirmar sin lugar a dudas que
    `route_and_predict` enruto cada fila al modelo correcto."""
    modelo = DummyClassifier(strategy="constant", constant=clase_constante)
    modelo.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))
    return modelo


def _guardar_pkl(tmp_path, nombre: str, modelo) -> Path:
    ruta = tmp_path / nombre
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    return ruta


def _clientes(n: int, prefijo: str = "CLI") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "client_id": [f"{prefijo}-{i:05d}" for i in range(n)],
        "ingreso_mensual": rng.normal(size=n),
        "dti": rng.normal(size=n),
    })


# --------------------------------------------------------------- determine_route

def test_el_mismo_client_id_siempre_cae_en_la_misma_cohorte():
    router = CanaryRouter()
    rutas = {router.determine_route("CLI-000123", 30) for _ in range(20)}
    assert len(rutas) == 1  # 20 llamadas, siempre la misma ruta


def test_el_mismo_client_id_da_la_misma_ruta_entre_instancias_distintas():
    assert CanaryRouter().determine_route("CLI-000123", 30) == CanaryRouter().determine_route("CLI-000123", 30)


def test_porcentaje_0_manda_todo_a_champion():
    router = CanaryRouter()
    for cid in [f"CLI-{i:05d}" for i in range(200)]:
        assert router.determine_route(cid, 0) == CHAMPION


def test_porcentaje_100_manda_todo_a_canary():
    router = CanaryRouter()
    for cid in [f"CLI-{i:05d}" for i in range(200)]:
        assert router.determine_route(cid, 100) == CANARY


def test_distribucion_de_1000_solicitudes_con_20_por_ciento_aproxima_el_20_por_ciento():
    router = CanaryRouter()
    rutas = [router.determine_route(f"CLI-{i:06d}", 20) for i in range(1000)]
    fraccion_canary = rutas.count(CANARY) / 1000

    assert 0.17 <= fraccion_canary <= 0.23  # 20% +/- 3 puntos porcentuales


def test_distribucion_de_1000_solicitudes_con_50_por_ciento_aproxima_el_50_por_ciento():
    router = CanaryRouter()
    rutas = [router.determine_route(f"CLI-{i:06d}", 50) for i in range(1000)]
    fraccion_canary = rutas.count(CANARY) / 1000

    assert 0.47 <= fraccion_canary <= 0.53


# ---------------------------------------------------------- porcentajes invalidos

@pytest.mark.parametrize("porcentaje", [-1, 101, -0.5, 100.5])
def test_porcentaje_fuera_de_rango_lanza_invalid_canary_percentage_error(porcentaje):
    router = CanaryRouter()
    with pytest.raises(InvalidCanaryPercentageError):
        router.determine_route("CLI-000001", porcentaje)


@pytest.mark.parametrize("porcentaje", ["veinte", None, [20]])
def test_porcentaje_no_numerico_lanza_invalid_canary_percentage_error(porcentaje):
    router = CanaryRouter()
    with pytest.raises(InvalidCanaryPercentageError):
        router.determine_route("CLI-000001", porcentaje)


def test_porcentajes_limite_0_y_100_son_validos():
    router = CanaryRouter()
    router.determine_route("CLI-000001", 0)
    router.determine_route("CLI-000001", 100)  # ninguno de los dos deberia lanzar


# ----------------------------------------------------------------- set_traffic_split

def test_set_traffic_split_escribe_el_json_con_porcentaje_y_fecha(tmp_path):
    router = CanaryRouter()
    config_path = tmp_path / "canary_config.json"

    resultado = router.set_traffic_split(25, config_path)

    assert config_path.exists()
    contenido = json.loads(config_path.read_text(encoding="utf-8"))
    assert contenido["canary_percentage"] == 25
    assert "updated_at" in contenido
    assert resultado == contenido


def test_set_traffic_split_con_porcentaje_invalido_no_escribe_nada(tmp_path):
    router = CanaryRouter()
    config_path = tmp_path / "canary_config.json"

    with pytest.raises(InvalidCanaryPercentageError):
        router.set_traffic_split(150, config_path)

    assert not config_path.exists()


# ------------------------------------------------------------------- trigger_rollback

def test_trigger_rollback_fuerza_el_trafico_a_0_de_inmediato(tmp_path):
    router = CanaryRouter()
    config_path = tmp_path / "canary_config.json"
    router.set_traffic_split(40, config_path)  # arranca en 40%

    resultado = router.trigger_rollback(config_path, reason="tasa de error del canary por encima del limite")

    assert resultado["canary_percentage"] == 0
    contenido = json.loads(config_path.read_text(encoding="utf-8"))
    assert contenido["canary_percentage"] == 0
    assert contenido["rollback"] is True
    assert contenido["rollback_reason"] == "tasa de error del canary por encima del limite"


def test_trigger_rollback_crea_el_archivo_si_no_existia(tmp_path):
    router = CanaryRouter()
    config_path = tmp_path / "sub" / "canary_config.json"  # ni el directorio existe

    router.trigger_rollback(config_path, reason="primer rollback, sin config previa")

    assert config_path.exists()


# ------------------------------------------------------------------- route_and_predict

def test_route_and_predict_enruta_cada_cliente_al_modelo_correcto(tmp_path):
    champion = _modelo_constante(0)  # el Champion "siempre dice bajo riesgo"
    canary = _modelo_constante(1)    # el Canary "siempre dice alto riesgo"

    clientes = _clientes(300)
    router = CanaryRouter()
    resultado = router.route_and_predict(champion, canary, clientes, canary_percentage=30)

    assert set(resultado["assigned_model"]) <= {CHAMPION, CANARY}
    assert list(resultado.columns) == ["client_id", "assigned_model", "prediction", "timestamp"]

    filas_canary = resultado[resultado["assigned_model"] == CANARY]
    filas_champion = resultado[resultado["assigned_model"] == CHAMPION]
    assert len(filas_canary) > 0 and len(filas_champion) > 0  # con 300 clientes y 30%, tienen que aparecer ambos

    assert (filas_canary["prediction"] == 1.0).all()
    assert (filas_champion["prediction"] == 0.0).all()


def test_route_and_predict_coincide_con_determine_route_fila_por_fila():
    champion = _modelo_constante(0)
    canary = _modelo_constante(1)
    clientes = _clientes(50)

    router = CanaryRouter()
    resultado = router.route_and_predict(champion, canary, clientes, canary_percentage=40)

    for _, fila in resultado.iterrows():
        assert fila["assigned_model"] == router.determine_route(fila["client_id"], 40)


def test_route_and_predict_acepta_rutas_a_pkl(tmp_path):
    champion_path = _guardar_pkl(tmp_path, "champion.pkl", _modelo_constante(0))
    canary_path = _guardar_pkl(tmp_path, "canary.pkl", _modelo_constante(1))
    clientes = _clientes(20)

    router = CanaryRouter()
    resultado = router.route_and_predict(champion_path, canary_path, clientes, canary_percentage=50)

    assert len(resultado) == 20
    assert not resultado["prediction"].isna().any()


# --------------------------------------------------------------- log_routing_decisions

def test_log_routing_decisions_crea_la_tabla_y_agrega_filas(tmp_path):
    champion = _modelo_constante(0)
    canary = _modelo_constante(1)
    router = CanaryRouter()
    db_path = tmp_path / "logs.duckdb"

    predicciones_1 = router.route_and_predict(champion, canary, _clientes(10), canary_percentage=30)
    n1 = router.log_routing_decisions(predicciones_1, db_path)
    predicciones_2 = router.route_and_predict(champion, canary, _clientes(5, prefijo="OTR"), canary_percentage=30)
    n2 = router.log_routing_decisions(predicciones_2, db_path)

    assert n1 == 10 and n2 == 5

    con = duckdb.connect(str(db_path))
    total = con.execute(f"SELECT count(*) FROM {LOG_TABLE_NAME}").fetchone()[0]
    con.close()
    assert total == 15  # append, no upsert -- las dos corridas coexisten


def test_log_routing_decisions_crea_el_directorio_padre_si_no_existe(tmp_path):
    """Regresion: duckdb.connect no crea directorios padre por su cuenta."""
    champion = _modelo_constante(0)
    canary = _modelo_constante(1)
    router = CanaryRouter()
    db_path = tmp_path / "no" / "existe" / "todavia" / "logs.duckdb"

    predicciones = router.route_and_predict(champion, canary, _clientes(5), canary_percentage=30)
    router.log_routing_decisions(predicciones, db_path)

    assert db_path.exists()


# ------------------------------------------------------------------- CLI: run_canary.py

def test_cli_actualiza_el_porcentaje(tmp_path):
    config_path = tmp_path / "canary_config.json"
    codigo = main(["--canary-percentage", "15", "--config-path", str(config_path)])

    assert codigo == 0
    contenido = json.loads(config_path.read_text(encoding="utf-8"))
    assert contenido["canary_percentage"] == 15


def test_cli_rollback_fuerza_0_por_ciento(tmp_path):
    config_path = tmp_path / "canary_config.json"
    main(["--canary-percentage", "50", "--config-path", str(config_path)])

    codigo = main(["--rollback", "--reason", "prueba", "--config-path", str(config_path)])

    assert codigo == 0
    contenido = json.loads(config_path.read_text(encoding="utf-8"))
    assert contenido["canary_percentage"] == 0
    assert contenido["rollback_reason"] == "prueba"


def test_cli_sin_porcentaje_explicito_usa_el_guardado_en_config(tmp_path):
    config_path = tmp_path / "canary_config.json"
    champion_path = _guardar_pkl(tmp_path, "champion.pkl", _modelo_constante(0))
    canary_path = _guardar_pkl(tmp_path, "canary.pkl", _modelo_constante(1))
    input_csv = tmp_path / "clientes.csv"
    _clientes(100).to_csv(input_csv, index=False)

    main(["--canary-percentage", "35", "--config-path", str(config_path)])  # fija 35% primero

    codigo = main([
        "--config-path", str(config_path),
        "--champion-model", str(champion_path), "--canary-model", str(canary_path),
        "--input-data", str(input_csv),
    ])

    assert codigo == 0


def test_cli_sin_config_previa_usa_el_default_de_10_por_ciento(tmp_path, capsys):
    codigo = main(["--config-path", str(tmp_path / "no-existe.json")])
    salida = capsys.readouterr().out

    assert codigo == 0
    assert "10%" in salida
