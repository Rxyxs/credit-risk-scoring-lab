"""Test de integracion: el modelo restringido tiene que pasar la auditoria
que el modelo libre reprueba.

Es el unico test que entrena de verdad (unos segundos), y esta porque la
afirmacion central del proyecto -- "restringir elimina las violaciones y
cuesta poco o nada" -- no puede quedar solo en el README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_applicants
from src.models import entrenar, metricas
from src.monotonicity_audit import auditar, resumen_auditoria
from src.preprocessing import etiqueta, matriz, split_tres


@pytest.fixture(scope="module")
def entrenados():
    df = generate_applicants(n=8000, seed=99)
    train, _, test = split_tres(df)
    modelos = entrenar(train)
    return modelos, test


def test_el_gbm_monotono_no_tiene_violaciones(entrenados):
    modelos, test = entrenados
    tabla = auditar(modelos["gbm_monotono"], matriz(test), muestra=600)
    assert resumen_auditoria(tabla)["auditoria_limpia"] is True
    assert tabla["violacion_maxima_pd"].max() == 0.0


def test_el_gbm_libre_si_las_tiene(entrenados):
    modelos, test = entrenados
    tabla = auditar(modelos["gbm_libre"], matriz(test), muestra=600)
    assert resumen_auditoria(tabla)["auditoria_limpia"] is False
    assert tabla["pct_casos_con_violacion"].max() > 0.10


def test_la_restriccion_no_destruye_el_desempeno(entrenados):
    modelos, test = entrenados
    X, y = matriz(test), etiqueta(test)
    auc_libre = metricas(y, modelos["gbm_libre"].predict_proba(X)[:, 1])["auc"]
    auc_mono = metricas(y, modelos["gbm_monotono"].predict_proba(X)[:, 1])["auc"]
    assert auc_mono > auc_libre - 0.02
    assert auc_mono > 0.70


def test_los_tres_modelos_predicen_probabilidades_validas(entrenados):
    modelos, test = entrenados
    X = matriz(test)
    for nombre, modelo in modelos.items():
        p = modelo.predict_proba(X)
        assert p.shape == (len(test), 2)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert np.all((p >= 0) & (p <= 1)), nombre


def test_metricas_son_consistentes_entre_si():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    m = metricas(y, p)
    assert m["auc"] == pytest.approx(1.0)
    assert m["gini"] == pytest.approx(1.0)
    assert 0 <= m["ks"] <= 1
    assert m["brier"] == pytest.approx(float(np.mean((p - y) ** 2)))
