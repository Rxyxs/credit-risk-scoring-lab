"""Tests de la auditoria de monotonia y de la politica de decision."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    FEATURES, MONOTONIC_CST, generate_applicants, generate_shifted,
)
from src.decision_policy import (
    COSTO_REVISION_CLP, LGD, MARGEN, decisiones_conformes, decisiones_por_banda,
    evaluar_politica,
)
from src.models import vector_restricciones
from src.monotonicity_audit import (
    auditar, curva_respuesta, grilla_de_valores, resumen_auditoria,
    violaciones_por_feature,
)
from src.preprocessing import split_tres


class ModeloMonotono:
    """PD que crece con dti (columna 0) por construccion."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(2.0 * X[:, 0] - 1.0)))
        return np.column_stack([1 - p, p])


class ModeloConEscalon:
    """PD que baja en un tramo de dti: exactamente lo que hay que detectar."""

    def predict_proba(self, X):
        d = X[:, 0]
        p = 1 / (1 + np.exp(-(2.0 * d - 1.0)))
        p = np.where((d > 0.30) & (d < 0.45), p - 0.12, p)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def _X(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    X[:, 0] = rng.uniform(0.02, 0.7, n)     # dti
    return X


# ------------------------------------------------------- auditoria
def test_modelo_monotono_no_registra_violaciones():
    r = violaciones_por_feature(ModeloMonotono(), _X(), "dti", 1)
    assert r["pct_casos_con_violacion"] == 0.0
    assert r["violacion_maxima_pd"] == 0.0
    assert r["efecto_medio_total"] > 0


def test_la_auditoria_detecta_un_escalon_a_la_baja():
    r = violaciones_por_feature(ModeloConEscalon(), _X(), "dti", 1, n_puntos=20)
    assert r["pct_casos_con_violacion"] > 0.9
    assert r["violacion_maxima_pd"] > 0.05


def test_direccion_esperada_invertida_marca_todo():
    """Si la direccion declarada es la contraria, todo el modelo la viola."""
    r = violaciones_por_feature(ModeloMonotono(), _X(), "dti", -1)
    assert r["pct_casos_con_violacion"] == 1.0


def test_no_se_audita_una_feature_sin_direccion():
    with pytest.raises(ValueError):
        violaciones_por_feature(ModeloMonotono(), _X(), "edad", 0)


def test_auditar_recorre_solo_las_features_restringidas():
    tabla = auditar(ModeloMonotono(), _X(), muestra=200)
    assert len(tabla) == sum(1 for v in MONOTONIC_CST.values() if v != 0)
    assert "edad" not in set(tabla["feature"])
    resumen = resumen_auditoria(tabla)
    assert resumen["features_auditadas"] == len(tabla)
    assert isinstance(resumen["auditoria_limpia"], bool)


def test_grilla_y_curva_de_respuesta_son_crecientes_en_el_eje():
    X = _X()
    g = grilla_de_valores(X[:, 0], n_puntos=10)
    assert g.size == 10
    assert np.all(np.diff(g) > 0)
    curva = curva_respuesta(ModeloMonotono(), X, "dti", n_puntos=15, muestra=300)
    assert len(curva) == 15
    assert curva["pd_media"].is_monotonic_increasing


def test_vector_de_restricciones_respeta_el_orden_de_las_columnas():
    v = vector_restricciones()
    assert len(v) == len(FEATURES)
    assert v == [MONOTONIC_CST[f] for f in FEATURES]
    assert v[FEATURES.index("edad")] == 0
    assert v[FEATURES.index("log_renta")] == -1


# ---------------------------------------------------------- datos
def test_el_escenario_de_deterioro_empeora_los_drivers():
    base = generate_applicants(n=4000, seed=1)
    shift = generate_shifted(n=4000, seed=1)
    assert shift["default_12m"].mean() > base["default_12m"].mean()
    assert shift["utilizacion_lineas"].mean() > base["utilizacion_lineas"].mean()
    assert shift["n_moras_12m"].mean() > base["n_moras_12m"].mean()
    assert shift["renta_liquida"].mean() < base["renta_liquida"].mean()


def test_split_en_tres_es_disjunto_y_estratificado():
    df = generate_applicants(n=5000, seed=2)
    train, calib, test = split_tres(df)
    assert len(train) + len(calib) + len(test) == len(df)
    ids = [set(p["applicant_id"]) for p in (train, calib, test)]
    assert ids[0] & ids[1] == set() and ids[0] & ids[2] == set() and ids[1] & ids[2] == set()
    tasas = [p["default_12m"].mean() for p in (train, calib, test)]
    assert max(tasas) - min(tasas) < 0.02
    with pytest.raises(ValueError):
        split_tres(df, frac_cal=0.6, frac_test=0.5)


# -------------------------------------------------------- politica
def test_mapeo_de_conjuntos_a_decisiones():
    conjuntos = np.array([[True, False], [False, True], [True, True], [False, False]])
    dec = decisiones_conformes(conjuntos)
    assert list(dec) == ["aprobar", "rechazar", "revisar", "revisar"]


def test_banda_de_score_manda_a_revision_el_volumen_pedido():
    pd_pred = np.linspace(0.01, 0.9, 1000)
    dec = decisiones_por_banda(pd_pred, frac_revision=0.20, corte_aprobacion=0.30)
    assert (dec == "revisar").sum() == 200
    # los revisados son los mas cercanos al corte
    revisados = pd_pred[dec == "revisar"]
    assert abs(revisados.mean() - 0.30) < 0.02
    assert set(np.unique(dec)) <= {"aprobar", "revisar", "rechazar"}


def test_evaluar_politica_calculada_a_mano():
    dec = np.array(["aprobar", "aprobar", "rechazar", "revisar"], dtype=object)
    y = np.array([0, 1, 0, 1])
    monto = np.array([1e6, 2e6, 3e6, 4e6])
    r = evaluar_politica(dec, y, monto, "prueba")

    assert r["pct_aprobado"] == pytest.approx(0.5)
    assert r["pct_revision_manual"] == pytest.approx(0.25)
    assert r["malos_aprobados"] == 1
    assert r["buenos_rechazados"] == 1
    assert r["tasa_error_decisiones_automaticas"] == pytest.approx(2 / 3)
    assert r["ingreso_clp"] == pytest.approx(1e6 * MARGEN)
    assert r["perdida_clp"] == pytest.approx(2e6 * LGD)
    assert r["costo_revision_clp"] == pytest.approx(COSTO_REVISION_CLP)
    assert r["utilidad_clp"] == pytest.approx(
        1e6 * MARGEN - 2e6 * LGD - COSTO_REVISION_CLP
    )
