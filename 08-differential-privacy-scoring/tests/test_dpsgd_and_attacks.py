"""Tests de DP-SGD, de los ataques y del simulador."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.attacks import (
    ataque_de_membresia, exposicion_de_canarios, riesgo_de_reidentificacion,
)
from src.data_generator import FEATURES, generar_canarios, generate_datasets
from src.dp_sgd import DPLogisticRegression, sigmoide


def _datos(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    eta = -1.0 + 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.6 * X[:, 2]
    y = (rng.random(n) < sigmoide(eta)).astype(float)
    return X, y


# ------------------------------------------------------------- DP-SGD
def test_sin_ruido_ni_recorte_se_parece_a_la_logistica_clasica():
    X, y = _datos(n=6000)
    dp = DPLogisticRegression(clip=1e6, sigma=0.0, q=0.25, pasos=3000, lr=1.0,
                              seed=1).fit(X, y)
    sk = LogisticRegression(max_iter=5000).fit(X, y)

    auc_dp = roc_auc_score(y, dp.predict_proba(X)[:, 1])
    auc_sk = roc_auc_score(y, sk.predict_proba(X)[:, 1])
    assert auc_dp == pytest.approx(auc_sk, abs=0.01)
    # y los coeficientes apuntan en la misma direccion
    coseno = float(np.dot(dp.coef_[1:], sk.coef_.ravel())
                   / (np.linalg.norm(dp.coef_[1:]) * np.linalg.norm(sk.coef_)))
    assert coseno > 0.98


def test_el_recorte_acota_la_norma_de_cada_gradiente():
    X, y = _datos(n=500)
    modelo = DPLogisticRegression(clip=0.7)
    Xa = modelo._con_intercepto(X)
    grads = modelo.gradientes_por_ejemplo(Xa, y, np.zeros(Xa.shape[1]))
    recortados, fraccion = modelo.recortar(grads)

    assert np.all(np.linalg.norm(recortados, axis=1) <= 0.7 + 1e-9)
    assert 0.0 < fraccion <= 1.0
    # los gradientes que ya cumplian la cota no se tocan
    pequenos = np.linalg.norm(grads, axis=1) <= 0.7
    assert np.allclose(recortados[pequenos], grads[pequenos])


def test_mas_ruido_degrada_la_utilidad():
    X, y = _datos(n=4000, seed=2)
    aucs = []
    for sigma in (0.0, 2.0, 20.0):
        m = DPLogisticRegression(clip=2.0, sigma=sigma, q=0.1, pasos=600, lr=0.8,
                                 seed=3).fit(X, y)
        aucs.append(roc_auc_score(y, m.predict_proba(X)[:, 1]))
    assert aucs[0] > aucs[1] > aucs[2]
    assert aucs[0] > 0.75


def test_el_presupuesto_reportado_calza_con_la_configuracion():
    X, y = _datos(n=1000)
    m = DPLogisticRegression(clip=1.0, sigma=4.0, q=0.1, pasos=500).fit(X, y)
    p = m.presupuesto_privacidad(delta=1e-5)
    assert p["sigma"] == 4.0 and p["pasos"] == 500 and p["q"] == pytest.approx(0.1)
    assert 0 < p["epsilon"] < 10

    sin_ruido = DPLogisticRegression(sigma=0.0, pasos=10).fit(X, y)
    assert np.isinf(sin_ruido.presupuesto_privacidad()["epsilon"])


def test_el_muestreo_de_poisson_da_lotes_de_tamano_variable():
    X, y = _datos(n=2000)
    m = DPLogisticRegression(clip=1.0, sigma=1.0, q=0.1, pasos=200, seed=5).fit(X, y)
    tamanos = np.array(m.historial_.tamano_lote)
    assert tamanos.std() > 5           # no es un lote fijo
    assert tamanos.mean() == pytest.approx(0.1 * 2000, rel=0.1)


def test_reproducible_con_la_misma_semilla_y_distinto_con_otra():
    X, y = _datos(n=1500)
    a = DPLogisticRegression(sigma=3.0, seed=7, pasos=200).fit(X, y).coef_
    b = DPLogisticRegression(sigma=3.0, seed=7, pasos=200).fit(X, y).coef_
    c = DPLogisticRegression(sigma=3.0, seed=8, pasos=200).fit(X, y).coef_
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        DPLogisticRegression(clip=0)
    with pytest.raises(ValueError):
        DPLogisticRegression(sigma=-1)
    with pytest.raises(ValueError):
        DPLogisticRegression(q=1.5)
    X, y = _datos(n=100)
    with pytest.raises(ValueError):
        DPLogisticRegression().fit(X, y[:50])


# ------------------------------------------------------------ ataques
def test_el_ataque_no_encuentra_nada_cuando_no_hay_membresia_real():
    """Miembros y no miembros de la misma distribucion, modelo sin memoria."""
    X, y = _datos(n=6000, seed=9)
    a, b = X[:3000], X[3000:]
    ya, yb = y[:3000], y[3000:]
    modelo = DPLogisticRegression(clip=2.0, sigma=0.0, q=0.2, pasos=400,
                                  seed=10).fit(X, y)   # vio a los dos grupos
    res = ataque_de_membresia(modelo, a, ya, b, yb)
    assert res["auc_ataque"] == pytest.approx(0.5, abs=0.03)


def test_el_ataque_detecta_un_modelo_que_memorizo():
    """Entrenado solo con la mitad: esa mitad tiene menor perdida."""
    rng = np.random.default_rng(11)
    # Tantos parametros como observaciones y etiquetas puramente aleatorias:
    # el unico ajuste posible es memorizar, que es justo lo que el ataque
    # tiene que detectar.
    n, p = 150, 150
    X = rng.normal(size=(2 * n, p))
    y = (rng.random(2 * n) < 0.5).astype(float)
    modelo = DPLogisticRegression(clip=1e6, sigma=0.0, q=0.5, pasos=8000, lr=2.0,
                                  seed=12).fit(X[:n], y[:n])
    res = ataque_de_membresia(modelo, X[:n], y[:n], X[n:], y[n:])
    assert res["auc_ataque"] > 0.70
    assert res["brecha_de_perdida"] > 0
    assert riesgo_de_reidentificacion(res)["lectura"] != "no hay filtracion detectable"


def test_la_estratificacion_neutraliza_la_diferencia_de_tasa_base():
    """Un modelo constante no filtra nada, aunque las tasas base difieran."""
    class ModeloConstante:
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])

        def perdida_por_ejemplo(self, X, y):
            p = np.clip(self.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
            y = np.asarray(y, dtype=float)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    rng = np.random.default_rng(13)
    Xm, Xn = rng.normal(size=(2000, 3)), rng.normal(size=(2000, 3))
    ym = (rng.random(2000) < 0.15).astype(float)     # miembros: 15% de malos
    yn = (rng.random(2000) < 0.30).astype(float)     # no miembros: 30%

    modelo = ModeloConstante()
    sin_estratificar = ataque_de_membresia(modelo, Xm, ym, Xn, yn,
                                           estratificado=False)["auc_ataque"]
    estratificado = ataque_de_membresia(modelo, Xm, ym, Xn, yn)["auc_ataque"]

    assert sin_estratificar > 0.55           # falsa senal por la tasa base
    assert estratificado == pytest.approx(0.5, abs=1e-9)


def test_la_exposicion_de_canarios_es_cero_si_el_modelo_no_los_vio():
    rng = np.random.default_rng(14)
    canarios = generar_canarios(60, rng)
    sombra = generar_canarios(60, np.random.default_rng(15))
    X, y = _datos(n=2000, seed=16)
    modelo = DPLogisticRegression(clip=2.0, sigma=0.0, q=0.2, pasos=300,
                                  seed=17).fit(X, y)

    Xc = canarios[FEATURES].to_numpy(float)[:, :4]
    Xs = sombra[FEATURES].to_numpy(float)[:, :4]
    res = exposicion_de_canarios(modelo, Xc, Xs)
    # ninguno de los dos grupos estuvo en el entrenamiento: la diferencia
    # solo puede venir de la variacion entre muestras, no de memorizacion
    assert abs(res["exposicion_pp"]) < 5.0
    assert res["n_canarios"] == 60


# --------------------------------------------------------------- datos
def test_los_conjuntos_son_disjuntos_y_los_canarios_van_solo_en_train():
    partes = generate_datasets(n_train=500, n_holdout=500, n_test=800, n_canarios=20)
    assert partes["train"]["es_canario"].sum() == 20
    assert partes["holdout"]["es_canario"].sum() == 0
    assert partes["test"]["es_canario"].sum() == 0
    assert len(partes["train"]) == 520
    ids = [set(partes[k]["record_id"]) for k in ("train", "holdout", "test")]
    assert ids[0] & ids[1] == set() and ids[1] & ids[2] == set()


def test_los_canarios_contradicen_su_propio_perfil():
    partes = generate_datasets(n_train=800, n_holdout=200, n_test=200, n_canarios=30)
    canarios = partes["canarios"]
    normales = partes["train"][~partes["train"]["es_canario"]]
    assert (canarios["default_12m"] == 1).all()
    assert canarios["renta_liquida"].min() > normales["renta_liquida"].quantile(0.95)
    assert canarios["dti"].max() < normales["dti"].quantile(0.05)
    assert (canarios["n_moras_12m"] == 0).all()


def test_el_simulador_es_reproducible():
    a = generate_datasets(n_train=300, n_holdout=300, n_test=300, seed=3)["train"]
    b = generate_datasets(n_train=300, n_holdout=300, n_test=300, seed=3)["train"]
    assert a.equals(b)
