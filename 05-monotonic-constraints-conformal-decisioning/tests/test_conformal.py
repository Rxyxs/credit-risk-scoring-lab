"""Tests de la prediccion conforme.

La propiedad que hay que verificar no es "corre sin error" sino la
garantia misma: sobre datos intercambiables, la cobertura empirica tiene
que quedar en el nivel pedido. Se testea con un modelo deliberadamente
malo ademas de uno bueno, porque la garantia conforme no depende de que el
modelo sea bueno -- y si el test solo pasara con un buen modelo, estaria
midiendo otra cosa.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.conformal import MondrianConformalClassifier


class ModeloConstante:
    """Modelo inutil: devuelve siempre la misma probabilidad."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _datos(n=6000, seed=0, desbalance=0.2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    eta = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
    eta += np.log(desbalance / (1 - desbalance)) - eta.mean()
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return X, y


def _split(X, y):
    n = len(X)
    a, b = int(0.5 * n), int(0.75 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def test_cobertura_se_cumple_con_un_buen_modelo():
    X, y = _datos()
    (Xtr, ytr), (Xc, yc), (Xt, yt) = _split(X, y)
    modelo = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    cp = MondrianConformalClassifier(modelo).calibrate(Xc, yc)

    for alpha in (0.05, 0.10, 0.20):
        c = cp.evaluar_cobertura(Xt, yt, alpha)
        assert c["cobertura_global"] > 1 - alpha - 0.03
        assert c["cobertura_clase_1"] > 1 - alpha - 0.05


def test_cobertura_se_cumple_incluso_con_un_modelo_inutil():
    """La garantia es del procedimiento, no del modelo."""
    X, y = _datos(seed=1)
    _, (Xc, yc), (Xt, yt) = _split(X, y)
    cp = MondrianConformalClassifier(ModeloConstante()).calibrate(Xc, yc)
    c = cp.evaluar_cobertura(Xt, yt, 0.10)
    assert c["cobertura_global"] > 0.87
    # el precio de un modelo malo es que casi no puede descartar nada
    assert c["tamano_medio_conjunto"] > 1.5


def test_mondrian_protege_a_la_clase_minoritaria():
    X, y = _datos(n=8000, seed=2, desbalance=0.12)
    (Xtr, ytr), (Xc, yc), (Xt, yt) = _split(X, y)
    modelo = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

    mondrian = MondrianConformalClassifier(modelo, mondrian=True).calibrate(Xc, yc)
    marginal = MondrianConformalClassifier(modelo, mondrian=False).calibrate(Xc, yc)

    c_m = mondrian.evaluar_cobertura(Xt, yt, 0.10)
    c_g = marginal.evaluar_cobertura(Xt, yt, 0.10)

    assert c_m["cobertura_clase_1"] > 1 - 0.10 - 0.05
    # la marginal cumple global pero deja atras a la clase 1
    assert c_g["cobertura_global"] > 0.85
    assert c_g["cobertura_clase_1"] < c_m["cobertura_clase_1"] - 0.10


def test_p_values_en_rango_y_conjuntos_anidados_en_alpha():
    X, y = _datos(seed=3)
    (Xtr, ytr), (Xc, yc), (Xt, yt) = _split(X, y)
    modelo = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    cp = MondrianConformalClassifier(modelo).calibrate(Xc, yc)

    p = cp.p_values(Xt)
    assert p.shape == (len(Xt), 2)
    assert np.all(p > 0) and np.all(p <= 1.0)

    grande = cp.prediction_sets(Xt, 0.05)
    chico = cp.prediction_sets(Xt, 0.20)
    # subir alpha solo puede sacar etiquetas del conjunto, nunca agregarlas
    assert np.all(chico <= grande)
    assert chico.sum() < grande.sum()


def test_tipos_de_conjunto_cubren_todos_los_casos():
    conjuntos = np.array([[True, False], [False, True], [True, True], [False, False]])
    tipos = MondrianConformalClassifier.clasificar_conjuntos(conjuntos)
    assert list(tipos) == ["solo_bueno", "solo_malo", "ambos", "vacio"]


def test_errores_de_uso():
    X, y = _datos(n=500, seed=4)
    modelo = LogisticRegression(max_iter=1000).fit(X, y)
    cp = MondrianConformalClassifier(modelo)
    with pytest.raises(RuntimeError):
        cp.p_values(X)                      # sin calibrar

    with pytest.raises(ValueError):
        cp.calibrate(X, np.zeros(len(X), dtype=int))   # una sola clase

    cp.calibrate(X, y)
    with pytest.raises(ValueError):
        cp.prediction_sets(X, alpha=1.5)
