"""Tests de FedAvg.

Las dos verificaciones centrales son identidades algebraicas exactas, no
aproximaciones: si la implementacion esta bien, tienen que calzar a
precision de maquina, no "aproximadamente".
"""

from __future__ import annotations

import numpy as np
import pytest

from src.federated import (
    FedAvgLogisticRegression, _con_intercepto, gradiente_medio,
    gradient_descent_centralizado, local_update, sigmoide,
)


def _datos(n=2000, seed=0, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(0, 0.6, size=p + 1)
    y = (rng.random(n) < sigmoide(beta[0] + X @ beta[1:])).astype(float)
    return X, y, beta


# --------------------------------------------------------------- gradiente
def test_gradiente_medio_contra_diferencias_finitas():
    X, y, beta = _datos(n=500)
    Xa = _con_intercepto(X)
    w = np.array([0.1, -0.2, 0.3, 0.05])

    def neg_ll_medio(w):
        p = np.clip(sigmoide(Xa @ w), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    grad = gradiente_medio(w, Xa, y)
    h = 1e-6
    grad_fd = np.zeros_like(w)
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += h
        wm[i] -= h
        grad_fd[i] = (neg_ll_medio(wp) - neg_ll_medio(wm)) / (2 * h)
    assert np.allclose(grad, grad_fd, atol=1e-5)


# ---------------------------------------------------- identidades exactas
def test_un_solo_cliente_fedavg_es_gd_centralizado():
    """Con un cliente, promediar no hace nada: FedAvg(E, R) tiene que dar
    EXACTAMENTE lo mismo que GD comun por R*E epocas seguidas."""
    X, y, _ = _datos(n=1500, seed=1)
    for E, R in [(3, 8), (1, 20), (10, 4)]:
        fed = FedAvgLogisticRegression(lr=0.4, local_epochs=E, n_rounds=R, seed=2).fit([X], [y])
        w_gd = gradient_descent_centralizado(
            np.zeros(X.shape[1] + 1), _con_intercepto(X), y, lr=0.4, epochs=E * R
        )
        assert np.allclose(fed.coef_, w_gd, atol=1e-10)


def test_e_igual_a_uno_multicliente_es_gd_sobre_el_pool():
    """Con un paso local por ronda, el promedio ponderado de gradientes es
    algebraicamente identico al gradiente sobre los datos agrupados --
    FedAvg con E=1 tiene que coincidir con GD centralizado sobre el pool,
    sin importar que tan desiguales sean los tamanos de los clientes."""
    rng = np.random.default_rng(3)
    X, y, _ = _datos(n=3000, seed=3)
    cortes = [0, 400, 1300, 1900, 3000]      # tamanos bien desiguales
    Xs = [X[cortes[i]:cortes[i + 1]] for i in range(4)]
    ys = [y[cortes[i]:cortes[i + 1]] for i in range(4)]

    fed = FedAvgLogisticRegression(lr=0.5, local_epochs=1, n_rounds=30, seed=4).fit(Xs, ys)
    w_gd = gradient_descent_centralizado(
        np.zeros(X.shape[1] + 1), _con_intercepto(X), y, lr=0.5, epochs=30
    )
    assert np.allclose(fed.coef_, w_gd, atol=1e-9)


def test_clientes_identicos_dan_el_mismo_resultado_que_uno_solo():
    """Un caso particular de la identidad anterior: si todos los clientes
    tienen exactamente los mismos datos, FedAvg con cualquier E tiene que
    coincidir con GD centralizado sobre esos mismos datos -- promediar
    K copias identicas del mismo update no cambia nada."""
    X, y, _ = _datos(n=600, seed=5)
    fed = FedAvgLogisticRegression(lr=0.5, local_epochs=4, n_rounds=15, seed=6) \
        .fit([X, X, X], [y, y, y])
    w_gd = gradient_descent_centralizado(
        np.zeros(X.shape[1] + 1), _con_intercepto(X), y, lr=0.5, epochs=4 * 15
    )
    assert np.allclose(fed.coef_, w_gd, atol=1e-9)


# ------------------------------------------------------------- comportamiento
def test_fedavg_converge_a_una_buena_solucion():
    X, y, beta = _datos(n=6000, seed=7)
    Xs = [X[:2000], X[2000:4000], X[4000:]]
    ys = [y[:2000], y[2000:4000], y[4000:]]
    fed = FedAvgLogisticRegression(lr=0.5, local_epochs=3, n_rounds=60, seed=8).fit(Xs, ys)

    from sklearn.metrics import roc_auc_score
    p = fed.predict_proba(X)[:, 1]
    assert roc_auc_score(y, p) > 0.70


def test_mas_clientes_no_cambian_el_largo_del_vector_de_coeficientes():
    X, y, _ = _datos(n=900, p=4, seed=9)
    fed = FedAvgLogisticRegression(n_rounds=5).fit([X[:300], X[300:600], X[600:]],
                                                    [y[:300], y[300:600], y[600:]])
    assert fed.coef_.shape == (5,)


def test_deriva_de_clientes_es_positiva_y_decrece_hacia_el_final_en_iid():
    """En datos IID, todos los clientes optimizan (en esperanza) el mismo
    objetivo: la deriva entre el modelo local y el global antes de agregar
    tiene que achicarse a medida que el modelo global converge."""
    rng = np.random.default_rng(10)
    X, y, _ = _datos(n=6000, seed=10)
    idx = rng.permutation(len(X))
    Xs = [X[idx[i::4]] for i in range(4)]
    ys = [y[idx[i::4]] for i in range(4)]

    fed = FedAvgLogisticRegression(lr=0.4, local_epochs=3, n_rounds=25, seed=11).fit(Xs, ys)
    deriva = np.array(fed.historia_.deriva_media_clientes)
    assert deriva[0] > 0
    assert deriva[-5:].mean() < deriva[:5].mean()


def test_deltas_primera_ronda_tienen_el_largo_correcto():
    X, y, _ = _datos(n=900, p=3, seed=12)
    Xs = [X[:300], X[300:600], X[600:]]
    ys = [y[:300], y[300:600], y[600:]]
    fed = FedAvgLogisticRegression(local_epochs=2, n_rounds=1)
    deltas = fed.deltas_primera_ronda(Xs, ys)
    assert deltas.shape == (3, 4)


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        FedAvgLogisticRegression(local_epochs=0)
    with pytest.raises(ValueError):
        FedAvgLogisticRegression(n_rounds=0)
    X, y, _ = _datos(n=100)
    with pytest.raises(ValueError):
        FedAvgLogisticRegression().fit([X], [y, y])
    with pytest.raises(ValueError):
        FedAvgLogisticRegression().fit([], [])


def test_predict_proba_da_probabilidades_validas():
    X, y, _ = _datos(n=500, seed=13)
    fed = FedAvgLogisticRegression(n_rounds=10).fit([X[:250], X[250:]], [y[:250], y[250:]])
    p = fed.predict_proba(X)
    assert p.shape == (500, 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all((p >= 0) & (p <= 1))
