"""Tests del modelo de Cox implementado desde cero.

El foco esta en la mecanica que es facil de equivocar y dificil de notar:
el gradiente/hessiano analiticos, el tratamiento de empates, y que las
predicciones de supervivencia sean funciones validas.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cox_ph import CoxPH


def _datos_ph(n=1500, seed=7, beta=(0.8, -0.5)):
    """Simula tiempos exponenciales con hazards proporcionales conocidos."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta)))
    eta = X @ np.array(beta)
    t = rng.exponential(scale=1.0 / (0.05 * np.exp(eta)))
    censura = rng.exponential(scale=40.0, size=n)
    duracion = np.minimum(t, censura)
    evento = (t <= censura).astype(int)
    return X, duracion, evento


def test_gradiente_analitico_coincide_con_diferencias_finitas():
    X, dur, ev = _datos_ph(n=400)
    m = CoxPH(ties="efron")
    Xs, ds, es = m._prepare(X, dur, ev)
    beta = np.array([0.3, -0.2])

    _, grad, _ = m._loglik_grad_hess(beta, Xs, ds, es)
    num = np.zeros_like(beta)
    for j in range(beta.size):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += 1e-6
        bm[j] -= 1e-6
        num[j] = (
            m._loglik_grad_hess(bp, Xs, ds, es)[0] - m._loglik_grad_hess(bm, Xs, ds, es)[0]
        ) / 2e-6
    assert np.allclose(grad, num, atol=1e-4)


def test_hessiano_analitico_coincide_con_diferencias_finitas():
    X, dur, ev = _datos_ph(n=400)
    m = CoxPH(ties="breslow")
    Xs, ds, es = m._prepare(X, dur, ev)
    beta = np.array([0.25, -0.15])

    _, _, hess = m._loglik_grad_hess(beta, Xs, ds, es)
    num = np.zeros((2, 2))
    for j in range(2):
        bp, bm = beta.copy(), beta.copy()
        bp[j] += 1e-6
        bm[j] -= 1e-6
        num[:, j] = (
            m._loglik_grad_hess(bp, Xs, ds, es)[1] - m._loglik_grad_hess(bm, Xs, ds, es)[1]
        ) / 2e-6
    assert np.allclose(hess, num, atol=1e-3)


def test_recupera_coeficientes_conocidos():
    beta_true = (0.8, -0.5)
    X, dur, ev = _datos_ph(n=4000, beta=beta_true)
    r = CoxPH(ties="efron").fit(X, dur, ev, feature_names=["x1", "x2"])
    assert r.converged
    assert np.allclose(r.coef, beta_true, atol=0.10)
    # los dos coeficientes deben ser claramente significativos
    assert np.all(r.p_values < 1e-6)


def test_sin_empates_breslow_y_efron_coinciden():
    """Las dos aproximaciones solo difieren cuando hay tiempos repetidos."""
    X, dur, ev = _datos_ph(n=800)
    assert len(np.unique(dur[ev == 1])) == int(ev.sum())  # tiempos continuos, sin empates
    r_b = CoxPH(ties="breslow").fit(X, dur, ev)
    r_e = CoxPH(ties="efron").fit(X, dur, ev)
    assert np.allclose(r_b.coef, r_e.coef, atol=1e-8)


def test_con_empates_breslow_atenua_hacia_cero():
    """Al redondear a meses aparecen empates masivos y Breslow encoge |beta|."""
    X, dur, ev = _datos_ph(n=3000, beta=(1.2, -0.9))
    dur_mes = np.ceil(dur / 2.0)                       # discretiza -> empates
    r_b = CoxPH(ties="breslow").fit(X, dur_mes, ev)
    r_e = CoxPH(ties="efron").fit(X, dur_mes, ev)
    assert np.all(np.abs(r_b.coef) < np.abs(r_e.coef))


def test_supervivencia_es_monotona_y_acotada():
    X, dur, ev = _datos_ph(n=1000)
    m = CoxPH(ties="efron")
    m.fit(X, dur, ev)
    t = np.linspace(0.5, 30, 40)
    S = m.predict_survival(X[:50], t)
    assert S.shape == (50, 40)
    assert np.all((S >= 0) & (S <= 1))
    assert np.all(np.diff(S, axis=1) <= 1e-12)
    assert np.all(np.diff(m.baseline_cumhazard_) >= 0)


def test_pd_acumulada_complementa_supervivencia():
    X, dur, ev = _datos_ph(n=600)
    m = CoxPH()
    m.fit(X, dur, ev)
    t = np.array([5.0, 12.0])
    assert np.allclose(
        m.predict_cumulative_default(X[:20], t), 1.0 - m.predict_survival(X[:20], t)
    )


def test_ties_invalido_y_datos_inconsistentes_fallan():
    with pytest.raises(ValueError):
        CoxPH(ties="exacto")
    X, dur, ev = _datos_ph(n=100)
    with pytest.raises(ValueError):
        CoxPH().fit(X, dur[:50], ev)
    with pytest.raises(ValueError):
        CoxPH().fit(X, dur, np.zeros_like(ev))       # sin eventos observados


def test_ph_test_detecta_efecto_variable_en_el_tiempo():
    """Con un efecto que decae en el tiempo, el test debe marcar violacion."""
    rng = np.random.default_rng(11)
    n, meses = 6000, 24
    x_ph = rng.normal(size=n)
    x_tv = rng.binomial(1, 0.5, size=n).astype(float)

    duracion = np.full(n, meses, dtype=float)
    evento = np.zeros(n, dtype=int)
    activo = np.ones(n, dtype=bool)
    for t in range(1, meses + 1):
        idx = np.where(activo)[0]
        log_h = -3.4 + 0.6 * x_ph[idx] + (1.4 * np.exp(-t / 6.0)) * x_tv[idx]
        cae = rng.random(idx.size) < (1 - np.exp(-np.exp(log_h)))
        duracion[idx[cae]] = t
        evento[idx[cae]] = 1
        activo[idx[cae]] = False

    m = CoxPH(ties="efron")
    m.fit(np.column_stack([x_ph, x_tv]), duracion, evento, feature_names=["ph", "tv"])
    res = {d["feature"]: d for d in m.ph_test()}
    assert res["tv"]["viola_ph"] is True
    assert res["ph"]["viola_ph"] is False
