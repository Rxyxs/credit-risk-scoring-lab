"""Tests del muestreador jerarquico y de las tres configuraciones de pooling."""

from __future__ import annotations

import numpy as np
import pytest

from src.hierarchical_logit import (
    HierarchicalLogit, posterior_pd, resumen_pd,
)


def _datos(n=2500, n_seg=8, seed=5, beta=(-1.6, 0.9, -0.6), tau=0.6, tam_desigual=True):
    """Simula datos jerarquicos con efectos de segmento conocidos."""
    rng = np.random.default_rng(seed)
    if tam_desigual:
        p = np.linspace(1.0, 8.0, n_seg)
        p = p / p.sum()
    else:
        p = np.full(n_seg, 1.0 / n_seg)
    seg = rng.choice(n_seg, size=n, p=p)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, len(beta) - 1))])
    efectos = rng.normal(0, tau, size=n_seg)
    efectos -= efectos.mean()
    eta = X @ np.array(beta) + efectos[seg]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y, seg, np.array(beta), efectos


def test_recupera_efectos_globales():
    X, y, seg, beta_true, _ = _datos(n=6000)
    draws = HierarchicalLogit("partial").fit(
        X, y, seg, 8, ["intercepto", "x1", "x2"], [f"s{j}" for j in range(8)],
        n_draws=600, n_warmup=300, n_chains=2, seed=1,
    )
    beta_post = draws.flat("beta").mean(axis=0)
    assert np.allclose(beta_post, beta_true, atol=0.15)
    assert draws.beta.shape == (2, 600, 3)


def test_recupera_tau_de_los_efectos_de_segmento():
    X, y, seg, _, _ = _datos(n=8000, n_seg=12, tau=0.8)
    draws = HierarchicalLogit("partial").fit(
        X, y, seg, 12, ["intercepto", "x1", "x2"], [f"s{j}" for j in range(12)],
        n_draws=800, n_warmup=400, n_chains=2, seed=2,
    )
    tau = draws.flat("tau")
    assert np.quantile(tau, 0.05) < 0.8 < np.quantile(tau, 0.95)


def test_pooling_completo_no_tiene_efectos_de_segmento():
    X, y, seg, _, _ = _datos(n=1500)
    draws = HierarchicalLogit("complete").fit(
        X, y, seg, 8, ["intercepto", "x1", "x2"], [f"s{j}" for j in range(8)],
        n_draws=200, n_warmup=100, n_chains=1, seed=3,
    )
    assert np.allclose(draws.b_segmento, 0.0)


def test_el_pooling_parcial_encoge_mas_que_el_ausente():
    """El shrinkage tiene que ser mayor donde hay menos datos."""
    X, y, seg, _, _ = _datos(n=3000, n_seg=8, seed=8)
    args = dict(n_draws=700, n_warmup=300, n_chains=2, seed=4)
    nombres = [f"s{j}" for j in range(8)]

    parcial = HierarchicalLogit("partial").fit(X, y, seg, 8, ["i", "x1", "x2"], nombres, **args)
    sin = HierarchicalLogit("none").fit(X, y, seg, 8, ["i", "x1", "x2"], nombres, **args)

    b_par = parcial.flat("b_segmento").mean(axis=0)
    b_sin = sin.flat("b_segmento").mean(axis=0)
    b_par = b_par - b_par.mean()
    b_sin = b_sin - b_sin.mean()

    tam = np.bincount(seg, minlength=8)
    chicos = np.argsort(tam)[:3]
    grandes = np.argsort(tam)[-3:]
    encogimiento = np.abs(b_sin) - np.abs(b_par)
    assert encogimiento[chicos].mean() > encogimiento[grandes].mean()
    assert np.abs(b_par).max() <= np.abs(b_sin).max() + 1e-9


def test_posterior_pd_forma_y_rango():
    X, y, seg, _, _ = _datos(n=1200)
    draws = HierarchicalLogit("partial").fit(
        X, y, seg, 8, ["i", "x1", "x2"], [f"s{j}" for j in range(8)],
        n_draws=300, n_warmup=150, n_chains=2, seed=6,
    )
    pd_draws = posterior_pd(draws, X[:50], seg[:50], thin=5)
    assert pd_draws.shape == (50, 2 * 300 // 5)
    assert np.all((pd_draws > 0) & (pd_draws < 1))

    r = resumen_pd(pd_draws)
    assert np.all(r["pd_q_lo"] <= r["pd_media"])
    assert np.all(r["pd_media"] <= r["pd_q_hi"])
    assert np.all(r["pd_sd"] > 0)


def test_la_incertidumbre_es_mayor_en_segmentos_chicos():
    X, y, seg, _, _ = _datos(n=4000, n_seg=8, seed=11)
    draws = HierarchicalLogit("partial").fit(
        X, y, seg, 8, ["i", "x1", "x2"], [f"s{j}" for j in range(8)],
        n_draws=800, n_warmup=300, n_chains=2, seed=7,
    )
    sd_efecto = draws.flat("b_segmento").std(axis=0)
    tam = np.bincount(seg, minlength=8)
    # correlacion negativa entre tamano del segmento e incertidumbre
    assert np.corrcoef(tam, sd_efecto)[0, 1] < -0.5


def test_validaciones_de_entrada():
    X, y, seg, _, _ = _datos(n=200)
    with pytest.raises(ValueError):
        HierarchicalLogit("pooling_magico")
    modelo = HierarchicalLogit("partial")
    with pytest.raises(ValueError):
        modelo.fit(X, y[:100], seg, 8, ["i", "x1", "x2"], [f"s{j}" for j in range(8)],
                   n_draws=10, n_warmup=5, n_chains=1)
    with pytest.raises(ValueError):
        modelo.fit(X, y, seg, 3, ["i", "x1", "x2"], ["a", "b", "c"],
                   n_draws=10, n_warmup=5, n_chains=1)
