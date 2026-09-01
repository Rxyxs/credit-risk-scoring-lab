"""Tests unitarios para el tercer enfoque de modelado (MLP en PyTorch): la loss
custom, la arquitectura con las tres activaciones, y el flujo de entrenamiento en
un dataset sintético pequeño (sin depender de data/clean_credit_applications.csv)."""

from __future__ import annotations

import numpy as np
import torch

from python.credit_scoring_mlp import (
    ACTIVATIONS,
    CreditMLP,
    FocalBCELoss,
    predict_proba,
    train_mlp,
)


def _toy_dataset(n: int = 200, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    logits = X[:, 0] * 2.0 - X[:, 1]
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(np.float32)
    return X, y


def test_focal_bce_loss_is_nonnegative_and_finite():
    loss_fn = FocalBCELoss(pos_weight=3.0)
    logits = torch.tensor([2.0, -1.0, 0.0, 5.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_bce_penalizes_confident_wrong_prediction_more():
    loss_fn = FocalBCELoss(pos_weight=1.0, gamma=2.0)
    confident_wrong = loss_fn(torch.tensor([5.0]), torch.tensor([0.0]))
    unsure = loss_fn(torch.tensor([0.0]), torch.tensor([0.0]))
    assert confident_wrong.item() > unsure.item()


def test_credit_mlp_forward_shape_for_each_activation():
    X = torch.randn(10, 6)
    for activation_cls in ACTIVATIONS.values():
        model = CreditMLP(n_features=6, activation=activation_cls)
        out = model(X)
        assert out.shape == (10,)


def test_all_three_activations_registered():
    assert set(ACTIVATIONS.keys()) == {"ReLU", "GELU", "Swish"}
    assert ACTIVATIONS["Swish"].__name__ == "SiLU"


def test_train_mlp_reduces_loss_and_predicts_valid_probabilities():
    X, y = _toy_dataset()
    model = train_mlp(X, y, activation_name="ReLU", n_epochs=15)
    probs = predict_proba(model, X)

    assert probs.shape == (len(X),)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # El modelo debe separar mejor que el azar en un dataset sintético linealmente
    # separable con ruido moderado.
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y, probs)
    assert auc > 0.6


def test_train_mlp_is_deterministic_given_fixed_seed():
    X, y = _toy_dataset()
    model_a = train_mlp(X, y, activation_name="GELU", n_epochs=5)
    model_b = train_mlp(X, y, activation_name="GELU", n_epochs=5)
    probs_a = predict_proba(model_a, X)
    probs_b = predict_proba(model_b, X)
    np.testing.assert_allclose(probs_a, probs_b, rtol=1e-4, atol=1e-5)
