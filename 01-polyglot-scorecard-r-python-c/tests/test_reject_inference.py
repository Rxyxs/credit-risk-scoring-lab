import numpy as np

from src.reject_inference import hard_cutoff_inference, parceling


def test_hard_cutoff_produces_expected_bad_rate():
    rng = np.random.default_rng(0)
    pd_rejects = rng.uniform(0, 1, size=1000)
    labels = hard_cutoff_inference(pd_rejects, expected_bad_rate=0.2)
    assert abs(labels.mean() - 0.2) < 0.02


def test_parceling_hard_returns_binary_labels():
    rng = np.random.default_rng(0)
    pd_approved = rng.uniform(0, 1, size=2000)
    y_approved = (rng.random(2000) < pd_approved * 0.3).astype(int)
    pd_rejects = rng.uniform(0, 1, size=500)

    labels, weights = parceling(pd_approved, y_approved, pd_rejects, method="hard")
    assert weights is None
    assert set(np.unique(labels)).issubset({0, 1})
    assert len(labels) == 500


def test_parceling_soft_returns_fractional_weights():
    rng = np.random.default_rng(0)
    pd_approved = rng.uniform(0, 1, size=2000)
    y_approved = (rng.random(2000) < pd_approved * 0.3).astype(int)
    pd_rejects = rng.uniform(0, 1, size=500)

    weights, _ = parceling(pd_approved, y_approved, pd_rejects, method="soft")
    assert ((weights >= 0.0) & (weights <= 1.0)).all()
