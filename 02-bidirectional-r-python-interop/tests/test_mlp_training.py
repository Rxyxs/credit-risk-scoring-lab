import numpy as np
import pytest
import torch

from python.credit_scoring_mlp import train_mlp


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 1.0).astype(np.float32)
    cut = int(n * 0.8)
    return X[:cut], y[:cut], X[cut:], y[cut:]


def test_early_stopping_respects_minimum_epochs_and_marks_best_epoch():
    X, y, Xv, yv = _toy()
    _model, history = train_mlp(X, y, "ReLU", X_val=Xv, y_val=yv, n_epochs=60, min_epochs=20,
                                patience=2, return_history=True)
    assert len(history) >= 20
    best = [h for h in history if h["is_best"]]
    assert len(best) == 1 and best[0]["val_loss"] == min(h["val_loss"] for h in history)


def test_interrupted_training_resumes_from_checkpoint(tmp_path, monkeypatch):
    X, y, Xv, yv = _toy(seed=1)
    kwargs = dict(X_val=Xv, y_val=yv, n_epochs=6, min_epochs=6, patience=100, return_history=True)
    reference, ref_history = train_mlp(X, y, "GELU", **kwargs)

    ckpt = tmp_path / "gelu.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}

    def flaky_step(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] > 5 * 3:  # 320 filas / 64 = 5 batches por época: corta en la época 4
            raise KeyboardInterrupt
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        train_mlp(X, y, "GELU", checkpoint_path=ckpt, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)

    resumed, history = train_mlp(X, y, "GELU", checkpoint_path=ckpt, **kwargs)
    assert [h["val_loss"] for h in history] == pytest.approx([h["val_loss"] for h in ref_history], rel=1e-5)
    for a, b in zip(reference.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-6)
