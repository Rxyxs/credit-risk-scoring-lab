import json

import numpy as np
import torch

from src.cleaning import run_cleaning_pipeline
from src.data_generator import generate_applicants
from src.deep_learning import ACTIVATIONS, DefaultMLP, FocalLoss, run_deep_learning_pipeline


def test_focal_loss_is_lower_for_confident_correct_predictions():
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    y = torch.tensor([1.0, 1.0])
    confident_correct_logits = torch.tensor([5.0, 5.0])
    unsure_logits = torch.tensor([0.0, 0.0])
    assert criterion(confident_correct_logits, y).item() < criterion(unsure_logits, y).item()


def test_default_mlp_forward_shape_matches_batch_size():
    for activation in ACTIVATIONS:
        model = DefaultMLP(n_features=10, activation=activation)
        model.eval()
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16,)


def test_run_deep_learning_pipeline_end_to_end(tmp_path):
    raw = generate_applicants(n=2000, seed=7)
    cleaned = run_cleaning_pipeline(raw)

    report = run_deep_learning_pipeline(cleaned, tmp_path / "models")

    assert report["best_activation"] in ACTIVATIONS
    assert (tmp_path / "models" / "best_dl_model.pt").exists()
    assert (tmp_path / "models" / "dl_model_meta.json").exists()
    assert (tmp_path / "reports" / "dl_model_comparison.json").exists()
    assert (tmp_path / "reports" / "dl_loss_curves.csv").exists()

    for activation, m in report["metrics"].items():
        assert 0.0 <= m["auc"] <= 1.0

    meta = json.loads((tmp_path / "models" / "dl_model_meta.json").read_text())
    assert meta["loss_function"].startswith("focal_loss")


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n) > 1.0).astype(np.float32)
    return X[: int(n * 0.8)], y[: int(n * 0.8)], X[int(n * 0.8):], y[int(n * 0.8):]


def test_early_stopping_respects_minimum_epochs_and_marks_best_epoch():
    from src.deep_learning import train_one

    X, y, Xv, yv = _toy()
    _model, history = train_one("relu", X, y, Xv, yv, n_epochs=60, min_epochs=20, patience=2, batch_size=64)
    assert len(history) >= 20
    best = [h for h in history if h["is_best"]]
    assert len(best) == 1 and best[0]["val_loss"] == min(h["val_loss"] for h in history)


def test_interrupted_training_resumes_from_checkpoint(tmp_path, monkeypatch):
    import pytest

    from src.deep_learning import train_one

    X, y, Xv, yv = _toy(seed=1)
    kwargs = dict(n_epochs=6, min_epochs=6, patience=100, batch_size=64)
    reference, ref_history = train_one("gelu", X, y, Xv, yv, **kwargs)

    ckpt = tmp_path / "gelu.ckpt"
    real_step = torch.optim.AdamW.step
    calls = {"n": 0}

    def flaky_step(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] > 5 * 3:  # 320 filas / 64 = 5 batches por epoca: corta en la epoca 4
            raise KeyboardInterrupt
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.AdamW, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        train_one("gelu", X, y, Xv, yv, checkpoint_path=ckpt, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", real_step)

    resumed, history = train_one("gelu", X, y, Xv, yv, checkpoint_path=ckpt, **kwargs)
    assert [h["val_loss"] for h in history] == pytest.approx([h["val_loss"] for h in ref_history], rel=1e-5)
    for a, b in zip(reference.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-6)
