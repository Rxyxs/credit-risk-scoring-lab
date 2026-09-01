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
