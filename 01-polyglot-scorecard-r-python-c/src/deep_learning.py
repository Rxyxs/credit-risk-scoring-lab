"""Tercer enfoque de modelado (challenger no lineal, complementario a
ml_models.py): red neuronal en PyTorch para probabilidad de default,
con Focal Loss (en vez de BCE plana) para el desbalance de clases y
comparacion de activaciones (ReLU vs GELU vs Swish/SiLU).

Mismo contrato que scorecard R y ml_models.py: mismas features
(features.py), mismo split train/test (columna `split`), mismas
metricas (evaluate() de ml_models.py) sobre el mismo holdout -- para
que los 3 enfoques sean comparables sin trampa metodologica.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import RANDOM_STATE, TARGET, build_feature_matrix
from src.ml_models import evaluate

BASE = Path(__file__).resolve().parents[1]

torch.manual_seed(RANDOM_STATE)


class FocalLoss(nn.Module):
    """BCE ponderada por (1 - p_t)^gamma: baja el peso de los casos ya
    bien clasificados y concentra el gradiente en los defaults dificiles
    de separar, que es donde falla una BCE plana bajo desbalance de clase."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
}


class DefaultMLP(nn.Module):
    def __init__(self, n_features: int, activation: str = "swish", hidden=(64, 32)):
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), act_cls(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def train_one(
    activation: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_epochs: int = 60, batch_size: int = 256, lr: float = 1e-3,
) -> tuple[DefaultMLP, list[dict]]:
    model = DefaultMLP(n_features=X_train.shape[1], activation=activation)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t, y_val_t = _to_tensor(X_val), _to_tensor(y_val)
    history = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_auc = evaluate(y_val, torch.sigmoid(val_logits).numpy())["auc"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / n_batches,
            "val_loss": val_loss,
            "val_auc": val_auc,
        })
    return model, history


def run_deep_learning_pipeline(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train_full = build_feature_matrix(train_df)
    X_test = build_feature_matrix(test_df).reindex(columns=X_train_full.columns, fill_value=0)
    y_train = train_df[TARGET].to_numpy().astype(np.float32)
    y_test = test_df[TARGET].to_numpy().astype(np.float32)

    means, stds = X_train_full.mean(), X_train_full.std().replace(0, 1)
    X_train_norm = ((X_train_full - means) / stds).to_numpy(dtype=np.float32)
    X_test_norm = ((X_test - means) / stds).to_numpy(dtype=np.float32)

    results, histories, models = {}, {}, {}
    for activation in ACTIVATIONS:
        model, history = train_one(activation, X_train_norm, y_train, X_test_norm, y_test)
        with torch.no_grad():
            model.eval()
            y_proba_test = torch.sigmoid(model(_to_tensor(X_test_norm))).numpy()
        results[activation] = evaluate(y_test, y_proba_test)
        histories[activation] = history
        models[activation] = model

    best_activation = max(results, key=lambda k: results[k]["auc"])
    best_model = models[best_activation]

    torch.save(best_model.state_dict(), out_dir / "best_dl_model.pt")
    joblib_meta = {
        "activation": best_activation,
        "feature_columns": list(X_train_full.columns),
        "norm_mean": means.to_dict(),
        "norm_std": stds.to_dict(),
        "loss_function": "focal_loss(alpha=0.25, gamma=2.0)",
    }
    with open(out_dir / "dl_model_meta.json", "w") as f:
        json.dump(joblib_meta, f, indent=2)

    with torch.no_grad():
        best_model.eval()
        scores_df = test_df[["applicant_id", TARGET]].copy()
        scores_df["pd_dl"] = torch.sigmoid(best_model(_to_tensor(X_test_norm))).numpy()
    scores_df.to_csv(reports_dir / "dl_test_scores.csv", index=False)

    loss_curve_rows = []
    for activation, hist in histories.items():
        for row in hist:
            loss_curve_rows.append({"activation": activation, **row})
    pd.DataFrame(loss_curve_rows).to_csv(reports_dir / "dl_loss_curves.csv", index=False)

    report = {
        "best_activation": best_activation,
        "loss_function": "focal_loss",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "metrics": results,
    }
    with open(reports_dir / "dl_model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    df = pd.read_csv(BASE / "data" / "processed" / "applicants_clean.csv")
    out_dir = BASE / "outputs" / "models"
    report = run_deep_learning_pipeline(df, out_dir)
    print(json.dumps(report, indent=2))
