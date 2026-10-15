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

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
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


MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 300
PATIENCE = 15
VAL_FRACTION = 0.15


def _fingerprint(*arrays, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def train_one(
    activation: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_epochs: int = MAX_EPOCHS, batch_size: int = 256, lr: float = 1e-3,
    min_epochs: int = MIN_EPOCHS, patience: int = PATIENCE,
    checkpoint_path: Path | None = None, seed: int = RANDOM_STATE,
) -> tuple[DefaultMLP, list[dict]]:
    """Entrena al menos ``min_epochs`` epocas (o ``n_epochs`` si es menor);
    desde ahi early stopping cuando la loss de validacion no mejora en
    ``patience`` epocas, con restauracion de la mejor epoca (``is_best`` en
    el historial). Con ``checkpoint_path`` el estado completo se guarda cada
    epoca y una corrida interrumpida retoma donde quedo."""
    torch.manual_seed(seed)
    model = DefaultMLP(n_features=X_train.shape[1], activation=activation)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    generator = torch.Generator()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    X_val_t, y_val_t = _to_tensor(X_val), _to_tensor(y_val)

    fp = _fingerprint(X_train, y_train, X_val, y_val, activation=activation, n_epochs=n_epochs,
                      batch_size=batch_size, lr=lr, min_epochs=min_epochs, patience=patience, seed=seed)
    state = {"epoch": 0, "history": [], "best_val": float("inf"), "best_epoch": 0, "best_state": None,
             "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            state = ckpt["state"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state}, tmp)
        tmp.replace(path)

    while not state["done"] and state["epoch"] < n_epochs:
        generator.manual_seed(seed + state["epoch"])
        torch.manual_seed(seed + 10_000 + state["epoch"])
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        state["epoch"] += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_auc = evaluate(y_val, torch.sigmoid(val_logits).numpy())["auc"]
        scheduler.step(val_loss)
        state["history"].append({
            "epoch": state["epoch"],
            "train_loss": epoch_loss / n_batches,
            "val_loss": val_loss,
            "val_auc": val_auc,
        })
        if val_loss < state["best_val"] - 1e-7:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, n_epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
        save_checkpoint()

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    history = [dict(row, is_best=row["epoch"] == state["best_epoch"]) for row in state["history"]]
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

    # Validacion = 15% estratificado del train, nunca el test: el early stopping y la eleccion de
    # activacion miran esta particion, y el test se evalua una sola vez por modelo ya entrenado.
    fit_idx, val_idx = train_test_split(np.arange(len(train_df)), test_size=VAL_FRACTION,
                                        stratify=y_train, random_state=RANDOM_STATE)
    means, stds = X_train_full.iloc[fit_idx].mean(), X_train_full.iloc[fit_idx].std().replace(0, 1)
    X_train_norm = ((X_train_full - means) / stds).to_numpy(dtype=np.float32)
    X_test_norm = ((X_test - means) / stds).to_numpy(dtype=np.float32)
    X_fit, y_fit = X_train_norm[fit_idx], y_train[fit_idx]
    X_val, y_val = X_train_norm[val_idx], y_train[val_idx]

    results, histories, models, val_auc = {}, {}, {}, {}
    for activation in ACTIVATIONS:
        model, history = train_one(activation, X_fit, y_fit, X_val, y_val,
                                   checkpoint_path=out_dir / "checkpoints" / f"mlp_{activation}.ckpt")
        best_row = next(row for row in history if row["is_best"])
        val_auc[activation] = best_row["val_auc"]
        print(f"  {activation:6s} mejor epoca {best_row['epoch']} de {len(history)}, "
              f"val_loss {best_row['val_loss']:.5f}, val_auc {best_row['val_auc']:.4f}")
        with torch.no_grad():
            model.eval()
            y_proba_test = torch.sigmoid(model(_to_tensor(X_test_norm))).numpy()
        results[activation] = evaluate(y_test, y_proba_test)
        histories[activation] = history
        models[activation] = model

    best_activation = max(val_auc, key=val_auc.get)  # elegida en validacion, no en test
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
