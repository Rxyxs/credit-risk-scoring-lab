"""Tercer enfoque de modelado de credit scoring: un MLP en PyTorch entrenado sobre
el mismo dataset/split que credit_scoring.py (Regresión Logística y XGBoost), con
una loss ponderada por clase (focal-BCE) para el desbalance de default (~12%) y
una comparación explícita de tres funciones de activación -- ReLU, GELU y Swish
(SiLU) -- en la misma arquitectura, mismos hiperparámetros de entrenamiento, misma
semilla. El objetivo no es "ganarle" a XGBoost sino completar la triada clásica de
riesgo crediticio: baseline interpretable (LogReg) + ensamble de árboles (XGBoost)
+ deep learning (MLP), reportando AUC/KS/Gini de los tres bajo el mismo protocolo.
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
from torch.utils.data import DataLoader, TensorDataset

from python.credit_scoring import gini_coefficient, ks_statistic, prepare_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TABLES_DIR = ROOT_DIR / "output" / "tables"
MODELS_DIR = ROOT_DIR / "output" / "models"

SEED = 42
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Swish": nn.SiLU,  # SiLU == Swish (x * sigmoid(x))
}


class FocalBCELoss(nn.Module):
    """Loss custom: BCE ponderada + término focal (Lin et al. 2017) que reduce el
    peso de ejemplos ya bien clasificados y refuerza los difíciles -- útil aquí
    porque el ~12% de defaults hace que la BCE plana sea dominada por la clase
    mayoritaria (no-default), igual que el scale_pos_weight usado en XGBoost."""

    def __init__(self, pos_weight: float, gamma: float = 2.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        weight = targets * self.pos_weight + (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (weight * focal_term * bce).mean()


class CreditMLP(nn.Module):
    """MLP de 2 capas ocultas -- misma capacidad para las 3 activaciones comparadas."""

    def __init__(self, n_features: int, activation: type[nn.Module], hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            activation(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            activation(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


MIN_EPOCHS = 100  # ninguna red se detiene antes de 100 épocas
MAX_EPOCHS = 300
PATIENCE = 15
VAL_FRACTION = 0.15
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"


def _fingerprint(*arrays, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    activation_name: str,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    n_epochs: int = MAX_EPOCHS,
    batch_size: int = 64,
    lr: float = 1e-3,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: Path | None = None,
    return_history: bool = False,
):
    """Con validación: al menos ``min_epochs`` épocas (o ``n_epochs`` si es
    menor) y desde ahí early stopping cuando la loss de validación no mejora en
    ``patience`` épocas, restaurando la mejor época. Sin validación entrena
    ``n_epochs`` completas. Con ``checkpoint_path`` el estado completo se guarda
    cada época y una corrida interrumpida retoma donde quedó."""
    torch.manual_seed(SEED)
    activation_cls = ACTIVATIONS[activation_name]
    model = CreditMLP(n_features=X_train.shape[1], activation=activation_cls)

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    loss_fn = FocalBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    )
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

    fp = _fingerprint(X_train, y_train, *( (X_val, y_val) if has_val else () ), activation=activation_name,
                      n_epochs=n_epochs, batch_size=batch_size, lr=lr, min_epochs=min_epochs, patience=patience)
    state = {"epoch": 0, "history": [], "best_val": float("inf"), "best_epoch": 0, "best_state": None,
             "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state}, tmp)
        tmp.replace(path)

    while not state["done"] and state["epoch"] < n_epochs:
        generator.manual_seed(SEED + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(SEED + 10_000 + state["epoch"])
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        state["epoch"] += 1
        row = {"epoch": state["epoch"], "train_loss": total / max(n, 1)}
        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = loss_fn(val_logits, y_val_t).item()
            row["val_loss"] = val_loss
            row["val_auc"] = float(roc_auc_score(y_val, torch.sigmoid(val_logits).numpy()))
            if val_loss < state["best_val"] - 1e-7:
                state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                             best_state=copy.deepcopy(model.state_dict()))
            else:
                state["bad_epochs"] += 1
            if state["epoch"] >= min(min_epochs, n_epochs) and state["bad_epochs"] >= patience:
                state["done"] = True
        state["history"].append(row)
        save_checkpoint()

    if has_val and state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    if return_history:
        best_epoch = state["best_epoch"] if has_val else state["epoch"]
        return model, [dict(r, is_best=r["epoch"] == best_epoch) for r in state["history"]]
    return model


@torch.no_grad()
def predict_proba(model: CreditMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def evaluate(y_true: np.ndarray, y_score: np.ndarray, activation_name: str) -> dict:
    auc = roc_auc_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)
    gini = gini_coefficient(y_true, y_score)
    return {
        "modelo": f"MLP-{activation_name}",
        "AUC_ROC": round(float(auc), 4),
        "KS": round(float(ks), 4),
        "Gini": round(float(gini), 4),
    }


def run_mlp_activation_comparison() -> dict:
    """Entrena el mismo MLP con ReLU/GELU/Swish sobre el mismo split usado por
    credit_scoring.py y devuelve métricas comparables + el mejor modelo entrenado."""
    clean_df = pd.read_csv(DATA_DIR / "clean_credit_applications.csv")
    X, y = prepare_dataset(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # Validación = 15% estratificado del train, nunca el test: el early stopping y la elección de
    # activación miran esta partición; el test se evalúa una sola vez por modelo ya entrenado.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRACTION, random_state=SEED, stratify=y_train
    )
    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    y_fit_arr = y_fit.to_numpy().astype(np.float32)
    y_val_arr = y_val.to_numpy().astype(np.float32)
    y_test_arr = y_test.to_numpy().astype(np.float32)

    results = []
    models = {}
    scores = {}
    val_auc = {}
    for activation_name in ACTIVATIONS:
        model, history = train_mlp(
            X_fit_scaled, y_fit_arr, activation_name, X_val=X_val_scaled, y_val=y_val_arr,
            checkpoint_path=CHECKPOINT_DIR / f"mlp_{activation_name.lower()}.ckpt", return_history=True,
        )
        best = next(r for r in history if r["is_best"])
        val_auc[activation_name] = best["val_auc"]
        print(f"MLP-{activation_name}: mejor época {best['epoch']} de {len(history)}, "
              f"val_loss {best['val_loss']:.4f}, val AUC {best['val_auc']:.4f}")
        score = predict_proba(model, X_test_scaled)
        results.append(evaluate(y_test_arr, score, activation_name))
        models[activation_name] = model
        scores[activation_name] = score

    best_name = max(val_auc, key=val_auc.get)  # elegida en validación, no en test

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(models[best_name].state_dict(), MODELS_DIR / f"mlp_{best_name.lower()}_state_dict.pt")

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(TABLES_DIR / "credit_scoring_mlp_activation_comparison.csv", index=False)

    return {
        "results": results,
        "models": models,
        "scores": scores,
        "best_activation": best_name,
        "y_test": y_test_arr,
        "X_test_scaled": X_test_scaled,
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    output = run_mlp_activation_comparison()
    for r in output["results"]:
        print(f"{r['modelo']}: AUC={r['AUC_ROC']}  KS={r['KS']}  Gini={r['Gini']}")
    print(f"\nMejor activación: {output['best_activation']}")
