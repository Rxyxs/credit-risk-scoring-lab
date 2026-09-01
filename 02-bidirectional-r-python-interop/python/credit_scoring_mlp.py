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


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    activation_name: str,
    n_epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> CreditMLP:
    torch.manual_seed(SEED)
    activation_cls = ACTIVATIONS[activation_name]
    model = CreditMLP(n_features=X_train.shape[1], activation=activation_cls)

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    loss_fn = FocalBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

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

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    y_train_arr = y_train.to_numpy().astype(np.float32)
    y_test_arr = y_test.to_numpy().astype(np.float32)

    results = []
    models = {}
    scores = {}
    for activation_name in ACTIVATIONS:
        model = train_mlp(X_train_scaled, y_train_arr, activation_name)
        score = predict_proba(model, X_test_scaled)
        results.append(evaluate(y_test_arr, score, activation_name))
        models[activation_name] = model
        scores[activation_name] = score

    best_name = max(results, key=lambda r: r["AUC_ROC"])["modelo"].replace("MLP-", "")

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
