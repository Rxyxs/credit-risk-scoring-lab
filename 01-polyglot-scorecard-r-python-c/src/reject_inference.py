"""Reject Inference: correccion del sesgo de seleccion que aparece cuando
un modelo de riesgo se entrena solo con la cartera aprobada (`known good/
bad`, KGB), porque el resultado real (default_12m) de un solicitante
rechazado nunca se observa en un banco real -- nunca recibio el credito.
Entrenar solo sobre aprobados sesga el modelo hacia la sub-poblacion menos
riesgosa del universo de solicitantes.

Como todo el dataset de este repo es sintetico, el `default_12m` de los
"rechazados" SI se conoce (lo genero `data_generator.py`) -- eso permite
algo que un banco real no puede hacer: validar honestamente si el reject
inference efectivamente reduce el sesgo, comparando contra el resultado
verdadero en la poblacion completa (aprobados + rechazados), no solo
estimarlo a ciegas. La politica de aprobacion se simula aqui (ver
`simulate_approval_decision`) y el paso de reject inference NUNCA usa el
`default_12m` real de los rechazados -- solo el score del modelo KGB -- para
que el metodo sea el mismo que se usaria en produccion.

Metodos implementados (Siddiqi, "Credit Risk Scorecards", cap. 9):
- **Hard cutoff**: el modelo KGB (entrenado solo con aprobados) scorea a
  los rechazados; a los que caen bajo un umbral de PD se les asigna
  default_12m=1 (malo) y al resto 0 (bueno) -- una asignacion binaria dura.
- **Parceling (hard)**: se agrupan los rechazados en "parcels" (deciles)
  segun su PD estimada por el modelo KGB, se calcula una tasa de malos
  objetivo por parcel (tasa de malos observada en aprobados de ese mismo
  rango de PD, multiplicada por un factor de inflacion porque la practica
  estandar asume que los rechazados son sistematicamente mas riesgosos que
  aprobados con el mismo score), y se asigna default_12m=1 a una muestra
  aleatoria de ese tamano dentro del parcel.
- **Parceling (soft)**: igual que arriba, pero en vez de asignar una
  etiqueta dura por sorteo, cada rechazado se duplica en dos filas
  ponderadas (peso = tasa de malos objetivo del parcel para la fila
  "mala", 1 - esa tasa para la fila "buena") y el modelo final se
  reentrena con `sample_weight` -- evita el ruido de un sorteo binario a
  costa de una etiqueta "difusa".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET, build_feature_matrix

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

TARGET_APPROVAL_RATE = 0.75  # a traves de la puerta -- rango tipico de aprobacion en credito de consumo
APPROVAL_STEEPNESS = 0.06    # que tan "duro" es el corte alrededor del cutoff de score
N_PARCELS = 10
INFLATION_FACTOR = 1.5       # rechazados asumidos ~1.5x mas riesgosos que aprobados del mismo parcel de PD


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    cum_bad = np.cumsum(y_sorted) / max(n_bad, 1)
    cum_good = np.cumsum(1 - y_sorted) / max(n_good, 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def load_population() -> pd.DataFrame:
    clean = pd.read_csv(DATA_DIR / "applicants_clean.csv")
    scored = pd.read_csv(DATA_DIR / "applicants_scored.csv")
    return clean.merge(scored[["applicant_id", "score"]], on="applicant_id")


def simulate_approval_decision(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.Series:
    """Politica de aprobacion simulada: probabilidad de aprobar creciente en
    el score del Champion (scorecard R), con una transicion suave (logistica)
    alrededor del cutoff en vez de un corte duro -- mas realista que un
    if/else, porque en la practica bancaria real hay overrides manuales
    cerca del punto de corte."""
    rng = np.random.default_rng(seed)
    cutoff_score = float(np.quantile(df["score"], 1 - TARGET_APPROVAL_RATE))
    p_approve = 1.0 / (1.0 + np.exp(-APPROVAL_STEEPNESS * (df["score"] - cutoff_score)))
    approved = pd.Series(rng.random(len(df)) < p_approve, index=df.index)
    return approved


def fit_logistic(X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_scaled, y, sample_weight=sample_weight)
    return model, scaler


def predict_pd(model: LogisticRegression, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(X))[:, 1]


def hard_cutoff_inference(kgb_pd_rejects: np.ndarray, expected_bad_rate: float) -> np.ndarray:
    """Asigna malo (1) al `expected_bad_rate` de rechazados con mayor PD
    estimada por el modelo KGB, bueno (0) al resto -- asignacion binaria
    dura basada en un umbral de score."""
    threshold = np.quantile(kgb_pd_rejects, 1 - expected_bad_rate)
    return (kgb_pd_rejects >= threshold).astype(int)


def parceling(
    kgb_pd_approved: np.ndarray, y_approved: np.ndarray, kgb_pd_rejects: np.ndarray,
    n_parcels: int = N_PARCELS, inflation_factor: float = INFLATION_FACTOR,
    method: str = "hard", seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Parcela por PD estimada (bins compartidos entre aprobados y
    rechazados) y calcula la tasa de malos observada entre aprobados de
    cada parcel, inflada por `inflation_factor`. Devuelve (labels, weights)
    -- weights es None para el metodo 'hard' (etiquetas 0/1 normales,
    peso implicito 1), y el vector de pesos "malo" para 'soft' (la fila
    "buena" complementaria se arma en `build_soft_parceling_training_set`)."""
    rng = np.random.default_rng(seed)
    edges = np.quantile(np.concatenate([kgb_pd_approved, kgb_pd_rejects]), np.linspace(0, 1, n_parcels + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)

    parcel_approved = np.digitize(kgb_pd_approved, edges[1:-1])
    parcel_rejects = np.digitize(kgb_pd_rejects, edges[1:-1])

    bad_rate_by_parcel = {}
    for p in np.unique(parcel_approved):
        mask = parcel_approved == p
        observed_bad_rate = y_approved[mask].mean() if mask.sum() > 0 else y_approved.mean()
        bad_rate_by_parcel[p] = float(np.clip(observed_bad_rate * inflation_factor, 0.0, 1.0))

    target_rate_rejects = np.array([bad_rate_by_parcel.get(p, y_approved.mean()) for p in parcel_rejects])

    if method == "hard":
        draws = rng.random(len(kgb_pd_rejects))
        labels = (draws < target_rate_rejects).astype(int)
        return labels, None
    else:  # soft
        return target_rate_rejects, target_rate_rejects


def build_training_sets(train_df: pd.DataFrame, approved: pd.Series, kgb_model, kgb_scaler) -> dict:
    approved_df = train_df[approved]
    rejected_df = train_df[~approved]

    X_approved = build_feature_matrix(approved_df)
    y_approved = approved_df[TARGET].to_numpy()

    X_rejects = build_feature_matrix(rejected_df).reindex(columns=X_approved.columns, fill_value=0)
    kgb_pd_rejects = predict_pd(kgb_model, kgb_scaler, X_rejects)
    kgb_pd_approved = predict_pd(kgb_model, kgb_scaler, X_approved)

    expected_bad_rate_rejects = float(np.clip(y_approved.mean() * INFLATION_FACTOR, 0.0, 1.0))
    hard_cutoff_labels = hard_cutoff_inference(kgb_pd_rejects, expected_bad_rate_rejects)

    parceling_hard_labels, _ = parceling(kgb_pd_approved, y_approved, kgb_pd_rejects, method="hard")
    parceling_soft_weights, _ = parceling(kgb_pd_approved, y_approved, kgb_pd_rejects, method="soft")

    sets = {
        "biased_approved_only": {
            "X": X_approved, "y": y_approved, "sample_weight": None,
        },
        "hard_cutoff": {
            "X": pd.concat([X_approved, X_rejects], ignore_index=True),
            "y": np.concatenate([y_approved, hard_cutoff_labels]),
            "sample_weight": None,
        },
        "parceling_hard": {
            "X": pd.concat([X_approved, X_rejects], ignore_index=True),
            "y": np.concatenate([y_approved, parceling_hard_labels]),
            "sample_weight": None,
        },
        "parceling_soft": {
            "X": pd.concat([X_approved, X_rejects, X_rejects], ignore_index=True),
            "y": np.concatenate([
                y_approved,
                np.ones(len(X_rejects)),   # copia "mala" de cada rechazado
                np.zeros(len(X_rejects)),  # copia "buena" de cada rechazado
            ]),
            "sample_weight": np.concatenate([
                np.ones(len(y_approved)),
                parceling_soft_weights,          # peso de la copia mala
                1.0 - parceling_soft_weights,    # peso de la copia buena
            ]),
        },
    }
    return sets, X_approved.columns, kgb_pd_rejects, rejected_df


def evaluate_on(model, scaler, feature_columns, df: pd.DataFrame) -> dict:
    X = build_feature_matrix(df).reindex(columns=feature_columns, fill_value=0)
    y = df[TARGET].to_numpy()
    pd_hat = predict_pd(model, scaler, X)
    auc = roc_auc_score(y, pd_hat)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks_statistic": compute_ks(y, pd_hat)}


def run_reject_inference_pipeline() -> dict:
    df = load_population()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    approved = simulate_approval_decision(train_df)
    approval_rate = float(approved.mean())

    # Modelo KGB: entrenado SOLO con aprobados -- el modelo sesgado de partida
    # que reject inference intenta corregir.
    X_approved_only = build_feature_matrix(train_df[approved])
    y_approved_only = train_df[approved][TARGET].to_numpy()
    kgb_model, kgb_scaler = fit_logistic(X_approved_only, y_approved_only)

    training_sets, feature_columns, kgb_pd_rejects, rejected_df = build_training_sets(
        train_df, approved, kgb_model, kgb_scaler
    )

    test_approved_mask = simulate_approval_decision(test_df)  # solo para el corte "monitoreo real"

    results = {}
    for method, spec in training_sets.items():
        model, scaler = fit_logistic(spec["X"], spec["y"], spec["sample_weight"])
        results[method] = {
            "n_train_rows": int(len(spec["y"])),
            "observed_bad_rate_in_training": float(np.average(spec["y"], weights=spec["sample_weight"])),
            "eval_test_approved_only": evaluate_on(model, scaler, feature_columns, test_df[test_approved_mask]),
            "eval_test_full_population": evaluate_on(model, scaler, feature_columns, test_df),
        }

    # Que tan riesgosos son realmente los rechazados de train (posible SOLO
    # porque los datos son sinteticos) -- el numero que un banco real jamas
    # podria calcular, y contra el que se valida que el reject inference no
    # se haya inventado una historia equivocada.
    true_bad_rate_rejects = float(rejected_df[TARGET].mean())
    kgb_predicted_bad_rate_rejects = float(kgb_pd_rejects.mean())

    report = {
        "approval_rate_simulated": approval_rate,
        "true_bad_rate_approved_train": float(train_df[approved][TARGET].mean()),
        "true_bad_rate_rejected_train": true_bad_rate_rejects,
        "kgb_model_predicted_mean_pd_on_rejects": kgb_predicted_bad_rate_rejects,
        "selection_bias_gap": true_bad_rate_rejects - float(train_df[approved][TARGET].mean()),
        "methods": results,
    }
    return report


if __name__ == "__main__":
    report = run_reject_inference_pipeline()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "reject_inference_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
