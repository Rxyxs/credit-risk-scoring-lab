"""Preparacion de la cartera para modelos de supervivencia.

Dos formatos distintos salen de aca, porque los dos modelos del proyecto
consumen la misma informacion de forma diferente:

- **Formato credito** (una fila por credito, con `duracion_meses` y
  `evento_default`): lo que consume Cox.
- **Formato persona-periodo** (una fila por credito y mes en riesgo, con
  una etiqueta binaria de "cayo en default este mes"): lo que consume el
  modelo de hazard en tiempo discreto.

El split es 70/30 estratificado por evento, con semilla fija. La
estandarizacion se ajusta SOLO en train y se aplica a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.30

FEATURES = [
    "edad_z", "log_renta_z", "dti_z", "antiguedad_z",
    "moras_z", "log_monto_z", "tasa_z", "plazo_z", "informal",
]

# Columnas crudas que se estandarizan, en el mismo orden que en el generador.
_NUMERICAS = {
    "edad_z": "edad",
    "log_renta_z": "log_renta",
    "dti_z": "dti",
    "antiguedad_z": "antiguedad_laboral_meses",
    "moras_z": "n_moras_12m",
    "log_monto_z": "log_monto",
    "tasa_z": "tasa_anual",
    "plazo_z": "plazo_meses",
}


def load_loan_book(path: Path | None = None) -> pd.DataFrame:
    path = path or (RAW_DIR / "loan_book.csv")
    df = pd.read_csv(path)
    df["log_renta"] = np.log(df["renta_liquida"])
    df["log_monto"] = np.log(df["monto_credito"])
    # Codificacion explicita en vez de get_dummies: la categoria que queda
    # como 1 es una decision del modelo, no del orden alfabetico.
    df["informal"] = (df["tipo_contrato"] == "informal").astype(float)
    return df


def split_train_test(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["evento_default"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_scaler(train: pd.DataFrame) -> dict:
    """Momentos de estandarizacion calculados solo sobre train."""
    return {
        z_name: {
            "mean": float(train[raw].mean()),
            "std": float(train[raw].std(ddof=0)),
        }
        for z_name, raw in _NUMERICAS.items()
    }


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for z_name, raw in _NUMERICAS.items():
        m = scaler[z_name]
        out[z_name] = (out[raw] - m["mean"]) / m["std"]
    return out


def design_matrix(df: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    features = features or FEATURES
    return df[features].to_numpy(dtype=float)


def to_person_period(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Expande a formato persona-periodo (una fila por credito-mes en riesgo).

    Un credito con `duracion_meses = 5` y `evento_default = 1` aporta 5
    filas (meses 1..5), con etiqueta 0,0,0,0,1. Si esta censurado, aporta
    las mismas 5 filas con etiqueta 0 en todas -- que es exactamente como
    la censura entra en la verosimilitud de un modelo de hazard discreto.
    """
    features = features or FEATURES
    dur = df["duracion_meses"].to_numpy(dtype=int)
    ev = df["evento_default"].to_numpy(dtype=int)

    rep = np.repeat(np.arange(len(df)), dur)
    mes = np.concatenate([np.arange(1, d + 1) for d in dur])

    out = df.iloc[rep][["loan_id", *features]].reset_index(drop=True)
    out["mes"] = mes
    out["duracion_meses"] = dur[rep]
    out["default_mes"] = ((out["mes"] == out["duracion_meses"]) & (ev[rep] == 1)).astype(int)
    return out


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_loan_book()
    train, test = split_train_test(df)

    scaler = fit_scaler(train)
    train = apply_scaler(train, scaler)
    test = apply_scaler(test, scaler)

    train.to_csv(PROC_DIR / "train_loans.csv", index=False)
    test.to_csv(PROC_DIR / "test_loans.csv", index=False)
    (PROC_DIR / "scaler.json").write_text(json.dumps(scaler, indent=2))

    pp_train = to_person_period(train)
    pp_test = to_person_period(test)
    pp_train.to_csv(PROC_DIR / "train_person_period.csv", index=False)
    pp_test.to_csv(PROC_DIR / "test_person_period.csv", index=False)

    print(f"Train: {len(train):,} creditos ({train['evento_default'].mean():.2%} default) "
          f"-> {len(pp_train):,} filas credito-mes")
    print(f"Test : {len(test):,} creditos ({test['evento_default'].mean():.2%} default) "
          f"-> {len(pp_test):,} filas credito-mes")
    print(f"Hazard mensual promedio en train: {pp_train['default_mes'].mean():.4%}")


if __name__ == "__main__":
    main()
