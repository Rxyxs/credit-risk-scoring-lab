"""Preparacion de datos: split, estandarizacion e indice de segmento.

Detalle que importa mas de lo que parece: los segmentos se indexan con la
lista completa de combinaciones posibles (region x contrato x canal), no
con las que aparecen en train. Asi, si en test llega un segmento que en
train tenia pocos o ningun caso, el modelo jerarquico igual puede
puntuarlo -- cae de vuelta al efecto promedio, que es precisamente lo que
un modelo con dummies por segmento no puede hacer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_generator import FEATURES, segmentos

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.30

_NUMERICAS = {
    "edad_z": "edad",
    "log_renta_z": "log_renta",
    "dti_z": "dti",
    "antiguedad_z": "antiguedad_laboral_meses",
    "moras_z": "n_moras_12m",
    "tasa_z": "tasa_anual",
}
SEGMENTOS = segmentos()
MAPA_SEGMENTO = {s: i for i, s in enumerate(SEGMENTOS)}


def load_applicants(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or (RAW_DIR / "applicants.csv"))
    df["log_renta"] = np.log(df["renta_liquida"])
    return df


def split_train_test(df: pd.DataFrame, test_size: float = TEST_SIZE,
                     seed: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["default_12m"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_scaler(train: pd.DataFrame) -> dict:
    return {
        z: {"mean": float(train[raw].mean()), "std": float(train[raw].std(ddof=0))}
        for z, raw in _NUMERICAS.items()
    }


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    out = df.copy()
    for z, raw in _NUMERICAS.items():
        out[z] = (out[raw] - scaler[z]["mean"]) / scaler[z]["std"]
    return out


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Matriz de diseno con intercepto en la primera columna."""
    X = df[FEATURES].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(df)), X])


def design_names() -> list[str]:
    return ["intercepto", *FEATURES]


def segment_index(df: pd.DataFrame) -> np.ndarray:
    faltantes = set(df["segmento"]) - set(MAPA_SEGMENTO)
    if faltantes:
        raise ValueError(f"segmentos desconocidos: {sorted(faltantes)}")
    return df["segmento"].map(MAPA_SEGMENTO).to_numpy(dtype=int)


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_applicants()
    train, test = split_train_test(df)
    scaler = fit_scaler(train)
    train = apply_scaler(train, scaler)
    test = apply_scaler(test, scaler)

    train.to_csv(PROC_DIR / "train.csv", index=False)
    test.to_csv(PROC_DIR / "test.csv", index=False)
    (PROC_DIR / "scaler.json").write_text(json.dumps(scaler, indent=2))

    tam = train["segmento"].value_counts()
    print(f"Train: {len(train):,} ({train['default_12m'].mean():.2%} default)")
    print(f"Test : {len(test):,} ({test['default_12m'].mean():.2%} default)")
    print(f"Segmentos posibles: {len(SEGMENTOS)} | presentes en train: {tam.size}")
    print(f"Casos por segmento en train: min {tam.min()}, mediana {int(tam.median())}, "
          f"max {tam.max()}")
    chicos = tam[tam < 50]
    print(f"Segmentos con menos de 50 casos en train: {len(chicos)}")


if __name__ == "__main__":
    main()
