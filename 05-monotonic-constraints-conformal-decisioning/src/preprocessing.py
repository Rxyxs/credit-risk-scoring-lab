"""Split en TRES conjuntos: train, calibracion y test.

La particion en tres no es un detalle: la prediccion conforme necesita un
conjunto de calibracion que el modelo **no** haya visto durante el
entrenamiento. Si se calibra sobre los mismos datos con los que se
entreno, los scores de no conformidad salen optimistas y la garantia de
cobertura -- que es todo el aporte del metodo -- deja de valer.

- `train` (60%)       : ajusta los modelos.
- `calibracion` (20%) : fija los umbrales conformes.
- `test` (20%)        : mide cobertura y decisiones, sin tocar nada mas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_generator import FEATURES

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"

RANDOM_STATE = 42
FRAC_CALIBRACION = 0.20
FRAC_TEST = 0.20


def load_applicants(nombre: str = "applicants.csv") -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / nombre)


def split_tres(df: pd.DataFrame, frac_cal: float = FRAC_CALIBRACION,
               frac_test: float = FRAC_TEST, seed: int = RANDOM_STATE
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado por default en train / calibracion / test."""
    if frac_cal + frac_test >= 1.0:
        raise ValueError("las fracciones de calibracion y test deben sumar menos de 1")

    resto, test = train_test_split(
        df, test_size=frac_test, random_state=seed, stratify=df["default_12m"]
    )
    frac_cal_en_resto = frac_cal / (1.0 - frac_test)
    train, calib = train_test_split(
        resto, test_size=frac_cal_en_resto, random_state=seed,
        stratify=resto["default_12m"],
    )
    return (train.reset_index(drop=True), calib.reset_index(drop=True),
            test.reset_index(drop=True))


def matriz(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=float)


def etiqueta(df: pd.DataFrame) -> np.ndarray:
    return df["default_12m"].to_numpy(dtype=int)


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_applicants()
    train, calib, test = split_tres(df)

    for nombre, parte in [("train", train), ("calib", calib), ("test", test)]:
        parte.to_csv(PROC_DIR / f"{nombre}.csv", index=False)
        print(f"{nombre:<6}: {len(parte):>6,} solicitudes | "
              f"default {parte['default_12m'].mean():.2%}")

    shifted = load_applicants("applicants_shifted.csv")
    shifted.to_csv(PROC_DIR / "test_shifted.csv", index=False)
    print(f"{'shift':<6}: {len(shifted):>6,} solicitudes | "
          f"default {shifted['default_12m'].mean():.2%} (escenario de deterioro)")
    print(f"\nFeatures del modelo ({len(FEATURES)}): {', '.join(FEATURES)}")


if __name__ == "__main__":
    main()
