"""Limpieza y validacion de consistencia de negocio sobre los solicitantes
crudos, antes de WOE/IV (R) y del entrenamiento de modelos ML (Python).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

NUMERIC_COLS = [
    "edad", "renta_liquida", "antiguedad_laboral_meses",
    "n_productos_activos", "deuda_total", "dti", "n_morosidad_reportes",
]


def enforce_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Invalida (convierte a NaN) combinaciones imposibles de negocio, en
    vez de simplemente recortar valores: una renta negativa o un DTI
    infinito no es un outlier estadistico, es un registro corrupto."""
    out = df.copy()
    out.loc[out["renta_liquida"] <= 0, "renta_liquida"] = np.nan
    out.loc[out["dti"] < 0, "dti"] = np.nan
    out.loc[out["edad"] < 18, "edad"] = np.nan
    out.loc[out["antiguedad_laboral_meses"] < 0, "antiguedad_laboral_meses"] = np.nan
    return out


def clip_outliers_iqr(df: pd.DataFrame, cols: list[str], factor: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        out[col] = out[col].clip(lower, upper)
    return out


def impute_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    imputer = SimpleImputer(strategy="median")
    out[cols] = imputer.fit_transform(out[cols])
    return out


def run_cleaning_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    validated = enforce_business_rules(raw)
    imputed = impute_missing(validated, NUMERIC_COLS)
    cleaned = clip_outliers_iqr(imputed, ["renta_liquida", "deuda_total"])
    # DTI se recalcula desde renta/deuda ya recortadas, para que quede
    # consistente con las dos variables de las que depende, y luego se
    # recorta por su cuenta (la razon puede tener su propia cola larga
    # aunque el numerador y denominador ya esten acotados).
    cleaned["dti"] = cleaned["deuda_total"] / cleaned["renta_liquida"].replace(0, np.nan)
    cleaned["dti"] = cleaned["dti"].fillna(cleaned["dti"].median())
    cleaned = clip_outliers_iqr(cleaned, ["dti"])

    # Split estratificado unico (75/25), compartido por el scorecard R y
    # los modelos ML de Python, para que la comparacion entre ambos sea
    # sobre exactamente el mismo holdout, no dos muestras distintas.
    train_ids, test_ids = train_test_split(
        cleaned["applicant_id"], test_size=0.25, stratify=cleaned["default_12m"], random_state=42,
    )
    cleaned["split"] = np.where(cleaned["applicant_id"].isin(set(train_ids)), "train", "test")
    return cleaned


if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / "data"
    raw = pd.read_csv(base / "raw" / "applicants_raw.csv")
    cleaned = run_cleaning_pipeline(raw)
    out_path = base / "processed" / "applicants_clean.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Limpiados {len(cleaned)} registros -> {out_path}")
    print(f"NaNs restantes: {int(cleaned.isna().sum().sum())}")
