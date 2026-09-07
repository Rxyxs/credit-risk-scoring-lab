"""Generador sintetico de solicitudes de credito de consumo organizadas en
cohortes mensuales (vintages), con un deterioro deliberado en las ultimas.

Dos cosas que este simulador necesita producir y que no son adorno:

1. **Relaciones no lineales pero monotonas** entre cada variable y el
   riesgo. El binning optimo existe para encontrar los cortes que mejor
   separan buenos de malos; si la relacion fuera lineal, cualquier
   discretizacion razonable daria lo mismo y no habria nada que comparar.
   Aca la carga financiera tiene un quiebre, la renta tiene rendimientos
   decrecientes, y la edad -- de nuevo -- es la excepcion no monotona.
2. **Un cambio de poblacion en las ultimas cohortes**: desde el vintage
   `VINTAGE_QUIEBRE` sube la informalidad, sube la utilizacion de lineas y
   baja la renta. Es el escenario que el monitoreo PSI/CSI tiene que
   levantar; sin el, un modulo de monitoreo se valida contra nada.

No usa datos reales de ninguna institucion ni del Boletin Comercial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"

RANDOM_STATE_DEFAULT = 42
N_VINTAGES = 24
VINTAGE_QUIEBRE = 18          # desde aca empieza el deterioro de la poblacion

NUMERICAS = [
    "dti", "utilizacion_lineas", "log_renta", "antiguedad_laboral_meses",
    "n_moras_12m", "consultas_6m", "edad", "tasa_anual",
]
CATEGORICAS = ["tipo_contrato", "region", "canal"]
FEATURES = NUMERICAS + CATEGORICAS

REGIONES = [
    "Metropolitana", "Valparaiso", "Biobio", "Antofagasta",
    "Araucania", "Maule", "Los Lagos", "Coquimbo",
]
REGION_RIESGO = {
    "Metropolitana": -0.10, "Valparaiso": 0.02, "Biobio": 0.10,
    "Antofagasta": -0.12, "Araucania": 0.22, "Maule": 0.10,
    "Los Lagos": 0.05, "Coquimbo": 0.08,
}

INTERCEPTO = -2.60


def _logit_riesgo(df: pd.DataFrame) -> np.ndarray:
    """Log-odds verdadero: monotono en casi todo, con curvatura real."""
    dti = df["dti"].to_numpy(float)
    util = df["utilizacion_lineas"].to_numpy(float)
    log_renta = df["log_renta"].to_numpy(float)
    antig = df["antiguedad_laboral_meses"].to_numpy(float)
    moras = df["n_moras_12m"].to_numpy(float)
    consultas = df["consultas_6m"].to_numpy(float)
    edad = df["edad"].to_numpy(float)
    tasa = df["tasa_anual"].to_numpy(float)

    eta = np.full(len(df), INTERCEPTO)
    # Quiebre en la carga financiera: casi plano abajo, empinado arriba.
    eta += 0.90 * dti + 4.60 * np.clip(dti - 0.22, 0, None)
    # Utilizacion: efecto que se acelera al copar la linea.
    eta += 0.70 * util + 2.10 * np.clip(util - 0.70, 0, None)
    # Renta: rendimientos decrecientes (protege mucho al principio).
    eta += -0.85 * np.clip(log_renta - 12.9, 0, None)
    eta += -0.0040 * antig
    eta += 0.60 * moras
    eta += 0.14 * consultas
    eta += 0.0022 * (edad - 45.0) ** 2          # no monotona a proposito
    eta += 0.028 * (tasa - 18.0)
    eta += df["region"].map(REGION_RIESGO).to_numpy(float)
    eta += 0.35 * (df["tipo_contrato"].to_numpy() == "informal")
    eta += 0.12 * (df["canal"].to_numpy() == "digital")
    return eta


def _cohorte(n: int, rng: np.random.Generator, deteriorada: bool) -> pd.DataFrame:
    p_informal = 0.46 if deteriorada else 0.31
    informal = rng.random(n) < p_informal
    mult_renta = 0.87 if deteriorada else 1.0

    renta = np.clip(
        np.where(informal, rng.lognormal(13.30, 0.55, n), rng.lognormal(13.70, 0.45, n))
        * mult_renta,
        320_000, 8_000_000,
    )
    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)
    antiguedad = np.where(
        informal,
        np.clip(rng.exponential(20, n), 1, 300),
        np.clip(rng.exponential(48, n), 1, 420),
    ).round().astype(int)
    moras = np.clip(
        rng.poisson((0.28 + 0.42 * informal + 0.28 * (renta < 600_000))
                    * (1.30 if deteriorada else 1.0)),
        0, 6,
    )
    util = np.clip(rng.beta(2.2, 3.0, n) + (0.12 if deteriorada else 0.0), 0.0, 1.0)
    consultas = np.clip(rng.poisson(1.1 + 0.7 * informal + (0.4 if deteriorada else 0.0)), 0, 12)

    plazo = rng.choice([12, 24, 36, 48], size=n, p=[0.18, 0.34, 0.33, 0.15])
    monto = np.clip(renta * rng.lognormal(1.05, 0.45, n), 300_000, 30_000_000)
    dti = np.clip((monto / plazo) / renta + rng.normal(0, 0.04, n), 0.02, 0.85)
    tasa = np.clip(
        18.0 + 3.0 * moras + 5.5 * informal + 8.0 * dti + rng.normal(0, 2.0, n), 9.0, 45.0
    )

    return pd.DataFrame({
        "tipo_contrato": np.where(informal, "informal", "formal"),
        "region": rng.choice(REGIONES, size=n,
                             p=[0.40, 0.13, 0.11, 0.07, 0.06, 0.08, 0.08, 0.07]),
        "canal": np.where(rng.random(n) < (0.55 if deteriorada else 0.45), "digital", "sucursal"),
        "edad": edad,
        "renta_liquida": renta.round(0),
        "log_renta": np.log(renta),
        "antiguedad_laboral_meses": antiguedad,
        "n_moras_12m": moras,
        "utilizacion_lineas": util.round(4),
        "consultas_6m": consultas,
        "monto_credito": monto.round(0),
        "plazo_meses": plazo,
        "dti": dti.round(4),
        "tasa_anual": tasa.round(2),
    })


def generate_applicants(n: int = 24_000, seed: int = RANDOM_STATE_DEFAULT,
                        n_vintages: int = N_VINTAGES) -> pd.DataFrame:
    """Genera solicitudes repartidas en `n_vintages` cohortes mensuales."""
    rng = np.random.default_rng(seed)
    por_vintage = n // n_vintages
    etiquetas = pd.PeriodIndex(
        pd.date_range("2023-01-01", periods=n_vintages, freq="MS"), freq="M"
    ).astype(str)

    partes = []
    for v in range(n_vintages):
        parte = _cohorte(por_vintage, rng, deteriorada=v >= VINTAGE_QUIEBRE)
        parte.insert(0, "vintage", etiquetas[v])
        parte.insert(1, "vintage_idx", v)
        partes.append(parte)

    df = pd.concat(partes, ignore_index=True)
    df.insert(0, "applicant_id", np.arange(1, len(df) + 1))
    eta = _logit_riesgo(df)
    p = 1.0 / (1.0 + np.exp(-eta))
    df["pd_verdadera"] = p
    df["default_12m"] = (rng.random(len(df)) < p).astype(int)
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_applicants()
    df.to_csv(RAW_DIR / "applicants.csv", index=False)
    (RAW_DIR / "feature_types.json").write_text(json.dumps(
        {"numericas": NUMERICAS, "categoricas": CATEGORICAS,
         "vintage_quiebre": VINTAGE_QUIEBRE}, indent=2))

    por_v = df.groupby("vintage_idx").agg(
        n=("applicant_id", "size"),
        default=("default_12m", "mean"),
        util=("utilizacion_lineas", "mean"),
        informal=("tipo_contrato", lambda s: (s == "informal").mean()),
    )
    estable = por_v.loc[por_v.index < VINTAGE_QUIEBRE]
    deteriorada = por_v.loc[por_v.index >= VINTAGE_QUIEBRE]

    print(f"Solicitudes         : {len(df):,} en {df['vintage'].nunique()} vintages")
    print(f"Tasa de default     : {df['default_12m'].mean():.2%}")
    print(f"\nVintages 0-{VINTAGE_QUIEBRE - 1} (poblacion estable)")
    print(f"  default {estable['default'].mean():.2%} | utilizacion "
          f"{estable['util'].mean():.3f} | informales {estable['informal'].mean():.1%}")
    print(f"Vintages {VINTAGE_QUIEBRE}-{N_VINTAGES - 1} (deterioro)")
    print(f"  default {deteriorada['default'].mean():.2%} | utilizacion "
          f"{deteriorada['util'].mean():.3f} | informales {deteriorada['informal'].mean():.1%}")
    print(f"-> {RAW_DIR / 'applicants.csv'}")


if __name__ == "__main__":
    main()
