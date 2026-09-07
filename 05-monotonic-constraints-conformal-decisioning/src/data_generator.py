"""Generador sintetico de solicitudes de credito de consumo con una
estructura de riesgo *deliberadamente monotona en casi todo*.

Por que importa la forma del proceso generador: este proyecto compara un
modelo libre contra uno con restricciones de monotonia. Esa comparacion
solo es informativa si la verdad es efectivamente monotona en las
variables donde se imponen las restricciones -- si no, se estaria midiendo
el dano de una restriccion equivocada, no el costo de una correcta.

Entonces:

- `dti`, `n_moras_12m`, `utilizacion_lineas`, `consultas_6m` y `tasa_anual`
  aumentan el riesgo de forma monotona (con curvatura: el efecto del DTI
  tiene un quiebre marcado sobre 0.20, como la carga financiera real).
- `log_renta` y `antiguedad_laboral` lo bajan de forma monotona.
- `edad` tiene un efecto genuinamente **no monotono** (forma de U: los muy
  jovenes y los muy mayores son mas riesgosos que la mitad de la
  distribucion). Queda sin restringir a proposito, para que el proyecto
  muestre que las restricciones se ponen donde el dominio las respalda y
  no como regla general.
- Hay dos interacciones reales (morosidad x utilizacion de lineas, y
  utilizacion x carga financiera), para que un modelo aditivo no pueda
  capturar toda la estructura y el gradient boosting tenga algo que ganar.

`generate_shifted` produce la misma cartera bajo un escenario de deterioro
macro (mas informalidad, menos renta, mas utilizacion). Sirve para probar
que pasa con la garantia de cobertura conforme cuando la poblacion cambia,
que es la situacion normal de una cartera real y el limite honesto del
metodo.

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

FEATURES = [
    "dti", "n_moras_12m", "utilizacion_lineas", "consultas_6m",
    "tasa_anual", "log_renta", "antiguedad_laboral_meses", "edad",
]

# Direccion del efecto sobre el riesgo, en el formato que espera
# HistGradientBoostingClassifier: 1 = creciente, -1 = decreciente, 0 = libre.
MONOTONIC_CST = {
    "dti": 1,
    "n_moras_12m": 1,
    "utilizacion_lineas": 1,
    "consultas_6m": 1,
    "tasa_anual": 1,
    "log_renta": -1,
    "antiguedad_laboral_meses": -1,
    "edad": 0,           # efecto real en U: restringirlo seria un error
}

INTERCEPTO = -3.15


def _logit_riesgo(df: pd.DataFrame) -> np.ndarray:
    """Log-odds verdadero del default a 12 meses."""
    dti = df["dti"].to_numpy(float)
    moras = df["n_moras_12m"].to_numpy(float)
    util = df["utilizacion_lineas"].to_numpy(float)
    consultas = df["consultas_6m"].to_numpy(float)
    tasa = df["tasa_anual"].to_numpy(float)
    log_renta = df["log_renta"].to_numpy(float)
    antig = df["antiguedad_laboral_meses"].to_numpy(float)
    edad = df["edad"].to_numpy(float)

    eta = np.full(len(df), INTERCEPTO)
    # Carga financiera con quiebre: suave hasta 0.20, empinada despues.
    eta += 1.10 * dti + 4.10 * np.clip(dti - 0.20, 0, None)
    eta += 0.62 * moras
    eta += 1.05 * util
    eta += 0.16 * consultas
    eta += 0.030 * (tasa - 18.0)
    eta += -0.55 * (log_renta - 13.6)
    eta += -0.0035 * antig
    # Efecto no monotono de la edad: minimo cerca de los 45 anos.
    eta += 0.0024 * (edad - 45.0) ** 2
    # Interacciones: una linea copada pesa mucho mas si ya hubo moras, y
    # todavia mas si ademas la carga financiera es alta. Un modelo aditivo
    # no puede representar ninguna de las dos.
    eta += 1.25 * moras * np.clip(util - 0.55, 0, None)
    eta += 9.0 * np.clip(util - 0.50, 0, None) * np.clip(dti - 0.15, 0, None)
    return eta


def _sample_features(n: int, rng: np.random.Generator, shift: bool = False) -> pd.DataFrame:
    informal_p = 0.44 if shift else 0.32
    informal = rng.random(n) < informal_p

    renta_mult = 0.88 if shift else 1.0
    renta = np.clip(
        np.where(informal, rng.lognormal(13.30, 0.55, n), rng.lognormal(13.70, 0.45, n))
        * renta_mult,
        320_000, 8_000_000,
    )
    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)
    antiguedad = np.where(
        informal,
        np.clip(rng.exponential(20, n), 1, 300),
        np.clip(rng.exponential(48, n), 1, 420),
    ).round().astype(int)

    lam_moras = (0.30 + 0.45 * informal + 0.30 * (renta < 600_000)) * (1.35 if shift else 1.0)
    moras = np.clip(rng.poisson(lam_moras), 0, 6)

    util_base = rng.beta(2.2, 3.0, n) + (0.10 if shift else 0.0)
    utilizacion = np.clip(util_base, 0.0, 1.0)

    consultas = np.clip(rng.poisson(1.1 + 0.8 * informal + (0.5 if shift else 0.0)), 0, 12)

    plazo = rng.choice([12, 24, 36, 48], size=n, p=[0.18, 0.34, 0.33, 0.15])
    monto = np.clip(renta * rng.lognormal(1.05, 0.45, n), 300_000, 30_000_000)
    dti = np.clip((monto / plazo) / renta + rng.normal(0, 0.04, n), 0.02, 0.85)

    tasa = np.clip(
        18.0 + 3.2 * moras + 6.0 * informal + 8.0 * dti + rng.normal(0, 2.0, n),
        9.0, 45.0,
    )

    return pd.DataFrame({
        "edad": edad,
        "tipo_contrato": np.where(informal, "informal", "formal"),
        "renta_liquida": renta.round(0),
        "log_renta": np.log(renta),
        "antiguedad_laboral_meses": antiguedad,
        "n_moras_12m": moras,
        "utilizacion_lineas": utilizacion.round(4),
        "consultas_6m": consultas,
        "monto_credito": monto.round(0),
        "plazo_meses": plazo,
        "dti": dti.round(4),
        "tasa_anual": tasa.round(2),
    })


def generate_applicants(n: int = 20_000, seed: int = RANDOM_STATE_DEFAULT,
                        shift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _sample_features(n, rng, shift=shift)
    eta = _logit_riesgo(df)
    p = 1.0 / (1.0 + np.exp(-eta))
    df.insert(0, "applicant_id", np.arange(1, n + 1))
    df["pd_verdadera"] = p
    df["default_12m"] = (rng.random(n) < p).astype(int)
    return df


def generate_shifted(n: int = 6_000, seed: int = 202) -> pd.DataFrame:
    """Cartera bajo deterioro macro: mas informalidad, menos renta, mas deuda."""
    return generate_applicants(n=n, seed=seed, shift=True)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_applicants()
    shifted = generate_shifted()
    df.to_csv(RAW_DIR / "applicants.csv", index=False)
    shifted.to_csv(RAW_DIR / "applicants_shifted.csv", index=False)
    (RAW_DIR / "monotonic_constraints.json").write_text(json.dumps(MONOTONIC_CST, indent=2))

    print(f"Cartera base    : {len(df):,} solicitudes, "
          f"{df['default_12m'].mean():.2%} de default")
    print(f"Cartera shifted : {len(shifted):,} solicitudes, "
          f"{shifted['default_12m'].mean():.2%} de default")
    print("\nDiferencias del escenario de deterioro:")
    for col in ("dti", "utilizacion_lineas", "n_moras_12m", "consultas_6m"):
        print(f"  {col:<22} {df[col].mean():>8.3f} -> {shifted[col].mean():>8.3f}")
    print(f"  {'renta_liquida':<22} {df['renta_liquida'].mean():>8,.0f} -> "
          f"{shifted['renta_liquida'].mean():>8,.0f}")
    print(f"\nRestricciones de monotonia: "
          f"{sum(v != 0 for v in MONOTONIC_CST.values())} de {len(MONOTONIC_CST)} features")
    print(f"-> {RAW_DIR / 'applicants.csv'}")


if __name__ == "__main__":
    main()
