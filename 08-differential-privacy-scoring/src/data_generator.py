"""Generador sintetico de solicitudes de credito, con dos particularidades
pensadas para poder *medir* la privacidad y no solo declararla.

1. **Conjunto de entrenamiento chico.** El riesgo de memorizacion no es un
   fenomeno abstracto: aparece cuando el modelo tiene pocos datos por
   parametro. Entrenar con 1.200 solicitudes es un escenario realista
   (un producto nuevo, un segmento chico) y ademas es donde un ataque de
   inferencia de membresia tiene algo que encontrar. Con 200.000 filas
   cualquier modelo se ve privado sin serlo.

2. **Canarios.** Un pequeno grupo de registros deliberadamente atipicos y
   con la etiqueta invertida respecto de lo que su perfil sugiere. Nada en
   los datos "normales" los explica, asi que la unica forma de que el
   modelo les acierte es haberlos memorizado. Son el equivalente crediticio
   del "secret sharer" de Carlini: convierten la memorizacion en algo
   observable con una regla simple.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"

RANDOM_STATE_DEFAULT = 42
N_CANARIOS = 40

FEATURES = [
    "dti", "n_moras_12m", "utilizacion_lineas", "consultas_6m",
    "log_renta", "antiguedad_laboral_meses", "edad",
]

INTERCEPTO = -2.60


def _logit_riesgo(df: pd.DataFrame) -> np.ndarray:
    eta = np.full(len(df), INTERCEPTO)
    eta += 1.00 * df["dti"].to_numpy(float)
    eta += 4.00 * np.clip(df["dti"].to_numpy(float) - 0.20, 0, None)
    eta += 0.65 * df["n_moras_12m"].to_numpy(float)
    eta += 1.10 * df["utilizacion_lineas"].to_numpy(float)
    eta += 0.16 * df["consultas_6m"].to_numpy(float)
    eta += -0.60 * np.clip(df["log_renta"].to_numpy(float) - 12.9, 0, None)
    eta += -0.0035 * df["antiguedad_laboral_meses"].to_numpy(float)
    eta += 0.0020 * (df["edad"].to_numpy(float) - 45.0) ** 2
    return eta


def _muestra(n: int, rng: np.random.Generator) -> pd.DataFrame:
    informal = rng.random(n) < 0.31
    renta = np.clip(
        np.where(informal, rng.lognormal(13.30, 0.55, n), rng.lognormal(13.70, 0.45, n)),
        320_000, 8_000_000,
    )
    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)
    antiguedad = np.where(
        informal, np.clip(rng.exponential(20, n), 1, 300),
        np.clip(rng.exponential(48, n), 1, 420),
    ).round().astype(int)
    moras = np.clip(rng.poisson(0.30 + 0.40 * informal + 0.25 * (renta < 600_000)), 0, 6)
    utilizacion = np.clip(rng.beta(2.2, 3.0, n), 0.0, 1.0)
    consultas = np.clip(rng.poisson(1.1 + 0.7 * informal), 0, 12)
    plazo = rng.choice([12, 24, 36, 48], size=n, p=[0.18, 0.34, 0.33, 0.15])
    monto = np.clip(renta * rng.lognormal(1.05, 0.45, n), 300_000, 30_000_000)
    dti = np.clip((monto / plazo) / renta + rng.normal(0, 0.04, n), 0.02, 0.85)

    return pd.DataFrame({
        "edad": edad,
        "renta_liquida": renta.round(0),
        "log_renta": np.log(renta),
        "antiguedad_laboral_meses": antiguedad,
        "n_moras_12m": moras,
        "utilizacion_lineas": utilizacion.round(4),
        "consultas_6m": consultas,
        "monto_credito": monto.round(0),
        "plazo_meses": plazo,
        "dti": dti.round(4),
    })


def generar_canarios(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Registros atipicos con etiqueta contraria a su perfil de riesgo.

    Perfil impecable (renta alta, sin moras, carga baja) pero etiquetados
    como default. Ningun patron general los justifica: si el modelo les
    asigna PD alta, es memorizacion.
    """
    df = pd.DataFrame({
        "edad": rng.integers(40, 52, n),
        "renta_liquida": rng.uniform(5_500_000, 7_800_000, n).round(0),
        "antiguedad_laboral_meses": rng.integers(280, 400, n),
        "n_moras_12m": np.zeros(n, dtype=int),
        "utilizacion_lineas": np.round(rng.uniform(0.01, 0.06, n), 4),
        "consultas_6m": np.zeros(n, dtype=int),
        "monto_credito": rng.uniform(400_000, 900_000, n).round(0),
        "plazo_meses": np.full(n, 48),
    })
    df["log_renta"] = np.log(df["renta_liquida"])
    df["dti"] = np.round((df["monto_credito"] / df["plazo_meses"]) / df["renta_liquida"], 4)
    df["pd_verdadera"] = np.nan
    df["default_12m"] = 1            # etiqueta invertida a proposito
    df["es_canario"] = True
    return df


def generate_datasets(n_train: int = 1_200, n_holdout: int = 1_200,
                      n_test: int = 4_000, n_canarios: int = N_CANARIOS,
                      seed: int = RANDOM_STATE_DEFAULT) -> dict[str, pd.DataFrame]:
    """Tres conjuntos disjuntos, mas los canarios inyectados solo en train.

    `holdout` cumple un rol especifico: es la poblacion de NO miembros con
    la que el ataque de membresia compara. Tiene que venir de la misma
    distribucion que train y no haber sido vista nunca, o el ataque estaria
    detectando diferencias de distribucion en vez de memorizacion.
    """
    rng = np.random.default_rng(seed)
    partes = {}
    for nombre, n in [("train", n_train), ("holdout", n_holdout), ("test", n_test)]:
        df = _muestra(n, rng)
        eta = _logit_riesgo(df)
        df["pd_verdadera"] = 1.0 / (1.0 + np.exp(-eta))
        df["default_12m"] = (rng.random(n) < df["pd_verdadera"]).astype(int)
        df["es_canario"] = False
        partes[nombre] = df

    canarios = generar_canarios(n_canarios, rng)
    partes["canarios"] = canarios
    partes["train"] = pd.concat([partes["train"], canarios], ignore_index=True)

    for nombre, df in partes.items():
        df.insert(0, "record_id", [f"{nombre}_{i}" for i in range(len(df))])
    return partes


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    partes = generate_datasets()
    for nombre, df in partes.items():
        df.to_csv(RAW_DIR / f"{nombre}.csv", index=False)

    (RAW_DIR / "diseno.json").write_text(json.dumps({
        "features": FEATURES,
        "n_canarios": int(len(partes["canarios"])),
        "nota": "los canarios estan solo en train; holdout es la poblacion "
                "de no miembros para el ataque de membresia",
    }, indent=2))

    for nombre in ("train", "holdout", "test"):
        df = partes[nombre]
        print(f"{nombre:<9}: {len(df):>5,} registros | "
              f"default {df['default_12m'].mean():.2%}")
    can = partes["canarios"]
    normal = partes["train"][~partes["train"]["es_canario"]]
    print(f"canarios : {len(can):>5,} registros con etiqueta invertida")
    print(f"\nPerfil de un canario contra la mediana del train:")
    for col in ("renta_liquida", "dti", "n_moras_12m", "utilizacion_lineas"):
        print(f"  {col:<22} canario {can[col].median():>12,.4f} | "
              f"train {normal[col].median():>12,.4f}")
    print(f"-> {RAW_DIR}")


if __name__ == "__main__":
    main()
