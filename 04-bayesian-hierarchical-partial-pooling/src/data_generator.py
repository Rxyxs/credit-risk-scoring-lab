"""Generador sintetico de solicitudes de credito de consumo con estructura
jerarquica por segmento comercial.

El punto del proyecto es el *pooling parcial*, asi que el simulador tiene
que producir el escenario donde el pooling parcial importa y no uno donde
da lo mismo:

- 32 segmentos (8 regiones x 2 tipos de contrato x 2 canales de venta)
  con tamanos muy desbalanceados: la Region Metropolitana formal por
  sucursal aporta mas de mil casos y varios segmentos regionales
  informales aportan menos de 60. Con tan pocos datos, un intercepto por
  segmento estimado de forma independiente es puro ruido, y estimar un
  solo intercepto para todos borra diferencias reales.
- Los efectos de segmento se sortean de una normal comun con desviacion
  conocida (`TAU_TRUE`), que es exactamente el supuesto jerarquico: los
  segmentos son distintos, pero se parecen entre si.

Guardar los efectos verdaderos permite despues medir cual de los tres
enfoques (pooling completo, sin pooling, pooling parcial) los recupera
mejor, en vez de discutirlo en abstracto.

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

REGIONES = [
    "Metropolitana", "Valparaiso", "Biobio", "Antofagasta",
    "Araucania", "Maule", "Los Lagos", "Coquimbo",
]
CONTRATOS = ["formal", "informal"]
CANALES = ["sucursal", "digital"]

# Peso comercial de cada region (la RM concentra la originacion real).
PESO_REGION = np.array([0.42, 0.13, 0.11, 0.07, 0.06, 0.08, 0.07, 0.06])
# Dentro de cada region, cuanta originacion es de contrato informal.
PROP_INFORMAL = np.array([0.22, 0.28, 0.33, 0.18, 0.44, 0.36, 0.30, 0.31])
# Penetracion del canal digital por region (la RM concentra lo digital).
PROP_DIGITAL = np.array([0.62, 0.44, 0.38, 0.41, 0.24, 0.31, 0.29, 0.33])

INTERCEPTO_TRUE = -2.35
BETA_TRUE = {
    "edad_z": -0.20,
    "log_renta_z": -0.35,
    "dti_z": 0.55,
    "antiguedad_z": -0.25,
    "moras_z": 0.80,
    "tasa_z": 0.30,
}
FEATURES = list(BETA_TRUE)

# Desviacion estandar verdadera de los efectos de segmento.
TAU_TRUE = 0.45


def _zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / sd, mu, sd


def segmentos() -> list[str]:
    return [f"{r}|{c}|{k}" for r in REGIONES for c in CONTRATOS for k in CANALES]


def generate_applicants(n: int = 9_000, seed: int = RANDOM_STATE_DEFAULT
                        ) -> tuple[pd.DataFrame, dict]:
    """Genera `n` solicitudes con efectos de segmento jerarquicos conocidos."""
    rng = np.random.default_rng(seed)

    region_idx = rng.choice(len(REGIONES), size=n, p=PESO_REGION)
    informal = rng.random(n) < PROP_INFORMAL[region_idx]
    region = np.array(REGIONES)[region_idx]
    tipo_contrato = np.where(informal, "informal", "formal")
    digital = rng.random(n) < PROP_DIGITAL[region_idx]
    canal = np.where(digital, "digital", "sucursal")
    segmento = np.array([f"{r}|{c}|{k}" for r, c, k in zip(region, tipo_contrato, canal)])

    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)
    renta = np.clip(
        np.where(informal,
                 rng.lognormal(13.30, 0.55, n),
                 rng.lognormal(13.70, 0.45, n)),
        320_000, 8_000_000,
    )
    antiguedad = np.where(
        informal,
        np.clip(rng.exponential(20, n), 1, 300),
        np.clip(rng.exponential(48, n), 1, 420),
    ).round().astype(int)
    n_moras_12m = np.clip(rng.poisson(0.35 + 0.45 * informal + 0.30 * (renta < 600_000)), 0, 6)
    monto = np.clip(renta * rng.lognormal(1.05, 0.45, n), 300_000, 30_000_000)
    plazo = rng.choice([12, 24, 36, 48], size=n, p=[0.18, 0.34, 0.33, 0.15])
    dti = np.clip((monto / plazo) / renta + rng.normal(0, 0.04, n), 0.02, 0.85)
    tasa_anual = np.clip(
        18.0 + 3.2 * n_moras_12m + 6.0 * informal + 8.0 * dti + rng.normal(0, 2.0, n),
        9.0, 45.0,
    )

    z, moments = {}, {}
    crudas = {
        "edad_z": edad.astype(float),
        "log_renta_z": np.log(renta),
        "dti_z": dti,
        "antiguedad_z": antiguedad.astype(float),
        "moras_z": n_moras_12m.astype(float),
        "tasa_z": tasa_anual,
    }
    for name, arr in crudas.items():
        z[name], mu, sd = _zscore(arr)
        moments[name] = {"mean": mu, "std": sd}

    # Efectos jerarquicos: un intercepto por segmento, sorteado de una
    # normal comun. Asi es como se comportan segmentos reales -- distintos,
    # pero no independientes.
    nombres_seg = segmentos()
    efecto_seg = rng.normal(0.0, TAU_TRUE, size=len(nombres_seg))
    efecto_seg -= efecto_seg.mean()          # identificabilidad vs el intercepto
    mapa_seg = {s: i for i, s in enumerate(nombres_seg)}
    seg_idx = np.array([mapa_seg[s] for s in segmento])

    eta = INTERCEPTO_TRUE + efecto_seg[seg_idx]
    for name, beta in BETA_TRUE.items():
        eta += beta * z[name]
    p_default = 1.0 / (1.0 + np.exp(-eta))
    default_12m = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "region": region,
        "tipo_contrato": tipo_contrato,
        "canal": canal,
        "segmento": segmento,
        "edad": edad,
        "renta_liquida": renta.round(0),
        "antiguedad_laboral_meses": antiguedad,
        "n_moras_12m": n_moras_12m,
        "monto_credito": monto.round(0),
        "plazo_meses": plazo,
        "tasa_anual": tasa_anual.round(2),
        "dti": dti.round(4),
        "pd_verdadera": p_default,
        "default_12m": default_12m,
    })

    ground_truth = {
        "intercepto_true": INTERCEPTO_TRUE,
        "beta_true": BETA_TRUE,
        "tau_true": TAU_TRUE,
        "efectos_segmento_true": dict(zip(nombres_seg, efecto_seg.tolist())),
        "standardization_moments": moments,
        "n": int(n),
        "seed": int(seed),
    }
    return df, ground_truth


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df, gt = generate_applicants()
    df.to_csv(RAW_DIR / "applicants.csv", index=False)
    (RAW_DIR / "ground_truth.json").write_text(json.dumps(gt, indent=2))

    tam = df["segmento"].value_counts()
    tasas = df.groupby("segmento")["default_12m"].mean()
    print(f"Solicitudes            : {len(df):,}")
    print(f"Tasa de default global : {df['default_12m'].mean():.2%}")
    print(f"Segmentos              : {df['segmento'].nunique()}")
    print(f"Tamano de segmento     : min {tam.min()}, mediana {int(tam.median())}, max {tam.max()}")
    print(f"Tasa de default por segmento: {tasas.min():.2%} a {tasas.max():.2%}")
    print("\nLos 5 segmentos mas chicos (donde el pooling parcial deberia pesar):")
    for seg, k in tam.tail(5).items():
        print(f"  {seg:<28} n={k:>4}  default observado {tasas[seg]:.2%}")
    print(f"-> {RAW_DIR / 'applicants.csv'}")


if __name__ == "__main__":
    main()
