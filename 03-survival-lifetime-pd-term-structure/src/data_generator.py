"""Generador sintetico de una cartera de creditos de consumo con historia
mensual: cada credito se sigue mes a mes hasta que cae en default, prepaga,
o llega al fin de la ventana de observacion (censura).

A diferencia de un dataset clasico de scoring (una fila = un solicitante,
una etiqueta default_12m), aca el proceso generador es un *hazard mensual*
conocido, lo que permite validar el modelo de supervivencia contra la
verdad de terreno: los coeficientes estimados deben recuperar los betas
con los que se simulo la cartera.

Dos elementos estan puestos a proposito para poder detectarlos despues:

1. Un hazard base con forma de joroba (sube, peak alrededor del mes 8-10,
   despues baja) -- el patron de "seasoning" que se observa en credito de
   consumo real, y que una PD plana a 12 meses no puede expresar.
2. Un efecto NO proporcional: el riesgo extra del contrato informal es
   fuerte al inicio y se desvanece con el tiempo. Viola el supuesto de
   hazards proporcionales de Cox, y el pipeline lo tiene que detectar en
   los residuos de Schoenfeld en vez de darlo por bueno.

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
HORIZONTE_MESES = 36

REGIONES = [
    "Metropolitana", "Valparaiso", "Biobio", "Antofagasta",
    "Araucania", "Maule", "Los Lagos", "Coquimbo",
]

# Coeficientes verdaderos sobre features estandarizadas (escala log-hazard).
BETA_TRUE = {
    "edad_z": -0.18,
    "log_renta_z": -0.25,
    "dti_z": 0.40,
    "antiguedad_z": -0.20,
    "moras_z": 0.55,
    "log_monto_z": 0.15,
    "tasa_z": 0.22,
    "plazo_z": 0.10,
}

# Efecto no proporcional del contrato informal: gamma(t) = G0 * exp(-t / TAU).
GAMMA_INFORMAL_0 = 0.75
GAMMA_INFORMAL_TAU = 10.0

# log h0(t) = A0 + A1*log(t) + A2*log(t)^2 -> joroba con peak cerca del mes 9.
BASELINE_A0 = -6.35
BASELINE_A1 = 1.35
BASELINE_A2 = -0.32

# Hazard mensual de prepago (riesgo competitivo): base + efecto de renta.
PREPAGO_C0 = -4.85
PREPAGO_C1 = 0.30


def baseline_log_hazard(t: np.ndarray) -> np.ndarray:
    """log del hazard base mensual en el mes `t` (t >= 1)."""
    lt = np.log(np.asarray(t, dtype=float))
    return BASELINE_A0 + BASELINE_A1 * lt + BASELINE_A2 * lt**2


def gamma_informal(t: np.ndarray) -> np.ndarray:
    """Efecto (decreciente en el tiempo) del contrato informal."""
    return GAMMA_INFORMAL_0 * np.exp(-np.asarray(t, dtype=float) / GAMMA_INFORMAL_TAU)


def _zscore(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / sd, mu, sd


def generate_portfolio(
    n: int = 15_000,
    seed: int = RANDOM_STATE_DEFAULT,
    horizonte: int = HORIZONTE_MESES,
) -> tuple[pd.DataFrame, dict]:
    """Genera `n` creditos con tiempo hasta default, prepago o censura.

    Devuelve el dataframe a nivel credito y un diccionario con la verdad de
    terreno (betas, momentos de estandarizacion, forma del hazard base).
    """
    rng = np.random.default_rng(seed)

    # --- originacion: 24 cohortes mensuales (vintages) 2022-01 .. 2023-12 ---
    vintage_idx = rng.integers(0, 24, size=n)
    vintages = np.array(
        pd.PeriodIndex(pd.date_range("2022-01-01", periods=24, freq="MS"), freq="M").astype(str)
    )

    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)

    tipo_contrato = rng.choice(["formal", "informal"], size=n, p=[0.66, 0.34])
    informal = (tipo_contrato == "informal").astype(float)

    renta = np.clip(
        np.where(
            informal == 1,
            rng.lognormal(13.30, 0.55, n),
            rng.lognormal(13.70, 0.45, n),
        ),
        320_000, 8_000_000,
    )

    antiguedad = np.where(
        informal == 1,
        np.clip(rng.exponential(20, n), 1, 300),
        np.clip(rng.exponential(48, n), 1, 420),
    ).round().astype(int)

    # Morosidad pasada: mas probable en informales y en rentas bajas.
    lam_moras = 0.35 + 0.45 * informal + 0.30 * (renta < 600_000)
    n_moras_12m = np.clip(rng.poisson(lam_moras), 0, 6)

    plazo_meses = rng.choice([12, 24, 36, 48], size=n, p=[0.18, 0.34, 0.33, 0.15])
    monto = np.clip(renta * rng.lognormal(1.05, 0.45, n), 300_000, 30_000_000)

    # Carga financiera: cuota estimada sobre renta liquida.
    cuota = monto / plazo_meses
    dti = np.clip(cuota / renta + rng.normal(0, 0.04, n), 0.02, 0.85)

    # Pricing por riesgo observable, como en originacion real.
    tasa_anual = np.clip(
        18.0 + 3.2 * n_moras_12m + 6.0 * informal + 8.0 * dti + rng.normal(0, 2.0, n),
        9.0, 45.0,
    )

    region = rng.choice(REGIONES, size=n)

    log_renta = np.log(renta)
    log_monto = np.log(monto)

    z, moments = {}, {}
    for name, arr in [
        ("edad_z", edad.astype(float)),
        ("log_renta_z", log_renta),
        ("dti_z", dti),
        ("antiguedad_z", antiguedad.astype(float)),
        ("moras_z", n_moras_12m.astype(float)),
        ("log_monto_z", log_monto),
        ("tasa_z", tasa_anual),
        ("plazo_z", plazo_meses.astype(float)),
    ]:
        z[name], mu, sd = _zscore(arr)
        moments[name] = {"mean": mu, "std": sd}

    eta = np.zeros(n)
    for name, beta in BETA_TRUE.items():
        eta += beta * z[name]

    # --- simulacion mes a mes con riesgos competitivos --------------------
    ventana = np.minimum(plazo_meses, horizonte)

    duracion = np.zeros(n, dtype=int)
    evento = np.zeros(n, dtype=int)          # 1 = default, 0 = censurado
    motivo = np.empty(n, dtype=object)

    activo = np.ones(n, dtype=bool)
    h_prepago = np.exp(PREPAGO_C0 + PREPAGO_C1 * z["log_renta_z"])
    p_prepago = 1.0 - np.exp(-h_prepago)

    for t in range(1, horizonte + 1):
        # Los que ya pasaron su ventana de observacion salen censurados.
        fin_ventana = activo & (ventana < t)
        duracion[fin_ventana] = ventana[fin_ventana]
        motivo[fin_ventana] = "fin_ventana"
        activo[fin_ventana] = False
        if not activo.any():
            break

        idx = np.where(activo)[0]
        t_arr = np.full(idx.size, t)
        log_h = (
            baseline_log_hazard(t_arr)
            + eta[idx]
            + gamma_informal(t_arr) * informal[idx]
        )
        p_default = 1.0 - np.exp(-np.exp(log_h))

        cae = rng.random(idx.size) < p_default
        idx_default = idx[cae]
        duracion[idx_default] = t
        evento[idx_default] = 1
        motivo[idx_default] = "default"
        activo[idx_default] = False

        vivos = idx[~cae]
        prepaga = rng.random(vivos.size) < p_prepago[vivos]
        idx_prepago = vivos[prepaga]
        duracion[idx_prepago] = t
        motivo[idx_prepago] = "prepago"
        activo[idx_prepago] = False

    # Lo que sigue vivo al final del horizonte: censura administrativa.
    duracion[activo] = np.minimum(ventana[activo], horizonte)
    motivo[activo] = "fin_ventana"

    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "vintage": vintages[vintage_idx],
        "edad": edad,
        "tipo_contrato": tipo_contrato,
        "renta_liquida": renta.round(0),
        "antiguedad_laboral_meses": antiguedad,
        "n_moras_12m": n_moras_12m,
        "monto_credito": monto.round(0),
        "plazo_meses": plazo_meses,
        "tasa_anual": tasa_anual.round(2),
        "dti": dti.round(4),
        "region": region,
        "ventana_obs_meses": ventana,
        "duracion_meses": duracion,
        "evento_default": evento,
        "motivo_salida": motivo,
    })

    ground_truth = {
        "beta_true": BETA_TRUE,
        "gamma_informal": {"g0": GAMMA_INFORMAL_0, "tau": GAMMA_INFORMAL_TAU},
        "baseline": {"a0": BASELINE_A0, "a1": BASELINE_A1, "a2": BASELINE_A2},
        "standardization_moments": moments,
        "n_loans": int(n),
        "horizonte_meses": int(horizonte),
        "seed": int(seed),
    }
    return df, ground_truth


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df, gt = generate_portfolio()
    df.to_csv(RAW_DIR / "loan_book.csv", index=False)
    (RAW_DIR / "ground_truth.json").write_text(json.dumps(gt, indent=2))

    print(f"Creditos generados      : {len(df):,}")
    print(f"Defaults observados     : {int(df['evento_default'].sum()):,} "
          f"({df['evento_default'].mean():.2%})")
    print(f"Censurados              : {(df['evento_default'] == 0).mean():.2%}")
    print(df["motivo_salida"].value_counts().to_string())
    print(f"Duracion mediana (meses): {df['duracion_meses'].median():.0f}")
    print(f"-> {RAW_DIR / 'loan_book.csv'}")


if __name__ == "__main__":
    main()
