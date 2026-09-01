"""Generador sintetico de solicitantes de credito de consumo (banca/retail
financiero chileno).

No usa datos reales de ninguna institucion ni del Boletin Comercial/DICOM:
es una simulacion con un proceso generador de default conocido (regresion
logistica latente sobre factores de riesgo estandar de la industria), para
que el pipeline completo (limpieza -> WOE/IV -> scorecard -> ML -> motor en
C) tenga una senal real y aprendible que validar de punta a punta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIONES = [
    "Metropolitana", "Valparaiso", "Biobio", "Antofagasta",
    "Araucania", "Maule", "Los Lagos", "Coquimbo",
]
REGION_RISK_ADJ = {
    "Metropolitana": -0.05, "Valparaiso": 0.05, "Biobio": 0.10,
    "Antofagasta": -0.10, "Araucania": 0.20, "Maule": 0.10,
    "Los Lagos": 0.05, "Coquimbo": 0.10,
}

RANDOM_STATE_DEFAULT = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_applicants(n: int = 20_000, seed: int = RANDOM_STATE_DEFAULT) -> pd.DataFrame:
    """Genera `n` solicitantes con un proceso de default logistico latente.

    El coeficiente mas fuerte es el historial de morosidad (consistente con
    la practica real de riesgo de credito: el pasado de pago es, con
    diferencia, el mejor predictor individual de default futuro).
    """
    rng = np.random.default_rng(seed)

    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)

    tipo_contrato = rng.choice(["formal", "informal"], size=n, p=[0.68, 0.32])
    es_informal = (tipo_contrato == "informal").astype(float)

    renta_base = np.where(
        tipo_contrato == "formal",
        rng.lognormal(mean=13.7, sigma=0.45, size=n),   # ~CLP 890k mediana aprox
        rng.lognormal(mean=13.3, sigma=0.55, size=n),   # informal: mediana mas baja, mas dispersa
    )
    renta_liquida = np.clip(renta_base, 320_000, 8_000_000)

    antiguedad_laboral_meses = np.where(
        tipo_contrato == "formal",
        np.clip(rng.exponential(48, n), 1, 420),
        np.clip(rng.exponential(20, n), 1, 300),
    ).round().astype(int)

    n_productos_activos = np.clip(rng.poisson(2.1, n), 0, 8)

    deuda_total = np.clip(
        renta_liquida * rng.gamma(shape=2.2, scale=0.35, size=n), 0, renta_liquida * 6
    )
    dti = np.clip(deuda_total / renta_liquida, 0.0, 6.0)

    # Reportes de morosidad comercial: proxy sintetico inspirado en el
    # concepto de boletines comerciales (Boletin de Informaciones
    # Comerciales / DICOM), SIN usar ni imitar datos reales de ningun
    # registro -- es una variable de conteo generada, no un dato de bureau.
    lambda_moros = np.clip(0.10 + 0.35 * es_informal + 0.5 * (dti > 2.0), 0.03, None)
    n_morosidad_reportes = rng.poisson(lambda_moros, n)

    region = rng.choice(REGIONES, size=n)
    region_risk = np.array([REGION_RISK_ADJ[r] for r in region])

    log_renta = np.log(renta_liquida)
    tenure_effect = -0.30 * np.log1p(antiguedad_laboral_meses / 12.0)

    # Calibrado para una tasa de default de cartera ~7-8% (rango citado para
    # segmentos de credito de consumo/retail en Chile), con el segmento
    # informal ~3-4x mas riesgoso que el formal -- no saturado cerca de 0%/100%
    # incluso en las colas de morosidad, a diferencia de una primera
    # calibracion que daba ~19% de tasa base y colas casi deterministas.
    logit_pd = (
        -3.20
        + 0.75 * es_informal
        + 0.45 * dti
        - 0.55 * (log_renta - log_renta.mean())
        + tenure_effect
        + 0.55 * n_morosidad_reportes
        - 0.08 * n_productos_activos
        + 0.010 * np.clip(30 - edad, 0, None)  # mas riesgo en solicitantes jovenes
        + region_risk
        + rng.normal(0, 0.30, n)  # heterogeneidad no observada
    )
    pd_default = _sigmoid(logit_pd)
    default_12m = (rng.random(n) < pd_default).astype(int)

    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "edad": edad,
        "region": region,
        "tipo_contrato": tipo_contrato,
        "renta_liquida": renta_liquida.round(0),
        "antiguedad_laboral_meses": antiguedad_laboral_meses,
        "n_productos_activos": n_productos_activos,
        "deuda_total": deuda_total.round(0),
        "dti": dti.round(3),
        "n_morosidad_reportes": n_morosidad_reportes,
        "default_12m": default_12m,
    })
    return df


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_applicants()
    df.to_csv(out_dir / "applicants_raw.csv", index=False)
    print(f"Generados {len(df)} solicitantes -> {out_dir / 'applicants_raw.csv'}")
    print(f"Tasa de default: {df['default_12m'].mean() * 100:.2f}%")
    print(df.describe(include="all").T)
