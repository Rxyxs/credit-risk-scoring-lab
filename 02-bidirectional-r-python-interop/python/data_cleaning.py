"""Genera solicitudes de crédito sintéticas pero realistas (con problemas de calidad
de datos inyectados a propósito) y define el pipeline de limpieza que los resuelve.

La probabilidad de default depende causalmente de las features (DTI, historial de
mora, score de buró, tipo de contrato) a través de un factor de riesgo latente
compartido -- no es una etiqueta aleatoria independiente de los datos, para que los
modelos de la Fase 2 tengan señal real que aprender.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42

CONTRATO_TIPOS = ["Indefinido", "Plazo Fijo", "Honorarios"]
CONTRATO_PROBS = [0.55, 0.25, 0.20]


def generate_raw_credit_data(n: int = 8000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Factor de riesgo latente no observado: correlaciona score de buro, historial
    # de mora y estabilidad laboral sin que unos causen directamente a los otros
    # (todos son consecuencia de una propensión de riesgo subyacente compartida).
    latent_risk = rng.normal(0, 1, n)

    edad = np.clip(rng.normal(40, 12, n), 18, 75).round(0)
    antiguedad_laboral_meses = np.clip(
        rng.gamma(shape=2.0, scale=24.0, size=n) - 6 * latent_risk, 0, 420
    ).round(0)
    tipo_contrato = rng.choice(CONTRATO_TIPOS, size=n, p=CONTRATO_PROBS)

    ingreso_mensual_clp = np.clip(rng.lognormal(mean=13.6, sigma=0.45, size=n), 350_000, 8_000_000).round(-3)
    deuda_total_clp = np.clip(
        ingreso_mensual_clp * rng.uniform(0.5, 6.0, n) * (1 + 0.3 * np.clip(latent_risk, -2, 2)),
        0, None,
    ).round(-3)
    monto_solicitado_clp = np.clip(
        ingreso_mensual_clp * rng.uniform(2.0, 12.0, n), 500_000, 40_000_000
    ).round(-3)
    plazo_meses = rng.choice([12, 24, 36, 48, 60], size=n)
    tasa_interes_pct = np.clip(rng.normal(18, 5, n) + 3 * np.clip(latent_risk, -2, 2), 6, 45).round(2)
    num_productos_financieros = np.clip(rng.poisson(2.2, n), 0, 8)

    score_buro_externo = np.clip(650 - 80 * latent_risk + rng.normal(0, 35, n), 300, 850).round(0)
    historial_moroso = (rng.uniform(0, 1, n) < _sigmoid(1.2 * latent_risk - 1.4)).astype(int)

    dti = deuda_total_clp / (ingreso_mensual_clp * 12)
    logit_p_default = (
        -3.0
        + 2.2 * np.clip(dti, 0, 3)
        + 1.1 * historial_moroso
        - 0.0045 * (score_buro_externo - 650)
        + 0.35 * (tipo_contrato == "Honorarios")
        - 0.10 * np.log1p(antiguedad_laboral_meses)
        + rng.normal(0, 0.4, n)
    )
    p_default = _sigmoid(logit_p_default)
    default = (rng.uniform(0, 1, n) < p_default).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n)],
            "edad": edad,
            "ingreso_mensual_clp": ingreso_mensual_clp,
            "antiguedad_laboral_meses": antiguedad_laboral_meses,
            "tipo_contrato": tipo_contrato,
            "deuda_total_clp": deuda_total_clp,
            "monto_solicitado_clp": monto_solicitado_clp,
            "plazo_meses": plazo_meses,
            "tasa_interes_pct": tasa_interes_pct,
            "num_productos_financieros": num_productos_financieros,
            "score_buro_externo": score_buro_externo,
            "historial_moroso": historial_moroso,
            "default": default,
        }
    )

    return _inject_data_quality_issues(df, rng)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _inject_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inyecta problemas de calidad de datos realistas -- todo lo que el pipeline
    de limpieza de `clean_credit_data()` debe detectar y resolver explícitamente."""
    df = df.copy()
    n = len(df)

    # Nulos: MCAR en antiguedad laboral, MAR en score de buro (clientes nuevos sin
    # historial crediticio previo suelen no tener score -- correlacionado con edad).
    df.loc[rng.uniform(0, 1, n) < 0.05, "ingreso_mensual_clp"] = np.nan
    mar_mask = (df["edad"] < 25) | (rng.uniform(0, 1, n) < 0.05)
    df.loc[mar_mask & (rng.uniform(0, 1, n) < 0.6), "score_buro_externo"] = np.nan
    df.loc[rng.uniform(0, 1, n) < 0.03, "antiguedad_laboral_meses"] = np.nan

    # Codificacion inconsistente de categoricos.
    contrato_variants = {
        "Indefinido": ["Indefinido", "INDEFINIDO", " indefinido", "indefinido "],
        "Plazo Fijo": ["Plazo Fijo", "PLAZO FIJO", "plazo fijo", "Plazo fijo"],
        "Honorarios": ["Honorarios", "HONORARIOS", "honorarios", " Honorarios"],
    }
    messy_idx = df.sample(frac=0.35, random_state=int(rng.integers(0, 1_000_000))).index
    df.loc[messy_idx, "tipo_contrato"] = df.loc[messy_idx, "tipo_contrato"].apply(
        lambda t: rng.choice(contrato_variants[t])
    )

    moroso_encodings = {0: ["0", "N", "No", "no"], 1: ["1", "Y", "Si", "si"]}
    moroso_idx = df.sample(frac=0.3, random_state=int(rng.integers(0, 1_000_000))).index
    df["historial_moroso"] = df["historial_moroso"].astype(object)
    df.loc[moroso_idx, "historial_moroso"] = df.loc[moroso_idx, "historial_moroso"].apply(
        lambda v: rng.choice(moroso_encodings[v])
    )

    # Outliers / errores de digitacion.
    outlier_idx = df.sample(frac=0.005, random_state=int(rng.integers(0, 1_000_000))).index
    df.loc[outlier_idx, "ingreso_mensual_clp"] = df.loc[outlier_idx, "ingreso_mensual_clp"] * 100
    neg_age_idx = df.sample(frac=0.003, random_state=int(rng.integers(0, 1_000_000))).index
    df.loc[neg_age_idx, "edad"] = -df.loc[neg_age_idx, "edad"]

    # Filas duplicadas (reenvio accidental de la misma solicitud).
    dup_rows = df.sample(frac=0.02, random_state=int(rng.integers(0, 1_000_000)))
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


def clean_credit_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pipeline de limpieza documentado -- cada paso justifica su decision y reporta
    cuantas filas/valores afecto, para que la limpieza sea auditable, no una caja negra."""
    report: dict = {}
    df = df_raw.copy()

    # 1. Deduplicar (mismo customer_id + mismos valores = reenvio accidental).
    n_before = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    report["duplicados_eliminados"] = n_before - len(df)

    # 2. Estandarizar categoricos (strip + upper-case canonico).
    df["tipo_contrato"] = df["tipo_contrato"].str.strip().str.title()
    df["tipo_contrato"] = df["tipo_contrato"].replace({"Plazo Fijo": "Plazo Fijo"})  # no-op explicito

    moroso_map = {
        "0": 0, "n": 0, "no": 0,
        "1": 1, "y": 1, "si": 1,
    }
    df["historial_moroso"] = (
        df["historial_moroso"].astype(str).str.strip().str.lower().map(moroso_map)
    )
    report["historial_moroso_no_mapeado"] = int(df["historial_moroso"].isna().sum())

    # 3. Outliers: edad fuera de rango plausible -> tratar como dato invalido (NaN,
    #    luego imputado en el paso 5), no eliminar la fila completa.
    edad_invalida = ~df["edad"].between(18, 90)
    report["edad_fuera_de_rango"] = int(edad_invalida.sum())
    df.loc[edad_invalida, "edad"] = np.nan

    # Ingreso: outliers por sobre el percentil 99.5 tratados como error de tipeo
    # (probable digito extra), no como ingresos altos legitimos -- se acotan
    # (winsorizacion) en vez de eliminarse, preservando el resto de la fila.
    p995 = df["ingreso_mensual_clp"].quantile(0.995)
    ingresos_extremos = df["ingreso_mensual_clp"] > p995
    report["ingresos_winsorizados"] = int(ingresos_extremos.sum())
    df.loc[ingresos_extremos, "ingreso_mensual_clp"] = p995

    # 4. Indicadores explicitos de "faltaba" ANTES de imputar -- la ausencia de
    #    score de buro es en si misma informativa (cliente nuevo sin historial),
    #    asi que se preserva como feature en vez de perderse en la imputacion.
    df["score_buro_faltante"] = df["score_buro_externo"].isna().astype(int)
    df["ingreso_faltante"] = df["ingreso_mensual_clp"].isna().astype(int)

    # 5. Imputacion (mediana para numericos -- robusta a outliers residuales; moda
    #    para historial_moroso no mapeado, tratando el caso ambiguo como "sin mora").
    for col in ["ingreso_mensual_clp", "antiguedad_laboral_meses", "score_buro_externo", "edad"]:
        df[col] = df[col].fillna(df[col].median())
    df["historial_moroso"] = df["historial_moroso"].fillna(0).astype(int)

    report["filas_finales"] = len(df)
    report["filas_originales"] = len(df_raw)

    return df, report


if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = generate_raw_credit_data()
    raw.to_csv(data_dir / "raw_credit_applications.csv", index=False)
    print(f"Datos crudos generados: {len(raw)} filas -> {data_dir / 'raw_credit_applications.csv'}")

    clean, cleaning_report = clean_credit_data(raw)
    clean.to_csv(data_dir / "clean_credit_applications.csv", index=False)
    print(f"Datos limpios: {len(clean)} filas -> {data_dir / 'clean_credit_applications.csv'}")
    print("\nReporte de limpieza:")
    for k, v in cleaning_report.items():
        print(f"  {k}: {v}")
    print(f"\nTasa de default (limpio): {clean['default'].mean():.2%}")
