"""Generador sintetico para una auditoria de trato justo en credito.

El diseno del simulador es el argumento del proyecto, asi que conviene ser
explicito sobre que se construyo y por que:

1. **El genero NO entra en el proceso generador del default.** El riesgo
   depende de carga financiera, morosidad, utilizacion de lineas, renta,
   antiguedad y edad. Ningun termino de genero, ni directo ni escondido.
   Cualquier disparidad que aparezca despues es, por construccion, o bien
   consecuencia de diferencias reales en esos factores, o bien un artefacto
   del modelo -- nunca un efecto causal del genero sobre el pago.
2. **El genero si correlaciona con factores legitimos.** Hay una brecha de
   ingresos y una diferencia de antiguedad laboral, que son hechos del
   mercado laboral chileno, no supuestos sobre las personas. Eso basta para
   producir **impacto dispar sin trato dispar**: el modelo nunca ve el
   genero y aun asi aprueba menos mujeres.
3. **Hay un proxy deliberado.** El sector laboral esta fuertemente
   segregado por genero (salud y educacion mayoritariamente femeninos;
   construccion y mineria mayoritariamente masculinos) pero aporta muy poca
   informacion de riesgo propia. Es el caso de manual de la
   discriminacion por proxy: una variable que transmite pertenencia al
   grupo casi sin pagar con poder predictivo.

Sin (1) el proyecto no podria distinguir disparidad de causalidad; sin (2)
no habria nada que auditar; sin (3) no habria nada que mitigar.

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

ATRIBUTO_PROTEGIDO = "genero"
GRUPO_REFERENCIA = "M"
GRUPO_PROTEGIDO = "F"

# Sectores laborales: proporcion de mujeres y efecto real sobre el riesgo.
# El efecto de riesgo es chico y NO esta alineado con la composicion de
# genero: por eso el sector funciona como proxy y no como factor legitimo
# dominante.
SECTORES = {
    "salud":         {"p_mujer": 0.80, "riesgo": -0.05, "peso": 0.11},
    "educacion":     {"p_mujer": 0.78, "riesgo": -0.02, "peso": 0.10},
    "retail":        {"p_mujer": 0.58, "riesgo": 0.10, "peso": 0.16},
    "servicios":     {"p_mujer": 0.52, "riesgo": 0.04, "peso": 0.18},
    "administracion": {"p_mujer": 0.55, "riesgo": -0.03, "peso": 0.13},
    "transporte":    {"p_mujer": 0.22, "riesgo": 0.08, "peso": 0.11},
    "construccion":  {"p_mujer": 0.14, "riesgo": 0.12, "peso": 0.12},
    "mineria":       {"p_mujer": 0.15, "riesgo": -0.06, "peso": 0.09},
}

# Brecha salarial y de antiguedad: hechos del mercado laboral, no del riesgo.
BRECHA_RENTA = 0.86          # la renta mediana femenina es 14% menor
BRECHA_ANTIGUEDAD = 0.82     # trayectorias mas interrumpidas

INTERCEPTO = -2.85

FEATURES_MODELO = [
    "dti", "n_moras_12m", "utilizacion_lineas", "consultas_6m",
    "log_renta", "antiguedad_laboral_meses", "edad", "sector",
]
FEATURES_NUMERICAS = [f for f in FEATURES_MODELO if f != "sector"]


def _logit_riesgo(df: pd.DataFrame) -> np.ndarray:
    """Log-odds del default. Notar que `genero` no aparece."""
    eta = np.full(len(df), INTERCEPTO)
    eta += 1.00 * df["dti"].to_numpy(float)
    eta += 4.20 * np.clip(df["dti"].to_numpy(float) - 0.20, 0, None)
    eta += 0.62 * df["n_moras_12m"].to_numpy(float)
    eta += 1.15 * df["utilizacion_lineas"].to_numpy(float)
    eta += 0.15 * df["consultas_6m"].to_numpy(float)
    eta += -0.60 * np.clip(df["log_renta"].to_numpy(float) - 12.9, 0, None)
    eta += -0.0038 * df["antiguedad_laboral_meses"].to_numpy(float)
    eta += 0.0020 * (df["edad"].to_numpy(float) - 45.0) ** 2
    eta += df["sector"].map({s: v["riesgo"] for s, v in SECTORES.items()}).to_numpy(float)
    return eta


def generate_applicants(n: int = 20_000, seed: int = RANDOM_STATE_DEFAULT) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sectores = list(SECTORES)
    pesos = np.array([SECTORES[s]["peso"] for s in sectores], dtype=float)
    pesos = pesos / pesos.sum()
    sector = rng.choice(sectores, size=n, p=pesos)

    p_mujer = np.array([SECTORES[s]["p_mujer"] for s in sector])
    es_mujer = rng.random(n) < p_mujer
    genero = np.where(es_mujer, GRUPO_PROTEGIDO, GRUPO_REFERENCIA)

    informal = rng.random(n) < 0.30
    renta_base = np.where(informal, rng.lognormal(13.30, 0.55, n),
                          rng.lognormal(13.70, 0.45, n))
    renta = np.clip(renta_base * np.where(es_mujer, BRECHA_RENTA, 1.0),
                    320_000, 8_000_000)

    antiguedad_base = np.where(informal,
                               np.clip(rng.exponential(20, n), 1, 300),
                               np.clip(rng.exponential(48, n), 1, 420))
    antiguedad = np.clip(
        antiguedad_base * np.where(es_mujer, BRECHA_ANTIGUEDAD, 1.0), 1, 420
    ).round().astype(int)

    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)
    moras = np.clip(rng.poisson(0.30 + 0.40 * informal + 0.25 * (renta < 600_000)), 0, 6)
    utilizacion = np.clip(rng.beta(2.2, 3.0, n), 0.0, 1.0)
    consultas = np.clip(rng.poisson(1.1 + 0.7 * informal), 0, 12)

    plazo = rng.choice([12, 24, 36, 48], size=n, p=[0.18, 0.34, 0.33, 0.15])
    monto = np.clip(renta * rng.lognormal(1.05, 0.45, n), 300_000, 30_000_000)
    dti = np.clip((monto / plazo) / renta + rng.normal(0, 0.04, n), 0.02, 0.85)

    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "genero": genero,
        "sector": sector,
        "tipo_contrato": np.where(informal, "informal", "formal"),
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

    eta = _logit_riesgo(df)
    df["pd_verdadera"] = 1.0 / (1.0 + np.exp(-eta))
    df["default_12m"] = (rng.random(n) < df["pd_verdadera"]).astype(int)
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_applicants()
    df.to_csv(RAW_DIR / "applicants.csv", index=False)
    (RAW_DIR / "diseno_simulador.json").write_text(json.dumps({
        "atributo_protegido": ATRIBUTO_PROTEGIDO,
        "grupo_protegido": GRUPO_PROTEGIDO,
        "grupo_referencia": GRUPO_REFERENCIA,
        "genero_en_el_proceso_generador": False,
        "brecha_renta": BRECHA_RENTA,
        "brecha_antiguedad": BRECHA_ANTIGUEDAD,
        "sectores": SECTORES,
        "features_modelo": FEATURES_MODELO,
    }, indent=2))

    por_genero = df.groupby("genero").agg(
        n=("applicant_id", "size"),
        default=("default_12m", "mean"),
        renta=("renta_liquida", "median"),
        antiguedad=("antiguedad_laboral_meses", "median"),
        dti=("dti", "mean"),
    )
    print(f"Solicitudes: {len(df):,} | tasa de default {df['default_12m'].mean():.2%}")
    print("\nPor grupo (el genero NO esta en el proceso generador del default):")
    print(por_genero.round(4).to_string())

    brecha = (por_genero.loc[GRUPO_PROTEGIDO, "default"]
              - por_genero.loc[GRUPO_REFERENCIA, "default"])
    print(f"\nBrecha observada en tasa de default: {brecha * 100:+.2f} pp "
          f"(consecuencia de las diferencias de renta y antiguedad, no del genero)")

    print("\nComposicion de genero por sector (el proxy):")
    comp = df.groupby("sector").agg(
        n=("applicant_id", "size"),
        pct_mujeres=("genero", lambda s: (s == GRUPO_PROTEGIDO).mean()),
        default=("default_12m", "mean"),
    ).sort_values("pct_mujeres", ascending=False)
    print(comp.round(4).to_string())
    print(f"-> {RAW_DIR / 'applicants.csv'}")


if __name__ == "__main__":
    main()
