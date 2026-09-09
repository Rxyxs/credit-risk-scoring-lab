"""Seis bancos, seis carteras que nunca se pueden juntar en una sola tabla.

El secreto bancario y la ley de proteccion de datos impiden que un banco
regional le pase su base de clientes a otro, o a un consorcio que quiera
entrenar un modelo conjunto. Cada banco solo ve a sus propios solicitantes.
El problema que eso genera no es solo legal: un modelo entrenado unicamente
con los clientes de un banco chico y especializado hereda el sesgo de esa
cartera, y le va mal con cualquier solicitante que no se parezca a los
suyos.

El simulador genera dos versiones de la misma cartera nacional, repartida
en 6 bancos, para poder medir exactamente ese problema:

- **IID**: los 6 bancos son 6 muestras aleatorias de la **misma**
  poblacion. Ningun banco tiene un sesgo propio -- solo tienen menos datos
  que el conjunto nacional.
- **No-IID (el caso real)**: cada banco sirve a un segmento geografico y
  socioeconomico distinto -- el banco minero del norte, el banco agricola
  del sur, el banco de microcreditos informales, el banco privado de renta
  alta -- con distribuciones de renta, informalidad y carga financiera
  bien distintas entre si ("covariate shift"). Y ademas, cada banco pesa
  la carga financiera un poco distinto y parte de un nivel de riesgo base
  propio ("concept shift" moderado: `DESVIO_CONCEPTO_NO_IID`) -- reflejo
  de politicas de garantias y cobranza distintas, no de una diferencia
  arbitraria. Las dos formas de heterogeneidad juntas son las que la
  literatura de aprendizaje federado identifica como el caso dificil real:
  con covariate shift puro, un banco chico igual identifica con precision
  razonable la relacion nacional a partir de unos pocos miles de casos (el
  parametro que se esta estimando sigue siendo el mismo en todas partes);
  el client drift que motiva a FedAvg aparece cuando la relacion misma
  cambia de banco en banco.

No usa datos reales de ninguna institucion ni del Boletin Comercial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"

RANDOM_STATE_DEFAULT = 42

FEATURES = [
    "dti", "n_moras_12m", "utilizacion_lineas", "consultas_6m",
    "log_renta_z", "antiguedad_z",
]

# La relacion entre riesgo y variables es UNA sola en todo el pais.
BETA_TRUE = {
    "dti": 1.65,
    "n_moras_12m": 0.40,
    "utilizacion_lineas": 0.75,
    "consultas_6m": 0.12,
    "log_renta_z": -0.30,
    "antiguedad_z": -0.22,
}
INTERCEPTO_TRUE = -1.55

# Cada banco es una region/segmento con una poblacion propia. Los
# parametros crudos (no estandarizados) son los que cambian entre bancos;
# la estandarizacion nacional (misma media/sd para todos) se aplica
# despues, para que el corrimiento de distribucion sea visible en las
# features del modelo y no se lo trague un z-score local por banco.
BANCOS = {
    "Banco_Minero_Norte": dict(informal_p=0.20, renta_mu=13.95, renta_sigma=0.40,
                               antiguedad_mu=52, moras_lambda=0.22, n=1400),
    "Banco_Agricola_Sur": dict(informal_p=0.42, renta_mu=13.35, renta_sigma=0.50,
                               antiguedad_mu=30, moras_lambda=0.45, n=1100),
    "Banco_Microcredito_Informal": dict(informal_p=0.78, renta_mu=12.85, renta_sigma=0.55,
                                        antiguedad_mu=14, moras_lambda=0.70, n=700),
    "Banco_Privado_Metropolitano": dict(informal_p=0.05, renta_mu=14.55, renta_sigma=0.35,
                                        antiguedad_mu=68, moras_lambda=0.10, n=1600),
    "Banco_Regional_Centro": dict(informal_p=0.30, renta_mu=13.60, renta_sigma=0.45,
                                  antiguedad_mu=40, moras_lambda=0.32, n=1300),
    "Banco_Cooperativo_Sur": dict(informal_p=0.35, renta_mu=13.45, renta_sigma=0.48,
                                  antiguedad_mu=36, moras_lambda=0.38, n=900),
}

# Parametros "nacionales" para el escenario IID: todos los bancos sortean
# de esta misma distribucion, sin sesgo propio.
POBLACION_NACIONAL = dict(informal_p=0.31, renta_mu=13.70, renta_sigma=0.48,
                          antiguedad_mu=42, moras_lambda=0.35)

# En el escenario no-IID cada banco ademas pesa el DTI distinto y parte de
# un nivel de riesgo base distinto -- no solo ve clientes distintos, sino
# que su propia relacion riesgo-variables se desvia un poco de la
# nacional. Es real: un banco que exige garantias reales pondera la carga
# financiera distinto de uno que presta sin colateral, y las politicas de
# cobranza cambian el nivel base de perdida entre instituciones. Sin este
# segundo tipo de heterogeneidad ("concept shift"), un banco chico igual
# identifica la relacion nacional con precision razonable a partir de
# unos pocos miles de casos -- el shift de covariables por si solo no
# alcanza a mostrar el client drift que motiva este proyecto.
DESVIO_CONCEPTO_NO_IID = {
    "Banco_Minero_Norte": {"mult_dti": 1.35, "delta_intercepto": -0.15},
    "Banco_Agricola_Sur": {"mult_dti": 0.65, "delta_intercepto": 0.10},
    "Banco_Microcredito_Informal": {"mult_dti": 1.70, "delta_intercepto": 0.40},
    "Banco_Privado_Metropolitano": {"mult_dti": 0.55, "delta_intercepto": -0.40},
    "Banco_Regional_Centro": {"mult_dti": 1.00, "delta_intercepto": 0.00},
    "Banco_Cooperativo_Sur": {"mult_dti": 1.20, "delta_intercepto": 0.08},
}


def _generar_banco(nombre: str, params: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = params["n"]

    informal = rng.random(n) < params["informal_p"]
    renta = np.clip(rng.lognormal(params["renta_mu"], params["renta_sigma"], n),
                    320_000, 8_000_000)
    antiguedad = np.clip(
        rng.exponential(params["antiguedad_mu"], n), 1, 420
    ).round().astype(int)
    moras = np.clip(rng.poisson(params["moras_lambda"] + 0.25 * informal, n), 0, 6)
    utilizacion = np.clip(rng.beta(2.2, 3.0, n), 0.0, 1.0)
    consultas = np.clip(rng.poisson(1.0 + 0.6 * informal, n), 0, 12)
    plazo = rng.choice([12, 24, 36, 48], size=n, p=[0.18, 0.34, 0.33, 0.15])
    monto = np.clip(renta * rng.lognormal(1.05, 0.45, n), 300_000, 30_000_000)
    dti = np.clip((monto / plazo) / renta + rng.normal(0, 0.04, n), 0.02, 0.85)

    return pd.DataFrame({
        "banco": nombre,
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
    })


def generar_carteras(escenario: str = "no_iid", seed: int = RANDOM_STATE_DEFAULT
                     ) -> pd.DataFrame:
    """Genera las 6 carteras bancarias, en el escenario IID o no-IID."""
    if escenario not in {"iid", "no_iid"}:
        raise ValueError("escenario debe ser 'iid' o 'no_iid'")

    partes = []
    for i, (nombre, params) in enumerate(BANCOS.items()):
        p = dict(POBLACION_NACIONAL, n=params["n"]) if escenario == "iid" else params
        partes.append(_generar_banco(nombre, p, seed=seed + 1000 * i))
    df = pd.concat(partes, ignore_index=True)
    df.insert(0, "applicant_id", np.arange(1, len(df) + 1))

    # Estandarizacion NACIONAL (una sola media/sd para todo el pais): asi
    # el corrimiento de distribucion entre bancos queda visible en las
    # features del modelo, en vez de que cada banco se autonormalice y
    # el corrimiento desaparezca de las variables que ve el algoritmo.
    log_renta = df["log_renta"].to_numpy()
    antiguedad = df["antiguedad_laboral_meses"].to_numpy(float)
    df["log_renta_z"] = (log_renta - log_renta.mean()) / log_renta.std()
    df["antiguedad_z"] = (antiguedad - antiguedad.mean()) / antiguedad.std()

    rng = np.random.default_rng(seed + 999)
    mult_dti = np.ones(len(df))
    delta_intercepto = np.zeros(len(df))
    if escenario == "no_iid":
        for banco, ajuste in DESVIO_CONCEPTO_NO_IID.items():
            m = (df["banco"] == banco).to_numpy()
            mult_dti[m] = ajuste["mult_dti"]
            delta_intercepto[m] = ajuste["delta_intercepto"]

    eta = np.full(len(df), INTERCEPTO_TRUE) + delta_intercepto
    for f, b in BETA_TRUE.items():
        coef = b * mult_dti if f == "dti" else b
        eta += coef * df[f].to_numpy(float)
    df["pd_verdadera"] = norm.cdf(eta)
    df["default_12m"] = (rng.random(len(df)) < df["pd_verdadera"]).astype(int)
    df["escenario"] = escenario
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for escenario in ("iid", "no_iid"):
        df = generar_carteras(escenario=escenario)
        df.to_csv(RAW_DIR / f"carteras_{escenario}.csv", index=False)

        por_banco = df.groupby("banco").agg(
            n=("applicant_id", "size"),
            pct_informal=("tipo_contrato", lambda s: (s == "informal").mean()),
            renta_mediana=("renta_liquida", "median"),
            default=("default_12m", "mean"),
        )
        resumen[escenario] = por_banco.to_dict(orient="index")

        print(f"\n=== Escenario {escenario.upper()} ===")
        print(f"{'banco':<30} {'n':>6} {'% informal':>11} {'renta mediana':>14} "
              f"{'default':>8}")
        for banco, r in por_banco.iterrows():
            print(f"{banco:<30} {r['n']:>6.0f} {r['pct_informal']:>11.1%} "
                  f"{r['renta_mediana']:>14,.0f} {r['default']:>8.2%}")

    (RAW_DIR / "ground_truth.json").write_text(json.dumps({
        "features": FEATURES, "beta_true": BETA_TRUE, "intercepto_true": INTERCEPTO_TRUE,
        "bancos": list(BANCOS), "resumen": resumen,
    }, indent=2))
    print(f"\nEn el escenario IID, las filas de arriba deberian verse casi identicas")
    print("entre bancos (misma poblacion, distinta muestra). En NO_IID, deberian")
    print("verse marcadamente distintas -- ese es el problema que el proyecto ataca.")
    print(f"-> {RAW_DIR}")


if __name__ == "__main__":
    main()
