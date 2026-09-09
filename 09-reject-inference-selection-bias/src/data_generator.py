"""Generador con el contrafactual que ningun banco tiene: el desenlace de
los solicitantes que fueron rechazados.

El problema del reject inference es que solo se observa el pago de quienes
fueron aprobados, y esa muestra la eligio una politica de credito anterior.
El modelo nuevo se entrena sobre una poblacion recortada, y nadie puede
verificar cuanto se equivoca fuera de ella -- porque los rechazados nunca
revelaron si habrian pagado.

Aca si se puede: el simulador genera la etiqueta de **todos**, aprobados y
rechazados, y despues aplica la politica historica. Eso convierte "este
metodo de reject inference funciona" en algo medible contra la verdad.

**Estructura latente.** Default y aprobacion se generan como dos indices
con errores normales correlacionados, que es exactamente el modelo que
supone el probit bivariado con seleccion:

    default   si   z_y + e_y > 0,     z_y = intercepto_y + x'beta
    aprobado  si   z_s + e_s > 0,     z_s = intercepto_s + x'alpha + tau*w

    e_y = gamma*u + sqrt(1 - gamma^2) * v_y
    e_s = delta*u + sqrt(1 - delta^2) * v_s

con u, v_y, v_s normales estandar independientes. Asi los dos errores
tienen varianza 1 y su correlacion es exactamente **rho = gamma * delta**,
un numero conocido contra el cual contrastar lo que estime cada metodo.

`u` es la informacion blanda del ejecutivo: la impresion de la entrevista,
el conocimiento del cliente. Existe, mueve el riesgo real, y no esta en
ninguna base de datos.

`w` es `presion_colocacion_z`: cuanta presion de meta comercial tenia la
sucursal el mes en que se evaluo la solicitud. Un ejecutivo bajo presion
aprueba mas -- sube `z_s` -- pero esa presion no cambia en nada la
capacidad de pago del solicitante, asi que no entra a `z_y`. Es la
**variable de exclusion**: la unica razon por la que el probit bivariado y
Heckman pueden identificar `rho` con algo mas que la curvatura de la
normal bivariada, y el proyecto lo demuestra ajustando los mismos dos
metodos con y sin ella.

Dos regimenes, y la distincion decide si el problema tiene arreglo:

- **MAR (seleccion sobre observables)**: delta = 0, luego rho = 0. La
  politica aprobaba mirando solo las variables que el modelo nuevo tambien
  tiene (mas la presion comercial, que es observable pero no predice
  riesgo).
- **MNAR (seleccion sobre no observables)**: delta > 0, luego rho != 0. La
  politica ademas usaba la informacion blanda, y los aprobados quedan
  seleccionados por algo que ninguna feature disponible puede explicar.

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
# La variable de exclusion: solo entra a la seleccion, nunca al desenlace.
INSTRUMENTO = "presion_colocacion_z"
FEATURES_SELECCION = FEATURES + [INSTRUMENTO]

# Coeficientes verdaderos del indice de default (escala probit).
BETA_TRUE = {
    "dti": 1.55,
    "n_moras_12m": 0.36,
    "utilizacion_lineas": 0.68,
    "consultas_6m": 0.09,
    "log_renta_z": -0.25,
    "antiguedad_z": -0.18,
}
INTERCEPTO_TRUE = -1.60

# Politica historica de aprobacion (escala probit).
ALPHA_SELECCION = {
    "dti": -1.35,
    "n_moras_12m": -0.45,
    "utilizacion_lineas": -0.55,
    "consultas_6m": -0.12,
    "log_renta_z": 0.32,
    "antiguedad_z": 0.20,
}
INTERCEPTO_SELECCION = 0.78
# Peso del instrumento en la seleccion. Fuerte a proposito: un instrumento
# debil identifica rho igual de mal que no tener ninguno.
TAU_INSTRUMENTO = 0.90

# Carga de la informacion blanda sobre cada error. rho = GAMMA_U * DELTA_U.
GAMMA_U = -0.60          # mejor impresion -> menos riesgo real
DELTA_U_MNAR = 0.70      # mejor impresion -> mas probable que lo aprueben
DELTA_U_MAR = 0.0


def rho_verdadero(regimen: str) -> float:
    """Correlacion verdadera entre los errores de seleccion y desenlace."""
    delta = DELTA_U_MNAR if regimen == "mnar" else DELTA_U_MAR
    return float(GAMMA_U * delta)


def generate_applicants(n: int = 30_000, seed: int = RANDOM_STATE_DEFAULT,
                        regimen: str = "mnar") -> pd.DataFrame:
    """Genera solicitantes, su desenlace real y la decision historica."""
    if regimen not in {"mar", "mnar"}:
        raise ValueError("regimen debe ser 'mar' o 'mnar'")
    rng = np.random.default_rng(seed)

    informal = rng.random(n) < 0.31
    renta = np.clip(
        np.where(informal, rng.lognormal(13.30, 0.55, n), rng.lognormal(13.70, 0.45, n)),
        320_000, 8_000_000,
    )
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

    log_renta = np.log(renta)
    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "edad": np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int),
        "tipo_contrato": np.where(informal, "informal", "formal"),
        "renta_liquida": renta.round(0),
        "log_renta_z": (log_renta - log_renta.mean()) / log_renta.std(),
        "antiguedad_z": (antiguedad - antiguedad.mean()) / antiguedad.std(),
        "antiguedad_laboral_meses": antiguedad,
        "n_moras_12m": moras,
        "utilizacion_lineas": utilizacion.round(4),
        "consultas_6m": consultas,
        "monto_credito": monto.round(0),
        "plazo_meses": plazo,
        "dti": dti.round(4),
    })

    # Presion comercial de la sucursal-mes: no depende de nada del
    # solicitante, y por diseno no entra al proceso de default.
    df[INSTRUMENTO] = rng.normal(0.0, 1.0, n)

    # --- errores correlacionados a traves de la informacion blanda -------
    delta = DELTA_U_MNAR if regimen == "mnar" else DELTA_U_MAR
    u = rng.normal(0.0, 1.0, n)
    e_y = GAMMA_U * u + np.sqrt(1.0 - GAMMA_U**2) * rng.normal(0.0, 1.0, n)
    e_s = delta * u + np.sqrt(1.0 - delta**2) * rng.normal(0.0, 1.0, n)
    df["soft_info_no_observada"] = u

    z_y = np.full(n, INTERCEPTO_TRUE, dtype=float)
    for f, b in BETA_TRUE.items():
        z_y += b * df[f].to_numpy(float)
    z_s = np.full(n, INTERCEPTO_SELECCION, dtype=float)
    for f, a in ALPHA_SELECCION.items():
        z_s += a * df[f].to_numpy(float)
    z_s += TAU_INSTRUMENTO * df[INSTRUMENTO].to_numpy(float)

    df["pd_verdadera"] = norm.cdf(z_y)          # PD marginal, sin la soft info
    df["default_12m_real"] = ((z_y + e_y) > 0).astype(int)
    df["aprobado"] = ((z_s + e_s) > 0).astype(int)
    df["score_politica_historica"] = z_s

    # Lo que el banco realmente tiene: la etiqueta solo de los aprobados.
    df["default_12m_observado"] = np.where(
        df["aprobado"] == 1, df["default_12m_real"], np.nan
    )
    df["regimen"] = regimen
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resumen = {}
    for regimen in ("mar", "mnar"):
        df = generate_applicants(regimen=regimen)
        df.to_csv(RAW_DIR / f"applicants_{regimen}.csv", index=False)

        aprobados = df[df["aprobado"] == 1]
        rechazados = df[df["aprobado"] == 0]
        resumen[regimen] = {
            "n": int(len(df)),
            "rho_verdadero": rho_verdadero(regimen),
            "tasa_aprobacion": float(df["aprobado"].mean()),
            "default_poblacion": float(df["default_12m_real"].mean()),
            "default_aprobados": float(aprobados["default_12m_real"].mean()),
            "default_rechazados": float(rechazados["default_12m_real"].mean()),
            "soft_info_media_aprobados": float(aprobados["soft_info_no_observada"].mean()),
            "soft_info_media_rechazados": float(rechazados["soft_info_no_observada"].mean()),
        }

    (RAW_DIR / "ground_truth.json").write_text(json.dumps({
        "beta_true": BETA_TRUE,
        "intercepto_true": INTERCEPTO_TRUE,
        "alpha_seleccion": ALPHA_SELECCION,
        "intercepto_seleccion": INTERCEPTO_SELECCION,
        "tau_instrumento": TAU_INSTRUMENTO,
        "gamma_u": GAMMA_U,
        "delta_u_mnar": DELTA_U_MNAR,
        "rho_mar": rho_verdadero("mar"),
        "rho_mnar": rho_verdadero("mnar"),
        "features": FEATURES,
        "features_seleccion": FEATURES_SELECCION,
        "instrumento": INSTRUMENTO,
        "resumen": resumen,
    }, indent=2))

    print(f"{'regimen':<8} {'rho real':>9} {'aprobacion':>11} {'default pobl.':>14} "
          f"{'aprobados':>11} {'rechazados':>12} {'soft info apr/rech':>22}")
    for regimen, r in resumen.items():
        print(f"{regimen:<8} {r['rho_verdadero']:>9.2f} {r['tasa_aprobacion']:>11.2%} "
              f"{r['default_poblacion']:>14.2%} {r['default_aprobados']:>11.2%} "
              f"{r['default_rechazados']:>12.2%} "
              f"{r['soft_info_media_aprobados']:>10.3f} /{r['soft_info_media_rechazados']:>9.3f}")

    print("\nLa diferencia entre los dos regimenes esta en las dos ultimas columnas:")
    print("  MAR : rho = 0. La politica no usaba la informacion blanda, asi que")
    print("        aprobados y rechazados no difieren en ella.")
    print("  MNAR: rho != 0. Los aprobados quedan seleccionados por algo que el")
    print("        modelo nuevo no puede observar ni corregir con sus features.")
    print(f"\n{INSTRUMENTO} entra solo a la seleccion (tau={TAU_INSTRUMENTO}), nunca al")
    print("desenlace: es la variable de exclusion que Heckman y el probit bivariado")
    print("necesitan para identificar rho con algo mas que la curvatura de la normal.")
    print(f"-> {RAW_DIR}")


if __name__ == "__main__":
    main()
