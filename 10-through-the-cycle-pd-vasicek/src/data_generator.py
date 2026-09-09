"""Simulador de una cartera bajo el modelo de un factor: 30 anos de
cohortes, 5 grados de riesgo, un solo factor sistematico compartido por
toda la cartera.

Cada grado de riesgo `g` tiene una PD through-the-cycle verdadera
`PD_ttc[g]` y una correlacion de activos verdadera `rho[g]` **igual a la
formula regulatoria de Basilea** evaluada en esa PD. Eso no es casualidad:
es lo que permite despues verificar dos cosas a la vez con el mismo
numero -- que el estimador de correlacion recupera la verdad del
simulador, y que esa verdad coincide con lo que asume el regulador.

El ciclo economico `Z_t` es un AR(1) persistente con dos recesiones
marcadas a proposito (anos 9-10 y 22-23), para que el patron de
"correlacion de activos = todos caen juntos en las malas" sea visible en
los datos y no solo en la teoria. Con `N` deudores por grado y por ano
(grande), la tasa de default observada de cada cohorte converge al limite
ASRF: el ruido idiosincratico se diversifica y lo que queda es
practicamente una funcion determinista de `Z_t`.

No usa datos reales de ninguna institucion ni del Boletin Comercial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.vasicek import basel_asset_correlation, conditional_pd

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"

RANDOM_STATE_DEFAULT = 42
N_ANOS = 30
N_POR_GRADO_ANO = 20_000
ANOS_RECESION = {9: -2.4, 10: -1.6, 22: -2.0, 23: -1.1}
PHI_AR = 0.65          # persistencia del ciclo economico

GRADOS = ["AAA-A", "BBB", "BB", "B", "CCC"]
PD_TTC_TRUE = {
    "AAA-A": 0.0015,
    "BBB": 0.006,
    "BB": 0.020,
    "B": 0.055,
    "CCC": 0.140,
}
LGD = 0.45
EAD_MEDIO = 1_000_000.0


def rho_true(grado: str) -> float:
    """La correlacion de activos que genero los datos: la formula de Basilea
    evaluada en la PD verdadera de ese grado -- ver el docstring del modulo."""
    return float(basel_asset_correlation(PD_TTC_TRUE[grado]))


def simular_ciclo(n_anos: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) gaussiano persistente, con recesiones marcadas eligiendo la
    innovacion de esos anos en vez de sumar un choque aparte.

    Un AR(1) impulsado por innovaciones N(0,1), `Z_t = phi*Z_{t-1} +
    sqrt(1-phi^2)*eta_t`, es exactamente gaussiano de varianza unitaria
    porque es una combinacion lineal de normales -- que es justo lo que el
    modelo de un factor supone sobre Z. Sumarle un choque aparte encima
    (una version anterior de este modulo lo hacia) rompe esa gaussianidad
    y sesga el estimador de correlacion, porque `mom_asset_correlation`
    invierte una formula que asume Z normal, no solo de varianza 1. Elegir
    la innovacion de un ano especifico para que sea una mala realizacion
    (en vez de agregar un termino extra) mantiene el proceso dentro del
    modelo: sigue siendo "una innovacion gaussiana", nada mas que una
    elegida a mano para ser una mala.
    """
    eta = rng.normal(size=n_anos)
    for t, valor in ANOS_RECESION.items():
        if t < n_anos:               # los anos de recesion son un calendario
            eta[t] = valor            # fijo; una corrida mas corta simplemente no los alcanza
    z = np.zeros(n_anos)
    for t in range(1, n_anos):
        z[t] = PHI_AR * z[t - 1] + np.sqrt(1.0 - PHI_AR**2) * eta[t]
    return z


def generar_cartera(n_anos: int = N_ANOS, n_por_grado_ano: int = N_POR_GRADO_ANO,
                    seed: int = RANDOM_STATE_DEFAULT) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    z = simular_ciclo(n_anos, rng)

    filas_cohorte = []
    for t in range(n_anos):
        for g in GRADOS:
            pd_ttc = PD_TTC_TRUE[g]
            rho = rho_true(g)
            umbral = norm.ppf(pd_ttc)

            eps = rng.normal(size=n_por_grado_ano)
            x = np.sqrt(rho) * z[t] + np.sqrt(1.0 - rho) * eps
            defaults = int(np.sum(x < umbral))

            filas_cohorte.append({
                "anio": t, "grado": g, "n_obligores": n_por_grado_ano,
                "n_defaults": defaults,
                "tasa_default_observada": defaults / n_por_grado_ano,
                "z_verdadero": float(z[t]),
                "pd_ttc_verdadera": pd_ttc,
                "rho_verdadero": rho,
                "pd_condicional_teorica": float(conditional_pd(pd_ttc, rho, z[t])),
            })

    cohortes = pd.DataFrame(filas_cohorte)

    resumen_grado = cohortes.groupby("grado").agg(
        pd_ttc_verdadera=("pd_ttc_verdadera", "first"),
        rho_verdadero=("rho_verdadero", "first"),
        tasa_default_media=("tasa_default_observada", "mean"),
        tasa_default_sd=("tasa_default_observada", "std"),
    ).reindex(GRADOS)

    return {"cohortes": cohortes, "ciclo": pd.DataFrame({"anio": np.arange(n_anos), "z": z}),
            "resumen_grado": resumen_grado}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    datos = generar_cartera()
    datos["cohortes"].to_csv(RAW_DIR / "cohortes.csv", index=False)
    datos["ciclo"].to_csv(RAW_DIR / "ciclo.csv", index=False)

    (RAW_DIR / "ground_truth.json").write_text(json.dumps({
        "grados": GRADOS,
        "pd_ttc_true": PD_TTC_TRUE,
        "rho_true": {g: rho_true(g) for g in GRADOS},
        "lgd": LGD,
        "ead_medio": EAD_MEDIO,
        "n_anos": N_ANOS,
        "n_por_grado_ano": N_POR_GRADO_ANO,
        "anos_recesion": ANOS_RECESION,
        "phi_ar": PHI_AR,
    }, indent=2))

    print(f"Cartera: {N_ANOS} anos, {len(GRADOS)} grados, "
          f"{N_POR_GRADO_ANO:,} deudores por grado-ano")
    print(f"\n{'grado':<8} {'PD_ttc':>8} {'rho verdadero':>14} {'default medio':>14} "
          f"{'sd default':>11}")
    for g in GRADOS:
        r = datos["resumen_grado"].loc[g]
        print(f"{g:<8} {r['pd_ttc_verdadera']:>8.4f} {r['rho_verdadero']:>14.4f} "
              f"{r['tasa_default_media']:>14.4f} {r['tasa_default_sd']:>11.4f}")

    print(f"\nAnos de recesion marcada: {sorted(ANOS_RECESION)} "
          f"(deberian verse como picos de default en TODOS los grados a la vez,")
    print("que es la firma del factor sistematico compartido)")
    print(f"-> {RAW_DIR}")


if __name__ == "__main__":
    main()
