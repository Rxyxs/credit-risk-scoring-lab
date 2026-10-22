"""El experimento central: recuperar lo que nunca se observo directamente
(la correlacion de activos y el ciclo economico), y despues mostrar por
que la distincion PIT/TTC no es un tecnicismo -- es la diferencia entre
capital regulatorio estable y capital que sube justo cuando el banco
menos puede permitirselo.

Cinco piezas:

1. **Recuperar rho** por metodo de momentos, grado por grado, contra la
   correlacion verdadera (que por diseno coincide con la formula de
   Basilea evaluada en la PD de cada grado).
2. **El sesgo de granularidad**: repetir la recuperacion de rho con una
   cartera mucho mas chica (200 deudores por cohorte en vez de 20.000),
   muchas veces, con tres estimadores -- el de momentos con el tamano de
   las cohortes, el de momentos sin el (limite de cartera infinita) y el
   limite ASRF -- para mostrar cuando confundir ruido idiosincratico con
   riesgo sistematico importa.
3. **Recuperar el ciclo economico** Z_t a partir de las tasas de default
   agregadas, sin haberlo visto nunca, y compararlo contra el verdadero.
4. **PIT vs TTC**: la PD condicional al ciclo (la que ve un modelo
   recalibrado cada ano) contra la PD promedio del ciclo completo (la que
   exige el marco IRB), y el capital regulatorio que resulta de cada una.
5. **Verificacion cruzada**: la distribucion cerrada de Vasicek contra una
   simulacion de Monte Carlo independiente de la serie temporal usada para
   estimar, para separar "el estimador funciona" de "la formula analitica
   es correcta".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.correlation_estimation import (
    asrf_limit_correlation, mom_asset_correlation,
)
from src.data_generator import GRADOS, LGD, PD_TTC_TRUE, rho_true, simular_ciclo
from src.vasicek import (
    basel_asset_correlation, basel_capital_requirement, conditional_pd,
    implied_z, vasicek_loss_cdf, vasicek_loss_pdf,
)

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
REPORTS_DIR = BASE / "outputs" / "reports"

# Composicion de la cartera para agregar capital a nivel de banco.
PESO_CARTERA = {"AAA-A": 0.15, "BBB": 0.30, "BB": 0.30, "B": 0.18, "CCC": 0.07}
EAD_TOTAL_CLP = 500_000_000_000.0     # 500.000 millones, orden de magnitud de un banco mediano

N_CHICO = 200                          # cartera "poco granular" para el sesgo de granularidad
N_REPLICAS_GRANULARIDAD = 200          # ciclos de 60 anos por grado y tamano
N_REPLICAS_MC = 4000                   # replicas para verificar la formula cerrada


def estimar_por_grado(cohortes: pd.DataFrame) -> pd.DataFrame:
    """PD_ttc y rho estimados por metodo de momentos, grado por grado."""
    filas = []
    for g in GRADOS:
        sub = cohortes[cohortes["grado"] == g]
        pd_hat = float(sub["tasa_default_observada"].mean())
        var_obs = float(sub["tasa_default_observada"].var(ddof=1))
        rho_hat = mom_asset_correlation(pd_hat, var_obs, n_por_cohorte=sub["n_obligores"].to_numpy())
        filas.append({
            "grado": g,
            "pd_ttc_true": PD_TTC_TRUE[g],
            "pd_ttc_hat": pd_hat,
            "rho_true": rho_true(g),
            "rho_hat_mom": rho_hat,
            "rho_basel_en_pd_hat": float(basel_asset_correlation(pd_hat)),
        })
    return pd.DataFrame(filas)


def sesgo_de_granularidad(seed: int = 123, n_replicas: int = N_REPLICAS_GRANULARIDAD) -> pd.DataFrame:
    """Repite la estimacion con una cartera chica y una grande, muchas veces.

    Tres estimadores sobre las mismas series: momentos con el tamano de las
    cohortes, momentos sin el (limite de cartera infinita, como estaba antes
    este modulo) y el limite ASRF. Una sola serie de 60 anos no alcanza para
    separar sesgo de ruido, asi que se reportan el promedio y el error
    absoluto medio sobre `n_replicas` ciclos. Dado el factor Z_t, los
    defaults de una cohorte son Binomial(N, p(Z_t)): es la misma
    distribucion que simular deudor por deudor, sin el costo.
    """
    rng = np.random.default_rng(seed)
    n_anos = 60                        # mas anos para que la comparacion no sea ruido puro

    filas = []
    for g in GRADOS:
        pd_ttc = PD_TTC_TRUE[g]
        rho = rho_true(g)
        umbral = norm.ppf(pd_ttc)

        for n_cartera, etiqueta in [(N_CHICO, "chica"), (20_000, "grande")]:
            est = {"mom": [], "mom_sin_n": [], "asrf": []}
            for _ in range(n_replicas):
                z = simular_ciclo(n_anos, rng)
                p = norm.cdf((umbral - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))
                tasas = rng.binomial(n_cartera, p) / n_cartera
                pd_hat = max(float(np.mean(tasas)), 1e-4)
                var_obs = float(np.var(tasas, ddof=1))
                if var_obs <= 0:
                    continue
                est["mom"].append(mom_asset_correlation(pd_hat, var_obs, n_por_cohorte=n_cartera))
                est["mom_sin_n"].append(mom_asset_correlation(pd_hat, var_obs))
                est["asrf"].append(asrf_limit_correlation(tasas))

            fila = {"grado": g, "n_por_cohorte": n_cartera, "tamano": etiqueta,
                    "rho_true": rho, "n_replicas": len(est["mom"])}
            for clave, nombre in [("mom", "rho_hat_mom"), ("mom_sin_n", "rho_hat_mom_sin_n"),
                                  ("asrf", "rho_hat_asrf_limite")]:
                valores = np.asarray(est[clave])
                fila[nombre] = float(valores.mean())
                fila[nombre.replace("rho_hat", "mae")] = float(np.abs(valores - rho).mean())
            filas.append(fila)
    return pd.DataFrame(filas)


def recuperar_ciclo(cohortes: pd.DataFrame, estimados: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye Z_t a partir de las tasas de default de cada grado, y
    promedia entre grados (cada uno es una vista ruidosa del mismo factor)."""
    est = estimados.set_index("grado")
    z_por_grado = {}
    for g in GRADOS:
        sub = cohortes[cohortes["grado"] == g].sort_values("anio")
        z_por_grado[g] = implied_z(
            est.loc[g, "pd_ttc_hat"], est.loc[g, "rho_hat_mom"],
            sub["tasa_default_observada"].to_numpy(),
        )

    anios = cohortes["anio"].unique()
    z_matriz = np.column_stack([z_por_grado[g] for g in GRADOS])
    z_recuperado = z_matriz.mean(axis=1)
    z_verdadero = (cohortes[cohortes["grado"] == GRADOS[0]]
                  .sort_values("anio")["z_verdadero"].to_numpy())

    return pd.DataFrame({
        "anio": sorted(anios), "z_verdadero": z_verdadero, "z_recuperado": z_recuperado,
        **{f"z_{g}": z_por_grado[g] for g in GRADOS},
    })


def capital_pit_vs_ttc(cohortes: pd.DataFrame, estimados: pd.DataFrame) -> pd.DataFrame:
    """RWA de la cartera completa, ano a ano, bajo dos filosofias de PD."""
    est = estimados.set_index("grado")
    filas = []
    for t in sorted(cohortes["anio"].unique()):
        rwa_ttc, rwa_pit, ead_total = 0.0, 0.0, 0.0
        for g in GRADOS:
            fila_g = cohortes[(cohortes["grado"] == g) & (cohortes["anio"] == t)].iloc[0]
            ead_g = PESO_CARTERA[g] * EAD_TOTAL_CLP
            pd_ttc_hat = est.loc[g, "pd_ttc_hat"]
            rho_hat = est.loc[g, "rho_hat_mom"]

            k_ttc = basel_capital_requirement(pd_ttc_hat, LGD, rho=rho_hat)
            pd_pit = conditional_pd(pd_ttc_hat, rho_hat, fila_g["z_verdadero"])
            k_pit = basel_capital_requirement(float(np.clip(pd_pit, 1e-6, 0.999)), LGD, rho=rho_hat)

            rwa_ttc += float(k_ttc) * 12.5 * ead_g
            rwa_pit += float(k_pit) * 12.5 * ead_g
            ead_total += ead_g

        filas.append({"anio": t, "rwa_ttc": rwa_ttc, "rwa_pit": rwa_pit,
                      "ead_total": ead_total,
                      # RWA como % de la exposicion (densidad de RWA) --
                      # no confundir con el ratio de capital regulatorio
                      # (capital / RWA, tipicamente 8-15%). Una cartera con
                      # peso fuerte en grados especulativos legitimamente
                      # supera el 100%: el risk weight de un CCC individual
                      # ya puede rondar 400-600%.
                      "densidad_rwa_ttc_pct": 100 * rwa_ttc / ead_total,
                      "densidad_rwa_pit_pct": 100 * rwa_pit / ead_total})
    return pd.DataFrame(filas)


def verificar_formula_cerrada(grado: str = "BB", n_cartera: int = 5000,
                              seed: int = 55) -> pd.DataFrame:
    """Monte Carlo independiente: histograma de la tasa de default de
    `N_REPLICAS_MC` carteras simuladas, contra la densidad cerrada de Vasicek."""
    rng = np.random.default_rng(seed)
    pd_ttc, rho = PD_TTC_TRUE[grado], rho_true(grado)
    umbral = norm.ppf(pd_ttc)

    z = rng.normal(size=N_REPLICAS_MC)
    tasas = np.empty(N_REPLICAS_MC)
    for r in range(N_REPLICAS_MC):
        eps = rng.normal(size=n_cartera)
        x = np.sqrt(rho) * z[r] + np.sqrt(1.0 - rho) * eps
        tasas[r] = np.mean(x < umbral)

    grilla = np.linspace(max(tasas.min(), 1e-4), min(tasas.max(), 0.999), 400)
    return pd.DataFrame({"tasa_default_mc": tasas}), pd.DataFrame({
        "x": grilla,
        "densidad_cerrada": vasicek_loss_pdf(grilla, pd_ttc, rho),
        "cdf_cerrada": vasicek_loss_cdf(grilla, pd_ttc, rho),
    })


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cohortes = pd.read_csv(RAW_DIR / "cohortes.csv")

    estimados = estimar_por_grado(cohortes)
    estimados.to_csv(REPORTS_DIR / "correlacion_estimada.csv", index=False)

    granularidad = sesgo_de_granularidad()
    granularidad.to_csv(REPORTS_DIR / "sesgo_granularidad.csv", index=False)

    ciclo = recuperar_ciclo(cohortes, estimados)
    ciclo.to_csv(REPORTS_DIR / "ciclo_recuperado.csv", index=False)
    corr_ciclo = float(np.corrcoef(ciclo["z_verdadero"], ciclo["z_recuperado"])[0, 1])
    rmse_ciclo = float(np.sqrt(np.mean((ciclo["z_verdadero"] - ciclo["z_recuperado"]) ** 2)))

    capital = capital_pit_vs_ttc(cohortes, estimados)
    capital.to_csv(REPORTS_DIR / "capital_pit_vs_ttc.csv", index=False)

    mc_muestras, mc_teorico = verificar_formula_cerrada()
    mc_muestras.to_csv(REPORTS_DIR / "monte_carlo_muestras.csv", index=False)
    mc_teorico.to_csv(REPORTS_DIR / "monte_carlo_teorico.csv", index=False)

    swing_ttc = float(capital["densidad_rwa_ttc_pct"].max() - capital["densidad_rwa_ttc_pct"].min())
    swing_pit = float(capital["densidad_rwa_pit_pct"].max() - capital["densidad_rwa_pit_pct"].min())

    reporte = {
        "correlacion_por_grado": estimados.to_dict(orient="records"),
        "sesgo_granularidad": granularidad.to_dict(orient="records"),
        "recuperacion_ciclo": {"correlacion": corr_ciclo, "rmse": rmse_ciclo},
        "capital": {
            "swing_ttc_pp": swing_ttc, "swing_pit_pp": swing_pit,
            "densidad_rwa_ttc_promedio_pct": float(capital["densidad_rwa_ttc_pct"].mean()),
            "densidad_rwa_pit_min_pct": float(capital["densidad_rwa_pit_pct"].min()),
            "densidad_rwa_pit_max_pct": float(capital["densidad_rwa_pit_pct"].max()),
        },
    }
    (REPORTS_DIR / "experiment_results.json").write_text(json.dumps(reporte, indent=2))

    print(f"{'grado':<8} {'PD_ttc real':>12} {'PD_ttc hat':>11} {'rho real':>9} "
          f"{'rho MOM':>9} {'rho Basilea(PD hat)':>20}")
    for _, f in estimados.iterrows():
        print(f"{f['grado']:<8} {f['pd_ttc_true']:>12.4f} {f['pd_ttc_hat']:>11.4f} "
              f"{f['rho_true']:>9.4f} {f['rho_hat_mom']:>9.4f} {f['rho_basel_en_pd_hat']:>20.4f}")

    print(f"\nRecuperacion del ciclo economico (sin haberlo observado nunca)")
    print(f"  Correlacion con el Z verdadero: {corr_ciclo:.4f}")
    print(f"  RMSE                          : {rmse_ciclo:.4f}")

    print(f"\nSesgo de granularidad (promedio de {N_REPLICAS_GRANULARIDAD} ciclos; "
          f"cartera chica de {N_CHICO} deudores vs 20.000)")
    print(f"  {'grado':<8} {'tamano':<7} {'rho real':>9} {'MOM con N':>10} {'MOM sin N':>10} {'ASRF':>8}")
    for _, f in granularidad.iterrows():
        print(f"  {f['grado']:<8} {f['tamano']:<7} {f['rho_true']:>9.4f} {f['rho_hat_mom']:>10.4f} "
              f"{f['rho_hat_mom_sin_n']:>10.4f} {f['rho_hat_asrf_limite']:>8.4f}")
    for tam in ("chica", "grande"):
        sub = granularidad[granularidad["tamano"] == tam]
        print(f"  error absoluto medio, cartera {tam}: MOM con N {sub['mae_mom'].mean():.4f} | "
              f"MOM sin N {sub['mae_mom_sin_n'].mean():.4f} | ASRF {sub['mae_asrf_limite'].mean():.4f}")

    print(f"\nDensidad de RWA (RWA / exposicion), PIT vs TTC, {len(GRADOS)} grados")
    print(f"  TTC: estable en {reporte['capital']['densidad_rwa_ttc_promedio_pct']:.2f}% "
          f"todos los anos, por construccion (PD fija -> capital fijo)")
    print(f"  PIT: entre {reporte['capital']['densidad_rwa_pit_min_pct']:.2f}% y "
          f"{reporte['capital']['densidad_rwa_pit_max_pct']:.2f}% "
          f"(oscilacion de {swing_pit:.1f} puntos porcentuales entre el mejor y el peor ano)")
    print(f"\nReportes en {REPORTS_DIR}")


if __name__ == "__main__":
    main()
