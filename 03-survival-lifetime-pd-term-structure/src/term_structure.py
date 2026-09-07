"""De la curva de hazard a una decision de negocio: estructura temporal de
PD, bandas de riesgo y provision esperada bajo IFRS 9.

Un scorecard clasico entrega un numero por cliente: PD a 12 meses. Con un
modelo de supervivencia se tiene la curva completa, y ahi aparecen dos
cosas que el numero unico esconde:

1. **Cuando** llega el riesgo (el peak de seasoning), que es lo que decide
   politicas de seguimiento temprano.
2. **Cuanta** perdida vive despues del mes 12 -- exactamente la diferencia
   entre provisionar a 12 meses (Stage 1) y provisionar a vida completa
   (Stage 2), que es el corazon del modelo de deterioro de IFRS 9.

Supuestos declarados: EAD = monto originado (no se modela amortizacion),
LGD plana de 45%, y ECL sin descontar a valor presente. Son supuestos de
laboratorio: el aporte del modulo es la PD por horizonte, no la severidad.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import km_cum_default_at

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

LGD = 0.45
BANDAS = ["A", "B", "C", "D", "E"]
UMBRAL_SICR = 2.5     # Stage 2 si PD 12m supera 2.5x la PD media de la cartera
HITOS = [6, 12, 24, 36]


def cargar() -> tuple[pd.DataFrame, np.ndarray]:
    test = pd.read_csv(PROC_DIR / "test_loans.csv")
    curvas = np.load(REPORTS_DIR / "discrete_pd_curves_test.npy")
    if len(curvas) != len(test):
        raise ValueError("curvas de PD y cartera de test no calzan")
    return test, curvas


def asignar_bandas(pd_12m: np.ndarray, n_bandas: int = 5) -> np.ndarray:
    """Bandas de riesgo por quintil de PD a 12 meses (A = mejor)."""
    q = pd.qcut(pd_12m, n_bandas, labels=BANDAS[:n_bandas])
    return np.asarray(q)


def tabla_por_banda(test: pd.DataFrame, curvas: np.ndarray) -> pd.DataFrame:
    df = test.copy()
    df["pd_12m"] = curvas[:, 11]
    df["pd_lifetime"] = curvas[:, -1]
    df["banda"] = asignar_bandas(df["pd_12m"].to_numpy())

    filas = []
    for banda, g in df.groupby("banda", observed=True):
        idx = g.index.to_numpy()
        fila = {"banda": banda, "n": len(g)}
        for h in HITOS:
            fila[f"pd_pred_{h}m"] = float(curvas[idx, h - 1].mean())
        fila["pd_obs_km_12m"] = km_cum_default_at(g["duracion_meses"], g["evento_default"], 12)
        fila["pd_obs_km_36m"] = km_cum_default_at(g["duracion_meses"], g["evento_default"], 36)
        fila["ead_total_clp"] = float(g["monto_credito"].sum())
        filas.append(fila)

    tabla = pd.DataFrame(filas).sort_values("banda").reset_index(drop=True)
    tabla["pct_riesgo_despues_de_12m"] = (
        100.0 * (tabla["pd_pred_36m"] - tabla["pd_pred_12m"]) / tabla["pd_pred_36m"]
    )
    return tabla


def curva_agregada(curvas: np.ndarray) -> pd.DataFrame:
    """PD acumulada y marginal promedio de la cartera, mes a mes."""
    cum = curvas.mean(axis=0)
    marginal = np.diff(np.concatenate([[0.0], cum]))
    return pd.DataFrame({
        "mes": np.arange(1, curvas.shape[1] + 1),
        "pd_acumulada": cum,
        "pd_marginal": marginal,
    })


def provision_ifrs9(test: pd.DataFrame, curvas: np.ndarray) -> dict:
    """Provision esperada con staging 12m/lifetime vs todo a 12 meses."""
    pd_12 = curvas[:, 11]
    pd_life = curvas[:, -1]
    ead = test["monto_credito"].to_numpy(dtype=float)

    umbral = UMBRAL_SICR * float(pd_12.mean())
    stage2 = pd_12 > umbral

    ecl_12 = ead * LGD * pd_12
    ecl_life = ead * LGD * pd_life
    ecl_ifrs9 = np.where(stage2, ecl_life, ecl_12)

    total_12 = float(ecl_12.sum())
    total_ifrs9 = float(ecl_ifrs9.sum())
    return {
        "lgd_supuesta": LGD,
        "umbral_sicr_pd_12m": float(umbral),
        "n_stage_1": int((~stage2).sum()),
        "n_stage_2": int(stage2.sum()),
        "pct_cartera_stage_2": float(stage2.mean()),
        "ead_total_clp": float(ead.sum()),
        "ecl_solo_12m_clp": total_12,
        "ecl_con_staging_ifrs9_clp": total_ifrs9,
        "uplift_pct": 100.0 * (total_ifrs9 / total_12 - 1.0),
        "ecl_sobre_ead_solo_12m": total_12 / float(ead.sum()),
        "ecl_sobre_ead_ifrs9": total_ifrs9 / float(ead.sum()),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test, curvas = cargar()

    tabla = tabla_por_banda(test, curvas)
    tabla.to_csv(REPORTS_DIR / "pd_term_structure_por_banda.csv", index=False)

    agregada = curva_agregada(curvas)
    agregada.to_csv(REPORTS_DIR / "pd_curva_agregada.csv", index=False)

    provision = provision_ifrs9(test, curvas)

    mes_peak = int(agregada.loc[agregada["pd_marginal"].idxmax(), "mes"])

    # El hazard *condicional* (base del modelo) y la PD *marginal* de la
    # cartera no piquean en el mismo mes: la marginal ya viene multiplicada
    # por la supervivencia, y como los creditos mas riesgosos caen primero,
    # la mezcla adelanta el peak. Se reportan los dos, suavizando el
    # condicional a 3 meses porque el baseline por dummies es ruidoso.
    base = pd.read_csv(REPORTS_DIR / "discrete_baseline_hazard.csv")
    col = "hazard_mensual_tvc" if "hazard_mensual_tvc" in base else "hazard_mensual_ph"
    suave = base[col].rolling(3, center=True, min_periods=1).mean()
    mes_peak_cond = int(base.loc[suave.idxmax(), "mes"])
    pd12 = float(agregada.loc[agregada["mes"] == 12, "pd_acumulada"].iloc[0])
    pd36 = float(agregada.loc[agregada["mes"] == 36, "pd_acumulada"].iloc[0])

    reporte = {
        "cartera_test": int(len(test)),
        "mes_peak_hazard_marginal": mes_peak,
        "mes_peak_hazard_condicional_suavizado": mes_peak_cond,
        "pd_cartera_12m": pd12,
        "pd_cartera_36m": pd36,
        "ratio_lifetime_sobre_12m": pd36 / pd12,
        "pct_riesgo_despues_de_12m": 100.0 * (pd36 - pd12) / pd36,
        "bandas": tabla.to_dict(orient="records"),
        "provision_ifrs9": provision,
    }
    (REPORTS_DIR / "term_structure_results.json").write_text(json.dumps(reporte, indent=2))

    print("Estructura temporal de PD por banda de riesgo")
    print(tabla.round(4).to_string(index=False))
    print(f"\nPeak del hazard condicional (suavizado 3m): mes {mes_peak_cond}")
    print(f"Peak de la PD marginal de la cartera      : mes {mes_peak}")
    print(f"PD cartera 12m -> 36m   : {pd12:.2%} -> {pd36:.2%} "
          f"({pd36 / pd12:.2f}x, {100 * (pd36 - pd12) / pd36:.1f}% del riesgo llega despues del mes 12)")
    print("\nProvision esperada (LGD 45%, EAD = monto originado, sin descuento)")
    print(f"  Stage 2 (SICR proxy)   : {provision['n_stage_2']:,} creditos "
          f"({provision['pct_cartera_stage_2']:.1%} de la cartera)")
    print(f"  ECL todo a 12 meses    : CLP {provision['ecl_solo_12m_clp']:,.0f}")
    print(f"  ECL con staging IFRS 9 : CLP {provision['ecl_con_staging_ifrs9_clp']:,.0f}")
    print(f"  Diferencia             : +{provision['uplift_pct']:.1f}%")


if __name__ == "__main__":
    main()
