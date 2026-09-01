"""Simulacion de estres tipo Basilea III / IFRS9 combinando: PD (XGBoost, Python,
credit_scoring.py) x LGD empirica estresada (GAM calibrado en R sobre datos macro
reales, r/lgd_calibration.R) x EAD, bajo staging IFRS9 de 3 etapas.

Interoperabilidad deliberadamente en la direccion OPUESTA al resto del repo: todo
el resto del pipeline usa reticulate (R llama a Python). Este script usa rpy2 --
Python llama a R directamente para cargar y evaluar el modelo GAM ya entrenado
(output/models/lgd_gam_fit.rds) sin necesidad de reimplementar la prediccion de un
GAM Beta-regression en Python. Interoperabilidad real y bidireccional, no solo
lectura de un CSV que R ya escribio.

Los tres escenarios macro (Base / Adverso / Severamente Adverso) estan anclados a
episodios REALMENTE OBSERVADOS del ciclo economico chileno (Banco Mundial, ver
data/chile_macro_indicators.csv) -- exactamente la logica de diseño de escenarios
de Basilea III/EBA/IFRS9: el escenario "severamente adverso" no es un supuesto
inventado, es el shock real del PIB chileno en 2020 (-6.14%, COVID).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "output" / "models"
TABLES_DIR = ROOT_DIR / "output" / "tables"

# Mezcla de garantia representativa del portafolio (misma que r/lgd_calibration.R
# usa para simular el panel de prestamos en default) -- para agregar la LGD
# predicha por tipo de garantia a una LGD de portafolio unica por escenario.
PORTFOLIO_GARANTIA_MIX = [
    {"tipo_garantia": "Hipotecaria", "peso": 0.30, "colateral_pct": 85.0},
    {"tipo_garantia": "Prendaria", "peso": 0.25, "colateral_pct": 45.0},
    {"tipo_garantia": "Sin Garantia", "peso": 0.45, "colateral_pct": 0.0},
]

PLAZO_PROMEDIO_MESES_FALLBACK = 36  # usado solo si el merge con plazo_meses falla


def _setup_r_home() -> None:
    """Auto-detecta la instalacion de R en esta maquina (no asume que R_HOME ya
    este seteado en el entorno -- una sesion de terminal nueva no lo tiene, mismo
    tipo de gotcha ya resuelto para vcvars64.bat/vswhere.exe en otros repos del
    portafolio)."""
    if os.environ.get("R_HOME"):
        return
    candidates = sorted(Path(r"C:\Program Files\R").glob("R-*"), reverse=True)
    if not candidates:
        raise RuntimeError("No se encontro una instalacion de R en C:\\Program Files\\R")
    r_home = candidates[0]
    os.environ["R_HOME"] = str(r_home)
    bin_x64 = r_home / "bin" / "x64"
    if str(bin_x64) not in os.environ["PATH"]:
        os.environ["PATH"] = f"{bin_x64};{os.environ['PATH']}"


def build_macro_scenarios(macro_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Ancla los 3 escenarios a episodios reales del ciclo chileno en vez de
    inventar numeros -- Base = ultimo anio disponible; Adverso = promedio de un
    tramo real de crecimiento lento (2014-2017, post-boom del cobre); Severamente
    Adverso = el shock COVID real de 2020, el peor anio del panel."""
    base_row = macro_df.sort_values("anio").iloc[-1]
    adverso_df = macro_df[macro_df["anio"].between(2014, 2017)]
    peor_anio = macro_df.loc[macro_df["pib_crecimiento_pct"].idxmin()]

    return {
        "Base": {
            "pib_crecimiento_pct": float(base_row["pib_crecimiento_pct"]),
            "desempleo_pct": float(base_row["desempleo_pct"]),
            "anio_ancla": f"{int(base_row['anio'])} (ultimo real disponible)",
        },
        "Adverso": {
            "pib_crecimiento_pct": float(adverso_df["pib_crecimiento_pct"].mean()),
            "desempleo_pct": float(adverso_df["desempleo_pct"].mean()),
            "anio_ancla": "2014-2017 promedio (desaceleracion real post-boom del cobre)",
        },
        "Severamente Adverso": {
            "pib_crecimiento_pct": float(peor_anio["pib_crecimiento_pct"]),
            "desempleo_pct": float(peor_anio["desempleo_pct"]),
            "anio_ancla": f"{int(peor_anio['anio'])} real (shock COVID)",
        },
    }


def predict_portfolio_lgd_by_scenario(scenarios: dict, gam_model) -> dict[str, float]:
    """Para cada escenario, predice LGD por tipo de garantia via el GAM de R (a
    traves de rpy2) y agrega por la mezcla de portafolio -- una unica LGD de
    portafolio empiricamente calibrada por escenario macro, reemplazando el
    supuesto fijo (45%, multiplicadores ilustrativos) del pipeline original."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    with (ro.default_converter + pandas2ri.converter).context():
        results: dict[str, float] = {}
        for name, scen in scenarios.items():
            rows = [
                {
                    "pib_crecimiento_pct": scen["pib_crecimiento_pct"],
                    "desempleo_pct": scen["desempleo_pct"],
                    "colateral_pct": g["colateral_pct"],
                    "tipo_garantia": g["tipo_garantia"],
                }
                for g in PORTFOLIO_GARANTIA_MIX
            ]
            newdata_r = ro.conversion.get_conversion().py2rpy(pd.DataFrame(rows))
            # tipo_garantia debe ser factor con los mismos niveles usados al
            # entrenar el GAM, o predict.gam falla silenciosamente con NA.
            ro.globalenv["newdata_r"] = newdata_r
            ro.globalenv["gam_model"] = gam_model
            ro.r(
                'newdata_r$tipo_garantia <- factor(newdata_r$tipo_garantia, '
                'levels = levels(gam_model$model$tipo_garantia))'
            )
            newdata_r = ro.globalenv["newdata_r"]
            pred_r = ro.r["predict"](gam_model, newdata=newdata_r, type="response")
            preds = list(pred_r)

            portfolio_lgd = sum(p * g["peso"] for p, g in zip(preds, PORTFOLIO_GARANTIA_MIX))
            results[name] = portfolio_lgd
    return results


def compute_ifrs9_ecl(risk_scores_df: pd.DataFrame, lgd_by_scenario: dict[str, float]) -> pd.DataFrame:
    """Staging IFRS9 de 3 etapas sobre la cartera de test de credit_scoring.py:
    Stage 3 (credit-impaired, default_real==1): ECL = LGD * EAD (PD=100%).
    Stage 2 (SICR, quintil Muy Alto de PD sin default observado aun): ECL de por
    vida, PD acumulada sobre el plazo restante via 1-(1-PD_12m)^(plazo_anios) --
    aproximacion estandar de supervivencia i.i.d. anual, documentada como tal.
    Stage 1 (resto): ECL a 12 meses, PD_12m * LGD * EAD.
    """
    df = risk_scores_df.copy()
    muy_alto_umbral = df["pd_xgb"].quantile(0.80)

    def stage_of(row) -> int:
        if row["default_real"] == 1:
            return 3
        if row["pd_xgb"] >= muy_alto_umbral:
            return 2
        return 1

    df["ifrs9_stage"] = df.apply(stage_of, axis=1)
    df["plazo_anios"] = df["plazo_meses"].fillna(PLAZO_PROMEDIO_MESES_FALLBACK) / 12.0

    rows = []
    for scenario, lgd in lgd_by_scenario.items():
        s = df.copy()
        s["lgd_estresada"] = lgd
        pd_lifetime = 1 - (1 - s["pd_xgb"]) ** s["plazo_anios"]

        ecl = pd.Series(index=s.index, dtype=float)
        stage3 = s["ifrs9_stage"] == 3
        stage2 = s["ifrs9_stage"] == 2
        stage1 = s["ifrs9_stage"] == 1

        ecl[stage3] = s.loc[stage3, "lgd_estresada"] * s.loc[stage3, "monto_solicitado_clp"]
        ecl[stage2] = pd_lifetime[stage2] * s.loc[stage2, "lgd_estresada"] * s.loc[stage2, "monto_solicitado_clp"]
        ecl[stage1] = s.loc[stage1, "pd_xgb"] * s.loc[stage1, "lgd_estresada"] * s.loc[stage1, "monto_solicitado_clp"]
        s["ecl_clp"] = ecl

        resumen = (
            s.groupby("ifrs9_stage")
            .agg(n_clientes=("customer_id", "count"), exposicion_clp=("monto_solicitado_clp", "sum"), ecl_clp=("ecl_clp", "sum"))
            .reset_index()
        )
        resumen["escenario"] = scenario
        resumen["lgd_estresada_portafolio"] = lgd
        rows.append(resumen)

    resultado = pd.concat(rows, ignore_index=True)
    resultado["tasa_cobertura"] = resultado["ecl_clp"] / resultado["exposicion_clp"]
    return resultado[["escenario", "ifrs9_stage", "n_clientes", "exposicion_clp", "lgd_estresada_portafolio", "ecl_clp", "tasa_cobertura"]]


def run_lgd_stress_test() -> pd.DataFrame:
    _setup_r_home()
    import rpy2.robjects as ro

    macro_df = pd.read_csv(DATA_DIR / "chile_macro_indicators.csv")
    risk_scores_df = pd.read_csv(TABLES_DIR / "credit_risk_scores.csv")
    clean_df = pd.read_csv(DATA_DIR / "clean_credit_applications.csv")[["customer_id", "plazo_meses"]]
    risk_scores_df = risk_scores_df.merge(clean_df, on="customer_id", how="left")

    gam_model = ro.r["readRDS"](str(MODELS_DIR / "lgd_gam_fit.rds"))

    scenarios = build_macro_scenarios(macro_df)
    lgd_by_scenario = predict_portfolio_lgd_by_scenario(scenarios, gam_model)

    ecl_df = compute_ifrs9_ecl(risk_scores_df, lgd_by_scenario)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ecl_df.to_csv(TABLES_DIR / "ifrs9_ecl_stress_test.csv", index=False)

    return ecl_df, scenarios, lgd_by_scenario


if __name__ == "__main__":
    ecl_df, scenarios, lgd_by_scenario = run_lgd_stress_test()

    print("=== Escenarios macro (anclados a datos reales del Banco Mundial) ===")
    for name, s in scenarios.items():
        print(f"  {name}: PIB {s['pib_crecimiento_pct']:+.2f}%  Desempleo {s['desempleo_pct']:.2f}%  ({s['anio_ancla']})")

    print("\n=== LGD de portafolio estresada (GAM empirico, via rpy2) ===")
    for name, lgd in lgd_by_scenario.items():
        print(f"  {name}: LGD = {lgd:.1%}")

    print("\n=== ECL IFRS9 por escenario y stage ===")
    print(ecl_df.to_string(index=False))

    total_por_escenario = ecl_df.groupby("escenario")["ecl_clp"].sum()
    print("\n=== ECL total de portafolio por escenario ===")
    for escenario, total in total_por_escenario.items():
        print(f"  {escenario}: ${total:,.0f} CLP")
