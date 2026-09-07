"""Ajuste y validacion del modelo de Cox sobre la cartera.

Corre las dos aproximaciones de empates (Efron y Breslow) sobre exactamente
los mismos datos, las contrasta contra los coeficientes verdaderos del
simulador, y aplica el test del supuesto de hazards proporcionales.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cox_ph import CoxPH
from src.evaluation import (
    auc_at_horizon, calibration_by_decile, calibration_summary,
    concordance_index, ks_statistic,
)
from src.preprocessing import FEATURES, design_matrix

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"
RAW_DIR = BASE / "data" / "raw"

HORIZONTE_12M = 12
HORIZONTE_LIFETIME = 36


def _load():
    train = pd.read_csv(PROC_DIR / "train_loans.csv")
    test = pd.read_csv(PROC_DIR / "test_loans.csv")
    gt = json.loads((RAW_DIR / "ground_truth.json").read_text())
    return train, test, gt


def fit_both_ties(train: pd.DataFrame) -> dict[str, CoxPH]:
    X = design_matrix(train)
    dur = train["duracion_meses"].to_numpy()
    ev = train["evento_default"].to_numpy()
    modelos = {}
    for ties in ("efron", "breslow"):
        m = CoxPH(ties=ties)
        m.fit(X, dur, ev, feature_names=FEATURES)
        modelos[ties] = m
    return modelos


def compare_with_truth(modelos: dict[str, CoxPH], gt: dict) -> pd.DataFrame:
    beta_true = gt["beta_true"]
    filas = []
    for feat_i, feat in enumerate(FEATURES):
        fila = {"feature": feat, "beta_true": beta_true.get(feat, np.nan)}
        for ties, m in modelos.items():
            r = m.result_
            fila[f"coef_{ties}"] = float(r.coef[feat_i])
            fila[f"se_{ties}"] = float(r.se[feat_i])
        filas.append(fila)
    df = pd.DataFrame(filas)
    for ties in modelos:
        df[f"error_{ties}"] = df[f"coef_{ties}"] - df["beta_true"]
    conocidos = df["beta_true"].notna()
    df.loc[conocidos, "atenuacion_breslow_vs_efron_pct"] = (
        100.0 * (1.0 - df.loc[conocidos, "coef_breslow"].abs() / df.loc[conocidos, "coef_efron"].abs())
    )
    return df


def evaluar(modelo: CoxPH, test: pd.DataFrame) -> dict:
    X = design_matrix(test)
    dur = test["duracion_meses"].to_numpy()
    ev = test["evento_default"].to_numpy()

    riesgo = modelo.predict_log_partial_hazard(X)
    pd_12 = modelo.predict_cumulative_default(X, np.array([HORIZONTE_12M])).ravel()
    pd_life = modelo.predict_cumulative_default(X, np.array([HORIZONTE_LIFETIME])).ravel()

    cal = calibration_by_decile(dur, ev, pd_12, HORIZONTE_12M)
    return {
        "metricas": {
            **concordance_index(dur, ev, riesgo),
            **auc_at_horizon(dur, ev, pd_12, HORIZONTE_12M),
            "ks_12m": ks_statistic(dur, ev, pd_12, HORIZONTE_12M),
            "calibracion_12m": calibration_summary(cal),
            "pd_12m_media": float(pd_12.mean()),
            "pd_lifetime_media": float(pd_life.mean()),
        },
        "calibracion": cal,
        "predicciones": pd.DataFrame({
            "loan_id": test["loan_id"],
            "duracion_meses": dur,
            "evento_default": ev,
            "riesgo_lineal": riesgo,
            "pd_12m": pd_12,
            "pd_lifetime_36m": pd_life,
        }),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train, test, gt = _load()

    modelos = fit_both_ties(train)
    comparacion = compare_with_truth(modelos, gt)
    comparacion.to_csv(REPORTS_DIR / "cox_coefficients_vs_truth.csv", index=False)

    modelo = modelos["efron"]
    ph = pd.DataFrame(modelo.ph_test(FEATURES))
    ph.to_csv(REPORTS_DIR / "cox_ph_test.csv", index=False)

    eval_efron = evaluar(modelo, test)
    eval_efron["predicciones"].to_csv(REPORTS_DIR / "cox_test_predictions.csv", index=False)
    eval_efron["calibracion"].to_csv(REPORTS_DIR / "cox_calibration_12m.csv", index=False)

    baseline = pd.DataFrame({
        "mes": modelo.baseline_times_,
        "hazard_base": modelo.baseline_hazard_,
        "hazard_base_acumulado": modelo.baseline_cumhazard_,
    })
    baseline.to_csv(REPORTS_DIR / "cox_baseline_hazard.csv", index=False)

    conocidos = comparacion["beta_true"].notna()
    reporte = {
        "modelo": "cox_ph_desde_cero",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "features": FEATURES,
        "ties": {
            ties: {
                "loglik": m.result_.loglik,
                "n_iter": m.result_.n_iter,
                "converged": m.result_.converged,
                "coeficientes": m.result_.summary(),
            }
            for ties, m in modelos.items()
        },
        "recuperacion_de_verdad": {
            "error_abs_medio_efron": float(comparacion.loc[conocidos, "error_efron"].abs().mean()),
            "error_abs_medio_breslow": float(comparacion.loc[conocidos, "error_breslow"].abs().mean()),
            "atenuacion_media_breslow_pct": float(
                comparacion.loc[conocidos, "atenuacion_breslow_vs_efron_pct"].mean()
            ),
        },
        "test_ph": ph.to_dict(orient="records"),
        "validacion_test": eval_efron["metricas"],
    }
    (REPORTS_DIR / "cox_results.json").write_text(json.dumps(reporte, indent=2))

    print("Coeficientes vs verdad de terreno")
    print(comparacion.round(4).to_string(index=False))
    print("\nTest del supuesto PH (Schoenfeld escalado vs rango del tiempo)")
    print(ph.round(4).to_string(index=False))
    print("\nValidacion en test")
    m = eval_efron["metricas"]
    print(f"  C-index            : {m['c_index']:.4f} ({m['pares_comparables']:,} pares comparables)")
    print(f"  AUC 12m            : {m['auc']:.4f} (Gini {m['gini']:.4f}, KS {m['ks_12m']:.4f})")
    print(f"  Brier 12m          : {m['brier']:.4f}")
    print(f"  Error calib. medio : {m['calibracion_12m']['error_absoluto_medio']:.4f}")
    print(f"  PD media 12m/36m   : {m['pd_12m_media']:.2%} / {m['pd_lifetime_media']:.2%}")


if __name__ == "__main__":
    main()
