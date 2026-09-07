"""Ajuste y validacion de los modelos de hazard en tiempo discreto.

Compara dos especificaciones sobre las mismas filas credito-mes:

- `ph`  : efectos constantes en el tiempo (el analogo discreto de Cox).
- `tvc` : agrega `informal x log(mes)`, o sea deja que ese efecto cambie
          con la antiguedad del credito -- la correccion directa a la
          violacion del supuesto PH que detecta el modelo de Cox.

El contraste se hace de dos formas: test de razon de verosimilitud dentro
de train (la especificacion mas rica, .es. significativamente mejor?) y
metricas fuera de muestra en test (.la mejora sobrevive a datos nuevos?).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.discrete_hazard import DiscreteTimeHazard, likelihood_ratio_test
from src.evaluation import (
    auc_at_horizon, calibration_by_decile, calibration_summary,
    concordance_index, ks_statistic,
)
from src.preprocessing import FEATURES

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

HORIZONTE_12M = 12
HORIZONTE_LIFETIME = 36
MAX_MES = 36


def _load():
    return (
        pd.read_csv(PROC_DIR / "train_person_period.csv"),
        pd.read_csv(PROC_DIR / "test_loans.csv"),
    )


def evaluar(modelo: DiscreteTimeHazard, test: pd.DataFrame) -> dict:
    dur = test["duracion_meses"].to_numpy()
    ev = test["evento_default"].to_numpy()
    curva = modelo.predict_cumulative_default(test)
    pd_12 = curva[:, HORIZONTE_12M - 1]
    pd_life = curva[:, HORIZONTE_LIFETIME - 1]

    cal = calibration_by_decile(dur, ev, pd_12, HORIZONTE_12M)
    return {
        "metricas": {
            **concordance_index(dur, ev, pd_life),
            **auc_at_horizon(dur, ev, pd_12, HORIZONTE_12M),
            "ks_12m": ks_statistic(dur, ev, pd_12, HORIZONTE_12M),
            "calibracion_12m": calibration_summary(cal),
            "pd_12m_media": float(pd_12.mean()),
            "pd_lifetime_media": float(pd_life.mean()),
        },
        "calibracion": cal,
        "curva": curva,
        "predicciones": pd.DataFrame({
            "loan_id": test["loan_id"],
            "duracion_meses": dur,
            "evento_default": ev,
            "pd_12m": pd_12,
            "pd_lifetime_36m": pd_life,
        }),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pp_train, test = _load()

    modelo_ph = DiscreteTimeHazard(FEATURES, max_mes=MAX_MES).fit(pp_train)
    modelo_tvc = DiscreteTimeHazard(
        FEATURES, tvc_features=["informal"], max_mes=MAX_MES
    ).fit(pp_train)

    lr = likelihood_ratio_test(modelo_ph, modelo_tvc)

    resultados = {}
    for nombre, modelo in [("ph", modelo_ph), ("tvc", modelo_tvc)]:
        resultados[nombre] = evaluar(modelo, test)

    # Criterio de seleccion: error de calibracion, no AUC. Lo que se le pide
    # a este modelo es una PD por mes que se pueda provisionar, no un
    # ranking -- y las dos especificaciones ordenan practicamente igual.
    def _err_calib(nombre: str) -> float:
        return resultados[nombre]["metricas"]["calibracion_12m"]["error_absoluto_medio"]

    mejor = min(("ph", "tvc"), key=_err_calib)

    resultados[mejor]["predicciones"].to_csv(
        REPORTS_DIR / "discrete_test_predictions.csv", index=False
    )
    resultados[mejor]["calibracion"].to_csv(
        REPORTS_DIR / "discrete_calibration_12m.csv", index=False
    )
    np.save(REPORTS_DIR / "discrete_pd_curves_test.npy", resultados[mejor]["curva"])

    baseline = pd.DataFrame({
        "mes": np.arange(1, MAX_MES + 1),
        "hazard_mensual_ph": modelo_ph.baseline_monthly_hazard(),
        "hazard_mensual_tvc": modelo_tvc.baseline_monthly_hazard(),
    })
    baseline.to_csv(REPORTS_DIR / "discrete_baseline_hazard.csv", index=False)

    coef_tvc = modelo_tvc.coefficients()
    coef_tvc[~coef_tvc["term"].str.startswith("mes_")].to_csv(
        REPORTS_DIR / "discrete_coefficients.csv", index=False
    )

    reporte = {
        "modelo": "hazard_tiempo_discreto",
        "n_filas_credito_mes_train": int(len(pp_train)),
        "n_test_creditos": int(len(test)),
        "modelo_elegido": mejor,
        "lr_test_ph_vs_tvc": lr,
        "coeficientes_tvc": coef_tvc[~coef_tvc["term"].str.startswith("mes_")].to_dict(
            orient="records"
        ),
        "validacion_test": {k: v["metricas"] for k, v in resultados.items()},
    }
    (REPORTS_DIR / "discrete_results.json").write_text(json.dumps(reporte, indent=2))

    print(f"Filas credito-mes en train: {len(pp_train):,}")
    print("\nLR test  PH vs time-varying (informal x log(mes))")
    print(f"  estadistico {lr['lr_stat']:.2f} con {lr['df']} gl -> p = {lr['p_value']:.3e}")
    print("\nValidacion fuera de muestra")
    for nombre in ("ph", "tvc"):
        m = resultados[nombre]["metricas"]
        print(f"  [{nombre}] C-index {m['c_index']:.4f} | AUC 12m {m['auc']:.4f} | "
              f"KS {m['ks_12m']:.4f} | Brier {m['brier']:.4f} | "
              f"err calib {m['calibracion_12m']['error_absoluto_medio']:.4f}")
    print(f"\nModelo elegido para la estructura temporal de PD: {mejor}")


if __name__ == "__main__":
    main()
