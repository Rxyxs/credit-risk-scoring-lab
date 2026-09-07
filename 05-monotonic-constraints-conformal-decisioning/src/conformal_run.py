"""Calibracion conforme y politica de decision en tres vias.

Etapas:

1. Se reentrena el modelo elegido (GBM monotono) sobre train.
2. Se calibra el predictor conforme sobre el conjunto de calibracion, que
   el modelo no vio.
3. Se mide cobertura empirica en test para varios alpha, en las dos
   variantes (Mondrian por clase y marginal), que es donde se ve por que
   con clases desbalanceadas la marginal no basta.
4. Se convierte todo en una decision operativa y se compara contra la
   practica habitual (banda de score) al mismo volumen de revision manual.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.conformal import MondrianConformalClassifier, cobertura_por_alpha
from src.decision_policy import barrido_alpha, comparar_a_igual_volumen
from src.models import entrenar, metricas
from src.preprocessing import etiqueta, matriz

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

MODELO_ELEGIDO = "gbm_monotono"
ALPHA = 0.10
ALPHAS = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30])


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PROC_DIR / "train.csv")
    calib = pd.read_csv(PROC_DIR / "calib.csv")
    test = pd.read_csv(PROC_DIR / "test.csv")

    modelos = entrenar(train)
    modelo = modelos[MODELO_ELEGIDO]

    X_cal, y_cal = matriz(calib), etiqueta(calib)
    X_test, y_test = matriz(test), etiqueta(test)
    monto = test["monto_credito"].to_numpy(float)
    pd_pred = modelo.predict_proba(X_test)[:, 1]

    cp = MondrianConformalClassifier(modelo, mondrian=True).calibrate(X_cal, y_cal)
    cp_marginal = MondrianConformalClassifier(modelo, mondrian=False).calibrate(X_cal, y_cal)

    cobertura = pd.DataFrame(cobertura_por_alpha(cp, X_test, y_test, ALPHAS))
    cobertura["variante"] = "mondrian"
    cobertura_marg = pd.DataFrame(cobertura_por_alpha(cp_marginal, X_test, y_test, ALPHAS))
    cobertura_marg["variante"] = "marginal"
    cobertura_total = pd.concat([cobertura, cobertura_marg], ignore_index=True)
    cobertura_total.to_csv(REPORTS_DIR / "cobertura_conforme.csv", index=False)

    politicas = comparar_a_igual_volumen(cp, X_test, y_test, monto, pd_pred, alpha=ALPHA)
    politicas.to_csv(REPORTS_DIR / "comparacion_politicas.csv", index=False)

    barrido = barrido_alpha(cp, X_test, y_test, monto, ALPHAS)
    barrido.to_csv(REPORTS_DIR / "barrido_alpha.csv", index=False)

    # Etiquetas conformes por solicitante, para inspeccion y graficos.
    conjuntos = cp.prediction_sets(X_test, ALPHA)
    p_vals = cp.p_values(X_test)
    pd.DataFrame({
        "applicant_id": test["applicant_id"],
        "default_12m": y_test,
        "pd_pred": pd_pred,
        "p_value_bueno": p_vals[:, 0],
        "p_value_malo": p_vals[:, 1],
        "tipo_conjunto": cp.clasificar_conjuntos(conjuntos),
        "monto_credito": monto,
    }).to_csv(REPORTS_DIR / "conformal_test_sets.csv", index=False)

    reporte = {
        "modelo": MODELO_ELEGIDO,
        "metricas_test": metricas(y_test, pd_pred),
        "n_calibracion": int(len(calib)),
        "alpha_operativo": ALPHA,
        "cobertura": cobertura_total.to_dict(orient="records"),
        "politicas_al_alpha_operativo": politicas.to_dict(orient="records"),
        "barrido_alpha": barrido.to_dict(orient="records"),
    }
    (REPORTS_DIR / "conformal_results.json").write_text(json.dumps(reporte, indent=2))

    print(f"Modelo: {MODELO_ELEGIDO} | calibracion sobre {len(calib):,} casos no vistos")
    print("\nCobertura empirica en test (objetivo = 1 - alpha)")
    print(f"  {'variante':<10} {'alpha':>6} {'objetivo':>9} {'global':>8} "
          f"{'clase 0':>9} {'clase 1':>9} {'% singleton':>12}")
    for _, f in cobertura_total.iterrows():
        print(f"  {f['variante']:<10} {f['alpha']:>6.2f} {f['cobertura_objetivo']:>9.2f} "
              f"{f['cobertura_global']:>8.4f} {f['cobertura_clase_0']:>9.4f} "
              f"{f['cobertura_clase_1']:>9.4f} {f['pct_singleton']:>11.1%}")

    print(f"\nDecision en tres vias al alpha operativo ({ALPHA}), "
          f"contra banda de score al mismo volumen de revision")
    cols = ["politica", "pct_aprobado", "pct_revision_manual",
            "tasa_error_decisiones_automaticas", "tasa_mala_entre_aprobados", "utilidad_clp"]
    print(politicas[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()
