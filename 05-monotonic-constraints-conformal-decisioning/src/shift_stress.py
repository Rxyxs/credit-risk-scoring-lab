"""Que pasa con la garantia conforme cuando la poblacion cambia.

La cobertura conforme se sostiene sobre un supuesto: intercambiabilidad
entre el conjunto de calibracion y los casos que llegan despues. Una
cartera de credito real rompe ese supuesto todo el tiempo -- cambia el mix
de canales, se deteriora el empleo, entra un producto nuevo.

Este modulo aplica el mismo predictor conforme (calibrado en la poblacion
normal) a una cartera bajo deterioro macro: mas informalidad, menos renta,
lineas mas copadas, mas consultas. Ahi se ve cuanto se degrada la
cobertura y en cual clase, que es la unica forma honesta de reportar una
garantia condicional a un supuesto.

Tambien se compara contra recalibrar con datos del escenario nuevo, para
mostrar cual es el arreglo cuando el supuesto se rompe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.conformal import MondrianConformalClassifier
from src.decision_policy import decisiones_conformes, evaluar_politica
from src.models import entrenar, metricas
from src.preprocessing import etiqueta, matriz

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

ALPHA = 0.10
FRAC_RECALIBRACION = 0.35     # cuanto del escenario nuevo se usa para recalibrar


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PROC_DIR / "train.csv")
    calib = pd.read_csv(PROC_DIR / "calib.csv")
    test = pd.read_csv(PROC_DIR / "test.csv")
    shifted = pd.read_csv(PROC_DIR / "test_shifted.csv")

    modelo = entrenar(train)["gbm_monotono"]
    cp = MondrianConformalClassifier(modelo).calibrate(matriz(calib), etiqueta(calib))

    X_test, y_test = matriz(test), etiqueta(test)
    X_shift, y_shift = matriz(shifted), etiqueta(shifted)
    monto_shift = shifted["monto_credito"].to_numpy(float)

    cob_normal = cp.evaluar_cobertura(X_test, y_test, ALPHA)
    cob_shift = cp.evaluar_cobertura(X_shift, y_shift, ALPHA)

    # Recalibracion con una parte del escenario nuevo.
    n_recal = int(FRAC_RECALIBRACION * len(shifted))
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(shifted))
    idx_recal, idx_eval = idx[:n_recal], idx[n_recal:]
    cp_recal = MondrianConformalClassifier(modelo).calibrate(
        X_shift[idx_recal], y_shift[idx_recal]
    )
    cob_recal = cp_recal.evaluar_cobertura(X_shift[idx_eval], y_shift[idx_eval], ALPHA)

    # Impacto operativo del shift con la calibracion vieja.
    dec_normal = decisiones_conformes(cp.prediction_sets(X_test, ALPHA))
    dec_shift = decisiones_conformes(cp.prediction_sets(X_shift, ALPHA))
    dec_recal = decisiones_conformes(cp_recal.prediction_sets(X_shift[idx_eval], ALPHA))

    politicas = pd.DataFrame([
        evaluar_politica(dec_normal, y_test, test["monto_credito"].to_numpy(float),
                         "poblacion_normal"),
        evaluar_politica(dec_shift, y_shift, monto_shift, "deterioro_calibracion_vieja"),
        evaluar_politica(dec_recal, y_shift[idx_eval], monto_shift[idx_eval],
                         "deterioro_recalibrado"),
    ])
    politicas.to_csv(REPORTS_DIR / "shift_politicas.csv", index=False)

    reporte = {
        "alpha": ALPHA,
        "tasa_default_normal": float(y_test.mean()),
        "tasa_default_deterioro": float(y_shift.mean()),
        "auc_normal": metricas(y_test, modelo.predict_proba(X_test)[:, 1])["auc"],
        "auc_deterioro": metricas(y_shift, modelo.predict_proba(X_shift)[:, 1])["auc"],
        "cobertura_normal": cob_normal,
        "cobertura_deterioro_calibracion_vieja": cob_shift,
        "cobertura_deterioro_recalibrado": cob_recal,
        "n_recalibracion": int(n_recal),
        "politicas": politicas.to_dict(orient="records"),
    }
    (REPORTS_DIR / "shift_results.json").write_text(json.dumps(reporte, indent=2))

    print(f"Tasa de default: {y_test.mean():.2%} (normal) -> {y_shift.mean():.2%} (deterioro)")
    print(f"AUC del modelo : {reporte['auc_normal']:.4f} -> {reporte['auc_deterioro']:.4f}")
    print(f"\nCobertura conforme al alpha {ALPHA} (objetivo {1 - ALPHA:.2f})")
    print(f"  {'escenario':<34} {'global':>8} {'clase 0':>9} {'clase 1':>9} {'% revision':>11}")
    for nombre, c in [
        ("poblacion normal", cob_normal),
        ("deterioro, calibracion vieja", cob_shift),
        ("deterioro, recalibrado", cob_recal),
    ]:
        print(f"  {nombre:<34} {c['cobertura_global']:>8.4f} {c['cobertura_clase_0']:>9.4f} "
              f"{c['cobertura_clase_1']:>9.4f} {1 - c['pct_singleton']:>10.1%}")

    print("\nImpacto operativo")
    cols = ["politica", "pct_aprobado", "pct_revision_manual",
            "tasa_error_decisiones_automaticas", "tasa_mala_entre_aprobados"]
    print(politicas[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()
