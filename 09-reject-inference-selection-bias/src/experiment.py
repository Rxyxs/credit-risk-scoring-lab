"""El experimento que un banco no puede hacer: evaluar reject inference
contra la verdad.

Cada metodo se juzga con cuatro medidas, y las cuatro necesitan la etiqueta
de los rechazados -- que solo existe porque los datos son simulados:

1. **AUC sobre la poblacion completa.** Lo que de verdad importa: el modelo
   nuevo va a decidir sobre solicitantes que la politica anterior habria
   rechazado. Medir solo sobre aprobados es medir en la region comoda.
2. **AUC sobre los rechazados.** La parte dificil, aislada.
3. **Error de los coeficientes** contra los betas verdaderos. Distingue un
   modelo que ordena bien de uno que ademas esta bien especificado.
4. **Tasa de default de la poblacion estimada vs real.** El numero con el
   que se aprovisiona y se pone precio; un modelo puede ordenar
   perfectamente y aun asi equivocarse por entero en el nivel.

Se corre en los dos regimenes de seleccion (MAR / MNAR) y, para los
metodos de seleccion conjunta, con y sin la variable de exclusion -- que es
la comparacion que muestra si el problema tiene arreglo o solo lo parece.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_generator import (
    BETA_TRUE, FEATURES, FEATURES_SELECCION, INTERCEPTO_TRUE, rho_verdadero,
)
from src.reject_inference import METODOS, parcelling
from src.selection_models import predict_probit

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
REPORTS_DIR = BASE / "outputs" / "reports"

REGIMENES = ("mar", "mnar")


def coeficientes_verdaderos() -> np.ndarray:
    return np.array([INTERCEPTO_TRUE, *[BETA_TRUE[f] for f in FEATURES]])


def evaluar(coef: np.ndarray, df: pd.DataFrame) -> dict:
    """Metricas de un conjunto de coeficientes contra la verdad de terreno."""
    X = df[FEATURES].to_numpy(float)
    y_real = df["default_12m_real"].to_numpy(int)
    rechazado = df["aprobado"].to_numpy(int) == 0

    pd_pred = predict_probit(coef, X)
    beta_true = coeficientes_verdaderos()

    return {
        "auc_poblacion": float(roc_auc_score(y_real, pd_pred)),
        "auc_rechazados": float(roc_auc_score(y_real[rechazado], pd_pred[rechazado])),
        "auc_aprobados": float(roc_auc_score(y_real[~rechazado], pd_pred[~rechazado])),
        "error_abs_medio_coef": float(np.mean(np.abs(coef - beta_true))),
        "error_intercepto": float(coef[0] - beta_true[0]),
        "sesgo_relativo_medio_pct": float(
            100 * np.mean((coef[1:] - beta_true[1:]) / np.abs(beta_true[1:]))
        ),
        "default_estimado_poblacion": float(pd_pred.mean()),
        "default_real_poblacion": float(y_real.mean()),
        "error_nivel_pp": float(100 * (pd_pred.mean() - y_real.mean())),
    }


def sensibilidad_parcelling(df: pd.DataFrame, factores=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0)) -> pd.DataFrame:
    """Cuanto depende parcelling del factor de castigo que alguien elige a dedo.

    En produccion no hay forma de validar el factor -- no se conoce la
    tasa mala real de los rechazados, que es precisamente el dato que
    falta. Aca si se conoce, y el barrido muestra que solo una franja
    angosta de factores da un error de nivel chico.
    """
    filas = []
    for factor in factores:
        res = parcelling(df[FEATURES].to_numpy(float),
                         np.nan_to_num(df["default_12m_observado"].to_numpy(float)),
                         df["aprobado"].to_numpy(int), factor=factor)
        m = evaluar(res["coef"], df)
        filas.append({"factor": factor, "error_nivel_pp": m["error_nivel_pp"],
                      "error_abs_medio_coef": m["error_abs_medio_coef"]})
    return pd.DataFrame(filas)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    gt = json.loads((RAW_DIR / "ground_truth.json").read_text())

    filas, coeficientes, detalles = [], [], {}
    for regimen in REGIMENES:
        df = pd.read_csv(RAW_DIR / f"applicants_{regimen}.csv")
        X = df[FEATURES].to_numpy(float)
        X_sel = df[FEATURES_SELECCION].to_numpy(float)
        aprobado = df["aprobado"].to_numpy(int)
        # La etiqueta que el banco realmente ve: NaN en los rechazados,
        # rellenada con 0 solo para que las funciones no fallen; ningun
        # metodo puede usarla, porque todos filtran por `aprobado`.
        y_obs = np.nan_to_num(df["default_12m_observado"].to_numpy(float))
        y_real = df["default_12m_real"].to_numpy(float)

        print(f"\n>> Regimen {regimen.upper()} "
              f"(rho verdadero = {rho_verdadero(regimen):+.2f}, "
              f"{aprobado.mean():.1%} aprobados)")

        for nombre, funcion in METODOS.items():
            t0 = time.time()
            res = funcion(X, X_sel, y_obs, aprobado, y_real)
            segundos = time.time() - t0

            metricas = evaluar(res["coef"], df)
            filas.append({
                "regimen": regimen, "metodo": nombre, "segundos": segundos,
                **metricas,
            })
            detalles[f"{regimen}/{nombre}"] = res["detalle"]
            coeficientes.append({
                "regimen": regimen, "metodo": nombre,
                **{f"coef_{n}": v for n, v in
                   zip(["intercepto", *FEATURES], res["coef"])},
            })
            print(f"   {nombre:<32} AUC pobl {metricas['auc_poblacion']:.4f} | "
                  f"AUC rech {metricas['auc_rechazados']:.4f} | "
                  f"err coef {metricas['error_abs_medio_coef']:.4f} | "
                  f"nivel {metricas['error_nivel_pp']:+.2f} pp | {segundos:.1f}s")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(REPORTS_DIR / "comparacion_metodos.csv", index=False)
    pd.DataFrame(coeficientes).to_csv(REPORTS_DIR / "coeficientes.csv", index=False)

    df_mnar = pd.read_csv(RAW_DIR / "applicants_mnar.csv")
    sens = sensibilidad_parcelling(df_mnar)
    sens.to_csv(REPORTS_DIR / "sensibilidad_parcelling.csv", index=False)
    print(f"\nSensibilidad de parcelling al factor de castigo (regimen MNAR)")
    print(sens.round(4).to_string(index=False))

    # rho estimado por el probit bivariado, con y sin instrumento
    rho_est = {
        regimen: {
            "con_instrumento": detalles[f"{regimen}/probit_bivariado"]["rho_estimado"],
            "sin_instrumento": detalles[
                f"{regimen}/probit_bivariado_sin_instrumento"]["rho_estimado"],
        }
        for regimen in REGIMENES
    }

    reporte = {
        "coeficientes_verdaderos": dict(zip(["intercepto", *FEATURES],
                                            coeficientes_verdaderos().tolist())),
        "rho_verdadero": {r: rho_verdadero(r) for r in REGIMENES},
        "rho_estimado_probit_bivariado": rho_est,
        "resumen_datos": gt["resumen"],
        "resultados": tabla.to_dict(orient="records"),
        "detalles": detalles,
        "sensibilidad_parcelling_mnar": sens.to_dict(orient="records"),
    }
    (REPORTS_DIR / "experiment_results.json").write_text(json.dumps(reporte, indent=2))

    print("\n" + "=" * 78)
    print("Correlacion de los errores (rho): lo que mide cuanta seleccion sobre no")
    print("observables hay, estimado sin verla nunca")
    print(f"{'':<8} {'verdadero':>10} {'sin instrumento':>17} {'con instrumento':>17}")
    for regimen in REGIMENES:
        r = rho_est[regimen]
        print(f"{regimen.upper():<8} {rho_verdadero(regimen):>10.3f} "
              f"{r['sin_instrumento']:>17.3f} {r['con_instrumento']:>17.3f}")


if __name__ == "__main__":
    main()
