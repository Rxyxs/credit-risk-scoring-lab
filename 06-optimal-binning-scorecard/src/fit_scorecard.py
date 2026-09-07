"""Compara tres formas de binear las mismas variables y arma el scorecard.

La comparacion se hace de punta a punta y en dos horizontes distintos:

- **In-time**: 30% de las cohortes estables, que es lo que un equipo de
  riesgo reporta como validacion.
- **Out-of-time**: las 6 cohortes con la poblacion deteriorada, que es lo
  que realmente pasa cuando el modelo lleva medio ano en produccion.

Separarlos importa: la diferencia entre los dos numeros dice mucho mas
sobre un scorecard que cualquiera de los dos por separado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.binning import (
    OptimalBinner, binning_arbol, binning_equifrecuente, interpretar_iv,
)
from src.data_generator import CATEGORICAS, NUMERICAS, VINTAGE_QUIEBRE
from src.scorecard import ajustar_scorecard, bandas_de_riesgo, metricas

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

SEED = 42
FRAC_VALIDACION = 0.30
MAX_BINS = 6
METODOS = ("dp_monotono", "dp_libre", "equifrecuente", "arbol")


def cargar_y_partir() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW_DIR / "applicants.csv")
    estables = df[df["vintage_idx"] < VINTAGE_QUIEBRE].reset_index(drop=True)
    fuera_de_tiempo = df[df["vintage_idx"] >= VINTAGE_QUIEBRE].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    mascara = rng.random(len(estables)) < FRAC_VALIDACION
    return (estables[~mascara].reset_index(drop=True),
            estables[mascara].reset_index(drop=True),
            fuera_de_tiempo)


def binear_todo(train: pd.DataFrame, y: np.ndarray, metodo: str) -> dict:
    """Aplica un metodo de binning a todas las variables del scorecard."""
    binnings = {}
    for var in NUMERICAS:
        x = train[var].to_numpy(float)
        if metodo == "dp_monotono":
            binnings[var] = OptimalBinner(max_bins=MAX_BINS).fit_numerica(x, y, var)
        elif metodo == "dp_libre":
            binnings[var] = OptimalBinner(
                max_bins=MAX_BINS, monotono=False).fit_numerica(x, y, var)
        elif metodo == "equifrecuente":
            binnings[var] = binning_equifrecuente(x, y, MAX_BINS, var)
        elif metodo == "arbol":
            binnings[var] = binning_arbol(x, y, MAX_BINS, nombre=var)
        else:
            raise ValueError(f"metodo desconocido: {metodo}")

    # Las categoricas se agrupan siempre con la DP: no hay version
    # "equifrecuente" sensata de una variable sin orden natural.
    for var in CATEGORICAS:
        binnings[var] = OptimalBinner(max_bins=MAX_BINS).fit_categorica(
            train[var].to_numpy(), y, var
        )
    return binnings


def convergencia_de_grilla(x: np.ndarray, y: np.ndarray,
                           grillas=(10, 20, 40, 80, 160)) -> pd.DataFrame:
    """IV alcanzado segun la resolucion del pre-binning.

    La DP es exacta *sobre la grilla*; este barrido muestra cuanto de la
    diferencia con un metodo de cortes libres es resolucion y no
    optimalidad.
    """
    filas = []
    for g in grillas:
        r = OptimalBinner(max_bins=MAX_BINS, monotono=False, n_prebins=g).fit_numerica(x, y)
        filas.append({"n_prebins": g, "iv": r.iv, "n_bins": r.n_bins})
    return pd.DataFrame(filas)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train, val_in_time, val_out_of_time = cargar_y_partir()
    y_tr = train["default_12m"].to_numpy(float)
    y_in = val_in_time["default_12m"].to_numpy()
    y_out = val_out_of_time["default_12m"].to_numpy()

    resultados, tablas_iv, scorecards = {}, {}, {}
    for metodo in METODOS:
        binnings = binear_todo(train, y_tr, metodo)
        sc = ajustar_scorecard(train, y_tr, binnings)
        scorecards[metodo] = sc

        iv_por_var = pd.DataFrame([
            {"variable": v, "iv": r.iv, "n_bins": r.n_bins, "monotono": r.monotono,
             "lectura": interpretar_iv(r.iv)}
            for v, r in binnings.items()
        ]).sort_values("iv", ascending=False)
        tablas_iv[metodo] = iv_por_var

        resultados[metodo] = {
            "iv_total": float(iv_por_var["iv"].sum()),
            "variables_no_monotonas": iv_por_var.loc[~iv_por_var["monotono"], "variable"].tolist(),
            "bins_totales": int(iv_por_var["n_bins"].sum()),
            "in_time": metricas(y_in, sc.predict_proba(val_in_time)),
            "out_of_time": metricas(y_out, sc.predict_proba(val_out_of_time)),
        }

    mejor = max(METODOS, key=lambda m: resultados[m]["out_of_time"]["auc"])
    sc = scorecards["dp_monotono"]        # el que se lleva a produccion

    sc.tarjeta.to_csv(REPORTS_DIR / "tarjeta_scorecard.csv", index=False)
    for metodo, tabla in tablas_iv.items():
        tabla.to_csv(REPORTS_DIR / f"iv_por_variable_{metodo}.csv", index=False)
    for var, res in sc.binnings.items():
        res.tabla.to_csv(REPORTS_DIR / f"bins_{var}.csv", index=False)

    score_in = sc.score(val_in_time)
    score_out = sc.score(val_out_of_time)
    bandas = bandas_de_riesgo(score_in, y_in)
    bandas.to_csv(REPORTS_DIR / "bandas_de_riesgo.csv", index=False)

    pd.DataFrame({
        "applicant_id": val_in_time["applicant_id"],
        "vintage_idx": val_in_time["vintage_idx"],
        "default_12m": y_in,
        "score": score_in,
        "pd": sc.predict_proba(val_in_time),
    }).to_csv(REPORTS_DIR / "scores_in_time.csv", index=False)
    pd.DataFrame({
        "applicant_id": val_out_of_time["applicant_id"],
        "vintage_idx": val_out_of_time["vintage_idx"],
        "default_12m": y_out,
        "score": score_out,
        "pd": sc.predict_proba(val_out_of_time),
    }).to_csv(REPORTS_DIR / "scores_out_of_time.csv", index=False)

    convergencia = convergencia_de_grilla(train["dti"].to_numpy(float), y_tr)
    convergencia.to_csv(REPORTS_DIR / "convergencia_grilla.csv", index=False)

    train.to_csv(PROC_DIR / "train.csv", index=False)
    val_in_time.to_csv(PROC_DIR / "val_in_time.csv", index=False)
    val_out_of_time.to_csv(PROC_DIR / "val_out_of_time.csv", index=False)

    reporte = {
        "n_train": len(train),
        "n_in_time": len(val_in_time),
        "n_out_of_time": len(val_out_of_time),
        "tasa_mala_train": float(y_tr.mean()),
        "tasa_mala_out_of_time": float(y_out.mean()),
        "metodos": resultados,
        "mejor_out_of_time": mejor,
        "convergencia_grilla_dti": convergencia.to_dict(orient="records"),
        "bandas_de_riesgo": bandas.to_dict(orient="records"),
        "scorecard_puntos": {
            "pdo": sc.pdo, "puntaje_base": sc.puntaje_base, "odds_base": sc.odds_base,
            "score_medio_in_time": float(score_in.mean()),
            "score_medio_out_of_time": float(score_out.mean()),
        },
    }
    (REPORTS_DIR / "scorecard_results.json").write_text(json.dumps(reporte, indent=2))

    print(f"Train {len(train):,} | in-time {len(val_in_time):,} | "
          f"out-of-time {len(val_out_of_time):,} "
          f"(tasa mala {y_tr.mean():.2%} vs {y_out.mean():.2%})")
    print(f"\n{'metodo':<16} {'IV total':>9} {'no monotonas':>13} "
          f"{'AUC in-time':>12} {'KS in-time':>11} {'AUC out':>9} {'KS out':>8}")
    for metodo in METODOS:
        r = resultados[metodo]
        print(f"{metodo:<16} {r['iv_total']:>9.4f} {len(r['variables_no_monotonas']):>13} "
              f"{r['in_time']['auc']:>12.4f} {r['in_time']['ks']:>11.4f} "
              f"{r['out_of_time']['auc']:>9.4f} {r['out_of_time']['ks']:>8.4f}")

    print(f"\nTarjeta del scorecard (dp_monotono), primeras filas")
    print(sc.tarjeta.head(12).to_string(index=False))
    print(f"\nBandas de riesgo (validacion in-time)")
    print(bandas.round(4).to_string(index=False))
    print(f"\nConvergencia del IV con la resolucion de la grilla (dti)")
    print(convergencia.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
