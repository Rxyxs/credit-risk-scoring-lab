"""Ajusta los tres enfoques de pooling y los compara entre si y contra la
verdad de terreno.

Comparaciones que interesan:

1. .Recupera los efectos globales? (contra `beta_true` del simulador)
2. .Recupera los efectos de segmento? Y sobre todo: .en cuales segmentos
   gana el pooling parcial? (la respuesta esperada es "en los chicos", y
   hay que verificarla, no suponerla)
3. .Sirve fuera de muestra? AUC, KS, Brier y log-loss en test, mas un
   benchmark frecuentista de regresion logistica con dummies por segmento.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data_generator import FEATURES
from src.diagnostics import (
    convergio, tabla_diagnostico, tabla_diagnostico_efecto_total,
)
from src.hierarchical_logit import HierarchicalLogit, posterior_pd, resumen_pd
from src.preprocessing import (
    SEGMENTOS, design_matrix, design_names, segment_index,
)

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

N_DRAWS = 3000
N_WARMUP = 1000
N_CHAINS = 4
SEED = 42
SEGMENTO_CHICO = 50      # umbral de "segmento con poca informacion" en train


def _ks(y_true: np.ndarray, score: np.ndarray) -> float:
    orden = np.argsort(score)
    y = np.asarray(y_true)[orden]
    cum_bad = np.cumsum(y) / max(y.sum(), 1)
    cum_good = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def metricas(y_true: np.ndarray, pd_pred: np.ndarray) -> dict:
    auc = float(roc_auc_score(y_true, pd_pred))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": _ks(y_true, pd_pred),
        "brier": float(brier_score_loss(y_true, pd_pred)),
        "log_loss": float(log_loss(y_true, pd_pred)),
    }


def benchmark_frecuentista(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Logistica con dummies por segmento: el equivalente 'sin pooling' clasico."""
    dummies_train = pd.get_dummies(train["segmento"]).reindex(columns=SEGMENTOS, fill_value=0)
    dummies_test = pd.get_dummies(test["segmento"]).reindex(columns=SEGMENTOS, fill_value=0)
    Xtr = np.column_stack([train[FEATURES].to_numpy(float), dummies_train.to_numpy(float)])
    Xte = np.column_stack([test[FEATURES].to_numpy(float), dummies_test.to_numpy(float)])

    modelo = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-8)
    modelo.fit(Xtr, train["default_12m"].to_numpy())
    pred = modelo.predict_proba(Xte)[:, 1]
    return {"metricas": metricas(test["default_12m"].to_numpy(), pred), "pd_pred": pred}


def efectos_segmento(draws, tabla_tamanos: pd.DataFrame, gt: dict) -> pd.DataFrame:
    """Efecto estimado por segmento contra el verdadero, con su tamano en train."""
    verdad = gt["efectos_segmento_true"]
    b = draws.flat("b_segmento")
    filas = []
    for j, nombre in enumerate(SEGMENTOS):
        col = b[:, j]
        filas.append({
            "segmento": nombre,
            "n_train": int(tabla_tamanos.get(nombre, 0)),
            "efecto_true": float(verdad[nombre]),
            "efecto_post_media": float(col.mean()),
            "efecto_post_sd": float(col.std(ddof=1)),
            "q05": float(np.quantile(col, 0.05)),
            "q95": float(np.quantile(col, 0.95)),
        })
    df = pd.DataFrame(filas)
    # Los efectos verdaderos estan centrados en 0 por construccion, y sin
    # pooling el nivel de los b_j es arbitrario (solo la suma
    # intercepto + b_j esta identificada). Comparar sin centrar castigaria
    # a ese modelo por una constante que no cambia ninguna prediccion.
    df["efecto_post_centrado"] = df["efecto_post_media"] - df["efecto_post_media"].mean()
    df["error"] = df["efecto_post_centrado"] - df["efecto_true"]
    df["chico"] = df["n_train"] < SEGMENTO_CHICO
    return df


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PROC_DIR / "train.csv")
    test = pd.read_csv(PROC_DIR / "test.csv")
    gt = json.loads((RAW_DIR / "ground_truth.json").read_text())

    Xtr, Xte = design_matrix(train), design_matrix(test)
    ytr = train["default_12m"].to_numpy(float)
    yte = test["default_12m"].to_numpy()
    str_idx, ste_idx = segment_index(train), segment_index(test)
    tamanos = train["segmento"].value_counts()

    beta_true = np.array([gt["intercepto_true"], *[gt["beta_true"][f] for f in FEATURES]])

    # Solicitudes de test que caen en segmentos con poca informacion en train.
    chicos = set(tamanos[tamanos < SEGMENTO_CHICO].index)
    mascara_chicos = test["segmento"].isin(chicos).to_numpy()

    resultados, diagnosticos, efectos = {}, {}, {}
    predicciones = {}

    for pooling in ("complete", "none", "partial"):
        print(f"\n>> Muestreando pooling = {pooling} "
              f"({N_CHAINS} cadenas x {N_DRAWS} draws, {N_WARMUP} de warmup)")
        modelo = HierarchicalLogit(pooling=pooling)
        draws = modelo.fit(
            Xtr, ytr, str_idx, len(SEGMENTOS), design_names(), SEGMENTOS,
            n_draws=N_DRAWS, n_warmup=N_WARMUP, n_chains=N_CHAINS, seed=SEED,
        )
        print(f"   {draws.segundos:.1f}s")

        tabla = tabla_diagnostico(draws)
        diagnosticos[pooling] = tabla
        estado = convergio(tabla)
        if pooling != "complete":
            tabla_alpha = tabla_diagnostico_efecto_total(draws)
            diagnosticos[f"{pooling}_efecto_total"] = tabla_alpha
            estado_alpha = convergio(tabla_alpha)
        else:
            estado_alpha = None

        pd_te = posterior_pd(draws, Xte, ste_idx)
        resumen = resumen_pd(pd_te)
        predicciones[pooling] = resumen

        beta_media = draws.flat("beta").mean(axis=0)
        efectos[pooling] = efectos_segmento(draws, tamanos, gt)

        resultados[pooling] = {
            "segundos": draws.segundos,
            "convergencia": estado,
            "convergencia_efecto_total": estado_alpha,
            "beta_posterior": {
                nombre: {
                    "media": float(beta_media[j]),
                    "true": float(beta_true[j]),
                    "error": float(beta_media[j] - beta_true[j]),
                }
                for j, nombre in enumerate(design_names())
            },
            "error_abs_medio_beta": float(np.abs(beta_media - beta_true).mean()),
            "metricas_test": metricas(yte, resumen["pd_media"]),
            # El RMSE de efectos sobre 5 segmentos chicos es ruidoso; lo que
            # decide es como predice el modelo en las solicitudes que caen
            # justamente en esos segmentos.
            "metricas_test_segmentos_chicos": metricas(
                yte[mascara_chicos], resumen["pd_media"][mascara_chicos]
            ),
            "metricas_test_segmentos_grandes": metricas(
                yte[~mascara_chicos], resumen["pd_media"][~mascara_chicos]
            ),
            "rmse_efectos_segmento": {
                "todos": float(np.sqrt((efectos[pooling]["error"] ** 2).mean())),
                "chicos": float(np.sqrt(
                    (efectos[pooling].loc[efectos[pooling]["chico"], "error"] ** 2).mean()
                )),
                "grandes": float(np.sqrt(
                    (efectos[pooling].loc[~efectos[pooling]["chico"], "error"] ** 2).mean()
                )),
            },
            "ancho_medio_ic90_pd_test": float(
                np.mean(resumen["pd_q_hi"] - resumen["pd_q_lo"])
            ),
        }
        if pooling == "partial":
            tau = draws.flat("tau")
            resultados[pooling]["tau_posterior"] = {
                "media": float(tau.mean()),
                "q05": float(np.quantile(tau, 0.05)),
                "q95": float(np.quantile(tau, 0.95)),
                "true": gt["tau_true"],
            }

    bench = benchmark_frecuentista(train, test)
    resultados["benchmark_logistica_dummies"] = {"metricas_test": bench["metricas"]}

    # --- persistencia ---------------------------------------------------
    for pooling, tabla in diagnosticos.items():
        tabla.to_csv(REPORTS_DIR / f"diagnostico_{pooling}.csv", index=False)
    for pooling, df in efectos.items():
        df.to_csv(REPORTS_DIR / f"efectos_segmento_{pooling}.csv", index=False)

    r = predicciones["partial"]
    pd.DataFrame({
        "applicant_id": test["applicant_id"],
        "segmento": test["segmento"],
        "default_12m": yte,
        "pd_media": r["pd_media"],
        "pd_q05": r["pd_q_lo"],
        "pd_q95": r["pd_q_hi"],
        "pd_sd": r["pd_sd"],
        "pd_media_sin_pooling": predicciones["none"]["pd_media"],
        "pd_media_pooling_completo": predicciones["complete"]["pd_media"],
        "monto_credito": test["monto_credito"],
    }).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    (REPORTS_DIR / "fit_results.json").write_text(json.dumps(resultados, indent=2))

    # --- resumen en consola ---------------------------------------------
    print("\nRecuperacion de efectos globales (error absoluto medio del vector beta)")
    for pooling in ("complete", "none", "partial"):
        print(f"  {pooling:<9} {resultados[pooling]['error_abs_medio_beta']:.4f}")

    print("\nRMSE de los efectos de segmento contra la verdad")
    print(f"  {'pooling':<9} {'todos':>8} {'chicos':>8} {'grandes':>8}")
    for pooling in ("complete", "none", "partial"):
        r_ = resultados[pooling]["rmse_efectos_segmento"]
        print(f"  {pooling:<9} {r_['todos']:>8.4f} {r_['chicos']:>8.4f} {r_['grandes']:>8.4f}")

    n_chicos = int(mascara_chicos.sum())
    print(f"\nDesempeno en las {n_chicos} solicitudes de test de segmentos chicos "
          f"(<{SEGMENTO_CHICO} casos en train)")
    print(f"  {'modelo':<12} {'AUC':>7} {'Brier':>8} {'logloss':>8}")
    for nombre in ("complete", "none", "partial"):
        m = resultados[nombre]["metricas_test_segmentos_chicos"]
        print(f"  {nombre:<12} {m['auc']:>7.4f} {m['brier']:>8.4f} {m['log_loss']:>8.4f}")

    print("\nDesempeno fuera de muestra (2.700 solicitudes de test)")
    print(f"  {'modelo':<28} {'AUC':>7} {'KS':>7} {'Brier':>8} {'logloss':>8}")
    for nombre in ("complete", "none", "partial", "benchmark_logistica_dummies"):
        m = resultados[nombre]["metricas_test"]
        print(f"  {nombre:<28} {m['auc']:>7.4f} {m['ks']:>7.4f} {m['brier']:>8.4f} "
              f"{m['log_loss']:>8.4f}")

    tp = resultados["partial"]["tau_posterior"]
    print(f"\ntau posterior: {tp['media']:.3f} [{tp['q05']:.3f}, {tp['q95']:.3f}] "
          f"contra un valor verdadero de {tp['true']:.3f}")
    print("\nDiagnostico de convergencia")
    for pooling in ("complete", "none", "partial"):
        c = resultados[pooling]["convergencia"]
        print(f"  [{pooling}] parametros crudos     : R-hat max {c['rhat_max']:.4f}, "
              f"ESS min {c['ess_min']:>6.0f} -> {'OK' if c['convergio'] else 'REVISAR'}")
        ca = resultados[pooling]["convergencia_efecto_total"]
        if ca:
            print(f"  [{pooling}] efecto total por segmento: R-hat max {ca['rhat_max']:.4f}, "
                  f"ESS min {ca['ess_min']:>6.0f} -> {'OK' if ca['convergio'] else 'REVISAR'}")


if __name__ == "__main__":
    main()
