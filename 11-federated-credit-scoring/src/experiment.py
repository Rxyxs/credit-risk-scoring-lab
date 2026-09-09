"""El experimento central: local, federado, y el oraculo centralizado que
la ley no permite construir -- en los dos regimenes de heterogeneidad.

Tres politicas de entrenamiento, comparadas siempre en la misma metrica
que importa: el desempeno sobre la poblacion **nacional agrupada**, no
sobre la propia cartera de cada banco (un banco chico y especializado
puede tener un AUC alto sobre sus propios clientes y aun asi ser inutil
para cualquiera que no se le parezca -- ese es justamente el problema que
el proyecto mide).

1. **Solo local**: cada banco entrena unicamente con sus datos.
2. **Federado (FedAvg)**: los datos nunca se juntan, solo los pesos.
3. **Oraculo centralizado**: como si los seis bancos pudieran juntar sus
   bases en una tabla -- imposible en la practica por secreto bancario,
   presente aca solo como cota superior.

Se corre en los dos escenarios de heterogeneidad (IID y no-IID) y se
barre `local_epochs` en el no-IID para medir el client drift: cuanta mas
computo local por ronda, menos comunicacion hace falta, pero mas se aleja
cada banco del optimo global antes de que la agregacion lo corrija.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_generator import BANCOS, FEATURES
from src.federated import (
    FedAvgLogisticRegression, _con_intercepto, gradient_descent_centralizado,
    local_update, sigmoide,
)

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
REPORTS_DIR = BASE / "outputs" / "reports"

LR = 0.6
EPOCAS_TOTALES = 200          # computo local total, fijo, para comparar politicas parejo
N_RONDAS_DEFAULT = 40
SEED = 42
BARRIDO_EPOCAS_LOCALES = [1, 2, 5, 10, 20, 40]


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    orden = np.argsort(score)
    yy = np.asarray(y)[orden]
    cum_bad = np.cumsum(yy) / max(yy.sum(), 1)
    cum_good = np.cumsum(1 - yy) / max((1 - yy).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def metricas(y: np.ndarray, pd_pred: np.ndarray) -> dict:
    """AUC/KS (ordenamiento) y calibracion (nivel), por separado.

    Un modelo puede ordenar bien a los solicitantes de una poblacion que
    nunca vio (AUC razonable) y aun asi decirle a esa poblacion un nivel
    de riesgo completamente equivocado -- que es exactamente lo que le
    pasa a un modelo entrenado solo con los clientes de un banco
    especializado cuando se aplica a la cartera nacional. La provision y
    el precio del credito dependen del NIVEL de PD, no solo del orden.
    """
    auc = float(roc_auc_score(y, pd_pred))
    y = np.asarray(y, dtype=float)
    pd_pred = np.asarray(pd_pred, dtype=float)
    return {
        "auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y, pd_pred),
        "brier": float(np.mean((pd_pred - y) ** 2)),
        "pd_media_predicha": float(pd_pred.mean()),
        "tasa_default_real": float(y.mean()),
        "sesgo_calibracion_pp": float(100 * (pd_pred.mean() - y.mean())),
    }


def _splits(df: pd.DataFrame, seed: int = SEED):
    """Split train/test por banco, mas los conjuntos agrupados (el
    'oraculo' de entrenamiento y el test nacional de evaluacion)."""
    trains, tests = {}, {}
    for banco in BANCOS:
        sub = df[df["banco"] == banco]
        tr, te = train_test_split(sub, test_size=0.30, random_state=seed,
                                  stratify=sub["default_12m"])
        trains[banco] = tr.reset_index(drop=True)
        tests[banco] = te.reset_index(drop=True)

    pool_train = pd.concat(trains.values(), ignore_index=True)
    pool_test = pd.concat(tests.values(), ignore_index=True)
    return trains, tests, pool_train, pool_test


def _Xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(float), df["default_12m"].to_numpy(float)


def evaluar_en(coef: np.ndarray, df: pd.DataFrame) -> dict:
    from src.federated import sigmoide, _con_intercepto
    X, y = _Xy(df)
    p = sigmoide(_con_intercepto(X) @ coef)
    return metricas(y, p)


def correr_escenario(escenario: str, local_epochs: int = 5,
                     n_rondas: int = N_RONDAS_DEFAULT) -> dict:
    df = pd.read_csv(RAW_DIR / f"carteras_{escenario}.csv")
    trains, tests, pool_train, pool_test = _splits(df)

    Xs_train = [_Xy(trains[b])[0] for b in BANCOS]
    ys_train = [_Xy(trains[b])[1] for b in BANCOS]
    p_features = len(FEATURES) + 1

    # --- 1. solo local: cada banco entrena SOLO con sus datos ------------
    # `local_update` espera la matriz de diseno CON intercepto, igual que
    # como la usa `FedAvgLogisticRegression` internamente -- se agrega aca
    # a mano para que las tres politicas corran exactamente el mismo paso
    # de optimizacion, y la comparacion sea sobre los datos, no sobre
    # detalles de implementacion.
    epocas_local = EPOCAS_TOTALES          # mismo presupuesto de computo que el resto
    coefs_locales = {
        b: local_update(np.zeros(p_features), _con_intercepto(_Xy(trains[b])[0]),
                        _Xy(trains[b])[1], lr=LR, epochs=epocas_local)
        for b in BANCOS
    }

    # --- 2. federado (FedAvg) ---------------------------------------------
    fed = FedAvgLogisticRegression(
        lr=LR, local_epochs=local_epochs,
        n_rounds=max(EPOCAS_TOTALES // local_epochs, 1), seed=SEED,
    ).fit(Xs_train, ys_train)

    # --- 3. oraculo centralizado -------------------------------------------
    coef_oraculo = local_update(np.zeros(p_features), _con_intercepto(_Xy(pool_train)[0]),
                                _Xy(pool_train)[1], lr=LR, epochs=EPOCAS_TOTALES)

    filas = []
    for b in BANCOS:
        filas.append({
            "escenario": escenario, "banco": b, "politica": "solo_local",
            **{f"eval_propio_{k}": v for k, v in evaluar_en(coefs_locales[b], tests[b]).items()},
            **{f"eval_nacional_{k}": v for k, v in evaluar_en(coefs_locales[b], pool_test).items()},
        })
        filas.append({
            "escenario": escenario, "banco": b, "politica": "federado",
            **{f"eval_propio_{k}": v for k, v in evaluar_en(fed.coef_, tests[b]).items()},
            **{f"eval_nacional_{k}": v for k, v in evaluar_en(fed.coef_, pool_test).items()},
        })
        filas.append({
            "escenario": escenario, "banco": b, "politica": "oraculo_centralizado",
            **{f"eval_propio_{k}": v for k, v in evaluar_en(coef_oraculo, tests[b]).items()},
            **{f"eval_nacional_{k}": v for k, v in evaluar_en(coef_oraculo, pool_test).items()},
        })

    deltas = fed.deltas_primera_ronda(Xs_train, ys_train)
    normas = np.linalg.norm(deltas, axis=1)
    cos_sim = deltas @ deltas.T / np.clip(np.outer(normas, normas), 1e-12, None)

    return {
        "tabla": pd.DataFrame(filas),
        "historia_fedavg": pd.DataFrame({
            "ronda": fed.historia_.ronda,
            "deriva_media_clientes": fed.historia_.deriva_media_clientes,
            "cambio_global": fed.historia_.cambio_global,
        }),
        "coef_oraculo": coef_oraculo,
        "coef_federado": fed.coef_,
        "similitud_updates_ronda1": pd.DataFrame(cos_sim, index=list(BANCOS), columns=list(BANCOS)),
        "nacional_oraculo": evaluar_en(coef_oraculo, pool_test),
        "nacional_federado": evaluar_en(fed.coef_, pool_test),
    }


def barrido_epocas_locales(escenario: str = "no_iid") -> pd.DataFrame:
    """Comunicacion vs client drift: mas epocas locales por ronda ahorra
    rondas, pero en datos no-IID aleja al modelo federado del oraculo."""
    df = pd.read_csv(RAW_DIR / f"carteras_{escenario}.csv")
    trains, tests, pool_train, pool_test = _splits(df)
    Xs_train = [_Xy(trains[b])[0] for b in BANCOS]
    ys_train = [_Xy(trains[b])[1] for b in BANCOS]

    filas = []
    for e in BARRIDO_EPOCAS_LOCALES:
        n_rondas = max(EPOCAS_TOTALES // e, 1)
        fed = FedAvgLogisticRegression(lr=LR, local_epochs=e, n_rounds=n_rondas, seed=SEED)
        fed.fit(Xs_train, ys_train)
        m = evaluar_en(fed.coef_, pool_test)
        filas.append({"local_epochs": e, "n_rondas": n_rondas,
                      "rondas_x_epocas_locales": n_rondas * e,
                      **{f"nacional_{k}": v for k, v in m.items()}})
    return pd.DataFrame(filas)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    resultados = {}
    for escenario in ("iid", "no_iid"):
        resultados[escenario] = correr_escenario(escenario)
        resultados[escenario]["tabla"].to_csv(
            REPORTS_DIR / f"comparacion_politicas_{escenario}.csv", index=False)
        resultados[escenario]["historia_fedavg"].to_csv(
            REPORTS_DIR / f"historia_fedavg_{escenario}.csv", index=False)
        resultados[escenario]["similitud_updates_ronda1"].to_csv(
            REPORTS_DIR / f"similitud_updates_{escenario}.csv")

    barrido = barrido_epocas_locales("no_iid")
    barrido.to_csv(REPORTS_DIR / "barrido_epocas_locales.csv", index=False)

    metricas_reportadas = ("auc", "gini", "ks", "brier", "sesgo_calibracion_pp")

    def _resumen_escenario(escenario: str) -> dict:
        tabla = resultados[escenario]["tabla"]
        solo_local = tabla[tabla["politica"] == "solo_local"].set_index("banco")
        return {
            "solo_local_promedio": {
                k: float(solo_local[f"eval_nacional_{k}"].mean())
                for k in metricas_reportadas
            },
            "solo_local_por_banco": {
                b: {
                    k: float(solo_local.loc[b, f"eval_nacional_{k}"])
                    for k in ("auc", "sesgo_calibracion_pp", "pd_media_predicha")
                }
                for b in BANCOS
            },
            "federado": resultados[escenario]["nacional_federado"],
            "oraculo_centralizado": resultados[escenario]["nacional_oraculo"],
        }

    reporte = {
        "lr": LR, "epocas_totales": EPOCAS_TOTALES, "bancos": list(BANCOS),
        "resumen_nacional": {
            escenario: _resumen_escenario(escenario) for escenario in ("iid", "no_iid")
        },
        "barrido_epocas_locales": barrido.to_dict(orient="records"),
    }
    (REPORTS_DIR / "experiment_results.json").write_text(json.dumps(reporte, indent=2))

    for escenario in ("iid", "no_iid"):
        r = reporte["resumen_nacional"][escenario]
        print(f"\n=== Escenario {escenario.upper()} -- desempeno sobre el test NACIONAL ===")
        print(f"  {'politica':<24} {'AUC':>7} {'Brier':>7} {'Sesgo calib. (pp)':>18}")
        for nombre, clave in [("Solo local (promedio)", "solo_local_promedio"),
                              ("Federado (FedAvg)", "federado"),
                              ("Oraculo centralizado", "oraculo_centralizado")]:
            m = r[clave]
            print(f"  {nombre:<24} {m['auc']:>7.4f} {m['brier']:>7.4f} "
                  f"{m['sesgo_calibracion_pp']:>+18.2f}")

        if escenario == "no_iid":
            print(f"\n  Sesgo de calibracion NACIONAL de cada modelo local, solo-local")
            print(f"  (cuanto se equivoca en el NIVEL de PD, no en el orden, al aplicarse "
                  f"fuera de su propia cartera)")
            for b, m in r["solo_local_por_banco"].items():
                print(f"    {b:<30} AUC nacional {m['auc']:.4f} | "
                      f"PD media predicha {m['pd_media_predicha']:.2%} | "
                      f"sesgo {m['sesgo_calibracion_pp']:>+7.2f} pp")

    print(f"\nBarrido de epocas locales por ronda (client drift medido en calibracion, NO-IID)")
    print(barrido[["local_epochs", "n_rondas", "nacional_auc",
                   "nacional_sesgo_calibracion_pp"]].round(4).to_string(index=False))
    print(f"\nReportes en {REPORTS_DIR}")


if __name__ == "__main__":
    main()
