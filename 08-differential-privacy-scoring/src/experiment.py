"""El experimento central: cuanto cuesta la privacidad y cuanto compra.

Para una grilla de presupuestos epsilon se calibra el ruido necesario, se
entrena el scorecard con DP-SGD, y se mide en la misma corrida:

- **Utilidad**: AUC, KS y calibracion sobre un test que no participo del
  entrenamiento.
- **Privacidad efectiva**: el AUC de un ataque de inferencia de membresia y
  la exposicion de los canarios.

Reportar solo lo primero deja la impresion de que la privacidad es puro
costo; reportar solo lo segundo deja la impresion de que es gratis. La
tabla completa es la unica forma honesta de mostrar el canje.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.accountant import calibrar_sigma, interpretar_epsilon
from src.attacks import (
    ataque_de_membresia, exposicion_de_canarios, riesgo_de_reidentificacion,
)
from src.data_generator import FEATURES, generar_canarios
from src.dp_sgd import DPLogisticRegression

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0]
DELTA = 1e-5
CLIP = 3.0
LR = 1.0
PASOS = 2500
TAM_LOTE = 128
SEED = 42
# DP-SGD es un mecanismo aleatorio: una sola corrida mezcla el efecto del
# ruido con la suerte de una semilla. Toda metrica se promedia sobre
# repeticiones independientes y se reporta con su desviacion.
N_REPETICIONES = 10


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    orden = np.argsort(score)
    yy = np.asarray(y)[orden]
    cum_bad = np.cumsum(yy) / max(yy.sum(), 1)
    cum_good = np.cumsum(1 - yy) / max((1 - yy).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def metricas(y: np.ndarray, pd_pred: np.ndarray) -> dict:
    auc = float(roc_auc_score(y, pd_pred))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, pd_pred),
        "brier": float(np.mean((pd_pred - y) ** 2)),
        "pd_media": float(pd_pred.mean()),
    }


def _estandarizar(train: pd.DataFrame):
    mu = train[FEATURES].mean()
    sd = train[FEATURES].std(ddof=0).replace(0, 1.0)
    return mu, sd


def _X(df: pd.DataFrame, mu, sd) -> np.ndarray:
    return ((df[FEATURES] - mu) / sd).to_numpy(float)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_DIR / "train.csv")
    holdout = pd.read_csv(RAW_DIR / "holdout.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")
    canarios = pd.read_csv(RAW_DIR / "canarios.csv")

    # Canarios sombra: mismo perfil, nunca vistos por el modelo.
    sombra = generar_canarios(len(canarios), np.random.default_rng(SEED + 777))

    mu, sd = _estandarizar(train)
    X_train, y_train = _X(train, mu, sd), train["default_12m"].to_numpy(float)
    X_hold, y_hold = _X(holdout, mu, sd), holdout["default_12m"].to_numpy(float)
    X_test, y_test = _X(test, mu, sd), test["default_12m"].to_numpy()
    X_can, X_sombra = _X(canarios, mu, sd), _X(sombra, mu, sd)

    q = TAM_LOTE / len(train)

    corridas = []
    modelos = {}
    for eps_objetivo in [None, *EPSILONS]:
        if eps_objetivo is None:
            nombre, sigma = "sin_privacidad", 0.0
        else:
            nombre = f"epsilon_{eps_objetivo:g}"
            sigma = calibrar_sigma(q, PASOS, eps_objetivo, DELTA)

        repeticiones = []
        for r in range(N_REPETICIONES):
            modelo = DPLogisticRegression(clip=CLIP, sigma=sigma, q=q, pasos=PASOS,
                                          lr=LR, seed=SEED + 101 * r).fit(X_train, y_train)
            if r == 0:
                modelos[nombre] = modelo        # representativo, para graficos
            membresia = ataque_de_membresia(modelo, X_train, y_train, X_hold, y_hold)
            canario = exposicion_de_canarios(modelo, X_can, X_sombra)
            repeticiones.append({
                **{f"utilidad_{k}": v for k, v in
                   metricas(y_test, modelo.predict_proba(X_test)[:, 1]).items()},
                "ataque_auc": membresia["auc_ataque"],
                "ataque_ventaja": membresia["ventaja"],
                "brecha_de_perdida": membresia["brecha_de_perdida"],
                "canario_exposicion_pp": canario["exposicion_pp"],
                "canario_pd_entrenados": canario["pd_media_canarios_entrenados"],
                "canario_pd_sombra": canario["pd_media_canarios_sombra"],
                "fraccion_recortada_media": float(
                    np.mean(modelo.historial_.fraccion_recortada)
                ),
                "distancia_coeficientes": float(np.linalg.norm(modelo.coef_)),
            })

        reps = pd.DataFrame(repeticiones)
        presupuesto = modelos[nombre].presupuesto_privacidad(DELTA)
        riesgo = riesgo_de_reidentificacion(
            {"ventaja": float(reps["ataque_ventaja"].mean())}
        )

        fila = {
            "escenario": nombre,
            "epsilon_objetivo": eps_objetivo,
            "epsilon_real": presupuesto["epsilon"],
            "sigma": sigma,
            "n_repeticiones": N_REPETICIONES,
            "lectura_epsilon": (
                "sin garantia formal" if sigma == 0
                else interpretar_epsilon(presupuesto["epsilon"])
            ),
            "lectura_filtracion": riesgo["lectura"],
        }
        for col in reps.columns:
            fila[col] = float(reps[col].mean())
            fila[f"{col}_sd"] = float(reps[col].std(ddof=1))

        # La pregunta operativa no es "cuanta fuga hay" sino "se puede
        # distinguir de cero". Con el ruido de DP, una exposicion promedio
        # de 2 pp con desviacion de 5 pp entre corridas no es evidencia de
        # nada; el estadistico t sobre las repeticiones lo hace explicito.
        error_estandar = fila["canario_exposicion_pp_sd"] / np.sqrt(N_REPETICIONES)
        t = fila["canario_exposicion_pp"] / max(error_estandar, 1e-12)
        fila["canario_t"] = float(t)
        fila["fuga_detectable"] = bool(abs(t) > 2.0)
        corridas.append(fila)

    tabla = pd.DataFrame(corridas)
    tabla.to_csv(REPORTS_DIR / "privacidad_vs_utilidad.csv", index=False)

    # Distancia de los coeficientes al modelo sin privacidad.
    base = modelos["sin_privacidad"].coef_
    coefs = pd.DataFrame(
        {n: m.coef_ for n, m in modelos.items()},
        index=["intercepto", *FEATURES],
    )
    coefs.to_csv(REPORTS_DIR / "coeficientes.csv")
    distancias = {
        n: float(np.linalg.norm(m.coef_ - base) / np.linalg.norm(base))
        for n, m in modelos.items()
    }

    # Curvas de perdida para ver el efecto del ruido en el entrenamiento.
    perdidas = pd.DataFrame({n: m.historial_.perdida for n, m in modelos.items()})
    perdidas.to_csv(REPORTS_DIR / "curvas_de_perdida.csv", index=False)

    pd.DataFrame({
        "record_id": test["record_id"],
        "default_12m": y_test,
        **{f"pd_{n}": m.predict_proba(X_test)[:, 1] for n, m in modelos.items()},
    }).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    reporte = {
        "configuracion": {
            "n_train": int(len(train)), "n_holdout": int(len(holdout)),
            "n_test": int(len(test)), "n_canarios": int(len(canarios)),
            "clip": CLIP, "lr": LR, "pasos": PASOS,
            "tamano_lote_esperado": TAM_LOTE, "q": q, "delta": DELTA,
        },
        "corridas": corridas,
        "distancia_relativa_de_coeficientes": distancias,
    }
    (REPORTS_DIR / "experiment_results.json").write_text(json.dumps(reporte, indent=2))

    print(f"Train {len(train):,} (incluye {len(canarios)} canarios) | "
          f"holdout {len(holdout):,} | test {len(test):,}")
    print(f"DP-SGD: clip {CLIP}, lr {LR}, {PASOS} pasos, lote esperado {TAM_LOTE} "
          f"(q = {q:.4f}), delta = {DELTA}\n")

    print(f"Cada fila es el promedio de {N_REPETICIONES} corridas independientes "
          f"(+- desviacion)\n")
    print(f"{'escenario':<16} {'sigma':>7} {'epsilon':>8} {'AUC':>16} "
          f"{'ataque AUC':>16} {'canarios pp':>16} {'t':>7} {'fuga':>4}")
    for c in corridas:
        eps = "inf" if np.isinf(c["epsilon_real"]) else f"{c['epsilon_real']:.2f}"
        print(f"{c['escenario']:<16} {c['sigma']:>7.2f} {eps:>8} "
              f"{c['utilidad_auc']:>9.4f} +-{c['utilidad_auc_sd']:<5.4f} "
              f"{c['ataque_auc']:>9.4f} +-{c['ataque_auc_sd']:<5.4f} "
              f"{c['canario_exposicion_pp']:>9.2f} +-{c['canario_exposicion_pp_sd']:<5.2f} "
              f"{c['canario_t']:>7.1f} {'SI' if c['fuga_detectable'] else 'no':>4}")

    print("\nDistancia relativa de los coeficientes respecto del modelo sin privacidad")
    for n, d in distancias.items():
        print(f"  {n:<16} {d:.4f}")


if __name__ == "__main__":
    main()
