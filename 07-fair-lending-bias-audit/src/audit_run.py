"""Auditoria completa: modelo base, deteccion de proxies y cuatro escenarios
de mitigacion, cada uno con su costo.

Todos los escenarios se comparan **a la misma tasa de aprobacion global**
(80%). Comparar a umbral fijo seria tramposo: una mitigacion que solo
aprueba menos gente se ve mas equitativa por razones que no tienen que ver
con equidad.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_generator import (
    ATRIBUTO_PROTEGIDO, FEATURES_MODELO, GRUPO_PROTEGIDO, GRUPO_REFERENCIA,
)
from src.fairness_metrics import (
    bootstrap_metricas, brecha_condicional, decisiones, metricas_de_equidad,
    resumen_brecha_condicional, tabla_por_grupo, umbral_por_tasa_de_aprobacion,
)
from src.mitigations import (
    aplicar_umbrales_por_grupo, frontera_equidad_utilidad, pesos_reponderacion,
    resultado_economico, umbrales_por_grupo,
)
from src.proxy_analysis import fuerza_de_proxy, poder_de_reconstruccion, resumen_proxies

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

SEED = 42
TEST_SIZE = 0.30
TASA_APROBACION = 0.80
PARAMS = dict(max_iter=250, learning_rate=0.07, max_leaf_nodes=31,
              min_samples_leaf=40, random_state=SEED)


def matriz(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Numericas tal cual; el sector entra como codigo ordinal estable."""
    partes = []
    for f in features:
        col = df[f]
        if col.dtype == object:
            categorias = sorted(col.astype(str).unique())
            mapa = {c: i for i, c in enumerate(categorias)}
            partes.append(col.astype(str).map(mapa).to_numpy(float))
        else:
            partes.append(col.to_numpy(float))
    return np.column_stack(partes)


def entrenar(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
             pesos: np.ndarray | None = None) -> np.ndarray:
    modelo = HistGradientBoostingClassifier(**PARAMS)
    modelo.fit(matriz(train, features), train["default_12m"].to_numpy(),
               sample_weight=pesos)
    return modelo.predict_proba(matriz(test, features))[:, 1]


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DIR / "applicants.csv")
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                   stratify=df["default_12m"])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(PROC_DIR / "train.csv", index=False)
    test.to_csv(PROC_DIR / "test.csv", index=False)

    y_test = test["default_12m"].to_numpy()
    grupo_test = test[ATRIBUTO_PROTEGIDO].to_numpy()
    monto_test = test["monto_credito"].to_numpy(float)

    # ---------------------------------------------------------------- proxies
    proxies = fuerza_de_proxy(train, FEATURES_MODELO, ATRIBUTO_PROTEGIDO, GRUPO_PROTEGIDO)
    proxies.to_csv(REPORTS_DIR / "fuerza_de_proxy.csv", index=False)
    reconstruccion = poder_de_reconstruccion(
        matriz(train, FEATURES_MODELO),
        (train[ATRIBUTO_PROTEGIDO] == GRUPO_PROTEGIDO).astype(int).to_numpy(),
    )
    resumen_px = resumen_proxies(proxies)
    sospechosas = resumen_px["features_sospechosas"]

    # ------------------------------------------------------------ escenarios
    features_sin_proxy = [f for f in FEATURES_MODELO if f not in sospechosas] or FEATURES_MODELO
    es_protegido_train = (train[ATRIBUTO_PROTEGIDO] == GRUPO_PROTEGIDO).astype(int).to_numpy()
    pesos = pesos_reponderacion(es_protegido_train, train["default_12m"].to_numpy())

    predicciones = {
        "base_ciego": entrenar(train, test, FEATURES_MODELO),
        "sin_proxies": entrenar(train, test, features_sin_proxy),
        "reponderado": entrenar(train, test, FEATURES_MODELO, pesos=pesos),
    }
    # El cuarto escenario reutiliza el modelo base y cambia solo la decision.
    predicciones["umbral_por_grupo"] = predicciones["base_ciego"]

    resultados, tablas_grupo = {}, {}
    for nombre, pd_pred in predicciones.items():
        if nombre == "umbral_por_grupo":
            umbrales = umbrales_por_grupo(pd_pred, grupo_test, TASA_APROBACION)
            aprobado = aplicar_umbrales_por_grupo(pd_pred, grupo_test, umbrales)
            # para las metricas se usa el umbral del grupo de referencia como
            # umbral nominal, y la decision ya viene calculada
            umbral = float(umbrales[GRUPO_REFERENCIA])
            tabla = tabla_por_grupo(y_test, pd_pred, grupo_test, umbral)
            tasas = {g: float(aprobado[grupo_test == g].mean()) for g in umbrales}
            ratio = tasas[GRUPO_PROTEGIDO] / tasas[GRUPO_REFERENCIA]
            metricas = metricas_de_equidad(y_test, pd_pred, grupo_test, umbral,
                                           GRUPO_PROTEGIDO, GRUPO_REFERENCIA)
            metricas.update({
                "umbral_pd": umbral,
                "umbrales_por_grupo": {g: float(u) for g, u in umbrales.items()},
                "tasa_seleccion_protegido": tasas[GRUPO_PROTEGIDO],
                "tasa_seleccion_referencia": tasas[GRUPO_REFERENCIA],
                "ratio_impacto_adverso": float(ratio),
                "cumple_regla_cuatro_quintos": bool(ratio >= 0.80),
                "paridad_demografica_pp": float(
                    100 * (tasas[GRUPO_PROTEGIDO] - tasas[GRUPO_REFERENCIA])
                ),
            })
        else:
            umbral = umbral_por_tasa_de_aprobacion(pd_pred, TASA_APROBACION)
            aprobado = decisiones(pd_pred, umbral)
            tabla = tabla_por_grupo(y_test, pd_pred, grupo_test, umbral)
            metricas = metricas_de_equidad(y_test, pd_pred, grupo_test, umbral,
                                           GRUPO_PROTEGIDO, GRUPO_REFERENCIA)

        tablas_grupo[nombre] = tabla
        resultados[nombre] = {
            "auc": float(roc_auc_score(y_test, pd_pred)),
            "features": FEATURES_MODELO if nombre != "sin_proxies" else features_sin_proxy,
            "equidad": metricas,
            "economia": resultado_economico(y_test, aprobado, monto_test),
            "por_grupo": tabla.to_dict(orient="records"),
        }
        tabla.to_csv(REPORTS_DIR / f"por_grupo_{nombre}.csv", index=False)

    # ------------------------------------------------- descomposicion y CIs
    pd_base = predicciones["base_ciego"]
    test_con_score = test.assign(pd_pred=pd_base)
    tabla_cond = brecha_condicional(
        test_con_score, ATRIBUTO_PROTEGIDO, GRUPO_PROTEGIDO, GRUPO_REFERENCIA,
        "pd_pred", ["log_renta", "dti"], n_bins=4,
    )
    brecha_bruta = float(
        test_con_score.loc[test_con_score[ATRIBUTO_PROTEGIDO] == GRUPO_PROTEGIDO, "pd_pred"].mean()
        - test_con_score.loc[test_con_score[ATRIBUTO_PROTEGIDO] == GRUPO_REFERENCIA, "pd_pred"].mean()
    )
    descomposicion = resumen_brecha_condicional(tabla_cond, brecha_bruta)
    tabla_cond.to_csv(REPORTS_DIR / "brecha_condicional.csv", index=False)

    umbral_base = umbral_por_tasa_de_aprobacion(pd_base, TASA_APROBACION)
    ic = bootstrap_metricas(y_test, pd_base, grupo_test, umbral_base,
                            GRUPO_PROTEGIDO, GRUPO_REFERENCIA)
    ic.to_csv(REPORTS_DIR / "intervalos_bootstrap.csv", index=False)

    frontera = frontera_equidad_utilidad(y_test, pd_base, grupo_test, monto_test,
                                         GRUPO_PROTEGIDO, GRUPO_REFERENCIA)
    frontera.to_csv(REPORTS_DIR / "frontera_equidad_utilidad.csv", index=False)

    pd.DataFrame({
        "applicant_id": test["applicant_id"],
        "genero": grupo_test,
        "sector": test["sector"],
        "default_12m": y_test,
        "pd_base": pd_base,
        "pd_sin_proxies": predicciones["sin_proxies"],
        "pd_reponderado": predicciones["reponderado"],
        "monto_credito": monto_test,
    }).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    reporte = {
        "atributo_protegido": ATRIBUTO_PROTEGIDO,
        "grupo_protegido": GRUPO_PROTEGIDO,
        "tasa_aprobacion_comparada": TASA_APROBACION,
        "n_train": len(train), "n_test": len(test),
        "proxies": {**resumen_px, "reconstruccion": reconstruccion,
                    "tabla": proxies.to_dict(orient="records")},
        "descomposicion_de_la_brecha": descomposicion,
        "intervalos_bootstrap": ic.to_dict(orient="records"),
        "escenarios": resultados,
    }
    (REPORTS_DIR / "audit_results.json").write_text(json.dumps(reporte, indent=2))

    # ------------------------------------------------------------- consola
    print(f"Auditoria sobre {len(test):,} solicitudes de test "
          f"({(grupo_test == GRUPO_PROTEGIDO).mean():.1%} del grupo protegido)")
    print(f"Todas las politicas comparadas al {TASA_APROBACION:.0%} de aprobacion global\n")

    print("Deteccion de proxies")
    print(f"  El grupo se puede reconstruir desde las features del modelo con AUC "
          f"{reconstruccion['auc_reconstruccion_grupo']:.4f} "
          f"({reconstruccion['interpretacion']})")
    print(proxies.round(4).to_string(index=False))

    print(f"\nDescomposicion de la brecha de PD predicha")
    d = descomposicion
    print(f"  Brecha bruta        : {d['brecha_bruta'] * 100:+.3f} pp")
    print(f"  Brecha condicional  : {d['brecha_condicional'] * 100:+.3f} pp "
          f"(comparando perfiles equivalentes en renta y carga financiera)")
    print(f"  Explicado por factores legitimos: "
          f"{d['pct_explicado_por_factores_legitimos']:.1f}%")

    print(f"\n{'escenario':<18} {'AUC':>7} {'ratio 4/5':>10} {'parid. dem.':>12} "
          f"{'igual. oport.':>14} {'utilidad CLP':>16}")
    for nombre, r in resultados.items():
        e, eco = r["equidad"], r["economia"]
        print(f"{nombre:<18} {r['auc']:>7.4f} {e['ratio_impacto_adverso']:>10.4f} "
              f"{e['paridad_demografica_pp']:>11.2f}p {e['igualdad_oportunidad_pp']:>13.2f}p "
              f"{eco['utilidad_clp']:>16,.0f}")

    print("\nIntervalos de confianza al 95% (modelo base, bootstrap)")
    print(ic.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
