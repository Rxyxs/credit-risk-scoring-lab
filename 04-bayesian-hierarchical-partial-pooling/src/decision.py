"""Politica de aprobacion con incertidumbre: usar la posterior completa,
no solo su media.

Un scorecard entrega un punto: PD = 8,4%. Un modelo bayesiano entrega una
distribucion, y dos solicitantes con la misma PD media pueden tener
incertidumbres muy distintas -- tipicamente porque uno viene de un segmento
con miles de casos y el otro de uno con cuarenta. La pregunta practica es
si conviene decidir con la media o con el borde superior del intervalo.

Se comparan dos politicas sobre exactamente la misma cartera de test:

- `media`        : aprueba a los de menor PD media posterior.
- `conservadora` : aprueba a los de menor percentil 95 de la PD posterior,
                   o sea exige estar razonablemente seguro de que el riesgo
                   es bajo, no solo que su mejor estimacion lo sea.

La comparacion se hace **a igual volumen aprobado**: para cada tasa de
aprobacion objetivo se toma el umbral que cada politica necesita para
aprobar esa misma fraccion. Comparar a umbral fijo seria tramposo, porque
el percentil 95 siempre es mayor que la media y aprobaria menos por
construccion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"

LGD = 0.45
MARGEN = 0.07            # ingreso esperado sobre el monto de un credito sano
SEGMENTO_CHICO = 50
TASAS_OBJETIVO = np.round(np.arange(0.50, 0.96, 0.05), 2)
POLITICAS = {"media": "pd_media", "conservadora": "pd_q95"}


def cargar() -> pd.DataFrame:
    pred = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    train = pd.read_csv(PROC_DIR / "train.csv")
    tam = train["segmento"].value_counts()
    pred["n_train_segmento"] = pred["segmento"].map(tam).fillna(0).astype(int)
    pred["segmento_chico"] = pred["n_train_segmento"] < SEGMENTO_CHICO
    return pred


def resultado_cartera(df: pd.DataFrame, aprobado: np.ndarray) -> dict:
    """Metricas realizadas de una decision, usando las etiquetas de test."""
    ap = df.loc[aprobado]
    if len(ap) == 0:
        return {"n_aprobados": 0, "tasa_aprobacion": 0.0, "tasa_mala_realizada": 0.0,
                "utilidad_realizada_clp": 0.0, "perdida_realizada_clp": 0.0}

    malos = ap["default_12m"].to_numpy(bool)
    monto = ap["monto_credito"].to_numpy(float)
    perdida = float((monto[malos] * LGD).sum())
    ingreso = float((monto[~malos] * MARGEN).sum())
    return {
        "n_aprobados": int(len(ap)),
        "tasa_aprobacion": float(aprobado.mean()),
        "tasa_mala_realizada": float(malos.mean()),
        "perdida_realizada_clp": perdida,
        "ingreso_realizado_clp": ingreso,
        "utilidad_realizada_clp": ingreso - perdida,
    }


def frontera(df: pd.DataFrame, tasas: np.ndarray = TASAS_OBJETIVO) -> pd.DataFrame:
    """Barrido de tasa de aprobacion objetivo para las dos politicas."""
    filas = []
    for nombre, col in POLITICAS.items():
        score = df[col].to_numpy(float)
        for tasa in tasas:
            umbral = float(np.quantile(score, tasa))
            aprobado = score <= umbral
            fila = {"politica": nombre, "tasa_objetivo": float(tasa), "umbral": umbral}
            fila.update(resultado_cartera(df, aprobado))
            filas.append(fila)
    return pd.DataFrame(filas)


def desacuerdos(df: pd.DataFrame, tasa: float = 0.80) -> dict:
    """Quienes cambian de decision al pasar de la media al percentil 95."""
    ap_media = df["pd_media"].to_numpy() <= np.quantile(df["pd_media"], tasa)
    ap_cons = df["pd_q95"].to_numpy() <= np.quantile(df["pd_q95"], tasa)

    rechazados_por_incertidumbre = ap_media & ~ap_cons
    aprobados_por_relajo = ~ap_media & ap_cons
    n = int(rechazados_por_incertidumbre.sum())

    resumen = {
        "tasa_aprobacion": float(tasa),
        "n_cambian_de_decision": n,
        "pct_cartera_que_cambia": float(n / len(df)),
        "tasa_mala_de_los_rechazados_por_incertidumbre": (
            float(df.loc[rechazados_por_incertidumbre, "default_12m"].mean()) if n else 0.0
        ),
        "tasa_mala_de_los_aprobados_por_relajo": (
            float(df.loc[aprobados_por_relajo, "default_12m"].mean())
            if aprobados_por_relajo.any() else 0.0
        ),
        "pct_de_segmentos_chicos_entre_los_rechazados": (
            float(df.loc[rechazados_por_incertidumbre, "segmento_chico"].mean()) if n else 0.0
        ),
        "pct_de_segmentos_chicos_en_la_cartera": float(df["segmento_chico"].mean()),
        "sd_posterior_media_rechazados": (
            float(df.loc[rechazados_por_incertidumbre, "pd_sd"].mean()) if n else 0.0
        ),
        "sd_posterior_media_cartera": float(df["pd_sd"].mean()),
    }
    return resumen


def comparar_a_igual_volumen(fr: pd.DataFrame, tasa: float = 0.80) -> dict:
    sel = fr[np.isclose(fr["tasa_objetivo"], tasa)].set_index("politica")
    m, c = sel.loc["media"], sel.loc["conservadora"]
    return {
        "tasa_aprobacion": float(tasa),
        "tasa_mala_media": float(m["tasa_mala_realizada"]),
        "tasa_mala_conservadora": float(c["tasa_mala_realizada"]),
        "delta_tasa_mala_pp": float(100 * (c["tasa_mala_realizada"] - m["tasa_mala_realizada"])),
        "utilidad_media_clp": float(m["utilidad_realizada_clp"]),
        "utilidad_conservadora_clp": float(c["utilidad_realizada_clp"]),
        "delta_utilidad_pct": float(
            100 * (c["utilidad_realizada_clp"] / m["utilidad_realizada_clp"] - 1)
        ),
    }


def experimento_datos_escasos(frac: float = 0.15, seed: int = 7) -> dict:
    """.Y si hubiera mucha menos data? Reajusta el modelo con una fraccion
    del train y repite la comparacion de politicas.

    La politica conservadora solo puede aportar algo cuando la
    incertidumbre posterior es material. Con la cartera completa esa
    incertidumbre es chica, asi que este experimento es la forma directa de
    probar el mecanismo en el regimen donde deberia importar, en vez de
    afirmarlo.
    """
    from src.hierarchical_logit import HierarchicalLogit, posterior_pd, resumen_pd
    from src.preprocessing import (
        SEGMENTOS, design_matrix, design_names, segment_index,
    )

    train = pd.read_csv(PROC_DIR / "train.csv")
    test = pd.read_csv(PROC_DIR / "test.csv")
    sub = train.sample(frac=frac, random_state=seed)

    draws = HierarchicalLogit(pooling="partial").fit(
        design_matrix(sub), sub["default_12m"].to_numpy(float), segment_index(sub),
        len(SEGMENTOS), design_names(), SEGMENTOS,
        n_draws=1500, n_warmup=500, n_chains=4, seed=seed,
    )
    r = resumen_pd(posterior_pd(draws, design_matrix(test), segment_index(test)))

    df = pd.DataFrame({
        "default_12m": test["default_12m"],
        "monto_credito": test["monto_credito"],
        "segmento": test["segmento"],
        "pd_media": r["pd_media"],
        "pd_q95": r["pd_q_hi"],
        "pd_sd": r["pd_sd"],
    })
    tam = sub["segmento"].value_counts()
    df["segmento_chico"] = df["segmento"].map(tam).fillna(0).astype(int) < SEGMENTO_CHICO

    fr = frontera(df)
    return {
        "n_train_reducido": int(len(sub)),
        "sd_posterior_media": float(df["pd_sd"].mean()),
        "comparacion_a_igual_volumen": {
            f"{t:.2f}": comparar_a_igual_volumen(fr, t) for t in (0.70, 0.80, 0.90)
        },
        "desacuerdos_al_80pct": desacuerdos(df, 0.80),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = cargar()

    fr = frontera(df)
    fr.to_csv(REPORTS_DIR / "frontera_decision.csv", index=False)

    comparaciones = {f"{t:.2f}": comparar_a_igual_volumen(fr, t) for t in (0.70, 0.80, 0.90)}
    desac = desacuerdos(df, 0.80)

    escaso = experimento_datos_escasos()

    reporte = {
        "supuestos": {"lgd": LGD, "margen_sobre_monto": MARGEN,
                      "umbral_segmento_chico": SEGMENTO_CHICO},
        "comparacion_a_igual_volumen": comparaciones,
        "desacuerdos_al_80pct": desac,
        "experimento_datos_escasos": escaso,
    }
    (REPORTS_DIR / "decision_results.json").write_text(json.dumps(reporte, indent=2))

    print("Politica de aprobacion: decidir con la media posterior vs con el percentil 95")
    print(f"  {'volumen':<9} {'mala (media)':>13} {'mala (cons.)':>13} {'delta pp':>9} "
          f"{'delta utilidad':>15}")
    for t, c in comparaciones.items():
        print(f"  {float(t):>7.0%}  {c['tasa_mala_media']:>12.2%} "
              f"{c['tasa_mala_conservadora']:>13.2%} {c['delta_tasa_mala_pp']:>9.2f} "
              f"{c['delta_utilidad_pct']:>14.2f}%")

    print(f"\nQuienes cambian de decision al 80% de aprobacion: "
          f"{desac['n_cambian_de_decision']} solicitudes "
          f"({desac['pct_cartera_que_cambia']:.1%} de la cartera)")
    print(f"  Tasa de default realizada de ese grupo : "
          f"{desac['tasa_mala_de_los_rechazados_por_incertidumbre']:.2%}")
    print(f"  Vienen de segmentos chicos             : "
          f"{desac['pct_de_segmentos_chicos_entre_los_rechazados']:.1%} "
          f"(en la cartera completa: {desac['pct_de_segmentos_chicos_en_la_cartera']:.1%})")
    print(f"  Desviacion posterior de su PD          : "
          f"{desac['sd_posterior_media_rechazados']:.4f} "
          f"(cartera: {desac['sd_posterior_media_cartera']:.4f})")

    print(f"\nMismo ejercicio con solo {escaso['n_train_reducido']:,} casos de "
          f"entrenamiento (sd posterior media {escaso['sd_posterior_media']:.4f})")
    print(f"  {'volumen':<9} {'mala (media)':>13} {'mala (cons.)':>13} {'delta pp':>9} "
          f"{'delta utilidad':>15}")
    for t, c in escaso["comparacion_a_igual_volumen"].items():
        print(f"  {float(t):>7.0%}  {c['tasa_mala_media']:>12.2%} "
              f"{c['tasa_mala_conservadora']:>13.2%} {c['delta_tasa_mala_pp']:>9.2f} "
              f"{c['delta_utilidad_pct']:>14.2f}%")


if __name__ == "__main__":
    main()
