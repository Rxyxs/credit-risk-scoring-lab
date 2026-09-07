"""Mitigaciones: cuatro formas de intervenir, y lo que cuesta cada una.

Ordenadas por donde intervienen en el pipeline, que es como conviene
pensarlas:

1. **Ciego (`unawareness`)**: no usar el atributo protegido. Es el punto de
   partida legal en credito y, como se ve en los resultados, no elimina la
   disparidad -- porque el modelo nunca lo estaba usando de forma directa.
2. **Sin proxies (`pre-proceso, features`)**: sacar las variables que
   cargan pertenencia al grupo sin aportar riesgo. Interviene en los datos.
3. **Reponderacion (`pre-proceso, pesos`, Kamiran & Calders 2012)**: pesar
   cada observacion por el cociente entre la frecuencia esperada bajo
   independencia grupo-etiqueta y la observada. No toca las features ni el
   algoritmo: solo la importancia relativa de cada caso en el entrenamiento.
4. **Umbrales por grupo (`post-proceso`)**: cortar en un punto distinto por
   grupo hasta igualar tasas. Es la mitigacion mas eficaz y **la mas
   restringida legalmente**: usar el atributo protegido en la decision es
   precisamente lo que la normativa de credito prohibe en la mayoria de las
   jurisdicciones. Se incluye como referencia analitica -- el limite de lo
   que se podria lograr -- y no como politica desplegable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fairness_metrics import decisiones, umbral_por_tasa_de_aprobacion


def pesos_reponderacion(es_protegido: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pesos de Kamiran & Calders: esperado bajo independencia / observado.

    Si en el grupo protegido hay menos casos buenos de los que habria bajo
    independencia entre grupo y etiqueta, esos casos pesan mas al entrenar.
    """
    g = np.asarray(es_protegido, dtype=int)
    y = np.asarray(y, dtype=int)
    if g.size != y.size:
        raise ValueError("grupo y etiqueta deben tener el mismo largo")
    n = g.size

    pesos = np.ones(n, dtype=float)
    for gv in (0, 1):
        for yv in (0, 1):
            celda = (g == gv) & (y == yv)
            n_celda = celda.sum()
            if n_celda == 0:
                continue
            esperado = (g == gv).sum() * (y == yv).sum() / n
            pesos[celda] = esperado / n_celda
    return pesos


def umbrales_por_grupo(pd_pred: np.ndarray, grupo: np.ndarray,
                       tasa_objetivo: float) -> dict:
    """Un umbral por grupo, cada uno calibrado a la misma tasa de aprobacion."""
    pd_pred = np.asarray(pd_pred, dtype=float)
    grupo = np.asarray(grupo)
    return {
        g: umbral_por_tasa_de_aprobacion(pd_pred[grupo == g], tasa_objetivo)
        for g in np.unique(grupo)
    }


def aplicar_umbrales_por_grupo(pd_pred: np.ndarray, grupo: np.ndarray,
                               umbrales: dict) -> np.ndarray:
    pd_pred = np.asarray(pd_pred, dtype=float)
    grupo = np.asarray(grupo)
    aprobado = np.zeros(pd_pred.size, dtype=int)
    for g, u in umbrales.items():
        m = grupo == g
        aprobado[m] = decisiones(pd_pred[m], u)
    return aprobado


def resultado_economico(y: np.ndarray, aprobado: np.ndarray, monto: np.ndarray,
                        lgd: float = 0.45, margen: float = 0.07) -> dict:
    """Utilidad realizada de una politica de aprobacion."""
    y = np.asarray(y, dtype=int)
    aprobado = np.asarray(aprobado, dtype=int).astype(bool)
    monto = np.asarray(monto, dtype=float)
    if not aprobado.any():
        return {"n_aprobados": 0, "tasa_aprobacion": 0.0, "tasa_mala_aprobados": 0.0,
                "utilidad_clp": 0.0}
    malos = y[aprobado] == 1
    ingreso = float((monto[aprobado][~malos] * margen).sum())
    perdida = float((monto[aprobado][malos] * lgd).sum())
    return {
        "n_aprobados": int(aprobado.sum()),
        "tasa_aprobacion": float(aprobado.mean()),
        "tasa_mala_aprobados": float(malos.mean()),
        "ingreso_clp": ingreso,
        "perdida_clp": perdida,
        "utilidad_clp": ingreso - perdida,
    }


def frontera_equidad_utilidad(y: np.ndarray, pd_pred: np.ndarray, grupo: np.ndarray,
                              monto: np.ndarray, grupo_protegido: str,
                              grupo_referencia: str,
                              tasas=np.arange(0.50, 0.96, 0.05)) -> pd.DataFrame:
    """Ratio de impacto adverso y utilidad segun cuanto se apruebe."""
    from src.fairness_metrics import metricas_de_equidad

    filas = []
    for tasa in tasas:
        umbral = umbral_por_tasa_de_aprobacion(pd_pred, float(tasa))
        m = metricas_de_equidad(y, pd_pred, grupo, umbral,
                                grupo_protegido, grupo_referencia)
        eco = resultado_economico(y, decisiones(pd_pred, umbral), monto)
        filas.append({
            "tasa_aprobacion_objetivo": float(tasa),
            "ratio_impacto_adverso": m["ratio_impacto_adverso"],
            "paridad_demografica_pp": m["paridad_demografica_pp"],
            "igualdad_oportunidad_pp": m["igualdad_oportunidad_pp"],
            **eco,
        })
    return pd.DataFrame(filas)
