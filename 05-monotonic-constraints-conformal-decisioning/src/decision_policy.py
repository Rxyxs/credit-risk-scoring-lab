"""Politica de decision en tres vias: aprobar, revisar, rechazar.

La comparacion central del modulo es entre dos formas de decidir a quien
mandar a revision manual:

- **Banda de score**: la practica habitual. Se define una franja de PD
  ("entre 10% y 25% lo ve un analista") y listo. Simple, y sin ninguna
  garantia sobre la tasa de error de lo que se automatiza.
- **Conjuntos conformes**: se automatiza solo cuando el conjunto de
  prediccion tiene una sola etiqueta, o sea cuando el metodo puede
  descartar la otra al nivel alpha elegido.

Para que la comparacion sea justa, la banda de score se calibra **al mismo
volumen de revision** que produjo el metodo conforme. Comparar contra una
banda arbitraria seria elegir el rival.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LGD = 0.45
MARGEN = 0.07
COSTO_REVISION_CLP = 12_000     # costo operativo de una revision manual


def decisiones_conformes(conjuntos: np.ndarray) -> np.ndarray:
    """Mapea conjuntos de prediccion a decisiones operativas."""
    solo_bueno = conjuntos[:, 0] & ~conjuntos[:, 1]
    solo_malo = ~conjuntos[:, 0] & conjuntos[:, 1]
    salida = np.full(len(conjuntos), "revisar", dtype=object)
    salida[solo_bueno] = "aprobar"
    salida[solo_malo] = "rechazar"
    return salida


def decisiones_por_banda(pd_pred: np.ndarray, frac_revision: float,
                         corte_aprobacion: float) -> np.ndarray:
    """Banda de revision centrada en el corte, con el volumen pedido.

    Se mandan a revision los `frac_revision` casos cuya PD esta mas cerca
    del corte de aprobacion -- la version mas favorable de la practica
    habitual, porque es justo donde la decision es mas dudosa.
    """
    pd_pred = np.asarray(pd_pred, dtype=float)
    n_revision = int(round(frac_revision * len(pd_pred)))
    salida = np.where(pd_pred <= corte_aprobacion, "aprobar", "rechazar").astype(object)
    if n_revision > 0:
        distancia = np.abs(pd_pred - corte_aprobacion)
        revisar = np.argsort(distancia)[:n_revision]
        salida[revisar] = "revisar"
    return salida


def evaluar_politica(decisiones: np.ndarray, y: np.ndarray, monto: np.ndarray,
                     nombre: str) -> dict:
    """Metricas operativas y economicas de una politica de decision."""
    decisiones = np.asarray(decisiones, dtype=object)
    y = np.asarray(y, dtype=int)
    monto = np.asarray(monto, dtype=float)

    aprobados = decisiones == "aprobar"
    rechazados = decisiones == "rechazar"
    revisados = decisiones == "revisar"
    automaticas = aprobados | rechazados

    malos_aprobados = int(y[aprobados].sum())
    buenos_rechazados = int((1 - y[rechazados]).sum())
    n_auto = int(automaticas.sum())

    ingreso = float((monto[aprobados & (y == 0)] * MARGEN).sum())
    perdida = float((monto[aprobados & (y == 1)] * LGD).sum())
    costo_revision = float(revisados.sum() * COSTO_REVISION_CLP)

    return {
        "politica": nombre,
        "pct_aprobado": float(aprobados.mean()),
        "pct_rechazado": float(rechazados.mean()),
        "pct_revision_manual": float(revisados.mean()),
        "tasa_mala_entre_aprobados": float(y[aprobados].mean()) if aprobados.any() else 0.0,
        "tasa_error_decisiones_automaticas": (
            float((malos_aprobados + buenos_rechazados) / n_auto) if n_auto else 0.0
        ),
        "malos_aprobados": malos_aprobados,
        "buenos_rechazados": buenos_rechazados,
        "utilidad_clp": ingreso - perdida - costo_revision,
        "ingreso_clp": ingreso,
        "perdida_clp": perdida,
        "costo_revision_clp": costo_revision,
    }


def comparar_a_igual_volumen(cp, X: np.ndarray, y: np.ndarray, monto: np.ndarray,
                             pd_pred: np.ndarray, alpha: float = 0.10,
                             corte_aprobacion: float | None = None) -> pd.DataFrame:
    """Conformal vs banda de score con el mismo volumen de revision manual."""
    conjuntos = cp.prediction_sets(X, alpha)
    dec_conf = decisiones_conformes(conjuntos)
    frac_revision = float((dec_conf == "revisar").mean())

    if corte_aprobacion is None:
        # corte que reproduce la misma tasa de aprobacion automatica
        pct_aprobado = float((dec_conf == "aprobar").mean())
        corte_aprobacion = float(np.quantile(pd_pred, pct_aprobado + frac_revision / 2.0))

    dec_banda = decisiones_por_banda(pd_pred, frac_revision, corte_aprobacion)

    return pd.DataFrame([
        evaluar_politica(dec_conf, y, monto, f"conformal_alpha_{alpha:g}"),
        evaluar_politica(dec_banda, y, monto, "banda_de_score"),
    ])


def barrido_alpha(cp, X: np.ndarray, y: np.ndarray, monto: np.ndarray,
                  alphas: np.ndarray) -> pd.DataFrame:
    """Como cambian volumen de revision y error automatico con alpha."""
    filas = []
    for a in alphas:
        dec = decisiones_conformes(cp.prediction_sets(X, float(a)))
        fila = evaluar_politica(dec, y, monto, f"conformal_alpha_{a:g}")
        fila["alpha"] = float(a)
        filas.append(fila)
    return pd.DataFrame(filas)
