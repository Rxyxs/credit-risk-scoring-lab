"""Ocho formas de tratar el problema de los rechazados.

Ordenadas por lo que suponen sobre *por que* alguien quedo fuera de la
muestra, que es la unica pregunta que importa:

1. `aprobados_solo` — ignorar el problema. Entrenar con los aprobados y
   aplicar el modelo a toda la poblacion. El punto de partida honesto: si
   la seleccion fue sobre observables y el modelo esta bien especificado,
   esto ya es correcto para la relacion condicional. Casi nadie lo dice.
2. `ipw` — reponderar cada aprobado por el inverso de su probabilidad de
   haber sido aprobado, para reconstruir la poblacion original. Corrige la
   *distribucion*; no puede corregir nada que dependa de lo no observado.
3. `parcelling` — la practica clasica de la industria: puntuar a los
   rechazados con el modelo actual, agruparlos por banda, y asignarles una
   tasa de malos igual a la de los aprobados de esa banda multiplicada por
   un factor de castigo. El factor se elige a dedo, y ese es su problema.
4. `augmentacion_em` — la version disciplinada de lo anterior: imputar la
   etiqueta de los rechazados con la PD del modelo actual, reajustar con
   esos pesos suaves, e iterar hasta converger. Elimina el factor
   arbitrario, pero hereda el supuesto de que el modelo actual extrapola
   bien fuera de la region donde se entreno.
5. `heckman_sin_instrumento` / `probit_bivariado_sin_instrumento` — los
   metodos de seleccion conjunta, alimentados con el mismo conjunto de
   variables en la ecuacion de seleccion y en la de desenlace. Tecnicamente
   identificados por la curvatura de la normal bivariada; en la practica,
   el fallo que el proyecto documenta.
6. `heckman_2etapas` / `probit_bivariado` — los mismos dos metodos, pero la
   ecuacion de seleccion incluye ademas la variable de exclusion (presion
   comercial de la sucursal): algo que mueve la aprobacion y no mueve el
   riesgo real. Es lo que la teoria pide para identificar `rho` de forma
   confiable, y el contraste contra la version anterior lo demuestra.
7. `oraculo` — entrenar con la etiqueta verdadera de todos, incluidos los
   rechazados. Imposible en la practica; esta para saber cuanto de la
   brecha es recuperable en principio.
"""

from __future__ import annotations

import numpy as np

from src.selection_models import (
    BivariateProbitSelection, fit_probit, heckman_dos_etapas, predict_probit,
)

MAX_ITER_EM = 25
TOL_EM = 1e-5
FACTOR_PARCELLING = 2.0
RECORTE_IPW = 0.02          # recorte de propensiones extremas


def aprobados_solo(X, y, aprobado) -> dict:
    """Ignorar a los rechazados: el baseline contra el que se mide todo."""
    sel = np.asarray(aprobado) == 1
    coef = fit_probit(np.asarray(X)[sel], np.asarray(y)[sel])
    return {"coef": coef, "detalle": {"n_entrenamiento": int(sel.sum())}}


def oraculo(X, y_real) -> dict:
    """Cota superior: la etiqueta verdadera de toda la poblacion."""
    return {"coef": fit_probit(X, y_real), "detalle": {"n_entrenamiento": int(len(X))}}


def ipw(X, X_seleccion, y, aprobado) -> dict:
    """Reponderacion inversa a la propension de ser aprobado.

    El modelo de propension usa `X_seleccion` (que puede incluir el
    instrumento): no necesita interpretacion causal, solo predecir bien
    quien fue aprobado, y mas variables ahi solo ayudan.
    """
    X = np.asarray(X, dtype=float)
    sel = np.asarray(aprobado) == 1
    coef_sel = fit_probit(X_seleccion, aprobado)
    propension = np.clip(predict_probit(coef_sel, X_seleccion), RECORTE_IPW, 1.0)
    pesos = 1.0 / propension[sel]
    coef = fit_probit(X[sel], np.asarray(y)[sel], w=pesos)
    return {
        "coef": coef,
        "detalle": {
            "peso_maximo": float(pesos.max()),
            "peso_medio": float(pesos.mean()),
            "propension_minima": float(propension.min()),
        },
    }


def parcelling(X, y, aprobado, factor: float = FACTOR_PARCELLING,
               n_bandas: int = 10) -> dict:
    """Asigna a cada rechazado la tasa mala de su banda, castigada por `factor`."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = np.asarray(aprobado) == 1

    base = fit_probit(X[sel], y[sel])
    score = predict_probit(base, X)

    cortes = np.quantile(score[sel], np.linspace(0, 1, n_bandas + 1)[1:-1])
    banda = np.searchsorted(cortes, score)

    # tasa mala observada por banda entre los aprobados
    tasa = np.array([
        y[sel & (banda == b)].mean() if (sel & (banda == b)).any() else y[sel].mean()
        for b in range(n_bandas)
    ])
    p_rechazado = np.clip(tasa[banda[~sel]] * factor, 1e-6, 1 - 1e-6)

    # cada rechazado entra dos veces, con peso fraccionario
    X_aug = np.vstack([X[sel], X[~sel], X[~sel]])
    y_aug = np.concatenate([y[sel], np.ones(p_rechazado.size), np.zeros(p_rechazado.size)])
    w_aug = np.concatenate([np.ones(int(sel.sum())), p_rechazado, 1.0 - p_rechazado])

    return {
        "coef": fit_probit(X_aug, y_aug, w=w_aug),
        "detalle": {
            "factor": float(factor),
            "tasa_mala_imputada_media": float(p_rechazado.mean()),
            "tasa_mala_aprobados": float(y[sel].mean()),
        },
    }


def augmentacion_em(X, y, aprobado, max_iter: int = MAX_ITER_EM,
                    tol: float = TOL_EM) -> dict:
    """EM: imputar la etiqueta de los rechazados y reajustar hasta converger."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = np.asarray(aprobado) == 1

    coef = fit_probit(X[sel], y[sel])
    historia = []
    for it in range(max_iter):
        p_rechazado = np.clip(predict_probit(coef, X[~sel]), 1e-6, 1 - 1e-6)

        X_aug = np.vstack([X[sel], X[~sel], X[~sel]])
        y_aug = np.concatenate([y[sel], np.ones(p_rechazado.size),
                                np.zeros(p_rechazado.size)])
        w_aug = np.concatenate([np.ones(int(sel.sum())), p_rechazado,
                                1.0 - p_rechazado])

        nuevo = fit_probit(X_aug, y_aug, w=w_aug)
        cambio = float(np.max(np.abs(nuevo - coef)))
        historia.append({"iteracion": it + 1, "cambio_maximo": cambio,
                         "pd_media_rechazados": float(p_rechazado.mean())})
        coef = nuevo
        if cambio < tol:
            break

    return {"coef": coef, "detalle": {"iteraciones": len(historia),
                                      "historia": historia}}


def heckman(X, X_seleccion, y, aprobado) -> dict:
    """Correccion en dos etapas con el inverse Mills ratio.

    `X_seleccion` decide si el metodo tiene variable de exclusion o no: si
    es igual a `X`, esta en el regimen "sin instrumento".
    """
    res = heckman_dos_etapas(X, X_seleccion, y, aprobado)
    return {
        "coef": res["coef_outcome"],
        "detalle": {"coef_imr": res["coef_imr"]},
    }


def probit_bivariado(X, X_seleccion, y, aprobado) -> dict:
    """MLE conjunta de seleccion y desenlace, con su correlacion."""
    modelo = BivariateProbitSelection().fit(X, X_seleccion, y, aprobado)
    return {
        "coef": modelo.coef_outcome_,
        "detalle": {
            "rho_estimado": modelo.rho_,
            "convergio": modelo.converged_,
            "loglik": modelo.loglik_,
            "iteraciones": modelo.n_iter_,
        },
    }


# Cada entrada recibe (X, X_seleccion, y, aprobado, y_real) para uniformar
# la interfaz; los metodos que no necesitan una parte simplemente la ignoran.
METODOS = {
    "aprobados_solo": lambda X, Xs, y, ap, yr: aprobados_solo(X, y, ap),
    "ipw": lambda X, Xs, y, ap, yr: ipw(X, Xs, y, ap),
    "parcelling": lambda X, Xs, y, ap, yr: parcelling(X, y, ap),
    "augmentacion_em": lambda X, Xs, y, ap, yr: augmentacion_em(X, y, ap),
    "heckman_sin_instrumento": lambda X, Xs, y, ap, yr: heckman(X, X, y, ap),
    "probit_bivariado_sin_instrumento": lambda X, Xs, y, ap, yr: probit_bivariado(X, X, y, ap),
    "heckman_2etapas": lambda X, Xs, y, ap, yr: heckman(X, Xs, y, ap),
    "probit_bivariado": lambda X, Xs, y, ap, yr: probit_bivariado(X, Xs, y, ap),
    "oraculo": lambda X, Xs, y, ap, yr: oraculo(X, yr),
}
