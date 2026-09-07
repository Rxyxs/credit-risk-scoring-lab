"""Ataques de privacidad: la validacion empirica de la garantia formal.

Un epsilon es una cota teorica. Sirve, pero por si solo no muestra nada:
hay que poder responder "y en la practica, .que se puede averiguar sobre
una persona a partir del modelo?". Aca se implementan los dos ataques que
contestan esa pregunta en un caso de credito:

- **Inferencia de membresia**: dado el modelo publicado, .se puede saber si
  una persona estuvo en la base de entrenamiento? El ataque clasico usa la
  perdida por ejemplo: un modelo que memorizo le asigna menor perdida a
  quienes vio. El AUC del ataque mide cuanto se filtra: 0.50 es no saber
  nada, 1.00 es identificacion perfecta. En credito el solo hecho de estar
  en la base ya es informacion sensible -- significa que la persona pidio
  un credito.

- **Exposicion de canarios**: se inyectan registros atipicos con etiqueta
  invertida y se compara la PD que el modelo les asigna contra la que
  asigna a registros identicos en distribucion que **no** estuvieron en el
  entrenamiento. Todo lo que exceda de cero es memorizacion pura, porque
  nada en el patron general justifica esa prediccion.

El segundo ataque es el mas informativo con pocos datos: detecta
memorizacion de casos puntuales incluso cuando el modelo no sobreajusta lo
suficiente como para que el ataque de membresia global se note.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def ataque_de_membresia(modelo, X_miembros, y_miembros,
                        X_no_miembros, y_no_miembros,
                        estratificado: bool = True) -> dict:
    """Ataque por umbral de perdida: menor perdida => probable miembro.

    El ataque se evalua **dentro de cada etiqueta** por defecto. Sin eso,
    cualquier diferencia de tasa de default entre el conjunto de miembros y
    el de no miembros (aca 20,5% contra 21,5%, pura variacion muestral) se
    cuela como si fuera filtracion: con un modelo muy ruidoso la perdida
    depende casi solo de la etiqueta, y el "ataque" termina detectando la
    tasa base en vez de la membresia. Estratificando, el atacante solo puede
    ganar si el modelo efectivamente trata distinto a quienes vio.
    """
    perdida_in = modelo.perdida_por_ejemplo(X_miembros, y_miembros)
    perdida_out = modelo.perdida_por_ejemplo(X_no_miembros, y_no_miembros)

    es_miembro = np.concatenate([np.ones(perdida_in.size), np.zeros(perdida_out.size)])
    score = -np.concatenate([perdida_in, perdida_out])      # menos perdida = mas sospecha
    etiquetas_y = np.concatenate([np.asarray(y_miembros, dtype=int),
                                  np.asarray(y_no_miembros, dtype=int)])

    if estratificado:
        aucs, pesos = [], []
        for valor in np.unique(etiquetas_y):
            m = etiquetas_y == valor
            if np.unique(es_miembro[m]).size < 2:
                continue
            aucs.append(roc_auc_score(es_miembro[m], score[m]))
            pesos.append(m.sum())
        auc = float(np.average(aucs, weights=pesos)) if aucs else 0.5
    else:
        auc = float(roc_auc_score(es_miembro, score))

    # Ventaja: maxima diferencia entre aciertos y falsas alarmas sobre
    # todos los umbrales posibles del atacante, calculada dentro de la
    # misma estratificacion que el AUC.
    def _ventaja(idx) -> float:
        orden = idx[np.argsort(-score[idx])]
        etiquetas = es_miembro[orden]
        tpr = np.cumsum(etiquetas) / max(etiquetas.sum(), 1)
        fpr = np.cumsum(1 - etiquetas) / max((1 - etiquetas).sum(), 1)
        return float(np.max(tpr - fpr))

    if estratificado:
        ventajas, pesos_v = [], []
        for valor in np.unique(etiquetas_y):
            idx = np.where(etiquetas_y == valor)[0]
            if np.unique(es_miembro[idx]).size < 2:
                continue
            ventajas.append(_ventaja(idx))
            pesos_v.append(idx.size)
        ventaja = float(np.average(ventajas, weights=pesos_v)) if ventajas else 0.0
    else:
        ventaja = _ventaja(np.arange(score.size))

    return {
        "auc_ataque": auc,
        "ventaja": ventaja,
        "perdida_media_miembros": float(perdida_in.mean()),
        "perdida_media_no_miembros": float(perdida_out.mean()),
        "brecha_de_perdida": float(perdida_out.mean() - perdida_in.mean()),
        "n_miembros": int(perdida_in.size),
        "n_no_miembros": int(perdida_out.size),
    }


def exposicion_de_canarios(modelo, X_canarios, X_canarios_sombra) -> dict:
    """Memorizacion medible: canarios entrenados vs canarios identicos no vistos.

    `X_canarios_sombra` viene del mismo generador y con el mismo perfil,
    pero nunca entro al entrenamiento. La diferencia de PD entre unos y
    otros no puede explicarse por el patron general: es memorizacion.
    """
    pd_entrenados = modelo.predict_proba(X_canarios)[:, 1]
    pd_sombra = modelo.predict_proba(X_canarios_sombra)[:, 1]
    return {
        "pd_media_canarios_entrenados": float(pd_entrenados.mean()),
        "pd_media_canarios_sombra": float(pd_sombra.mean()),
        "exposicion_pp": float(100 * (pd_entrenados.mean() - pd_sombra.mean())),
        "pd_maxima_canario": float(pd_entrenados.max()),
        "n_canarios": int(pd_entrenados.size),
    }


def riesgo_de_reidentificacion(resultado_membresia: dict,
                               prevalencia: float = 0.5) -> dict:
    """Traduce el AUC del ataque a algo que se pueda explicar en un comite.

    Con la ventaja del atacante se acota cuanto mejora respecto de adivinar:
    la precision alcanzable al operar en el punto de maxima ventaja.
    """
    ventaja = resultado_membresia["ventaja"]
    return {
        "ventaja": float(ventaja),
        "precision_del_atacante": float(
            (prevalencia * (1 + ventaja)) / max(prevalencia * (1 + ventaja)
                                                + (1 - prevalencia) * (1 - ventaja), 1e-12)
        ),
        "lectura": (
            "no hay filtracion detectable" if ventaja < 0.05 else
            "filtracion leve" if ventaja < 0.15 else
            "filtracion material"
        ),
    }
