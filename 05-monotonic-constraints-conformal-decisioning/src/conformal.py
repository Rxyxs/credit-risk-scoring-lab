"""Prediccion conforme dividida (split conformal), variante Mondrian.

Que agrega sobre una PD calibrada: una **garantia de cobertura sin
supuestos distribucionales**. Fijado un nivel alpha, el conjunto de
prediccion contiene la etiqueta verdadera al menos el (1 - alpha) de las
veces, y eso vale para cualquier modelo subyacente, sin asumir normalidad,
correcta especificacion ni nada parecido. El unico supuesto es
intercambiabilidad entre calibracion y aplicacion -- supuesto real, y el
modulo `shift_stress.py` muestra exactamente que pasa cuando se rompe.

Mecanica:

1. Score de no conformidad de un caso con etiqueta y: `s = 1 - p_hat(y|x)`
   (mientras menos probabilidad le da el modelo a la etiqueta verdadera,
   mas "raro" es el caso).
2. Sobre el conjunto de calibracion se guardan esos scores.
3. Para un caso nuevo y cada etiqueta candidata y, el p-value conforme es
   la fraccion de scores de calibracion que son al menos tan raros:
   `p_y = (#{s_cal >= s(x,y)} + 1) / (n_cal + 1)`.
4. El conjunto de prediccion al nivel alpha es `{y : p_y > alpha}`.

**Mondrian** (condicional por clase) significa que el paso 3 se hace
contra los scores de calibracion *de esa misma clase*. Con una tasa de
default de 18%, una calibracion marginal cumple la cobertura global
apoyandose casi entera en los buenos y deja a los malos por debajo; la
version condicional garantiza cobertura clase a clase, que es lo que
importa cuando el error caro es el de la clase minoritaria.

Cuatro conjuntos posibles, con lectura de negocio directa:

- `{0}`    -> el modelo descarta el default: aprobacion automatica.
- `{1}`    -> descarta el pago: rechazo automatico.
- `{0, 1}` -> no puede descartar ninguno: revision manual.
- `{}`     -> el caso es raro para ambas clases: revision manual.
"""

from __future__ import annotations

import numpy as np


class MondrianConformalClassifier:
    """Conformal split por clase para clasificacion binaria.

    Parameters
    ----------
    modelo : clasificador ya entrenado con `predict_proba`.
    mondrian : si es False usa una calibracion marginal (un solo conjunto
        de scores para las dos clases). Se deja como opcion justamente para
        poder mostrar que con clases desbalanceadas la version marginal
        cumple la cobertura global a costa de la clase minoritaria.
    """

    def __init__(self, modelo, mondrian: bool = True):
        self.modelo = modelo
        self.mondrian = bool(mondrian)
        self.scores_ = None

    # ------------------------------------------------------------------
    def _no_conformidad(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        proba = self.modelo.predict_proba(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=int)
        return 1.0 - proba[np.arange(len(y)), y]

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "MondrianConformalClassifier":
        y_cal = np.asarray(y_cal, dtype=int)
        if set(np.unique(y_cal)) != {0, 1}:
            raise ValueError("la calibracion necesita casos de ambas clases")
        s = self._no_conformidad(X_cal, y_cal)
        if self.mondrian:
            self.scores_ = {c: np.sort(s[y_cal == c]) for c in (0, 1)}
        else:
            todos = np.sort(s)
            self.scores_ = {0: todos, 1: todos}
        self.n_cal_ = {c: int(self.scores_[c].size) for c in (0, 1)}
        return self

    # ------------------------------------------------------------------
    def p_values(self, X: np.ndarray) -> np.ndarray:
        """p-values conformes por clase: matriz (n, 2)."""
        if self.scores_ is None:
            raise RuntimeError("hay que llamar a calibrate() antes de predecir")
        proba = self.modelo.predict_proba(np.asarray(X, dtype=float))
        p = np.empty_like(proba)
        for c in (0, 1):
            s_nuevo = 1.0 - proba[:, c]
            cal = self.scores_[c]
            # cuantos scores de calibracion son >= al del caso nuevo
            mayores = cal.size - np.searchsorted(cal, s_nuevo, side="left")
            p[:, c] = (mayores + 1.0) / (cal.size + 1.0)
        return p

    def prediction_sets(self, X: np.ndarray, alpha: float = 0.10) -> np.ndarray:
        """Matriz booleana (n, 2): que etiquetas quedan en el conjunto."""
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha debe estar en (0, 1)")
        return self.p_values(X) > alpha

    # ------------------------------------------------------------------
    @staticmethod
    def clasificar_conjuntos(conjuntos: np.ndarray) -> np.ndarray:
        """Traduce los conjuntos a etiquetas legibles."""
        n0, n1 = conjuntos[:, 0], conjuntos[:, 1]
        salida = np.empty(len(conjuntos), dtype=object)
        salida[n0 & ~n1] = "solo_bueno"
        salida[~n0 & n1] = "solo_malo"
        salida[n0 & n1] = "ambos"
        salida[~n0 & ~n1] = "vacio"
        return salida

    def evaluar_cobertura(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.10) -> dict:
        """Cobertura empirica global y por clase, mas tamano de los conjuntos."""
        y = np.asarray(y, dtype=int)
        conjuntos = self.prediction_sets(X, alpha)
        cubre = conjuntos[np.arange(len(y)), y]
        tipos = self.clasificar_conjuntos(conjuntos)
        return {
            "alpha": float(alpha),
            "cobertura_objetivo": 1.0 - alpha,
            "cobertura_global": float(cubre.mean()),
            "cobertura_clase_0": float(cubre[y == 0].mean()),
            "cobertura_clase_1": float(cubre[y == 1].mean()),
            "tamano_medio_conjunto": float(conjuntos.sum(axis=1).mean()),
            "pct_singleton": float((conjuntos.sum(axis=1) == 1).mean()),
            "pct_ambos": float((tipos == "ambos").mean()),
            "pct_vacio": float((tipos == "vacio").mean()),
        }


def cobertura_por_alpha(cp: MondrianConformalClassifier, X: np.ndarray, y: np.ndarray,
                        alphas: np.ndarray) -> "list[dict]":
    """Barrido de alpha para contrastar cobertura empirica vs objetivo."""
    return [cp.evaluar_cobertura(X, y, float(a)) for a in alphas]
