"""DP-SGD: regresion logistica entrenada con privacidad diferencial.

Tres cambios sobre un SGD normal, y cada uno esta ahi por una razon
precisa:

1. **Gradiente por ejemplo, no por lote.** Para acotar cuanto puede influir
   una persona, hay que poder mirar su contribucion por separado.
2. **Recorte de norma (`clipping`)**: cada gradiente individual se escala
   para que su norma no supere `C`. Eso fija la *sensibilidad* del paso: sin
   una cota, un solo cliente atipico podria mover el modelo tanto como se
   quiera, y no habria nada que proteger con ruido.
3. **Ruido gaussiano sobre la suma**: se agrega N(0, sigma^2 C^2) a la suma
   de gradientes recortados antes de promediar. La combinacion recorte +
   ruido es el mecanismo gaussiano, y el contador de `accountant.py`
   traduce (sigma, tasa de muestreo, pasos) en epsilon.

El muestreo es de Poisson -- cada ejemplo entra al lote de forma
independiente con probabilidad `q` -- porque es el supuesto bajo el cual
vale la amplificacion por submuestreo que usa el contador. Usar lotes
barajados de tamano fijo y contabilizar como si fueran Poisson es un error
comun que reporta menos epsilon del que corresponde.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.accountant import epsilon_del_entrenamiento


@dataclass
class HistorialEntrenamiento:
    perdida: list[float] = field(default_factory=list)
    norma_gradiente: list[float] = field(default_factory=list)
    fraccion_recortada: list[float] = field(default_factory=list)
    tamano_lote: list[int] = field(default_factory=list)


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))       # estable en los extremos


class DPLogisticRegression:
    """Regresion logistica con DP-SGD.

    Parameters
    ----------
    clip : cota `C` de la norma L2 del gradiente por ejemplo.
    sigma : multiplicador de ruido (desviacion del ruido = sigma * C).
        `sigma = 0` desactiva el ruido y entrega el modelo NO privado, util
        como linea de base.
    q : tasa de muestreo de Poisson (tamano de lote esperado = q * n).
    pasos : iteraciones de entrenamiento.
    lr : tasa de aprendizaje.
    """

    def __init__(self, clip: float = 1.0, sigma: float = 1.0, q: float = 0.02,
                 pasos: int = 1000, lr: float = 0.5, seed: int = 42):
        if clip <= 0:
            raise ValueError("clip debe ser positivo")
        if sigma < 0:
            raise ValueError("sigma no puede ser negativo")
        if not 0 < q <= 1:
            raise ValueError("q debe estar en (0, 1]")
        self.clip = float(clip)
        self.sigma = float(sigma)
        self.q = float(q)
        self.pasos = int(pasos)
        self.lr = float(lr)
        self.seed = int(seed)

    # ------------------------------------------------------------------
    @staticmethod
    def _con_intercepto(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])

    def gradientes_por_ejemplo(self, X: np.ndarray, y: np.ndarray,
                               w: np.ndarray) -> np.ndarray:
        """Matriz (n, p) con el gradiente de cada observacion."""
        p = sigmoide(X @ w)
        return (p - y)[:, None] * X

    def recortar(self, grads: np.ndarray) -> tuple[np.ndarray, float]:
        """Escala cada gradiente para que su norma no supere `clip`."""
        normas = np.linalg.norm(grads, axis=1)
        factor = np.minimum(1.0, self.clip / np.maximum(normas, 1e-12))
        return grads * factor[:, None], float(np.mean(normas > self.clip))

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "DPLogisticRegression":
        Xa = self._con_intercepto(X)
        y = np.asarray(y, dtype=float)
        if Xa.shape[0] != y.size:
            raise ValueError("X e y deben tener el mismo largo")

        rng = np.random.default_rng(self.seed)
        n, p = Xa.shape
        w = np.zeros(p)
        historial = HistorialEntrenamiento()
        lote_esperado = max(self.q * n, 1.0)

        for _ in range(self.pasos):
            # Muestreo de Poisson: cada ejemplo entra de forma independiente.
            en_lote = rng.random(n) < self.q
            idx = np.where(en_lote)[0]
            if idx.size == 0:
                historial.tamano_lote.append(0)
                continue

            grads = self.gradientes_por_ejemplo(Xa[idx], y[idx], w)
            recortados, frac = self.recortar(grads)
            suma = recortados.sum(axis=0)

            if self.sigma > 0:
                suma = suma + rng.normal(0.0, self.sigma * self.clip, size=p)

            grad = suma / lote_esperado
            w = w - self.lr * grad

            historial.perdida.append(self._perdida(Xa, y, w))
            historial.norma_gradiente.append(float(np.linalg.norm(grad)))
            historial.fraccion_recortada.append(frac)
            historial.tamano_lote.append(int(idx.size))

        self.coef_ = w
        self.historial_ = historial
        self.n_train_ = n
        return self

    @staticmethod
    def _perdida(X, y, w) -> float:
        p = np.clip(sigmoide(X @ w), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        p = sigmoide(self._con_intercepto(X) @ self.coef_)
        return np.column_stack([1 - p, p])

    def perdida_por_ejemplo(self, X, y) -> np.ndarray:
        """Log-loss individual: el insumo del ataque de membresia."""
        p = np.clip(self.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        y = np.asarray(y, dtype=float)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def presupuesto_privacidad(self, delta: float = 1e-5) -> dict:
        """Epsilon gastado por esta corrida (infinito si no hubo ruido)."""
        if self.sigma == 0:
            return {"epsilon": float("inf"), "delta": float(delta),
                    "sigma": 0.0, "q": self.q, "pasos": self.pasos,
                    "orden_optimo": None}
        return epsilon_del_entrenamiento(self.q, self.sigma, self.pasos, delta)
