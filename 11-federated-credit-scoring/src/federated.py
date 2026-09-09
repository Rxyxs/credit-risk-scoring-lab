"""FedAvg (McMahan et al., 2017) para una regresion logistica, escrito
desde cero.

La idea del aprendizaje federado es exactamente la restriccion legal que
enfrenta un consorcio de bancos: los datos de cada participante nunca
salen de su servidor. Lo unico que viaja es el modelo. En cada ronda:

1. El servidor manda los pesos globales actuales `w` a cada banco.
2. Cada banco entrena localmente, arrancando desde `w`, por `E` epocas
   de descenso de gradiente sobre **sus propios datos y nada mas**.
3. Cada banco manda de vuelta sus pesos actualizados (no sus datos).
4. El servidor promedia los pesos de todos los bancos, ponderado por
   cuantos solicitantes aporto cada uno, y esa es la nueva `w`.

Dos identidades exactas anclan la implementacion, y los tests las
verifican bit a bit:

- **Con un solo banco**, promediar no hace nada: FedAvg con `E` epocas
  locales por ronda y `R` rondas tiene que dar exactamente lo mismo que
  correr descenso de gradiente centralizado por `R*E` epocas seguidas
  sobre los datos de ese banco.
- **Con `E=1`** (cada banco da un solo paso de gradiente por ronda antes
  de promediar), FedAvg tiene que coincidir exactamente con descenso de
  gradiente centralizado sobre los datos **agrupados** de todos los
  bancos -- porque el promedio ponderado de un paso de gradiente por
  cliente, ponderado por su tamano, es algebraicamente identico a un
  paso de gradiente calculado directamente sobre el conjunto union. Esta
  es la identidad "FedAvg con E=1 es FedSGD es descenso centralizado" del
  paper original.

`E > 1` es donde el metodo deja de ser una curiosidad algebraica: ahorra
comunicacion (menos rondas por la misma cantidad de computo local), pero
en datos no-IID cada banco se aleja del optimo global antes de que se le
corrija ("client drift"), y esa tension es lo que el proyecto mide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _con_intercepto(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def gradiente_medio(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradiente de la log-verosimilitud logistica, promediado por ejemplo.

    Promediar (no sumar) es lo que hace que la identidad de agregacion de
    FedAvg funcione limpio: el promedio ponderado por tamano de gradientes
    *promedio* de cada cliente es exactamente el gradiente *promedio* del
    conjunto agrupado.
    """
    p = sigmoide(X @ w)
    return X.T @ (p - y) / len(y)


def local_update(w: np.ndarray, X: np.ndarray, y: np.ndarray, lr: float,
                 epochs: int) -> np.ndarray:
    """`epochs` pasos de descenso de gradiente de lote completo, desde `w`."""
    w = w.copy()
    for _ in range(epochs):
        w = w - lr * gradiente_medio(w, X, y)
    return w


def gradient_descent_centralizado(w0: np.ndarray, X: np.ndarray, y: np.ndarray,
                                  lr: float, epochs: int) -> np.ndarray:
    """Descenso de gradiente comun, para comparar contra FedAvg."""
    return local_update(w0, X, y, lr, epochs)


@dataclass
class HistorialFedAvg:
    ronda: list[int] = field(default_factory=list)
    deriva_media_clientes: list[float] = field(default_factory=list)
    cambio_global: list[float] = field(default_factory=list)


class FedAvgLogisticRegression:
    """Regresion logistica entrenada por FedAvg sobre datos de varios
    clientes que nunca se juntan en una sola matriz.

    Parameters
    ----------
    lr : tasa de aprendizaje del descenso de gradiente local.
    local_epochs : epocas de entrenamiento local por ronda (`E` en el
        paper). `E=1` es FedSGD; `E` grande ahorra comunicacion pero
        expone al metodo al client drift en datos no-IID.
    n_rounds : rondas de comunicacion.
    """

    def __init__(self, lr: float = 0.5, local_epochs: int = 1, n_rounds: int = 40,
                seed: int = 42):
        if local_epochs < 1:
            raise ValueError("local_epochs debe ser >= 1")
        if n_rounds < 1:
            raise ValueError("n_rounds debe ser >= 1")
        self.lr = float(lr)
        self.local_epochs = int(local_epochs)
        self.n_rounds = int(n_rounds)
        self.seed = int(seed)

    def fit(self, X_por_cliente: list[np.ndarray], y_por_cliente: list[np.ndarray],
           w0: np.ndarray | None = None) -> "FedAvgLogisticRegression":
        if len(X_por_cliente) != len(y_por_cliente):
            raise ValueError("X_por_cliente e y_por_cliente deben tener el mismo largo")
        if len(X_por_cliente) == 0:
            raise ValueError("se necesita al menos un cliente")

        Xs = [_con_intercepto(X) for X in X_por_cliente]
        ys = [np.asarray(y, dtype=float) for y in y_por_cliente]
        n = np.array([len(y) for y in ys], dtype=float)
        peso = n / n.sum()

        p = Xs[0].shape[1]
        w = np.zeros(p) if w0 is None else np.asarray(w0, dtype=float).copy()

        historia = HistorialFedAvg()
        for r in range(self.n_rounds):
            w_locales = np.array([
                local_update(w, Xs[k], ys[k], self.lr, self.local_epochs)
                for k in range(len(Xs))
            ])
            w_nuevo = (peso[:, None] * w_locales).sum(axis=0)

            historia.ronda.append(r + 1)
            historia.deriva_media_clientes.append(
                float(np.mean(np.linalg.norm(w_locales - w, axis=1)))
            )
            historia.cambio_global.append(float(np.linalg.norm(w_nuevo - w)))
            w = w_nuevo

        self.coef_ = w
        self.historia_ = historia
        self.n_clientes_ = len(Xs)
        return self

    def deltas_primera_ronda(self, X_por_cliente: list[np.ndarray],
                             y_por_cliente: list[np.ndarray],
                             w0: np.ndarray | None = None) -> np.ndarray:
        """El update que cada cliente manda en la ronda 1, antes de agregar.

        No hace falta entrenar el modelo completo para calcularlo: es
        exactamente el primer paso de `fit`, expuesto por separado porque
        es lo que un servidor honesto-pero-curioso observaria de cada
        participante antes de que la agregacion mezcle todo.
        """
        Xs = [_con_intercepto(X) for X in X_por_cliente]
        ys = [np.asarray(y, dtype=float) for y in y_por_cliente]
        p = Xs[0].shape[1]
        w = np.zeros(p) if w0 is None else np.asarray(w0, dtype=float).copy()
        return np.array([
            local_update(w, Xs[k], ys[k], self.lr, self.local_epochs) - w
            for k in range(len(Xs))
        ])

    def predict_proba(self, X) -> np.ndarray:
        p = sigmoide(_con_intercepto(X) @ self.coef_)
        return np.column_stack([1 - p, p])
