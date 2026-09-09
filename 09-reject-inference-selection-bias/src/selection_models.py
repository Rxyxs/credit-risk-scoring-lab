"""Modelos de seleccion: probit, normal bivariada y probit bivariado con
seleccion, implementados desde cero.

Cuando la muestra de aprobados quedo seleccionada por informacion que el
modelo no observa, ninguna reponderacion arregla el problema: el sesgo no
esta en las proporciones sino en la relacion misma. La correccion clasica
consiste en modelar la seleccion y el desenlace **conjuntamente**,
permitiendo que sus errores esten correlacionados.

Dos versiones, y la diferencia entre ellas es exactamente el punto:

- **Heckman en dos etapas**: probit de seleccion, se calcula el inverse
  Mills ratio de cada aprobado, y se agrega como regresor al modelo de
  desenlace. Es rapido y es lo que se aplica en la practica, pero para un
  desenlace binario es una aproximacion: la derivacion original es para
  una variable continua.
- **Probit bivariado con seleccion**: la version correcta para desenlace
  binario. Estima por maxima verosimilitud los dos indices y la
  correlacion `rho` entre sus errores. `rho != 0` es, literalmente, la
  medida de cuanta seleccion sobre no observables hay.

Para la verosimilitud del probit bivariado hace falta la CDF normal
bivariada, que numpy no trae. Se implementa por cuadratura de
Gauss-Legendre sobre la identidad

    Phi_2(a, b, rho) = Phi(a) Phi(b) + integral_0^rho phi_2(a, b, r) dr

que es exacta y vectorizable sobre todas las observaciones a la vez. Los
tests la contrastan contra `scipy.stats.multivariate_normal`, que es una
implementacion independiente.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.stats import norm

N_CUADRATURA = 48


def _phi2(a: np.ndarray, b: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Densidad normal bivariada estandar con correlacion `r`."""
    un_menos = 1.0 - r**2
    return np.exp(-(a**2 - 2 * r * a * b + b**2) / (2 * un_menos)) / (
        2 * np.pi * np.sqrt(un_menos)
    )


def bivariate_normal_cdf(a, b, rho: float, n_nodos: int = N_CUADRATURA) -> np.ndarray:
    """Phi_2(a, b; rho) vectorizada, por cuadratura sobre la correlacion."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if not -1.0 < rho < 1.0:
        raise ValueError("rho debe estar en (-1, 1)")

    base = norm.cdf(a) * norm.cdf(b)
    if abs(rho) < 1e-12:
        return base

    nodos, pesos = np.polynomial.legendre.leggauss(n_nodos)
    # cambio de variable de [-1, 1] a [0, rho]
    r = 0.5 * rho * (nodos + 1.0)
    escala = 0.5 * rho
    integral = np.sum(
        pesos[None, :] * _phi2(a[:, None], b[:, None], r[None, :]), axis=1
    ) * escala
    return np.clip(base + integral, 1e-12, 1.0 - 1e-12)


def bivariate_normal_cdf_partials(a, b, rho: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivadas parciales de Phi_2(a, b; rho), en forma cerrada.

    Sin esto, ajustar el probit bivariado obliga a scipy a diferenciar
    numericamente una funcion de 7+ parametros que ya de por si se evalua
    por cuadratura -- y en la practica eso hace que el optimizador pare
    lejos del maximo (confirmado con un barrido de la verosimilitud en el
    beta/alpha verdaderos: el minimo esta exactamente en el rho verdadero,
    pero L-BFGS-B con gradiente numerico converge a otro punto). Las tres
    derivadas son identidades estandar:

        dPhi2/da   = phi(a) * Phi( (b - rho*a) / sqrt(1 - rho^2) )
        dPhi2/db   = phi(b) * Phi( (a - rho*b) / sqrt(1 - rho^2) )
        dPhi2/drho = phi_2(a, b; rho)              (formula de Plackett)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    raiz = np.sqrt(1.0 - rho**2)
    d_a = norm.pdf(a) * norm.cdf((b - rho * a) / raiz)
    d_b = norm.pdf(b) * norm.cdf((a - rho * b) / raiz)
    d_rho = _phi2(a, b, np.full_like(a, rho))
    return d_a, d_b, d_rho


def inverse_mills_ratio(indice: np.ndarray, seleccionado: bool = True) -> np.ndarray:
    """IMR: phi(z)/Phi(z) para los seleccionados, -phi(z)/(1-Phi(z)) si no."""
    z = np.asarray(indice, dtype=float)
    if seleccionado:
        return norm.pdf(z) / np.clip(norm.cdf(z), 1e-12, None)
    return -norm.pdf(z) / np.clip(1.0 - norm.cdf(z), 1e-12, None)


# ----------------------------------------------------------------------
# Probit
# ----------------------------------------------------------------------
def _con_intercepto(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def fit_probit(X, y, w=None, max_iter: int = 200) -> np.ndarray:
    """Probit por maxima verosimilitud (gradiente analitico).

    Acepta pesos por observacion, que es lo que necesitan los metodos de
    reject inference: reponderacion inversa a la propension, etiquetas
    fraccionarias del parcelling, y los pesos suaves del EM.
    """
    Xa = _con_intercepto(X)
    y = np.asarray(y, dtype=float)
    if Xa.shape[0] != y.size:
        raise ValueError("X e y deben tener el mismo largo")
    w = np.ones(y.size) if w is None else np.asarray(w, dtype=float)
    if w.size != y.size:
        raise ValueError("los pesos deben tener el mismo largo que y")
    if np.any(w < 0):
        raise ValueError("los pesos no pueden ser negativos")

    def neg_ll(beta):
        z = Xa @ beta
        p = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
        return -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))

    def grad(beta):
        z = Xa @ beta
        p = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
        lam = w * norm.pdf(z) * (y - p) / (p * (1 - p))
        return -Xa.T @ lam

    res = optimize.minimize(neg_ll, np.zeros(Xa.shape[1]), jac=grad,
                            method="L-BFGS-B", options={"maxiter": max_iter})
    return res.x


def predict_probit(coef: np.ndarray, X) -> np.ndarray:
    return norm.cdf(_con_intercepto(X) @ np.asarray(coef, dtype=float))


# ----------------------------------------------------------------------
# Probit bivariado con seleccion
# ----------------------------------------------------------------------
class BivariateProbitSelection:
    """Modelo de Heckman para desenlace binario, estimado por MLE conjunta.

    La verosimilitud tiene tres piezas, una por cada cosa que efectivamente
    se observa:

        rechazado                 ->  1 - Phi(z_s)
        aprobado y no cae en mora ->  Phi_2(-z_y,  z_s, -rho)
        aprobado y cae en mora    ->  Phi_2( z_y,  z_s,  rho)

    Notar que de los rechazados solo se usa que fueron rechazados: su
    desenlace no existe en los datos. Esa es toda la informacion que el
    metodo puede extraer de ellos, y es mas de lo que usa un modelo
    entrenado unicamente con aprobados.
    """

    def __init__(self, max_iter: int = 300):
        self.max_iter = int(max_iter)

    def _neg_ll(self, params, Xy, Xs, y, aprobado):
        return self._neg_ll_grad(params, Xy, Xs, y, aprobado)[0]

    def _neg_ll_grad(self, params, Xy, Xs, y, aprobado):
        """Log-verosimilitud negativa y su gradiente analitico.

        El gradiente conjunto es indispensable aca: sin el, scipy tiene que
        diferenciar numericamente sobre 2p+1 parametros una funcion que ya
        se evalua por cuadratura, y en la practica eso deja al optimizador
        lejos del maximo aunque reporte "convergio". Con el gradiente
        cerrado (regla de la cadena sobre las derivadas de Phi_2) el ajuste
        es a la vez mas rapido y mas confiable.
        """
        p_y = Xy.shape[1]
        beta = params[:p_y]
        alpha = params[p_y:-1]
        u = params[-1]
        rho = np.tanh(u)                    # reparametrizacion a (-1, 1)
        drho_du = 1.0 - rho**2               # d(tanh)/du

        z_y = Xy @ beta
        z_s = Xs @ alpha

        ll = np.empty(len(y))
        g_beta = np.zeros(p_y)
        g_alpha = np.zeros(Xs.shape[1])
        g_rho = 0.0

        no_sel = aprobado == 0
        surv = np.clip(1.0 - norm.cdf(z_s[no_sel]), 1e-12, None)
        ll[no_sel] = np.log(surv)
        # d/dz_s[-log(1-Phi(z_s))] = phi(z_s)/(1-Phi(z_s)); aca es +log, signo invertido
        g_alpha += -(norm.pdf(z_s[no_sel]) / surv) @ Xs[no_sel]

        sel_bueno = (aprobado == 1) & (y == 0)
        a_b, b_b = -z_y[sel_bueno], z_s[sel_bueno]
        p_b = np.clip(bivariate_normal_cdf(a_b, b_b, -rho), 1e-12, None)
        ll[sel_bueno] = np.log(p_b)
        da_b, db_b, dr_b = bivariate_normal_cdf_partials(a_b, b_b, -rho)
        g_beta += ((da_b / p_b) * -1.0) @ Xy[sel_bueno]
        g_alpha += (db_b / p_b) @ Xs[sel_bueno]
        g_rho += float(np.sum((dr_b / p_b) * -1.0))

        sel_malo = (aprobado == 1) & (y == 1)
        a_m, b_m = z_y[sel_malo], z_s[sel_malo]
        p_m = np.clip(bivariate_normal_cdf(a_m, b_m, rho), 1e-12, None)
        ll[sel_malo] = np.log(p_m)
        da_m, db_m, dr_m = bivariate_normal_cdf_partials(a_m, b_m, rho)
        g_beta += (da_m / p_m) @ Xy[sel_malo]
        g_alpha += (db_m / p_m) @ Xs[sel_malo]
        g_rho += float(np.sum(dr_m / p_m))

        grad = np.concatenate([g_beta, g_alpha, [g_rho * drho_du]])
        return -float(np.sum(ll)), -grad

    def fit(self, X_outcome, X_selection, y, aprobado) -> "BivariateProbitSelection":
        Xy = _con_intercepto(X_outcome)
        Xs = _con_intercepto(X_selection)
        y = np.asarray(y, dtype=float)
        aprobado = np.asarray(aprobado, dtype=int)
        if not (len(Xy) == len(Xs) == y.size == aprobado.size):
            raise ValueError("todas las entradas deben tener el mismo largo")
        if aprobado.sum() == 0:
            raise ValueError("no hay observaciones aprobadas")

        # Arranque desde las estimaciones por separado, que es donde el
        # optimizador conjunto converge mas rapido y mas estable.
        beta0 = fit_probit(X_outcome[aprobado == 1], y[aprobado == 1])
        alpha0 = fit_probit(X_selection, aprobado)
        inicio = np.concatenate([beta0, alpha0, [0.0]])

        res = optimize.minimize(
            self._neg_ll_grad, inicio, args=(Xy, Xs, y, aprobado),
            jac=True, method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )
        p_y = Xy.shape[1]
        self.coef_outcome_ = res.x[:p_y]
        self.coef_selection_ = res.x[p_y:-1]
        self.rho_ = float(np.tanh(res.x[-1]))
        self.loglik_ = -float(res.fun)
        self.converged_ = bool(res.success)
        self.n_iter_ = int(res.nit)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """PD marginal (no condicional a ser aprobado), que es la que sirve."""
        return norm.cdf(_con_intercepto(X) @ self.coef_outcome_)


def heckman_dos_etapas(X_outcome, X_selection, y, aprobado) -> dict:
    """Correccion de Heckman en dos etapas con el inverse Mills ratio.

    Rapida y estandar en la practica; para un desenlace binario es una
    aproximacion, y el proyecto la compara justamente contra la version
    exacta (probit bivariado) para mostrar cuanto se pierde.
    """
    aprobado = np.asarray(aprobado, dtype=int)
    y = np.asarray(y, dtype=float)

    alpha = fit_probit(X_selection, aprobado)
    z_s = _con_intercepto(X_selection) @ alpha
    imr = inverse_mills_ratio(z_s, seleccionado=True)

    sel = aprobado == 1
    X_aug = np.column_stack([np.asarray(X_outcome, dtype=float)[sel], imr[sel]])
    beta_aug = fit_probit(X_aug, y[sel])

    return {
        "coef_seleccion": alpha,
        "coef_outcome": beta_aug[:-1],       # sin el termino del IMR
        "coef_imr": float(beta_aug[-1]),
        "imr_todos": imr,
    }
