"""Metricas de validacion para modelos de supervivencia con censura.

Las metricas de scoring clasico (AUC, KS sobre una etiqueta binaria) no se
pueden aplicar tal cual cuando parte de la cartera esta censurada: un
credito que salio de la ventana en el mes 8 sin caer en default no es un
"bueno", es un dato incompleto. Aca estan las versiones que si respetan la
censura:

- **Kaplan-Meier** (con varianza de Greenwood) para la curva observada.
- **C-index de Harrell**, que solo cuenta los pares comparables.
- **AUC / Brier a 12 meses** sobre el subconjunto con estado conocido a
  los 12 meses -- explicitamente excluyendo los censurados antes del corte,
  en vez de contarlos como buenos.
- **Calibracion por decil** de PD acumulada predicha contra la observada
  por Kaplan-Meier dentro de cada decil.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------
# Kaplan-Meier
# ----------------------------------------------------------------------
def kaplan_meier(duration, event, times: np.ndarray | None = None) -> pd.DataFrame:
    """Estimador de Kaplan-Meier con error estandar de Greenwood."""
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    uniq = np.unique(duration[event == 1])

    n_risk, n_event, surv, var_sum = [], [], [], []
    s, acc = 1.0, 0.0
    for t in uniq:
        at_risk = int((duration >= t).sum())
        d = int(((duration == t) & (event == 1)).sum())
        s *= 1.0 - d / at_risk
        if at_risk > d:
            acc += d / (at_risk * (at_risk - d))
        n_risk.append(at_risk)
        n_event.append(d)
        surv.append(s)
        var_sum.append(acc)

    km = pd.DataFrame({
        "tiempo": uniq,
        "n_en_riesgo": n_risk,
        "n_eventos": n_event,
        "survival": surv,
        "cum_default": 1.0 - np.array(surv),
        "se_greenwood": np.array(surv) * np.sqrt(var_sum),
    })
    if times is None:
        return km

    times = np.atleast_1d(np.asarray(times, dtype=float))
    idx = np.searchsorted(km["tiempo"].to_numpy(), times, side="right") - 1
    s_at = np.where(idx >= 0, km["survival"].to_numpy()[np.clip(idx, 0, None)], 1.0)
    return pd.DataFrame({"tiempo": times, "survival": s_at, "cum_default": 1.0 - s_at})


def km_cum_default_at(duration, event, t: float) -> float:
    """PD acumulada observada (KM) en el mes `t`."""
    return float(kaplan_meier(duration, event, times=np.array([t]))["cum_default"].iloc[0])


# ----------------------------------------------------------------------
# Discriminacion
# ----------------------------------------------------------------------
def concordance_index(duration, event, risk_score, chunk: int = 512) -> dict:
    """C-index de Harrell.

    Un par (i, j) es comparable si el que tiene el tiempo menor tuvo el
    evento. Es concordante si ese mismo tiene el score de riesgo mayor.
    Los empates de score cuentan medio par, que es la convencion estandar.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)
    n = duration.size

    concordantes = discordantes = empates = 0.0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        di = duration[start:stop, None]
        ei = event[start:stop, None]
        ri = risk[start:stop, None]

        comparable = (ei == 1) & (di < duration[None, :])
        if not comparable.any():
            continue
        dr = ri - risk[None, :]
        concordantes += float(np.sum(comparable & (dr > 0)))
        discordantes += float(np.sum(comparable & (dr < 0)))
        empates += float(np.sum(comparable & (dr == 0)))

    total = concordantes + discordantes + empates
    if total == 0:
        raise ValueError("no hay pares comparables")
    return {
        "c_index": (concordantes + 0.5 * empates) / total,
        "pares_comparables": int(total),
        "concordantes": int(concordantes),
        "discordantes": int(discordantes),
        "empates": int(empates),
    }


def status_at_horizon(duration, event, horizonte: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Etiqueta binaria a `horizonte` meses, con mascara de casos usables.

    - default en o antes del corte -> 1
    - sigue observado despues del corte -> 0
    - censurado antes del corte      -> excluido (estado desconocido)
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    y = ((event == 1) & (duration <= horizonte)).astype(int)
    usable = y.astype(bool) | (duration >= horizonte)
    return y, usable


def auc_at_horizon(duration, event, pd_pred, horizonte: int = 12) -> dict:
    y, usable = status_at_horizon(duration, event, horizonte)
    pd_pred = np.asarray(pd_pred, dtype=float)
    auc = float(roc_auc_score(y[usable], pd_pred[usable]))
    brier = float(np.mean((pd_pred[usable] - y[usable]) ** 2))
    return {
        "horizonte_meses": int(horizonte),
        "auc": auc,
        "gini": 2 * auc - 1,
        "brier": brier,
        "n_usables": int(usable.sum()),
        "n_excluidos_por_censura": int((~usable).sum()),
        "tasa_default_observada": float(y[usable].mean()),
    }


def ks_statistic(duration, event, pd_pred, horizonte: int = 12) -> float:
    """KS clasico sobre el estado conocido al horizonte."""
    y, usable = status_at_horizon(duration, event, horizonte)
    p = np.asarray(pd_pred, dtype=float)[usable]
    yy = y[usable]
    order = np.argsort(p)
    yy = yy[order]
    cum_bad = np.cumsum(yy) / max(yy.sum(), 1)
    cum_good = np.cumsum(1 - yy) / max((1 - yy).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


# ----------------------------------------------------------------------
# Calibracion
# ----------------------------------------------------------------------
def calibration_by_decile(duration, event, pd_pred, horizonte: int = 12,
                          n_bins: int = 10) -> pd.DataFrame:
    """Compara PD predicha vs PD observada por Kaplan-Meier, por decil."""
    df = pd.DataFrame({
        "duracion": np.asarray(duration, dtype=float),
        "evento": np.asarray(event, dtype=int),
        "pd_pred": np.asarray(pd_pred, dtype=float),
    })
    df["decil"] = pd.qcut(df["pd_pred"], n_bins, labels=False, duplicates="drop")

    filas = []
    for decil, g in df.groupby("decil"):
        filas.append({
            "decil": int(decil) + 1,
            "n": len(g),
            "pd_predicha": float(g["pd_pred"].mean()),
            "pd_observada_km": km_cum_default_at(g["duracion"], g["evento"], horizonte),
        })
    out = pd.DataFrame(filas)
    out["error_abs"] = (out["pd_predicha"] - out["pd_observada_km"]).abs()
    return out


def calibration_summary(cal: pd.DataFrame) -> dict:
    return {
        "error_absoluto_medio": float(cal["error_abs"].mean()),
        "error_absoluto_max": float(cal["error_abs"].max()),
        "sesgo_medio": float((cal["pd_predicha"] - cal["pd_observada_km"]).mean()),
    }
