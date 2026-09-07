"""Metricas de trato justo, implementadas desde cero.

Un punto que conviene fijar antes de cualquier numero: **no existe "la"
metrica de equidad**. Existen varias, cada una formaliza una idea distinta
de justicia, y son matematicamente incompatibles entre si cuando las tasas
base difieren entre grupos (Kleinberg, Chouldechova). Elegir cual se
reporta es una decision de politica, no de estadistica -- asi que aca se
calculan todas y se muestran juntas, en vez de elegir la que deje mejor
parada a la decision que ya se tomo.

Con "resultado favorable" = credito aprobado y "cliente bueno" = no cayo en
default:

- **Tasa de seleccion**: cuanto se aprueba en cada grupo.
- **Ratio de impacto adverso (regla de los 4/5)**: tasa del grupo protegido
  sobre la del grupo de referencia. Bajo 0.80 es el umbral clasico de
  alerta regulatoria en credito y empleo.
- **Paridad demografica**: diferencia absoluta de tasas de aprobacion.
  Ignora si los grupos tienen riesgo distinto, y por eso es la mas
  exigente y la mas discutida.
- **Igualdad de oportunidad**: diferencia de tasa de aprobacion *entre los
  clientes que efectivamente pagaron*. Es la que pregunta si el modelo
  reconoce igual de bien a un buen cliente en cada grupo.
- **Odds igualados**: agrega la diferencia en la tasa de aprobacion de los
  que si cayeron en default.
- **Calibracion por grupo**: si el modelo dice 10% de PD, .en los dos grupos
  ocurre un 10%?

Todas se reportan con intervalos de confianza por bootstrap: una brecha de
2 puntos sin barra de error no es un hallazgo, es un numero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

UMBRAL_REGLA_CUATRO_QUINTOS = 0.80


def _validar(y, pd_pred, grupo):
    y = np.asarray(y, dtype=int)
    pd_pred = np.asarray(pd_pred, dtype=float)
    grupo = np.asarray(grupo)
    if not (y.size == pd_pred.size == grupo.size):
        raise ValueError("y, pd_pred y grupo deben tener el mismo largo")
    if np.unique(grupo).size < 2:
        raise ValueError("se necesitan al menos dos grupos para comparar")
    return y, pd_pred, grupo


def decisiones(pd_pred: np.ndarray, umbral: float) -> np.ndarray:
    """Aprobado = 1 cuando la PD estimada queda bajo el umbral."""
    return (np.asarray(pd_pred, dtype=float) <= umbral).astype(int)


def umbral_por_tasa_de_aprobacion(pd_pred: np.ndarray, tasa: float) -> float:
    """Umbral global que aprueba la fraccion pedida de la cartera."""
    if not 0 < tasa < 1:
        raise ValueError("la tasa de aprobacion debe estar en (0, 1)")
    return float(np.quantile(np.asarray(pd_pred, dtype=float), tasa))


def tabla_por_grupo(y, pd_pred, grupo, umbral: float) -> pd.DataFrame:
    """Aprobacion, riesgo observado y calibracion, grupo por grupo."""
    y, pd_pred, grupo = _validar(y, pd_pred, grupo)
    aprobado = decisiones(pd_pred, umbral)

    filas = []
    for g in np.unique(grupo):
        m = grupo == g
        buenos = m & (y == 0)
        malos = m & (y == 1)
        filas.append({
            "grupo": g,
            "n": int(m.sum()),
            "tasa_seleccion": float(aprobado[m].mean()),
            "tasa_default_real": float(y[m].mean()),
            "pd_media_predicha": float(pd_pred[m].mean()),
            "tasa_aprobacion_buenos": float(aprobado[buenos].mean()) if buenos.any() else np.nan,
            "tasa_aprobacion_malos": float(aprobado[malos].mean()) if malos.any() else np.nan,
            "tasa_mala_entre_aprobados": (
                float(y[m & (aprobado == 1)].mean()) if (m & (aprobado == 1)).any() else np.nan
            ),
            "auc": float(roc_auc_score(y[m], pd_pred[m])) if 0 < y[m].mean() < 1 else np.nan,
        })
    tabla = pd.DataFrame(filas)
    tabla["sesgo_calibracion_pp"] = 100.0 * (
        tabla["pd_media_predicha"] - tabla["tasa_default_real"]
    )
    return tabla


def metricas_de_equidad(y, pd_pred, grupo, umbral: float,
                        grupo_protegido: str, grupo_referencia: str) -> dict:
    """Las cinco metricas, calculadas sobre la misma decision."""
    tabla = tabla_por_grupo(y, pd_pred, grupo, umbral).set_index("grupo")
    if grupo_protegido not in tabla.index or grupo_referencia not in tabla.index:
        raise ValueError("los grupos indicados no estan en los datos")

    p, r = tabla.loc[grupo_protegido], tabla.loc[grupo_referencia]
    ratio = p["tasa_seleccion"] / r["tasa_seleccion"] if r["tasa_seleccion"] > 0 else np.nan

    return {
        "umbral_pd": float(umbral),
        "tasa_seleccion_protegido": float(p["tasa_seleccion"]),
        "tasa_seleccion_referencia": float(r["tasa_seleccion"]),
        "ratio_impacto_adverso": float(ratio),
        "cumple_regla_cuatro_quintos": bool(ratio >= UMBRAL_REGLA_CUATRO_QUINTOS),
        "paridad_demografica_pp": float(
            100 * (p["tasa_seleccion"] - r["tasa_seleccion"])
        ),
        "igualdad_oportunidad_pp": float(
            100 * (p["tasa_aprobacion_buenos"] - r["tasa_aprobacion_buenos"])
        ),
        "odds_igualados_pp": float(max(
            abs(100 * (p["tasa_aprobacion_buenos"] - r["tasa_aprobacion_buenos"])),
            abs(100 * (p["tasa_aprobacion_malos"] - r["tasa_aprobacion_malos"])),
        )),
        "brecha_calibracion_pp": float(
            p["sesgo_calibracion_pp"] - r["sesgo_calibracion_pp"]
        ),
        "brecha_auc": float(p["auc"] - r["auc"]),
        "tasa_mala_aprobados_protegido": float(p["tasa_mala_entre_aprobados"]),
        "tasa_mala_aprobados_referencia": float(r["tasa_mala_entre_aprobados"]),
    }


def bootstrap_metricas(y, pd_pred, grupo, umbral: float, grupo_protegido: str,
                       grupo_referencia: str, n_boot: int = 400,
                       seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """Intervalos de confianza por bootstrap para cada metrica."""
    y, pd_pred, grupo = _validar(y, pd_pred, grupo)
    rng = np.random.default_rng(seed)
    n = y.size

    claves = [
        "ratio_impacto_adverso", "paridad_demografica_pp",
        "igualdad_oportunidad_pp", "odds_igualados_pp", "brecha_calibracion_pp",
    ]
    muestras = {k: [] for k in claves}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            m = metricas_de_equidad(y[idx], pd_pred[idx], grupo[idx], umbral,
                                    grupo_protegido, grupo_referencia)
        except ValueError:
            continue
        for k in claves:
            muestras[k].append(m[k])

    punto = metricas_de_equidad(y, pd_pred, grupo, umbral,
                                grupo_protegido, grupo_referencia)
    filas = []
    for k in claves:
        v = np.array(muestras[k], dtype=float)
        filas.append({
            "metrica": k,
            "estimacion": punto[k],
            "ic_inferior": float(np.quantile(v, alpha / 2)),
            "ic_superior": float(np.quantile(v, 1 - alpha / 2)),
            "n_boot": int(v.size),
        })
    return pd.DataFrame(filas)


def brecha_condicional(df: pd.DataFrame, columna_grupo: str, grupo_protegido: str,
                       grupo_referencia: str, columna_score: str,
                       columnas_control: list[str], n_bins: int = 5) -> pd.DataFrame:
    """.Cuanto de la brecha sobrevive al comparar perfiles equivalentes?

    Estratifica por los factores de riesgo legitimos (en cuantiles) y mide
    la diferencia de score dentro de cada estrato. Si la brecha desaparece
    al controlar, es impacto dispar via factores legitimos; si sobrevive,
    el modelo esta usando algo mas.
    """
    d = df.copy()
    for col in columnas_control:
        d[f"_q_{col}"] = pd.qcut(d[col], n_bins, labels=False, duplicates="drop")
    llaves = [f"_q_{c}" for c in columnas_control]

    filas = []
    for llave, g in d.groupby(llaves, observed=True):
        p = g.loc[g[columna_grupo] == grupo_protegido, columna_score]
        r = g.loc[g[columna_grupo] == grupo_referencia, columna_score]
        if len(p) < 20 or len(r) < 20:
            continue
        filas.append({
            "estrato": llave if isinstance(llave, tuple) else (llave,),
            "n_protegido": len(p),
            "n_referencia": len(r),
            "score_protegido": float(p.mean()),
            "score_referencia": float(r.mean()),
            "brecha": float(p.mean() - r.mean()),
        })
    return pd.DataFrame(filas)


def resumen_brecha_condicional(tabla: pd.DataFrame, brecha_bruta: float) -> dict:
    """Compara la brecha bruta con la brecha promedio dentro de estratos."""
    if tabla.empty:
        return {"brecha_bruta": brecha_bruta, "brecha_condicional": np.nan,
                "pct_explicado_por_factores_legitimos": np.nan}
    peso = tabla["n_protegido"] + tabla["n_referencia"]
    condicional = float(np.average(tabla["brecha"], weights=peso))
    return {
        "brecha_bruta": float(brecha_bruta),
        "brecha_condicional": condicional,
        "pct_explicado_por_factores_legitimos": float(
            100 * (1 - condicional / brecha_bruta)
        ) if brecha_bruta != 0 else np.nan,
        "n_estratos": int(len(tabla)),
    }
