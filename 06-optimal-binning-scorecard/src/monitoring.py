"""Monitoreo de estabilidad: PSI sobre el puntaje y CSI por variable.

Un scorecard se degrada por dos vias distintas y conviene no confundirlas:

- **PSI (Population Stability Index)** sobre el puntaje: la distribucion de
  score de la poblacion nueva se corrio respecto de la de desarrollo. Dice
  que el mix de clientes cambio, no necesariamente que el modelo este mal.
- **CSI (Characteristic Stability Index)**: el mismo calculo pero variable
  por variable, sobre los bins del scorecard. Dice *cual* variable se
  movio, que es lo que uno necesita para decidir si recalibrar, re-binear o
  no hacer nada.

    PSI = sum_bins (pct_nuevo - pct_base) * ln(pct_nuevo / pct_base)

Umbrales de la practica: bajo 0.10 estable, entre 0.10 y 0.25 alerta,
sobre 0.25 cambio material. No son teoremas -- son convencion -- y por eso
el modulo reporta el valor, no solo el semaforo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.binning import ResultadoBinning, indice_bin

UMBRAL_ALERTA = 0.10
UMBRAL_CRITICO = 0.25
SUAVIZADO_PSI = 1e-6


def semaforo(indice: float) -> str:
    if indice < UMBRAL_ALERTA:
        return "estable"
    if indice < UMBRAL_CRITICO:
        return "alerta"
    return "critico"


def psi_desde_conteos(base: np.ndarray, nuevo: np.ndarray) -> float:
    """PSI entre dos vectores de conteos por bin."""
    base = np.asarray(base, dtype=float)
    nuevo = np.asarray(nuevo, dtype=float)
    if base.shape != nuevo.shape:
        raise ValueError("base y nuevo deben tener el mismo numero de bins")
    if base.sum() == 0 or nuevo.sum() == 0:
        raise ValueError("no se puede calcular PSI con una poblacion vacia")

    p_base = np.clip(base / base.sum(), SUAVIZADO_PSI, None)
    p_nuevo = np.clip(nuevo / nuevo.sum(), SUAVIZADO_PSI, None)
    return float(np.sum((p_nuevo - p_base) * np.log(p_nuevo / p_base)))


def psi_score(score_base: np.ndarray, score_nuevo: np.ndarray,
              n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """PSI del puntaje, con los cortes fijados por los deciles de la base."""
    cortes = np.quantile(score_base, np.linspace(0, 1, n_bins + 1)[1:-1])
    idx_base = np.searchsorted(cortes, score_base, side="right")
    idx_nuevo = np.searchsorted(cortes, score_nuevo, side="right")
    conteo_base = np.bincount(idx_base, minlength=n_bins).astype(float)
    conteo_nuevo = np.bincount(idx_nuevo, minlength=n_bins).astype(float)

    p_base = conteo_base / conteo_base.sum()
    p_nuevo = conteo_nuevo / conteo_nuevo.sum()
    detalle = pd.DataFrame({
        "bin": np.arange(n_bins),
        "pct_base": p_base,
        "pct_nuevo": p_nuevo,
        "aporte_psi": (p_nuevo - np.clip(p_base, SUAVIZADO_PSI, None))
        * np.log(np.clip(p_nuevo, SUAVIZADO_PSI, None) / np.clip(p_base, SUAVIZADO_PSI, None)),
    })
    return psi_desde_conteos(conteo_base, conteo_nuevo), detalle


def csi_variable(binning: ResultadoBinning, x_base, x_nuevo) -> float:
    """CSI de una variable, usando los bins del scorecard ya ajustado."""
    n_bins = len(binning.tabla)
    base = np.bincount(indice_bin(binning, x_base), minlength=n_bins).astype(float)
    nuevo = np.bincount(indice_bin(binning, x_nuevo), minlength=n_bins).astype(float)
    return psi_desde_conteos(base, nuevo)


def monitoreo_por_vintage(scorecard, df_base: pd.DataFrame, df: pd.DataFrame,
                          columna_vintage: str = "vintage_idx") -> pd.DataFrame:
    """PSI del score y CSI de cada variable, vintage a vintage."""
    score_base = scorecard.score(df_base)
    filas = []
    for vintage, grupo in df.groupby(columna_vintage):
        psi, _ = psi_score(score_base, scorecard.score(grupo))
        fila = {
            "vintage_idx": int(vintage),
            "n": len(grupo),
            "tasa_mala": float(grupo["default_12m"].mean()),
            "score_medio": float(scorecard.score(grupo).mean()),
            "psi_score": psi,
            "estado_psi": semaforo(psi),
        }
        for var, binning in scorecard.binnings.items():
            fila[f"csi_{var}"] = csi_variable(
                binning, df_base[var].to_numpy(), grupo[var].to_numpy()
            )
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("vintage_idx").reset_index(drop=True)


def resumen_monitoreo(tabla: pd.DataFrame) -> dict:
    """Primera alerta, primer critico y la variable que mas se movio."""
    cols_csi = [c for c in tabla.columns if c.startswith("csi_")]
    csi_max = tabla[cols_csi].max()
    alerta = tabla.loc[tabla["psi_score"] >= UMBRAL_ALERTA, "vintage_idx"]
    critico = tabla.loc[tabla["psi_score"] >= UMBRAL_CRITICO, "vintage_idx"]
    return {
        "psi_max": float(tabla["psi_score"].max()),
        "primer_vintage_en_alerta": int(alerta.min()) if len(alerta) else None,
        "primer_vintage_critico": int(critico.min()) if len(critico) else None,
        "variable_mas_inestable": csi_max.idxmax().replace("csi_", ""),
        "csi_maximo": float(csi_max.max()),
        "vintages_estables": int((tabla["estado_psi"] == "estable").sum()),
        "vintages_en_alerta": int((tabla["estado_psi"] == "alerta").sum()),
        "vintages_criticos": int((tabla["estado_psi"] == "critico").sum()),
    }
