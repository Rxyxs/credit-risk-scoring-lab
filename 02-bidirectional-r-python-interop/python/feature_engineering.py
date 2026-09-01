"""Features derivadas para el modelo de credit scoring -- razones financieras
estándar de la industria (DTI, carga de cuota sobre ingreso, utilización relativa)
en vez de alimentar los modelos solo con las variables crudas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dti"] = df["deuda_total_clp"] / (df["ingreso_mensual_clp"] * 12)
    df["dti"] = df["dti"].clip(upper=5)

    # Cuota mensual aproximada (amortizacion simple, sin interes compuesto exacto --
    # suficiente como feature de "carga de pago", no para efectos contractuales).
    df["cuota_mensual_aprox_clp"] = df["monto_solicitado_clp"] / df["plazo_meses"] * (
        1 + df["tasa_interes_pct"] / 100
    )
    df["carga_cuota_sobre_ingreso"] = (df["cuota_mensual_aprox_clp"] / df["ingreso_mensual_clp"]).clip(upper=3)

    df["monto_sobre_ingreso_anual"] = df["monto_solicitado_clp"] / (df["ingreso_mensual_clp"] * 12)
    df["antiguedad_laboral_anos"] = df["antiguedad_laboral_meses"] / 12
    df["log_ingreso"] = np.log1p(df["ingreso_mensual_clp"])

    # Codificacion explicita (Indefinido como referencia) en vez de
    # pd.get_dummies(..., drop_first=True): drop_first descarta la categoria que
    # queda primera en orden alfabetico ("Honorarios"), no necesariamente la que
    # uno espera como referencia -- eso dejaba tipo_contrato_Honorarios como una
    # columna constante en cero, perdiendo en silencio una señal de riesgo real
    # (se detecto al inspeccionar el heatmap de correlacion y ver la fila vacia).
    df["tipo_contrato_Plazo Fijo"] = (df["tipo_contrato"] == "Plazo Fijo").astype(int)
    df["tipo_contrato_Honorarios"] = (df["tipo_contrato"] == "Honorarios").astype(int)
    df = df.drop(columns=["tipo_contrato"])

    return df


FEATURE_COLUMNS = [
    "edad",
    "log_ingreso",
    "antiguedad_laboral_anos",
    "dti",
    "carga_cuota_sobre_ingreso",
    "monto_sobre_ingreso_anual",
    "tasa_interes_pct",
    "num_productos_financieros",
    "score_buro_externo",
    "score_buro_faltante",
    "ingreso_faltante",
    "historial_moroso",
    "tipo_contrato_Plazo Fijo",
    "tipo_contrato_Honorarios",
]
