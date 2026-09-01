"""Descarga indicadores macroeconómicos REALES de Chile desde la API pública del
Banco Mundial (sin llave, sin autenticación) para usarlos como variables explicativas
en la calibración empírica de LGD (r/lgd_calibration.R). A diferencia del resto del
pipeline (datos sintéticos con estructura causal inyectada), estos son series
históricas reales -- el ciclo macroeconómico que después se usa para "estresar" LGD
bajo Basilea III/IFRS9 es el ciclo chileno de verdad (incl. la crisis 2020 y el
proceso de ajuste 2022-2023 post-estallido social), no un supuesto.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

# Indicadores del World Bank Open Data API, todos anuales, país Chile (CHL).
# https://api.worldbank.org/v2/country/CHL/indicator/<code>?format=json
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "pib_crecimiento_pct",       # GDP growth (annual %)
    "SL.UEM.TOTL.ZS": "desempleo_pct",                 # Unemployment, total (% of labor force)
    "FP.CPI.TOTL.ZG": "inflacion_pct",                  # Inflation, consumer prices (annual %)
    "FR.INR.DPST": "tasa_interes_deposito_pct",         # Deposit interest rate (%)
}

API_URL = "https://api.worldbank.org/v2/country/CHL/indicator/{code}?format=json&per_page=100"


def fetch_indicator(code: str, timeout: float = 20.0) -> pd.DataFrame:
    url = API_URL.format(code=code)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.load(resp)

    if len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"World Bank API no devolvio datos para el indicador {code}")

    rows = [
        {"anio": int(row["date"]), "valor": row["value"]}
        for row in payload[1]
        if row["value"] is not None
    ]
    return pd.DataFrame(rows)


def fetch_chile_macro_indicators() -> pd.DataFrame:
    """Descarga los 4 indicadores y los une en un panel anual anio x indicador.
    Recorta al rango donde los 4 series tienen datos simultaneamente (evita huecos
    que romperian el join con el panel de prestamos sinteticos)."""
    merged: pd.DataFrame | None = None
    for code, col_name in INDICATORS.items():
        serie = fetch_indicator(code).rename(columns={"valor": col_name})
        merged = serie if merged is None else merged.merge(serie, on="anio", how="inner")

    merged = merged.dropna().sort_values("anio").reset_index(drop=True)
    return merged


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    macro = fetch_chile_macro_indicators()
    out_path = DATA_DIR / "chile_macro_indicators.csv"
    macro.to_csv(out_path, index=False)
    print(f"Indicadores macro de Chile (Banco Mundial, real): {len(macro)} anios -> {out_path}")
    print(f"Rango: {macro['anio'].min()}-{macro['anio'].max()}")
    print(macro.tail(10).to_string(index=False))
