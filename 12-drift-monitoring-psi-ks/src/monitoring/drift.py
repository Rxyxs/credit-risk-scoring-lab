"""Detección de data drift / concept drift entre una línea base (train) y un
conjunto de scoring (OOT o producción), vía dos pruebas independientes:

- **PSI (Population Stability Index)**: compara las proporciones por bucket
  entre `expected` (base) y `actual` (scoring). Es la métrica estándar de
  riesgo de crédito para esto -- barata, interpretable, y con umbrales de
  práctica ya asentados en la industria (ver `PSI_ALERT_THRESHOLD` /
  `PSI_CRITICAL_THRESHOLD` abajo).
- **KS de dos muestras** (`scipy.stats.ks_2samp`): compara las CDFs
  empíricas directamente, sin discretizar en buckets -- una segunda señal,
  sensible a formas de drift que PSI por deciles puede diluir (ej. un cambio
  concentrado en un solo extremo de la distribución).

Ninguna de las dos requiere un modelo ajustado: a diferencia de
`06-optimal-binning-scorecard/src/monitoring.py` (que mide PSI/CSI sobre el
*score* y los *bins* de un scorecard ya entrenado), este módulo opera sobre
cualquier columna numérica cruda -- el caso de uso es detectar drift en las
features de entrada antes de siquiera llegar al modelo.
"""
from __future__ import annotations

import datetime as dt
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EPSILON = 1e-6
PSI_ALERT_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
KS_ALPHA = 0.05

# Colores de badge por estado, en un solo lugar para que
# export_drift_html no pueda desincronizarse de classify_psi.
_BADGE_COLOR = {"green": "#28a745", "yellow": "#ffc107", "red": "#dc3545"}
_BADGE_LABEL = {"green": "ESTABLE", "yellow": "ALERTA", "red": "CRITICO"}
# Tope para la barra de magnitud: un PSI de 0.5 ya es un drift severo varias
# veces por encima del umbral crítico (0.25) -- capar ahí evita que un
# outlier de PSI=3.0 deje a todas las demás barras ilegibles por comparación.
_PSI_BAR_CAP = 0.5


def _bucket_edges(expected: np.ndarray, num_buckets: int) -> np.ndarray:
    """Cortes de cuantiles de `expected`, deduplicados.

    Con pocos datos o una columna casi constante, los cuantiles de
    `num_buckets` colapsan en valores repetidos (`np.quantile` no lo evita
    por sí solo); `np.unique` reduce el número de buckets efectivos en vez
    de fallar o dividir por cero en un bucket vacío.
    """
    cuantiles = np.linspace(0, 1, num_buckets + 1)[1:-1]
    return np.unique(np.quantile(expected, cuantiles))


def _bucket_proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right")
    conteos = np.bincount(indices, minlength=len(edges) + 1).astype(float)
    return conteos / conteos.sum()


def calculate_psi(expected, actual, num_buckets: int = 10) -> float:
    """Population Stability Index entre `expected` (base) y `actual` (scoring).

    PSI = sum_bucket (pct_actual - pct_expected) * ln(pct_actual / pct_expected)

    Los cortes de bucket son los cuantiles de `expected` -- PSI mide cuánto
    se corrió `actual` respecto de esa referencia, no al revés. NaNs se
    descartan de cada lado antes de calcular (ver `_valores_finitos`);
    ambos lados vacíos, o completamente vacíos tras filtrar, son un error
    explícito, no un PSI silenciosamente inválido.

    Límite conocido: si `expected` es (casi) constante, sus cuantiles
    colapsan en un único corte y el PSI resultante no puede distinguir
    valores nuevos entre sí -- da 0.0 en vez de fallar, pero no detecta el
    drift en ese caso degenerado. Es inherente a discretizar por cuantiles
    de la propia base, no algo que este módulo intente resolver.
    """
    expected = _valores_finitos(expected, "expected")
    actual = _valores_finitos(actual, "actual")

    edges = _bucket_edges(expected, num_buckets)
    pct_expected = np.clip(_bucket_proportions(expected, edges), EPSILON, None)
    pct_actual = np.clip(_bucket_proportions(actual, edges), EPSILON, None)

    return float(np.sum((pct_actual - pct_expected) * np.log(pct_actual / pct_expected)))


def calculate_ks_drift(expected, actual) -> dict:
    """Prueba KS de dos muestras entre `expected` y `actual`.

    Devuelve el estadístico (distancia máxima entre las dos CDFs empíricas),
    el p-valor, y `drift_detected` -- True si el p-valor cae bajo `KS_ALPHA`,
    es decir, si se rechaza la hipótesis nula de que vienen de la misma
    distribución.
    """
    expected = _valores_finitos(expected, "expected")
    actual = _valores_finitos(actual, "actual")

    resultado = stats.ks_2samp(expected, actual)
    return {
        "statistic": float(resultado.statistic),
        "p_value": float(resultado.pvalue),
        "drift_detected": bool(resultado.pvalue < KS_ALPHA),
    }


def classify_psi(psi: float) -> str:
    """Semáforo normativo: 'green' (<0.10), 'yellow' (0.10-0.25), 'red' (>0.25)."""
    if psi < PSI_ALERT_THRESHOLD:
        return "green"
    if psi <= PSI_CRITICAL_THRESHOLD:
        return "yellow"
    return "red"


def generate_drift_report(baseline_df: pd.DataFrame, scoring_df: pd.DataFrame,
                           features: list[str], num_buckets: int = 10) -> dict:
    """PSI + KS por feature, comparando `baseline_df` (train) contra
    `scoring_df` (OOT/producción).

    Devuelve un dict serializable a JSON con una entrada por feature y un
    resumen (`features_en_alerta`, `features_criticas`) para no tener que
    recorrer todas las features a mano buscando cuáles superaron el umbral.
    """
    por_feature = {}
    for feature in features:
        psi = calculate_psi(baseline_df[feature], scoring_df[feature], num_buckets=num_buckets)
        ks = calculate_ks_drift(baseline_df[feature], scoring_df[feature])
        por_feature[feature] = {
            "psi": psi,
            "psi_status": classify_psi(psi),
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "ks_drift_detected": ks["drift_detected"],
        }

    en_alerta = [f for f, r in por_feature.items() if r["psi_status"] == "yellow"]
    criticas = [f for f, r in por_feature.items() if r["psi_status"] == "red"]

    return {
        "features": por_feature,
        "features_en_alerta": en_alerta,
        "features_criticas": criticas,
        "n_baseline": int(len(baseline_df)),
        "n_scoring": int(len(scoring_df)),
    }


def _valores_finitos(x, nombre: str) -> np.ndarray:
    valores = np.asarray(x, dtype=float)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        raise ValueError(f"'{nombre}' no tiene ningun valor finito para calcular drift")
    return valores


def _overall_status(report: dict) -> str:
    """Semáforo del reporte completo: el peor estado entre todas las features."""
    if report["features_criticas"]:
        return "red"
    if report["features_en_alerta"]:
        return "yellow"
    return "green"


def export_drift_json(report: dict, json_path) -> Path:
    """Exporta `report` (de `generate_drift_report`) a JSON formateado, con
    timestamp de generación y el semáforo agregado del reporte completo.
    """
    payload = {
        **report,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "overall_status": _overall_status(report),
    }
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def _badge(status: str) -> str:
    color = _BADGE_COLOR[status]
    label = _BADGE_LABEL[status]
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.85em;font-weight:600;">{label}</span>'
    )


def _psi_bar(psi: float, status: str) -> str:
    color = _BADGE_COLOR[status]
    ancho = min(max(psi, 0.0) / _PSI_BAR_CAP, 1.0) * 100
    return (
        '<div style="background:#e9ecef;border-radius:4px;width:140px;height:10px;'
        'display:inline-block;vertical-align:middle;overflow:hidden;">'
        f'<div style="background:{color};width:{ancho:.1f}%;height:100%;"></div>'
        "</div>"
    )


def export_drift_html(report: dict, html_path) -> Path:
    """Genera un reporte HTML autónomo (CSS inline, sin JS ni dependencias
    externas) con una fila por feature: PSI, badge de color, barra de
    magnitud, y el resultado de KS.
    """
    estado_general = _overall_status(report)
    generado = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    filas = []
    for feature, datos in report["features"].items():
        filas.append(
            "<tr>"
            f"<td>{escape(feature)}</td>"
            f"<td>{datos['psi']:.4f}</td>"
            f"<td>{_psi_bar(datos['psi'], datos['psi_status'])}</td>"
            f"<td>{_badge(datos['psi_status'])}</td>"
            f"<td>{datos['ks_statistic']:.4f}</td>"
            f"<td>{datos['ks_p_value']:.4g}</td>"
            f"<td>{'si' if datos['ks_drift_detected'] else 'no'}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte de Drift</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; color: #212529; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
  .meta {{ color: #6c757d; font-size: 0.9em; margin-bottom: 1.2rem; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
  th {{ background: #f8f9fa; font-weight: 600; }}
  .resumen {{ margin-bottom: 1rem; }}
</style>
</head>
<body>
  <h1>Reporte de Drift</h1>
  <div class="meta">Generado {escape(generado)} -- {report['n_baseline']} filas base vs. {report['n_scoring']} filas scoring</div>
  <div class="resumen">Estado general: {_badge(estado_general)}</div>
  <table>
    <thead>
      <tr><th>Feature</th><th>PSI</th><th>Magnitud</th><th>Estado</th><th>KS stat</th><th>KS p-valor</th><th>Drift KS</th></tr>
    </thead>
    <tbody>
      {''.join(filas)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    return html_path
