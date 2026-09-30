"""Lo que un cutover no puede responder por si solo: el candidato paso el
canary con datos de unas pocas horas o dias, pero recien meses despues,
cuando los prestamos que curso como Champion maduran, se sabe si de verdad
predijo bien el riesgo real. Esta tecnica mide eso -- desempeno realizado
contra verdad de campo observada, mas si la distribucion de sus propios
scores se corrio respecto de cuando se entreno -- y no antes de que haya
suficiente verdad de campo para que la medicion signifique algo.

Ninguna tecnica anterior de este portafolio registra "prediccion del
Champion en produccion" y "resultado real observado" para el mismo
cliente: 18-canary-deployment loguea predicciones durante la fase canaria
(antes del cutover, con CHAMPION y CANARY mezclados), y ninguna etapa
anterior tiene un concepto de default madurado. Por eso
`PostCutoverTelemetryEngine` recibe DataFrames directamente en vez de leer
un esquema rio arriba que no existe: es `run_telemetry.py`, no el motor,
quien decide de donde salen esos DataFrames.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ("target", "default_flag")
MIN_OBSERVATIONS_DEFAULT = 50

SEVERITY_THRESHOLDS = (
    (0.10, "no_shift"),
    (0.25, "moderate_shift"),
)
SEVERITY_BEYOND = "significant_shift"
PSI_EPSILON = 1e-4


class InsufficientTelemetryDataError(RuntimeError):
    """Sin filas en comun entre predicciones y verdad de campo, o sin datos
    suficientes para construir los buckets del PSI -- la razon concreta va
    en el mensaje."""


def _severity(psi: float) -> str:
    for umbral, etiqueta in SEVERITY_THRESHOLDS:
        if psi < umbral:
            return etiqueta
    return SEVERITY_BEYOND


class PostCutoverTelemetryEngine:
    """`compute_realized_performance` y `compute_prediction_drift` se
    pueden llamar en cualquier orden, o uno solo -- cada uno deja su
    resultado en `self.performance` / `self.drift` para que
    `generate_telemetry_report` combine lo que haya, igual que
    `CanaryHealthMonitor` (tecnica 19) combina metricas y decision."""

    def __init__(self):
        self.performance: dict | None = None
        self.drift: dict | None = None

    def compute_realized_performance(self, predictions_df: pd.DataFrame,
                                      ground_truth_df: pd.DataFrame) -> dict:
        if "client_id" not in predictions_df.columns or "prediction" not in predictions_df.columns:
            raise ValueError("predictions_df necesita las columnas 'client_id' y 'prediction'")

        columna_etiqueta = next((c for c in LABEL_COLUMNS if c in ground_truth_df.columns), None)
        if columna_etiqueta is None or "client_id" not in ground_truth_df.columns:
            raise ValueError(
                f"ground_truth_df necesita 'client_id' y una de {LABEL_COLUMNS}")

        fusion = predictions_df[["client_id", "prediction"]].merge(
            ground_truth_df[["client_id", columna_etiqueta]].rename(columns={columna_etiqueta: "target"}),
            on="client_id", how="inner",
        )
        if len(fusion) == 0:
            raise InsufficientTelemetryDataError(
                "ningun client_id en comun entre predictions_df y ground_truth_df")

        y = fusion["target"].to_numpy(dtype=float)
        # Clip, no para el AUC (invariante a la escala) sino para que
        # log_loss nunca evalue ln(0) si algun score llego exactamente en
        # 0.0 o 1.0.
        p = np.clip(fusion["prediction"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)

        clases_presentes = np.unique(y)
        if clases_presentes.size < 2:
            realized_roc_auc = None
            razon_auc_no_definida = (
                f"solo se observo la clase {int(clases_presentes[0])} en este lote "
                "-- el AUC no esta definido sin ambas clases"
            )
        else:
            realized_roc_auc = float(roc_auc_score(y, p))
            razon_auc_no_definida = None

        resultado = {
            "status": "EVALUATED",
            "realized_roc_auc": realized_roc_auc,
            "auc_undefined_reason": razon_auc_no_definida,
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "observed_default_rate": float(y.mean()),
            "predicted_default_rate": float(p.mean()),
            "n_evaluated": int(len(fusion)),
        }
        self.performance = resultado
        return dict(resultado)

    def note_insufficient_maturity(self, n_observed: int,
                                    min_required: int = MIN_OBSERVATIONS_DEFAULT) -> dict:
        """La cartera todavia no tiene suficiente verdad de campo madurada
        para que un AUC o un Brier score signifiquen algo -- es el estado
        normal en los primeros dias despues de un cutover, no un error."""
        nota = {
            "status": "INSUFFICIENT_MATURITY",
            "n_observed": int(n_observed),
            "min_required": int(min_required),
            "reason": (
                f"solo {n_observed} etiqueta(s) de verdad de campo observada(s), "
                f"se necesitan al menos {min_required} -- la cartera necesita mas "
                "tiempo de maduracion antes de evaluar desempeno realizado"
            ),
        }
        self.performance = nota
        return dict(nota)

    def compute_prediction_drift(self, baseline_preds, recent_preds,
                                  num_buckets: int = 10) -> dict:
        """PSI de las distribuciones de score, exclusivamente -- nunca de
        las features de entrada. Los cortes de los buckets son los deciles
        (o `num_buckets` cuantiles) del baseline; `recent` se mide contra
        esos mismos cortes, no contra los suyos propios, porque el punto
        del PSI es detectar que la distribucion de produccion se movio
        *respecto del baseline*, no describir la distribucion de produccion
        en el vacio."""
        baseline = np.asarray(baseline_preds, dtype=float)
        recent = np.asarray(recent_preds, dtype=float)

        if baseline.size < num_buckets:
            raise InsufficientTelemetryDataError(
                f"baseline_preds tiene {baseline.size} valores, se necesitan al "
                f"menos {num_buckets} para construir {num_buckets} buckets")
        if recent.size == 0:
            raise InsufficientTelemetryDataError("recent_preds esta vacio")

        # num_buckets - 1 cortes interiores; los cuantiles 0 y 1 no se
        # necesitan como corte porque searchsorted ya manda todo lo que cae
        # antes del primer corte o despues del ultimo a los buckets extremos.
        cortes = np.quantile(baseline, np.linspace(0.0, 1.0, num_buckets + 1)[1:-1])

        idx_baseline = np.searchsorted(cortes, baseline, side="right")
        idx_recent = np.searchsorted(cortes, recent, side="right")

        conteo_baseline = np.bincount(idx_baseline, minlength=num_buckets)[:num_buckets]
        conteo_recent = np.bincount(idx_recent, minlength=num_buckets)[:num_buckets]

        pct_baseline = np.clip(conteo_baseline / baseline.size, PSI_EPSILON, None)
        pct_recent = np.clip(conteo_recent / recent.size, PSI_EPSILON, None)

        contribuciones = (pct_recent - pct_baseline) * np.log(pct_recent / pct_baseline)
        psi = float(contribuciones.sum())

        resultado = {
            "psi": psi,
            "num_buckets": num_buckets,
            "severity": _severity(psi),
            "n_baseline": int(baseline.size),
            "n_recent": int(recent.size),
            "bucket_detail": [
                {"bucket": i, "pct_baseline": float(pct_baseline[i]), "pct_recent": float(pct_recent[i])}
                for i in range(num_buckets)
            ],
        }
        self.drift = resultado
        return dict(resultado)

    def _summary(self) -> str:
        partes = []
        if self.performance is not None:
            if self.performance.get("status") == "INSUFFICIENT_MATURITY":
                partes.append(self.performance["reason"])
            else:
                auc_txt = (f"{self.performance['realized_roc_auc']:.4f}"
                          if self.performance["realized_roc_auc"] is not None
                          else "no definido")
                sesgo_pp = 100 * (self.performance["predicted_default_rate"]
                                   - self.performance["observed_default_rate"])
                partes.append(
                    f"Desempeno realizado sobre {self.performance['n_evaluated']} "
                    f"observaciones: AUC={auc_txt}, Brier={self.performance['brier_score']:.4f}, "
                    f"log-loss={self.performance['log_loss']:.4f}, "
                    f"sesgo de calibracion={sesgo_pp:+.2f}pp "
                    f"(predicho {self.performance['predicted_default_rate']:.2%} vs. "
                    f"observado {self.performance['observed_default_rate']:.2%})."
                )
        if self.drift is not None:
            partes.append(
                f"Drift de prediccion: PSI={self.drift['psi']:.4f} ({self.drift['severity']}) "
                f"sobre {self.drift['num_buckets']} buckets "
                f"({self.drift['n_baseline']} baseline vs. {self.drift['n_recent']} recientes)."
            )
        return " ".join(partes) if partes else "Sin metricas calculadas todavia."

    def generate_telemetry_report(self, output_dir) -> Path:
        if self.performance is None and self.drift is None:
            raise RuntimeError(
                "generate_telemetry_report() llamado sin haber calculado desempeno "
                "realizado ni drift de prediccion en esta instancia.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ahora = dt.datetime.now(dt.timezone.utc)
        payload = {
            "realized_performance": self.performance,
            "prediction_drift": self.drift,
            "summary": self._summary(),
            "generated_at": ahora.isoformat(timespec="seconds"),
        }
        timestamp = ahora.strftime("%Y%m%dT%H%M%S%f") + "Z"
        report_path = output_dir / f"post_cutover_telemetry_{timestamp}.json"
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        logger.info("Reporte de telemetria post-cutover: %s -> %s", payload["summary"], report_path)
        return report_path
