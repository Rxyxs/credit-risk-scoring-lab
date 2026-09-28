"""Último eslabón de la cadena de remediación por drift: 12 detecta y
dispara, 13 ingiere, 14 entrena el modelo sombra, 15 decide si ese modelo
sombra cruza la barra mínima para producción -- una decisión de negocio
(PROMOTED/REJECTED), nunca un error técnico. Esto no despliega nada: solo
deja escrita la decisión para que otro proceso (fuera de este repo) actúe
sobre ella.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MIN_ROC_AUC = 0.75
DEFAULT_MIN_SAMPLES = 1000

_MODEL_PREFIX = "shadow_model_"
_METRICS_PREFIX = "shadow_metrics_"


class PromotionEvaluationError(RuntimeError):
    """El archivo de métricas no existe, no es JSON válido, o no tiene los
    campos que `evaluate_candidate` necesita (`roc_auc`, `n_samples`) --
    todos los motivos por los que `run_promotion.py` aborta con código 0 en
    vez de fallar."""


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class ModelPromoter:
    """`evaluate_candidate` -> `generate_decision`, en ese orden -- la
    segunda escribe el resultado que dejó la primera en `self.evaluation`,
    igual que `ShadowModelTrainer` en la técnica 14."""

    def __init__(self):
        self.evaluation: dict | None = None

    def evaluate_candidate(self, metrics_path, min_roc_auc: float = DEFAULT_MIN_ROC_AUC,
                            min_samples: int = DEFAULT_MIN_SAMPLES) -> dict:
        metrics_path = Path(metrics_path)

        try:
            contenido = metrics_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PromotionEvaluationError(f"no pude leer '{metrics_path}': {exc}") from exc

        try:
            metricas = json.loads(contenido)
        except json.JSONDecodeError as exc:
            raise PromotionEvaluationError(f"'{metrics_path}' no es JSON valido: {exc}") from exc

        try:
            roc_auc = float(metricas["roc_auc"])
            n_samples = int(metricas["n_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionEvaluationError(
                f"'{metrics_path}' no tiene 'roc_auc'/'n_samples' numericos validos: {exc}"
            ) from exc

        razones_de_fallo = []
        if roc_auc < min_roc_auc:
            razones_de_fallo.append(f"Failed ROC-AUC threshold: {roc_auc:.2f} < {min_roc_auc:.2f}")
        if n_samples < min_samples:
            razones_de_fallo.append(f"Failed sample size threshold: {n_samples} < {min_samples}")

        decision = "REJECTED" if razones_de_fallo else "PROMOTED"
        reason = "; ".join(razones_de_fallo) if razones_de_fallo \
            else "Meets minimum ROC-AUC and sample size thresholds"

        self.evaluation = {
            "candidate_model": self._derivar_nombre_modelo(metrics_path),
            "decision": decision,
            "reason": reason,
            "roc_auc": roc_auc,
            "n_samples": n_samples,
            "min_roc_auc": min_roc_auc,
            "min_samples": min_samples,
        }
        return dict(self.evaluation)

    @staticmethod
    def _derivar_nombre_modelo(metrics_path: Path) -> str:
        """`shadow_metrics_<timestamp>.json` -> `shadow_model_<timestamp>.pkl`
        -- mismo timestamp para el par, tal como los escribe
        `ShadowModelTrainer.save_model` (técnica 14). No hace falta que el
        .pkl exista de verdad: esto es solo una referencia por nombre para
        el manifiesto de decisión, no un enlace que se resuelva acá."""
        nombre = metrics_path.name
        if nombre.startswith(_METRICS_PREFIX) and nombre.endswith(".json"):
            timestamp = nombre[len(_METRICS_PREFIX):-len(".json")]
            return f"{_MODEL_PREFIX}{timestamp}.pkl"
        return nombre  # nombre de archivo atipico: se deja tal cual, sin adivinar

    def generate_decision(self, output_dir) -> dict:
        if self.evaluation is None:
            raise RuntimeError("no hay una evaluacion todavia -- llama evaluate_candidate() primero")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "candidate_model": self.evaluation["candidate_model"],
            "decision": self.evaluation["decision"],
            "reason": self.evaluation["reason"],
            "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        decision_path = output_dir / f"promotion_decision_{_timestamp()}.json"
        decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "Decision de promocion: %s -- %s (modelo: %s)",
            payload["decision"], payload["reason"], payload["candidate_model"],
        )

        return {"decision_path": str(decision_path), "decision": payload}
