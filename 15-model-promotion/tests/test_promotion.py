"""Tests de src.promotion_logic.ModelPromoter y de run_promotion.py."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from run_promotion import find_latest_metrics, main
from src.promotion_logic import ModelPromoter, PromotionEvaluationError


def _escribir_metrics(tmp_path, nombre: str, roc_auc: float, n_samples: int) -> Path:
    ruta = tmp_path / nombre
    ruta.write_text(json.dumps({
        "roc_auc": roc_auc,
        "n_samples": n_samples,
        "n_train": int(n_samples * 0.8),
        "n_test": int(n_samples * 0.2),
        "trained_at": "2026-09-27T00:00:00+00:00",
    }, indent=2), encoding="utf-8")
    return ruta


# --------------------------------------------------------------------- PROMOTED

def test_candidato_por_encima_de_los_umbrales_es_promoted(tmp_path):
    metrics_path = _escribir_metrics(tmp_path, "shadow_metrics_20260927T195118Z.json",
                                      roc_auc=0.95, n_samples=2000)

    promoter = ModelPromoter()
    evaluacion = promoter.evaluate_candidate(metrics_path, min_roc_auc=0.75, min_samples=1000)

    assert evaluacion["decision"] == "PROMOTED"
    assert evaluacion["reason"] == "Meets minimum ROC-AUC and sample size thresholds"
    assert evaluacion["candidate_model"] == "shadow_model_20260927T195118Z.pkl"


def test_candidato_exactamente_en_el_umbral_es_promoted(tmp_path):
    """El umbral es inclusivo (roc_auc >= min_roc_auc), no estricto."""
    metrics_path = _escribir_metrics(tmp_path, "shadow_metrics_a.json", roc_auc=0.75, n_samples=1000)

    promoter = ModelPromoter()
    evaluacion = promoter.evaluate_candidate(metrics_path, min_roc_auc=0.75, min_samples=1000)

    assert evaluacion["decision"] == "PROMOTED"


# --------------------------------------------------------------------- REJECTED

def test_roc_auc_insuficiente_es_rejected_con_el_mensaje_exacto(tmp_path):
    metrics_path = _escribir_metrics(tmp_path, "shadow_metrics_b.json", roc_auc=0.68, n_samples=2000)

    promoter = ModelPromoter()
    evaluacion = promoter.evaluate_candidate(metrics_path, min_roc_auc=0.75, min_samples=1000)

    assert evaluacion["decision"] == "REJECTED"
    assert evaluacion["reason"] == "Failed ROC-AUC threshold: 0.68 < 0.75"


def test_muestra_insuficiente_es_rejected(tmp_path):
    metrics_path = _escribir_metrics(tmp_path, "shadow_metrics_c.json", roc_auc=0.90, n_samples=300)

    promoter = ModelPromoter()
    evaluacion = promoter.evaluate_candidate(metrics_path, min_roc_auc=0.75, min_samples=1000)

    assert evaluacion["decision"] == "REJECTED"
    assert evaluacion["reason"] == "Failed sample size threshold: 300 < 1000"


def test_ambos_umbrales_fallidos_incluye_las_dos_razones(tmp_path):
    metrics_path = _escribir_metrics(tmp_path, "shadow_metrics_d.json", roc_auc=0.50, n_samples=200)

    promoter = ModelPromoter()
    evaluacion = promoter.evaluate_candidate(metrics_path, min_roc_auc=0.75, min_samples=1000)

    assert evaluacion["decision"] == "REJECTED"
    assert "ROC-AUC" in evaluacion["reason"]
    assert "sample size" in evaluacion["reason"]


def test_usa_los_umbrales_por_defecto_si_no_se_pasan(tmp_path):
    metrics_path = _escribir_metrics(tmp_path, "shadow_metrics_e.json", roc_auc=0.76, n_samples=1001)

    promoter = ModelPromoter()
    evaluacion = promoter.evaluate_candidate(metrics_path)  # sin min_roc_auc/min_samples explicitos

    assert evaluacion["decision"] == "PROMOTED"


# ------------------------------------------------------------ generate_decision

def test_generate_decision_escribe_el_json_con_los_campos_pedidos(tmp_path):
    metrics_path = _escribir_metrics(tmp_path, "shadow_metrics_20260927T195118Z.json",
                                      roc_auc=0.95, n_samples=2000)

    promoter = ModelPromoter()
    promoter.evaluate_candidate(metrics_path)
    resultado = promoter.generate_decision(tmp_path / "decisions")

    decision_path = Path(resultado["decision_path"])
    assert decision_path.exists()
    assert decision_path.name.startswith("promotion_decision_")

    contenido = json.loads(decision_path.read_text(encoding="utf-8"))
    assert contenido["candidate_model"] == "shadow_model_20260927T195118Z.pkl"
    assert contenido["decision"] == "PROMOTED"
    assert contenido["reason"] == "Meets minimum ROC-AUC and sample size thresholds"
    assert "evaluated_at" in contenido


def test_generate_decision_sin_evaluar_antes_lanza_error(tmp_path):
    promoter = ModelPromoter()
    with pytest.raises(RuntimeError, match="no hay una evaluacion"):
        promoter.generate_decision(tmp_path)


# --------------------------------------------------------------- manejo seguro

def test_archivo_de_metricas_inexistente_lanza_promotion_evaluation_error(tmp_path):
    promoter = ModelPromoter()
    with pytest.raises(PromotionEvaluationError):
        promoter.evaluate_candidate(tmp_path / "no-existe.json")


def test_json_corrupto_lanza_promotion_evaluation_error(tmp_path):
    ruta = tmp_path / "roto.json"
    ruta.write_text("{esto no es json valido,,,", encoding="utf-8")

    promoter = ModelPromoter()
    with pytest.raises(PromotionEvaluationError):
        promoter.evaluate_candidate(ruta)


def test_json_sin_los_campos_esperados_lanza_promotion_evaluation_error(tmp_path):
    ruta = tmp_path / "incompleto.json"
    ruta.write_text(json.dumps({"otra_cosa": 1}), encoding="utf-8")

    promoter = ModelPromoter()
    with pytest.raises(PromotionEvaluationError):
        promoter.evaluate_candidate(ruta)


# ------------------------------------------------------------ CLI: run_promotion.py

def test_find_latest_metrics_elige_el_de_timestamp_mas_alto(tmp_path):
    (tmp_path / "shadow_metrics_20260101T000000Z.json").write_text("{}")
    mas_reciente = tmp_path / "shadow_metrics_20260927T235959Z.json"
    mas_reciente.write_text("{}")
    (tmp_path / "shadow_metrics_20260615T120000Z.json").write_text("{}")

    assert find_latest_metrics(tmp_path) == mas_reciente


def test_find_latest_metrics_devuelve_none_si_no_hay_nada(tmp_path):
    assert find_latest_metrics(tmp_path / "no-existe") is None
    assert find_latest_metrics(tmp_path) is None


def test_cli_evalua_el_mas_reciente_y_devuelve_0_sea_cual_sea_la_decision(tmp_path):
    _escribir_metrics(tmp_path, "shadow_metrics_20260927T000000Z.json", roc_auc=0.30, n_samples=50)
    output_dir = tmp_path / "decisions"

    codigo = main([
        "--metrics-dir", str(tmp_path), "--output-dir", str(output_dir),
    ])

    assert codigo == 0  # REJECTED no es un error de CI
    decisiones = list(output_dir.glob("promotion_decision_*.json"))
    assert len(decisiones) == 1
    contenido = json.loads(decisiones[0].read_text(encoding="utf-8"))
    assert contenido["decision"] == "REJECTED"


def test_cli_no_lanza_excepcion_sin_ningun_archivo_de_metricas(tmp_path, caplog):
    output_dir = tmp_path / "decisions"

    with caplog.at_level(logging.INFO):
        codigo = main(["--metrics-dir", str(tmp_path / "vacio"), "--output-dir", str(output_dir)])

    assert codigo == 0
    assert not output_dir.exists()


def test_cli_no_lanza_excepcion_con_metrics_corrupto(tmp_path, caplog):
    (tmp_path / "shadow_metrics_20260927T000000Z.json").write_text("no es json", encoding="utf-8")
    output_dir = tmp_path / "decisions"

    with caplog.at_level(logging.INFO):
        codigo = main(["--metrics-dir", str(tmp_path), "--output-dir", str(output_dir)])

    assert codigo == 0
    assert not output_dir.exists()
    assert any("abortando sin error" in r.message for r in caplog.records)


def test_cli_respeta_umbrales_pasados_por_linea_de_comandos(tmp_path):
    _escribir_metrics(tmp_path, "shadow_metrics_20260927T000000Z.json", roc_auc=0.60, n_samples=600)
    output_dir = tmp_path / "decisions"

    codigo = main([
        "--metrics-dir", str(tmp_path), "--output-dir", str(output_dir),
        "--min-roc-auc", "0.55", "--min-samples", "500",
    ])

    assert codigo == 0
    contenido = json.loads(next(output_dir.glob("promotion_decision_*.json")).read_text(encoding="utf-8"))
    assert contenido["decision"] == "PROMOTED"
