"""Tests de la capa de decision, del simulador y del preprocesamiento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import TAU_TRUE, generate_applicants, segmentos
from src.decision import (
    LGD, MARGEN, comparar_a_igual_volumen, desacuerdos, frontera, resultado_cartera,
)
from src.preprocessing import (
    MAPA_SEGMENTO, apply_scaler, design_matrix, design_names, fit_scaler,
    segment_index, split_train_test,
)


# ---------------------------------------------------------------- datos
def test_simulador_es_reproducible_y_tiene_estructura_jerarquica():
    df, gt = generate_applicants(n=3000, seed=1)
    df2, _ = generate_applicants(n=3000, seed=1)
    assert df.equals(df2)
    assert df["segmento"].nunique() <= len(segmentos())
    assert gt["tau_true"] == TAU_TRUE
    assert len(gt["efectos_segmento_true"]) == len(segmentos())
    # los efectos verdaderos estan centrados: el nivel vive en el intercepto
    assert np.mean(list(gt["efectos_segmento_true"].values())) == pytest.approx(0, abs=1e-12)


def test_los_segmentos_tienen_tamanos_desbalanceados():
    df, _ = generate_applicants(n=9000, seed=42)
    tam = df["segmento"].value_counts()
    assert tam.min() < 100          # hay segmentos con poca informacion
    assert tam.max() > 10 * tam.min()


def test_la_senal_esta_en_las_features():
    df, _ = generate_applicants(n=8000, seed=3)
    con_moras = df.loc[df["n_moras_12m"] >= 2, "default_12m"].mean()
    sin_moras = df.loc[df["n_moras_12m"] == 0, "default_12m"].mean()
    assert con_moras > sin_moras * 1.5


# -------------------------------------------------------- preprocesamiento
def test_indice_de_segmento_cubre_todas_las_combinaciones():
    df, _ = generate_applicants(n=2000, seed=4)
    idx = segment_index(df)
    assert idx.min() >= 0
    assert idx.max() < len(MAPA_SEGMENTO)
    assert len(MAPA_SEGMENTO) == 32


def test_segmento_desconocido_falla_en_vez_de_puntuar_mal():
    df, _ = generate_applicants(n=100, seed=5)
    df.loc[0, "segmento"] = "Marte|formal|digital"
    with pytest.raises(ValueError):
        segment_index(df)


def test_estandarizacion_se_ajusta_solo_en_train():
    df, _ = generate_applicants(n=4000, seed=6)
    df["log_renta"] = np.log(df["renta_liquida"])
    train, test = split_train_test(df)
    scaler = fit_scaler(train)
    train_z, test_z = apply_scaler(train, scaler), apply_scaler(test, scaler)

    assert train_z["dti_z"].mean() == pytest.approx(0.0, abs=1e-9)
    assert train_z["dti_z"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)
    # test NO queda centrado exactamente: usa los momentos de train
    assert abs(test_z["dti_z"].mean()) > 0
    assert len(train) + len(test) == len(df)


def test_design_matrix_lleva_intercepto():
    df, _ = generate_applicants(n=500, seed=7)
    df["log_renta"] = np.log(df["renta_liquida"])
    scaler = fit_scaler(df)
    X = design_matrix(apply_scaler(df, scaler))
    assert X.shape == (500, len(design_names()))
    assert np.allclose(X[:, 0], 1.0)
    assert design_names()[0] == "intercepto"


# ------------------------------------------------------------- decision
def _cartera_predicha(n=800, seed=9):
    rng = np.random.default_rng(seed)
    pd_media = np.clip(rng.beta(2, 10, n), 0.005, 0.95)
    pd_sd = rng.uniform(0.01, 0.08, n)
    return pd.DataFrame({
        "default_12m": (rng.random(n) < pd_media).astype(int),
        "monto_credito": rng.uniform(500_000, 8_000_000, n),
        "segmento": rng.choice(["a", "b", "c"], n),
        "segmento_chico": rng.random(n) < 0.1,
        "pd_media": pd_media,
        "pd_sd": pd_sd,
        "pd_q95": pd_media + 1.64 * pd_sd,
    })


def test_resultado_cartera_calculado_a_mano():
    df = pd.DataFrame({
        "default_12m": [0, 1, 0],
        "monto_credito": [1_000_000.0, 2_000_000.0, 3_000_000.0],
    })
    r = resultado_cartera(df, np.array([True, True, False]))
    assert r["n_aprobados"] == 2
    assert r["tasa_mala_realizada"] == pytest.approx(0.5)
    assert r["perdida_realizada_clp"] == pytest.approx(2_000_000 * LGD)
    assert r["ingreso_realizado_clp"] == pytest.approx(1_000_000 * MARGEN)
    assert r["utilidad_realizada_clp"] == pytest.approx(1_000_000 * MARGEN - 2_000_000 * LGD)


def test_cartera_vacia_no_revienta():
    df = _cartera_predicha(n=10)
    r = resultado_cartera(df, np.zeros(10, dtype=bool))
    assert r["n_aprobados"] == 0
    assert r["utilidad_realizada_clp"] == 0.0


def test_frontera_aprueba_lo_pedido_y_empeora_con_el_volumen():
    df = _cartera_predicha()
    fr = frontera(df, tasas=np.array([0.5, 0.7, 0.9]))
    assert set(fr["politica"]) == {"media", "conservadora"}
    for politica, g in fr.groupby("politica"):
        g = g.sort_values("tasa_objetivo")
        assert np.allclose(g["tasa_aprobacion"], g["tasa_objetivo"], atol=0.01)
        # aprobar mas gente sube la tasa de default realizada
        assert g["tasa_mala_realizada"].is_monotonic_increasing


def test_comparacion_a_igual_volumen_usa_el_mismo_volumen():
    df = _cartera_predicha()
    fr = frontera(df, tasas=np.array([0.8]))
    c = comparar_a_igual_volumen(fr, 0.8)
    sel = fr.set_index("politica")
    assert sel.loc["media", "n_aprobados"] == sel.loc["conservadora", "n_aprobados"]
    assert c["delta_tasa_mala_pp"] == pytest.approx(
        100 * (c["tasa_mala_conservadora"] - c["tasa_mala_media"])
    )


def test_los_desacuerdos_son_casos_de_alta_incertidumbre():
    df = _cartera_predicha(n=1500, seed=12)
    d = desacuerdos(df, 0.8)
    assert d["n_cambian_de_decision"] > 0
    assert d["sd_posterior_media_rechazados"] > d["sd_posterior_media_cartera"]
