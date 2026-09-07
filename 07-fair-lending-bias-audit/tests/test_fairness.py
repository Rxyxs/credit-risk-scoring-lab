"""Tests de las metricas de equidad.

Las metricas de equidad se prestan para errores silenciosos: un signo
cambiado invierte quien esta siendo perjudicado, y el numero sigue
pareciendo razonable. Por eso casi todos los casos de abajo son escenarios
construidos a mano, donde la respuesta correcta se conoce de antemano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness_metrics import (
    UMBRAL_REGLA_CUATRO_QUINTOS, bootstrap_metricas, brecha_condicional,
    decisiones, metricas_de_equidad, resumen_brecha_condicional, tabla_por_grupo,
    umbral_por_tasa_de_aprobacion,
)


def _escenario(n=4000, brecha=0.0, seed=0):
    """Dos grupos; `brecha` desplaza la PD del grupo protegido hacia arriba."""
    rng = np.random.default_rng(seed)
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    pd_pred = np.clip(rng.beta(2, 8, n) + (grupo == "F") * brecha, 0.001, 0.999)
    y = (rng.random(n) < pd_pred).astype(int)
    return y, pd_pred, grupo


def test_sin_brecha_las_metricas_dan_paridad():
    y, pd_pred, grupo = _escenario(n=20_000, brecha=0.0, seed=1)
    umbral = umbral_por_tasa_de_aprobacion(pd_pred, 0.80)
    m = metricas_de_equidad(y, pd_pred, grupo, umbral, "F", "M")
    assert m["ratio_impacto_adverso"] == pytest.approx(1.0, abs=0.02)
    assert abs(m["paridad_demografica_pp"]) < 2.0
    assert m["cumple_regla_cuatro_quintos"]


def test_una_brecha_grande_rompe_la_regla_de_los_cuatro_quintos():
    y, pd_pred, grupo = _escenario(n=20_000, brecha=0.22, seed=2)
    umbral = umbral_por_tasa_de_aprobacion(pd_pred, 0.60)
    m = metricas_de_equidad(y, pd_pred, grupo, umbral, "F", "M")
    assert m["ratio_impacto_adverso"] < UMBRAL_REGLA_CUATRO_QUINTOS
    assert not m["cumple_regla_cuatro_quintos"]
    assert m["paridad_demografica_pp"] < -10


def test_el_signo_indica_quien_queda_perjudicado():
    """Invertir los roles de los grupos tiene que invertir el signo."""
    y, pd_pred, grupo = _escenario(n=8000, brecha=0.12, seed=3)
    umbral = umbral_por_tasa_de_aprobacion(pd_pred, 0.75)
    m1 = metricas_de_equidad(y, pd_pred, grupo, umbral, "F", "M")
    m2 = metricas_de_equidad(y, pd_pred, grupo, umbral, "M", "F")
    assert m1["paridad_demografica_pp"] < 0 < m2["paridad_demografica_pp"]
    assert m1["paridad_demografica_pp"] == pytest.approx(-m2["paridad_demografica_pp"])
    assert m1["ratio_impacto_adverso"] < 1 < m2["ratio_impacto_adverso"]


def test_tasas_calculadas_a_mano():
    # 4 solicitantes del grupo F (2 aprobados) y 4 del grupo M (4 aprobados)
    pd_pred = np.array([0.05, 0.05, 0.90, 0.90, 0.05, 0.05, 0.05, 0.05])
    grupo = np.array(["F"] * 4 + ["M"] * 4)
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    m = metricas_de_equidad(y, pd_pred, grupo, umbral=0.5,
                            grupo_protegido="F", grupo_referencia="M")
    assert m["tasa_seleccion_protegido"] == pytest.approx(0.5)
    assert m["tasa_seleccion_referencia"] == pytest.approx(1.0)
    assert m["ratio_impacto_adverso"] == pytest.approx(0.5)
    assert m["paridad_demografica_pp"] == pytest.approx(-50.0)
    assert not m["cumple_regla_cuatro_quintos"]


def test_igualdad_de_oportunidad_solo_mira_a_los_buenos():
    # Todos los buenos aprobados en ambos grupos, pero mas malos aprobados en M
    pd_pred = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    grupo = np.array(["F", "F", "F", "M", "M", "M"])
    y = np.array([0, 0, 1, 0, 0, 1])
    m = metricas_de_equidad(y, pd_pred, grupo, 0.5, "F", "M")
    assert m["igualdad_oportunidad_pp"] == pytest.approx(0.0)   # buenos: 100% en ambos
    assert m["odds_igualados_pp"] == pytest.approx(100.0)       # malos: 0% vs 100%


def test_umbral_por_tasa_de_aprobacion_aprueba_lo_pedido():
    rng = np.random.default_rng(5)
    pd_pred = rng.beta(2, 8, 10_000)
    for tasa in (0.5, 0.8, 0.95):
        u = umbral_por_tasa_de_aprobacion(pd_pred, tasa)
        assert decisiones(pd_pred, u).mean() == pytest.approx(tasa, abs=0.01)
    with pytest.raises(ValueError):
        umbral_por_tasa_de_aprobacion(pd_pred, 1.5)


def test_tabla_por_grupo_reporta_calibracion_y_auc():
    y, pd_pred, grupo = _escenario(n=6000, brecha=0.05, seed=6)
    tabla = tabla_por_grupo(y, pd_pred, grupo, umbral=0.15)
    assert set(tabla["grupo"]) == {"F", "M"}
    assert (tabla["n"] > 0).all()
    assert tabla["auc"].between(0.5, 1.0).all()
    # como la PD es la verdadera, el sesgo de calibracion debe ser chico
    assert tabla["sesgo_calibracion_pp"].abs().max() < 3.0


def test_bootstrap_entrega_intervalos_que_contienen_la_estimacion():
    y, pd_pred, grupo = _escenario(n=4000, brecha=0.08, seed=7)
    umbral = umbral_por_tasa_de_aprobacion(pd_pred, 0.80)
    ic = bootstrap_metricas(y, pd_pred, grupo, umbral, "F", "M", n_boot=120, seed=1)
    assert len(ic) == 5
    assert (ic["ic_inferior"] <= ic["estimacion"]).all()
    assert (ic["estimacion"] <= ic["ic_superior"]).all()
    assert (ic["n_boot"] > 100).all()


def test_la_brecha_condicional_desaparece_si_solo_hay_factores_legitimos():
    """Si el grupo solo difiere en el factor de riesgo, controlarlo cierra la brecha."""
    rng = np.random.default_rng(8)
    n = 8000
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    # el grupo F tiene menor renta, y la PD depende SOLO de la renta
    log_renta = rng.normal(13.6, 0.4, n) - (grupo == "F") * 0.25
    dti = rng.beta(2, 8, n)
    pd_pred = 1 / (1 + np.exp(-(-0.8 * (log_renta - 13.6))))
    df = pd.DataFrame({"genero": grupo, "log_renta": log_renta, "dti": dti,
                       "pd_pred": pd_pred})

    bruta = (df.loc[df["genero"] == "F", "pd_pred"].mean()
             - df.loc[df["genero"] == "M", "pd_pred"].mean())
    tabla = brecha_condicional(df, "genero", "F", "M", "pd_pred", ["log_renta", "dti"], 4)
    resumen = resumen_brecha_condicional(tabla, bruta)

    assert bruta > 0.01
    assert abs(resumen["brecha_condicional"]) < abs(bruta) * 0.25
    assert resumen["pct_explicado_por_factores_legitimos"] > 75


def test_entradas_invalidas():
    y, pd_pred, grupo = _escenario(n=100)
    with pytest.raises(ValueError):
        metricas_de_equidad(y[:50], pd_pred, grupo, 0.2, "F", "M")
    with pytest.raises(ValueError):
        metricas_de_equidad(y, pd_pred, np.array(["F"] * 100), 0.2, "F", "M")
    with pytest.raises(ValueError):
        metricas_de_equidad(y, pd_pred, grupo, 0.2, "X", "M")
