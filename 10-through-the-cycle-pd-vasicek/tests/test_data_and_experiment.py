"""Tests del simulador de cartera y del experimento de punta a punta."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    ANOS_RECESION, GRADOS, PD_TTC_TRUE, generar_cartera, rho_true, simular_ciclo,
)
from src.experiment import (
    capital_pit_vs_ttc, estimar_por_grado, recuperar_ciclo, sesgo_de_granularidad,
)


# --------------------------------------------------------------- ciclo
def test_ciclo_es_reproducible():
    a = simular_ciclo(30, np.random.default_rng(9))
    b = simular_ciclo(30, np.random.default_rng(9))
    assert np.allclose(a, b)


def test_ciclo_cae_marcadamente_en_los_anos_de_recesion():
    """Los anos elegidos a mano tienen que quedar entre los peores del
    ciclo simulado, no perderse en el ruido del AR(1)."""
    z = simular_ciclo(30, np.random.default_rng(10))
    orden = np.argsort(z)               # de peor a mejor
    peor_tercio = set(orden[:10])
    anos_recesion = set(ANOS_RECESION)
    # al menos la mitad de los anos de recesion marcados deben quedar
    # entre los diez peores del ciclo completo
    assert len(anos_recesion & peor_tercio) >= len(anos_recesion) // 2


def test_rho_true_coincide_con_la_formula_de_basilea():
    """Por diseno: la correlacion verdadera de cada grado es exactamente
    la formula regulatoria evaluada en su PD_ttc."""
    from src.vasicek import basel_asset_correlation

    for g in GRADOS:
        assert rho_true(g) == pytest.approx(float(basel_asset_correlation(PD_TTC_TRUE[g])))


# -------------------------------------------------------------- cartera
def test_cartera_tiene_un_grado_y_una_fila_por_ano():
    datos = generar_cartera(n_anos=10, n_por_grado_ano=500, seed=11)
    cohortes = datos["cohortes"]
    assert len(cohortes) == 10 * len(GRADOS)
    assert set(cohortes["grado"]) == set(GRADOS)
    assert set(cohortes["anio"]) == set(range(10))


def test_mas_deudores_por_cohorte_reduce_el_ruido_de_la_tasa_observada():
    """A mayor N, la tasa de default observada tiene que acercarse mas a
    la PD condicional teorica -- el ruido idiosincratico se diversifica."""
    chica = generar_cartera(n_anos=25, n_por_grado_ano=100, seed=12)["cohortes"]
    grande = generar_cartera(n_anos=25, n_por_grado_ano=50_000, seed=12)["cohortes"]

    err_chica = (chica["tasa_default_observada"] - chica["pd_condicional_teorica"]).abs().mean()
    err_grande = (grande["tasa_default_observada"] - grande["pd_condicional_teorica"]).abs().mean()
    assert err_grande < err_chica


def test_grados_mas_riesgosos_tienen_menor_correlacion_verdadera():
    """Propiedad de la formula de Basilea: a mayor PD, menor rho."""
    orden_pd = sorted(GRADOS, key=lambda g: PD_TTC_TRUE[g])
    rhos = [rho_true(g) for g in orden_pd]
    assert rhos == sorted(rhos, reverse=True)


# ----------------------------------------------------------- experimento
@pytest.fixture(scope="module")
def datos_chicos():
    return generar_cartera(n_anos=40, n_por_grado_ano=8000, seed=13)


def test_estimar_por_grado_recupera_algo_cercano_a_la_verdad(datos_chicos):
    est = estimar_por_grado(datos_chicos["cohortes"])
    assert len(est) == len(GRADOS)
    for _, f in est.iterrows():
        assert abs(f["pd_ttc_hat"] - f["pd_ttc_true"]) < 0.15 * f["pd_ttc_true"] + 0.01
        assert abs(f["rho_hat_mom"] - f["rho_true"]) < 0.15


def test_recuperar_ciclo_da_una_fila_por_ano_y_correlaciona_bien(datos_chicos):
    est = estimar_por_grado(datos_chicos["cohortes"])
    ciclo = recuperar_ciclo(datos_chicos["cohortes"], est)
    assert len(ciclo) == 40
    correlacion = np.corrcoef(ciclo["z_verdadero"], ciclo["z_recuperado"])[0, 1]
    assert correlacion > 0.8


def test_capital_ttc_es_constante_entre_anos(datos_chicos):
    """Por construccion: la PD through-the-cycle no cambia con el ano, asi
    que el capital TTC tiene que ser exactamente el mismo todos los anos."""
    est = estimar_por_grado(datos_chicos["cohortes"])
    capital = capital_pit_vs_ttc(datos_chicos["cohortes"], est)
    assert capital["densidad_rwa_ttc_pct"].nunique() == 1


def test_capital_pit_varia_mas_que_el_ttc(datos_chicos):
    est = estimar_por_grado(datos_chicos["cohortes"])
    capital = capital_pit_vs_ttc(datos_chicos["cohortes"], est)
    assert capital["densidad_rwa_pit_pct"].std() > capital["densidad_rwa_ttc_pct"].std()


def test_capital_sube_en_anos_de_recesion(datos_chicos):
    est = estimar_por_grado(datos_chicos["cohortes"])
    capital = capital_pit_vs_ttc(datos_chicos["cohortes"], est).set_index("anio")
    anos_normales = [a for a in capital.index if a not in ANOS_RECESION]
    for anio in ANOS_RECESION:
        if anio in capital.index:
            assert capital.loc[anio, "densidad_rwa_pit_pct"] > \
                   capital.loc[anos_normales, "densidad_rwa_pit_pct"].median()


def test_sesgo_de_granularidad_devuelve_dos_tamanos_por_grado():
    tabla = sesgo_de_granularidad(seed=14)
    assert set(tabla["tamano"]) == {"chica", "grande"}
    assert len(tabla) == 2 * len(GRADOS)
