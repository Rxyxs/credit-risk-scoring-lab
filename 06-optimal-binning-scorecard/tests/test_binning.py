"""Tests del binning optimo.

El test que sostiene todo el modulo es el primero: sobre una instancia
chica se enumeran **todas** las particiones posibles y se compara el mejor
IV con el que devuelve la programacion dinamica. Si la DP fuera un
heuristico disfrazado, ahi se nota; sin ese test, "optimo" es una palabra
en un README.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.binning import (
    OptimalBinner, aplicar_binning, binning_arbol, binning_equifrecuente,
    indice_bin, interpretar_iv, woe_iv_tramo,
)


def _iv_de_particion(n, malos, cortes) -> tuple[float, list[float]]:
    """IV total y WOE de cada tramo para una particion dada por `cortes`."""
    total_malos = malos.sum()
    total_buenos = n.sum() - total_malos
    bordes = [0, *cortes, n.size]
    iv_total, woes = 0.0, []
    for a, b in zip(bordes[:-1], bordes[1:]):
        woe, iv = woe_iv_tramo(np.array([n[a:b].sum()]), np.array([malos[a:b].sum()]),
                               total_malos, total_buenos)
        iv_total += float(iv[0])
        woes.append(float(woe[0]))
    return iv_total, woes


def _mejor_por_fuerza_bruta(n, malos, max_bins, min_fraccion, min_eventos, monotono):
    """Enumera todas las particiones en tramos contiguos y devuelve la mejor."""
    m = n.size
    total = n.sum()
    mejor = -np.inf
    for k in range(1, max_bins + 1):
        for cortes in combinations(range(1, m), k - 1):
            bordes = [0, *cortes, m]
            tramos = list(zip(bordes[:-1], bordes[1:]))
            ok = all(
                n[a:b].sum() >= min_fraccion * total
                and malos[a:b].sum() >= min_eventos
                and (n[a:b].sum() - malos[a:b].sum()) >= min_eventos
                for a, b in tramos
            )
            if not ok:
                continue
            iv, woes = _iv_de_particion(n, malos, list(cortes))
            if monotono:
                d = np.diff(woes)
                if not (np.all(d > 0) or np.all(d < 0)):
                    continue
            mejor = max(mejor, iv)
    return mejor


def _instancia(seed=0, m=10, n_por_bin=200):
    """Prebins sinteticos con una relacion ruidosa pero creciente."""
    rng = np.random.default_rng(seed)
    n = np.full(m, float(n_por_bin))
    tasa = np.clip(np.linspace(0.10, 0.45, m) + rng.normal(0, 0.05, m), 0.02, 0.9)
    malos = np.round(n * tasa)
    return n, malos


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_la_dp_iguala_a_la_busqueda_exhaustiva_sin_monotonia(seed):
    n, malos = _instancia(seed)
    binner = OptimalBinner(max_bins=4, min_fraccion=0.05, min_eventos=10, monotono=False)
    valor, tramos = binner._resolver_dp(n, malos, direccion=0)
    mejor = _mejor_por_fuerza_bruta(n, malos, 4, 0.05, 10, monotono=False)
    assert valor == pytest.approx(mejor, rel=1e-12)
    assert tramos[0][0] == 0 and tramos[-1][1] == n.size - 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_la_dp_iguala_a_la_busqueda_exhaustiva_con_monotonia(seed):
    n, malos = _instancia(seed, m=9)
    binner = OptimalBinner(max_bins=4, min_fraccion=0.05, min_eventos=10, monotono=True)
    mejor_dp = max(
        binner._resolver_dp(n, malos, direccion=d)[0] for d in (1, -1)
    )
    mejor = _mejor_por_fuerza_bruta(n, malos, 4, 0.05, 10, monotono=True)
    assert mejor_dp == pytest.approx(mejor, rel=1e-12)


def test_los_tramos_cubren_todo_sin_solaparse():
    n, malos = _instancia(7, m=12)
    _, tramos = OptimalBinner(max_bins=5, min_eventos=10)._resolver_dp(n, malos, 1)
    cubiertos = [i for (a, b) in tramos for i in range(a, b + 1)]
    assert cubiertos == list(range(n.size))


def _datos_reales(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 0.12, n)
    p = 1 / (1 + np.exp(-(-2.2 + 4.5 * x)))
    y = (rng.random(n) < p).astype(float)
    return x, y


def test_respeta_las_restricciones_de_tamano():
    x, y = _datos_reales()
    r = OptimalBinner(max_bins=6, min_fraccion=0.10, min_eventos=50).fit_numerica(x, y, "x")
    assert r.n_bins <= 6
    assert (r.tabla["pct_poblacion"] >= 0.10 - 1e-9).all()
    assert (r.tabla["malos"] >= 50).all()
    assert (r.tabla["n"] - r.tabla["malos"] >= 50).all()


def test_la_monotonia_se_cumple_cuando_se_pide():
    x, y = _datos_reales(seed=5)
    r = OptimalBinner(monotono=True).fit_numerica(x, y, "x")
    d = np.diff(r.tabla["woe"].to_numpy())
    assert r.monotono
    assert np.all(d >= -1e-12) or np.all(d <= 1e-12)


def test_la_dp_no_puede_perder_contra_cortes_equifrecuentes():
    """Con la grilla alineada, la particion equifrecuente es factible para la DP."""
    x, y = _datos_reales(seed=9)
    dp = OptimalBinner(max_bins=6, monotono=False, n_prebins=60).fit_numerica(x, y, "x")
    ef = binning_equifrecuente(x, y, 6, "x")
    assert dp.iv >= ef.iv - 1e-9


def test_woe_e_iv_calculados_a_mano():
    # 100 casos: 40 malos de 200 en el bin, 60 malos de 300 fuera.
    woe, iv = woe_iv_tramo(np.array([200.0]), np.array([40.0]),
                           total_malos=100.0, total_buenos=400.0)
    p_malos = (40 + 0.5) / (100 + 1)
    p_buenos = (160 + 0.5) / (400 + 1)
    esperado = np.log(p_malos / p_buenos)
    assert woe[0] == pytest.approx(esperado)
    assert iv[0] == pytest.approx((p_malos - p_buenos) * esperado)


def test_woe_negativo_donde_hay_menos_malos():
    woe, _ = woe_iv_tramo(np.array([100.0, 100.0]), np.array([5.0, 50.0]), 55.0, 145.0)
    assert woe[0] < 0 < woe[1]


def test_aplicar_binning_devuelve_el_woe_del_tramo_correcto():
    x, y = _datos_reales(seed=11)
    r = OptimalBinner().fit_numerica(x, y, "x")
    woe = aplicar_binning(r, x)
    idx = indice_bin(r, x)
    assert np.allclose(woe, r.tabla["woe"].to_numpy()[idx])
    assert set(np.unique(idx)) <= set(range(r.n_bins))
    # un valor por debajo del minimo cae al primer bin, uno enorme al ultimo
    extremos = aplicar_binning(r, np.array([-1e9, 1e9]))
    assert extremos[0] == r.tabla["woe"].iloc[0]
    assert extremos[1] == r.tabla["woe"].iloc[-1]


def test_categoricas_se_agrupan_por_tasa_de_evento():
    rng = np.random.default_rng(2)
    cats = np.repeat(["a", "b", "c", "d"], 1500)
    tasas = {"a": 0.05, "b": 0.10, "c": 0.30, "d": 0.35}
    y = np.array([rng.random() < tasas[c] for c in cats], dtype=float)
    r = OptimalBinner(max_bins=3, min_eventos=10).fit_categorica(cats, y, "cat")

    assert r.grupos is not None
    orden = [c for grupo in r.grupos for c in grupo]
    assert orden == ["a", "b", "c", "d"]           # ordenadas por tasa de malos
    assert r.tabla["woe"].is_monotonic_increasing


def test_categoria_no_vista_cae_en_el_bin_mas_poblado():
    rng = np.random.default_rng(4)
    cats = np.repeat(["a", "b", "c"], [3000, 500, 500])
    y = (rng.random(4000) < 0.2).astype(float)
    r = OptimalBinner(max_bins=3, min_eventos=10).fit_categorica(cats, y, "cat")
    woe = aplicar_binning(r, np.array(["zzz"]))
    bin_refugio = int(r.tabla["n"].idxmax())
    assert woe[0] == r.tabla["woe"].iloc[bin_refugio]


def test_lectura_del_iv_y_parametros_invalidos():
    assert interpretar_iv(0.01) == "sin poder predictivo"
    assert interpretar_iv(0.05) == "debil"
    assert interpretar_iv(0.20) == "medio"
    assert interpretar_iv(0.40) == "fuerte"
    assert "sospechoso" in interpretar_iv(0.90)

    with pytest.raises(ValueError):
        OptimalBinner(max_bins=1)
    with pytest.raises(ValueError):
        OptimalBinner(min_fraccion=0.9)
    x, y = _datos_reales(n=500)
    with pytest.raises(ValueError):
        OptimalBinner().fit_numerica(x, y[:100], "x")
    with pytest.raises(ValueError):
        OptimalBinner().fit_numerica(x, np.full(x.size, 2.0), "x")


def test_el_arbol_tambien_produce_una_tabla_comparable():
    x, y = _datos_reales(seed=13)
    r = binning_arbol(x, y, 6, nombre="x")
    assert r.n_bins <= 6
    assert set(r.tabla.columns) >= {"woe", "iv", "n", "tasa_mala"}
    assert r.iv > 0
