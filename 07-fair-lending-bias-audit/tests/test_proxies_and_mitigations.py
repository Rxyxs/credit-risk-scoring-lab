"""Tests de la deteccion de proxies, de las mitigaciones y del simulador."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_generator import (
    GRUPO_PROTEGIDO, GRUPO_REFERENCIA, SECTORES, generate_applicants,
)
from src.mitigations import (
    aplicar_umbrales_por_grupo, frontera_equidad_utilidad, pesos_reponderacion,
    resultado_economico, umbrales_por_grupo,
)
from src.proxy_analysis import fuerza_de_proxy, poder_de_reconstruccion, resumen_proxies


def _datos_con_proxy(n=6000, seed=0):
    """Un proxy puro, un factor de riesgo puro, y una variable neutra."""
    rng = np.random.default_rng(seed)
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    es_f = grupo == "F"
    return pd.DataFrame({
        "genero": grupo,
        # proxy: separa casi perfecto por grupo, no afecta el riesgo
        "proxy": rng.normal(0, 1, n) + es_f * 2.5,
        # riesgo: no tiene relacion con el grupo
        "riesgo": rng.normal(0, 1, n),
        "neutra": rng.normal(0, 1, n),
        "default_12m": (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 1.2 * rng.normal(0, 1, n))))).astype(int),
    })


# ------------------------------------------------------------- proxies
def test_la_deteccion_separa_proxy_de_factor_de_riesgo():
    rng = np.random.default_rng(1)
    n = 8000
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    proxy = rng.normal(0, 1, n) + (grupo == "F") * 2.5
    riesgo = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.2 + 1.4 * riesgo)))).astype(int)
    df = pd.DataFrame({"genero": grupo, "proxy": proxy, "riesgo": riesgo,
                       "default_12m": y})

    tabla = fuerza_de_proxy(df, ["proxy", "riesgo"], "genero", "F")
    fila_proxy = tabla.set_index("feature").loc["proxy"]
    fila_riesgo = tabla.set_index("feature").loc["riesgo"]

    assert fila_proxy["auc_predice_grupo"] > 0.85
    assert fila_riesgo["auc_predice_grupo"] < 0.55
    assert fila_proxy["razon_proxy"] > fila_riesgo["razon_proxy"] * 10
    assert bool(fila_proxy["es_proxy_sospechoso"])
    assert not bool(fila_riesgo["es_proxy_sospechoso"])
    assert resumen_proxies(tabla)["feature_con_mayor_razon_proxy"] == "proxy"


def test_sin_proxies_la_reconstruccion_es_azarosa():
    rng = np.random.default_rng(2)
    n = 4000
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    X = rng.normal(0, 1, (n, 4))              # ninguna columna sabe del grupo
    r = poder_de_reconstruccion(X, (grupo == "F").astype(int), cv=3)
    assert r["auc_reconstruccion_grupo"] < 0.56
    assert "casi no permiten inferir" in r["interpretacion"]


def test_con_un_proxy_fuerte_la_reconstruccion_es_alta():
    rng = np.random.default_rng(3)
    n = 4000
    es_f = rng.random(n) < 0.5
    X = np.column_stack([rng.normal(0, 1, n) + es_f * 2.5, rng.normal(0, 1, n)])
    r = poder_de_reconstruccion(X, es_f.astype(int), cv=3)
    assert r["auc_reconstruccion_grupo"] > 0.85
    assert "practicamente inferible" in r["interpretacion"]


def test_las_categoricas_tambien_se_evaluan():
    df = _datos_con_proxy()
    df["sector"] = np.where(df["genero"] == "F",
                            np.random.default_rng(4).choice(["salud", "retail"], len(df),
                                                            p=[0.8, 0.2]),
                            np.random.default_rng(5).choice(["salud", "retail"], len(df),
                                                            p=[0.2, 0.8]))
    tabla = fuerza_de_proxy(df, ["sector", "riesgo"], "genero", "F")
    assert tabla.set_index("feature").loc["sector", "auc_predice_grupo"] > 0.7


# ---------------------------------------------------------- mitigaciones
def test_la_reponderacion_iguala_la_tasa_de_malos_ponderada():
    rng = np.random.default_rng(6)
    n = 5000
    g = (rng.random(n) < 0.4).astype(int)
    # el grupo 1 tiene el doble de tasa mala
    y = (rng.random(n) < np.where(g == 1, 0.30, 0.15)).astype(int)
    w = pesos_reponderacion(g, y)

    tasa_pond = {
        gv: float(np.average(y[g == gv], weights=w[g == gv])) for gv in (0, 1)
    }
    assert tasa_pond[0] == pytest.approx(tasa_pond[1], abs=1e-9)
    assert w.sum() == pytest.approx(n, rel=1e-9)
    assert np.all(w > 0)


def test_la_reponderacion_no_hace_nada_si_ya_hay_independencia():
    rng = np.random.default_rng(7)
    n = 4000
    g = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < 0.2).astype(int)      # tasa igual en ambos grupos
    w = pesos_reponderacion(g, y)
    assert np.allclose(w, 1.0, atol=0.1)


def test_umbrales_por_grupo_igualan_la_tasa_de_aprobacion():
    rng = np.random.default_rng(8)
    n = 8000
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    pd_pred = np.clip(rng.beta(2, 8, n) + (grupo == "F") * 0.10, 0.001, 0.999)

    umbrales = umbrales_por_grupo(pd_pred, grupo, 0.75)
    aprobado = aplicar_umbrales_por_grupo(pd_pred, grupo, umbrales)
    tasa_f = aprobado[grupo == "F"].mean()
    tasa_m = aprobado[grupo == "M"].mean()

    assert tasa_f == pytest.approx(0.75, abs=0.01)
    assert tasa_m == pytest.approx(0.75, abs=0.01)
    # el grupo con PD mas alta necesita un umbral mas laxo
    assert umbrales["F"] > umbrales["M"]


def test_economia_calculada_a_mano():
    y = np.array([0, 1, 0])
    aprobado = np.array([1, 1, 0])
    monto = np.array([1e6, 2e6, 3e6])
    r = resultado_economico(y, aprobado, monto, lgd=0.45, margen=0.07)
    assert r["n_aprobados"] == 2
    assert r["tasa_mala_aprobados"] == pytest.approx(0.5)
    assert r["utilidad_clp"] == pytest.approx(1e6 * 0.07 - 2e6 * 0.45)


def test_la_frontera_recorre_las_tasas_pedidas():
    rng = np.random.default_rng(9)
    n = 5000
    grupo = np.where(rng.random(n) < 0.5, "F", "M")
    pd_pred = np.clip(rng.beta(2, 8, n) + (grupo == "F") * 0.05, 0.001, 0.999)
    y = (rng.random(n) < pd_pred).astype(int)
    monto = rng.uniform(1e6, 8e6, n)

    fr = frontera_equidad_utilidad(y, pd_pred, grupo, monto, "F", "M",
                                   tasas=np.array([0.6, 0.8, 0.95]))
    assert len(fr) == 3
    assert np.allclose(fr["tasa_aprobacion"], [0.6, 0.8, 0.95], atol=0.01)
    # aprobar mas gente acerca las tasas entre grupos
    assert fr["ratio_impacto_adverso"].iloc[-1] > fr["ratio_impacto_adverso"].iloc[0]


# --------------------------------------------------------------- datos
def test_el_genero_no_entra_en_el_proceso_generador():
    """Controlando por los factores de riesgo, la PD verdadera no depende del grupo."""
    df = generate_applicants(n=12_000, seed=11)
    df["estrato"] = (pd.qcut(df["log_renta"], 4, labels=False).astype(str) + "_"
                     + pd.qcut(df["dti"], 4, labels=False).astype(str) + "_"
                     + df["n_moras_12m"].clip(0, 2).astype(str))
    brechas = []
    for _, g in df.groupby("estrato"):
        f = g.loc[g["genero"] == GRUPO_PROTEGIDO, "pd_verdadera"]
        m = g.loc[g["genero"] == GRUPO_REFERENCIA, "pd_verdadera"]
        if len(f) > 30 and len(m) > 30:
            brechas.append(f.mean() - m.mean())
    assert abs(float(np.mean(brechas))) < 0.01


def test_hay_brecha_bruta_pero_por_factores_legitimos():
    df = generate_applicants(n=12_000, seed=12)
    f = df[df["genero"] == GRUPO_PROTEGIDO]
    m = df[df["genero"] == GRUPO_REFERENCIA]
    assert f["default_12m"].mean() > m["default_12m"].mean()
    assert f["renta_liquida"].median() < m["renta_liquida"].median()
    assert f["antiguedad_laboral_meses"].median() < m["antiguedad_laboral_meses"].median()


def test_el_sector_esta_segregado_pero_no_es_el_gran_factor_de_riesgo():
    df = generate_applicants(n=12_000, seed=13)
    comp = df.groupby("sector").agg(
        pct_f=("genero", lambda s: (s == GRUPO_PROTEGIDO).mean()),
        default=("default_12m", "mean"),
    )
    assert comp["pct_f"].max() - comp["pct_f"].min() > 0.5      # fuerte segregacion
    assert comp["default"].max() - comp["default"].min() < 0.08  # poca senal de riesgo
    assert set(comp.index) == set(SECTORES)
