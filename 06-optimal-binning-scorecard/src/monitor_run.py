"""Corre el monitoreo de estabilidad del scorecard vintage por vintage.

La pregunta operativa que contesta: si este scorecard hubiera entrado en
produccion con las cohortes estables, .en que mes habria sonado la alarma?
Y sobre todo: .habria sonado antes de que la tasa de default lo hiciera
evidente?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generator import VINTAGE_QUIEBRE
from src.fit_scorecard import binear_todo
from src.monitoring import monitoreo_por_vintage, resumen_monitoreo, semaforo
from src.scorecard import ajustar_scorecard, metricas

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
PROC_DIR = BASE / "data" / "processed"
REPORTS_DIR = BASE / "outputs" / "reports"


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PROC_DIR / "train.csv")
    todo = pd.read_csv(RAW_DIR / "applicants.csv")

    y_tr = train["default_12m"].to_numpy(float)
    binnings = binear_todo(train, y_tr, "dp_monotono")
    sc = ajustar_scorecard(train, y_tr, binnings)

    tabla = monitoreo_por_vintage(sc, train, todo)
    tabla.to_csv(REPORTS_DIR / "monitoreo_por_vintage.csv", index=False)
    resumen = resumen_monitoreo(tabla)

    # .Cuanto se degrado el poder discriminante, y no solo la poblacion?
    estables = todo[todo["vintage_idx"] < VINTAGE_QUIEBRE]
    deteriorados = todo[todo["vintage_idx"] >= VINTAGE_QUIEBRE]
    degradacion = {
        "auc_vintages_estables": metricas(
            estables["default_12m"].to_numpy(), sc.predict_proba(estables))["auc"],
        "auc_vintages_deteriorados": metricas(
            deteriorados["default_12m"].to_numpy(), sc.predict_proba(deteriorados))["auc"],
        "pd_predicha_media_estables": float(sc.predict_proba(estables).mean()),
        "tasa_mala_real_estables": float(estables["default_12m"].mean()),
        "pd_predicha_media_deteriorados": float(sc.predict_proba(deteriorados).mean()),
        "tasa_mala_real_deteriorados": float(deteriorados["default_12m"].mean()),
    }
    degradacion["sesgo_de_calibracion_deteriorados_pp"] = 100.0 * (
        degradacion["pd_predicha_media_deteriorados"]
        - degradacion["tasa_mala_real_deteriorados"]
    )

    reporte = {
        "vintage_quiebre_real": VINTAGE_QUIEBRE,
        "resumen": resumen,
        "degradacion": degradacion,
        "monitoreo": tabla.to_dict(orient="records"),
    }
    (REPORTS_DIR / "monitoring_results.json").write_text(json.dumps(reporte, indent=2))

    cols_csi = [c for c in tabla.columns if c.startswith("csi_")]
    top_csi = tabla[cols_csi].max().sort_values(ascending=False).head(3)

    print(f"Referencia: {len(train):,} solicitudes de las cohortes estables")
    print(f"Quiebre real de poblacion: vintage {VINTAGE_QUIEBRE}\n")
    print(f"{'vintage':>8} {'n':>6} {'tasa mala':>10} {'score medio':>12} "
          f"{'PSI':>7}  estado")
    for _, f in tabla.iterrows():
        print(f"{int(f['vintage_idx']):>8} {int(f['n']):>6} {f['tasa_mala']:>10.2%} "
              f"{f['score_medio']:>12.1f} {f['psi_score']:>7.4f}  {f['estado_psi']}")

    print(f"\nPrimer vintage en alerta (PSI >= 0.10): {resumen['primer_vintage_en_alerta']}")
    print(f"Primer vintage critico  (PSI >= 0.25): {resumen['primer_vintage_critico']}")
    print(f"Variable mas inestable por CSI: {resumen['variable_mas_inestable']} "
          f"({resumen['csi_maximo']:.4f}, {semaforo(resumen['csi_maximo'])})")
    print("Top 3 por CSI maximo:")
    for var, valor in top_csi.items():
        print(f"  {var.replace('csi_', ''):<26} {valor:.4f}")

    d = degradacion
    print(f"\nAUC {d['auc_vintages_estables']:.4f} (estables) -> "
          f"{d['auc_vintages_deteriorados']:.4f} (deteriorados)")
    print(f"PD media predicha vs real, cohortes deterioradas: "
          f"{d['pd_predicha_media_deteriorados']:.2%} vs "
          f"{d['tasa_mala_real_deteriorados']:.2%} "
          f"({d['sesgo_de_calibracion_deteriorados_pp']:+.2f} pp)")


if __name__ == "__main__":
    main()
