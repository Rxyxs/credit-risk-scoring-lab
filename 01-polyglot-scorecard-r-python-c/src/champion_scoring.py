"""Scoring en vivo del modelo Champion (scorecard R, WOE + puntos PDO) para
un solicitante nuevo, usando el motor compilado en C (`ScoreEngine`) como
hot-path -- el mismo binning que aplico R en `01_woe_binning.R`, pero
reimplementado en Python puro (sin R en el runtime de produccion) a partir
de los artefactos que R ya dejo escritos: `woe_bin_edges.json` (cortes de
bin) y `scorecard_points_table.csv` + `scorecard_meta.json` (puntos y
parametros PDO). No reentrena ni reajusta nada, solo aplica el binning
congelado en el entrenamiento -- igual que en produccion real, donde el
binning de un scorecard aprobado no cambia entre corridas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ctypes_bridge import ScoreEngine

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "outputs" / "reports"


class ChampionScorer:
    """Encapsula binning WOE congelado + motor C para scorear un solicitante
    (o un batch) sin pasar por R. Se levanta una sola vez al iniciar el
    servicio (carga DLL + tablas), y luego cada request solo hace lookups."""

    def __init__(self):
        self.bin_edges = json.loads((REPORTS_DIR / "woe_bin_edges.json").read_text())
        self.meta = json.loads((REPORTS_DIR / "scorecard_meta.json").read_text())
        self.feature_order = self.meta["feature_order"]
        self.max_bins = self.meta["max_bins"]
        self.base_points = self.meta["base_points"]
        self.factor = self.meta["factor"]
        self.offset = self.meta["offset"]

        points_df = pd.read_csv(REPORTS_DIR / "scorecard_points_table.csv")
        self.points_table = np.zeros((len(self.feature_order), self.max_bins), dtype=np.float64)
        for i, feat in enumerate(self.feature_order):
            rows = points_df[points_df["feature"] == feat]
            for _, row in rows.iterrows():
                self.points_table[i, int(row["bin_id"])] = row["points"]

        self.engine = ScoreEngine()

    def _bin_one_feature(self, feature: str, value) -> int:
        spec = self.bin_edges[feature]
        if spec["type"] == "numeric":
            breaks = [
                -np.inf if b == "-Inf" else np.inf if b == "Inf" else float(b)
                for b in spec["breaks"]
            ]
            for bin_id, (lo, hi) in enumerate(zip(breaks[:-1], breaks[1:])):
                if lo < value <= hi or (bin_id == 0 and value <= hi):
                    return bin_id
            return len(breaks) - 2  # por seguridad, ultimo bin
        else:
            levels = spec["levels"]
            if value in levels:
                return levels.index(value)
            return len(levels) - 1  # categoria no vista -> ultimo bin, igual que en R

    def bin_indices_for(self, features: dict) -> np.ndarray:
        return np.array(
            [[self._bin_one_feature(feat, features[feat]) for feat in self.feature_order]],
            dtype=np.int32,
        )

    def score_one(self, features: dict) -> dict:
        bin_indices = self.bin_indices_for(features)
        score = float(self.engine.score_batch(bin_indices, self.points_table, self.base_points)[0])
        log_odds_good_bad = (score - self.offset) / self.factor
        pd_estimate = 1.0 / (1.0 + np.exp(log_odds_good_bad))
        return {"score": score, "pd_estimate": float(pd_estimate)}
