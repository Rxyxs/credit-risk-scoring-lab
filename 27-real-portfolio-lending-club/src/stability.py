"""Techniques 12 and 22 on the real book: does drift monitoring see what matters?

The monotone-DP scorecard fitted on 2007-2013 scores every quarter of 2014
and 2015. For each quarter: the PSI of the score and of the main variables
against the training population (technique 12's `calculate_psi`), the
realised AUC, and the predicted against the observed default rate. Technique
22 retrains when the realised AUC drops below 0.72 or a PSI goes above 0.20;
the table says in which quarters that rule would have fired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.lab import TRIGGER_MAX_PSI, TRIGGER_MIN_AUC, drift

WATCHED = ["fico", "dti", "log_income", "loan_to_income", "revol_util", "inq_last_6mths"]


def quarterly(card, train: pd.DataFrame, later: pd.DataFrame) -> pd.DataFrame:
    base_score = card.score(train)
    rows = []
    for q, sub in later.groupby("quarter"):
        y = sub["bad"].to_numpy(int)
        pd_pred = card.predict_proba(sub)
        row = {"quarter": q, "n": int(len(sub)), "observed_rate": float(y.mean()),
               "mean_pd": float(pd_pred.mean()), "auc": float(roc_auc_score(y, pd_pred)),
               "psi_score": drift.calculate_psi(base_score, card.score(sub))}
        for v in WATCHED:
            row[f"psi_{v}"] = drift.calculate_psi(train[v].to_numpy(float), sub[v].to_numpy(float))
        psis = [row["psi_score"], *(row[f"psi_{v}"] for v in WATCHED)]
        row["max_psi"] = float(np.max(psis))
        row["fires_auc"] = row["auc"] < TRIGGER_MIN_AUC
        row["fires_psi"] = row["max_psi"] > TRIGGER_MAX_PSI
        rows.append(row)
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> dict:
    return {
        "quarters": int(len(table)),
        "fires_on_auc": int(table["fires_auc"].sum()),
        "fires_on_psi": int(table["fires_psi"].sum()),
        "auc_range": [float(table["auc"].min()), float(table["auc"].max())],
        "max_psi_score": float(table["psi_score"].max()),
        "max_psi_any": float(table["max_psi"].max()),
        "corr_psi_score_vs_auc": float(np.corrcoef(table["psi_score"], table["auc"])[0, 1]),
        "underprediction_pp": [float(100 * (o - p)) for o, p in zip(table["observed_rate"], table["mean_pd"])],
    }
