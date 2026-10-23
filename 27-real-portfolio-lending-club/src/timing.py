"""Technique 03's question on the real book: when does a 36-month loan default?

Technique 03 reports that 50.5% of lifetime risk arrives after month 12, which
is what makes a lifetime (IFRS 9 stage 2) provision so much larger than a
12-month one. Here the default date is the month of the last payment of a
charged-off loan (the onset of the delinquency; Lending Club charges off about
five months later). Two readings, because real loans also leave early by
prepaying:

- Kaplan-Meier with prepayment as censoring (technique 03's own estimator):
  the default curve of a loan that stays on the book.
- The plain share of defaulted loans whose last payment came after month 12:
  what actually happened to the book, prepayments included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TERM_MONTHS
from src.lab import survival

NINETY_DAYS = 3        # a 90-days-past-due definition dates each default three months later


def default_timing(book: pd.DataFrame, shift: int = 0) -> dict:
    """Default curve and the share of lifetime risk after month 12."""
    bad = book["bad"].to_numpy(int)
    month = book["last_payment_month"].to_numpy(float)
    # Dated at 90 days past due, a loan can still default up to `shift`
    # months after its last scheduled instalment, so the horizon grows with it.
    # A loan repaid early leaves the book (censored) at its payoff month; one
    # repaid on schedule has survived the whole horizon, since it can no
    # longer default.
    horizon = TERM_MONTHS + shift
    survived = np.where(month >= TERM_MONTHS - 1, horizon, month)
    duration = np.where(bad == 1, month + shift, survived).clip(0, horizon)
    km = survival.kaplan_meier(duration, bad, times=np.arange(0, horizon + 1))
    cum = km["cum_default"].to_numpy()
    pd12, pd_life = float(cum[12]), float(cum[horizon])
    after = (month[bad == 1] + shift) > 12
    return {
        "shift_months": shift,
        "horizon_months": horizon,
        "km_pd_12m": pd12,
        "km_pd_lifetime": pd_life,
        "km_lifetime_over_12m": pd_life / pd12 if pd12 > 0 else None,
        "km_share_after_12m": 1.0 - pd12 / pd_life if pd_life > 0 else None,
        "observed_share_of_defaults_after_12m": float(after.mean()),
        "median_default_month": float(np.median(month[bad == 1] + shift)),
        "curve": [round(float(c), 5) for c in cum],
    }


def by_grade(book: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, sub in book.groupby("grade"):
        t = default_timing(sub)
        rows.append({"grade": g, "n": int(len(sub)), "bad_rate": float(sub["bad"].mean()),
                     "km_pd_12m": t["km_pd_12m"], "km_pd_lifetime": t["km_pd_lifetime"],
                     "km_share_after_12m": t["km_share_after_12m"]})
    return pd.DataFrame(rows)
