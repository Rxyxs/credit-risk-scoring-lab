"""Technique 10 on the real book: asset correlation from vintage default rates.

Each Lending Club grade (A-G) is followed one issue quarter at a time; a
cohort's default rate is the share of its 36-month loans that were charged
off. Technique 10's method of moments turns the spread of those rates across
quarters into an asset correlation rho, with the cohort sizes (from tens of
loans in 2007 to tens of thousands in 2015), and without them for comparison.

Two cautions shape the reading:

- The rate is a 36-month (lifetime) default rate. Basel's correlation formulas
  are written for a one-year PD, so they are evaluated at the annualised rate
  1 - (1 - PD_36)^(1/3). Lending Club loans are unsecured consumer credit:
  the matching formula is the one for "other retail" exposures (between 0.03
  and 0.16); technique 10 uses the corporate one (0.12 to 0.24).
- The one-factor model assumes a grade means the same risk every quarter.
  Lending Club re-drew its grades and widened its credit box over these
  years, so part of the spread is a trend, not a cycle. The estimate is
  repeated after removing a linear trend from each grade's series, as a
  bound on how much of rho that trend explains.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import GRADES
from src.lab import correlation, vasicek

MIN_COHORT = 30


def cohorts(book: pd.DataFrame, min_cohort: int = MIN_COHORT) -> pd.DataFrame:
    c = (book.groupby(["grade", "quarter"])["bad"].agg(n="size", defaults="sum").reset_index())
    c["rate"] = c["defaults"] / c["n"]
    return c[c["n"] >= min_cohort].reset_index(drop=True)


def basel_retail_correlation(pd_1y) -> np.ndarray:
    """Basel IRB correlation for other retail exposures (between 0.03 and 0.16)."""
    pd_1y = np.asarray(pd_1y, dtype=float)
    w = (1.0 - np.exp(-35.0 * pd_1y)) / (1.0 - np.exp(-35.0))
    return 0.03 * w + 0.16 * (1.0 - w)


def annualise(pd_36: float, years: float = 3.0) -> float:
    return 1.0 - (1.0 - pd_36) ** (1.0 / years)


def _detrended_variance(rates: np.ndarray) -> float:
    t = np.arange(len(rates), dtype=float)
    resid = rates - np.polyval(np.polyfit(t, rates, 1), t)
    return float(np.sum(resid**2) / (len(rates) - 2))


def estimate(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GRADES:
        sub = c[c["grade"] == g].sort_values("quarter")
        if len(sub) < 8:
            continue
        rates, n = sub["rate"].to_numpy(float), sub["n"].to_numpy(float)
        pd_hat = float(rates.mean())
        var_obs = float(rates.var(ddof=1))
        var_detr = _detrended_variance(rates)
        binomial = float(pd_hat * (1 - pd_hat) * np.mean(1 / n))
        pd_1y = annualise(pd_hat)
        rows.append({
            "grade": g, "quarters": int(len(sub)), "loans": int(n.sum()),
            "first_quarter": sub["quarter"].iloc[0], "last_quarter": sub["quarter"].iloc[-1],
            "pd_36m": pd_hat, "pd_1y": pd_1y,
            "share_of_variance_binomial": min(binomial / var_obs, 1.0),
            "rho_mom_with_n": correlation.mom_asset_correlation(pd_hat, var_obs, n_por_cohorte=n),
            "rho_mom_without_n": correlation.mom_asset_correlation(pd_hat, var_obs),
            "rho_asrf_limit": correlation.asrf_limit_correlation(rates),
            "rho_mom_with_n_detrended": correlation.mom_asset_correlation(pd_hat, var_detr, n_por_cohorte=n),
            "rho_basel_retail": float(basel_retail_correlation(pd_1y)),
            "rho_basel_corporate": float(vasicek.basel_asset_correlation(pd_1y)),
        })
    return pd.DataFrame(rows)


def portfolio_series(book: pd.DataFrame, unrate: pd.Series) -> pd.DataFrame:
    """Default rate by issue quarter (all grades) next to unemployment.

    Unemployment is averaged over each cohort's first 18 months, the part of
    a 36-month loan's life where most defaults start.
    """
    q = book.groupby("quarter").agg(n=("bad", "size"), rate=("bad", "mean")).reset_index()
    starts = pd.PeriodIndex(q["quarter"], freq="Q").start_time
    q["unemployment_first_18m"] = [float(unrate[(unrate.index >= s) & (unrate.index < s + pd.DateOffset(months=18))].mean())
                                   for s in starts]
    q["grade_mix_a_b_share"] = book.groupby("quarter")["grade"].apply(lambda s: s.isin(["A", "B"]).mean()).to_numpy()
    return q
