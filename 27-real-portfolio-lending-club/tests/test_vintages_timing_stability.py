from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import stability, timing, vintages


def test_basel_retail_correlation_bounds():
    assert vintages.basel_retail_correlation(1e-6) == pytest.approx(0.16, abs=1e-3)
    assert vintages.basel_retail_correlation(0.5) == pytest.approx(0.03, abs=1e-3)


def test_annualise():
    assert vintages.annualise(1 - 0.9**3) == pytest.approx(0.1)


def _cohorts(rho: float, trend: float = 0.0, quarters: int = 34, n: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    rows = []
    for g, pd_ttc in (("B", 0.12), ("D", 0.22)):
        for t in range(quarters):
            thr = norm.ppf(pd_ttc) + trend * (t - quarters / 2)
            p = norm.cdf((thr - np.sqrt(rho) * rng.normal()) / np.sqrt(1 - rho))
            d = rng.binomial(n, p)
            rows.append({"grade": g, "quarter": f"{2007 + t // 4}Q{t % 4 + 1}", "n": n,
                         "defaults": d, "rate": d / n})
    return pd.DataFrame(rows)


def test_without_n_always_reads_more_correlation():
    est = vintages.estimate(_cohorts(rho=0.03))
    assert (est["rho_mom_without_n"] > est["rho_mom_with_n"]).all()


def test_a_trend_reads_as_correlation_until_it_is_removed():
    flat = vintages.estimate(_cohorts(rho=0.02))
    trended = vintages.estimate(_cohorts(rho=0.02, trend=0.02))
    assert (trended["rho_mom_with_n"] > flat["rho_mom_with_n"] + 0.02).all()
    assert (trended["rho_mom_with_n_detrended"] < trended["rho_mom_with_n"]).all()


def test_small_cohorts_are_dropped(book):
    c = vintages.cohorts(book, min_cohort=30)
    assert (c["n"] >= 30).all()


def _timing_book(default_month: int, n: int = 200):
    bad = np.r_[np.ones(n // 2, int), np.zeros(n // 2, int)]
    month = np.r_[np.full(n // 2, default_month), np.full(n // 2, 36)]
    return pd.DataFrame({"bad": bad, "last_payment_month": month})


def test_share_after_month_12():
    early = timing.default_timing(_timing_book(6))
    late = timing.default_timing(_timing_book(20))
    assert early["km_share_after_12m"] == pytest.approx(0.0)
    assert late["km_share_after_12m"] == pytest.approx(1.0)
    assert late["observed_share_of_defaults_after_12m"] == 1.0


def test_ninety_day_dating_extends_the_horizon():
    t = timing.default_timing(_timing_book(35), shift=timing.NINETY_DAYS)
    assert t["horizon_months"] == 39
    assert len(t["curve"]) == 40
    assert np.all(np.diff(t["curve"]) >= 0)
    # Half the book defaults, the other half matures: lifetime PD is one half,
    # not one (loans repaid on schedule stay in the risk set to the end).
    assert t["km_pd_lifetime"] == pytest.approx(0.5)


class _ConstantCard:
    """Scores every loan by FICO; enough to exercise the quarterly table."""

    def score(self, df):
        return df["fico"].to_numpy(float)

    def predict_proba(self, df):
        return 1 / (1 + np.exp((df["fico"].to_numpy(float) - 640) / 40))


def test_quarterly_stability_flags(book):
    train = book[book["year"] <= 2013]
    later = book[book["year"] >= 2014]
    table = stability.quarterly(_ConstantCard(), train, later)
    assert set(table["quarter"]) == set(later["quarter"])
    assert (table["fires_auc"] == (table["auc"] < 0.72)).all()
    assert (table["fires_psi"] == (table["max_psi"] > 0.20)).all()
    s = stability.summary(table)
    assert s["quarters"] == len(table)
