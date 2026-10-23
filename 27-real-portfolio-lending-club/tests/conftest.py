"""A small file shaped like Lending Club's, so the tests run without the data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _month(year: int, month: int) -> str:
    year += (month - 1) // 12
    return f"{MONTHS[(month - 1) % 12]}-{year}"


def fake_raw(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year = rng.choice(np.arange(2010, 2017), n)
    month = rng.integers(1, 13, n)
    fico = rng.integers(660, 800, n)
    dti = rng.uniform(0, 35, n)
    risk = -2.0 - 0.02 * (fico - 700) + 0.03 * (dti - 15)
    bad = rng.random(n) < 1 / (1 + np.exp(-risk))
    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    status[rng.random(n) < 0.02] = "Current"
    term = np.where(rng.random(n) < 0.8, " 36 months", " 60 months")
    pay_month = np.where(bad, rng.integers(0, 30, n), rng.integers(10, 37, n))
    grade_idx = np.clip(((800 - fico) / 22).astype(int), 0, 6)
    return pd.DataFrame({
        "issue_d": [_month(y, m) for y, m in zip(year, month)],
        "term": term,
        "loan_status": status,
        "grade": [ "ABCDEFG"[g] for g in grade_idx],
        "sub_grade": [f"{'ABCDEFG'[g]}{rng.integers(1, 6)}" for g in grade_idx],
        "int_rate": 6 + 3 * grade_idx + rng.uniform(0, 1, n),
        "loan_amnt": rng.integers(1000, 35000, n).astype(float),
        "annual_inc": rng.lognormal(11, 0.5, n),
        "dti": dti,
        "fico_range_low": fico.astype(float),
        "emp_length": rng.choice(["< 1 year", "5 years", "10+ years", None], n),
        "home_ownership": rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], n),
        "verification_status": rng.choice(["Verified", "Not Verified"], n),
        "revol_util": rng.uniform(0, 100, n),
        "revol_bal": rng.uniform(0, 30000, n),
        "inq_last_6mths": rng.integers(0, 5, n).astype(float),
        "delinq_2yrs": rng.integers(0, 3, n).astype(float),
        "pub_rec": rng.integers(0, 2, n).astype(float),
        "open_acc": rng.integers(2, 25, n).astype(float),
        "total_acc": rng.integers(5, 50, n).astype(float),
        "earliest_cr_line": [_month(y - rng.integers(3, 25), m) for y, m in zip(year, month)],
        "last_pymnt_d": [None if (b and p == 0) else _month(y, m + p)
                         for y, m, p, b in zip(year, month, pay_month, bad)],
    })


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return fake_raw()


@pytest.fixture(scope="session")
def book(raw):
    from src.data import prepare
    return prepare(raw)[0]
