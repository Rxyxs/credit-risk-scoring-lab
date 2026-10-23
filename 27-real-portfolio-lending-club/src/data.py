"""The real book: Lending Club 36-month loans issued 2007-2015.

Source: Kaggle `wordsforthewise/lending-club`, file
`accepted_2007_to_2018Q4.csv.gz` (every loan Lending Club issued through
2018Q4, 151 columns). Only 36-month loans issued up to December 2015 are used,
because they have all reached maturity in the file: their outcome is known.

Only what was known when the loan was approved goes into a model. Lending
Club's own risk assessment (grade, sub-grade, interest rate, and the
instalment, which is computed from the rate) is kept out of the features and
used as a benchmark instead: it is what the lender actually priced with.
Everything recorded after origination (payments, recoveries, last FICO,
hardship and settlement fields) is never read, except the last payment date,
which only the default-timing analysis uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW_FILE = BASE / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
UNRATE_FILE = BASE / "data" / "raw" / "UNRATE.csv"

READ_COLUMNS = [
    "issue_d", "term", "loan_status", "grade", "sub_grade", "int_rate", "loan_amnt",
    "annual_inc", "dti", "fico_range_low", "emp_length", "home_ownership", "purpose",
    "verification_status", "revol_util", "revol_bal", "inq_last_6mths", "delinq_2yrs",
    "pub_rec", "open_acc", "total_acc", "earliest_cr_line", "last_pymnt_d",
]
BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
LAST_YEAR = 2015
TRAIN_LAST_YEAR, VALID_YEAR, TEST_YEAR = 2013, 2014, 2015
TERM_MONTHS = 36

NUMERIC = [
    "fico", "log_income", "dti", "revol_util", "inq_last_6mths", "delinq_2yrs", "pub_rec",
    "open_acc", "total_acc", "credit_history_months", "loan_to_income", "revol_bal_to_income",
    "loan_amnt",
]
CATEGORICAL = ["home_ownership", "purpose", "emp_length", "verification_status"]
FEATURES = NUMERIC + CATEGORICAL
# Expected direction of each variable's effect on default risk, in the format
# HistGradientBoostingClassifier takes (1 up, -1 down, 0 free), as in technique 05.
MONOTONIC = {
    "fico": -1, "log_income": -1, "dti": 1, "revol_util": 1, "inq_last_6mths": 1,
    "delinq_2yrs": 1, "pub_rec": 1, "credit_history_months": -1, "loan_to_income": 1,
    "open_acc": 0, "total_acc": 0, "revol_bal_to_income": 0, "loan_amnt": 0,
}
LENDER_COLUMNS = ["grade", "sub_grade", "int_rate"]   # benchmark only, never a feature
GRADES = list("ABCDEFG")


def read_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    return pd.read_csv(path, usecols=READ_COLUMNS, low_memory=False)


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter to resolved 36-month loans up to LAST_YEAR and build the features.

    Returns the book and a small report of what was dropped and why.
    """
    df = raw[raw["issue_d"].notna()].copy()
    df["issue"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["year"] = df["issue"].dt.year
    df = df[(df["term"].str.strip() == f"{TERM_MONTHS} months") & (df["year"] <= LAST_YEAR)]
    resolved = df["loan_status"].isin(BAD | GOOD)
    report = {"loans_36m_through_last_year": int(len(df)),
              "unresolved_dropped": int((~resolved).sum())}
    df = df[resolved].copy()
    df["bad"] = df["loan_status"].isin(BAD).astype(int)
    df["quarter"] = df["issue"].dt.to_period("Q").astype(str)

    income = df["annual_inc"].clip(lower=1.0)
    df["fico"] = df["fico_range_low"].astype(float)
    df["log_income"] = np.log(income)
    df["dti"] = df["dti"].clip(0, 100)
    df["revol_util"] = df["revol_util"].clip(0, 150)
    df["credit_history_months"] = _months_between(
        pd.to_datetime(df["earliest_cr_line"], format="%b-%Y"), df["issue"]).clip(lower=0)
    df["loan_to_income"] = (df["loan_amnt"] / income).clip(upper=5)
    df["revol_bal_to_income"] = (df["revol_bal"] / income).clip(upper=10)

    df["home_ownership"] = df["home_ownership"].where(
        df["home_ownership"].isin(["MORTGAGE", "RENT", "OWN"]), "OTHER")
    df["emp_length"] = df["emp_length"].fillna("missing")
    for c in ("purpose", "verification_status"):
        df[c] = df[c].fillna("missing")

    # Month of the last payment: the onset of the default for a charged-off
    # loan, the payoff month for one that was repaid. No payment at all is month 0.
    last = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y")
    df["last_payment_month"] = _months_between(df["issue"], last).fillna(0).clip(0, 120).astype(int)

    report["loans"] = int(len(df))
    report["bad_rate"] = float(df["bad"].mean())
    report["missing_share"] = {c: float(df[c].isna().mean()) for c in NUMERIC if df[c].isna().any()}
    keep = ["issue", "year", "quarter", "loan_status", "bad", "last_payment_month",
            *LENDER_COLUMNS, *FEATURES]
    return df[keep].reset_index(drop=True), report


def split(book: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Out of time: fit on 2007-2013, choose on 2014, test on 2015."""
    train = book[book["year"] <= TRAIN_LAST_YEAR].reset_index(drop=True)
    valid = book[book["year"] == VALID_YEAR].reset_index(drop=True)
    test = book[book["year"] == TEST_YEAR].reset_index(drop=True)
    return train, valid, test


def impute(train: pd.DataFrame, *others: pd.DataFrame) -> list[pd.DataFrame]:
    """Fill the few missing numeric values with the training median."""
    medians = train[NUMERIC].median()
    return [d.fillna({c: medians[c] for c in NUMERIC}) for d in (train, *others)]


def sub_grade_rank(sub_grade: pd.Series) -> np.ndarray:
    """A1 -> 1 ... G5 -> 35: Lending Club's own ranking, as a risk score."""
    s = sub_grade.astype(str)
    return np.array([GRADES.index(x[0]) * 5 + int(x[1]) for x in s], dtype=float)


def read_unemployment(path: Path = UNRATE_FILE) -> pd.Series:
    """US unemployment rate (FRED series UNRATE), monthly, in percent."""
    u = pd.read_csv(path, parse_dates=["observation_date"])
    return u.set_index("observation_date")["UNRATE"]
