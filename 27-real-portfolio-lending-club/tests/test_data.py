from __future__ import annotations

import numpy as np
import pandas as pd

from src import data


def test_only_resolved_36_month_loans_up_to_2015(raw):
    book, report = data.prepare(raw)
    assert book["year"].max() <= data.LAST_YEAR
    assert set(book["loan_status"]) <= data.BAD | data.GOOD
    assert report["unresolved_dropped"] > 0
    expected = ((raw["term"].str.strip() == "36 months")
                & (pd.to_datetime(raw["issue_d"], format="%b-%Y").dt.year <= 2015)
                & raw["loan_status"].isin(data.BAD | data.GOOD)).sum()
    assert len(book) == expected


def test_bad_includes_policy_charge_offs():
    assert "Does not meet the credit policy. Status:Charged Off" in data.BAD
    assert "Does not meet the credit policy. Status:Fully Paid" in data.GOOD
    assert not data.BAD & data.GOOD


def test_lender_pricing_never_enters_a_model():
    for col in ("grade", "sub_grade", "int_rate", "installment"):
        assert col not in data.FEATURES
    post_origination = {"total_pymnt", "recoveries", "last_fico_range_low", "last_pymnt_d",
                        "out_prncp", "debt_settlement_flag", "hardship_flag"}
    assert not post_origination & set(data.FEATURES)


def test_derived_features(book):
    assert (book["credit_history_months"] >= 36).all()       # fake file: 3 to 24 years
    assert book["loan_to_income"].between(0, 5).all()
    assert set(book["home_ownership"]) <= {"MORTGAGE", "RENT", "OWN", "OTHER"}
    assert "missing" in set(book["emp_length"])
    assert book["last_payment_month"].min() == 0               # no payment at all


def test_months_between():
    a = pd.Series(pd.to_datetime(["2012-11-01", "2015-01-01"]))
    b = pd.Series(pd.to_datetime(["2013-02-01", "2015-01-01"]))
    assert data._months_between(a, b).tolist() == [3, 0]


def test_out_of_time_split(book):
    train, valid, test = data.split(book)
    assert train["year"].max() <= data.TRAIN_LAST_YEAR
    assert set(valid["year"]) == {data.VALID_YEAR}
    assert set(test["year"]) == {data.TEST_YEAR}
    assert len(train) + len(valid) + len(test) == len(book)


def test_impute_uses_the_training_median(book):
    train, valid, _ = data.split(book)
    valid = valid.copy()
    valid.loc[valid.index[0], "dti"] = np.nan
    _, filled = data.impute(train, valid)
    assert filled.loc[filled.index[0], "dti"] == train["dti"].median()


def test_sub_grade_rank():
    r = data.sub_grade_rank(pd.Series(["A1", "A5", "B1", "G5"]))
    assert r.tolist() == [1, 5, 6, 35]
