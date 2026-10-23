from __future__ import annotations

import pytest

from src import data, scoring


@pytest.fixture(scope="module")
def splits(book):
    return data.impute(*data.split(book))


def test_four_binnings_and_their_scorecards(splits):
    results, cards = scoring.binning_comparison(*splits)
    assert set(results) == set(scoring.BINNING_METHODS)
    assert results["dp_monotono"]["non_monotone"] == []
    for r in results.values():
        assert 0 < r["test_2015"]["auc"] < 1
    # fico drives default in the fake file, so it should carry the most IV
    iv = results["dp_monotono"]["iv_by_variable"]
    assert max(iv, key=iv.get) == "fico"


def test_models_audit_and_conformal(splits):
    train, valid, test = splits
    _, cards = scoring.binning_comparison(train, valid, test)
    models, fitted = scoring.model_comparison(train, valid, test, cards["dp_monotono"])
    assert models["gbm_monotono"]["audit"]["variables_with_any_violation"] == 0
    assert "lending_club_sub_grade" in models
    out = scoring.conformal_decisions(fitted, valid, test)
    for d in out.values():
        assert d["approve"] + d["review"] + d["decline"] == pytest.approx(1.0)
        assert 0.0 <= d["coverage"] <= 1.0
