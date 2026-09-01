import numpy as np

from src.cleaning import enforce_business_rules, run_cleaning_pipeline
from src.data_generator import generate_applicants


def test_enforce_business_rules_invalidates_impossible_values():
    df = generate_applicants(n=200, seed=3)
    df.loc[0, "renta_liquida"] = -100
    df.loc[1, "dti"] = -0.5
    out = enforce_business_rules(df)
    assert np.isnan(out.loc[0, "renta_liquida"])
    assert np.isnan(out.loc[1, "dti"])


def test_run_cleaning_pipeline_leaves_no_nans():
    df = generate_applicants(n=3000, seed=8)
    cleaned = run_cleaning_pipeline(df)
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) == len(df)


def test_split_is_stratified():
    df = generate_applicants(n=5000, seed=9)
    cleaned = run_cleaning_pipeline(df)
    rates = cleaned.groupby("split")["default_12m"].mean()
    assert abs(rates["train"] - rates["test"]) < 0.02


def test_dti_is_non_negative_and_finite():
    df = generate_applicants(n=2000, seed=10)
    cleaned = run_cleaning_pipeline(df)
    assert (cleaned["dti"] >= 0).all()
    assert np.isfinite(cleaned["dti"]).all()
