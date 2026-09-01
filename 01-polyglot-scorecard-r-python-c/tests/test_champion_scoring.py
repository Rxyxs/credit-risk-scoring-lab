import pandas as pd
import pytest

from src.champion_scoring import ChampionScorer


@pytest.fixture(scope="module")
def scorer():
    try:
        return ChampionScorer()
    except FileNotFoundError:
        pytest.skip("score_engine.dll no compilada -- correr powershell -File c/build.ps1 primero")


def test_matches_r_computed_score_on_sample(scorer):
    clean = pd.read_csv("data/processed/applicants_clean.csv")
    scored = pd.read_csv("data/processed/applicants_scored.csv")
    sample = clean.sample(20, random_state=7)

    for _, row in sample.iterrows():
        features = {f: row[f] for f in scorer.feature_order}
        result = scorer.score_one(features)
        r_score = scored.loc[scored.applicant_id == row.applicant_id, "score"].iloc[0]
        assert abs(result["score"] - r_score) < 1e-6


def test_unseen_category_falls_back_to_last_bin(scorer):
    features = {
        "tipo_contrato": "no_existe",
        "n_morosidad_reportes": 0,
        "antiguedad_laboral_meses": 24,
        "renta_liquida": 700_000,
        "dti": 0.5,
    }
    result = scorer.score_one(features)
    assert isinstance(result["score"], float)
    assert 0.0 <= result["pd_estimate"] <= 1.0
