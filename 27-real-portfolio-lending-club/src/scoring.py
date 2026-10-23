"""Techniques 01, 05 and 06 on the real book.

- 06: the four binning methods (monotone DP, free DP, equal frequency, tree),
  each turned into a WOE scorecard, scored out of time.
- 01 and 05: the monotone-DP scorecard against gradient boosting with and
  without monotonic constraints (technique 05's own settings), and against
  Lending Club's sub-grade; the counterfactual monotonicity audit of 05.
- 05: the Mondrian conformal decision (approve / review / decline) at
  alpha = 0.10, calibrated on 2014 and tested on 2015, and calibrated and
  tested inside 2015.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data import CATEGORICAL, FEATURES, MONOTONIC, NUMERIC, sub_grade_rank
from src.lab import audit, binning, conformal, policy, scorecard

MAX_BINS = 6                       # technique 06's setting
BINNING_METHODS = ("dp_monotono", "dp_libre", "equifrecuente", "arbol")
# Technique 05's gradient boosting settings, unchanged.
PARAMS_GBM = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  random_state=42)
ALPHA = 0.10


def metrics(y: np.ndarray, pd_pred: np.ndarray) -> dict:
    out = scorecard.metricas(np.asarray(y), np.asarray(pd_pred, dtype=float))
    out["mean_pd"] = float(np.mean(pd_pred))
    out["observed_rate"] = float(np.mean(y))
    return out


def bin_all(train: pd.DataFrame, method: str) -> dict:
    """Technique 06's `binear_todo`, on this book's variables."""
    y = train["bad"].to_numpy(float)
    out = {}
    for var in NUMERIC:
        x = train[var].to_numpy(float)
        if method == "dp_monotono":
            out[var] = binning.OptimalBinner(max_bins=MAX_BINS).fit_numerica(x, y, var)
        elif method == "dp_libre":
            out[var] = binning.OptimalBinner(max_bins=MAX_BINS, monotono=False).fit_numerica(x, y, var)
        elif method == "equifrecuente":
            out[var] = binning.binning_equifrecuente(x, y, MAX_BINS, var)
        elif method == "arbol":
            out[var] = binning.binning_arbol(x, y, MAX_BINS, nombre=var)
        else:
            raise ValueError(f"unknown method: {method}")
    for var in CATEGORICAL:
        out[var] = binning.OptimalBinner(max_bins=MAX_BINS).fit_categorica(train[var].to_numpy(), y, var)
    return out


def binning_comparison(train, valid, test) -> tuple[dict, dict]:
    results, cards = {}, {}
    y_tr = train["bad"].to_numpy(float)
    for method in BINNING_METHODS:
        bins = bin_all(train, method)
        card = scorecard.ajustar_scorecard(train, y_tr, bins)
        cards[method] = card
        numeric_bins = {v: bins[v] for v in NUMERIC}
        results[method] = {
            "total_iv": float(sum(r.iv for r in bins.values())),
            "numeric_iv": float(sum(r.iv for r in numeric_bins.values())),
            "non_monotone": [v for v, r in numeric_bins.items() if not r.monotono],
            "iv_by_variable": {v: round(float(r.iv), 4) for v, r in bins.items()},
            "valid_2014": metrics(valid["bad"], card.predict_proba(valid)),
            "test_2015": metrics(test["bad"], card.predict_proba(test)),
        }
    return results, cards


class Encoder:
    """Numeric columns as they are, categoricals as integer codes learned on train."""

    def __init__(self, train: pd.DataFrame):
        self.codes = {c: {v: i for i, v in enumerate(sorted(train[c].astype(str).unique()))}
                      for c in CATEGORICAL}

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        cols = [df[c].to_numpy(float) for c in NUMERIC]
        for c in CATEGORICAL:
            m = self.codes[c]
            cols.append(df[c].astype(str).map(m).fillna(len(m)).to_numpy(float))
        return np.column_stack(cols)


def fit_gbm(X: np.ndarray, y: np.ndarray, monotone: bool) -> HistGradientBoostingClassifier:
    cat_mask = [f in CATEGORICAL for f in FEATURES]
    cst = [MONOTONIC.get(f, 0) for f in FEATURES] if monotone else None
    model = HistGradientBoostingClassifier(categorical_features=cat_mask, monotonic_cst=cst, **PARAMS_GBM)
    return model.fit(X, y)


def monotonicity_audit(model, X: np.ndarray, sample: int = 2000) -> dict:
    """Technique 05's counterfactual audit. It reads the column order from its
    module-level FEATURES, so point that at this book's columns first."""
    audit.FEATURES = FEATURES
    rows = [audit.violaciones_por_feature(model, X, f, d, muestra=sample)
            for f, d in MONOTONIC.items() if d != 0]
    table = pd.DataFrame(rows)
    # Share of applicants with a violation in at least one variable needs the
    # per-applicant flags, which the audit summarises per variable; the
    # largest per-variable share is a lower bound and is what 05 reports too.
    return {
        "by_variable": table[["feature", "pct_casos_con_violacion", "violacion_maxima_pd"]]
        .round(4).to_dict(orient="records"),
        "max_share_of_applicants_with_violation": float(table["pct_casos_con_violacion"].max()),
        "variables_with_any_violation": int((table["pct_casos_con_violacion"] > 0).sum()),
    }


def model_comparison(train, valid, test, card) -> tuple[dict, dict]:
    enc = Encoder(train)
    X_tr, X_va, X_te = enc(train), enc(valid), enc(test)
    y_tr = train["bad"].to_numpy(int)
    models = {"gbm_libre": fit_gbm(X_tr, y_tr, monotone=False),
              "gbm_monotono": fit_gbm(X_tr, y_tr, monotone=True)}
    out = {"scorecard_dp_monotono": {"valid_2014": metrics(valid["bad"], card.predict_proba(valid)),
                                     "test_2015": metrics(test["bad"], card.predict_proba(test))}}
    for name, m in models.items():
        out[name] = {"valid_2014": metrics(valid["bad"], m.predict_proba(X_va)[:, 1]),
                     "test_2015": metrics(test["bad"], m.predict_proba(X_te)[:, 1]),
                     "iterations": int(m.n_iter_),
                     "audit": monotonicity_audit(m, X_te)}
    # Lending Club's own ranking: the sub-grade it priced each loan with.
    for name, df in (("valid_2014", valid), ("test_2015", test)):
        out.setdefault("lending_club_sub_grade", {})[name] = {
            "auc": float(roc_auc_score(df["bad"], sub_grade_rank(df["sub_grade"]))),
            "ks": scorecard.ks_statistic(df["bad"].to_numpy(), sub_grade_rank(df["sub_grade"])),
        }
    return out, {"encoder": enc, **models}


def _decisions(cp, X: np.ndarray, y: np.ndarray, amount: np.ndarray) -> dict:
    sets = cp.prediction_sets(X, ALPHA)
    decision = policy.decisiones_conformes(sets)
    covered = sets[np.arange(len(y)), y]
    automated = decision != "revisar"
    wrong = ((decision == "aprobar") & (y == 1)) | ((decision == "rechazar") & (y == 0))
    return {
        "alpha": ALPHA,
        "coverage": float(covered.mean()),
        "coverage_good": float(covered[y == 0].mean()),
        "coverage_bad": float(covered[y == 1].mean()),
        "approve": float((decision == "aprobar").mean()),
        "review": float((decision == "revisar").mean()),
        "decline": float((decision == "rechazar").mean()),
        "error_rate_automated": float(wrong[automated].mean()) if automated.any() else None,
        "bad_rate_approved": float(y[decision == "aprobar"].mean()) if (decision == "aprobar").any() else None,
        "amount_reviewed_share": float(amount[decision == "revisar"].sum() / amount.sum()),
    }


def conformal_decisions(models: dict, valid, test, seed: int = 42) -> dict:
    """Technique 05's Mondrian conformal sets on the monotone model."""
    enc, model = models["encoder"], models["gbm_monotono"]
    X_va, X_te = enc(valid), enc(test)
    y_va, y_te = valid["bad"].to_numpy(int), test["bad"].to_numpy(int)
    amount = test["loan_amnt"].to_numpy(float)

    across = conformal.MondrianConformalClassifier(model).calibrate(X_va, y_va)
    out = {"calibrated_2014_tested_2015": _decisions(across, X_te, y_te, amount)}

    # Exchangeable version: calibrate on half of 2015, test on the other half.
    rng = np.random.default_rng(seed)
    half = rng.random(len(test)) < 0.5
    within = conformal.MondrianConformalClassifier(model).calibrate(X_te[half], y_te[half])
    out["calibrated_and_tested_in_2015"] = _decisions(within, X_te[~half], y_te[~half], amount[~half])
    return out
