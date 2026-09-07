[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Credit Risk Scoring Lab

Five self-contained approaches to the same question — *how likely is this borrower to default, and what should be done about it?* — each answering it with a different method, and each reporting what its method costs as well as what it buys. Every folder has its own README, dependencies, tests, and a pipeline that runs end to end from one command.

## Techniques

| # | Technique | Folder | What it does |
|---|---|---|---|
| 01 | Polyglot scorecard (R + Python + C) | [`01-polyglot-scorecard-r-python-c`](01-polyglot-scorecard-r-python-c) | R does the regulatory WOE/IV + logistic-regression scorecard, Python does ML challengers (XGBoost/LightGBM) + SHAP, C implements a compiled scoring hot-path verified bit-for-bit against R's score. |
| 02 | Bidirectional R↔Python interop | [`02-bidirectional-r-python-interop`](02-bidirectional-r-python-interop) | Python handles data cleaning and credit scoring; R handles candlestick market analysis, GARCH volatility, and empirical LGD calibration (Tobit/GAM); connected both ways via `reticulate` (R calls Python) and `rpy2` (Python calls R), stress-tested against real Chilean macro data under an IFRS9/Basel III framing. |
| 03 | Lifetime PD from survival analysis | [`03-survival-lifetime-pd-term-structure`](03-survival-lifetime-pd-term-structure) | Models *when* default arrives, not just whether: Cox proportional hazards written from scratch (Efron and Breslow partial likelihoods, analytic gradient/Hessian, Schoenfeld diagnostics) plus a discrete-time hazard model, turned into an IFRS 9 PD term structure — 50.5% of lifetime risk lands after month 12. |
| 04 | Bayesian hierarchical scorecard | [`04-bayesian-hierarchical-partial-pooling`](04-bayesian-hierarchical-partial-pooling) | Partial pooling across 32 commercial segments, sampled with a Gibbs sampler written from scratch on Pólya-Gamma augmentation. Every PD is a posterior distribution; segment-effect error drops 39% against both no pooling and complete pooling, and the approval policy is allowed to use the uncertainty. |
| 05 | Monotonic constraints + conformal decisioning | [`05-monotonic-constraints-conformal-decisioning`](05-monotonic-constraints-conformal-decisioning) | A model a supervisor can accept and that knows when to abstain: monotonic-constrained gradient boosting audited by counterfactual perturbation (97% of applicants had a violation before constraining, 0% after, at no accuracy cost), wrapped in a Mondrian conformal predictor that turns PD into approve / manual review / decline with a distribution-free error guarantee. |

## What ties them together

Each technique targets a different failure mode of a plain PD model:

- **01** — the same scorecard has to be interpretable *and* fast in production, and those pull in different directions.
- **02** — credit risk is not estimated in isolation from market risk, and the tooling for each lives in a different language.
- **03** — a 12-month PD says nothing about *when* risk arrives, which is exactly what provisioning under IFRS 9 turns on.
- **04** — one model for a heterogeneous portfolio is wrong, and one model per segment is noise; and a point estimate hides how much the model actually knows.
- **05** — a model that contradicts the domain does not get approved, and a model that cannot abstain automates the decisions it should not be making.

## Running a technique

Each folder is self-contained — see its own README for the exact setup and entry point, real results from an actual run, and its honest negative findings.

```bash
cd 03-survival-lifetime-pd-term-structure    # or any other
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
pytest -q
```

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
