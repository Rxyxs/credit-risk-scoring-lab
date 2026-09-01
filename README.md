[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Credit Risk Scoring Lab

Two credit-risk projects, both deliberately polyglot but for different reasons — one splits stages of a single scorecard across languages for performance and regulatory fit, the other proves genuine bidirectional interop between R and Python. Each folder is self-contained with its own README, dependencies, and tests. This repo replaces two separate repos that used to live on this profile.

## Techniques

| # | Technique | Folder | What it does |
|---|---|---|---|
| 01 | Polyglot scorecard (R + Python + C) | [`01-polyglot-scorecard-r-python-c`](01-polyglot-scorecard-r-python-c) | R does the regulatory WOE/IV + logistic-regression scorecard, Python does ML challengers (XGBoost/LightGBM) + SHAP, C implements a compiled scoring hot-path verified bit-for-bit against R's score. |
| 02 | Bidirectional R↔Python interop | [`02-bidirectional-r-python-interop`](02-bidirectional-r-python-interop) | Python handles data cleaning and credit scoring; R handles candlestick market analysis, GARCH volatility, and empirical LGD calibration (Tobit/GAM); connected both ways via `reticulate` (R calls Python) and `rpy2` (Python calls R), stress-tested against real Chilean macro data under an IFRS9/Basel III framing. |

## Why one repo instead of two

Both projects are real, runnable, and independently tested — this isn't about hiding scope, it's about representing it accurately. Two repos both titled around "credit risk" and "R+Python" read as duplication; one lab makes the actual distinction visible: one project is about *where* each language's stage lives in a pipeline (a systems-design choice), the other is about *how* two languages call each other directly (an interop-engineering choice).

## Running a technique

Each folder is self-contained — see its own README for the exact setup and entry point, real results from an actual run, and any honest negative findings.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
