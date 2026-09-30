[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# 12 — Drift Monitoring (PSI + KS)

A model degrades two ways: the population that reaches it shifts (data drift), or the
relationship between features and outcome shifts (concept drift). Neither shows up as an
exception — both look like nothing, right up until the backtest that catches it three months
late. This technique gives every other technique in this lab a way to check for the first kind
early, on raw features, without needing a fitted scorecard.

## What it measures

- **PSI (Population Stability Index)** — buckets `expected` (train/baseline) by its own
  quantiles, then compares what fraction of `actual` (OOT/scoring) falls in each bucket:

  `PSI = sum_bucket (pct_actual - pct_expected) * ln(pct_actual / pct_expected)`

  Industry-standard bands: **< 0.10** stable, **0.10–0.25** moderate shift (alert), **> 0.25**
  severe drift (recalibration).
- **Two-sample KS** (`scipy.stats.ks_2samp`) — compares the empirical CDFs directly, no
  binning. A second, independent signal: PSI-by-deciles can dilute a shift concentrated at one
  tail; KS doesn't discretize, so it catches that case too.

Unlike [`06-optimal-binning-scorecard/src/monitoring.py`](../06-optimal-binning-scorecard/src/monitoring.py)
(PSI/CSI on a fitted scorecard's *score* and *bins*), this module needs no trained model — it
runs on any numeric column, which is the point: catching drift in the inputs before it ever
reaches a model.

## API

```python
from src.monitoring.drift import calculate_psi, calculate_ks_drift, generate_drift_report

calculate_psi(expected, actual, num_buckets=10)          # -> float
calculate_ks_drift(expected, actual)                       # -> {"statistic", "p_value", "drift_detected"}
generate_drift_report(baseline_df, scoring_df, features)   # -> dict, one entry per feature + summary
```

## Real run

```bash
python run_pipeline.py
```

Synthetic baseline (5,000 rows) vs. a scoring set (2,000 rows) with one feature shifted on
purpose — a +1σ move in mean monthly income, the other two features left alone:

| Feature | PSI | Status | KS p-value | KS drift |
|---|---:|:---:|---:|:---:|
| `ingreso_mensual` (shifted +1σ) | **0.873** | 🔴 red | 1.9e-194 | True |
| `dti` (untouched) | 0.006 | 🟢 green | 0.454 | False |
| `antiguedad_laboral_meses` (untouched) | 0.005 | 🟢 green | 0.738 | False |

Both methods agree on the shifted feature and both stay quiet on the stable ones — the point
of running two independent tests instead of one.

## Known limitation

PSI buckets on `expected`'s own quantiles. If `expected` is constant (or near-constant), its
quantiles collapse into a single cut point, and PSI can't distinguish any two different values
in `actual` from each other — it reports 0.0, not an error. This isn't a bug introduced here;
it's inherent to quantile-of-the-baseline binning and it's the same limitation
`06-optimal-binning-scorecard`'s own `psi_score` has. `tests/test_drift_monitoring.py` documents
it as an explicit test rather than hiding it.

## Tests

```bash
pytest -v
```

**17 tests**: identical distributions → PSI≈0 and KS p≈1; independent samples from the same
distribution stay well under the alert threshold; a severe shift crosses 0.25 on both PSI and
KS; a moderate shift lands in the yellow band specifically (not just green/red); classification
boundaries; and safe handling of all-zero columns, NaNs, too few rows for the requested bucket
count, and a constant baseline.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
