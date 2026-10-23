"""Run every comparison on the real book and write the results.

    python -m src.study            # ~5 minutes on a laptop CPU
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src import data, scoring, stability, timing, vintages

BASE = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE / "outputs" / "reports"
RESULTS = REPORTS_DIR / "lending_club_results.json"


def main() -> dict:
    start = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    book, report = data.prepare(data.read_raw())
    print(f"book: {report['loans']:,} resolved 36-month loans, bad rate {report['bad_rate']:.4f}")
    train, valid, test = data.impute(*data.split(book))
    report["split"] = {name: {"loans": int(len(d)), "bad_rate": float(d["bad"].mean())}
                       for name, d in (("train_2007_2013", train), ("valid_2014", valid), ("test_2015", test))}

    binning_results, cards = scoring.binning_comparison(train, valid, test)
    for m, r in binning_results.items():
        print(f"  {m:14s} IV {r['total_iv']:.4f}  non-monotone {len(r['non_monotone'])}  "
              f"AUC 2015 {r['test_2015']['auc']:.4f}")
    card = cards["dp_monotono"]
    card.tarjeta.to_csv(REPORTS_DIR / "scorecard_dp_monotono.csv", index=False)

    models, fitted = scoring.model_comparison(train, valid, test, card)
    for name, r in models.items():
        print(f"  {name:24s} AUC 2015 {r['test_2015']['auc']:.4f}")
    decisions = scoring.conformal_decisions(fitted, valid, test)
    print("  conformal:", {k: round(v["review"], 3) for k, v in decisions.items()})

    whole = pd.concat([train, valid, test], ignore_index=True)
    timing_results = {"last_payment": timing.default_timing(whole),
                      "ninety_days": timing.default_timing(whole, shift=timing.NINETY_DAYS)}
    grade_timing = timing.by_grade(whole)
    print(f"  defaults after month 12: KM {timing_results['last_payment']['km_share_after_12m']:.3f}, "
          f"observed {timing_results['last_payment']['observed_share_of_defaults_after_12m']:.3f}")

    cohorts = vintages.cohorts(whole)
    cohorts.to_csv(REPORTS_DIR / "vintage_cohorts.csv", index=False)
    rho = vintages.estimate(cohorts)
    series = vintages.portfolio_series(whole, data.read_unemployment())
    series.to_csv(REPORTS_DIR / "vintage_portfolio.csv", index=False)
    print(rho[["grade", "pd_36m", "rho_mom_with_n", "rho_mom_without_n", "rho_mom_with_n_detrended",
               "rho_basel_retail"]].round(4).to_string(index=False))

    later = pd.concat([valid, test], ignore_index=True)
    drift_table = stability.quarterly(card, train, later)
    drift_table.to_csv(REPORTS_DIR / "quarterly_stability.csv", index=False)
    print(drift_table[["quarter", "auc", "psi_score", "max_psi", "observed_rate", "mean_pd"]].round(4).to_string(index=False))

    results = {
        "data": {"source": "Kaggle wordsforthewise/lending-club, accepted_2007_to_2018Q4.csv.gz",
                 "unemployment": "FRED UNRATE", **report},
        "binning_06": binning_results,
        "models_01_05": models,
        "conformal_05": decisions,
        "timing_03": {**timing_results, "by_grade": grade_timing.to_dict(orient="records")},
        "vintages_10": {"min_cohort": vintages.MIN_COHORT, "cohorts": int(len(cohorts)),
                        "by_grade": rho.to_dict(orient="records")},
        "stability_12_22": {"summary": stability.summary(drift_table),
                            "quarters": drift_table.to_dict(orient="records")},
        "elapsed_seconds": round(time.perf_counter() - start, 1),
    }
    RESULTS.write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"saved {RESULTS} in {results['elapsed_seconds']} s")
    return results


if __name__ == "__main__":
    main()
