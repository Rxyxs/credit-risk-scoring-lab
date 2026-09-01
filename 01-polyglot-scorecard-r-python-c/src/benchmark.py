"""Benchmark C vs. NumPy vs. Python puro para el motor de scoring, mas
validacion de correctitud: los tres deben reproducir exactamente el score
que R calculo para los mismos solicitantes (no solo ser rapidos, sino
ser el MISMO numero).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ctypes_bridge import ScoreEngine, score_batch_numpy, score_batch_python

BASE = Path(__file__).resolve().parents[1]
REPLICATE_FACTOR = 50  # replica el batch real para tener suficientes filas y medir tiempos estables


def load_scorecard_artifacts():
    meta = json.loads((BASE / "outputs" / "reports" / "scorecard_meta.json").read_text())
    points_df = pd.read_csv(BASE / "outputs" / "reports" / "scorecard_points_table.csv")
    binned = pd.read_csv(BASE / "data" / "processed" / "applicants_woe_binned.csv")

    feature_order = meta["feature_order"]
    max_bins = meta["max_bins"]

    points_table = np.zeros((len(feature_order), max_bins), dtype=np.float64)
    for i, feat in enumerate(feature_order):
        rows = points_df[points_df["feature"] == feat]
        for _, row in rows.iterrows():
            points_table[i, int(row["bin_id"])] = row["points"]

    bin_cols = [f"{feat}_bin" for feat in feature_order]
    bin_indices = binned[bin_cols].to_numpy(dtype=np.int32)

    return meta, points_table, bin_indices, binned


def check_correctness(meta, points_table, bin_indices, binned) -> dict:
    scored = pd.read_csv(BASE / "data" / "processed" / "applicants_scored.csv")
    r_scores = scored.set_index("applicant_id").loc[binned["applicant_id"], "score"].to_numpy()

    engine = ScoreEngine()
    c_scores = engine.score_batch(bin_indices, points_table, meta["base_points"])
    np_scores = score_batch_numpy(bin_indices, points_table, meta["base_points"])
    py_scores = score_batch_python(bin_indices[:2000], points_table, meta["base_points"])

    return {
        "max_abs_diff_c_vs_r": float(np.max(np.abs(c_scores - r_scores))),
        "max_abs_diff_numpy_vs_r": float(np.max(np.abs(np_scores - r_scores))),
        "max_abs_diff_python_vs_r": float(np.max(np.abs(py_scores - r_scores[:2000]))),
        "max_abs_diff_c_vs_numpy": float(np.max(np.abs(c_scores - np_scores))),
    }


def time_method(fn, *args, n_repeats: int = 3) -> float:
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def run_benchmark() -> dict:
    meta, points_table, bin_indices, binned = load_scorecard_artifacts()
    correctness = check_correctness(meta, points_table, bin_indices, binned)

    bin_indices_big = np.tile(bin_indices, (REPLICATE_FACTOR, 1))
    n_rows = bin_indices_big.shape[0]

    engine = ScoreEngine()
    t_c = time_method(engine.score_batch, bin_indices_big, points_table, meta["base_points"])
    t_numpy = time_method(score_batch_numpy, bin_indices_big, points_table, meta["base_points"])

    n_python_subset = 20_000
    t_python_subset = time_method(
        score_batch_python, bin_indices_big[:n_python_subset], points_table, meta["base_points"], n_repeats=1
    )
    t_python_full_estimate = t_python_subset * (n_rows / n_python_subset)

    results = {
        "n_rows_benchmarked": int(n_rows),
        "n_features": int(bin_indices.shape[1]),
        "correctness": correctness,
        "throughput": {
            "c_engine": {"seconds": t_c, "rows_per_sec": n_rows / t_c},
            "numpy_vectorized": {"seconds": t_numpy, "rows_per_sec": n_rows / t_numpy},
            "python_pure_loop": {
                "seconds_measured_subset": t_python_subset,
                "n_rows_measured": n_python_subset,
                "seconds_extrapolated_full": t_python_full_estimate,
                "rows_per_sec": n_python_subset / t_python_subset,
            },
        },
        "speedup_c_vs_python_loop": (n_python_subset / t_python_subset) and (n_rows / t_c) / (n_python_subset / t_python_subset),
        "speedup_c_vs_numpy": (n_rows / t_c) / (n_rows / t_numpy),
    }
    return results


if __name__ == "__main__":
    results = run_benchmark()
    out_path = BASE / "outputs" / "reports" / "c_engine_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("Correctitud (diferencia maxima absoluta vs. score de R):")
    for k, v in results["correctness"].items():
        print(f"  {k}: {v:.2e}")

    tp = results["throughput"]
    print(f"\nThroughput sobre {results['n_rows_benchmarked']:,} filas:")
    print(f"  C (ctypes):       {tp['c_engine']['rows_per_sec']:,.0f} filas/seg")
    print(f"  NumPy vectorizado: {tp['numpy_vectorized']['rows_per_sec']:,.0f} filas/seg")
    print(f"  Python puro:      {tp['python_pure_loop']['rows_per_sec']:,.0f} filas/seg")
    print(f"\nSpeedup C vs. Python puro: {results['speedup_c_vs_python_loop']:.1f}x")
    print(f"Speedup C vs. NumPy: {results['speedup_c_vs_numpy']:.2f}x")
