"""Orquestador end-to-end: Python (datos + limpieza) -> R (WOE/IV +
scorecard + validacion) -> C (build del motor de scoring) -> Python
(benchmark C vs NumPy vs Python, modelos ML challenger, graficos).

Cada etapa corre como el mismo comando que se documenta para ejecutarla
de forma aislada (subprocess `Rscript`/`powershell`, no una reimplementacion
paralela), para que "correr todo" y "correr un paso solo" sean
consistentes.

Uso:
    python run_pipeline.py
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def find_rscript() -> str:
    import shutil

    on_path = shutil.which("Rscript")
    if on_path:
        return on_path

    candidates = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"), reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No se encontro Rscript.exe. Instala R (https://cran.r-project.org/bin/windows/base/) "
        "o agrega su carpeta bin/ al PATH."
    )


def run_step(description: str, cmd: list[str], cwd: Path = BASE_DIR):
    print(f"\n{'=' * 70}\n>> {description}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Paso fallido: {description} (codigo {result.returncode})")


def main():
    for d in ["data/raw", "data/processed", "outputs/models", "outputs/reports", "outputs/plots"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    rscript_exe = find_rscript()

    run_step("Python: generar solicitantes sinteticos", [python_exe, "-m", "src.data_generator"])
    run_step("Python: limpieza + split train/test", [python_exe, "-m", "src.cleaning"])

    run_step("R: binning WOE + IV", [rscript_exe, "R/01_woe_binning.R"])
    run_step("R: scorecard (logit + puntos PDO)", [rscript_exe, "R/02_scorecard_model.R"])
    run_step("R: validacion (AUC/Gini/KS/PSI)", [rscript_exe, "R/03_validation.R"])

    run_step(
        "C: compilar motor de scoring (DLL + benchmark exe)",
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", "c/build.ps1"],
    )

    run_step("Python: benchmark C vs NumPy vs Python + verificacion de correctitud",
              [python_exe, "-m", "src.benchmark"])
    run_step("Python: modelos ML challenger (XGBoost/LightGBM/RF/Logit) + SHAP",
              [python_exe, "-m", "src.ml_models"])
    run_step("Python: MLP PyTorch (Focal Loss, ReLU/GELU/Swish)",
              [python_exe, "-m", "src.deep_learning"])
    run_step("Python: graficos de resultados", [python_exe, "-m", "src.visualization.plots"])
    run_step("Python: persistir metricas en DuckDB", [python_exe, "-m", "src.metrics_store"])

    summarize()


def summarize():
    reports = BASE_DIR / "outputs" / "reports"
    r_metrics = json.loads((reports / "scorecard_validation_metrics.json").read_text())
    ml_report = json.loads((reports / "ml_model_comparison.json").read_text())
    dl_report = json.loads((reports / "dl_model_comparison.json").read_text())
    bench = json.loads((reports / "c_engine_benchmark.json").read_text())

    summary = {
        "scorecard_r": {
            "auc": r_metrics["auc"], "gini": r_metrics["gini"], "ks": r_metrics["ks_statistic"],
            "psi_test_vs_train": r_metrics["psi_test_vs_train"],
            "psi_drifted_vs_train": r_metrics["psi_drifted_vs_train"],
        },
        "best_ml_model": ml_report["best_model"],
        "best_ml_test_auc": ml_report["metrics"][ml_report["best_model"]]["test"]["auc"],
        "best_dl_activation": dl_report["best_activation"],
        "best_dl_test_auc": dl_report["metrics"][dl_report["best_activation"]]["auc"],
        "c_engine_correctness_max_diff_vs_r": bench["correctness"]["max_abs_diff_c_vs_r"],
        "c_engine_speedup_vs_python": bench["speedup_c_vs_python_loop"],
        "c_engine_speedup_vs_numpy": bench["speedup_c_vs_numpy"],
    }
    (reports / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 70}\nPipeline completo. Resumen:\n{'=' * 70}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
