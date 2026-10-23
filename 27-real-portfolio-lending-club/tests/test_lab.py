from __future__ import annotations

import re
import sys

from src import lab


def test_lab_code_is_loaded_from_each_technique():
    assert lab.binning.__file__.endswith("06-optimal-binning-scorecard\\src\\binning.py") or \
        lab.binning.__file__.replace("\\", "/").endswith("06-optimal-binning-scorecard/src/binning.py")
    assert hasattr(lab.binning, "OptimalBinner")
    assert hasattr(lab.conformal, "MondrianConformalClassifier")
    assert hasattr(lab.survival, "kaplan_meier")
    assert hasattr(lab.drift, "calculate_psi")


def test_the_corrected_method_of_moments_is_the_one_used():
    """Technique 10's estimator takes the cohort size since the granularity fix."""
    import inspect
    assert "n_por_cohorte" in inspect.signature(lab.correlation.mom_asset_correlation).parameters


def test_this_folders_src_is_restored():
    import src.data  # noqa: F401
    assert sys.modules["src"].__file__.replace("\\", "/").endswith("27-real-portfolio-lending-club/src/__init__.py")


def test_trigger_thresholds_match_technique_22():
    text = (lab.LAB / "22-automated-retraining-trigger" / "src" / "retraining_trigger.py").read_text(encoding="utf-8")
    auc = float(re.search(r"MIN_REALIZED_AUC_DEFAULT = ([0-9.]+)", text).group(1))
    psi = float(re.search(r"MAX_PSI_DEFAULT = ([0-9.]+)", text).group(1))
    assert (auc, psi) == (lab.TRIGGER_MIN_AUC, lab.TRIGGER_MAX_PSI)
