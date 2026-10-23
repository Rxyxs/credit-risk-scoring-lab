"""Load the lab's own implementations, so the real book runs through the same code.

Every technique in this repository is a self-contained folder whose code lives
in a package called `src`, so two of them cannot be imported side by side the
usual way. `load(folder, *modules)` puts one folder first on the path, imports
the requested modules, and then takes that folder's `src` back out of
`sys.modules` (restoring whatever `src` was there before, including this
folder's own). The loaded module objects keep working: they hold references to
everything they imported.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]


def _src_modules() -> dict:
    return {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}


def load(folder: str, *modules: str):
    """Import `src.<module>` for each name from the technique in `LAB/folder`."""
    root = LAB / folder
    if not (root / "src").is_dir():
        raise FileNotFoundError(f"{root} has no src package")
    saved = _src_modules()
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(root))
    try:
        loaded = tuple(importlib.import_module(f"src.{m}") for m in modules)
    finally:
        sys.path.remove(str(root))
        for k in _src_modules():
            del sys.modules[k]
        sys.modules.update(saved)
    return loaded if len(loaded) > 1 else loaded[0]


# The techniques this folder reuses, loaded once.
binning, scorecard = load("06-optimal-binning-scorecard", "binning", "scorecard")
audit, conformal, policy = load("05-monotonic-constraints-conformal-decisioning",
                                "monotonicity_audit", "conformal", "decision_policy")
survival = load("03-survival-lifetime-pd-term-structure", "evaluation")
correlation, vasicek = load("10-through-the-cycle-pd-vasicek", "correlation_estimation", "vasicek")
drift = load("12-drift-monitoring-psi-ks", "monitoring.drift")

# Technique 22's default retraining thresholds. Importing it would pull in
# DuckDB for two numbers, so they are copied here and a test checks that they
# still match its source.
TRIGGER_MIN_AUC = 0.72
TRIGGER_MAX_PSI = 0.20
