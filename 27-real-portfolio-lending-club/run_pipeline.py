"""Technique 27 end to end: the lab's claims on a real loan book.

Usage (the data is not in the repository, see the README):
    kaggle datasets download wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/raw
    curl -o data/raw/UNRATE.csv "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    python run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("Run techniques 01, 03, 05, 06, 10, 12 and 22 on Lending Club 2007-2015", "src.study"),
    ("Plots", "src.plots"),
]


def main():
    for description, module in STEPS:
        print(f"\n{'=' * 78}\n>> {description}\n{'=' * 78}")
        r = subprocess.run([sys.executable, "-m", module], cwd=BASE_DIR)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
