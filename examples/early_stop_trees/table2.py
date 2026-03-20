#!/usr/bin/env python
"""
Table 2. Joint probabilities P(speedup ≥ x% and loss ≥ y%).

Rows: speedup ≥ x% for x = 0, 5, 10, ..., 50.
Columns: loss ≥ y% for y = 0, 5, 10, ..., 50.
Cell (x, y) = proportion of datasets (for that task and method) where
  speedup_median >= 1 + x/100  and  loss_median >= y/100.

One such table per (task, method): regression, classification Gini, classification Entropy
and each method in SPLITTERS.

Output: tables/table2_joint_{task}_{method}.csv for each (task, method).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_results_utils import load_all

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "tables"

# x and y from 0 to 50 by step of 5
SPEEDUP_PCT = list(range(0, 55, 5))   # 0, 5, ..., 50
LOSS_PCT = list(range(0, 55, 5))      # 0, 5, ..., 50


def _joint_probability_table(sub: pd.DataFrame, loss_col: str) -> pd.DataFrame:
    """Build 11×11 matrix: cell (i,j) = P(speedup ≥ speedup_pct[i] and loss ≥ loss_pct[j])."""
    if sub is None or sub.empty:
        return pd.DataFrame()
    sub = sub.dropna(subset=["speedup_median", loss_col])
    n = len(sub)
    if n == 0:
        return pd.DataFrame()
    speedup = sub["speedup_median"].values
    loss = sub[loss_col].values  # in [0,1]
    rows = []
    for x in SPEEDUP_PCT:
        sp_min = 1.0 + x / 100.0
        row = []
        for y in LOSS_PCT:
            loss_min = y / 100.0
            count = np.sum((speedup >= sp_min) & (loss >= loss_min))
            row.append(count / n)
        rows.append(row)
    df = pd.DataFrame(rows, index=[f"speedup≥{x}%" for x in SPEEDUP_PCT], columns=[f"loss≥{y}%" for y in LOSS_PCT])
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(BENCHMARK_DIR, exclude_secretary_par=True, by_variant=True)

    configs = [
        ("regression", data["regression_summary"], "loss_rmse_bounded_median"),
        ("gini", data["classification_gini_summary"], "loss_f1_median"),
        ("entropy", data["classification_entropy_summary"], "loss_f1_median"),
    ]

    for tag, summary, loss_col in configs:
        if summary is None:
            continue
        summary = summary.dropna(subset=["speedup_median", loss_col]).copy()
        if "variant" not in summary.columns:
            summary["variant"] = ""
        summary["method_key"] = summary["splitter"].astype(str) + "|" + summary["variant"].fillna("").astype(str)
        for method_key in summary["method_key"].unique():
            sub = summary[summary["method_key"] == method_key]
            table = _joint_probability_table(sub, loss_col)
            if table.empty:
                continue
            # Filename-safe: replace | with __
            method_slug = str(method_key).replace("|", "__")
            out_csv = OUT_DIR / f"table2_joint_{tag}_{method_slug}.csv"
            table.to_csv(out_csv)
            print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
