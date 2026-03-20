#!/usr/bin/env python
"""
Supplementary: pairwise method comparison at the **dataset level** (medians across runs).

For each pair (method A, method B) and each task, on datasets where **both** methods exist:
  - frac_A_faster: fraction where median speedup(A) > median speedup(B)
  - frac_A_better_loss: fraction where median loss(A) < median loss(B)
  - frac_A_dominates_both: both hold
  - optional Wilcoxon p-values on paired differences (speedup_A - speedup_B), (loss_A - loss_B)

Loss columns match the main figures:
  - Regression: loss_rmse_bounded_median
  - Classification: loss_f1_median

Default pairs include block-rank vs S_all (1/e) as requested; extend DEFAULT_PAIRS below.

Outputs:
  - examples/early_stop_trees/tables/pairwise_method_comparison.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_utils import TASKS, load_task, pairwise_method_pair_metrics

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
TABLES_DIR = SCRIPT_DIR / "tables"

# (method_a, method_b) — interpret as "A vs B" (fractions are from A's perspective)
DEFAULT_PAIRS = [
    ("block_rank|", "secretary_all|1overe"),
    ("block_rank|", "secretary|1overe"),
    ("block_rank|", "double_secretary|1overe"),
    ("block_rank|", "prophet_1sample|"),
    ("secretary_all|1overe", "prophet_1sample|"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument(
        "--pairs",
        type=str,
        default=None,
        help='Optional: extra pairs as "a||b,c||d" (use || between method keys). Appended to defaults.',
    )
    args = ap.parse_args()

    indir = Path(args.indir) if args.indir else BENCHMARK_DIR
    outdir = Path(args.outdir) if args.outdir else TABLES_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = list(DEFAULT_PAIRS)
    if args.pairs:
        for chunk in args.pairs.split(","):
            chunk = chunk.strip()
            if "||" in chunk:
                a, b = chunk.split("||", 1)
                pairs.append((a.strip(), b.strip()))

    rows = []
    for task in ("regression", "classification_gini", "classification_entropy"):
        _, ds_sum = load_task(indir, task, exclude_secretary_par=True, by_variant=True)
        if ds_sum is None or ds_sum.empty:
            continue
        spec = TASKS[task]
        if task == "regression":
            loss_col = "loss_rmse_bounded_median"
        else:
            loss_col = "loss_f1_median"
        speed_col = f"{spec.speedup_col}_median"

        for method_a, method_b in pairs:
            m = pairwise_method_pair_metrics(
                ds_sum,
                method_a,
                method_b,
                speed_median_col=speed_col,
                loss_median_col=loss_col,
            )
            m["task"] = task
            m["loss_column"] = loss_col
            rows.append(m)

    df = pd.DataFrame(rows)
    # Readable column order
    cols = [
        "task",
        "method_a",
        "method_b",
        "n_common_datasets",
        "frac_A_faster",
        "frac_A_better_loss",
        "frac_A_dominates_both",
        "frac_tie_speed",
        "frac_tie_loss",
        "wilcoxon_speed_diff_pvalue",
        "wilcoxon_loss_diff_pvalue",
        "loss_column",
    ]
    df = df[[c for c in cols if c in df.columns]]
    out = outdir / "pairwise_method_comparison.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
