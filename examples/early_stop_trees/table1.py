#!/usr/bin/env python
"""
Table 1. Global summary by method.

For each method and each task (Regression, Gini, Entropy), report:
- median speedup across datasets
- median predictive loss across datasets (%)
- 90th percentile predictive loss (%)
- proportion of datasets for which loss stays below a fixed tolerance
- proportion of datasets for which speedup exceeds a fixed threshold.

Output: CSV and optionally LaTeX in tables/ (table1_*.csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_results_utils import load_all, SPLITTERS_NO_PAR

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "tables"

# Fixed tolerance and threshold for the proportion columns
LOSS_TOLERANCE = 0.05   # 5% loss (loss in [0,1] so 0.05)
SPEEDUP_THRESHOLD_PCT = 20  # P(speedup ≥ 20%) -> speedup >= 1.20
SPEEDUP_THRESHOLD = 1.0 + SPEEDUP_THRESHOLD_PCT / 100.0


def _global_summary_one_task(summary: pd.DataFrame, loss_col: str) -> pd.DataFrame:
    """Per (splitter, variant): median speedup, median loss (%), P90 loss (%), P(loss ≤ τ), P(speedup ≥ threshold)."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    summary = summary.dropna(subset=["speedup_median", loss_col]).copy()
    if "variant" not in summary.columns:
        summary["variant"] = ""
    summary["method_key"] = summary["splitter"].astype(str) + "|" + summary["variant"].fillna("").astype(str)
    rows = []
    for method_key in summary["method_key"].unique():
        sub = summary[summary["method_key"] == method_key]
        if sub.empty:
            continue
        s = sub["splitter"].iloc[0]
        v = sub["variant"].iloc[0]
        if pd.isna(v):
            v = ""
        method_label = f"{s} ({v})" if v else s
        sp = sub["speedup_median"].values
        loss = sub[loss_col].values
        loss_pct = 100.0 * loss
        rows.append({
            "method": method_label,
            "splitter": s,
            "variant": v,
            "median_speedup": np.median(sp),
            "median_loss_pct": np.median(loss_pct),
            "p90_loss_pct": np.percentile(loss_pct, 90),
            "p_loss_below_tol": np.mean(loss <= LOSS_TOLERANCE),
            "p_speedup_above_thr": np.mean(sp >= SPEEDUP_THRESHOLD),
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(BENCHMARK_DIR, exclude_secretary_par=True, by_variant=True)

    configs = [
        ("regression", data["regression_summary"], "loss_rmse_bounded_median", "Regression"),
        ("gini", data["classification_gini_summary"], "loss_f1_median", "Classification (Gini)"),
        ("entropy", data["classification_entropy_summary"], "loss_f1_median", "Classification (Entropy)"),
    ]

    all_tables = []
    for tag, summary, loss_col, task_label in configs:
        if summary is None:
            continue
        df = _global_summary_one_task(summary, loss_col)
        if df.empty:
            continue
        df["task"] = task_label
        # Reorder columns: task, method, then metrics
        cols = ["task", "method", "median_speedup", "median_loss_pct", "p90_loss_pct", "p_loss_below_tol", "p_speedup_above_thr"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.rename(columns={
            "median_speedup": "median_speedup",
            "median_loss_pct": "median_loss_%",
            "p90_loss_pct": "p90_loss_%",
            "p_loss_below_tol": f"P(loss≤{int(LOSS_TOLERANCE*100)}%)",
            "p_speedup_above_thr": f"P(speedup≥{SPEEDUP_THRESHOLD_PCT}%)",
        })
        out_csv = OUT_DIR / f"table1_{tag}.csv"
        df.to_csv(out_csv, index=False, float_format="%.4g")
        all_tables.append(df)
        print(f"Saved {out_csv}")

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_csv(OUT_DIR / "table1_all.csv", index=False, float_format="%.4g")
        print(f"Saved {OUT_DIR / 'table1_all.csv'}")
        # LaTeX (one table with task as first column)
        out_tex = OUT_DIR / "table1.tex"
        cols = [c for c in combined.columns if c not in ("task", "method")]
        with open(out_tex, "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write("\\caption{Global summary by method: median speedup, median and 90th percentile loss (\\%), ")
            f.write(f"proportion of datasets with loss $\\leq$ {int(LOSS_TOLERANCE*100)}\\% and speedup $\\geq$ {SPEEDUP_THRESHOLD_PCT}\\% .}}\n")
            f.write("\\label{tab:global-summary}\n")
            f.write("\\begin{tabular}{ll" + "r" * len(cols) + "}\n\\toprule\n")
            f.write("Task & Method & " + " & ".join(c.replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\\n\\midrule\n")
            for _, row in combined.iterrows():
                vals = [str(row["task"]), str(row["method"]).replace("_", " ")]
                for c in cols:
                    v = row[c]
                    vals.append(f"{v:.3g}" if pd.notna(v) and isinstance(v, (int, float)) else str(v))
                f.write(" & ".join(vals) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        print(f"Saved {out_tex}")


if __name__ == "__main__":
    main()
