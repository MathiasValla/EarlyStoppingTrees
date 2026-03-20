#!/usr/bin/env python
"""
Compact dataset summary for the benchmark actually run (datasets present in result CSVs).

Per task (Regression, Classification Gini, Classification Entropy):
  - n_datasets (unique dataset names in run-level files)
  - median n, median p
  - min/max n, min/max p

Outputs:
  - examples/early_stop_trees/tables/dataset_benchmark_summary.csv
  - examples/early_stop_trees/tables/dataset_benchmark_summary.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark_results_utils import load_all
from analysis_utils import task_dataset_benchmark_stats

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
TABLES_DIR = SCRIPT_DIR / "tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default=None, help="benchmark_results directory")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: examples/early_stop_trees/tables)")
    args = ap.parse_args()

    indir = Path(args.indir) if args.indir else BENCHMARK_DIR
    outdir = Path(args.outdir) if args.outdir else TABLES_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_all(indir, exclude_secretary_par=True, by_variant=True)

    rows = []
    specs = [
        ("regression", "Regression", data.get("regression_run")),
        ("classification_gini", "Classification (Gini)", data.get("classification_gini_run")),
        ("classification_entropy", "Classification (Entropy)", data.get("classification_entropy_run")),
    ]
    for _key, label, run_df in specs:
        r = task_dataset_benchmark_stats(run_df, label)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    csv_path = outdir / "dataset_benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    tex_path = outdir / "dataset_benchmark_summary.tex"
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Task & $N$ & median $n$ & median $p$ & $n_{\min}$--$n_{\max}$ & $p_{\min}$--$p_{\max}$ \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        task = str(row["task"]).replace("&", r"\&")
        lines.append(
            f"{task} & {int(row['n_datasets'])} & "
            f"{row['median_n']:.0f} & {row['median_p']:.0f} & "
            f"{row['min_n']:.0f}--{row['max_n']:.0f} & "
            f"{row['min_p']:.0f}--{row['max_p']:.0f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
