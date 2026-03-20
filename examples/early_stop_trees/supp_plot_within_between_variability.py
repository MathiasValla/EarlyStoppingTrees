#!/usr/bin/env python
"""
Supplementary: within-dataset randomness vs between-dataset heterogeneity.

We visualize two complementary summaries derived from the same dataset-level medians:

1) Within-dataset (run-to-run randomness):
   point = dataset-level median performance,
   whisker = run-to-run variability (IQR across runs for that dataset-method).

2) Between-dataset (dataset heterogeneity):
   forest plot across datasets of the paired deltas vs a reference (IQR across datasets).

Outputs (per task):
- within whisker plots (speedup + loss)
- between forest plots (speedup + loss)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_utils import TASKS, load_task, method_sort_key


def _task_metric_cols(task: str):
    spec = TASKS[task]
    speed_med = f"{spec.speedup_col}_median"
    speed_iqr = f"{spec.speedup_col}_iqr"
    loss_med = f"{spec.loss_col}_median"
    loss_iqr = f"{spec.loss_col}_iqr"
    return speed_med, speed_iqr, loss_med, loss_iqr


def _nice_metric_label(task: str, which: str) -> str:
    if which == "speedup":
        return "Speedup (median across runs)"
    if which == "loss":
        if task == "regression":
            return "Loss (RMSE relative, median across runs)"
        return "Loss (F1 relative, median across runs)"
    raise ValueError(which)


def _plot_within_whiskers(
    ds_sum: pd.DataFrame,
    *,
    metric_median_col: str,
    metric_iqr_col: str,
    methods: list[str],
    title: str,
    outpath: Path,
    baseline_x: float,
):
    df = ds_sum[["dataset", "method_key", metric_median_col, metric_iqr_col]].copy()
    df = df.dropna(subset=[metric_median_col, metric_iqr_col])
    df = df[df["method_key"].isin(methods)]
    df["method_key"] = df["method_key"].astype(str)

    methods_sorted = sorted(methods, key=method_sort_key)
    n = len(methods_sorted)
    fig_h = max(4.0, 0.95 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, mk in zip(axes, methods_sorted):
        sub = df[df["method_key"] == mk].sort_values(metric_median_col)
        x = sub[metric_median_col].to_numpy(dtype=float)
        # symmetric half-IQR whiskers
        xerr = 0.5 * sub[metric_iqr_col].to_numpy(dtype=float)
        y = np.arange(sub.shape[0])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            markersize=2.5,
            color="#1f77b4",
            ecolor="#9ecae1",
            elinewidth=0.8,
            capsize=1,
        )
        ax.axvline(baseline_x, color="#bbbbbb", linewidth=1.0)
        ax.set_ylabel(mk, rotation=0, ha="right", va="center", labelpad=28)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.2)

    axes[0].set_title(title)
    axes[-1].set_xlabel(f"{metric_median_col} (point) with +/- 0.5 IQR whiskers")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_between_forest(
    forest_df: pd.DataFrame,
    *,
    methods: list[str],
    title: str,
    outpath: Path,
):
    df = forest_df.copy()
    df = df[df["method_key"].isin(methods)]
    df = df.sort_values("method_key", key=lambda s: s.map(method_sort_key))
    y = np.arange(df.shape[0])

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.35 * df.shape[0] + 1.5)))
    ax.errorbar(
        df["delta_median_across_datasets"].to_numpy(dtype=float),
        y,
        xerr=df["delta_iqr_across_datasets"].to_numpy(dtype=float),
        fmt="o",
        color="#222222",
        ecolor="#666666",
        elinewidth=1.2,
        capsize=2,
    )
    ax.axvline(0.0, color="#999999", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method_key"].tolist())
    ax.set_xlabel("Paired delta vs reference (median across datasets)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=str, default=None, help="analysis_additional directory")
    ap.add_argument("--task", type=str, default="regression", choices=sorted(TASKS.keys()))
    ap.add_argument("--top-k", type=int, default=6, help="Keep top-k methods (by speedup win_rate vs best|)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: analysis-dir/supplementary)")
    ap.add_argument("--ref", type=str, default="best|", help="Reference method_key used by paired analysis")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else (script_dir / "analysis_additional")
    paired_dir = analysis_dir / "paired"

    outdir = Path(args.outdir) if args.outdir else (analysis_dir / "supplementary")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    # Select methods by paired speedup win_rate
    speed_csv = paired_dir / f"{args.task}_paired_summary_speedup_vs_best__.csv"
    speed_win = pd.read_csv(speed_csv)
    speed_win = speed_win[speed_win["method_key"].astype(str) != "best|"].copy()
    speed_win = speed_win.sort_values("win_rate", ascending=False)
    methods = speed_win.head(args.top_k)["method_key"].astype(str).tolist()

    run_df, ds_sum = load_task(Path(analysis_dir.parent / "benchmark_results"), args.task, exclude_secretary_par=True, by_variant=True)
    if ds_sum is None or ds_sum.empty:
        raise SystemExit(f"Empty ds_sum for task={args.task}")

    speed_med, speed_iqr, loss_med, loss_iqr = _task_metric_cols(args.task)

    _plot_within_whiskers(
        ds_sum,
        metric_median_col=speed_med,
        metric_iqr_col=speed_iqr,
        methods=methods,
        title=f"{args.task}: within-dataset variability (whiskers = run-to-run IQR) - speedup",
        outpath=outdir / "figures" / f"{args.task}_supp_within_whiskers_speedup.pdf",
        baseline_x=1.0,
    )
    _plot_within_whiskers(
        ds_sum,
        metric_median_col=loss_med,
        metric_iqr_col=loss_iqr,
        methods=methods,
        title=f"{args.task}: within-dataset variability (whiskers = run-to-run IQR) - loss",
        outpath=outdir / "figures" / f"{args.task}_supp_within_whiskers_loss.pdf",
        baseline_x=0.0,
    )

    # Between-dataset forest from the precomputed CSVs written by analysis_paired.py
    forest_speed = pd.read_csv(paired_dir / f"{args.task}_forest_speedup_vs_best__.csv")
    forest_loss = pd.read_csv(paired_dir / f"{args.task}_forest_loss_vs_best__.csv")

    _plot_between_forest(
        forest_speed,
        methods=methods,
        title=f"{args.task}: between-dataset variability (forest of paired delta vs best|) - speedup",
        outpath=outdir / "figures" / f"{args.task}_supp_between_forest_speedup.pdf",
    )
    _plot_between_forest(
        forest_loss,
        methods=methods,
        title=f"{args.task}: between-dataset variability (forest of paired delta vs best|) - loss",
        outpath=outdir / "figures" / f"{args.task}_supp_between_forest_loss.pdf",
    )

    print(f"Wrote supplementary variability figures to {outdir / 'figures'}")


if __name__ == "__main__":
    main()

