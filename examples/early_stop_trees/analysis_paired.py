#!/usr/bin/env python
"""
Dataset-level paired comparisons + within/between variability decomposition.

Outputs:
- CSVs in examples/early_stop_trees/analysis/
- A few simple figures in examples/early_stop_trees/analysis/figures/

Key idea: summarize 100 runs within each (dataset, method) first, then compare methods across datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    TASKS,
    add_basic_dataset_meta_from_results,
    add_method_key,
    paired_deltas,
    summarize_paired_deltas,
    within_between_decomposition,
    method_sort_key,
    load_task,
)


def _ensure_dirs(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)


def _wilcoxon_signed_rank_pvalue(d: np.ndarray) -> float:
    """
    Two-sided Wilcoxon signed-rank p-value for paired differences d.
    Uses SciPy if available, else returns NaN (still writes effect summaries).
    """
    d = d[np.isfinite(d)]
    if d.size < 5:
        return np.nan
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return np.nan
    try:
        # zero_method='wilcox' drops zeros; consistent default
        stat = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(stat.pvalue)
    except Exception:
        return np.nan


def add_pvalues_wilcoxon(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mk, sub in deltas.groupby("method_key"):
        d = sub["delta"].to_numpy(dtype=float)
        rows.append({"method_key": mk, "wilcoxon_pvalue": _wilcoxon_signed_rank_pvalue(d), "n_pairs": int(np.sum(np.isfinite(d)))})
    out = pd.DataFrame(rows)
    return out


def _plot_forest_whiskers(summary: pd.DataFrame, *, x_col: str, xerr_col: str, title: str, outpath: Path):
    """
    Forest plot: point=median across datasets; whisker=IQR across datasets of dataset-level medians.
    This is "between-dataset" variability (not run-to-run). Run-to-run is handled in separate figures.
    """
    import matplotlib.pyplot as plt

    df = summary.copy()
    df = df.sort_values("method_key", key=lambda s: s.map(method_sort_key))
    y = np.arange(len(df))
    fig_h = max(3.0, 0.28 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.errorbar(df[x_col], y, xerr=df[xerr_col], fmt="o", color="#222222", ecolor="#666666", elinewidth=1.2, capsize=2)
    ax.axvline(0.0, color="#999999", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["method_key"].tolist())
    ax.set_xlabel(x_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _plot_dataset_whiskers(dataset_summary: pd.DataFrame, *, metric_median_col: str, metric_iqr_col: str, outpath: Path, title: str):
    """
    For each method, show distribution across datasets:
    - x = dataset-level median (point)
    - whisker length = within-dataset IQR across runs (run-to-run randomness)

    Plot is faceted vertically by method to keep it readable.
    """
    import matplotlib.pyplot as plt

    df = dataset_summary[["dataset", "method_key", metric_median_col, metric_iqr_col]].dropna().copy()
    methods = sorted(df["method_key"].unique().tolist(), key=method_sort_key)
    n = len(methods)
    fig_h = max(4.0, 1.0 + 0.65 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, mk in zip(axes, methods):
        sub = df[df["method_key"] == mk].sort_values(metric_median_col)
        y = np.arange(sub.shape[0])
        x = sub[metric_median_col].to_numpy(dtype=float)
        xerr = 0.5 * sub[metric_iqr_col].to_numpy(dtype=float)  # half-IQR as symmetric whisker
        ax.errorbar(x, y, xerr=xerr, fmt="o", markersize=2.5, color="#1f77b4", ecolor="#9ecae1", elinewidth=0.8, capsize=1)
        ax.axvline(0.0, color="#bbbbbb", linewidth=1.0)
        ax.set_ylabel(mk, rotation=0, ha="right", va="center", labelpad=30)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.2)
    axes[0].set_title(title)
    axes[-1].set_xlabel(metric_median_col + " (point=dataset median, whisker=½ IQR across runs)")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default=None, help="Benchmark results directory (default: examples/early_stop_trees/benchmark_results)")
    ap.add_argument("--outdir", type=str, default=None, help="Output analysis directory (default: examples/early_stop_trees/analysis)")
    ap.add_argument("--task", type=str, default="regression", choices=sorted(TASKS.keys()))
    ap.add_argument("--ref", type=str, default="best|", help="Reference method_key for paired deltas (default: best|)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    indir = Path(args.indir) if args.indir else (script_dir / "benchmark_results")
    outdir = Path(args.outdir) if args.outdir else (script_dir / "analysis")
    _ensure_dirs(outdir)

    run_df, ds_sum = load_task(indir, args.task, exclude_secretary_par=True, by_variant=True)
    if ds_sum is None or ds_sum.empty:
        raise SystemExit(f"No dataset summary found for task={args.task} in {indir}")

    spec = TASKS[args.task]
    # Identify columns in summary (median + iqr)
    speed_med = f"{spec.speedup_col}_median"
    speed_iqr = f"{spec.speedup_col}_iqr"
    loss_med = f"{spec.loss_col}_median"
    loss_iqr = f"{spec.loss_col}_iqr"

    # Add dataset meta from results (n, p, p/n)
    meta = add_basic_dataset_meta_from_results(ds_sum)
    meta.to_csv(outdir / f"{args.task}_dataset_meta_basic.csv", index=False)

    # Within/between decomposition per method for speedup and loss
    wb_speed = within_between_decomposition(ds_sum, metric_median_col=speed_med, metric_iqr_col=speed_iqr)
    wb_loss = within_between_decomposition(ds_sum, metric_median_col=loss_med, metric_iqr_col=loss_iqr)
    wb_speed.to_csv(outdir / f"{args.task}_within_between_speedup.csv", index=False)
    wb_loss.to_csv(outdir / f"{args.task}_within_between_loss.csv", index=False)

    # Paired deltas vs reference
    deltas_speed = paired_deltas(ds_sum, metric_median_col=speed_med, ref_method_key=args.ref)
    deltas_loss = paired_deltas(ds_sum, metric_median_col=loss_med, ref_method_key=args.ref)
    deltas_speed.to_csv(outdir / f"{args.task}_paired_deltas_speedup_vs_{args.ref.replace('|','__')}.csv", index=False)
    deltas_loss.to_csv(outdir / f"{args.task}_paired_deltas_loss_vs_{args.ref.replace('|','__')}.csv", index=False)

    # Summaries + Wilcoxon p-values
    # For speedup: delta = speedup(method) - speedup(best); higher is better => win when delta > 0.
    # For loss: delta = loss(method) - loss(best); lower is better => win when delta < 0.
    summ_speed = summarize_paired_deltas(deltas_speed, better_if_delta_positive=True).merge(
        add_pvalues_wilcoxon(deltas_speed), on=["method_key", "n_pairs"], how="left"
    )
    summ_loss = summarize_paired_deltas(deltas_loss, better_if_delta_positive=False).merge(
        add_pvalues_wilcoxon(deltas_loss), on=["method_key", "n_pairs"], how="left"
    )
    summ_speed.to_csv(outdir / f"{args.task}_paired_summary_speedup_vs_{args.ref.replace('|','__')}.csv", index=False)
    summ_loss.to_csv(outdir / f"{args.task}_paired_summary_loss_vs_{args.ref.replace('|','__')}.csv", index=False)

    # Simple figures
    # Between-dataset forest (median delta with IQR across datasets)
    # Compute IQR across datasets of delta
    for deltas, name in [(deltas_speed, "speedup"), (deltas_loss, "loss")]:
        g = deltas.groupby("method_key")["delta"]
        forest = pd.DataFrame(
            {
                "method_key": g.median().index,
                "delta_median_across_datasets": g.median().values,
                "delta_iqr_across_datasets": (g.quantile(0.75) - g.quantile(0.25)).values,
            }
        )
        forest = forest.sort_values("method_key", key=lambda s: s.map(method_sort_key))
        forest.to_csv(outdir / f"{args.task}_forest_{name}_vs_{args.ref.replace('|','__')}.csv", index=False)
        _plot_forest_whiskers(
            forest,
            x_col="delta_median_across_datasets",
            xerr_col="delta_iqr_across_datasets",
            title=f"{args.task}: paired delta ({name}) vs {args.ref} (between-dataset IQR)",
            outpath=outdir / "figures" / f"{args.task}_forest_{name}_vs_{args.ref.replace('|','__')}.pdf",
        )

    # Run-to-run whiskers around dataset medians (within-dataset randomness)
    _plot_dataset_whiskers(
        ds_sum,
        metric_median_col=speed_med,
        metric_iqr_col=speed_iqr,
        outpath=outdir / "figures" / f"{args.task}_dataset_whiskers_speedup.pdf",
        title=f"{args.task}: dataset medians with run-to-run whiskers (speedup)",
    )
    _plot_dataset_whiskers(
        ds_sum,
        metric_median_col=loss_med,
        metric_iqr_col=loss_iqr,
        outpath=outdir / "figures" / f"{args.task}_dataset_whiskers_loss.pdf",
        title=f"{args.task}: dataset medians with run-to-run whiskers (loss)",
    )

    print(f"Wrote paired + within/between analysis to {outdir}")


if __name__ == "__main__":
    main()

