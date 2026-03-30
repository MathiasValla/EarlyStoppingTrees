#!/usr/bin/env python
"""
Supplementary effort figures.

Writes:
  - SUPP_FIGURES/supp_figure_14_effort_loss.png
  - SUPP_FIGURES/supp_figure_15_time_effort_calibration.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from benchmark_results_utils import (
    add_method_key,
    get_variant_method_order_and_colors,
    load_all,
    per_dataset_median_iqr,
    plot_grouped_variant_legend,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"

TASK_CONFIGS = (
    ("regression", "Regression", "loss_rmse_bounded"),
    ("classification_gini", "Classification (Gini)", "loss_f1"),
    ("classification_entropy", "Classification (Entropy)", "loss_f1"),
)


def _time_saved_pct(speedup: pd.Series) -> pd.Series:
    speedup = pd.to_numeric(speedup, errors="coerce")
    return 100.0 * (1.0 - 1.0 / np.maximum(speedup, 1e-6))


def _prepare_run_df(df: pd.DataFrame, loss_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return None
    df = add_method_key(df)
    required = [loss_col, "speedup", "effort_saved_total", "effort_saved_per_split"]
    for col in required:
        if col not in df.columns:
            return None
    df["loss_pct"] = 100.0 * pd.to_numeric(df[loss_col], errors="coerce")
    df["time_saved_pct"] = _time_saved_pct(df["speedup"])
    df["effort_saved_total_pct"] = 100.0 * pd.to_numeric(df["effort_saved_total"], errors="coerce")
    df["effort_saved_per_split_pct"] = 100.0 * pd.to_numeric(
        df["effort_saved_per_split"], errors="coerce"
    )
    return df


def _load_all_runs(indir: Path):
    data = load_all(indir, exclude_secretary_par=False, by_variant=True)
    return {
        task_key: _prepare_run_df(data.get(f"{task_key}_run"), loss_col)
        for task_key, _, loss_col in TASK_CONFIGS
    }


def _dataset_summary(run_df: pd.DataFrame, value_cols) -> pd.DataFrame:
    if run_df is None or run_df.empty:
        return None
    summary = per_dataset_median_iqr(
        run_df,
        value_cols,
        group_cols=["dataset", "splitter", "variant"],
    )
    keep = np.zeros(len(summary), dtype=bool)
    for col in value_cols:
        median_col = f"{col}_median"
        if median_col in summary.columns:
            keep |= np.isfinite(pd.to_numeric(summary[median_col], errors="coerce"))
    if not np.any(keep):
        return summary.iloc[0:0].copy()
    return summary.loc[keep].copy()


def _scatter_panel(ax, summary_df, x_col, y_col, title, method_order, method_colors):
    if summary_df is None or summary_df.empty:
        ax.set_visible(False)
        return

    for method_key in method_order:
        splitter, variant = method_key.split("|", 1)
        sub = summary_df[
            (summary_df["splitter"] == splitter)
            & (summary_df["variant"].fillna("").astype(str) == variant)
        ]
        if sub.empty:
            continue
        x = sub[f"{x_col}_median"].to_numpy(dtype=float)
        y = sub[f"{y_col}_median"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if not np.any(ok):
            continue
        color = method_colors.get(method_key, "#888888")
        ax.scatter(x[ok], y[ok], s=12, alpha=0.16, color=color, edgecolors="none")
        ax.scatter(
            np.median(x[ok]),
            np.median(y[ok]),
            s=72,
            color=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )

    ax.axhline(0, color="#d7d7d7", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#d7d7d7", linewidth=0.8, zorder=0)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)


def _build_method_palette(task_summaries):
    summaries = [df for df in task_summaries if df is not None and not df.empty]
    if not summaries:
        return [], {}, []
    return get_variant_method_order_and_colors(*summaries, include_secretary_par=True)


def plot_effort_loss(indir: Path):
    runs = _load_all_runs(indir)
    summaries = [
        _dataset_summary(runs[task_key], ["effort_saved_total_pct", "loss_pct"])
        for task_key, _, _ in TASK_CONFIGS
    ]
    method_order, method_colors, method_labels = _build_method_palette(summaries)

    fig = plt.figure(figsize=(14, 5.8), layout="constrained")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.56], hspace=0.08, wspace=0.08)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_leg = fig.add_subplot(gs[1, :])

    for ax, (task_key, title, _), summary in zip(axes, TASK_CONFIGS, summaries):
        _scatter_panel(
            ax,
            summary,
            "effort_saved_total_pct",
            "loss_pct",
            title,
            method_order,
            method_colors,
        )
    axes[0].set_ylabel("Median loss vs best (%)")
    for ax in axes:
        ax.set_xlabel("Median effort saved vs best (%)")

    plot_grouped_variant_legend(
        ax_leg,
        method_order,
        method_colors,
        method_labels,
        fontsize=7,
        legend_style="point",
    )

    out = SUPP_DIR / "supp_figure_14_effort_loss.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_time_effort(indir: Path):
    runs = _load_all_runs(indir)
    total_summaries = [
        _dataset_summary(runs[task_key], ["time_saved_pct", "effort_saved_total_pct"])
        for task_key, _, _ in TASK_CONFIGS
    ]
    per_split_summaries = [
        _dataset_summary(runs[task_key], ["time_saved_pct", "effort_saved_per_split_pct"])
        for task_key, _, _ in TASK_CONFIGS
    ]
    method_order, method_colors, method_labels = _build_method_palette(total_summaries + per_split_summaries)

    fig = plt.figure(figsize=(14, 8.2), layout="constrained")
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.56], hspace=0.10, wspace=0.08)
    axes_top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    axes_bottom = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax_leg = fig.add_subplot(gs[2, :])

    for ax, (_, title, _), summary in zip(axes_top, TASK_CONFIGS, total_summaries):
        _scatter_panel(
            ax,
            summary,
            "time_saved_pct",
            "effort_saved_total_pct",
            title,
            method_order,
            method_colors,
        )
    for ax, (_, title, _), summary in zip(axes_bottom, TASK_CONFIGS, per_split_summaries):
        _scatter_panel(
            ax,
            summary,
            "time_saved_pct",
            "effort_saved_per_split_pct",
            title,
            method_order,
            method_colors,
        )

    axes_top[0].set_ylabel("Median total effort saved (%)")
    axes_bottom[0].set_ylabel("Median per-split effort saved (%)")
    for ax in axes_top + axes_bottom:
        ax.set_xlabel("Median wall-clock time saved (%)")

    plot_grouped_variant_legend(
        ax_leg,
        method_order,
        method_colors,
        method_labels,
        fontsize=7,
        legend_style="point",
    )

    out = SUPP_DIR / "supp_figure_15_time_effort_calibration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    plot_effort_loss(BENCHMARK_DIR)
    plot_time_effort(BENCHMARK_DIR)


if __name__ == "__main__":
    main()
