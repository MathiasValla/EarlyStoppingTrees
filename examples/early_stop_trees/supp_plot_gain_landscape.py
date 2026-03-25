#!/usr/bin/env python
"""
Supplementary figure for exhaustive gain-landscape diagnostics.

Reads the dataset-level output from analysis_gain_landscape.py and produces one
three-panel figure comparing regression, Gini classification, and entropy
classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR / "analysis_gain_landscape"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"
OUT_NAME = "supp_figure_13_gain_landscape.png"
TASK_ORDER = ["regression", "classification_gini", "classification_entropy"]
TASK_LABELS = {
    "regression": "Regression",
    "classification_gini": "Classification (Gini)",
    "classification_entropy": "Classification (Entropy)",
}
COLORS = {
    "05": "#1f77b4",
    "10": "#9ecae1",
    "gap": "#d62728",
}


def _grouped_boxplot(ax, df: pd.DataFrame, low_col: str, high_col: str, title: str, ylabel: str):
    positions_low = np.arange(len(TASK_ORDER)) - 0.18
    positions_high = np.arange(len(TASK_ORDER)) + 0.18
    data_low = [df.loc[df["task"] == task, low_col].dropna().to_numpy(dtype=float) for task in TASK_ORDER]
    data_high = [df.loc[df["task"] == task, high_col].dropna().to_numpy(dtype=float) for task in TASK_ORDER]

    box_low = ax.boxplot(
        data_low,
        positions=positions_low,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    box_high = ax.boxplot(
        data_high,
        positions=positions_high,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box_low["boxes"]:
        patch.set_facecolor(COLORS["05"])
        patch.set_alpha(0.75)
    for patch in box_high["boxes"]:
        patch.set_facecolor(COLORS["10"])
        patch.set_alpha(0.75)

    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=12, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def _single_boxplot(ax, df: pd.DataFrame, col: str, title: str, ylabel: str):
    data = [df.loc[df["task"] == task, col].dropna().to_numpy(dtype=float) for task in TASK_ORDER]
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor(COLORS["gap"])
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(TASK_ORDER) + 1))
    ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=12, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description="Plot supplementary gain-landscape diagnostics")
    parser.add_argument(
        "--indir",
        type=str,
        default=None,
        help="Directory containing gain_landscape_dataset_summary.csv",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for the supplementary PNG",
    )
    args = parser.parse_args()

    indir = Path(args.indir) if args.indir else ANALYSIS_DIR
    outdir = Path(args.outdir) if args.outdir else SUPP_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_path = indir / "gain_landscape_dataset_summary.csv"
    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError(f"No rows found in {dataset_path}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    _grouped_boxplot(
        axes[0],
        df,
        "near_optimal_prop_05_median",
        "near_optimal_prop_10_median",
        title="Near-optimal threshold mass",
        ylabel="Dataset-level median across internal nodes",
    )
    _single_boxplot(
        axes[1],
        df,
        "relative_median_gap_median",
        title="Best-vs-median random-threshold gap",
        ylabel="Relative gap",
    )
    _grouped_boxplot(
        axes[2],
        df,
        "winner_width_05_median",
        "winner_width_10_median",
        title="Near-optimal region width",
        ylabel="Winning-feature contiguous width",
    )

    handles = [
        plt.Line2D([0], [0], color=COLORS["05"], lw=8, alpha=0.75, label="within 5% of best"),
        plt.Line2D([0], [0], color=COLORS["10"], lw=8, alpha=0.75, label="within 10% of best"),
        plt.Line2D([0], [0], color=COLORS["gap"], lw=8, alpha=0.75, label="relative median gap"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)

    outpath = outdir / OUT_NAME
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
