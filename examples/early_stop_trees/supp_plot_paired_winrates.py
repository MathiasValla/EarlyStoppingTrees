#!/usr/bin/env python
"""
Supplementary: paired dataset-level comparisons (win rates vs reference).

We use the paired-analysis CSVs (median deltas across datasets) produced by:
  examples/early_stop_trees/analysis_paired.py

For each method_key, win_rate is computed over datasets as:
- speedup: delta = speedup(method) - speedup(best) => win if delta > 0
- loss:    delta = loss(method) - loss(best)       => win if delta < 0

Outputs:
- one PDF per task with two panels: speedup win-rate and loss win-rate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from benchmark_results_utils import get_variant_method_order_and_colors
from analysis_utils import method_sort_key


def _method_label(method_key: str) -> str:
    if "|" not in method_key:
        return method_key.replace("_", " ")
    s, v = method_key.split("|", 1)
    if v:
        return f"{s.replace('_', ' ')} ({v})"
    return s.replace("_", " ")


def _methods_to_dummy_summary_df(method_keys: list[str]) -> pd.DataFrame:
    rows = []
    for mk in method_keys:
        if "|" not in mk:
            rows.append({"splitter": mk, "variant": ""})
        else:
            s, v = mk.split("|", 1)
            rows.append({"splitter": s, "variant": v})
    return pd.DataFrame(rows)


def _error_bar_binom(p: np.ndarray, n: np.ndarray):
    # Wald standard error
    se = np.sqrt(p * (1 - p) / np.maximum(n, 1))
    return se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=str, default=None, help="analysis_additional directory")
    ap.add_argument("--task", type=str, default="regression", help="regression|classification_gini|classification_entropy")
    ap.add_argument("--top-k", type=int, default=8, help="Keep top-k methods by speedup win-rate (exclude best)")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: analysis-dir/supplementary)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else (script_dir / "analysis_additional")
    paired_dir = analysis_dir / "paired"

    outdir = Path(args.outdir) if args.outdir else (analysis_dir / "supplementary")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    speed_csv = paired_dir / f"{args.task}_paired_summary_speedup_vs_best__.csv"
    loss_csv = paired_dir / f"{args.task}_paired_summary_loss_vs_best__.csv"
    speed = pd.read_csv(speed_csv)
    loss = pd.read_csv(loss_csv)

    speed = speed[speed["method_key"].astype(str) != "best|"].copy()
    speed = speed.sort_values("win_rate", ascending=False)
    top_methods = speed.head(args.top_k)["method_key"].astype(str).tolist()

    # Colors consistent with paper style
    dummy = _methods_to_dummy_summary_df(top_methods)
    _, colors, labels = get_variant_method_order_and_colors(dummy)
    # labels correspond to keys returned by get_variant...; build mapping
    # (get_variant returns labels in same order as keys)
    method_order = list(colors.keys())
    label_map = {k: lab for k, lab in zip(method_order, labels)}

    # Ensure consistent order in plot: by speedup win_rate
    speed = speed[speed["method_key"].isin(top_methods)].set_index("method_key").loc[top_methods].reset_index()
    loss = loss[loss["method_key"].isin(top_methods)].set_index("method_key").loc[top_methods].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    x = np.arange(len(top_methods))

    for ax, df, panel_title, metric_is_loss in [
        (axes[0], speed, "Speedup win-rate vs best", False),
        (axes[1], loss, "Loss win-rate vs best", True),
    ]:
        p = df["win_rate"].to_numpy(dtype=float)
        n = df["n_pairs"].to_numpy(dtype=float)
        se = _error_bar_binom(p, n)

        bar_colors = [colors.get(mk, "#555555") for mk in top_methods]
        ax.bar(x, p, color=bar_colors, edgecolor="white", linewidth=0.5, yerr=se, capsize=2)
        ax.set_title(panel_title)
        ax.set_xticks(x)
        ax.set_xticklabels([label_map.get(mk, _method_label(mk)) for mk in top_methods], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(0.5, color="#999999", linewidth=1.0, alpha=0.9)
        ax.set_ylabel("Win fraction across datasets" if ax is axes[0] else "")

    fig.suptitle(f"Paired dataset-level comparisons (task={args.task})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = outdir / "figures" / f"{args.task}_supp_paired_winrates.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote supplementary paired win-rate figure: {outpath}")


if __name__ == "__main__":
    main()

