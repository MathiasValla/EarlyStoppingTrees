#!/usr/bin/env python
"""
Supplementary: predicted regime maps from meta-regression.

Reads the CSV grid produced by:
  examples/early_stop_trees/analysis_meta_regression.py

For each task, we render:
- predicted best method for speedup median (via spline predictions)
- predicted best method for loss median (via spline predictions)

Outputs:
- one PDF per task with two panels (speedup + loss)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from benchmark_results_utils import get_variant_method_order_and_colors


def _methods_to_dummy_summary_df(method_keys: list[str]) -> pd.DataFrame:
    rows = []
    for mk in method_keys:
        if "|" not in mk:
            rows.append({"splitter": mk, "variant": ""})
        else:
            s, v = mk.split("|", 1)
            rows.append({"splitter": s, "variant": v})
    return pd.DataFrame(rows)


def _method_label(method_key: str) -> str:
    if "|" not in method_key:
        return method_key.replace("_", " ")
    s, v = method_key.split("|", 1)
    if v:
        return f"{s.replace('_', ' ')} ({v})"
    return s.replace("_", " ")


def _plot_one_heatmap(ax, grid_df: pd.DataFrame, *, value_col: str, title: str, colors: dict, method_order: list[str]):
    sub = grid_df[["log_n", "p_over_n", value_col]].dropna().copy()
    # Pivot to a dense matrix (index = p_over_n, columns = log_n)
    x_vals = np.sort(sub["log_n"].unique())
    y_vals = np.sort(sub["p_over_n"].unique())
    pivot = sub.pivot(index="p_over_n", columns="log_n", values=value_col)
    pivot = pivot.reindex(index=y_vals, columns=x_vals)

    # Map categorical labels to integer classes
    z = np.full(pivot.shape, -1, dtype=int)
    for i, mk in enumerate(method_order):
        z[pivot.to_numpy() == mk] = i

    cmap = ListedColormap([colors.get(mk, "#888888") for mk in method_order])

    # Use numeric indices since pcolormesh doesn't accept categorical values
    ax.pcolormesh(x_vals, y_vals, z, cmap=cmap, shading="auto", alpha=0.95, vmin=0, vmax=len(method_order) - 1)

    ax.set_xlabel(r"log(n) (natural log)")
    ax.set_ylabel(r"p/n")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # Legend
    handles = [Patch(facecolor=colors.get(mk, "#888888"), edgecolor="white", label=_method_label(mk)) for mk in method_order]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=str, default=None, help="analysis_additional directory")
    ap.add_argument("--task", type=str, default="regression", help="regression|classification_gini|classification_entropy")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: analysis-dir/supplementary)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else (script_dir / "analysis_additional")
    meta_dir = analysis_dir / "meta"

    outdir = Path(args.outdir) if args.outdir else (analysis_dir / "supplementary")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    csv_path = meta_dir / f"{args.task}_regime_map_predicted_spline.csv"
    grid_df = pd.read_csv(csv_path)

    # Collect methods from both columns to get a stable color scheme.
    cols = ["best_method_speedup", "best_method_loss"]
    methods = []
    for c in cols:
        if c in grid_df.columns:
            methods.extend(grid_df[c].dropna().astype(str).unique().tolist())
    methods = sorted(set(methods))

    dummy = _methods_to_dummy_summary_df(methods)
    method_order, colors, _labels = get_variant_method_order_and_colors(dummy)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _plot_one_heatmap(
        axes[0],
        grid_df,
        value_col="best_method_speedup",
        title=f"{args.task}: predicted best (speedup)",
        colors=colors,
        method_order=method_order,
    )
    _plot_one_heatmap(
        axes[1],
        grid_df,
        value_col="best_method_loss",
        title=f"{args.task}: predicted best (loss)",
        colors=colors,
        method_order=method_order,
    )

    fig.suptitle("Predicted regime maps from dataset-level meta-regression", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = outdir / "figures" / f"{args.task}_supp_predicted_regime_map.pdf"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote supplementary predicted regime map: {outpath}")


if __name__ == "__main__":
    main()

