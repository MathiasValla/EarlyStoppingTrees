#!/usr/bin/env python
"""
Supplementary: predicted regime maps from meta-regression (merged).

Reads CSV grids from ``analysis_additional/meta/`` (see ``analysis_meta_regression.py``).

Produces one figure: 3 tasks × 2 metrics (speedup | loss), single shared legend, PNG only.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from benchmark_results_utils import get_variant_method_order_and_colors

SCRIPT_DIR = Path(__file__).resolve().parent
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"


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


def _plot_one_heatmap(
    ax,
    grid_df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    colors: dict,
    method_order: list[str],
    show_legend: bool = True,
):
    sub = grid_df[["log_n", "p_over_n", value_col]].dropna().copy()
    x_vals = np.sort(sub["log_n"].unique())
    y_vals = np.sort(sub["p_over_n"].unique())
    pivot = sub.pivot(index="p_over_n", columns="log_n", values=value_col)
    pivot = pivot.reindex(index=y_vals, columns=x_vals)

    z = np.full(pivot.shape, -1, dtype=int)
    for i, mk in enumerate(method_order):
        z[pivot.to_numpy() == mk] = i

    cmap = ListedColormap([colors.get(mk, "#888888") for mk in method_order])

    ax.pcolormesh(x_vals, y_vals, z, cmap=cmap, shading="auto", alpha=0.95, vmin=0, vmax=len(method_order) - 1)

    ax.set_xlabel(r"log(n) (natural log)")
    ax.set_ylabel(r"p/n")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)

    if show_legend:
        handles = [Patch(facecolor=colors.get(mk, "#888888"), edgecolor="white", label=_method_label(mk)) for mk in method_order]
        ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=str, default=None, help="analysis_additional directory")
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for legacy per-task PDFs (optional; merged PNG always goes to SUPP_FIGURES/).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else (script_dir / "analysis_additional")
    meta_dir = analysis_dir / "meta"

    tasks = [
        ("regression", "Regression"),
        ("classification_gini", "Gini"),
        ("classification_entropy", "Entropy"),
    ]

    all_methods: set[str] = set()
    grids: dict[str, pd.DataFrame] = {}
    for task_key, _ in tasks:
        csv_path = meta_dir / f"{task_key}_regime_map_predicted_spline.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing {csv_path}")
        grid_df = pd.read_csv(csv_path)
        grids[task_key] = grid_df
        for c in ("best_method_speedup", "best_method_loss"):
            if c in grid_df.columns:
                all_methods.update(grid_df[c].dropna().astype(str).unique().tolist())

    methods = sorted(all_methods)
    dummy = _methods_to_dummy_summary_df(methods)
    method_order, colors, _labels = get_variant_method_order_and_colors(dummy)

    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 14), layout="constrained")
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.35], hspace=0.18, wspace=0.12)

    for row, (task_key, task_name) in enumerate(tasks):
        grid_df = grids[task_key]
        ax_sp = fig.add_subplot(gs[row, 0])
        ax_lo = fig.add_subplot(gs[row, 1])
        _plot_one_heatmap(
            ax_sp,
            grid_df,
            value_col="best_method_speedup",
            title=f"{task_name}: predicted best (speedup)",
            colors=colors,
            method_order=method_order,
            show_legend=False,
        )
        _plot_one_heatmap(
            ax_lo,
            grid_df,
            value_col="best_method_loss",
            title=f"{task_name}: predicted best (loss)",
            colors=colors,
            method_order=method_order,
            show_legend=False,
        )

    leg_ax = fig.add_subplot(gs[3, :])
    leg_ax.set_axis_off()
    handles = [Patch(facecolor=colors.get(mk, "#888888"), edgecolor="white", label=_method_label(mk)) for mk in method_order]
    leg_ax.legend(handles=handles, loc="center", ncol=min(4, len(handles)), fontsize=7, frameon=True)

    out_merged = SUPP_DIR / "supp_figure_08_predicted_regime_maps.png"
    fig.savefig(out_merged, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_merged}")

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "figures").mkdir(parents=True, exist_ok=True)
        for task_key, _ in tasks:
            grid_df = grids[task_key]
            fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
            _plot_one_heatmap(
                axes2[0],
                grid_df,
                value_col="best_method_speedup",
                title=f"{task_key}: predicted best (speedup)",
                colors=colors,
                method_order=method_order,
                show_legend=True,
            )
            _plot_one_heatmap(
                axes2[1],
                grid_df,
                value_col="best_method_loss",
                title=f"{task_key}: predicted best (loss)",
                colors=colors,
                method_order=method_order,
                show_legend=True,
            )
            fig2.tight_layout()
            legacy = outdir / "figures" / f"{task_key}_supp_predicted_regime_map.pdf"
            fig2.savefig(legacy, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"Wrote legacy {legacy}")


if __name__ == "__main__":
    main()
