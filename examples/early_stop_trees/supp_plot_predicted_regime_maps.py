#!/usr/bin/env python
"""
Supplementary Figure S8: tolerance-constrained quickest-method regime maps.

For each dataset, and for each tolerance tau in {0.5%, 1%, 2.5%}, assign the
quickest method among those whose median loss stays below tau. If no
non-exhaustive method satisfies the tolerance, the dataset is assigned to the
exhaustive baseline. The figure places datasets in the (log(n), p/n) plane and
uses a light k-NN background to emphasize broad empirical regimes.
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

from benchmark_results_utils import (
    get_variant_method_order_and_colors,
    keep_secretary_par_representative,
    load_all,
    plot_grouped_variant_legend,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"

TAU_VALUES = (0.005, 0.01, 0.025)
BEST_METHOD_KEY = "best|"
KNN_NEIGHBORS = 7
GRID_RESOLUTION = 150
DEFAULT_REGION_ALPHA = 0.35


def _pct_label(tau: float) -> str:
    return f"{tau * 100:.3g}"


def _knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    """Predict class for each query row by majority vote among the k nearest datasets."""
    n_query = X_query.shape[0]
    n_classes = int(y_train.max()) + 1
    out = np.empty(n_query, dtype=y_train.dtype)
    for i in range(n_query):
        d = np.sqrt(np.sum((X_train - X_query[i]) ** 2, axis=1))
        idx = np.argpartition(d, k)[:k]
        out[i] = np.bincount(y_train[idx], minlength=n_classes).argmax()
    return out


def _add_dataset_axes(summary: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    """Merge n and p, then compute log(n) and p/n for each dataset."""
    if summary is None or run_df is None or summary.empty or run_df.empty:
        return summary
    summary = summary.copy()
    if "variant" not in summary.columns:
        summary["variant"] = ""
    summary["variant"] = summary["variant"].fillna("").astype(str)
    if "method_key" not in summary.columns:
        summary["method_key"] = summary["splitter"].astype(str) + "|" + summary["variant"]
    meta = run_df[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset").copy()
    meta["log_n"] = np.log(meta["n_samples"].fillna(1).astype(float).clip(lower=1.0))
    meta["p_over_n"] = (
        meta["n_features"].fillna(0).astype(float)
        / meta["n_samples"].fillna(1).astype(float).clip(lower=1.0)
    )
    return summary.merge(meta, on="dataset", how="left")


def _winners_per_tau(summary: pd.DataFrame, loss_col: str, tau: float) -> pd.Series:
    """
    For each dataset, keep the fastest non-exhaustive method whose median loss is <= tau.
    Datasets with no admissible non-best method are filled with the exhaustive baseline later.
    """
    sub = summary[(summary["splitter"] != "best") & (summary[loss_col] <= tau)].copy()
    if sub.empty:
        return pd.Series(dtype=object)
    idx = sub.groupby("dataset")["speedup_median"].idxmax()
    return sub.loc[idx].set_index("dataset")["method_key"]


def _scatter_only(ax, meta: pd.DataFrame, class_order: list[str], method_colors: dict, *, show_ylabel: bool):
    for method in class_order:
        m = meta[meta["winner"] == method]
        if m.empty:
            continue
        ax.scatter(
            m["log_n"],
            m["p_over_n"],
            c=method_colors.get(method, "#888888"),
            s=42,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )
    ax.set_xlabel(r"$\log(n)$")
    ax.set_ylabel(r"$p/n$" if show_ylabel else "")
    ax.grid(True, alpha=0.25)


def _quickest_regime_panel(
    ax,
    summary: pd.DataFrame,
    loss_col: str,
    tau: float,
    method_order: list[str],
    method_colors: dict,
    *,
    region_alpha: float,
    show_ylabel: bool,
):
    if summary is None or summary.empty:
        ax.set_axis_off()
        return
    summary = summary.dropna(subset=["log_n", "p_over_n", "speedup_median", loss_col]).copy()
    if summary.empty:
        ax.set_axis_off()
        return

    winners = _winners_per_tau(summary, loss_col, tau)
    meta = summary[["dataset", "log_n", "p_over_n"]].drop_duplicates("dataset").set_index("dataset")
    meta["winner"] = winners.reindex(meta.index).fillna(BEST_METHOD_KEY)

    order_no_best = [m for m in method_order if m != "best" and not str(m).startswith("best|")]
    class_order = order_no_best + [BEST_METHOD_KEY]
    meta["class_idx"] = meta["winner"].map({m: i for i, m in enumerate(class_order)})
    meta = meta.dropna(subset=["class_idx"]).astype({"class_idx": int})
    if meta.empty:
        ax.set_axis_off()
        return

    x_min = meta["log_n"].min() - 0.25
    x_max = meta["log_n"].max() + 0.25
    y_min = max(0.0, meta["p_over_n"].min() - 0.05)
    y_max = meta["p_over_n"].max() + 0.05 * max(1.0, meta["p_over_n"].max())

    if meta["class_idx"].nunique() < 2 or len(meta) <= 2:
        _scatter_only(ax, meta, class_order, method_colors, show_ylabel=show_ylabel)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        return

    X = meta[["log_n", "p_over_n"]].values
    y = meta["class_idx"].values
    k = min(KNN_NEIGHBORS, len(meta) - 1)
    if k < 1:
        _scatter_only(ax, meta, class_order, method_colors, show_ylabel=show_ylabel)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        return

    xx = np.linspace(x_min, x_max, GRID_RESOLUTION)
    yy = np.linspace(y_min, y_max, GRID_RESOLUTION)
    X_grid = np.meshgrid(xx, yy)
    X_flat = np.column_stack([X_grid[0].ravel(), X_grid[1].ravel()])
    Z = _knn_predict(X, y, X_flat, k).reshape(X_grid[0].shape)

    cmap = ListedColormap([method_colors.get(m, "#888888") for m in class_order])
    ax.pcolormesh(
        xx,
        yy,
        Z,
        cmap=cmap,
        shading="auto",
        alpha=region_alpha,
        vmin=0,
        vmax=len(class_order) - 1,
        zorder=0,
    )

    for method in class_order:
        m = meta[meta["winner"] == method]
        if m.empty:
            continue
        ax.scatter(
            m["log_n"],
            m["p_over_n"],
            c=method_colors.get(method, "#888888"),
            s=42,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

    ax.set_xlabel(r"$\log(n)$")
    ax.set_ylabel(r"$p/n$" if show_ylabel else "")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=str, default=None, help="Unused compatibility argument.")
    ap.add_argument(
        "--indir",
        type=str,
        default=None,
        help="Input benchmark-results directory (default: examples/early_stop_trees/benchmark_results).",
    )
    ap.add_argument(
        "--region-alpha",
        type=float,
        default=DEFAULT_REGION_ALPHA,
        help=f"Opacity of the k-NN background in [0, 1] (default: {DEFAULT_REGION_ALPHA}).",
    )
    args = ap.parse_args()

    indir = Path(args.indir) if args.indir else BENCHMARK_DIR
    region_alpha = float(np.clip(args.region_alpha, 0.0, 1.0))

    data = load_all(indir, exclude_secretary_par=False, by_variant=True)

    regression_summary = data["regression_summary"]
    if regression_summary is not None:
        regression_summary = regression_summary[regression_summary["splitter"] != "secretary_par"].copy()
    classification_gini_summary = keep_secretary_par_representative(data["classification_gini_summary"])
    classification_entropy_summary = keep_secretary_par_representative(data["classification_entropy_summary"])

    method_order, method_colors, method_labels = get_variant_method_order_and_colors(
        regression_summary,
        classification_gini_summary,
        classification_entropy_summary,
        include_secretary_par=True,
    )

    configs = [
        ("Regression", _add_dataset_axes(regression_summary, data["regression_run"]), "loss_rmse_bounded_median"),
        ("Gini", _add_dataset_axes(classification_gini_summary, data["classification_gini_run"]), "loss_f1_median"),
        ("Entropy", _add_dataset_axes(classification_entropy_summary, data["classification_entropy_run"]), "loss_f1_median"),
    ]

    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.5, 12.8), layout="constrained")
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 0.5], hspace=0.08, wspace=0.08)

    axes = [[None] * 3 for _ in range(3)]
    for r, (task_label, summary, loss_col) in enumerate(configs):
        for c, tau in enumerate(TAU_VALUES):
            ax = fig.add_subplot(gs[r, c])
            axes[r][c] = ax
            _quickest_regime_panel(
                ax,
                summary,
                loss_col,
                tau,
                method_order,
                method_colors,
                region_alpha=region_alpha,
                show_ylabel=(c == 0),
            )
            if r == 0:
                ax.set_title(rf"$\tau = {_pct_label(tau)}\%$", fontsize=10, fontweight="bold", pad=6)
            if c != 0:
                ax.tick_params(axis="y", labelleft=False)
        axes[r][0].annotate(
            task_label,
            xy=(-0.18, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="bold",
        )

    leg_ax = fig.add_subplot(gs[3, :])
    plot_grouped_variant_legend(
        leg_ax,
        method_order,
        method_colors,
        method_labels,
        fontsize=7,
        legend_style="point",
        y_header=0.95,
        y_top=0.82,
        y_bot=0.12,
    )

    out = SUPP_DIR / "supp_figure_08_predicted_regime_maps.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
