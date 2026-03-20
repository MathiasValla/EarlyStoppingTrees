#!/usr/bin/env python
"""
Figure 3: Dataset-regime map – who wins depending on n and p.

Tolerance rule: for each dataset, winner = method with best speedup among those
with loss ≤ τ (secretary methods only). τ = 5%, 10%, 20%.

If no non-best method satisfies loss ≤ τ, the dataset is assigned to **best** (exhaustive)
for the background / scatter, instead of a separate "none" class.

Layout options:
- **Combined (flagship):** 3×3 grid — rows = Regression, Classification (Gini), Classification (Entropy);
  columns = τ ∈ {5%, 10%, 20%}. Shared legend column. Saves ``figure3_regime_combined``.
- **Per task:** 2×2 — three τ panels + one legend panel. Saves ``figure3_regime_{regression|...}``.

x-axis: log10(n), y-axis: log10(p). Each point = one dataset, colored by winner.

REGION_ALPHA: Opacity of the KNN-filled background in [0, 1]. Passed to
matplotlib ``pcolormesh(..., alpha=REGION_ALPHA)``: 0 = fully transparent
(only scatter visible), 1 = fully opaque colored cells. Tune for overlay vs grid.
"""
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

from benchmark_results_utils import load_all, get_variant_method_order_and_colors

# KNN for dominant-region shading (pure NumPy, no sklearn)
KNN_NEIGHBORS = 7
# Default: semi-transparent overlay (original Figure 3 style). Override with --region-alpha.
DEFAULT_REGION_ALPHA = 0.35
GRID_RESOLUTION = 150

# method_key for exhaustive baseline (must match benchmark_results_utils / summaries)
BEST_METHOD_KEY = "best|"


def _knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    """Predict class for each row of X_query by k-NN majority vote. y_train are int class indices."""
    n_query = X_query.shape[0]
    n_classes = int(y_train.max()) + 1
    out = np.empty(n_query, dtype=y_train.dtype)
    for i in range(n_query):
        d = np.sqrt(np.sum((X_train - X_query[i]) ** 2, axis=1))
        idx = np.argpartition(d, k)[:k]
        neighbors = y_train[idx]
        out[i] = np.bincount(neighbors, minlength=n_classes).argmax()
    return out

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "figures"

TAU_VALUES = (0.05, 0.10, 0.20)  # 5%, 10%, 20%

# method_colors built from get_variant_method_order_and_colors (variant shades)


def _add_dataset_info(summary: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    """Add n_samples, n_features per dataset from run-level."""
    if run_df is None or "n_features" not in run_df.columns:
        return summary
    meta = run_df[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    return summary.merge(meta, on="dataset", how="left")


def _winners_per_tau(summary: pd.DataFrame, loss_col: str, tau: float) -> pd.Series:
    """
    For each dataset, winner = method_key with max speedup_median among rows with loss ≤ tau.
    Non-best only. Returns Series index=dataset, value=method_key.
    Datasets with no qualifying non-best row are filled with best (see _regime_map_panel).
    """
    sub = summary[
        (summary["splitter"] != "best")
        & (summary[loss_col] <= tau)
    ].copy()
    if sub.empty:
        return pd.Series(dtype=object)
    idx = sub.groupby("dataset")["speedup_median"].idxmax()
    return sub.loc[idx].set_index("dataset")["method_key"]


def _regime_map_panel(
    ax,
    summary: pd.DataFrame,
    loss_col: str,
    tau: float,
    title: str,
    method_order: list,
    method_colors: dict,
    *,
    region_alpha: float,
):
    """Scatter log10(n) vs log10(p), color by winner (method_key); KNN-derived regime shading behind."""
    summary = summary.dropna(subset=["n_samples", "n_features", "speedup_median", loss_col])
    if summary.empty:
        ax.set_axis_off()
        ax.set_title(title)
        return

    winners = _winners_per_tau(summary, loss_col, tau)
    if winners.empty:
        ax.set_axis_off()
        ax.set_title(title)
        return

    meta = summary[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    meta = meta.set_index("dataset")
    meta["log10_n"] = np.log10(meta["n_samples"].fillna(1).astype(float).clip(lower=0.1))
    meta["log10_p"] = np.log10(meta["n_features"].fillna(1).astype(float).clip(lower=0.1))
    # No qualifying non-best method → assign exhaustive baseline for background (not "none")
    meta["winner"] = winners.reindex(meta.index).fillna(BEST_METHOD_KEY)
    meta = meta.dropna(subset=["winner"])

    # Heatmap / KNN classes: non-best methods that can win + best (for unqualified datasets)
    order_no_best = [m for m in method_order if m != "best" and not str(m).startswith("best|")]
    class_order = order_no_best + [BEST_METHOD_KEY]
    meta["class_idx"] = meta["winner"].map({m: i for i, m in enumerate(class_order)})
    meta = meta.dropna(subset=["class_idx"]).astype({"class_idx": int})

    if meta.empty or meta["class_idx"].nunique() < 2:
        _scatter_only(ax, meta, class_order, title, method_colors)
        return

    X = meta[["log10_n", "log10_p"]].values
    y = meta["class_idx"].values
    k = min(KNN_NEIGHBORS, len(meta) - 1)
    if k < 1:
        _scatter_only(ax, meta, class_order, title, method_colors)
        return

    # Grid over data range + padding
    pad = 0.3
    x_min = meta["log10_n"].min() - pad
    x_max = meta["log10_n"].max() + pad
    y_min = meta["log10_p"].min() - pad
    y_max = meta["log10_p"].max() + pad
    xx = np.linspace(x_min, x_max, GRID_RESOLUTION)
    yy = np.linspace(y_min, y_max, GRID_RESOLUTION)
    X_grid = np.meshgrid(xx, yy)
    X_flat = np.column_stack([X_grid[0].ravel(), X_grid[1].ravel()])
    Z = _knn_predict(X, y, X_flat, k).reshape(X_grid[0].shape)

    colors = [method_colors.get(m, "#888888") for m in class_order]
    cmap = ListedColormap(colors)
    ax.pcolormesh(
        xx, yy, Z,
        cmap=cmap,
        alpha=region_alpha,
        zorder=0,
        shading="auto",
        vmin=0,
        vmax=len(class_order) - 1,
    )

    for method in class_order:
        m = meta[meta["winner"] == method]
        if m.empty:
            continue
        color = method_colors.get(method, "#888888")
        if method == BEST_METHOD_KEY or str(method).startswith("best|"):
            label = "best"
        else:
            label = method.replace("_", " ")
        if "|" in str(method) and method != BEST_METHOD_KEY:
            label = method.split("|", 1)[0].replace("_", " ") + " (" + method.split("|", 1)[1] + ")"
        ax.scatter(
            m["log10_n"],
            m["log10_p"],
            c=color,
            s=45,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=1,
            label=label,
        )

    ax.set_xlabel(r"$\log_{10}(n)$")
    ax.set_ylabel(r"$\log_{10}(p)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _scatter_only(ax, meta: pd.DataFrame, class_order: list, title: str, method_colors: dict):
    """No KNN regions; just scatter points."""
    for method in class_order:
        m = meta[meta["winner"] == method]
        if m.empty:
            continue
        color = method_colors.get(method, "#888888")
        ax.scatter(
            m["log10_n"], m["log10_p"],
            c=color, s=45, alpha=0.85, edgecolors="white", linewidths=0.5,
        )
    ax.set_xlabel(r"$\log_{10}(n)$")
    ax.set_ylabel(r"$\log_{10}(p)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


def _legend_handles(method_order: list, method_colors: dict, method_labels: list):
    """Patches + labels for regime-map legend (non-best winners + exhaustive baseline)."""
    order_no_best = [m for m in method_order if m != "best" and not str(m).startswith("best|")]
    labels_no_best = [method_labels[method_order.index(m)] for m in order_no_best] if method_labels else None
    handles = [
        Patch(facecolor=method_colors.get(m, "#888888"), edgecolor="white", label=labels_no_best[i] if labels_no_best and i < len(labels_no_best) else str(m).replace("_", " "))
        for i, m in enumerate(order_no_best)
    ]
    best_label = "best"
    if method_labels and BEST_METHOD_KEY in method_order:
        best_label = method_labels[method_order.index(BEST_METHOD_KEY)]
    handles.append(
        Patch(facecolor=method_colors.get(BEST_METHOD_KEY, "#333333"), edgecolor="white", label=best_label)
    )
    return handles


def _legend_panel(ax, method_order: list, method_colors: dict, method_labels: list):
    """Shared legend: non-best methods (potential winners) + best (fallback / background)."""
    ax.set_axis_off()
    handles = _legend_handles(method_order, method_colors, method_labels)
    ax.legend(handles=handles, loc="center", fontsize=8, frameon=True)


def main():
    p = argparse.ArgumentParser(description="Figure 3: regime maps (KNN background + scatter)")
    p.add_argument(
        "--region-alpha",
        type=float,
        default=DEFAULT_REGION_ALPHA,
        help=f"Opacity of KNN background [0,1] (default: {DEFAULT_REGION_ALPHA}). 0=transparent, 1=opaque.",
    )
    args = p.parse_args()
    region_alpha = float(np.clip(args.region_alpha, 0.0, 1.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(BENCHMARK_DIR, exclude_secretary_par=True, by_variant=True)

    method_order, method_colors, method_labels = get_variant_method_order_and_colors(
        data["regression_summary"], data["classification_gini_summary"], data["classification_entropy_summary"]
    )

    configs = [
        (
            "regression",
            data["regression_summary"],
            data["regression_run"],
            "loss_rmse_bounded_median",
            "Regression",
        ),
        (
            "classification_gini",
            data["classification_gini_summary"],
            data["classification_gini_run"],
            "loss_f1_median",
            "Classification (Gini)",
        ),
        (
            "classification_entropy",
            data["classification_entropy_summary"],
            data["classification_entropy_run"],
            "loss_f1_median",
            "Classification (Entropy)",
        ),
    ]

    prepared = []
    for tag, summary, run_df, loss_col, task_label in configs:
        if summary is None:
            continue
        summary = _add_dataset_info(summary, run_df)
        if "n_samples" not in summary.columns:
            continue
        if "variant" in summary.columns:
            summary = summary.copy()
            summary["method_key"] = summary["splitter"].astype(str) + "|" + summary["variant"].fillna("").astype(str)
        prepared.append((tag, summary, loss_col, task_label))

    # Combined flagship: 3 rows (Regression, Gini, Entropy) × 3 columns (τ = 5%, 10%, 20%) + legend column
    if len(prepared) == 3:
        fig = plt.figure(figsize=(14, 11), constrained_layout=True)
        gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.42])
        axes_grid = [[None] * 3 for _ in range(3)]
        for row in range(3):
            _tag, summary, loss_col, task_label = prepared[row]
            for col in range(3):
                tau = TAU_VALUES[col]
                if col == 0:
                    ax = fig.add_subplot(gs[row, col])
                    axes_grid[row][0] = ax
                else:
                    ax = fig.add_subplot(gs[row, col], sharex=axes_grid[row][0], sharey=axes_grid[row][0])
                    axes_grid[row][col] = ax
                title = f"{task_label}\nτ = {int(tau * 100)}% (loss ≤ {int(tau * 100)}%)"
                _regime_map_panel(
                    ax, summary, loss_col, tau, title, method_order, method_colors, region_alpha=region_alpha
                )
        leg_ax = fig.add_subplot(gs[:, 3])
        _legend_panel(leg_ax, method_order, method_colors, method_labels)
        fig.suptitle(
            "Usefulness depends on dataset structure: winner in the (n, p) plane under loss tolerance τ",
            fontsize=11,
        )
        for ext in ("pdf", "png"):
            out = OUT_DIR / f"figure3_regime_combined.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=(150 if ext == "png" else None))
        plt.close(fig)
        print("Saved figure3_regime_combined.pdf and figure3_regime_combined.png")

    for tag, summary, loss_col, task_label in prepared:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        titles = [f"{task_label} – τ = {int(t * 100)}% (loss ≤ {int(t * 100)}%)" for t in TAU_VALUES]
        for i, (tau, title) in enumerate(zip(TAU_VALUES, titles)):
            ax = axes[i // 2, i % 2]
            _regime_map_panel(
                ax, summary, loss_col, tau, title, method_order, method_colors, region_alpha=region_alpha
            )
        _legend_panel(axes[1, 1], method_order, method_colors, method_labels)

        plt.tight_layout()
        for ext in ("pdf", "png"):
            out = OUT_DIR / f"figure3_regime_{tag}.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=(150 if ext == "png" else None))
        plt.close(fig)
        print(f"Saved figure3_regime_{tag}.pdf and figure3_regime_{tag}.png")


if __name__ == "__main__":
    main()
