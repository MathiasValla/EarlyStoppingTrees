#!/usr/bin/env python
"""Figure 3: observed winner regimes over dataset size and dimensionality.

For each dataset and tolerance, the winner is the non-exhaustive method with the
largest median speedup among methods whose median loss is at most the tolerance.
If none qualifies, the dataset is assigned to the exhaustive ``best`` method.
Every ``secretary_par`` configuration is excluded from this main-figure
competition; it remains supplementary exploratory material.

The flagship layout has task columns (Regression, Gini, Entropy) and tolerance
rows (0.5%, 1%, 2.5%). The x-axis is log10(n), the y-axis is log10(p), outlined
symbols are observed datasets, and the muted field is a descriptive 7-NN
interpolation of their winner labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from benchmark_results_utils import (
    get_variant_method_order_and_colors,
    load_all,
)

# KNN for dominant-region shading (pure NumPy, no sklearn)
KNN_NEIGHBORS = 7
# Keep interpolation visibly subordinate to observed dataset markers.
DEFAULT_REGION_ALPHA = 0.18
GRID_RESOLUTION = 150
DEFAULT_PNG_DPI = 450
FULL_PAGE_WIDTH_IN = 7.35
COMBINED_HEIGHT_IN = 8.2
POINT_SIZE = 29

# method_key for exhaustive baseline (must match benchmark_results_utils / summaries)
BEST_METHOD_KEY = "best|"

# Okabe-Ito-derived family colors, supplemented by distinct brown and magenta.
# Marker shapes redundantly encode variants, so color is never the sole cue.
ACCESSIBLE_BASE_COLORS = {
    "best": "#3F3F3F",
    "secretary": "#0072B2",
    "secretary_all": "#009E73",
    "double_secretary": "#D55E00",
    "block_rank": "#A23B72",
    "prophet_1sample": "#7A5C00",
    "extra_tree": "#E69F00",
}
VARIANT_ORDERS = {
    "secretary": ("1overe", "sqrt_n", "ln_n", "0.1n"),
    "secretary_all": ("1overe", "sqrt_n", "ln_n", "0.1n"),
    "double_secretary": ("1overe", "sqrt_n", "ln_n", "0.1n"),
    "extra_tree": (
        "max_features=1",
        "max_features=1over3",
        "max_features=2over3",
        "max_features=all",
    ),
}
VARIANT_MARKERS = {
    "1overe": "o",
    "sqrt_n": "s",
    "ln_n": "D",
    "0.1n": "^",
    "max_features=1": "o",
    "max_features=1over3": "s",
    "max_features=2over3": "D",
    "max_features=all": "^",
}
SINGLETON_MARKERS = {
    "best": "X",
    "block_rank": "P",
    "prophet_1sample": "h",
}
VARIANT_TONES = (-0.10, 0.03, 0.16, 0.29)


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

TAU_VALUES = (0.005, 0.01, 0.025)  # 0.5%, 1%, 2.5%


def _pct_label(tau: float) -> str:
    """Format tolerance in percent without spurious trailing zeros."""
    return f"{tau * 100:.3g}"


def _method_parts(method_key: str) -> tuple[str, str]:
    """Split a normalized ``splitter|variant`` key."""
    parts = str(method_key).split("|", 1)
    return parts[0], parts[1] if len(parts) == 2 else ""


def _tone_color(color: str, tone: float) -> str:
    """Mix a color toward black (negative) or white (positive)."""
    rgb = np.asarray(to_rgb(color))
    target = np.ones(3) if tone >= 0 else np.zeros(3)
    mixed = rgb + (target - rgb) * abs(tone)
    return to_hex(mixed)


def _figure3_method_colors(method_order: list[str]) -> dict[str, str]:
    """Build accessible family colors with controlled variant tones."""
    colors = {}
    for method_key in method_order:
        splitter, variant = _method_parts(method_key)
        base = ACCESSIBLE_BASE_COLORS.get(splitter, "#777777")
        variants = VARIANT_ORDERS.get(splitter, ())
        tone = VARIANT_TONES[variants.index(variant)] if variant in variants else 0.0
        colors[method_key] = _tone_color(base, tone)
    return colors


def _method_marker(method_key: str) -> str:
    """Return the redundant marker encoding for a method variant."""
    splitter, variant = _method_parts(method_key)
    return VARIANT_MARKERS.get(variant, SINGLETON_MARKERS.get(splitter, "o"))


def _add_dataset_info(summary: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    """Add n_samples, n_features per dataset from run-level."""
    if run_df is None or "n_features" not in run_df.columns:
        return summary
    meta = run_df[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    return summary.merge(meta, on="dataset", how="left")


def _winners_per_tau(summary: pd.DataFrame, loss_col: str, tau: float) -> pd.Series:
    """
    For each dataset, winner = method_key with max speedup_median among rows with loss <= tau.
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
    show_title: bool = True,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Draw observed winners over a muted KNN interpolation field."""
    summary = summary.dropna(subset=["n_samples", "n_features", "speedup_median", loss_col])
    if summary.empty:
        ax.set_axis_off()
        if show_title and title:
            ax.set_title(title)
        return

    winners = _winners_per_tau(summary, loss_col, tau)
    meta = summary[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    meta = meta.set_index("dataset")
    meta["log10_n"] = np.log10(meta["n_samples"].fillna(1).astype(float).clip(lower=0.1))
    meta["log10_p"] = np.log10(meta["n_features"].fillna(1).astype(float).clip(lower=0.1))
    # No qualifying non-best method maps to the exhaustive baseline, not a "none" class.
    meta["winner"] = winners.reindex(meta.index).fillna(BEST_METHOD_KEY)
    meta = meta.dropna(subset=["winner"])

    # KNN classes are all eligible non-best variants plus the exhaustive fallback.
    order_no_best = [m for m in method_order if m != "best" and not str(m).startswith("best|")]
    class_order = order_no_best + [BEST_METHOD_KEY]
    meta["class_idx"] = meta["winner"].map({m: i for i, m in enumerate(class_order)})
    meta = meta.dropna(subset=["class_idx"]).astype({"class_idx": int})

    if meta.empty:
        ax.set_axis_off()
        if show_title and title:
            ax.set_title(title)
        return

    if xlim is None:
        xlim = _padded_limits(meta["log10_n"])
    if ylim is None:
        ylim = _padded_limits(meta["log10_p"])

    if meta["class_idx"].nunique() >= 2 and region_alpha > 0:
        X = meta[["log10_n", "log10_p"]].values
        y = meta["class_idx"].values
        k = min(KNN_NEIGHBORS, len(meta) - 1)
        if k >= 1:
            xx = np.linspace(*xlim, GRID_RESOLUTION)
            yy = np.linspace(*ylim, GRID_RESOLUTION)
            X_grid = np.meshgrid(xx, yy)
            X_flat = np.column_stack([X_grid[0].ravel(), X_grid[1].ravel()])
            Z = _knn_predict(X, y, X_flat, k).reshape(X_grid[0].shape)

            background_colors = [
                _tone_color(method_colors.get(method, "#888888"), 0.28)
                for method in class_order
            ]
            levels = np.arange(len(class_order) + 1) - 0.5
            ax.contourf(
                xx,
                yy,
                Z,
                levels=levels,
                cmap=ListedColormap(background_colors),
                alpha=region_alpha,
                antialiased=False,
                corner_mask=False,
                zorder=0,
            )

    _draw_observed_points(ax, meta, class_order, method_colors)
    _style_panel_axes(
        ax,
        title=title if show_title else "",
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _padded_limits(values: pd.Series, pad: float = 0.18) -> tuple[float, float]:
    """Return stable plotting limits around finite log-scale values."""
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        return lo - 0.5, hi + 0.5
    return lo - pad, hi + pad


def _summary_log_limits(summary: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute task limits from unique observed dataset metadata."""
    meta = summary[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    log_n = np.log10(meta["n_samples"].astype(float).clip(lower=0.1))
    log_p = np.log10(meta["n_features"].astype(float).clip(lower=0.1))
    return _padded_limits(log_n), _padded_limits(log_p)


def _draw_observed_points(ax, meta: pd.DataFrame, class_order: list, method_colors: dict):
    """Draw observations with a dark hairline and white separation halo."""
    for method in class_order:
        observed = meta[meta["winner"] == method]
        if observed.empty:
            continue
        marker = _method_marker(method)
        x = observed["log10_n"]
        y = observed["log10_p"]
        ax.scatter(
            x,
            y,
            marker=marker,
            s=POINT_SIZE + 17,
            facecolors="white",
            edgecolors="#252525",
            linewidths=0.45,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            marker=marker,
            s=POINT_SIZE,
            facecolors=method_colors.get(method, "#777777"),
            edgecolors="none",
            zorder=3.1,
        )


def _style_panel_axes(ax, *, title: str, show_xlabel: bool, show_ylabel: bool):
    """Apply publication-scale axis styling consistently across panels."""
    ax.set_facecolor("#FCFCFA")
    ax.set_axisbelow(True)
    ax.grid(True, color="#D5D5CF", linewidth=0.45, alpha=0.75)
    ax.set_box_aspect(0.72)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.65, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#404040")
        spine.set_linewidth(0.65)
    ax.set_xlabel(r"Dataset size, $\log_{10}(n)$" if show_xlabel else "", fontsize=9)
    ax.set_ylabel(r"Feature count, $\log_{10}(p)$" if show_ylabel else "", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)


def _legend_handles(method_order: list, method_colors: dict, method_labels: list):
    """Build marker-aware handles for the per-task review figures."""
    order_no_best = [m for m in method_order if m != "best" and not str(m).startswith("best|")]
    labels_no_best = [method_labels[method_order.index(m)] for m in order_no_best] if method_labels else None

    def _point_handle(method: str, lab: str):
        return Line2D(
            [0],
            [0],
            linestyle="none",
            marker=_method_marker(method),
            markersize=7,
            markerfacecolor=method_colors.get(method, "#777777"),
            markeredgecolor="#252525",
            markeredgewidth=0.55,
            label=lab,
        )

    handles = [
        _point_handle(
            m,
            labels_no_best[i] if labels_no_best and i < len(labels_no_best) else str(m).replace("_", " "),
        )
        for i, m in enumerate(order_no_best)
    ]
    best_label = "best"
    if method_labels and BEST_METHOD_KEY in method_order:
        best_label = method_labels[method_order.index(BEST_METHOD_KEY)]
    handles.append(_point_handle(BEST_METHOD_KEY, best_label))
    return handles


def _legend_panel(ax, method_order: list, method_colors: dict, method_labels: list, ncol=None):
    """Shared legend for per-task review figures."""
    ax.set_axis_off()
    handles = _legend_handles(method_order, method_colors, method_labels)
    n = len(handles)
    if ncol is None:
        ncol = min(6, max(4, int(np.ceil(n / 4))))
    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        fontsize=7.5,
        frameon=True,
        columnspacing=0.8,
        handlelength=1.0,
    )


def _compact_legend_label(method_key: str) -> str:
    """Return compact, unambiguous labels for the grouped main legend."""
    splitter, variant = _method_parts(method_key)
    schedule_labels = {
        "1overe": r"$f=1/e$",
        "sqrt_n": r"$f=1/\sqrt{n_{\mathrm{node}}}$",
        "ln_n": r"$f=1/\log N_{\mathrm{data}}$",
        "0.1n": r"$f=0.1$",
    }
    mtry_labels = {
        "max_features=1": r"$m_{\rm try}=1$",
        "max_features=1over3": r"$m_{\rm try}=p/3$",
        "max_features=2over3": r"$m_{\rm try}=2p/3$",
        "max_features=all": r"$m_{\rm try}=p$",
    }
    singleton_labels = {
        "best": "Exhaustive (best)",
        "block_rank": "Rank-inspired",
        "prophet_1sample": "Prophet-style",
    }
    if splitter == "extra_tree":
        return mtry_labels.get(variant, variant)
    if splitter in VARIANT_ORDERS:
        return schedule_labels.get(variant, variant)
    return singleton_labels.get(splitter, splitter.replace("_", " "))


def _combined_legend_panel(ax, method_order: list, method_colors: dict):
    """Draw the grouped, marker-aware legend and interpretation note."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    family_order = ("secretary", "secretary_all", "double_secretary", "extra_tree")
    groups = {family: [] for family in family_order}
    groups["others"] = []
    for method_key in method_order:
        splitter, variant = _method_parts(method_key)
        target = splitter if splitter in family_order else "others"
        groups[target].append(method_key)

    for family in family_order:
        variants = VARIANT_ORDERS[family]
        groups[family].sort(key=lambda key: variants.index(_method_parts(key)[1]))
    other_priority = {"best": 0, "block_rank": 1, "prophet_1sample": 2}
    groups["others"].sort(key=lambda key: other_priority.get(_method_parts(key)[0], 99))

    headers = {
        "secretary": "Secretary (S)",
        "secretary_all": r"$S_{\rm all}$",
        "double_secretary": r"$S^2$",
        "extra_tree": "Extra Trees (ERT)",
        "others": "Other methods",
    }
    widths = np.asarray([0.175, 0.175, 0.175, 0.205, 0.27])
    starts = np.r_[0.0, np.cumsum(widths[:-1])]

    for start, width, family in zip(starts, widths, (*family_order, "others")):
        center = start + width / 2
        ax.text(
            center,
            0.92,
            headers[family],
            ha="center",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            transform=ax.transAxes,
        )
        entries = groups[family]
        y_positions = np.linspace(0.72, 0.29, len(entries)) if entries else []
        for method_key, y in zip(entries, y_positions):
            marker_x = start + width * 0.08
            ax.scatter(
                [marker_x],
                [y],
                marker=_method_marker(method_key),
                s=38,
                facecolors=method_colors.get(method_key, "#777777"),
                edgecolors="#252525",
                linewidths=0.55,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                start + width * 0.19,
                y,
                _compact_legend_label(method_key),
                ha="left",
                va="center",
                fontsize=8.0,
                transform=ax.transAxes,
            )

    ax.text(
        0.5,
        0.12,
        r"Outlined symbols: observed datasets; winner = fastest method with loss $\leq\tau$ "
        "(exhaustive if none qualifies).",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#303030",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.015,
        f"Background tint: descriptive {KNN_NEIGHBORS}-NN interpolation of observed winner labels "
        "(not model estimates).",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#303030",
        transform=ax.transAxes,
    )


def _save_figure_pair(fig, outdir: Path, stem: str, png_dpi: int) -> tuple[Path, Path]:
    """Save a vector PDF and exact-size high-DPI PNG."""
    pdf_path = outdir / f"{stem}.pdf"
    png_path = outdir / f"{stem}.png"
    fig.savefig(
        pdf_path,
        facecolor="white",
        metadata={
            "Title": "Figure 3 - observed dataset winner regimes",
            "Subject": "Observed winners with descriptive 7-NN background interpolation",
        },
    )
    fig.savefig(png_path, dpi=png_dpi, facecolor="white")
    return pdf_path, png_path


def main():
    p = argparse.ArgumentParser(
        description="Figure 3: observed regime maps with descriptive KNN backgrounds"
    )
    p.add_argument(
        "--indir",
        type=str,
        default=None,
        help="Input directory with benchmark_results (default: examples/early_stop_trees/benchmark_results).",
    )
    p.add_argument(
        "--region-alpha",
        type=float,
        default=DEFAULT_REGION_ALPHA,
        help=f"Opacity of KNN background [0,1] (default: {DEFAULT_REGION_ALPHA}). 0=transparent, 1=opaque.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: examples/early_stop_trees/figures).",
    )
    p.add_argument(
        "--png-dpi",
        type=int,
        default=DEFAULT_PNG_DPI,
        help=f"PNG resolution in dots per inch (default: {DEFAULT_PNG_DPI}).",
    )
    p.add_argument(
        "--combined-only",
        action="store_true",
        help="Generate only the 3-by-3 main Figure 3 assets.",
    )
    args = p.parse_args()
    region_alpha = float(np.clip(args.region_alpha, 0.0, 1.0))
    if args.png_dpi < 1:
        p.error("--png-dpi must be positive")
    indir = Path(args.indir) if args.indir is not None else BENCHMARK_DIR
    outdir = Path(args.outdir) if args.outdir is not None else OUT_DIR

    outdir.mkdir(parents=True, exist_ok=True)
    data = load_all(indir, exclude_secretary_par=True, by_variant=True)

    regression_summary = data["regression_summary"]
    classification_gini_summary = data["classification_gini_summary"]
    classification_entropy_summary = data["classification_entropy_summary"]

    method_order, _utility_colors, method_labels = get_variant_method_order_and_colors(
        regression_summary,
        classification_gini_summary,
        classification_entropy_summary,
        include_secretary_par=False,
    )
    if any(_method_parts(method)[0] == "secretary_par" for method in method_order):
        raise RuntimeError("secretary_par must not enter the main Figure 3 competition")
    method_colors = _figure3_method_colors(method_order)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    configs = [
        (
            "regression",
            regression_summary,
            data["regression_run"],
            "loss_rmse_bounded_median",
            "Regression",
        ),
        (
            "classification_gini",
            classification_gini_summary,
            data["classification_gini_run"],
            "loss_f1_median",
            "Gini",
        ),
        (
            "classification_entropy",
            classification_entropy_summary,
            data["classification_entropy_run"],
            "loss_f1_median",
            "Entropy",
        ),
    ]

    prepared = []
    for tag, summary, run_df, loss_col, task_label in configs:
        if summary is None:
            continue
        summary = _add_dataset_info(summary, run_df)
        if "n_samples" not in summary.columns:
            continue
        summary = summary.copy()
        if summary["splitter"].eq("secretary_par").any():
            raise RuntimeError(f"secretary_par remained in the filtered {tag} summary")
        variant = summary["variant"].fillna("").astype(str) if "variant" in summary else ""
        summary["method_key"] = summary["splitter"].astype(str) + "|" + variant
        prepared.append((tag, summary, loss_col, task_label))

    # Combined flagship: task columns, tolerance rows, then a dedicated legend band.
    if len(prepared) == 3:
        task_limits = [_summary_log_limits(summary) for _, summary, _, _ in prepared]
        global_ylim = (
            min(limits[1][0] for limits in task_limits),
            max(limits[1][1] for limits in task_limits),
        )

        fig = plt.figure(figsize=(FULL_PAGE_WIDTH_IN, COMBINED_HEIGHT_IN), facecolor="white")
        outer = GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[3.25, 1.0],
            hspace=0.17,
            left=0.115,
            right=0.985,
            bottom=0.035,
            top=0.955,
        )
        panel_gs = outer[0].subgridspec(3, 3, hspace=0.15, wspace=0.12)
        axes_grid = [[None] * 3 for _ in range(3)]
        col_titles = ["Regression", "Gini", "Entropy"]
        row_titles = [rf"$\tau={_pct_label(t)}\%$" for t in TAU_VALUES]
        # prepared order matches configs: [regression, gini, entropy]
        for r in range(3):
            tau = TAU_VALUES[r]
            for c in range(3):
                _tag, summary, loss_col, _task_label = prepared[c]
                sharex = axes_grid[0][c] if r > 0 else None
                sharey = axes_grid[0][0] if (r, c) != (0, 0) else None
                ax = fig.add_subplot(panel_gs[r, c], sharex=sharex, sharey=sharey)
                axes_grid[r][c] = ax
                _regime_map_panel(
                    ax,
                    summary,
                    loss_col,
                    tau,
                    col_titles[c] if r == 0 else "",
                    method_order,
                    method_colors,
                    region_alpha=region_alpha,
                    show_title=True,
                    show_xlabel=False,
                    show_ylabel=False,
                    xlim=task_limits[c][0],
                    ylim=global_ylim,
                )

        for r in range(3):
            axes_grid[r][0].annotate(
                row_titles[r],
                xy=(-0.245, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                rotation=90,
                fontsize=9.5,
                fontweight="bold",
                annotation_clip=False,
            )

        for r in range(3):
            for c in range(3):
                ax = axes_grid[r][c]
                ax.tick_params(labelbottom=(r == 2), labelleft=(c == 0))

        legend_ax = fig.add_subplot(outer[1])
        _combined_legend_panel(legend_ax, method_order, method_colors)

        # Position shared labels from the realized panel bounds to prevent collisions.
        fig.canvas.draw()
        panel_boxes = [ax.get_position() for row in axes_grid for ax in row]
        panel_left = min(box.x0 for box in panel_boxes)
        panel_right = max(box.x1 for box in panel_boxes)
        panel_bottom = min(box.y0 for box in panel_boxes)
        panel_top = max(box.y1 for box in panel_boxes)
        fig.text(
            (panel_left + panel_right) / 2,
            panel_bottom - 0.044,
            r"Dataset size, $\log_{10}(n)$",
            ha="center",
            va="center",
            fontsize=9.5,
        )
        fig.text(
            0.018,
            (panel_bottom + panel_top) / 2,
            r"Feature count, $\log_{10}(p)$",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.5,
        )

        pdf_path, png_path = _save_figure_pair(
            fig, outdir, "figure3_regime_combined", args.png_dpi
        )
        plt.close(fig)
        print(f"Saved {pdf_path}")
        print(f"Saved {png_path}")
        print(f"Main Figure 3 legend entries: {len(method_order)} (secretary_par excluded)")

    if args.combined_only:
        return

    for tag, summary, loss_col, task_label in prepared:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        titles = [
            rf"{task_label} - $\tau={_pct_label(t)}\%$ (loss $\leq{_pct_label(t)}\%$)"
            for t in TAU_VALUES
        ]
        for i, (tau, title) in enumerate(zip(TAU_VALUES, titles)):
            ax = axes[i // 2, i % 2]
            _regime_map_panel(
                ax, summary, loss_col, tau, title, method_order, method_colors, region_alpha=region_alpha
            )
        _legend_panel(axes[1, 1], method_order, method_colors, method_labels)

        plt.tight_layout()
        pdf_path, png_path = _save_figure_pair(
            fig, outdir, f"figure3_regime_{tag}", args.png_dpi
        )
        plt.close(fig)
        print(f"Saved {pdf_path}")
        print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
