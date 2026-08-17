#!/usr/bin/env python
"""Dataset-level reliability figures for the early-stopping benchmark.

Every plotted observation is a dataset/method point. Run-level outcomes are
first paired to exhaustive CART within the same run by
``benchmark_results_utils``; speedup and predictive loss are then summarized
by their medians across runs. Reliability is the equal-weight fraction of
dataset points meeting a prespecified criterion, never a fraction of runs.

The main figure combines:

* loss-only reliability at tolerances 0.5%, 1%, and 2.5%; and
* joint reliability at the same loss tolerances and minimum time savings of
  0%, 10%, 25%, and 50%.

If ``--inference-dir`` contains ``joint_reliability_intervals.csv``, the joint
profiles include the selected bootstrap interval scope. Otherwise the output
is visibly marked as a point-estimate preview. S_par is excluded throughout;
its screened configurations remain an exploratory analysis elsewhere.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from benchmark_results_utils import (
    get_classification_run_level,
    get_regression_run_level,
    get_variant_method_order_and_colors,
    plot_grouped_variant_legend,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "figures"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"

SMALL_SIZE = 2500
LARGE_SIZE = 25000
TIME_SAVED_THRESHOLDS_PCT = (0.0, 10.0, 25.0, 50.0)
LOSS_TOLERANCES_PCT = (0.5, 1.0, 2.5)

TASK_ORDER = (
    "regression",
    "classification_gini",
    "classification_entropy",
)
MAIN_REPRESENTATIVES = (
    "best|",
    "secretary|1overe",
    "double_secretary|1overe",
    "secretary_all|1overe",
    "block_rank|",
    "prophet_1sample|",
    "extra_tree|max_features=1",
    "extra_tree|max_features=1over3",
    "extra_tree|max_features=2over3",
    "extra_tree|max_features=all",
)
TASK_CONFIG = {
    "regression": {
        "short_title": "Regression",
        "panel_title": "Regression\nbounded relative RMSE loss (%)",
        "loss_col": "loss_rmse_bounded",
        "loss_unit": "percent_bounded_relative_RMSE_loss",
        "tolerance_label": r"Tolerance $\varepsilon$ (%)",
    },
    "classification_gini": {
        "short_title": "Classification (Gini)",
        "panel_title": "Classification (Gini)\nweighted-F1 loss (pp)",
        "loss_col": "loss_f1",
        "loss_unit": "F1_percentage_points",
        "tolerance_label": r"Tolerance $\varepsilon$ (F1 pp)",
    },
    "classification_entropy": {
        "short_title": "Classification (Entropy)",
        "panel_title": "Classification (Entropy)\nweighted-F1 loss (pp)",
        "loss_col": "loss_f1",
        "loss_unit": "F1_percentage_points",
        "tolerance_label": r"Tolerance $\varepsilon$ (F1 pp)",
    },
}

# The one-per-family operating choices are drawn more strongly, while all
# non-parametric variants remain visible and are identified in the legend.
EMPHASIZED_METHOD_KEYS = frozenset(
    {
        "best|",
        "secretary|1overe",
        "secretary_all|1overe",
        "double_secretary|1overe",
        "block_rank|",
        "prophet_1sample|",
        "extra_tree|max_features=all",
    }
)


def _time_saved_pct(speedup: np.ndarray | pd.Series) -> np.ndarray:
    """Convert speedup to percentage training time saved."""
    values = np.asarray(speedup, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        saved = 100.0 * (1.0 - 1.0 / values)
    saved[~np.isfinite(saved)] = np.nan
    return saved


def _add_method_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize variants and attach the method identifier used by inference."""
    out = frame.copy()
    if "variant" not in out.columns:
        out["variant"] = ""
    out["variant"] = out["variant"].fillna("").astype(str)
    out["method_key"] = out["splitter"].astype(str) + "|" + out["variant"]
    return out


def _dataset_median_points(
    run_df: pd.DataFrame,
    *,
    task: str,
) -> pd.DataFrame:
    """Return one paired-run-median point per dataset and method.

    ``speedup`` and the task-specific loss already contrast each method with
    exhaustive CART in the same dataset/run. The speedup median is transformed
    only after aggregation, matching ``analysis_inference.py`` exactly.
    """
    if run_df is None or run_df.empty:
        raise FileNotFoundError(f"No run-level rows available for {task}")
    loss_col = TASK_CONFIG[task]["loss_col"]
    required = {
        "run",
        "dataset",
        "splitter",
        "n_samples",
        "n_features",
        "speedup",
        loss_col,
    }
    missing = sorted(required - set(run_df.columns))
    if missing:
        raise ValueError(f"{task} run data is missing columns: {missing}")

    frame = _add_method_key(run_df)
    duplicate = frame.duplicated(["dataset", "run", "method_key"], keep=False)
    if duplicate.any():
        sample = frame.loc[
            duplicate, ["dataset", "run", "method_key"]
        ].head()
        raise ValueError(
            "Duplicate dataset/run/method rows would overweight seeds:\n"
            f"{sample}"
        )

    frame["speedup"] = pd.to_numeric(frame["speedup"], errors="coerce")
    frame[loss_col] = pd.to_numeric(frame[loss_col], errors="coerce")

    # Use one complete run block across every method in a dataset. This is the
    # observed-point counterpart of the paired bootstrap in analysis_inference.
    paired_blocks = []
    for dataset, dataset_rows in frame.groupby("dataset", sort=True):
        speedup = dataset_rows.pivot(
            index="run", columns="method_key", values="speedup"
        )
        loss = dataset_rows.pivot(
            index="run", columns="method_key", values=loss_col
        )
        common_methods = sorted(set(speedup.columns) & set(loss.columns))
        speedup = speedup.reindex(columns=common_methods)
        loss = loss.reindex(columns=common_methods)
        complete = (
            np.isfinite(speedup.to_numpy(dtype=float)).all(axis=1)
            & np.isfinite(loss.to_numpy(dtype=float)).all(axis=1)
        )
        common_runs = speedup.index[complete]
        if common_runs.empty:
            raise ValueError(f"No complete paired runs for {task}/{dataset}")
        paired_blocks.append(dataset_rows[dataset_rows["run"].isin(common_runs)])
    paired = pd.concat(paired_blocks, ignore_index=True)
    if paired.empty:
        raise ValueError(f"No finite paired speedup/loss rows for {task}")

    group_cols = ["dataset", "splitter", "variant", "method_key"]
    points = (
        paired.groupby(group_cols, as_index=False, sort=True)
        .agg(
            speedup_median=("speedup", "median"),
            loss_raw_median=(loss_col, "median"),
            n_paired_runs=("run", "nunique"),
            n_samples=("n_samples", "first"),
            n_features=("n_features", "first"),
        )
        .reset_index(drop=True)
    )
    points["time_saved_pct"] = _time_saved_pct(points["speedup_median"])
    points["loss_pct"] = 100.0 * points["loss_raw_median"].astype(float)
    points["_size"] = (
        pd.to_numeric(points["n_samples"], errors="coerce")
        * pd.to_numeric(points["n_features"], errors="coerce")
    )
    points["task"] = task
    points["loss_unit"] = TASK_CONFIG[task]["loss_unit"]
    return points


def _filter_points_by_size(points: pd.DataFrame, size_filter: str) -> pd.DataFrame:
    """Filter dataset points using the established n_samples * n_features cutoffs."""
    if size_filter == "all":
        return points.copy()
    if size_filter == "small":
        return points[points["_size"] <= SMALL_SIZE].copy()
    if size_filter == "large":
        return points[points["_size"] >= LARGE_SIZE].copy()
    raise ValueError(f"Unknown size filter: {size_filter}")


def _loss_reliability_table(points: pd.DataFrame) -> pd.DataFrame:
    """Fraction of dataset median points below each loss tolerance."""
    rows = []
    for method_key, method_points in points.groupby("method_key", sort=False):
        losses = method_points["loss_pct"].to_numpy(dtype=float)
        losses = losses[np.isfinite(losses)]
        if losses.size == 0:
            continue
        for tolerance in LOSS_TOLERANCES_PCT:
            rows.append(
                {
                    "task": str(method_points["task"].iloc[0]),
                    "method_key": str(method_key),
                    "loss_tolerance_pct": float(tolerance),
                    "estimate": float(np.mean(losses <= tolerance)),
                    "n_datasets": int(losses.size),
                    "experimental_unit": "dataset",
                }
            )
    return pd.DataFrame(rows)


def _joint_reliability_table(points: pd.DataFrame) -> pd.DataFrame:
    """Fraction of dataset median points meeting each joint operating point."""
    rows = []
    for method_key, method_points in points.groupby("method_key", sort=False):
        time_saved = method_points["time_saved_pct"].to_numpy(dtype=float)
        losses = method_points["loss_pct"].to_numpy(dtype=float)
        valid = np.isfinite(time_saved) & np.isfinite(losses)
        time_saved = time_saved[valid]
        losses = losses[valid]
        if losses.size == 0:
            continue
        for time_threshold in TIME_SAVED_THRESHOLDS_PCT:
            for tolerance in LOSS_TOLERANCES_PCT:
                success = (time_saved >= time_threshold) & (losses <= tolerance)
                rows.append(
                    {
                        "task": str(method_points["task"].iloc[0]),
                        "method_key": str(method_key),
                        "time_saved_threshold_pct": float(time_threshold),
                        "loss_tolerance_pct": float(tolerance),
                        "estimate": float(np.mean(success)),
                        "n_datasets": int(losses.size),
                        "experimental_unit": "dataset",
                    }
                )
    return pd.DataFrame(rows)


def _load_joint_intervals(
    inference_dir: Path | None,
    *,
    interval_scope: str,
) -> tuple[pd.DataFrame | None, Path | None]:
    """Load optional dataset-level bootstrap intervals for joint reliability."""
    if inference_dir is None:
        return None, None
    path = inference_dir / "joint_reliability_intervals.csv"
    if not path.is_file():
        warnings.warn(
            f"{path} is unavailable; generating a point-estimate preview only.",
            stacklevel=2,
        )
        return None, path

    frame = pd.read_csv(path)
    low_col = f"{interval_scope}_ci_low"
    high_col = f"{interval_scope}_ci_high"
    required = {
        "task",
        "method_key",
        "time_saved_threshold_pct",
        "loss_tolerance_pct",
        "estimate",
        "n_datasets",
        "experimental_unit",
        "loss_unit",
        "confidence",
        low_col,
        high_col,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    units = set(frame["experimental_unit"].dropna().astype(str))
    if units != {"dataset"}:
        raise ValueError(
            f"{path} must contain dataset-level intervals; found units {sorted(units)}"
        )

    frame = frame.copy()
    frame = frame[~frame["method_key"].astype(str).str.startswith("secretary_par|")]
    for task, expected_unit in (
        ("classification_gini", "F1_percentage_points"),
        ("classification_entropy", "F1_percentage_points"),
    ):
        task_units = set(
            frame.loc[frame["task"] == task, "loss_unit"].dropna().astype(str)
        )
        if task_units and task_units != {expected_unit}:
            raise ValueError(
                f"{path} labels {task} loss as {sorted(task_units)}, expected "
                f"{expected_unit}"
            )
    frame = frame.rename(
        columns={
            "estimate": "inference_estimate",
            "n_datasets": "inference_n_datasets",
            low_col: "ci_low",
            high_col: "ci_high",
        }
    )
    return frame, path


def _attach_joint_intervals(
    point_table: pd.DataFrame,
    interval_frame: pd.DataFrame | None,
) -> tuple[pd.DataFrame, int]:
    """Attach intervals and reject stale files with different point estimates."""
    out = point_table.copy()
    out["ci_low"] = np.nan
    out["ci_high"] = np.nan
    out["confidence"] = np.nan
    if interval_frame is None:
        return out, 0

    keys = [
        "task",
        "method_key",
        "time_saved_threshold_pct",
        "loss_tolerance_pct",
    ]
    intervals = interval_frame[
        keys
        + [
            "inference_estimate",
            "inference_n_datasets",
            "ci_low",
            "ci_high",
            "confidence",
        ]
    ].copy()
    for col in ("time_saved_threshold_pct", "loss_tolerance_pct"):
        out[col] = pd.to_numeric(out[col], errors="raise").round(10)
        intervals[col] = pd.to_numeric(intervals[col], errors="raise").round(10)
    duplicate = intervals.duplicated(keys, keep=False)
    if duplicate.any():
        sample = intervals.loc[duplicate, keys].head()
        raise ValueError(f"Duplicate joint reliability interval rows:\n{sample}")

    out = out.drop(columns=["ci_low", "ci_high", "confidence"]).merge(
        intervals,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    matched = out["inference_estimate"].notna()
    discrepancies = (
        out.loc[matched, "estimate"] - out.loc[matched, "inference_estimate"]
    ).abs()
    if (discrepancies > 1e-12).any():
        bad = out.loc[
            matched & (discrepancies > 1e-12),
            keys + ["estimate", "inference_estimate"],
        ].head()
        raise ValueError(
            "Inference intervals do not match dataset-median point estimates; "
            f"the inference bundle may be stale:\n{bad}"
        )
    count_mismatch = matched & (
        out["n_datasets"].astype(float) != out["inference_n_datasets"]
    )
    if count_mismatch.any():
        bad = out.loc[
            count_mismatch,
            keys + ["n_datasets", "inference_n_datasets"],
        ].head()
        raise ValueError(f"Inference dataset counts do not match figure data:\n{bad}")
    return out, int(matched.sum())


def _style_probability_axis(ax: plt.Axes) -> None:
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.grid(axis="y", color="#c8c8c8", linewidth=0.55, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.2, length=3)


def _plot_profiles(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    x_col: str,
    x_values: tuple[float, ...],
    method_order: list[str],
    method_colors: dict[str, str],
) -> None:
    """Plot one reliability profile per method, with optional interval bands."""
    if table.empty:
        ax.set_axis_off()
        return
    has_intervals = {"ci_low", "ci_high"}.issubset(table.columns)
    for method_key in method_order:
        rows = table[table["method_key"] == method_key].sort_values(x_col)
        if rows.empty:
            continue
        x = rows[x_col].to_numpy(dtype=float)
        y = rows["estimate"].to_numpy(dtype=float)
        color = method_colors.get(method_key, "#666666")
        emphasized = method_key in EMPHASIZED_METHOD_KEYS
        alpha = 0.96 if emphasized else 0.68
        linewidth = 1.45 if emphasized else 0.9
        zorder = 4 if emphasized else 2
        if has_intervals:
            low = rows["ci_low"].to_numpy(dtype=float)
            high = rows["ci_high"].to_numpy(dtype=float)
            finite = np.isfinite(low) & np.isfinite(high)
            if finite.any():
                ax.fill_between(
                    x[finite],
                    np.clip(low[finite], 0.0, 1.0),
                    np.clip(high[finite], 0.0, 1.0),
                    color=color,
                    alpha=0.075 if emphasized else 0.035,
                    linewidth=0,
                    zorder=zorder - 1,
                )
        ax.plot(
            x,
            y,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            marker="o",
            markersize=3.2 if emphasized else 2.4,
            markeredgewidth=0.35,
            markeredgecolor="white",
            linestyle="--" if method_key == "best|" else "-",
            zorder=zorder,
        )
    ax.set_xticks(x_values)
    _style_probability_axis(ax)


def _legend_axis(
    ax: plt.Axes,
    *,
    method_order: list[str],
    method_colors: dict[str, str],
    method_labels: list[str],
    note: str,
) -> None:
    plot_grouped_variant_legend(
        ax,
        method_order,
        method_colors,
        method_labels,
        fontsize=6.45,
        header_fontsize=7.1,
        legend_style="point",
        y_header=0.73,
        y_top=0.63,
        y_bot=0.01,
    )
    ax.text(
        0.5,
        0.96,
        note,
        ha="center",
        va="top",
        fontsize=6.7,
        color="#444444",
        transform=ax.transAxes,
    )


def _save_figure(
    fig: plt.Figure,
    output_stem: Path,
    *,
    png_dpi: int = 240,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(
            path,
            bbox_inches="tight",
            dpi=png_dpi if extension == "png" else None,
        )
        print(f"Saved {path}")


def _size_description(size_filter: str) -> str:
    if size_filter == "small":
        return rf"small datasets ($n\times p\leq {SMALL_SIZE:,}$)"
    if size_filter == "large":
        return rf"large datasets ($n\times p\geq {LARGE_SIZE:,}$)"
    return "all datasets"


def _interval_note(
    joint_tables: dict[str, pd.DataFrame],
    *,
    interval_scope: str,
) -> tuple[str, bool]:
    confidences = []
    for table in joint_tables.values():
        if "confidence" in table.columns:
            confidences.extend(table["confidence"].dropna().astype(float).tolist())
    if not confidences:
        return (
            "POINT-ESTIMATE PREVIEW ONLY: joint bootstrap intervals were not supplied.",
            False,
        )
    confidence = 100.0 * float(np.median(confidences))
    return (
        f"Joint profiles: {confidence:.0f}% {interval_scope} percentile-bootstrap bands; "
        "loss-only profiles: point estimates.",
        True,
    )


def _save_combined_reliability(
    *,
    loss_tables: dict[str, pd.DataFrame],
    joint_tables: dict[str, pd.DataFrame],
    method_order: list[str],
    method_colors: dict[str, str],
    method_labels: list[str],
    output_stem: Path,
    size_filter: str,
    interval_scope: str,
) -> None:
    """Save the main loss-only plus joint reliability profile."""
    note, has_intervals = _interval_note(
        joint_tables, interval_scope=interval_scope
    )
    fig = plt.figure(figsize=(7.45, 9.55), layout="constrained")
    grid = GridSpec(
        5,
        3,
        figure=fig,
        height_ratios=(0.92, 1.0, 1.0, 1.0, 1.05),
        hspace=0.08,
        wspace=0.08,
    )
    axes = [[None] * 3 for _ in range(4)]

    for column, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(grid[0, column])
        axes[0][column] = ax
        _plot_profiles(
            ax,
            loss_tables[task],
            x_col="loss_tolerance_pct",
            x_values=LOSS_TOLERANCES_PCT,
            method_order=method_order,
            method_colors=method_colors,
        )
        ax.set_title(
            TASK_CONFIG[task]["panel_title"],
            fontsize=8.4,
            fontweight="bold",
            pad=5,
        )
        ax.set_xlabel(TASK_CONFIG[task]["tolerance_label"], fontsize=7.4)
        if column == 0:
            ax.set_ylabel("Loss-only fraction\nof datasets", fontsize=7.6)
        else:
            ax.tick_params(labelleft=False)

    for row, tolerance in enumerate(LOSS_TOLERANCES_PCT, start=1):
        for column, task in enumerate(TASK_ORDER):
            ax = fig.add_subplot(grid[row, column])
            axes[row][column] = ax
            subset = joint_tables[task]
            subset = subset[subset["loss_tolerance_pct"] == tolerance]
            _plot_profiles(
                ax,
                subset,
                x_col="time_saved_threshold_pct",
                x_values=TIME_SAVED_THRESHOLDS_PCT,
                method_order=method_order,
                method_colors=method_colors,
            )
            if row == len(LOSS_TOLERANCES_PCT):
                ax.set_xlabel(r"Minimum training time saved, $s$ (%)", fontsize=7.4)
            else:
                ax.tick_params(labelbottom=False)
            if column == 0:
                unit = "%" if task == "regression" else " F1 pp"
                ax.set_ylabel(
                    "Joint fraction\n"
                    + rf"$\varepsilon\leq {tolerance:g}${unit}",
                    fontsize=7.6,
                )
            else:
                ax.tick_params(labelleft=False)

    axes[0][0].text(
        -0.28,
        1.19,
        "a",
        transform=axes[0][0].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    axes[1][0].text(
        -0.28,
        1.08,
        "b",
        transform=axes[1][0].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )

    legend_ax = fig.add_subplot(grid[4, :])
    _legend_axis(
        legend_ax,
        method_order=method_order,
        method_colors=method_colors,
        method_labels=method_labels,
        note=note,
    )
    if size_filter == "all" and not has_intervals:
        status = "POINT-ESTIMATE PREVIEW ONLY | dataset-level reliability"
        status_color = "#9a3412"
    else:
        status = f"Dataset-level reliability | {_size_description(size_filter)}"
        status_color = "#333333"
    fig.suptitle(status, fontsize=9.2, fontweight="bold", color=status_color)
    _save_figure(fig, output_stem)
    plt.close(fig)


def _save_joint_profiles(
    *,
    joint_tables: dict[str, pd.DataFrame],
    method_order: list[str],
    method_colors: dict[str, str],
    method_labels: list[str],
    output_stem: Path,
    size_filter: str,
    interval_scope: str,
    supplementary_copy: Path | None = None,
) -> None:
    """Save the compact grid of joint operating-point profiles."""
    note, has_intervals = _interval_note(
        joint_tables, interval_scope=interval_scope
    )
    fig = plt.figure(figsize=(7.45, 7.55), layout="constrained")
    grid = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=(1.0, 1.0, 1.0, 1.0),
        hspace=0.08,
        wspace=0.08,
    )
    axes = [[None] * 3 for _ in LOSS_TOLERANCES_PCT]
    for row, tolerance in enumerate(LOSS_TOLERANCES_PCT):
        for column, task in enumerate(TASK_ORDER):
            ax = fig.add_subplot(grid[row, column])
            axes[row][column] = ax
            subset = joint_tables[task]
            subset = subset[subset["loss_tolerance_pct"] == tolerance]
            _plot_profiles(
                ax,
                subset,
                x_col="time_saved_threshold_pct",
                x_values=TIME_SAVED_THRESHOLDS_PCT,
                method_order=method_order,
                method_colors=method_colors,
            )
            if row == 0:
                ax.set_title(
                    TASK_CONFIG[task]["short_title"],
                    fontsize=8.4,
                    fontweight="bold",
                    pad=5,
                )
            if row == len(LOSS_TOLERANCES_PCT) - 1:
                ax.set_xlabel(r"Minimum time saved, $s$ (%)", fontsize=7.4)
            else:
                ax.tick_params(labelbottom=False)
            if column == 0:
                unit = "%" if task == "regression" else " F1 pp"
                ax.set_ylabel(
                    "Fraction of datasets\n"
                    + rf"$\varepsilon\leq {tolerance:g}${unit}",
                    fontsize=7.5,
                )
            else:
                ax.tick_params(labelleft=False)

    legend_ax = fig.add_subplot(grid[3, :])
    _legend_axis(
        legend_ax,
        method_order=method_order,
        method_colors=method_colors,
        method_labels=method_labels,
        note=note,
    )
    if size_filter == "all" and not has_intervals:
        title = "POINT-ESTIMATE PREVIEW ONLY | joint dataset reliability"
        title_color = "#9a3412"
    else:
        title = f"Joint dataset reliability | {_size_description(size_filter)}"
        title_color = "#333333"
    fig.suptitle(title, fontsize=9.2, fontweight="bold", color=title_color)
    _save_figure(fig, output_stem)
    if supplementary_copy is not None:
        supplementary_copy.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(supplementary_copy, bbox_inches="tight", dpi=600)
        print(f"Saved {supplementary_copy}")
    plt.close(fig)


def _plot_loss_cdf_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    *,
    task: str,
    method_order: list[str],
    method_colors: dict[str, str],
) -> None:
    """Plot a continuous loss-only CDF over dataset median points."""
    if points.empty:
        ax.set_axis_off()
        return
    epsilon = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 2.5, 126),
                np.linspace(2.5, 10.0, 76),
                np.linspace(10.0, 50.0, 81),
            )
        )
    )
    for method_key in method_order:
        losses = points.loc[
            points["method_key"] == method_key, "loss_pct"
        ].dropna()
        if losses.empty:
            continue
        fraction = np.asarray([(losses <= value).mean() for value in epsilon])
        emphasized = method_key in EMPHASIZED_METHOD_KEYS
        ax.plot(
            epsilon,
            fraction,
            color=method_colors.get(method_key, "#666666"),
            linewidth=1.35 if emphasized else 0.85,
            alpha=0.95 if emphasized else 0.66,
            linestyle="--" if method_key == "best|" else "-",
            zorder=4 if emphasized else 2,
        )
    for tolerance in LOSS_TOLERANCES_PCT:
        ax.axvline(tolerance, color="#555555", linewidth=0.55, alpha=0.38)
    ax.set_xscale("symlog", linthresh=2.5, linscale=1.0)
    ax.set_xlim(0.0, 50.0)
    ax.set_xticks((0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0))
    ax.set_xticklabels(("0", "0.5", "1", "2.5", "5", "10", "25", "50"))
    ax.set_xlabel(TASK_CONFIG[task]["tolerance_label"], fontsize=7.4)
    _style_probability_axis(ax)


def _save_loss_only_profiles(
    *,
    points_by_task: dict[str, pd.DataFrame],
    method_order: list[str],
    method_colors: dict[str, str],
    method_labels: list[str],
    output_stem: Path,
    size_filter: str,
) -> None:
    fig = plt.figure(figsize=(7.45, 3.95), layout="constrained")
    grid = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=(1.0, 0.7),
        hspace=0.08,
        wspace=0.08,
    )
    for column, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(grid[0, column])
        _plot_loss_cdf_panel(
            ax,
            points_by_task[task],
            task=task,
            method_order=method_order,
            method_colors=method_colors,
        )
        ax.set_title(
            TASK_CONFIG[task]["short_title"],
            fontsize=8.4,
            fontweight="bold",
            pad=5,
        )
        if column == 0:
            ax.set_ylabel("Fraction of datasets", fontsize=7.6)
        else:
            ax.tick_params(labelleft=False)
    legend_ax = fig.add_subplot(grid[1, :])
    _legend_axis(
        legend_ax,
        method_order=method_order,
        method_colors=method_colors,
        method_labels=method_labels,
        note="Each curve is an empirical CDF over dataset-level paired-run medians.",
    )
    fig.suptitle(
        f"Loss-only dataset reliability | {_size_description(size_filter)}",
        fontsize=9.2,
        fontweight="bold",
    )
    _save_figure(fig, output_stem)
    plt.close(fig)


def _save_size_stratified_joint(
    *,
    joint_by_size: dict[str, dict[str, pd.DataFrame]],
    method_order: list[str],
    method_colors: dict[str, str],
    method_labels: list[str],
    output_path: Path,
) -> None:
    """Save small/large joint profiles; the unit remains a dataset."""
    fig = plt.figure(figsize=(7.45, 13.1), layout="constrained")
    grid = GridSpec(
        7,
        3,
        figure=fig,
        height_ratios=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95),
        hspace=0.08,
        wspace=0.08,
    )
    row = 0
    for size_filter in ("large", "small"):
        for tolerance in LOSS_TOLERANCES_PCT:
            for column, task in enumerate(TASK_ORDER):
                ax = fig.add_subplot(grid[row, column])
                subset = joint_by_size[size_filter][task]
                subset = subset[subset["loss_tolerance_pct"] == tolerance]
                _plot_profiles(
                    ax,
                    subset,
                    x_col="time_saved_threshold_pct",
                    x_values=TIME_SAVED_THRESHOLDS_PCT,
                    method_order=method_order,
                    method_colors=method_colors,
                )
                if row in (0, 3):
                    ax.set_title(
                        TASK_CONFIG[task]["short_title"],
                        fontsize=8.4,
                        fontweight="bold",
                        pad=5,
                    )
                if row in (2, 5):
                    ax.set_xlabel(r"Minimum time saved, $s$ (%)", fontsize=7.4)
                else:
                    ax.tick_params(labelbottom=False)
                if column == 0:
                    stratum = "Large" if size_filter == "large" else "Small"
                    unit = "%" if task == "regression" else " F1 pp"
                    ax.set_ylabel(
                        f"{stratum}: fraction\n"
                        + rf"$\varepsilon\leq {tolerance:g}${unit}",
                        fontsize=7.4,
                    )
                else:
                    ax.tick_params(labelleft=False)
            row += 1
    legend_ax = fig.add_subplot(grid[6, :])
    _legend_axis(
        legend_ax,
        method_order=method_order,
        method_colors=method_colors,
        method_labels=method_labels,
        note="Point estimates; each success fraction weights datasets equally.",
    )
    fig.suptitle(
        "Joint dataset reliability by benchmark size",
        fontsize=9.2,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")


def _save_size_stratified_loss(
    *,
    points_by_size: dict[str, dict[str, pd.DataFrame]],
    method_order: list[str],
    method_colors: dict[str, str],
    method_labels: list[str],
    output_path: Path,
) -> None:
    """Save small/large loss-only CDFs over dataset median points."""
    fig = plt.figure(figsize=(7.45, 6.25), layout="constrained")
    grid = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=(1.0, 1.0, 0.82),
        hspace=0.08,
        wspace=0.08,
    )
    for row, size_filter in enumerate(("large", "small")):
        for column, task in enumerate(TASK_ORDER):
            ax = fig.add_subplot(grid[row, column])
            _plot_loss_cdf_panel(
                ax,
                points_by_size[size_filter][task],
                task=task,
                method_order=method_order,
                method_colors=method_colors,
            )
            if row == 0:
                ax.set_title(
                    TASK_CONFIG[task]["short_title"],
                    fontsize=8.4,
                    fontweight="bold",
                    pad=5,
                )
            if column == 0:
                stratum = "Large" if size_filter == "large" else "Small"
                ax.set_ylabel(f"{stratum}: fraction\nof datasets", fontsize=7.5)
            else:
                ax.tick_params(labelleft=False)
    legend_ax = fig.add_subplot(grid[2, :])
    _legend_axis(
        legend_ax,
        method_order=method_order,
        method_colors=method_colors,
        method_labels=method_labels,
        note="Each curve is an empirical CDF over dataset-level paired-run medians.",
    )
    fig.suptitle(
        "Loss-only dataset reliability by benchmark size",
        fontsize=9.2,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indir",
        type=Path,
        default=BENCHMARK_DIR,
        help="Directory containing benchmark run CSVs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Figure output directory. Defaults to examples/early_stop_trees/figures; "
            "pass a /tmp path for previews."
        ),
    )
    parser.add_argument(
        "--supp-outdir",
        type=Path,
        default=None,
        help=(
            "Supplementary PNG directory. With a custom --outdir, defaults to "
            "<outdir>/SUPP_FIGURES."
        ),
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=None,
        help="Optional directory containing joint_reliability_intervals.csv.",
    )
    parser.add_argument(
        "--interval-scope",
        choices=("within", "between", "hierarchical"),
        default="hierarchical",
        help="Bootstrap interval scope to display (default: hierarchical).",
    )
    parser.add_argument(
        "--no-supp",
        action="store_true",
        help="Do not write the merged small/large supplementary PNGs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.outdir if args.outdir is not None else OUT_DIR
    if args.supp_outdir is not None:
        supp_dir = args.supp_outdir
    elif args.outdir is not None:
        supp_dir = out_dir / "SUPP_FIGURES"
    else:
        supp_dir = SUPP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_by_task = {
        "regression": get_regression_run_level(args.indir),
        "classification_gini": get_classification_run_level(args.indir, "gini"),
        "classification_entropy": get_classification_run_level(
            args.indir, "entropy"
        ),
    }
    display_runs = {}
    for task, frame in runs_by_task.items():
        if frame is None or frame.empty:
            raise FileNotFoundError(f"No benchmark run files found for {task}")
        frame = _add_method_key(frame)
        runs_by_task[task] = frame
        display_runs[task] = frame[frame["splitter"] != "secretary_par"].copy()

    method_order, method_colors, method_labels = (
        get_variant_method_order_and_colors(
            *(display_runs[task] for task in TASK_ORDER),
            include_secretary_par=False,
        )
    )
    if any(method.startswith("secretary_par|") for method in method_order):
        raise RuntimeError("S_par must not appear in the reliability figure")
    label_by_method = dict(zip(method_order, method_labels))
    missing_main = sorted(set(MAIN_REPRESENTATIVES) - set(method_order))
    if missing_main:
        raise RuntimeError(f"Missing main-figure representatives: {missing_main}")
    main_method_order = list(MAIN_REPRESENTATIVES)
    main_method_colors = {key: method_colors[key] for key in main_method_order}
    main_method_labels = [label_by_method[key] for key in main_method_order]

    all_points = {}
    for task in TASK_ORDER:
        task_points = _dataset_median_points(runs_by_task[task], task=task)
        all_points[task] = task_points[
            task_points["splitter"] != "secretary_par"
        ].copy()
    for task, points in all_points.items():
        n_datasets = int(points["dataset"].nunique())
        run_min = int(points["n_paired_runs"].min())
        run_max = int(points["n_paired_runs"].max())
        print(
            f"[{task}] {n_datasets} dataset units; paired-run medians use "
            f"{run_min}-{run_max} runs"
        )

    points_by_size = {
        size_filter: {
            task: _filter_points_by_size(all_points[task], size_filter)
            for task in TASK_ORDER
        }
        for size_filter in ("all", "small", "large")
    }
    loss_by_size = {
        size_filter: {
            task: _loss_reliability_table(points_by_size[size_filter][task])
            for task in TASK_ORDER
        }
        for size_filter in ("all", "small", "large")
    }
    joint_by_size = {
        size_filter: {
            task: _joint_reliability_table(points_by_size[size_filter][task])
            for task in TASK_ORDER
        }
        for size_filter in ("all", "small", "large")
    }

    interval_frame, interval_path = _load_joint_intervals(
        args.inference_dir,
        interval_scope=args.interval_scope,
    )
    matched_total = 0
    for task in TASK_ORDER:
        joint_by_size["all"][task], matched = _attach_joint_intervals(
            joint_by_size["all"][task], interval_frame
        )
        matched_total += matched
    if interval_frame is None:
        source = interval_path if interval_path is not None else "no inference directory"
        print(f"Joint intervals unavailable ({source}); point-estimate preview mode.")
    else:
        expected = sum(len(joint_by_size["all"][task]) for task in TASK_ORDER)
        if matched_total != expected:
            warnings.warn(
                f"Matched {matched_total}/{expected} joint interval rows; "
                "unmatched profiles will show point estimates only.",
                stacklevel=2,
            )
        print(
            f"Loaded {matched_total} {args.interval_scope} joint intervals from "
            f"{interval_path}"
        )

    for size_filter in ("all", "small", "large"):
        suffix = "" if size_filter == "all" else f"_{size_filter}"
        _save_combined_reliability(
            loss_tables=loss_by_size[size_filter],
            joint_tables=joint_by_size[size_filter],
            method_order=main_method_order,
            method_colors=main_method_colors,
            method_labels=main_method_labels,
            output_stem=out_dir / f"figure4_success_combined{suffix}",
            size_filter=size_filter,
            interval_scope=args.interval_scope,
        )
        _save_joint_profiles(
            joint_tables=joint_by_size[size_filter],
            method_order=method_order,
            method_colors=method_colors,
            method_labels=method_labels,
            output_stem=out_dir / f"figure4_success_joint{suffix}",
            size_filter=size_filter,
            interval_scope=args.interval_scope,
            supplementary_copy=(
                supp_dir / "supp_figure_12_success_joint_all.png"
                if size_filter == "all"
                else None
            ),
        )
        _save_loss_only_profiles(
            points_by_task=points_by_size[size_filter],
            method_order=method_order,
            method_colors=method_colors,
            method_labels=method_labels,
            output_stem=out_dir / f"figure4_success_loss_only{suffix}",
            size_filter=size_filter,
        )

    if not args.no_supp:
        _save_size_stratified_joint(
            joint_by_size=joint_by_size,
            method_order=method_order,
            method_colors=method_colors,
            method_labels=method_labels,
            output_path=supp_dir
            / "supp_figure_06_success_joint_large_small.png",
        )
        _save_size_stratified_loss(
            points_by_size=points_by_size,
            method_order=method_order,
            method_colors=method_colors,
            method_labels=method_labels,
            output_path=supp_dir
            / "supp_figure_07_success_loss_only_large_small.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
