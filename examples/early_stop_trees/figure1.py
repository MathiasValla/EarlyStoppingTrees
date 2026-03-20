#!/usr/bin/env python
"""
Figure 1: Pareto scatter plots (main figure).

For each dataset d and method m: one point at
  (log2(median speedup), median predictive loss)
with horizontal/vertical error bars for run-to-run variability (IQR).
Separate panels: regression, classification (F1 loss).
Optional: point size by log2(n_features); smoothed Pareto envelope per method.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# Allow running from project root or from this directory
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_SCRIPT_DIR))
from benchmark_results_utils import (
    load_all,
    SECRETARY_SPLITTERS,
    SECRETARY_SPLITTERS_NO_PAR,
    BASE_COLOR_SECRETARY_PAR,
    per_dataset_median_iqr,
    get_variant_method_order_and_colors,
)

BENCHMARK_DIR = _SCRIPT_DIR / "benchmark_results"
OUT_DIR = _SCRIPT_DIR / "figures"

# Method labels and colors (best = baseline, others distinct)
METHOD_COLORS = {
    "best": "#444444",
    "secretary": "#1f77b4",
    "secretary_par": "#ff7f0e",
    "secretary_all": "#2ca02c",
    "double_secretary": "#d62728",
    "block_rank": "#9467bd",
    "prophet_1sample": "#8c564b",
}
METHOD_ORDER = ["best", "secretary", "secretary_par", "secretary_all", "double_secretary", "block_rank", "prophet_1sample"]

# Figure 1 flagship: one “default” representative per family (1/e style), for emphasis vs other variants.
FIG1_DEFAULT_FAMILY_KEYS = frozenset(
    {
        "best|",
        "secretary|1overe",
        "double_secretary|1overe",
        "secretary_all|1overe",
        "block_rank|",
        "prophet_1sample|",
    }
)


def _pareto_visual_tier(method_key: str) -> str:
    """Return 'emph' (default representatives), 'par' (secretary_par, screened), or 'other' (other variants)."""
    mk = str(method_key)
    if mk.startswith("secretary_par|"):
        return "par"
    if mk in FIG1_DEFAULT_FAMILY_KEYS:
        return "emph"
    return "other"


def _add_dataset_info(summary: pd.DataFrame, run_df: pd.DataFrame) -> pd.DataFrame:
    """Add n_samples, n_features per dataset from run-level (constant per dataset)."""
    if run_df is None or "n_features" not in run_df.columns:
        return summary
    meta = run_df[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    return summary.merge(meta, on="dataset", how="left")


def _time_saved_pct_and_error(speedup_median, speedup_iqr):
    """
    Convert speedup median/IQR to percentage time saved and symmetric errors.

    TimeSaved = 100 * (1 - 1 / speedup).
    We approximate low/high speedup as median ± IQR/2, then propagate through the transform.
    """
    s_med = np.asarray(speedup_median, dtype=float)
    iqr = np.asarray(speedup_iqr, dtype=float)
    s_low = np.maximum(s_med - 0.5 * iqr, 1e-6)
    s_high = s_med + 0.5 * iqr
    ts_med = 100.0 * (1.0 - 1.0 / np.maximum(s_med, 1e-6))
    ts_low = 100.0 * (1.0 - 1.0 / np.maximum(s_high, 1e-6))  # smaller speedup -> less time saved
    ts_high = 100.0 * (1.0 - 1.0 / np.maximum(s_low, 1e-6))
    # Ensure non-negative error bar lengths
    xerr_left = np.maximum(ts_med - ts_low, 0.0)
    xerr_right = np.maximum(ts_high - ts_med, 0.0)
    return ts_med, xerr_left, xerr_right


def _pareto_envelope(x: np.ndarray, y: np.ndarray) -> tuple:
    """Pareto envelope: sort by x ascending, then cumulative min of y from right. Returns (x_sorted, y_envelope)."""
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]
    # For each position, envelope = min loss among all points with x >= x_s[i]
    y_env = np.minimum.accumulate(y_s[::-1])[::-1]
    return x_s, y_env


def _weighted_centroid_and_ellipse(x: np.ndarray, y: np.ndarray, w: np.ndarray, n_sigma: float = 2.0):
    """
    Weighted mean (centroid) and weighted covariance. Return (cx, cy, ellipse_width, ellipse_height, angle_deg)
    for drawing an ellipse at n_sigma standard deviations (approx 95% for n_sigma=2 in 2D Gaussian).
    """
    w = np.asarray(w, dtype=float)
    w = w / np.maximum(w.sum(), 1e-20)
    cx = np.sum(w * x)
    cy = np.sum(w * y)
    vx = np.sum(w * (x - cx) ** 2)
    vy = np.sum(w * (y - cy) ** 2)
    cov_xy = np.sum(w * (x - cx) * (y - cy))
    # Eigenvalues of [[vx, cov_xy], [cov_xy, vy]]
    trace = vx + vy
    det = vx * vy - cov_xy * cov_xy
    disc = max(trace ** 2 / 4 - det, 0.0)
    lam1 = trace / 2 + np.sqrt(disc)
    lam2 = trace / 2 - np.sqrt(disc)
    lam1 = max(lam1, 1e-20)
    lam2 = max(lam2, 1e-20)
    width = 2 * n_sigma * np.sqrt(lam1)
    height = 2 * n_sigma * np.sqrt(lam2)
    angle = np.degrees(np.arctan2(2 * cov_xy, vx - vy) / 2) if (vx != vy or cov_xy != 0) else 0.0
    return cx, cy, width, height, angle


def _pareto_envelope_smooth(x: np.ndarray, y: np.ndarray, n_grid: int = 400) -> tuple:
    """
    Smoothed Pareto envelope by evaluating the envelope on an x-grid and then
    applying a short moving-average smoothing while preserving monotonicity.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 2:
        return x, y
    x_grid = np.linspace(np.min(x), np.max(x), n_grid)
    y_grid = np.empty_like(x_grid)
    for i, x0 in enumerate(x_grid):
        mask = x >= x0
        y_grid[i] = np.min(y[mask]) if np.any(mask) else np.nan
    # Enforce Pareto (non-increasing in y) and smooth with moving average, then enforce again
    y_grid = np.minimum.accumulate(y_grid[::-1])[::-1]
    window = max(int(n_grid // 25), 5)  # ~4% of grid, at least 5
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_pad = np.pad(y_grid, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    y_smooth = np.minimum.accumulate(y_smooth[::-1])[::-1]
    return x_grid, y_smooth


def _plot_density_fade(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    x_lim,
    y_lim,
    n_grid: int = 80,
    zorder: int = 0,
    *,
    alpha_scale: float = 1.0,
):
    """
    Draw a 2D KDE of (x,y) as a smooth density with opacity fading where density is lower.
    Uses contourf with a custom colormap from transparent to method color.
    """
    if len(x) < 3:
        return
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return
    x_lo, x_hi = x_lim
    y_lo, y_hi = y_lim
    xs = np.linspace(x_lo, x_hi, n_grid)
    ys = np.linspace(y_lo, y_hi, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    positions = np.vstack([XX.ravel(), YY.ravel()])
    Z = kde(positions).reshape(XX.shape)
    Z = np.maximum(Z, 0)
    z_max = Z.max()
    if z_max <= 0:
        return
    Z = Z / z_max
    # Colormap: density 0 -> transparent, density 1 -> method color with alpha ~0.4 (scaled)
    a_hi = 0.42 * float(np.clip(alpha_scale, 0.0, 1.0))
    cmap = LinearSegmentedColormap.from_list(
        "dens",
        [to_rgba(color, 0.0), to_rgba(color, a_hi)],
        N=256,
    )
    levels = np.linspace(0.03, 1.0, 22)
    ax.contourf(XX, YY, Z, levels=levels, cmap=cmap, extend="min", zorder=zorder, antialiased=True)


def _plot_pareto_panel(
    ax,
    summary: pd.DataFrame,
    loss_col: str,
    loss_iqr_col: str,
    title: str,
    size_by_logp: bool = True,
    plot_envelope: bool = True,
):
    """One panel: x = % time saved, y = % loss; error bars from IQR; optional size by log2(p)."""
    # Drop rows with missing key columns
    df = summary.dropna(subset=["speedup_median", loss_col, "speedup_iqr", loss_iqr_col])
    df = df[df["speedup_median"] > 0].copy()
    # X axis: percentage time saved vs best
    ts_med, xerr_left, xerr_right = _time_saved_pct_and_error(
        df["speedup_median"].values, df["speedup_iqr"].values
    )
    df["x"] = ts_med
    # Y axis: percentage (or percentage-point) loss
    df["y"] = 100.0 * df[loss_col]

    # Point size by log2(n_features+1) if available
    if size_by_logp and "n_features" in df.columns:
        df["size"] = 20 + 25 * np.log2(df["n_features"].fillna(1) + 1)
    else:
        df["size"] = 40
    # Opacity by dataset size n*p (log-scaled)
    if "n_samples" in df.columns and "n_features" in df.columns:
        size_np = (df["n_samples"].fillna(1).astype(float) * df["n_features"].fillna(1).astype(float)).values
        log_np = np.log10(np.maximum(size_np, 1.0))
        lo, hi = np.nanpercentile(log_np, 5), np.nanpercentile(log_np, 95)
        denom = (hi - lo) if hi > lo else 1.0
        a = (log_np - lo) / denom
        a = np.clip(a, 0.0, 1.0)
        df["alpha_pt"] = 0.15 + 0.80 * a
    else:
        df["alpha_pt"] = 0.8

    for method in METHOD_ORDER:
        sub = df[df["splitter"] == method]
        if sub.empty:
            continue
        color = METHOD_COLORS.get(method, "gray")
        x = sub["x"].values
        y = sub["y"].values
        label = method.replace("_", " ")
        if method != "best":
            _, xerr_left_sub, xerr_right_sub = _time_saved_pct_and_error(
                sub["speedup_median"].values, sub["speedup_iqr"].values
            )
            yerr = 0.5 * 100.0 * sub[loss_iqr_col].values
            ax.errorbar(
                x, y,
                xerr=[xerr_left_sub, xerr_right_sub],
                yerr=yerr,
                fmt="none",
                color=color,
                capsize=1.5,
                capthick=0.8,
                elinewidth=0.8,
                alpha=0.55,
                zorder=1,
            )
        rgba = [to_rgba(color, alpha=a) for a in sub["alpha_pt"].values]
        ax.scatter(
            x, y,
            s=sub["size"].values,
            c=rgba,
            label=label,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )

    if plot_envelope:
        for method in SECRETARY_SPLITTERS:
            sub = df[df["splitter"] == method]
            if len(sub) < 5:
                continue
            x_s, y_env = _pareto_envelope_smooth(sub["x"].values, sub["y"].values)
            color = METHOD_COLORS.get(method, "gray")
            ax.plot(x_s, y_env, "-", color=color, linewidth=1.8, alpha=0.75, zorder=0)
        all_sec = df[df["splitter"].isin(SECRETARY_SPLITTERS)]
        if len(all_sec) >= 10:
            x_g, y_g = _pareto_envelope_smooth(all_sec["x"].values, all_sec["y"].values)
            ax.plot(x_g, y_g, "--", color="black", linewidth=2.0, alpha=0.8, zorder=0, label="global pareto")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Median % time saved vs best")
    ax.set_ylabel("Median % loss vs best")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Center on center of mass (equal weight per point)
    x_vals = df["x"].values
    y_vals = df["y"].values
    cx, cy = np.mean(x_vals), np.mean(y_vals)
    margin_x = np.max(np.abs(x_vals - cx)) + 5
    margin_y = np.max(np.abs(y_vals - cy)) + 5
    ax.set_xlim(left=cx - margin_x, right=cx + margin_x)
    ax.set_ylim(bottom=cy - margin_y, top=cy + margin_y)


def _plot_pareto_panel_centroids(
    ax,
    summary: pd.DataFrame,
    loss_col: str,
    loss_iqr_col: str,
    title: str,
    use_size_weights: bool,
    plot_envelope: bool = True,
    n_sigma: float = 2.0,
    method_order=None,
    method_colors=None,
    *,
    flagship_style: bool = False,
):
    """
    One panel: centroid per method + 2D density (KDE) with opacity fading; individual points on top.
    use_size_weights: True = weight centroid by n_samples*n_features, False = equal weight per dataset.
    Axes: x in [-50, 100]; y bottom = global Pareto min * 1.1, top from data.
    """
    df = summary.dropna(subset=["speedup_median", loss_col, "speedup_iqr", loss_iqr_col])
    df = df[df["speedup_median"] > 0].copy()
    ts_med, _, _ = _time_saved_pct_and_error(df["speedup_median"].values, df["speedup_iqr"].values)
    df["x"] = ts_med
    df["y"] = 100.0 * df[loss_col]

    if use_size_weights and "n_samples" in df.columns and "n_features" in df.columns:
        df["w"] = (df["n_samples"].fillna(1).astype(float) * df["n_features"].fillna(1).astype(float)).values
    else:
        df["w"] = 1.0

    x_all = df["x"].values
    y_all = df["y"].values
    cy_all = np.mean(y_all)
    # Envelope: all non-best (use SECRETARY_SPLITTERS_NO_PAR when summary has variant/method_key to exclude par)
    sec_col = "method_key" if "method_key" in df.columns else "splitter"
    all_sec = df[df["splitter"] != "best"] if "method_key" not in df.columns else df[df["splitter"].isin(SECRETARY_SPLITTERS_NO_PAR)]
    global_env = None
    y_lowest = None
    if len(all_sec) >= 2:
        x_g, y_g = _pareto_envelope_smooth(all_sec["x"].values, all_sec["y"].values)
        if y_g.size > 0 and np.any(np.isfinite(x_g)) and np.any(np.isfinite(y_g)):
            global_env = (x_g, y_g)
            y_lowest = np.nanmin(y_g)
    # Shared x-scale; y: bottom = lowest point of global Pareto * 1.1
    left, right_border = -50, 100
    if y_lowest is not None:
        bottom = float(y_lowest) * 1.1
        margin_y = np.max(np.abs(y_all - cy_all)) + 5
        top = max(2 * cy_all - bottom, bottom + max(margin_y, 1.0))
    else:
        bottom, top = -10, 80
    x_lim = (left, right_border)
    y_lim = (bottom, top)

    method_order = METHOD_ORDER if method_order is None else list(method_order)
    method_colors = METHOD_COLORS if method_colors is None else dict(method_colors)
    id_col = "method_key" if "method_key" in df.columns else "splitter"

    # Draw 2D density (KDE) per method, then points, then centroids
    for method in method_order:
        sub = df[df[id_col] == method]
        if sub.empty:
            continue
        color = method_colors.get(method, "gray")
        x = sub["x"].values
        y = sub["y"].values
        w = sub["w"].values
        tier = _pareto_visual_tier(method) if flagship_style else "emph"
        if flagship_style:
            if tier == "par":
                dens_scale, pt_alpha, cent_alpha, cent_lw = 0.35, 0.12, 0.38, 0.5
            elif tier == "emph":
                dens_scale, pt_alpha, cent_alpha, cent_lw = 1.0, 0.25, 1.0, 0.8
            else:
                dens_scale, pt_alpha, cent_alpha, cent_lw = 0.55, 0.18, 0.78, 0.65
        else:
            dens_scale, pt_alpha, cent_alpha, cent_lw = 1.0, 0.25, 1.0, 0.8

        _plot_density_fade(ax, x, y, color, x_lim, y_lim, n_grid=80, zorder=0, alpha_scale=dens_scale)
        ax.scatter(
            x, y,
            s=25,
            c=[to_rgba(color, alpha=pt_alpha)] * len(x),
            edgecolors="none",
            zorder=1,
        )
        w_safe = np.asarray(w, dtype=float)
        if np.all(w_safe <= 0):
            w_safe = np.ones_like(w_safe)
        cx = np.average(x, weights=w_safe)
        cy = np.average(y, weights=w_safe)
        cent_s = 68 if (flagship_style and tier == "par") else 80
        ax.scatter(
            cx,
            cy,
            s=cent_s,
            color=to_rgba(color, cent_alpha),
            edgecolors="white",
            linewidths=cent_lw,
            zorder=2,
        )

    if plot_envelope and global_env is not None:
        x_g, y_g = global_env
        ax.plot(x_g, y_g, "--", color="black", linewidth=2.0, alpha=0.8, zorder=3)

    ax.set_xlabel("Median % time saved vs best")
    ax.set_ylabel("Median % loss vs best")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=left, right=right_border)
    ax.set_ylim(bottom=bottom, top=top)


def _plot_legend_panel(ax):
    """Bottom-left panel: common legend for all methods."""
    return _plot_legend_panel_custom(ax, METHOD_ORDER, METHOD_COLORS)


def _plot_legend_panel_custom(ax, method_order, method_colors, method_labels=None):
    """Legend panel for an arbitrary set of methods. method_labels: optional list parallel to method_order."""
    ax.set_axis_off()
    labels = method_labels if method_labels is not None else [str(m).replace("_", " ") for m in method_order]
    handles = [
        Patch(facecolor=method_colors.get(m, "gray"), edgecolor="white", label=labels[i])
        for i, m in enumerate(method_order)
    ]
    ax.legend(handles=handles, loc="center", fontsize=9, frameon=True)


def _plot_legend_flagship_compact(ax, method_colors: dict):
    """
    Main-text legend: emphasize default 1/e family representatives; show S_par as a faded screened-out method.

    Does not list every variant — only the methods referenced in the paper's main comparison.
    """
    ax.set_axis_off()
    entries = [
        ("best|", "Exhaustive (best)"),
        ("secretary|1overe", "S (n/e)"),
        ("double_secretary|1overe", "S² (n/e)"),
        ("secretary_all|1overe", "S_all (n/e)"),
        ("block_rank|", "block-rank"),
        ("prophet_1sample|", "1-sample Prophet"),
    ]
    handles = []
    for key, label in entries:
        fc = method_colors.get(key, "#888888")
        handles.append(
            Patch(
                facecolor=fc,
                edgecolor="black",
                linewidth=1.6,
                label=label,
            )
        )
    # Parametric secretary: screened out — faded swatch (family color, not a single variant)
    handles.append(
        Patch(
            facecolor=to_rgba(BASE_COLOR_SECRETARY_PAR, 0.32),
            edgecolor=to_rgba(BASE_COLOR_SECRETARY_PAR, 0.55),
            linewidth=0.9,
            label="S_par (screened)",
        )
    )
    leg = ax.legend(handles=handles, loc="center", fontsize=9, frameon=True, handlelength=1.8)
    for t in leg.get_texts():
        if "S_par" in t.get_text():
            t.set_fontweight("normal")
            t.set_color("0.35")
        else:
            t.set_fontweight("bold")


def _variant_summaries(data: dict, target_splitter: str):
    """
    Build per-dataset summaries where each variant of target_splitter is treated as its own "method",
    plus the baseline "best".
    Returns (reg_summary, clf_gini_summary, clf_entropy_summary) with columns compatible with plotting.
    """

    def _prep(run_df: pd.DataFrame, loss_cols: list):
        if run_df is None or run_df.empty:
            return None
        keep = run_df[run_df["splitter"].isin(("best", target_splitter))].copy()
        if keep.empty:
            return None

        if "variant" not in keep.columns:
            keep["variant"] = ""
        keep["variant"] = keep["variant"].fillna("")
        keep.loc[keep["splitter"] == "best", "variant"] = "best"
        keep.loc[(keep["splitter"] == target_splitter) & (keep["variant"] == ""), "variant"] = "default"
        keep["method"] = keep["variant"].astype(str)

        value_cols = ["speedup"] + list(loss_cols)
        summary = per_dataset_median_iqr(keep, value_cols=value_cols, group_cols=["dataset", "method"])
        summary = summary.rename(columns={"method": "splitter"})

        # Bring over n_samples/n_features per dataset for size filtering
        meta = keep[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
        summary = summary.merge(meta, on="dataset", how="left")
        return summary

    reg = _prep(data.get("regression_run"), ["loss_rmse_bounded"])
    gini = _prep(data.get("classification_gini_run"), ["loss_f1"])
    ent = _prep(data.get("classification_entropy_run"), ["loss_f1"])
    return reg, gini, ent


def main():
    parser = argparse.ArgumentParser(description="Figure 1: Pareto scatter plots from benchmark results.")
    parser.add_argument(
        "--indir",
        type=str,
        default=None,
        help="Input directory with benchmark_results (default: examples/early_stop_trees/benchmark_results)",
    )
    parser.add_argument(
        "--variants-of",
        type=str,
        default=None,
        choices=["secretary", "double_secretary", "secretary_all", "secretary_par"],
        help="If set, plot best + all variants of this splitter (one color per variant).",
    )
    args = parser.parse_args()

    indir = Path(args.indir) if args.indir is not None else BENCHMARK_DIR

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flagship_style = False
    if args.variants_of is None:
        # Include secretary_par in data; flagship figure fades S_par and emphasizes default 1/e representatives.
        data = load_all(indir, exclude_secretary_par=False, by_variant=True)
        reg_summary = data["regression_summary"]
        clf_gini_summary = data["classification_gini_summary"]
        clf_entropy_summary = data["classification_entropy_summary"]
        if reg_summary is None:
            raise FileNotFoundError("No regression summary (run-level CSVs missing?)")
        if clf_gini_summary is None:
            raise FileNotFoundError("No classification gini summary (run-level CSVs missing?)")
        if clf_entropy_summary is None:
            raise FileNotFoundError("No classification entropy summary (run-level CSVs missing?)")

        reg_summary = _add_dataset_info(reg_summary, data["regression_run"])
        clf_gini_summary = _add_dataset_info(clf_gini_summary, data["classification_gini_run"])
        clf_entropy_summary = _add_dataset_info(clf_entropy_summary, data["classification_entropy_run"])

        for summ in (reg_summary, clf_gini_summary, clf_entropy_summary):
            v = summ.get("variant", pd.Series([""] * len(summ)))
            summ["method_key"] = summ["splitter"].astype(str) + "|" + v.fillna("").astype(str)

        method_order, method_colors, method_labels = get_variant_method_order_and_colors(
            reg_summary, clf_gini_summary, clf_entropy_summary, include_secretary_par=True
        )
        out_prefix = "figure1_pareto"
        legend_fn = lambda ax: _plot_legend_flagship_compact(ax, method_colors)
        flagship_style = True
    else:
        data = load_all(indir)
        reg_summary, clf_gini_summary, clf_entropy_summary = _variant_summaries(data, args.variants_of)
        if reg_summary is None or clf_gini_summary is None or clf_entropy_summary is None:
            raise FileNotFoundError("Missing run-level CSVs for variant plotting.")

        out_prefix = f"figure1_variants_{args.variants_of}"
        variants = sorted(
            set(reg_summary["splitter"]).union(set(clf_gini_summary["splitter"])).union(set(clf_entropy_summary["splitter"]))
        )
        method_order = ["best"] + [v for v in variants if v != "best"]
        cmap = plt.get_cmap("tab20")
        method_colors = {"best": "#444444"}
        for i, v in enumerate([m for m in method_order if m != "best"]):
            method_colors[v] = cmap(i % cmap.N)
        legend_fn = lambda ax: _plot_legend_panel_custom(ax, method_order, method_colors)

    def _filter_by_size(df: pd.DataFrame, min_size=None, max_size=None) -> pd.DataFrame:
        """Filter by dataset size = n_samples * n_features."""
        if df is None or "n_samples" not in df.columns or "n_features" not in df.columns:
            return df
        out = df.copy()
        out["_size"] = out["n_samples"].fillna(1).astype(float) * out["n_features"].fillna(1).astype(float)
        if min_size is not None:
            out = out[out["_size"] >= min_size]
        if max_size is not None:
            out = out[out["_size"] < max_size]
        out = out.drop(columns=["_size"], errors="ignore")
        return out

    configs = [
        ("all", reg_summary, clf_gini_summary, clf_entropy_summary),
        ("small", _filter_by_size(reg_summary, min_size=None, max_size=2501), _filter_by_size(clf_gini_summary, min_size=None, max_size=2501), _filter_by_size(clf_entropy_summary, min_size=None, max_size=2501)),
        ("large", _filter_by_size(reg_summary, min_size=25000, max_size=None), _filter_by_size(clf_gini_summary, min_size=25000, max_size=None), _filter_by_size(clf_entropy_summary, min_size=25000, max_size=None)),
    ]

    for tag, reg_sub, clf_gini_sub, clf_entropy_sub in configs:
        if reg_sub is None or reg_sub.empty or clf_gini_sub is None or clf_gini_sub.empty or clf_entropy_sub is None or clf_entropy_sub.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False)
        # Top left: regression
        _plot_pareto_panel_centroids(
            axes[0, 0],
            reg_sub,
            loss_col="loss_rmse_bounded_median",
            loss_iqr_col="loss_rmse_bounded_iqr",
            title="Regression (RMSE loss, bounded %)",
            use_size_weights=False,
            plot_envelope=True,
            method_order=method_order,
            method_colors=method_colors,
            flagship_style=flagship_style,
        )
        # Top right: classification (Gini)
        _plot_pareto_panel_centroids(
            axes[0, 1],
            clf_gini_sub,
            loss_col="loss_f1_median",
            loss_iqr_col="loss_f1_iqr",
            title="Classification (F1 loss, Gini)",
            use_size_weights=False,
            plot_envelope=True,
            method_order=method_order,
            method_colors=method_colors,
            flagship_style=flagship_style,
        )
        # Bottom left: legend
        legend_fn(axes[1, 0])
        # Bottom right: classification (entropy)
        _plot_pareto_panel_centroids(
            axes[1, 1],
            clf_entropy_sub,
            loss_col="loss_f1_median",
            loss_iqr_col="loss_f1_iqr",
            title="Classification (F1 loss, Entropy)",
            use_size_weights=False,
            plot_envelope=True,
            method_order=method_order,
            method_colors=method_colors,
            flagship_style=flagship_style,
        )
        plt.tight_layout()
        for ext in ("pdf", "png"):
            out = OUT_DIR / f"{out_prefix}_{tag}.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=(150 if ext == "png" else None))
        plt.close()
        print(f"Saved {out_prefix}_{tag}.pdf and {out_prefix}_{tag}.png")


if __name__ == "__main__":
    main()
