#!/usr/bin/env python
"""
Figure 2: Ridgeline plots of per-dataset distributions.

One ridgeline per dataset (ordered by increasing n_samples * n_features).
On each line: distribution across the 100 runs for each method (all methods on
the same ridge); "best" is excluded (it defines the 0 reference).
A dashed vertical line at 0 crosses all ridges.

Three separate figures (each 1×2, A4 height):
  - figure2_ridgelines_regression: time saved | loss (RMSE bounded)
  - figure2_ridgelines_classification_gini: time saved | loss (F1)
  - figure2_ridgelines_classification_entropy: time saved | loss (F1)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

from benchmark_results_utils import (
    load_all,
    get_regression_run_level,
    get_classification_run_level,
    SECRETARY_SPLITTERS_NO_PAR,
    get_variant_method_order_and_colors,
)

# A4 portrait: 210 × 297 mm → inches
A4_INCHES = (8.27, 11.69)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "figures"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"


def _time_saved_pct(speedup: np.ndarray) -> np.ndarray:
    """% time saved vs best from run-level speedup."""
    s = np.asarray(speedup, dtype=float)
    return 100.0 * (1.0 - 1.0 / np.maximum(s, 1e-6))


def _dataset_order_by_size(df: pd.DataFrame) -> list:
    """Unique datasets sorted by n_samples * n_features ascending (smallest first)."""
    meta = _dataset_meta_by_size(df)
    return meta["dataset"].tolist()


def _dataset_meta_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame of dataset, n_samples, n_features, size sorted by size ascending."""
    meta = df[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    meta = meta.assign(
        size=(meta["n_samples"].fillna(1).astype(float) * meta["n_features"].fillna(1).astype(float))
    )
    meta = meta.sort_values("size", ascending=True)
    return meta


# Default x-axis (time saved); loss panels use (-40, 100)
RIDGE_X_MIN, RIDGE_X_MAX = -100.0, 100.0
RIDGE_LOSS_X_MIN, RIDGE_LOSS_X_MAX = -40.0, 100.0


def _ridgeline_panel(
    ax,
    run_df: pd.DataFrame,
    value_col: str,
    xlabel: str,
    method_order: list,
    method_colors: dict,
    ridge_height: float = 1.0,
    ridge_step: float = 0.82,
    show_ridge_labels: bool = True,
    x_min: float = None,
    x_max: float = None,
    n_grid: int = 200,
):
    """
    One panel: one ridgeline per dataset (ordered by size); on each ridge, KDE over 100 runs
    for each method (all methods on the same line). method_order/list colors use method_key (splitter|variant).
    Vertical dashed line at 0.
    """
    datasets = _dataset_order_by_size(run_df)
    if not datasets:
        ax.set_axis_off()
        return

    all_vals = run_df[value_col].dropna().values
    if all_vals.size == 0:
        ax.set_axis_off()
        return

    if x_min is None:
        x_min = RIDGE_X_MIN
    if x_max is None:
        x_max = RIDGE_X_MAX
    xs = np.linspace(x_min, x_max, n_grid)
    id_col = "method_key" if "method_key" in run_df.columns else "splitter"

    N = len(datasets)
    y_offsets = (np.arange(N)[::-1].astype(float)) * ridge_step
    meta = _dataset_meta_by_size(run_df)

    for idx, ds in enumerate(datasets):
        y0 = y_offsets[idx]
        sub = run_df[run_df["dataset"] == ds]
        if sub.empty:
            continue

        for method in method_order:
            vals = sub.loc[sub[id_col] == method, value_col].dropna().values
            if vals.size < 2:
                continue
            try:
                kde = gaussian_kde(vals)
                ys = kde(xs)
            except np.linalg.LinAlgError:
                continue
            ys = np.maximum(ys, 0)
            if ys.max() > 0:
                ys = ys / ys.max() * ridge_height
            y_top = y0 + ys
            color = method_colors.get(method, "#555555")
            ax.fill_between(
                xs,
                y0,
                y_top,
                color=to_rgba(color, alpha=0.5),
                linewidth=0.6,
                edgecolor=to_rgba(color, alpha=0.85),
            )

        if show_ridge_labels:
            row = meta[meta["dataset"] == ds].iloc[0]
            n, p = int(row["n_samples"]), int(row["n_features"])
            d = int(row["size"])
            label = f"n={n} p={p} d={d}"
            ax.text(
                x_min - 0.02 * (x_max - x_min),
                y0 + 0.5 * ridge_height,
                label,
                ha="right",
                va="center",
                fontsize=6,
                color="#333",
            )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.9, zorder=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, (N - 1) * ridge_step + ridge_height + 0.2)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)


def main():
    ap = argparse.ArgumentParser(description="Figure 2: ridgeline plots of per-dataset distributions.")
    ap.add_argument(
        "--no-supp",
        action="store_true",
        help="Do not write supplementary PNG copies to SUPP_FIGURES/ (supp_figure_03–05).",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(BENCHMARK_DIR, exclude_secretary_par=True)

    def _prepare_run_df(raw, loss_col: str):
        if raw is None:
            return None
        df = raw[raw["splitter"].isin(SECRETARY_SPLITTERS_NO_PAR)].copy()
        if "variant" not in df.columns:
            df["variant"] = ""
        df["variant"] = df["variant"].fillna("").astype(str)
        df["method_key"] = df["splitter"].astype(str) + "|" + df["variant"]
        df["ts_pct"] = _time_saved_pct(df["speedup"].values)
        df["loss_pct"] = 100.0 * df[loss_col].values
        return df

    reg_raw = data.get("regression_run")
    if reg_raw is None:
        reg_raw = get_regression_run_level(BENCHMARK_DIR)
        if reg_raw is not None:
            reg_raw = reg_raw[reg_raw["splitter"].isin(SECRETARY_SPLITTERS_NO_PAR)].copy()
    gini_raw = data.get("classification_gini_run")
    if gini_raw is None:
        gini_raw = get_classification_run_level(BENCHMARK_DIR, "gini")
    entropy_raw = data.get("classification_entropy_run")
    if entropy_raw is None:
        entropy_raw = get_classification_run_level(BENCHMARK_DIR, "entropy")

    if reg_raw is None:
        raise FileNotFoundError("Missing regression run-level data.")
    reg = _prepare_run_df(reg_raw, "loss_rmse_bounded")
    method_order, method_colors, _ = get_variant_method_order_and_colors(reg_raw, gini_raw, entropy_raw)

    figures_config = [
        ("regression", reg, "Regression", "% loss vs best (RMSE bounded)"),
    ]
    if gini_raw is not None:
        figures_config.append(("classification_gini", _prepare_run_df(gini_raw, "loss_f1"), "Classification (Gini)", "% loss vs best (F1)"))
    if entropy_raw is not None:
        figures_config.append(("classification_entropy", _prepare_run_df(entropy_raw, "loss_f1"), "Classification (Entropy)", "% loss vs best (F1)"))

    for tag, run_df, _, loss_xlabel in figures_config:
        if run_df is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=A4_INCHES, sharey=False)
        _ridgeline_panel(
            axes[0],
            run_df,
            "ts_pct",
            xlabel="% time saved vs best",
            method_order=method_order,
            method_colors=method_colors,
            show_ridge_labels=True,
        )
        _ridgeline_panel(
            axes[1],
            run_df,
            "loss_pct",
            xlabel=loss_xlabel,
            method_order=method_order,
            method_colors=method_colors,
            show_ridge_labels=False,
            x_min=RIDGE_LOSS_X_MIN,
            x_max=RIDGE_LOSS_X_MAX,
        )
        plt.tight_layout()
        for ext in ("pdf", "png"):
            out = OUT_DIR / f"figure2_ridgelines_{tag}.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=(150 if ext == "png" else None))
        if not args.no_supp:
            SUPP_DIR.mkdir(parents=True, exist_ok=True)
            supp_names = {
                "regression": "supp_figure_03_ridgelines_regression.png",
                "classification_gini": "supp_figure_04_ridgelines_classification_gini.png",
                "classification_entropy": "supp_figure_05_ridgelines_classification_entropy.png",
            }
            if tag in supp_names:
                supp_path = SUPP_DIR / supp_names[tag]
                fig.savefig(supp_path, bbox_inches="tight", dpi=200)
                print(f"Saved {supp_path.name}")
        plt.close(fig)
        print(f"Saved figure2_ridgelines_{tag}.pdf and figure2_ridgelines_{tag}.png")


if __name__ == "__main__":
    main()
