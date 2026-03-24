#!/usr/bin/env python
"""
Figure 4: Success-probability curves under a tolerance criterion.

Figure 4a: For each method, fraction of runs satisfying loss vs best ≤ ε, as ε varies.
  Three panels: Regression, Classification (Gini), Classification (Entropy).
  Generated for: all datasets, small (n×p ≤ SMALL_SIZE), large (n×p ≥ LARGE_SIZE).

Figure 4b: P(speedup ≥ s and loss ≤ ε) for grid (s, ε), one figure with grouped bars
  (regression | gini | entropy) per method per panel.
  s = 0, 10, 25, 50 (%); ε = 5, 10, 20 (%). 4×3 panels.
  Generated for: all, small, large.

Main text export (``MAIN_FIGURES``): **loss CDF row only** + grouped legend (marker swatches) → ``figure4_success_combined{suffix}`` — styled like ``figure4_success_loss_only`` (same titles) but one bottom legend instead of per-panel line legends.

Joint bar grid (4×3) + grouped legend → ``figure4_success_joint{suffix}``; **all** datasets also saved to ``SUPP_FIGURES/supp_figure_12_success_joint_all.png``.

Also saves ``figure4_success_loss_only{suffix}`` (curves with per-panel matplotlib line legends — not for MAIN export).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from benchmark_results_utils import (
    load_all,
    get_regression_run_level,
    get_classification_run_level,
    get_variant_method_order_and_colors,
    plot_grouped_variant_legend,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "figures"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"

SMALL_SIZE = 2500
LARGE_SIZE = 25000


# Joint criterion grid for Figure 4b
S_PERCENT = (0, 10, 25, 50)
EPSILON_PERCENT = (5, 10, 20)


def _time_saved_pct(speedup: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 - 1.0 / np.maximum(np.asarray(speedup, dtype=float), 1e-6))


def _add_size_and_filter(run_df: pd.DataFrame, size_filter: str) -> pd.DataFrame:
    """Add size = n_samples * n_features; filter by size_filter: 'all' | 'small' | 'large'."""
    if run_df is None:
        return None
    df = run_df.copy()
    df["_size"] = df["n_samples"].fillna(1).astype(float) * df["n_features"].fillna(1).astype(float)
    if size_filter == "small":
        df = df[df["_size"] <= SMALL_SIZE]
    elif size_filter == "large":
        df = df[df["_size"] >= LARGE_SIZE]
    # 'all': no filter
    df = df.drop(columns=["_size"], errors="ignore")
    return df


def _prepare_run_with_pct(df, loss_col: str):
    """Add ts_pct and loss_pct (in %) to run-level df."""
    if df is None or df.empty:
        return None
    out = df.copy()
    out["ts_pct"] = _time_saved_pct(out["speedup"].values)
    out["loss_pct"] = 100.0 * out[loss_col].values
    return out


def _plot_loss_cdf_panel(
    ax,
    run_df: pd.DataFrame,
    loss_pct_col: str,
    title: str,
    method_order: list,
    method_colors: dict,
    method_labels: list,
    *,
    show_legend: bool = True,
    show_title: bool = True,
):
    """Plot P(loss ≤ ε) vs ε (in %) for each method/variant (empirical CDF of loss)."""
    if run_df is None or run_df.empty:
        ax.set_axis_off()
        if show_title and title:
            ax.set_title(title)
        return
    id_col = "method_key" if "method_key" in run_df.columns else "splitter"
    eps = np.linspace(0, 50, 201)
    for i, method in enumerate(method_order):
        sub = run_df[run_df[id_col] == method][loss_pct_col].dropna()
        if sub.empty:
            continue
        vals = sub.values
        frac = np.array([np.mean(vals <= e) for e in eps])
        color = method_colors.get(method, "#555555")
        label = method_labels[i] if method_labels and i < len(method_labels) else str(method).replace("_", " ")
        ax.plot(eps, frac, color=color, linewidth=2, label=label)
    ax.set_xlabel(r"Tolerance ε (% loss vs best)")
    ax.set_ylabel("Fraction of runs with loss ≤ ε")
    if show_title and title:
        ax.set_title(title)
    if show_legend:
        ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)


def _plot_joint_panel_three_tasks(
    ax,
    run_dfs: list,
    task_labels: list,
    s_pct: float,
    eps_pct: float,
    method_order: list,
    method_colors: dict,
    method_labels: list,
    *,
    show_legend: bool = True,
    show_title: bool = True,
):
    """One panel: P(ts ≥ s and loss ≤ ε). Bars grouped by task; color by method/variant (no best)."""
    id_col = "method_key" if (run_dfs and run_dfs[0] is not None and "method_key" in run_dfs[0].columns) else "splitter"
    # Exclude best from bar groups
    methods_no_best = [m for m in method_order if not m.startswith("best|") and m != "best"]
    n_tasks = len(task_labels)
    n_methods = len(methods_no_best)
    probs = np.zeros((n_tasks, n_methods))
    for t, run_df in enumerate(run_dfs):
        if run_df is None or run_df.empty:
            continue
        for m, method in enumerate(methods_no_best):
            sub = run_df[run_df[id_col] == method]
            if sub.empty:
                continue
            ok = (sub["ts_pct"] >= s_pct) & (sub["loss_pct"] <= eps_pct)
            probs[t, m] = ok.mean()

    x = np.arange(n_tasks)
    bar_w = max(0.08, 0.35 / max(n_methods, 1))
    offsets = np.linspace(-(n_methods - 1) * bar_w / 2, (n_methods - 1) * bar_w / 2, n_methods) if n_methods else []
    labels_no_best = [method_labels[method_order.index(m)] if m in method_order and method_labels else m.replace("_", " ") for m in methods_no_best]
    for m, method in enumerate(methods_no_best):
        ax.bar(
            x + offsets[m],
            probs[:, m],
            width=bar_w,
            label=labels_no_best[m] if m < len(labels_no_best) else method.replace("_", " "),
            color=method_colors.get(method, "#555555"),
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.02)
    if show_title:
        ax.set_title(f"s ≥ {s_pct}%, ε ≤ {eps_pct}%")
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def _save_supp_figure4_joint_large_small(
    reg_run,
    gini_run,
    entropy_run,
    method_order: list,
    method_colors: dict,
    method_labels: list,
):
    """SUPP fig 6: joint 4×3 grid for large + same for small + one grouped legend. PNG only."""
    task_labels = ["Regression", "Gini", "Entropy"]
    run_dfs_large = []
    run_dfs_small = []
    for run_df, loss_col in (
        (reg_run, "loss_rmse_bounded"),
        (gini_run, "loss_f1"),
        (entropy_run, "loss_f1"),
    ):
        r_l = _add_size_and_filter(run_df, "large")
        r_s = _add_size_and_filter(run_df, "small")
        run_dfs_large.append(_prepare_run_with_pct(r_l, loss_col) if r_l is not None else None)
        run_dfs_small.append(_prepare_run_with_pct(r_s, loss_col) if r_s is not None else None)

    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 24), layout="constrained")
    gs = GridSpec(
        9,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.52],
        hspace=0.12,
        wspace=0.10,
    )
    axes_joint = [[None] * 3 for _ in range(8)]
    for i, s_pct in enumerate(S_PERCENT):
        for j, eps_pct in enumerate(EPSILON_PERCENT):
            ax_j = fig.add_subplot(gs[i, j])
            axes_joint[i][j] = ax_j
            _plot_joint_panel_three_tasks(
                ax_j,
                run_dfs_large,
                task_labels,
                s_pct,
                eps_pct,
                method_order,
                method_colors,
                method_labels,
                show_legend=False,
                show_title=False,
            )
    for i, s_pct in enumerate(S_PERCENT):
        for j, eps_pct in enumerate(EPSILON_PERCENT):
            ax_j = fig.add_subplot(gs[i + 4, j])
            axes_joint[i + 4][j] = ax_j
            _plot_joint_panel_three_tasks(
                ax_j,
                run_dfs_small,
                task_labels,
                s_pct,
                eps_pct,
                method_order,
                method_colors,
                method_labels,
                show_legend=False,
                show_title=False,
            )
    for j, eps_pct in enumerate(EPSILON_PERCENT):
        axes_joint[0][j].set_title(rf"$\varepsilon \leq {eps_pct}\%$", fontsize=9, fontweight="bold", pad=5)
        axes_joint[4][j].set_title(rf"$\varepsilon \leq {eps_pct}\%$", fontsize=9, fontweight="bold", pad=5)
    for i, s_pct in enumerate(S_PERCENT):
        axes_joint[i][0].annotate(
            rf"Large · $s \geq {s_pct}\%$ saved",
            xy=(-0.12, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=9,
            fontweight="bold",
        )
        axes_joint[i + 4][0].annotate(
            rf"Small · $s \geq {s_pct}\%$ saved",
            xy=(-0.12, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=9,
            fontweight="bold",
        )

    leg_ax = fig.add_subplot(gs[8, :])
    plot_grouped_variant_legend(
        leg_ax, method_order, method_colors, method_labels, fontsize=7, legend_style="patch"
    )
    out = SUPP_DIR / "supp_figure_06_success_joint_large_small.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def _save_supp_figure4_loss_only_large_small(
    reg_run,
    gini_run,
    entropy_run,
    method_order: list,
    method_colors: dict,
    method_labels: list,
):
    """SUPP fig 7: loss CDF row for large + row for small + one grouped legend. PNG only."""
    task_configs = [
        (reg_run, "loss_rmse_bounded", "Regression"),
        (gini_run, "loss_f1", "Gini"),
        (entropy_run, "loss_f1", "Entropy"),
    ]

    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 8.2), layout="constrained")
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.52],
        hspace=0.14,
        wspace=0.10,
    )
    axes_top = []
    axes_bot = []
    for j, (run_df, loss_col, task_label) in enumerate(task_configs):
        r_l = _add_size_and_filter(run_df, "large")
        r_s = _add_size_and_filter(run_df, "small")
        r_l = _prepare_run_with_pct(r_l, loss_col) if r_l is not None else None
        r_s = _prepare_run_with_pct(r_s, loss_col) if r_s is not None else None
        ax_top = fig.add_subplot(gs[0, j])
        ax_bot = fig.add_subplot(gs[1, j])
        _plot_loss_cdf_panel(
            ax_top,
            r_l,
            "loss_pct",
            title="",
            method_order=method_order,
            method_colors=method_colors,
            method_labels=method_labels,
            show_legend=False,
            show_title=False,
        )
        _plot_loss_cdf_panel(
            ax_bot,
            r_s,
            "loss_pct",
            title="",
            method_order=method_order,
            method_colors=method_colors,
            method_labels=method_labels,
            show_legend=False,
            show_title=False,
        )
        ax_top.set_title(task_label, fontsize=10, fontweight="bold", pad=6)
        axes_top.append(ax_top)
        axes_bot.append(ax_bot)

    axes_top[0].annotate(
        r"P(loss $\leq \varepsilon$)",
        xy=(-0.11, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
    )
    axes_bot[0].annotate(
        r"P(loss $\leq \varepsilon$)",
        xy=(-0.11, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
    )
    axes_top[0].annotate(
        "Large",
        xy=(-0.20, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
    )
    axes_bot[0].annotate(
        "Small",
        xy=(-0.20, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
    )

    leg_ax = fig.add_subplot(gs[2, :])
    plot_grouped_variant_legend(
        leg_ax, method_order, method_colors, method_labels, fontsize=7, legend_style="patch"
    )
    out = SUPP_DIR / "supp_figure_07_success_loss_only_large_small.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Figure 4: success-probability curves.")
    ap.add_argument(
        "--no-supp",
        action="store_true",
        help="Do not write merged supplementary PNGs to SUPP_FIGURES/ (supp_figure_06, supp_figure_07).",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(BENCHMARK_DIR, exclude_secretary_par=True, by_variant=True)

    reg_run = data.get("regression_run")
    if reg_run is None:
        reg_run = get_regression_run_level(BENCHMARK_DIR)
        if reg_run is not None:
            from benchmark_results_utils import SPLITTERS_NO_PAR
            reg_run = reg_run[reg_run["splitter"].isin(SPLITTERS_NO_PAR)].copy()
    gini_run = data.get("classification_gini_run")
    if gini_run is None:
        gini_run = get_classification_run_level(BENCHMARK_DIR, "gini")
    entropy_run = data.get("classification_entropy_run")
    if entropy_run is None:
        entropy_run = get_classification_run_level(BENCHMARK_DIR, "entropy")

    for df in (reg_run, gini_run, entropy_run):
        if df is not None and "variant" in df.columns:
            df["method_key"] = df["splitter"].astype(str) + "|" + df["variant"].fillna("").astype(str)
        elif df is not None:
            df["method_key"] = df["splitter"].astype(str)

    method_order, method_colors, method_labels = get_variant_method_order_and_colors(reg_run, gini_run, entropy_run)

    task_configs = [
        (reg_run, "loss_rmse_bounded", "Regression"),
        (gini_run, "loss_f1", "Gini"),
        (entropy_run, "loss_f1", "Entropy"),
    ]

    for size_filter in ("all", "small", "large"):
        reg_f = _add_size_and_filter(reg_run, size_filter)
        gini_f = _add_size_and_filter(gini_run, size_filter)
        entropy_f = _add_size_and_filter(entropy_run, size_filter)

        suffix = f"_{size_filter}" if size_filter != "all" else ""

        reg_p = _prepare_run_with_pct(reg_f, "loss_rmse_bounded") if reg_f is not None else None
        gini_p = _prepare_run_with_pct(gini_f, "loss_f1") if gini_f is not None else None
        entropy_p = _prepare_run_with_pct(entropy_f, "loss_f1") if entropy_f is not None else None
        run_dfs = [reg_p, gini_p, entropy_p]
        task_labels = ["Regression", "Gini", "Entropy"]

        # --- figure4_success_combined: loss CDF row only + grouped legend (patch style) ---
        fig_c = plt.figure(figsize=(11, 5.8), layout="constrained")
        gs = GridSpec(
            2,
            3,
            figure=fig_c,
            height_ratios=[1, 0.52],
            hspace=0.14,
            wspace=0.10,
        )
        axes_top = []
        for j, (run_df, loss_col, task_label) in enumerate(task_configs):
            r = _add_size_and_filter(run_df, size_filter)
            r = _prepare_run_with_pct(r, loss_col) if r is not None else None
            ax_top = fig_c.add_subplot(gs[0, j])
            axes_top.append(ax_top)
            _plot_loss_cdf_panel(
                ax_top,
                r,
                "loss_pct",
                title=f"{task_label} – P(loss ≤ ε) vs ε",
                method_order=method_order,
                method_colors=method_colors,
                method_labels=method_labels,
                show_legend=False,
                show_title=False,
            )
        # Match figure4_success_loss_only panel titles (MAIN text figure)
        col_titles_top = [
            "Regression – P(loss ≤ ε) vs ε",
            "Classification (Gini) – P(loss ≤ ε) vs ε",
            "Classification (Entropy) – P(loss ≤ ε) vs ε",
        ]
        for j in range(3):
            axes_top[j].set_title(col_titles_top[j], fontsize=10, fontweight="bold", pad=6)
        axes_top[0].annotate(
            r"P(loss $\leq \varepsilon$)",
            xy=(-0.11, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize=9,
            fontweight="bold",
        )

        leg_ax = fig_c.add_subplot(gs[1, :])
        plot_grouped_variant_legend(
            leg_ax, method_order, method_colors, method_labels, fontsize=7, legend_style="point"
        )
        for ext in ("pdf", "png"):
            fig_c.savefig(
                OUT_DIR / f"figure4_success_combined{suffix}.{ext}",
                bbox_inches="tight",
                dpi=(150 if ext == "png" else None),
            )
        plt.close(fig_c)
        print(f"Saved figure4_success_combined{suffix}.pdf and .png")

        # --- figure4_success_joint: 4×3 joint panels + row/column titles + grouped legend ---
        fig_j = plt.figure(figsize=(11, 14.5), layout="constrained")
        gs_j = GridSpec(
            5,
            3,
            figure=fig_j,
            height_ratios=[1.0, 1.0, 1.0, 1.0, 0.55],
            hspace=0.14,
            wspace=0.10,
        )
        axes_joint = [[None] * 3 for _ in range(len(S_PERCENT))]
        for i, s_pct in enumerate(S_PERCENT):
            for j, eps_pct in enumerate(EPSILON_PERCENT):
                ax_j = fig_j.add_subplot(gs_j[i, j])
                axes_joint[i][j] = ax_j
                _plot_joint_panel_three_tasks(
                    ax_j,
                    run_dfs,
                    task_labels,
                    s_pct,
                    eps_pct,
                    method_order,
                    method_colors,
                    method_labels,
                    show_legend=False,
                    show_title=False,
                )
        for j, eps_pct in enumerate(EPSILON_PERCENT):
            axes_joint[0][j].set_title(rf"$\varepsilon \leq {eps_pct}\%$", fontsize=9, fontweight="bold", pad=5)
        for i, s_pct in enumerate(S_PERCENT):
            axes_joint[i][0].annotate(
                rf"$s \geq {s_pct}\%$ saved",
                xy=(-0.12, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                rotation=90,
                fontsize=9,
                fontweight="bold",
            )

        leg_ax_j = fig_j.add_subplot(gs_j[4, :])
        plot_grouped_variant_legend(
            leg_ax_j, method_order, method_colors, method_labels, fontsize=7, legend_style="patch"
        )
        for ext in ("pdf", "png"):
            fig_j.savefig(
                OUT_DIR / f"figure4_success_joint{suffix}.{ext}",
                bbox_inches="tight",
                dpi=(150 if ext == "png" else None),
            )
        if size_filter == "all":
            SUPP_DIR.mkdir(parents=True, exist_ok=True)
            supp_joint = SUPP_DIR / "supp_figure_12_success_joint_all.png"
            fig_j.savefig(supp_joint, bbox_inches="tight", dpi=200)
            print(f"Saved {supp_joint}")
        plt.close(fig_j)
        print(f"Saved figure4_success_joint{suffix}.pdf and .png")

        fig4a, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        for ax, (run_df, loss_col, task_label) in zip(axes, task_configs):
            r = _add_size_and_filter(run_df, size_filter)
            r = _prepare_run_with_pct(r, loss_col) if r is not None else None
            _plot_loss_cdf_panel(
                ax, r, "loss_pct",
                title=f"{task_label} – P(loss ≤ ε) vs ε",
                method_order=method_order,
                method_colors=method_colors,
                method_labels=method_labels,
            )
        fig4a.suptitle(f"Success probability (loss ≤ ε)" + (f" – {size_filter} datasets" if size_filter != "all" else ""), fontsize=11)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig4a.savefig(OUT_DIR / f"figure4_success_loss_only{suffix}.{ext}", bbox_inches="tight", dpi=(150 if ext == "png" else None))
        plt.close(fig4a)
        print(f"Saved figure4_success_loss_only{suffix}.pdf and .png")

    if not args.no_supp:
        _save_supp_figure4_joint_large_small(
            reg_run, gini_run, entropy_run, method_order, method_colors, method_labels
        )
        _save_supp_figure4_loss_only_large_small(
            reg_run, gini_run, entropy_run, method_order, method_colors, method_labels
        )


if __name__ == "__main__":
    main()
