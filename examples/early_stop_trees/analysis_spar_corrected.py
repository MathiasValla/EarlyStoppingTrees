#!/usr/bin/env python
"""Effort-loss analysis of the corrected S_par benchmark campaign.

Predictive loss and gain-evaluation effort are paired by dataset and seed with
exhaustive CART. Wall-clock results are deliberately excluded because S_par was
rerun in a later software/timing campaign.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmark_results_utils import load_all


TASKS = {
    "regression": {
        "prefix": "regression",
        "shards": ("reg_1", "reg_2"),
        "performance": "rmse_mean",
        "title": "Regression",
    },
    "classification_gini": {
        "prefix": "classification_gini",
        "shards": ("clf_1", "clf_2", "clf_3", "clf_4"),
        "performance": "f1_weighted_mean",
        "title": "Gini",
    },
    "classification_entropy": {
        "prefix": "classification_entropy",
        "shards": ("clf_1", "clf_2", "clf_3", "clf_4"),
        "performance": "f1_weighted_mean",
        "title": "Entropy",
    },
}

BUDGET_ORDER = ("2", "10", "ln_n", "0.1n", "sqrt_n")
BUDGET_LABELS = {
    "2": r"$B=2$",
    "10": r"$B=10$",
    "ln_n": r"$B=\mathrm{round}(\log N)$",
    "0.1n": r"$\rho=0.1$",
    "sqrt_n": r"$\rho=1/\sqrt{N}$",
}
BUDGET_COLORS = {
    "2": "#7f3c8d",
    "10": "#11a579",
    "ln_n": "#3969ac",
    "0.1n": "#e68310",
    "sqrt_n": "#d62728",
}
Q_MARKERS = {0.5: "o", 0.75: "s", 0.9: "^", 0.95: "D"}
DISPLAY_VARIANT = "samples=sqrt_n,q=0.9"
BOOTSTRAP_SEED = 20260818
N_BOOTSTRAP = 10_000


def _run_number(path: Path) -> int:
    match = re.search(r"run(\d{3})\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse run number from {path}")
    return int(match.group(1))


def _load_baseline(indir: Path, prefix: str) -> pd.DataFrame:
    frames = []
    for path in sorted(indir.glob(f"{prefix}_run*.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["splitter"].eq("best")].copy()
        frame["run"] = _run_number(path)
        frames.append(frame)
    if len(frames) != 100:
        raise ValueError(f"Expected 100 {prefix} baseline files, found {len(frames)}")
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["run", "dataset"]).any():
        raise ValueError(f"Duplicate exhaustive rows in {prefix}")
    return out


def _load_corrected(indir: Path, prefix: str, shards: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for shard in shards:
        paths = sorted((indir / shard).glob(f"{prefix}_run*.csv"))
        if len(paths) != 100:
            raise ValueError(f"Expected 100 {prefix} files in {shard}, found {len(paths)}")
        for path in paths:
            frame = pd.read_csv(path)
            if set(frame["splitter"]) != {"secretary_par"}:
                raise ValueError(f"Unexpected splitter in {path}")
            frame["run"] = _run_number(path)
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    key = ["run", "dataset", "variant"]
    if out.duplicated(key).any():
        raise ValueError(f"Duplicate corrected rows in {prefix}")
    counts = out.groupby(["run", "dataset"])["variant"].nunique()
    if not counts.eq(20).all():
        raise ValueError(f"Incomplete S_par grid in {prefix}")
    return out


def _parse_variant(variant: str) -> tuple[str, float]:
    fields = dict(part.split("=", 1) for part in str(variant).split(","))
    return fields["samples"], float(fields["q"])


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(N_BOOTSTRAP, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, (0.025, 0.975)))


def _task_points(task: str, baseline_dir: Path, corrected_dir: Path) -> pd.DataFrame:
    spec = TASKS[task]
    baseline = _load_baseline(baseline_dir, spec["prefix"])
    corrected = _load_corrected(corrected_dir, spec["prefix"], spec["shards"])
    baseline = baseline[
        [
            "run",
            "dataset",
            spec["performance"],
            "gain_evaluations_mean",
        ]
    ]
    joined = corrected.merge(
        baseline,
        on=["run", "dataset"],
        how="inner",
        suffixes=("_method", "_best"),
        validate="many_to_one",
    )
    if task == "regression":
        joined["loss_pct"] = 100.0 * (
            1.0 - joined["rmse_mean_best"] / joined["rmse_mean_method"]
        )
    else:
        joined["loss_pct"] = 100.0 * (
            joined["f1_weighted_mean_best"] - joined["f1_weighted_mean_method"]
        )
    joined["effort_saved_pct"] = 100.0 * (
        1.0
        - joined["gain_evaluations_mean_method"]
        / joined["gain_evaluations_mean_best"]
    )

    paired = (
        joined.groupby(["dataset", "variant"], as_index=False)
        .agg(
            loss_pct=("loss_pct", "median"),
            effort_saved_pct=("effort_saved_pct", "median"),
            n_paired_runs=("run", "nunique"),
        )
    )
    points = paired
    points.insert(0, "task", task)
    parsed = points["variant"].map(_parse_variant)
    points["sample_budget"] = parsed.map(lambda value: value[0])
    points["q"] = parsed.map(lambda value: value[1])
    return points


def _summarize(points: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (task, variant), frame in points.groupby(["task", "variant"], sort=True):
        row = {
            "task": task,
            "variant": variant,
            "n_datasets": frame["dataset"].nunique(),
            "centroid_loss_pct": frame["loss_pct"].mean(),
            "centroid_effort_saved_pct": frame["effort_saved_pct"].mean(),
            "median_loss_pct": frame["loss_pct"].median(),
        }
        for metric in ("loss_pct", "effort_saved_pct"):
            low, high = _bootstrap_mean(frame[metric].to_numpy(), rng)
            row[f"centroid_{metric}_ci_low"] = low
            row[f"centroid_{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _nonparametric_centroids(baseline_dir: Path, task: str) -> pd.DataFrame:
    loaded = load_all(baseline_dir, exclude_secretary_par=True, by_variant=True)
    if task == "regression":
        frame = loaded["regression_summary"].copy()
        loss_col = "loss_rmse_bounded_median"
    else:
        criterion = task.removeprefix("classification_")
        frame = loaded[f"classification_{criterion}_summary"].copy()
        loss_col = "loss_f1_median"
    frame["variant"] = frame["variant"].fillna("")
    frame["loss_pct"] = 100.0 * frame[loss_col]
    frame["effort_saved_pct"] = 100.0 * frame["effort_saved_total_median"]
    return (
        frame.groupby(["splitter", "variant"], as_index=False)
        .agg(
            loss_pct=("loss_pct", "mean"),
            effort_saved_pct=("effort_saved_pct", "mean"),
        )
    )


def _pareto_indices(frame: pd.DataFrame, x: str, y: str) -> list[int]:
    keep = []
    for idx, row in frame.iterrows():
        dominated = (
            (frame[x] >= row[x])
            & (frame[y] <= row[y])
            & ((frame[x] > row[x]) | (frame[y] < row[y]))
        ).any()
        if not dominated:
            keep.append(idx)
    return keep


def _mark_full_frontier(summary: pd.DataFrame, baseline_dir: Path) -> pd.DataFrame:
    out = summary.copy()
    out["on_full_effort_loss_frontier"] = False
    for task, spar in out.groupby("task"):
        other = _nonparametric_centroids(baseline_dir, task)
        renamed = spar.rename(
            columns={
                "centroid_loss_pct": "loss_pct",
                "centroid_effort_saved_pct": "effort_saved_pct",
            }
        )
        combined = pd.concat(
            [
                other.assign(source="validated", method_key=other["splitter"] + "|" + other["variant"]),
                renamed.assign(
                    source="secretary_par",
                    splitter="secretary_par",
                    method_key="secretary_par|" + renamed["variant"],
                ),
            ],
            ignore_index=True,
        )
        frontier = combined.loc[
            _pareto_indices(combined, "effort_saved_pct", "loss_pct"), "method_key"
        ]
        out.loc[out["task"].eq(task), "on_full_effort_loss_frontier"] = (
            "secretary_par|" + out.loc[out["task"].eq(task), "variant"]
        ).isin(set(frontier))
    return out


def _write_table(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[summary["variant"].eq(DISPLAY_VARIANT)].set_index("task")
    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Task & Loss & Effort saved (\%) \\",
        r"\midrule",
    ]
    for task in TASKS:
        row = selected.loc[task]
        lines.append(
            f"{TASKS[task]['title']} & {row.centroid_loss_pct:.3f} "
            f"[{row.centroid_loss_pct_ci_low:.3f}, {row.centroid_loss_pct_ci_high:.3f}] & "
            f"{row.centroid_effort_saved_pct:.2f} "
            f"[{row.centroid_effort_saved_pct_ci_low:.2f}, {row.centroid_effort_saved_pct_ci_high:.2f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def _plot(points: pd.DataFrame, summary: pd.DataFrame, baseline_dir: Path, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    for col, task in enumerate(TASKS):
        spar = summary[summary["task"].eq(task)].copy()
        other = _nonparametric_centroids(baseline_dir, task)
        ax = axes[col]
        front = other.loc[
            _pareto_indices(other, "effort_saved_pct", "loss_pct")
        ].sort_values("effort_saved_pct")
        ax.plot(
            front["effort_saved_pct"],
            front["loss_pct"],
            color="#8c8c8c",
            lw=1.2,
            alpha=0.7,
            zorder=1,
        )
        ax.scatter(
            other["effort_saved_pct"],
            other["loss_pct"],
            s=14,
            color="#bdbdbd",
            alpha=0.45,
            zorder=1,
        )
        for _, point in spar.iterrows():
            budget, q = _parse_variant(point["variant"])
            ax.scatter(
                point["centroid_effort_saved_pct"],
                point["centroid_loss_pct"],
                s=48,
                marker=Q_MARKERS[q],
                color=BUDGET_COLORS[budget],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        chosen = spar[spar["variant"].eq(DISPLAY_VARIANT)].iloc[0]
        ax.scatter(
            chosen["centroid_effort_saved_pct"],
            chosen["centroid_loss_pct"],
            s=120,
            marker="^",
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.2,
            zorder=4,
        )
        ax.axvline(0, color="#555555", lw=0.7, ls="--", alpha=0.65)
        ax.axhline(0, color="#555555", lw=0.7, ls="--", alpha=0.65)
        ax.grid(alpha=0.2)
        ax.set_title(TASKS[task]["title"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Centroid gain-evaluation effort saved (%)")
        ax.set_ylabel(
            "Relative RMSE loss (%)"
            if task == "regression"
            else "Weighted-F1 loss (points)"
        )
    budget_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BUDGET_COLORS[b],
               markeredgecolor="white", markersize=7, label=BUDGET_LABELS[b])
        for b in BUDGET_ORDER
    ]
    q_handles = [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="#666666",
               markeredgecolor="white", markersize=7, label=f"q={q:g}")
        for q, m in Q_MARKERS.items()
    ]
    context = Line2D([0], [0], color="#8c8c8c", marker="o", markersize=4,
                     label="validated-method frontier/context")
    chosen = Line2D([0], [0], marker="^", color="#111111", markerfacecolor="none",
                    markersize=9, linestyle="none", label=r"display: $\rho=1/\sqrt{N},q=0.9$")
    fig.legend(
        handles=budget_handles + q_handles + [context, chosen],
        loc="outside lower center", ncol=6, frameon=False, fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--baseline-dir", type=Path, default=script_dir / "benchmark_results")
    parser.add_argument("--corrected-dir", type=Path, default=script_dir / "benchmark_spar_corrected_100")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "spar_corrected_results")
    parser.add_argument("--article-dir", type=Path, default=script_dir.parents[1] / "RESEARCH_ARTICLE")
    args = parser.parse_args()

    points = pd.concat(
        [_task_points(task, args.baseline_dir, args.corrected_dir) for task in TASKS],
        ignore_index=True,
    )
    summary = _mark_full_frontier(_summarize(points), args.baseline_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output_dir / "spar_corrected_dataset_points.csv", index=False)
    summary.to_csv(args.output_dir / "spar_corrected_variant_summary.csv", index=False)

    article_table = args.article_dir / "TABLES" / "spar_corrected_summary.tex"
    example_table = script_dir / "tables" / "spar_corrected_summary.tex"
    _write_table(summary, article_table)
    _write_table(summary, example_table)
    figure_name = "supp_figure_16_secretary_par_screening.png"
    _plot(points, summary, args.baseline_dir, args.article_dir / "SUPP_FIGURES" / figure_name)
    _plot(points, summary, args.baseline_dir, script_dir / "SUPP_FIGURES" / figure_name)

    selected = summary[summary["variant"].eq(DISPLAY_VARIANT)]
    print(selected.to_string(index=False))
    if summary["on_full_effort_loss_frontier"].any():
        raise RuntimeError("An S_par variant entered the full effort-loss frontier; revise interpretation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
