"""Analyze the paired random-feature and ensemble revision controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")


TASK_ORDER = ("regression", "classification_gini", "classification_entropy")
TASK_LABELS = {
    "regression": "Regression",
    "classification_gini": "Classification (Gini)",
    "classification_entropy": "Classification (entropy)",
}
METHOD_LABELS = {
    ("exhaustive", "all"): "Exhaustive CART",
    ("secretary_all", "1overe"): r"$S_{\mathrm{all}}(1/e)$",
    ("random_feature_exact", "1"): "Exact random subset (1)",
    ("random_feature_exact", "0.1p"): "Exact random subset (0.1p)",
    ("random_feature_exact", "1overe"): "Exact random subset (p/e)",
    ("random_feature_exact", "0.5p"): "Exact random subset (p/2)",
    ("random_feature_exact", "2over3p"): "Exact random subset (2p/3)",
    ("ert", "1"): "ERT (1)",
    ("ert", "1over3p"): "ERT (p/3)",
    ("ert", "2over3p"): "ERT (2p/3)",
    ("ert", "all"): "ERT (p)",
}
COLORS = {
    "exhaustive": "#333333",
    "secretary_all": "#00876c",
    "random_feature_exact": "#3b6fb6",
    "ert": "#d98c00",
}
MARKERS = {
    "exhaustive": "s",
    "secretary_all": "*",
    "random_feature_exact": "o",
    "ert": "^",
}


def _load_control_shards(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("shards/*/*_run*.csv")):
        frame = pd.read_csv(path)
        match = re.search(r"run(\d+)", path.stem)
        if match and "run" not in frame:
            frame["run"] = int(match.group(1))
        frame["source_file"] = str(path.relative_to(root))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No control run files found below {root}")
    out = pd.concat(frames, ignore_index=True)
    duplicate = out.duplicated(["task", "dataset", "seed", "method", "variant"], keep=False)
    if duplicate.any():
        sample = out.loc[duplicate, ["task", "dataset", "seed", "method", "variant"]].head()
        raise ValueError(f"Duplicate control rows:\n{sample}")
    return out


def _paired_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "dataset", "seed"]
    baseline_cols = keys + ["fit_time_mean"]
    if (raw["task"] == "regression").any():
        baseline_cols.append("rmse_mean")
    for column in ("f1_weighted_mean", "f1_macro_mean", "recall_min_mean", "accuracy_mean"):
        if column in raw:
            baseline_cols.append(column)
    baseline = raw[(raw["method"] == "exhaustive") & (raw["variant"] == "all")][baseline_cols]
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in baseline_cols if column not in keys})
    merged = raw.merge(baseline, on=keys, how="left", validate="many_to_one")
    if merged["baseline_fit_time_mean"].isna().any():
        raise ValueError("At least one control row lacks a same-task, same-dataset, same-seed baseline")
    merged["time_saved_pct"] = 100.0 * (
        1.0 - merged["fit_time_mean"] / merged["baseline_fit_time_mean"]
    )
    merged["method_key"] = merged["method"].astype(str) + "|" + merged["variant"].astype(str)
    merged["label"] = [
        METHOD_LABELS.get((method, variant), f"{method} ({variant})")
        for method, variant in zip(merged["method"], merged["variant"])
    ]
    is_regression = merged["task"] == "regression"
    merged["primary_loss_pct"] = np.nan
    merged.loc[is_regression, "primary_loss_pct"] = 100.0 * (
        1.0
        - merged.loc[is_regression, "baseline_rmse_mean"]
        / merged.loc[is_regression, "rmse_mean"]
    )
    for metric in ("f1_weighted", "f1_macro", "recall_min", "accuracy"):
        column = f"{metric}_mean"
        baseline_column = f"baseline_{column}"
        loss_column = f"{metric}_loss_pct"
        if column in merged and baseline_column in merged:
            merged[loss_column] = 100.0 * (merged[baseline_column] - merged[column])
    classification = ~is_regression
    if "f1_weighted_loss_pct" in merged:
        merged.loc[classification, "primary_loss_pct"] = merged.loc[
            classification, "f1_weighted_loss_pct"
        ]
    return merged


def _hierarchical_centroid_interval(matrix: np.ndarray, rng, n_bootstrap: int):
    matrix = np.asarray(matrix, dtype=float)
    n_datasets, n_runs = matrix.shape
    observed = float(np.mean(np.median(matrix, axis=1)))
    draws = np.empty(n_bootstrap, dtype=float)
    chunk_size = 100
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        size = stop - start
        dataset_indices = rng.integers(0, n_datasets, size=(size, n_datasets))
        run_indices = rng.integers(0, n_runs, size=(size, n_datasets, n_runs))
        selected = matrix[dataset_indices[:, :, None], run_indices]
        draws[start:stop] = np.mean(np.median(selected, axis=2), axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def _complete_matrix(subset: pd.DataFrame, value: str):
    pivot = subset.pivot(index="dataset", columns="seed", values=value).sort_index()
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError(f"No complete dataset-by-seed matrix for {value}")
    return pivot


def _classification_family(dataset: str) -> str:
    dataset = str(dataset)
    strong_patterns = (
        (r"^GAMETES_Epistasis_2_Way", "gametes_epistasis_2way"),
        (r"^GAMETES_Epistasis_3_Way", "gametes_epistasis_3way"),
        (r"^GAMETES_Heterogeneity", "gametes_heterogeneity"),
        (r"^mfeat_", "mfeat"),
        (r"^monk[123]$", "monks"),
        (r"^waveform_(21|40)$", "waveform"),
        (r"^Hill_Valley_(with|without)_noise$", "hill_valley"),
        (r"^wine_quality_(red|white)$", "wine_quality"),
        (r"^led(7|24)$", "led"),
        (r"^analcatdata_boxing[12]$", "analcatdata_boxing"),
    )
    for pattern, family in strong_patterns:
        if re.search(pattern, dataset):
            return family
    return dataset


def _summaries(metrics: pd.DataFrame, n_bootstrap: int, seed: int):
    summary_rows = []
    dataset_rows = []
    tail_rows = []
    reliability_rows = []
    family_rows = []
    rng = np.random.default_rng(seed)
    metric_columns = ["time_saved_pct", "primary_loss_pct"]
    for optional in ("f1_macro_loss_pct", "recall_min_loss_pct", "accuracy_loss_pct"):
        if optional in metrics:
            metric_columns.append(optional)

    for (task, method, variant), subset in metrics.groupby(
        ["task", "method", "variant"], sort=True
    ):
        dataset_medians = (
            subset.groupby("dataset", as_index=False)[metric_columns].median()
        )
        dataset_medians["task"] = task
        dataset_medians["method"] = method
        dataset_medians["variant"] = variant
        dataset_medians["label"] = METHOD_LABELS.get((method, variant), f"{method} ({variant})")
        dataset_rows.append(dataset_medians)

        for metric in metric_columns:
            available = subset.dropna(subset=[metric])
            if available.empty:
                continue
            matrix = _complete_matrix(available, metric)
            estimate, low, high = _hierarchical_centroid_interval(
                matrix.to_numpy(), rng, n_bootstrap
            )
            summary_rows.append(
                {
                    "task": task,
                    "method": method,
                    "variant": variant,
                    "label": METHOD_LABELS.get((method, variant), f"{method} ({variant})"),
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_datasets": matrix.shape[0],
                    "n_seeds": matrix.shape[1],
                }
            )

        losses = dataset_medians["primary_loss_pct"].dropna().to_numpy()
        tail_rows.append(
            {
                "task": task,
                "method": method,
                "variant": variant,
                "label": METHOD_LABELS.get((method, variant), f"{method} ({variant})"),
                "median_loss_pct": float(np.median(losses)),
                "p90_loss_pct": float(np.quantile(losses, 0.90)),
                "p95_loss_pct": float(np.quantile(losses, 0.95)),
                "max_loss_pct": float(np.max(losses)),
            }
        )
        for min_saving in (0.0, 10.0, 20.0):
            for tolerance in (0.5, 1.0, 2.5):
                success = (
                    (dataset_medians["time_saved_pct"] >= min_saving)
                    & (dataset_medians["primary_loss_pct"] < tolerance)
                )
                reliability_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "variant": variant,
                        "min_time_saved_pct": min_saving,
                        "loss_tolerance_pct": tolerance,
                        "success_fraction": float(success.mean()),
                        "success_count": int(success.sum()),
                        "n_datasets": int(success.size),
                    }
                )

        if task.startswith("classification"):
            family_data = dataset_medians.copy()
            family_data["family"] = family_data["dataset"].map(_classification_family)
            family_summary = family_data.groupby("family", as_index=False)[
                ["time_saved_pct", "primary_loss_pct"]
            ].median()
            family_rows.append(
                {
                    "task": task,
                    "method": method,
                    "variant": variant,
                    "n_families": int(family_summary.shape[0]),
                    "family_centroid_time_saved_pct": float(family_summary["time_saved_pct"].mean()),
                    "family_centroid_loss_pct": float(family_summary["primary_loss_pct"].mean()),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.concat(dataset_rows, ignore_index=True),
        pd.DataFrame(tail_rows),
        pd.DataFrame(reliability_rows),
        pd.DataFrame(family_rows),
    )


def _order_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, method, variant), subset in metrics.groupby(["task", "method", "variant"]):
        correlation, pvalue = spearmanr(
            subset["evaluation_position"], subset["fit_time_mean"], nan_policy="omit"
        )
        counts = subset["evaluation_position"].value_counts()
        rows.append(
            {
                "task": task,
                "method": method,
                "variant": variant,
                "spearman_fit_time_vs_position": float(correlation),
                "spearman_pvalue": float(pvalue),
                "min_position_count": int(counts.min()),
                "max_position_count": int(counts.max()),
                "n_rows": int(subset.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _summary_lookup(summary: pd.DataFrame, task: str, metric: str):
    subset = summary[(summary["task"] == task) & (summary["metric"] == metric)]
    return subset.set_index(["method", "variant"])


def _plot_control(summary: pd.DataFrame, output: Path, loss_metric: str, ylabel: str):
    tasks = TASK_ORDER if loss_metric == "primary_loss_pct" else TASK_ORDER[1:]
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.6 * len(tasks), 4.2),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, task in zip(axes, tasks):
        x = _summary_lookup(summary, task, "time_saved_pct")
        y = _summary_lookup(summary, task, loss_metric)
        common = x.index.intersection(y.index)
        for family in ("random_feature_exact", "ert"):
            points = [idx for idx in common if idx[0] == family]
            points.sort(key=lambda idx: float(x.loc[idx, "estimate"]))
            if points:
                ax.plot(
                    [x.loc[idx, "estimate"] for idx in points],
                    [y.loc[idx, "estimate"] for idx in points],
                    color=COLORS[family],
                    linewidth=1.3,
                    alpha=0.75,
                    zorder=2,
                )
        for method, variant in common:
            xrow = x.loc[(method, variant)]
            yrow = y.loc[(method, variant)]
            label = METHOD_LABELS.get((method, variant), f"{method} ({variant})")
            ax.errorbar(
                xrow["estimate"],
                yrow["estimate"],
                xerr=[[xrow["estimate"] - xrow["ci_low"]], [xrow["ci_high"] - xrow["estimate"]]],
                yerr=[[yrow["estimate"] - yrow["ci_low"]], [yrow["ci_high"] - yrow["estimate"]]],
                fmt=MARKERS[method],
                markersize=10 if method == "secretary_all" else 6,
                color=COLORS[method],
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=0.8,
                capsize=2,
                label=label,
                zorder=4 if method == "secretary_all" else 3,
            )
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title(TASK_LABELS[task], fontsize=14, fontweight="bold")
        ax.set_xlabel("Centroid time saved (%)", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(ylabel, fontsize=12)
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="outside lower center",
        ncol=4,
        fontsize=11.5,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.33, wspace=0.24)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def _analyze_ensembles(root: Path, outdir: Path, n_bootstrap: int, seed: int):
    frames = []
    for path in sorted(root.glob("*/ensemble_*.csv")):
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        return
    raw = pd.concat(frames, ignore_index=True)
    raw["variant"] = "ensemble"
    raw["method_key"] = raw["method"]
    keys = ["task", "dataset", "seed"]
    baseline_columns = keys + ["fit_time_mean"]
    for column in ("rmse_mean", "f1_weighted_mean", "f1_macro_mean", "recall_min_mean", "accuracy_mean"):
        if column in raw:
            baseline_columns.append(column)
    baseline = raw[raw["method"] == "exhaustive"][baseline_columns].rename(
        columns={column: f"baseline_{column}" for column in baseline_columns if column not in keys}
    )
    metrics = raw.merge(baseline, on=keys, how="left", validate="many_to_one")
    metrics["time_saved_pct"] = 100.0 * (1.0 - metrics["fit_time_mean"] / metrics["baseline_fit_time_mean"])
    metrics["primary_loss_pct"] = np.where(
        metrics["task"] == "regression",
        100.0 * (1.0 - metrics["baseline_rmse_mean"] / metrics["rmse_mean"]),
        100.0 * (metrics["baseline_f1_weighted_mean"] - metrics["f1_weighted_mean"]),
    )
    if "f1_macro_mean" in metrics:
        metrics["f1_macro_loss_pct"] = 100.0 * (
            metrics["baseline_f1_macro_mean"] - metrics["f1_macro_mean"]
        )
        metrics["recall_min_loss_pct"] = 100.0 * (
            metrics["baseline_recall_min_mean"] - metrics["recall_min_mean"]
        )
    summary_rows = []
    rng = np.random.default_rng(seed)
    for (task, method), subset in metrics.groupby(["task", "method"]):
        for metric in ("time_saved_pct", "primary_loss_pct", "f1_macro_loss_pct", "recall_min_loss_pct"):
            if metric not in subset or subset[metric].isna().all():
                continue
            matrix = _complete_matrix(subset.dropna(subset=[metric]), metric)
            estimate, low, high = _hierarchical_centroid_interval(matrix.to_numpy(), rng, n_bootstrap)
            summary_rows.append(
                {
                    "task": task,
                    "method": method,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_datasets": matrix.shape[0],
                    "n_seeds": matrix.shape[1],
                }
            )
    metrics.to_csv(outdir / "ensemble_paired_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(outdir / "ensemble_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--ensemble-root", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw = _load_control_shards(args.control_root)
    metrics = _paired_metrics(raw)
    summary, dataset, tails, reliability, family = _summaries(
        metrics, args.n_bootstrap, args.seed
    )
    order = _order_diagnostics(metrics)
    raw.to_csv(args.outdir / "revision_control_raw_merged.csv", index=False)
    metrics.to_csv(args.outdir / "revision_control_paired_metrics.csv", index=False)
    summary.to_csv(args.outdir / "revision_control_summary.csv", index=False)
    dataset.to_csv(args.outdir / "revision_control_dataset_medians.csv", index=False)
    tails.to_csv(args.outdir / "revision_control_tail_losses.csv", index=False)
    reliability.to_csv(args.outdir / "revision_control_reliability.csv", index=False)
    family.to_csv(args.outdir / "revision_control_classification_family_sensitivity.csv", index=False)
    order.to_csv(args.outdir / "revision_control_order_diagnostics.csv", index=False)
    _plot_control(
        summary,
        args.outdir / "revision_control_primary_tradeoff",
        "primary_loss_pct",
        "Centroid predictive loss (%)",
    )
    if "f1_macro_loss_pct" in summary["metric"].unique():
        _plot_control(
            summary,
            args.outdir / "revision_control_macro_f1_tradeoff",
            "f1_macro_loss_pct",
            "Centroid macro-F1 loss (percentage points)",
        )
    if args.ensemble_root:
        _analyze_ensembles(
            args.ensemble_root, args.outdir, args.n_bootstrap, args.seed + 1
        )
    metadata = {
        "control_root": str(args.control_root.resolve()),
        "ensemble_root": str(args.ensemble_root.resolve()) if args.ensemble_root else None,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "interval": "paired two-stage percentile bootstrap over seeds within datasets and datasets",
        "classification_family_rule": "Only strongly named generator/alias groups are collapsed; all other entries remain singleton families.",
    }
    (args.outdir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
