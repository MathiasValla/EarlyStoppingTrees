#!/usr/bin/env python
"""Hierarchical inference for the repeated PMLB split-search benchmark.

The benchmark has two nested sampling levels that answer different questions:

1. Runs (random seeds) quantify stochastic variability conditional on a dataset.
   CV folds are fixed by the benchmark, so folds are not treated as independent
   replicates.  Every method is first contrasted with exhaustive CART in the
   same dataset and run, and run identifiers are resampled jointly across all
   methods within a dataset.
2. Datasets quantify heterogeneity across benchmark problems.  Resampling whole
   datasets gives an interval for generalization to a conceptual population of
   related datasets, subject to the important caveat that PMLB is not a random
   probability sample.

The script reports three percentile-bootstrap intervals for global estimands:

* ``within``: datasets fixed; paired run identifiers resampled within datasets;
* ``between``: observed dataset medians fixed; datasets resampled as clusters;
* ``hierarchical``: datasets and paired run identifiers both resampled.

Dataset-level median intervals use the paired run bootstrap.  Global summaries
include both the equal-dataset-weight centroid used by Figure 1 (the mean of
dataset-level medians) and the cross-dataset median used by summary tables.
For revision-designated representative methods only, paired Wilcoxon tests versus
exhaustive CART are Holm-adjusted within each task/metric family and a Friedman
test supplies an omnibus check.  Screened variants, including all
``S_par`` configurations, receive descriptive intervals but remain exploratory.

No generated result is written into the repository by default.  Pass a fresh
``--outdir`` under ``/tmp`` for an auditable analysis bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.stats import binomtest, friedmanchisquare, rankdata, spearmanr, wilcoxon


BASELINE_METHOD = "best|"
DEFAULT_SEED = 20260810
DEFAULT_BOOTSTRAPS = 10000
DEFAULT_CONFIDENCE = 0.95

AUDITED_SOURCE_RELATIVE_PATHS = (
    ("benchmark_source", "examples/early_stop_trees/benchmark_secretary_pmlb.py"),
    ("benchmark_source", "examples/early_stop_trees/benchmark_results_utils.py"),
    ("splitter_source", "treeple/tree/_early_stop_splitter.pyx"),
    ("splitter_source", "treeple/tree/_early_stop_splitter.pxd"),
    ("splitter_source", "treeple/tree/_classes.py"),
    (
        "reference_splitter_source",
        "treeple/_lib/sklearn_fork/sklearn/tree/_splitter.pyx",
    ),
    ("build_source", "pyproject.toml"),
    ("build_source", "meson.build"),
    ("build_source", "treeple/tree/meson.build"),
)


@dataclass(frozen=True)
class TaskConfig:
    """Raw-file and outcome configuration for one benchmark task."""

    name: str
    prefix: str
    performance_col: str
    baseline_performance_col: str


@dataclass(frozen=True)
class TimingDatasetBlock:
    """Complete paired timing runs for one dataset."""

    dataset: str
    runs: np.ndarray
    methods: tuple[str, ...]
    speedup: np.ndarray


TASKS = {
    "regression": TaskConfig(
        name="regression",
        prefix="regression",
        performance_col="rmse_mean",
        baseline_performance_col="rmse_best",
    ),
    "classification_gini": TaskConfig(
        name="classification_gini",
        prefix="classification_gini",
        performance_col="f1_weighted_mean",
        baseline_performance_col="f1_best",
    ),
    "classification_entropy": TaskConfig(
        name="classification_entropy",
        prefix="classification_entropy",
        performance_col="f1_weighted_mean",
        baseline_performance_col="f1_best",
    ),
}


RUN_METRICS = ("speedup", "predictive_loss", "effort_saved")
PRESENTATION_METRICS = (
    "time_saved_pct",
    "predictive_loss_pct",
    "effort_saved_pct",
)
TIME_SAVED_THRESHOLDS_PCT = (0.0, 10.0, 25.0, 50.0)
LOSS_TOLERANCES_PCT = (0.5, 1.0, 2.5)
PRIMARY_LOSS_MARGIN_PCT = 1.0
SENSITIVITY_LOSS_MARGINS_PCT = (0.5, 2.5)
TIMING_BLOCK_LENGTHS = (5, 10)
CONFIRMATORY_METHODS = (
    BASELINE_METHOD,
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

REGRESSION_ALIAS_FAMILIES = {
    "197_cpu_act": "cpu_act_alias",
    "573_cpu_act": "cpu_act_alias",
    "227_cpu_small": "cpu_small_alias",
    "562_cpu_small": "cpu_small_alias",
}


def analysis_tier(method_key: str) -> str:
    """Label revision-designated representatives versus screened variants."""
    return "confirmatory" if method_key in CONFIRMATORY_METHODS else "exploratory"


def require_confirmatory_methods(methods: list[str]) -> list[str]:
    """Require and return the complete revision-designated representative family."""
    available = set(methods)
    missing = [method for method in CONFIRMATORY_METHODS if method not in available]
    if missing:
        raise ValueError(
            "Missing required confirmatory method keys: " + ", ".join(missing)
        )
    return list(CONFIRMATORY_METHODS)


def dataset_family_id(task: str, dataset: str) -> str:
    """Return the conservative family used in dependence sensitivities."""
    dataset = str(dataset)
    if task != "regression":
        return dataset
    if dataset in REGRESSION_ALIAS_FAMILIES:
        return REGRESSION_ALIAS_FAMILIES[dataset]
    friedman = re.search(r"_fri_c([0-4])_", dataset)
    if friedman:
        return f"friedman_generator_c{friedman.group(1)}"
    return dataset


def _identity_paired_resample_indices(
    dataset_ids: tuple[str, ...],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Cache paired outer/inner draws by ordered dataset identities, not count."""
    if not dataset_ids:
        raise ValueError("At least one dataset identity is required")
    if len(set(dataset_ids)) != len(dataset_ids):
        raise ValueError("Dataset identities must be unique within a resampling pool")
    if dataset_ids not in cache:
        n_datasets = len(dataset_ids)
        cache[dataset_ids] = (
            rng.integers(0, n_datasets, size=(n_bootstrap, n_datasets)),
            rng.integers(0, n_bootstrap, size=(n_bootstrap, n_datasets)),
        )
    return cache[dataset_ids]


def metric_unit(task: str, metric: str) -> str:
    """Human-readable unit, distinguishing F1 percentage-point loss."""
    if metric == "time_saved_pct":
        return "percent_training_time"
    if metric == "effort_saved_pct":
        return "percent_gain_evaluations"
    if metric == "predictive_loss_pct" and task.startswith("classification_"):
        return "F1_percentage_points"
    if metric == "predictive_loss_pct":
        return "percent_bounded_relative_RMSE_loss"
    raise ValueError(f"Unknown task/metric unit: {task}/{metric}")


def _identity_percent(x: np.ndarray) -> np.ndarray:
    return 100.0 * x


def _speedup_to_time_saved(x: np.ndarray) -> np.ndarray:
    """Match the manuscript transform: 100 * (1 - 1 / median speedup)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (1.0 - 1.0 / x)
    out[~np.isfinite(out)] = np.nan
    return out


METRIC_TRANSFORMS: tuple[Callable[[np.ndarray], np.ndarray], ...] = (
    _speedup_to_time_saved,
    _identity_percent,
    _identity_percent,
)


def _method_key(splitter: pd.Series, variant: pd.Series) -> pd.Series:
    return splitter.astype(str) + "|" + variant.fillna("").astype(str)


def _run_number(path: Path) -> int:
    match = re.search(r"_run(\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot infer run number from {path}")
    return int(match.group(1))


def load_task_archive(indir: Path, task: str) -> tuple[pd.DataFrame, list[Path]]:
    """Load raw per-run CSVs for one task and attach run/method identifiers."""
    config = TASKS[task]
    paths = sorted(indir.glob(f"{config.prefix}_run*.csv"))
    if not paths:
        raise FileNotFoundError(f"No {config.prefix}_run*.csv files under {indir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Header-only benchmark file: {path}")
        frame["run"] = _run_number(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    required = {
        "run",
        "dataset",
        "splitter",
        "variant",
        "fit_time_mean",
        "gain_evaluations_mean",
        config.performance_col,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{task} archive is missing columns: {missing}")

    raw["variant"] = raw["variant"].fillna("").astype(str)
    raw["method_key"] = _method_key(raw["splitter"], raw["variant"])
    duplicate = raw.duplicated(["run", "dataset", "method_key"], keep=False)
    if duplicate.any():
        sample = raw.loc[duplicate, ["run", "dataset", "method_key"]].head()
        raise ValueError(f"Duplicate dataset/run/method rows in {task}:\n{sample}")
    return raw, paths


def prepare_paired_run_metrics(raw: pd.DataFrame, task: str) -> pd.DataFrame:
    """Create same-run effects relative to exhaustive CART.

    Regression uses the bounded RMSE loss from the manuscript,
    ``1 - RMSE_best / RMSE_method``.  Classification uses weighted-F1 loss,
    ``F1_best - F1_method``.  Timing and effort are paired ratios.
    """
    config = TASKS[task]
    baseline = raw[raw["method_key"] == BASELINE_METHOD].copy()
    if baseline.empty:
        raise ValueError(f"No exhaustive baseline ({BASELINE_METHOD}) in {task}")
    if baseline.duplicated(["run", "dataset"]).any():
        raise ValueError(f"Non-unique exhaustive baseline in {task}")

    baseline = baseline[
        [
            "run",
            "dataset",
            "fit_time_mean",
            "gain_evaluations_mean",
            config.performance_col,
        ]
    ].rename(
        columns={
            "fit_time_mean": "fit_time_best",
            "gain_evaluations_mean": "gain_evaluations_best",
            config.performance_col: config.baseline_performance_col,
        }
    )
    paired = raw.merge(
        baseline,
        on=["run", "dataset"],
        how="left",
        validate="many_to_one",
    )
    if paired["fit_time_best"].isna().any():
        missing = paired.loc[paired["fit_time_best"].isna(), ["run", "dataset"]]
        raise ValueError(f"Rows without a same-run exhaustive baseline in {task}:\n{missing.head()}")

    paired["speedup"] = paired["fit_time_best"] / paired["fit_time_mean"].replace(0, np.nan)
    paired["effort_saved"] = 1.0 - (
        paired["gain_evaluations_mean"]
        / paired["gain_evaluations_best"].replace(0, np.nan)
    )
    if task == "regression":
        paired["predictive_loss"] = 1.0 - (
            paired[config.baseline_performance_col]
            / paired[config.performance_col].replace(0, np.nan)
        )
        paired["predictive_loss"] = paired["predictive_loss"].clip(upper=1.0)
    else:
        paired["predictive_loss"] = (
            paired[config.baseline_performance_col] - paired[config.performance_col]
        )

    keep = [
        "run",
        "dataset",
        "splitter",
        "variant",
        "method_key",
        "n_samples",
        "n_features",
        *RUN_METRICS,
    ]
    return paired[keep].copy()


def _transform_run_medians(medians: np.ndarray) -> np.ndarray:
    """Transform (..., three run-median metrics) to manuscript percentages."""
    transformed = np.empty_like(medians, dtype=float)
    for index, transform in enumerate(METRIC_TRANSFORMS):
        transformed[..., index] = transform(medians[..., index])
    return transformed


def _percentile_interval(draws: np.ndarray, confidence: float, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    tail = 0.5 * (1.0 - confidence)
    return (
        np.nanquantile(draws, tail, axis=axis),
        np.nanquantile(draws, 1.0 - tail, axis=axis),
    )


def paired_run_bootstrap(
    run_metrics: pd.DataFrame,
    *,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
    chunk_size: int = 100,
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """Bootstrap run medians jointly across methods within each dataset.

    Returns dataset names, method keys, observed presentation metrics with shape
    ``(D, M, K)``, bootstrap draws with shape ``(D, M, K, B)``, run counts with
    shape ``(D, M)``, and diagnostics.  The same sampled run identifiers are
    used for every method and metric in a dataset, preserving pairing.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    duplicate = run_metrics.duplicated(
        ["dataset", "run", "method_key"], keep=False
    )
    if duplicate.any():
        sample = run_metrics.loc[
            duplicate, ["dataset", "run", "method_key"]
        ].head()
        raise ValueError(
            "Duplicated seed rows would artificially narrow uncertainty; "
            f"refusing input:\n{sample}"
        )
    datasets = sorted(run_metrics["dataset"].astype(str).unique())
    methods = sorted(run_metrics["method_key"].astype(str).unique())
    expected_runs = sorted(int(value) for value in run_metrics["run"].unique())
    method_index = {method: index for index, method in enumerate(methods)}
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_metrics = len(RUN_METRICS)

    observed = np.full((n_datasets, n_methods, n_metrics), np.nan, dtype=float)
    bootstrap = np.full(
        (n_datasets, n_methods, n_metrics, n_bootstrap), np.nan, dtype=float
    )
    run_counts = np.zeros((n_datasets, n_methods), dtype=int)
    diagnostic_rows = []

    for dataset_index, dataset in enumerate(datasets):
        subset = run_metrics[run_metrics["dataset"].astype(str) == dataset]
        methods_here = sorted(subset["method_key"].astype(str).unique())
        pivots = {
            metric: subset.pivot(index="run", columns="method_key", values=metric).reindex(
                columns=methods_here
            )
            for metric in RUN_METRICS
        }
        common_runs = set(pivots[RUN_METRICS[0]].index)
        for pivot in pivots.values():
            finite_rows = pivot.index[np.isfinite(pivot.to_numpy(dtype=float)).all(axis=1)]
            common_runs &= set(finite_rows)
        common_runs = sorted(common_runs)
        if not common_runs:
            raise ValueError(f"No complete paired runs for dataset {dataset}")

        block = np.stack(
            [pivots[metric].loc[common_runs].to_numpy(dtype=float) for metric in RUN_METRICS],
            axis=-1,
        )
        method_positions = np.asarray([method_index[method] for method in methods_here])
        run_counts[dataset_index, method_positions] = len(common_runs)
        observed_raw = np.median(block, axis=0)
        observed[dataset_index, method_positions, :] = _transform_run_medians(observed_raw)

        for start in range(0, n_bootstrap, chunk_size):
            stop = min(start + chunk_size, n_bootstrap)
            sample_index = rng.integers(
                0,
                len(common_runs),
                size=(stop - start, len(common_runs)),
            )
            sampled_medians = np.median(block[sample_index, :, :], axis=1)
            transformed = _transform_run_medians(sampled_medians)
            for local_method, global_method in enumerate(method_positions):
                bootstrap[
                    dataset_index,
                    global_method,
                    :,
                    start:stop,
                ] = transformed[:, local_method, :].T

        diagnostic_rows.append(
            {
                "dataset": dataset,
                "n_methods": len(methods_here),
                "n_runs_union": int(subset["run"].nunique()),
                "n_complete_paired_runs": len(common_runs),
                "min_method_runs": int(subset.groupby("method_key")["run"].nunique().min()),
                "max_method_runs": int(subset.groupby("method_key")["run"].nunique().max()),
                "excluded_run_ids": ";".join(
                    str(run) for run in sorted(set(expected_runs) - set(common_runs))
                ),
                "complete_block_status": (
                    "complete" if len(common_runs) == len(expected_runs) else "incomplete_flagged"
                ),
            }
        )

    return (
        datasets,
        methods,
        observed,
        bootstrap,
        run_counts,
        pd.DataFrame(diagnostic_rows),
    )


def dataset_interval_table(
    *,
    task: str,
    datasets: list[str],
    methods: list[str],
    observed: np.ndarray,
    bootstrap: np.ndarray,
    run_counts: np.ndarray,
    confidence: float,
) -> pd.DataFrame:
    """Return long-form paired-run intervals for every dataset/method/metric."""
    low, high = _percentile_interval(bootstrap, confidence, axis=-1)
    rows = []
    for dataset_index, dataset in enumerate(datasets):
        for method_index, method_key in enumerate(methods):
            splitter, _, variant = method_key.partition("|")
            for metric_index, metric in enumerate(PRESENTATION_METRICS):
                estimate = observed[dataset_index, method_index, metric_index]
                if not np.isfinite(estimate):
                    continue
                rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "method_key": method_key,
                        "splitter": splitter,
                        "variant": variant,
                        "analysis_tier": analysis_tier(method_key),
                        "metric": metric,
                        "unit": metric_unit(task, metric),
                        "estimate": estimate,
                        "ci_low": low[dataset_index, method_index, metric_index],
                        "ci_high": high[dataset_index, method_index, metric_index],
                        "confidence": confidence,
                        "n_paired_runs": int(run_counts[dataset_index, method_index]),
                        "resampling_scope": "paired_runs_within_dataset",
                    }
                )
    return pd.DataFrame(rows)


def point_estimate_consistency_check(
    *,
    run_metrics: pd.DataFrame,
    datasets: list[str],
    methods: list[str],
    observed: np.ndarray,
) -> dict:
    """Verify points/centroids against the direct manuscript aggregation."""
    direct = (
        run_metrics.groupby(["dataset", "method_key"], sort=True)[list(RUN_METRICS)]
        .median()
        .reset_index()
    )
    direct_values = _transform_run_medians(
        direct[list(RUN_METRICS)].to_numpy(dtype=float)
    )
    direct[list(PRESENTATION_METRICS)] = direct_values
    dataset_index = {dataset: index for index, dataset in enumerate(datasets)}
    method_index = {method: index for index, method in enumerate(methods)}
    discrepancies = []
    for row in direct.itertuples(index=False):
        expected = np.asarray(
            [getattr(row, metric) for metric in PRESENTATION_METRICS], dtype=float
        )
        actual = observed[
            dataset_index[str(row.dataset)], method_index[str(row.method_key)], :
        ]
        discrepancies.append(np.abs(actual - expected))
    max_dataset_error = float(np.nanmax(np.asarray(discrepancies)))

    direct_centroids = direct.groupby("method_key")[list(PRESENTATION_METRICS)].mean()
    observed_centroids = np.nanmean(observed, axis=0)
    centroid_discrepancies = []
    for method, row in direct_centroids.iterrows():
        centroid_discrepancies.append(
            np.abs(
                observed_centroids[method_index[str(method)], :]
                - row.to_numpy(dtype=float)
            )
        )
    max_centroid_error = float(np.nanmax(np.asarray(centroid_discrepancies)))
    return {
        "dataset_point_max_abs_error": max_dataset_error,
        "figure1_centroid_max_abs_error": max_centroid_error,
        "passed_atol_1e-12": bool(
            max_dataset_error <= 1e-12 and max_centroid_error <= 1e-12
        ),
    }


def _aggregate(values: np.ndarray, estimand: str, axis: int) -> np.ndarray:
    if estimand == "centroid_mean":
        return np.nanmean(values, axis=axis)
    if estimand == "cross_dataset_median":
        return np.nanmedian(values, axis=axis)
    raise ValueError(f"Unknown estimand: {estimand}")


def global_interval_table(
    *,
    task: str,
    datasets: list[str],
    methods: list[str],
    observed: np.ndarray,
    bootstrap: np.ndarray,
    confidence: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compute within, between, and two-stage hierarchical global intervals."""
    n_bootstrap = bootstrap.shape[-1]
    rows = []
    # Cache common outer resamples so method contrasts retain dataset pairing.
    resample_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    dataset_array = np.asarray(datasets, dtype=object)

    for method_index, method_key in enumerate(methods):
        splitter, _, variant = method_key.partition("|")
        for metric_index, metric in enumerate(PRESENTATION_METRICS):
            point_values = observed[:, method_index, metric_index]
            pool = bootstrap[:, method_index, metric_index, :]
            valid = np.isfinite(point_values) & np.isfinite(pool).any(axis=1)
            point_values = point_values[valid]
            pool = pool[valid, :]
            n_datasets = point_values.size
            if n_datasets == 0:
                continue
            valid_dataset_ids = tuple(str(value) for value in dataset_array[valid])
            dataset_index, inner_index = _identity_paired_resample_indices(
                valid_dataset_ids,
                n_bootstrap=n_bootstrap,
                rng=rng,
                cache=resample_cache,
            )

            for estimand in ("centroid_mean", "cross_dataset_median"):
                estimate = float(_aggregate(point_values, estimand, axis=0))
                within_draws = _aggregate(pool.T, estimand, axis=1)
                between_draws = _aggregate(point_values[dataset_index], estimand, axis=1)
                hierarchical_values = pool[dataset_index, inner_index]
                hierarchical_draws = _aggregate(hierarchical_values, estimand, axis=1)

                row = {
                    "task": task,
                    "method_key": method_key,
                    "splitter": splitter,
                    "variant": variant,
                    "analysis_tier": analysis_tier(method_key),
                    "metric": metric,
                    "unit": metric_unit(task, metric),
                    "estimand": estimand,
                    "estimate": estimate,
                    "confidence": confidence,
                    "n_datasets": n_datasets,
                    "n_bootstrap": n_bootstrap,
                    "interval_method": "percentile_bootstrap",
                }
                for scope, draws in (
                    ("within", within_draws),
                    ("between", between_draws),
                    ("hierarchical", hierarchical_draws),
                ):
                    low, high = _percentile_interval(draws, confidence, axis=0)
                    row[f"{scope}_ci_low"] = float(low)
                    row[f"{scope}_ci_high"] = float(high)
                    row[f"{scope}_se"] = float(np.nanstd(draws, ddof=1))
                rows.append(row)
    return pd.DataFrame(rows)


def dataset_family_map_table(task: str, datasets: list[str]) -> pd.DataFrame:
    """Document every entry-to-family assignment used by the sensitivity."""
    rows = []
    family_ids = [dataset_family_id(task, dataset) for dataset in datasets]
    family_sizes = pd.Series(family_ids).value_counts().to_dict()
    for dataset, family_id in zip(datasets, family_ids):
        if family_id.startswith("friedman_generator_"):
            rule = "related Friedman synthetic entries grouped by generator c0-c4"
        elif family_id.endswith("_alias"):
            rule = "exact duplicate PMLB aliases grouped together"
        else:
            rule = "singleton PMLB entry"
        rows.append(
            {
                "task": task,
                "dataset": dataset,
                "family_id": family_id,
                "family_size": int(family_sizes[family_id]),
                "grouping_rule": rule,
            }
        )
    return pd.DataFrame(rows)


def _family_aggregate_arrays(
    *,
    task: str,
    datasets: list[str],
    observed: np.ndarray,
    bootstrap: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Collapse related entries by their within-family median."""
    family_ids = np.asarray(
        [dataset_family_id(task, dataset) for dataset in datasets], dtype=object
    )
    families = sorted(str(value) for value in np.unique(family_ids))
    family_observed = np.full((len(families), *observed.shape[1:]), np.nan)
    family_bootstrap = np.full((len(families), *bootstrap.shape[1:]), np.nan)
    for family_index, family_id in enumerate(families):
        members = family_ids == family_id
        family_observed[family_index] = np.nanmedian(observed[members], axis=0)
        family_bootstrap[family_index] = np.nanmedian(bootstrap[members], axis=0)
    return families, family_observed, family_bootstrap


def family_balanced_global_interval_table(
    *,
    task: str,
    datasets: list[str],
    methods: list[str],
    observed: np.ndarray,
    bootstrap: np.ndarray,
    confidence: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Hierarchical intervals after equal-weighting related dataset families."""
    families, family_observed, family_bootstrap = _family_aggregate_arrays(
        task=task,
        datasets=datasets,
        observed=observed,
        bootstrap=bootstrap,
    )
    table = global_interval_table(
        task=task,
        datasets=families,
        methods=methods,
        observed=family_observed,
        bootstrap=family_bootstrap,
        confidence=confidence,
        rng=rng,
    )
    table = table.rename(columns={"n_datasets": "n_families"})
    table["estimand"] = table["estimand"].replace(
        {
            "centroid_mean": "family_centroid_mean",
            "cross_dataset_median": "cross_family_median",
        }
    )
    table["experimental_unit"] = "dataset_family"
    table["n_source_datasets"] = len(datasets)
    table["sensitivity_estimand"] = (
        "equal family weight after within-family median aggregation"
    )
    return table


def family_balanced_joint_test_table(
    *,
    task: str,
    datasets: list[str],
    methods: list[str],
    observed: np.ndarray,
    confidence: float,
) -> pd.DataFrame:
    """Simultaneous-success tests after collapsing related dataset families."""
    placeholder_bootstrap = observed[..., None]
    families, family_observed, _ = _family_aggregate_arrays(
        task=task,
        datasets=datasets,
        observed=observed,
        bootstrap=placeholder_bootstrap,
    )
    table = joint_time_loss_margin_test_table(
        task=task,
        methods=methods,
        observed=family_observed,
        confidence=confidence,
    )
    table = table.rename(columns={"n_complete_datasets": "n_complete_families"})
    table["joint_test_rule"] = (
        "one-sided exact binomial test of simultaneous family-level success"
    )
    table["joint_null"] = "P_family(time > 0 AND loss < margin) <= 0.5"
    table["joint_alternative"] = "P_family(time > 0 AND loss < margin) > 0.5"
    table["n_source_datasets"] = len(datasets)
    table["n_total_families"] = len(families)
    table["experimental_unit"] = "dataset_family"
    return table


def joint_reliability_interval_table(
    *,
    task: str,
    datasets: list[str],
    methods: list[str],
    observed: np.ndarray,
    bootstrap: np.ndarray,
    confidence: float,
    rng: np.random.Generator,
    time_thresholds_pct: tuple[float, ...] = TIME_SAVED_THRESHOLDS_PCT,
    loss_tolerances_pct: tuple[float, ...] = LOSS_TOLERANCES_PCT,
) -> pd.DataFrame:
    """Intervals for the dataset-level joint reliability probability.

    The experimental unit is a dataset.  A dataset is successful when its
    run-median point simultaneously meets the time-saving and predictive-loss
    constraints.  This deliberately avoids pooling dataset/run rows.
    """
    n_bootstrap = bootstrap.shape[-1]
    time_index = PRESENTATION_METRICS.index("time_saved_pct")
    loss_index = PRESENTATION_METRICS.index("predictive_loss_pct")
    rows = []
    resample_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    dataset_ids = np.asarray(datasets, dtype=object)
    if dataset_ids.size != observed.shape[0]:
        raise ValueError("Dataset identities do not match the observed array")

    for method_index, method_key in enumerate(methods):
        point_time = observed[:, method_index, time_index]
        point_loss = observed[:, method_index, loss_index]
        time_pool = bootstrap[:, method_index, time_index, :]
        loss_pool = bootstrap[:, method_index, loss_index, :]
        valid = (
            np.isfinite(point_time)
            & np.isfinite(point_loss)
            & np.isfinite(time_pool).any(axis=1)
            & np.isfinite(loss_pool).any(axis=1)
        )
        point_time = point_time[valid]
        point_loss = point_loss[valid]
        time_pool = time_pool[valid, :]
        loss_pool = loss_pool[valid, :]
        n_datasets = point_time.size
        if n_datasets == 0:
            continue
        valid_dataset_ids = tuple(str(value) for value in dataset_ids[valid])
        dataset_index, inner_index = _identity_paired_resample_indices(
            valid_dataset_ids,
            n_bootstrap=n_bootstrap,
            rng=rng,
            cache=resample_cache,
        )

        splitter, _, variant = method_key.partition("|")
        for time_threshold in time_thresholds_pct:
            for loss_tolerance in loss_tolerances_pct:
                point_success = (
                    (point_time >= time_threshold)
                    & (point_loss <= loss_tolerance)
                )
                within_success = (
                    (time_pool >= time_threshold)
                    & (loss_pool <= loss_tolerance)
                ).T
                between_success = point_success[dataset_index]
                hierarchical_success = (
                    (time_pool[dataset_index, inner_index] >= time_threshold)
                    & (loss_pool[dataset_index, inner_index] <= loss_tolerance)
                )
                draws_by_scope = {
                    "within": np.mean(within_success, axis=1),
                    "between": np.mean(between_success, axis=1),
                    "hierarchical": np.mean(hierarchical_success, axis=1),
                }
                row = {
                    "task": task,
                    "method_key": method_key,
                    "splitter": splitter,
                    "variant": variant,
                    "analysis_tier": analysis_tier(method_key),
                    "time_saved_threshold_pct": time_threshold,
                    "loss_tolerance_pct": loss_tolerance,
                    "loss_unit": metric_unit(task, "predictive_loss_pct"),
                    "estimate": float(np.mean(point_success)),
                    "confidence": confidence,
                    "n_datasets": n_datasets,
                    "n_bootstrap": n_bootstrap,
                    "experimental_unit": "dataset",
                    "estimand": "P_dataset(median time saved >= s and median loss <= epsilon)",
                }
                for scope, draws in draws_by_scope.items():
                    low, high = _percentile_interval(draws, confidence, axis=0)
                    row[f"{scope}_ci_low"] = float(low)
                    row[f"{scope}_ci_high"] = float(high)
                    row[f"{scope}_se"] = float(np.std(draws, ddof=1))
                rows.append(row)
    return pd.DataFrame(rows)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down family-wise-error adjustment, preserving NaNs."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    finite_positions = np.flatnonzero(np.isfinite(pvalues))
    if finite_positions.size == 0:
        return adjusted
    order = finite_positions[np.argsort(pvalues[finite_positions], kind="mergesort")]
    m = order.size
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (m - rank) * pvalues[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted


def _wilcoxon_summary(difference: np.ndarray) -> tuple[float, float, int]:
    difference = np.asarray(difference, dtype=float)
    difference = difference[np.isfinite(difference)]
    nonzero = difference[~np.isclose(difference, 0.0, rtol=0.0, atol=1e-12)]
    if nonzero.size == 0:
        return 1.0, 0.0, 0
    result = wilcoxon(
        nonzero,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    ranks = rankdata(np.abs(nonzero), method="average")
    signed_rank_biserial = float(
        (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    )
    return float(result.pvalue), signed_rank_biserial, int(nonzero.size)


def inferential_test_tables(
    *,
    task: str,
    datasets: list[str],
    methods: list[str],
    observed: np.ndarray,
    global_intervals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Holm-adjusted baseline comparisons and Friedman omnibus tests."""
    confirmatory_methods = require_confirmatory_methods(methods)
    confirmatory_indices = [methods.index(method) for method in confirmatory_methods]
    baseline_index = confirmatory_methods.index(BASELINE_METHOD)
    pairwise_rows = []
    omnibus_rows = []

    for metric_index, metric in enumerate(PRESENTATION_METRICS):
        matrix = observed[:, confirmatory_indices, metric_index]
        complete = np.isfinite(matrix).all(axis=1)
        complete_matrix = matrix[complete, :]
        if complete_matrix.shape[0] >= 2 and complete_matrix.shape[1] >= 3:
            result = friedmanchisquare(
                *[complete_matrix[:, index] for index in range(complete_matrix.shape[1])]
            )
            statistic = float(result.statistic)
            n_complete, n_methods = complete_matrix.shape
            kendalls_w = statistic / (n_complete * (n_methods - 1))
            omnibus_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "unit": metric_unit(task, metric),
                    "test": "friedman",
                    "statistic": statistic,
                    "df": n_methods - 1,
                    "pvalue": float(result.pvalue),
                    "kendalls_w": kendalls_w,
                    "n_complete_datasets": n_complete,
                    "n_methods": n_methods,
                }
            )

        metric_rows = []
        for method_index, method_key in enumerate(confirmatory_methods):
            if method_key == BASELINE_METHOD:
                continue
            valid = np.isfinite(matrix[:, method_index]) & np.isfinite(
                matrix[:, baseline_index]
            )
            difference = (
                matrix[valid, method_index] - matrix[valid, baseline_index]
            )
            pvalue, rank_biserial, n_nonzero = _wilcoxon_summary(difference)
            favorable = difference >= 0 if metric != "predictive_loss_pct" else difference <= 0

            global_rows = global_intervals[
                (global_intervals["method_key"] == method_key)
                & (global_intervals["metric"] == metric)
            ]
            centroid = global_rows[global_rows["estimand"] == "centroid_mean"]
            cross_median = global_rows[
                global_rows["estimand"] == "cross_dataset_median"
            ]
            splitter, _, variant = method_key.partition("|")
            metric_rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "unit": metric_unit(task, metric),
                    "method_key": method_key,
                    "splitter": splitter,
                    "variant": variant,
                    "analysis_tier": "confirmatory",
                    "reference_method": BASELINE_METHOD,
                    "method_selection_tier": "revision_designated_representative",
                    "n_paired_datasets": int(valid.sum()),
                    "n_nonzero_differences": n_nonzero,
                    "mean_paired_effect": float(np.mean(difference)),
                    "median_paired_effect": float(np.median(difference)),
                    "fraction_favorable_or_tied": float(np.mean(favorable)),
                    "wilcoxon_pvalue_raw": pvalue,
                    "signed_rank_biserial": rank_biserial,
                    "centroid_hierarchical_ci_low": float(
                        centroid["hierarchical_ci_low"].iloc[0]
                    ),
                    "centroid_hierarchical_ci_high": float(
                        centroid["hierarchical_ci_high"].iloc[0]
                    ),
                    "median_hierarchical_ci_low": float(
                        cross_median["hierarchical_ci_low"].iloc[0]
                    ),
                    "median_hierarchical_ci_high": float(
                        cross_median["hierarchical_ci_high"].iloc[0]
                    ),
                }
            )
        pvalues = np.asarray(
            [row["wilcoxon_pvalue_raw"] for row in metric_rows], dtype=float
        )
        adjusted = holm_adjust(pvalues)
        for row, pvalue_adjusted in zip(metric_rows, adjusted):
            row["wilcoxon_pvalue_holm"] = pvalue_adjusted
            row["reject_holm_0.05"] = bool(pvalue_adjusted < 0.05)
        pairwise_rows.extend(metric_rows)

    return pd.DataFrame(pairwise_rows), pd.DataFrame(omnibus_rows)


def _one_sided_sign_summary(
    favorable_difference: np.ndarray,
    *,
    confidence: float,
    tie_atol: float = 1e-12,
) -> dict[str, float | int]:
    """Test whether favorable signs occur with probability greater than 1/2."""
    difference = np.asarray(favorable_difference, dtype=float)
    difference = difference[np.isfinite(difference)]
    tied = np.isclose(difference, 0.0, rtol=0.0, atol=tie_atol)
    positive = int(np.sum((difference > 0.0) & ~tied))
    negative = int(np.sum((difference < 0.0) & ~tied))
    ties = int(np.sum(tied))
    n_non_ties = positive + negative
    if n_non_ties:
        test = binomtest(positive, n_non_ties, p=0.5, alternative="greater")
        interval = binomtest(positive, n_non_ties).proportion_ci(
            confidence_level=confidence, method="exact"
        )
        pvalue = float(test.pvalue)
        probability_low = float(interval.low)
        probability_high = float(interval.high)
    else:
        pvalue = np.nan
        probability_low = np.nan
        probability_high = np.nan
    return {
        "n_datasets": int(difference.size),
        "n_non_ties": n_non_ties,
        "n_favorable": positive,
        "n_unfavorable": negative,
        "n_ties": ties,
        "fraction_favorable_non_ties": (
            positive / n_non_ties if n_non_ties else np.nan
        ),
        "favorable_probability_ci_low": probability_low,
        "favorable_probability_ci_high": probability_high,
        "pvalue_raw": pvalue,
    }


def _apply_margin_family_holm(
    table: pd.DataFrame,
    *,
    pvalue_column: str,
    adjusted_column: str,
) -> pd.DataFrame:
    """Separate the primary method family from combined sensitivity analyses."""
    table = table.copy()
    table[adjusted_column] = np.nan
    table["multiplicity_family"] = ""
    table["multiplicity_family_size"] = 0
    table["evidence_role"] = ""

    primary = np.isclose(
        table["loss_margin_pct"].to_numpy(dtype=float),
        PRIMARY_LOSS_MARGIN_PCT,
    )
    sensitivity = table["loss_margin_pct"].isin(SENSITIVITY_LOSS_MARGINS_PCT).to_numpy()
    recognized = primary | sensitivity
    if not np.all(recognized):
        unexpected = sorted(table.loc[~recognized, "loss_margin_pct"].unique())
        raise ValueError(f"Unclassified loss margins: {unexpected}")

    families = (
        (
            primary,
            "primary_1.0pct_across_methods",
            "primary_confirmatory",
        ),
        (
            sensitivity,
            "sensitivity_0.5_and_2.5pct_across_methods_and_margins",
            "sensitivity_analysis",
        ),
    )
    for mask, family, role in families:
        positions = np.flatnonzero(mask)
        adjusted = holm_adjust(
            table.iloc[positions][pvalue_column].to_numpy(dtype=float)
        )
        table.loc[table.index[positions], adjusted_column] = adjusted
        table.loc[table.index[positions], "multiplicity_family"] = family
        table.loc[table.index[positions], "multiplicity_family_size"] = len(positions)
        table.loc[table.index[positions], "evidence_role"] = role
    return table


def loss_margin_sign_test_table(
    *,
    task: str,
    methods: list[str],
    observed: np.ndarray,
    confidence: float,
    margins_pct: tuple[float, ...] = LOSS_TOLERANCES_PCT,
) -> pd.DataFrame:
    """Exact sign tests for loss below revision-designated margins.

    For each margin, the null is that a favorable sign, ``loss < margin``, has
    probability at most 1/2 among non-tied datasets.  This is descriptive support
    for a loss margin; it does not estimate a model-based treatment contrast.
    """
    loss_index = PRESENTATION_METRICS.index("predictive_loss_pct")
    rows = []
    for margin in margins_pct:
        for method_key in require_confirmatory_methods(methods):
            if method_key == BASELINE_METHOD:
                continue
            method_index = methods.index(method_key)
            loss = observed[:, method_index, loss_index]
            summary = _one_sided_sign_summary(
                margin - loss,
                confidence=confidence,
            )
            splitter, _, variant = method_key.partition("|")
            rows.append(
                {
                    "task": task,
                    "method_key": method_key,
                    "splitter": splitter,
                    "variant": variant,
                    "method_selection_tier": "revision_designated_representative",
                    "loss_margin_pct": margin,
                    "is_primary_margin": bool(
                        np.isclose(margin, PRIMARY_LOSS_MARGIN_PCT)
                    ),
                    "loss_unit": metric_unit(task, "predictive_loss_pct"),
                    "n_datasets": summary["n_datasets"],
                    "n_non_ties": summary["n_non_ties"],
                    "n_below_margin": summary["n_favorable"],
                    "n_above_margin": summary["n_unfavorable"],
                    "n_equal_margin": summary["n_ties"],
                    "fraction_below_margin_non_ties": summary[
                        "fraction_favorable_non_ties"
                    ],
                    "below_margin_probability_ci_low": summary[
                        "favorable_probability_ci_low"
                    ],
                    "below_margin_probability_ci_high": summary[
                        "favorable_probability_ci_high"
                    ],
                    "loss_margin_sign_pvalue_raw": summary["pvalue_raw"],
                    "null_hypothesis": "P_dataset(loss < margin | non-tie) <= 0.5",
                    "alternative_hypothesis": "P_dataset(loss < margin | non-tie) > 0.5",
                }
            )
    table = _apply_margin_family_holm(
        pd.DataFrame(rows),
        pvalue_column="loss_margin_sign_pvalue_raw",
        adjusted_column="loss_margin_sign_pvalue_holm",
    )
    table["reject_component_holm_0.05"] = (
        table["loss_margin_sign_pvalue_holm"] < 0.05
    )
    return table


def runtime_superiority_sign_test_table(
    *,
    task: str,
    methods: list[str],
    observed: np.ndarray,
    confidence: float,
) -> pd.DataFrame:
    """Exact sign tests for positive dataset-level time saving versus CART."""
    time_index = PRESENTATION_METRICS.index("time_saved_pct")
    rows = []
    for method_key in require_confirmatory_methods(methods):
        if method_key == BASELINE_METHOD:
            continue
        method_index = methods.index(method_key)
        time_saved = observed[:, method_index, time_index]
        summary = _one_sided_sign_summary(time_saved, confidence=confidence)
        splitter, _, variant = method_key.partition("|")
        rows.append(
            {
                "task": task,
                "method_key": method_key,
                "splitter": splitter,
                "variant": variant,
                "analysis_tier": "confirmatory",
                "unit": metric_unit(task, "time_saved_pct"),
                "n_datasets": summary["n_datasets"],
                "n_non_ties": summary["n_non_ties"],
                "n_positive_time_saving": summary["n_favorable"],
                "n_negative_time_saving": summary["n_unfavorable"],
                "n_zero_ties": summary["n_ties"],
                "fraction_positive_non_ties": summary[
                    "fraction_favorable_non_ties"
                ],
                "positive_probability_ci_low": summary[
                    "favorable_probability_ci_low"
                ],
                "positive_probability_ci_high": summary[
                    "favorable_probability_ci_high"
                ],
                "runtime_sign_pvalue_raw": summary["pvalue_raw"],
                "null_hypothesis": "P_dataset(time saved > 0 | non-tie) <= 0.5",
                "alternative_hypothesis": "P_dataset(time saved > 0 | non-tie) > 0.5",
            }
        )
    adjusted = holm_adjust(
        np.asarray([row["runtime_sign_pvalue_raw"] for row in rows], dtype=float)
    )
    for row, pvalue_adjusted in zip(rows, adjusted):
        row["runtime_sign_pvalue_holm"] = pvalue_adjusted
        row["multiplicity_family"] = "runtime_superiority_across_methods"
        row["multiplicity_family_size"] = len(rows)
        row["reject_component_holm_0.05"] = bool(pvalue_adjusted < 0.05)
    return pd.DataFrame(rows)


def joint_time_loss_margin_test_table(
    *,
    task: str,
    methods: list[str],
    observed: np.ndarray,
    confidence: float,
    margins_pct: tuple[float, ...] = LOSS_TOLERANCES_PCT,
) -> pd.DataFrame:
    """Exact tests of simultaneous dataset-level runtime and loss success.

    A dataset succeeds only when that same dataset has positive time saving and
    predictive loss below the margin.  The one-sided exact binomial test asks
    whether the simultaneous-success probability exceeds one half.
    """
    if PRIMARY_LOSS_MARGIN_PCT not in margins_pct:
        raise ValueError(
            f"Primary loss margin {PRIMARY_LOSS_MARGIN_PCT} must be included"
        )
    time_index = PRESENTATION_METRICS.index("time_saved_pct")
    loss_index = PRESENTATION_METRICS.index("predictive_loss_pct")
    rows = []
    for margin in margins_pct:
        for method_key in require_confirmatory_methods(methods):
            if method_key == BASELINE_METHOD:
                continue
            method_index = methods.index(method_key)
            time_saved = observed[:, method_index, time_index]
            loss = observed[:, method_index, loss_index]
            complete = np.isfinite(time_saved) & np.isfinite(loss)
            time_summary = _one_sided_sign_summary(
                time_saved[complete], confidence=confidence
            )
            loss_summary = _one_sided_sign_summary(
                margin - loss[complete], confidence=confidence
            )
            joint_success = (
                (time_saved[complete] > 0.0) & (loss[complete] < margin)
            )
            joint_summary = _one_sided_sign_summary(
                np.where(joint_success, 1.0, -1.0),
                confidence=confidence,
            )
            splitter, _, variant = method_key.partition("|")
            rows.append(
                {
                    "task": task,
                    "method_key": method_key,
                    "splitter": splitter,
                    "variant": variant,
                    "method_selection_tier": "revision_designated_representative",
                    "loss_margin_pct": margin,
                    "is_primary_margin": bool(
                        np.isclose(margin, PRIMARY_LOSS_MARGIN_PCT)
                    ),
                    "loss_unit": metric_unit(task, "predictive_loss_pct"),
                    "time_unit": metric_unit(task, "time_saved_pct"),
                    "n_complete_datasets": int(np.sum(complete)),
                    "runtime_n_non_ties": time_summary["n_non_ties"],
                    "runtime_n_positive": time_summary["n_favorable"],
                    "runtime_n_negative": time_summary["n_unfavorable"],
                    "runtime_n_ties": time_summary["n_ties"],
                    "loss_n_non_ties": loss_summary["n_non_ties"],
                    "loss_n_below_margin": loss_summary["n_favorable"],
                    "loss_n_above_margin": loss_summary["n_unfavorable"],
                    "loss_n_equal_margin": loss_summary["n_ties"],
                    "runtime_sign_pvalue_raw": time_summary["pvalue_raw"],
                    "loss_margin_sign_pvalue_raw": loss_summary["pvalue_raw"],
                    "joint_n_success": joint_summary["n_favorable"],
                    "joint_n_failure": joint_summary["n_unfavorable"],
                    "joint_success_fraction": joint_summary[
                        "fraction_favorable_non_ties"
                    ],
                    "joint_success_probability_ci_low": joint_summary[
                        "favorable_probability_ci_low"
                    ],
                    "joint_success_probability_ci_high": joint_summary[
                        "favorable_probability_ci_high"
                    ],
                    "joint_success_pvalue_raw": joint_summary["pvalue_raw"],
                    "joint_test_rule": (
                        "one-sided exact binomial test of simultaneous "
                        "dataset-level success"
                    ),
                    "joint_null": "P_dataset(time > 0 AND loss < margin) <= 0.5",
                    "joint_alternative": "P_dataset(time > 0 AND loss < margin) > 0.5",
                }
            )
    table = _apply_margin_family_holm(
        pd.DataFrame(rows),
        pvalue_column="joint_success_pvalue_raw",
        adjusted_column="joint_success_pvalue_holm",
    )
    rejected = table["joint_success_pvalue_holm"] < 0.05
    primary = table["is_primary_margin"].astype(bool)
    table["supports_primary_joint_claim_0.05"] = rejected & primary
    table["sensitivity_joint_signal_0.05"] = rejected & ~primary
    return table


def _complete_timing_blocks(
    run_metrics: pd.DataFrame,
) -> tuple[list[str], list[str], list[TimingDatasetBlock]]:
    """Extract common finite run blocks, preserving method pairing."""
    duplicate = run_metrics.duplicated(
        ["dataset", "run", "method_key"], keep=False
    )
    if duplicate.any():
        raise ValueError("Duplicate dataset/run/method rows in timing input")
    datasets = sorted(run_metrics["dataset"].astype(str).unique())
    methods = sorted(run_metrics["method_key"].astype(str).unique())
    blocks = []
    for dataset in datasets:
        subset = run_metrics[run_metrics["dataset"].astype(str) == dataset]
        methods_here = tuple(sorted(subset["method_key"].astype(str).unique()))
        pivots = {
            metric: subset.pivot(
                index="run", columns="method_key", values=metric
            ).reindex(columns=methods_here)
            for metric in RUN_METRICS
        }
        common_runs = set(pivots[RUN_METRICS[0]].index)
        for pivot in pivots.values():
            finite = np.isfinite(pivot.to_numpy(dtype=float)).all(axis=1)
            common_runs &= set(pivot.index[finite])
        ordered_runs = np.asarray(sorted(common_runs), dtype=int)
        if ordered_runs.size == 0:
            raise ValueError(f"No complete paired timing runs for {dataset}")
        blocks.append(
            TimingDatasetBlock(
                dataset=dataset,
                runs=ordered_runs,
                methods=methods_here,
                speedup=pivots["speedup"].loc[ordered_runs].to_numpy(dtype=float),
            )
        )
    return datasets, methods, blocks


def timing_run_index_diagnostic_table(
    *,
    task: str,
    run_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Describe serial timing drift against the benchmark run index."""
    _, _, blocks = _complete_timing_blocks(run_metrics)
    rows = []
    for block in blocks:
        n_runs = block.runs.size
        midpoint = max(1, n_runs // 2)
        n_run_gaps = int(np.sum(np.diff(block.runs) != 1))
        for method_index, method_key in enumerate(block.methods):
            time_saved = _speedup_to_time_saved(block.speedup[:, method_index])
            if n_runs >= 2 and np.ptp(time_saved) > 0.0:
                correlation = spearmanr(block.runs, time_saved)
                rho = float(correlation.statistic)
                rho_pvalue = float(correlation.pvalue)
                slope_per_10_runs = float(
                    10.0 * np.polyfit(block.runs, time_saved, deg=1)[0]
                )
            else:
                rho = np.nan
                rho_pvalue = np.nan
                slope_per_10_runs = 0.0 if n_runs >= 2 else np.nan
            first_half = time_saved[:midpoint]
            second_half = time_saved[midpoint:]
            second_minus_first = (
                float(np.median(second_half) - np.median(first_half))
                if second_half.size
                else np.nan
            )
            splitter, _, variant = method_key.partition("|")
            rows.append(
                {
                    "task": task,
                    "dataset": block.dataset,
                    "method_key": method_key,
                    "splitter": splitter,
                    "variant": variant,
                    "analysis_tier": analysis_tier(method_key),
                    "unit": metric_unit(task, "time_saved_pct"),
                    "n_complete_paired_runs": n_runs,
                    "run_index_min": int(block.runs.min()),
                    "run_index_max": int(block.runs.max()),
                    "n_run_index_gaps": n_run_gaps,
                    "spearman_rho_run_index_vs_time_saved": rho,
                    "spearman_pvalue_unadjusted": rho_pvalue,
                    "linear_slope_pct_points_per_10_runs": slope_per_10_runs,
                    "second_half_minus_first_half_median_pct_points": second_minus_first,
                    "diagnostic_role": "serial_run_order_sensitivity",
                }
            )
    return pd.DataFrame(rows)


def _circular_block_sample_indices(
    *,
    n_runs: int,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample run positions using fixed-length circular moving blocks."""
    if n_runs < 1 or block_length < 1:
        raise ValueError("n_runs and block_length must be positive")
    n_blocks = int(np.ceil(n_runs / block_length))
    starts = rng.integers(0, n_runs, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_length, dtype=int)
    sampled = (starts[..., None] + offsets) % n_runs
    return sampled.reshape(n_bootstrap, -1)[:, :n_runs]


def timing_block_bootstrap_tables(
    *,
    task: str,
    run_metrics: pd.DataFrame,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.Generator,
    block_lengths: tuple[int, ...] = TIMING_BLOCK_LENGTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Circular block-bootstrap sensitivity for timing medians and centroids."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    datasets, methods, blocks = _complete_timing_blocks(run_metrics)
    dataset_index_by_name = {dataset: index for index, dataset in enumerate(datasets)}
    method_index_by_key = {method: index for index, method in enumerate(methods)}
    n_datasets = len(datasets)
    n_methods = len(methods)
    dataset_rows = []
    centroid_rows = []

    for block_length in block_lengths:
        if block_length < 1:
            raise ValueError("Timing block lengths must be positive")
        point = np.full((n_datasets, n_methods), np.nan, dtype=float)
        pool = np.full(
            (n_datasets, n_methods, n_bootstrap), np.nan, dtype=float
        )
        run_count = np.zeros((n_datasets, n_methods), dtype=int)

        for block in blocks:
            dataset_index = dataset_index_by_name[block.dataset]
            method_positions = np.asarray(
                [method_index_by_key[method] for method in block.methods], dtype=int
            )
            sample_index = _circular_block_sample_indices(
                n_runs=block.runs.size,
                block_length=block_length,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            sampled_speedup_medians = np.median(
                block.speedup[sample_index, :], axis=1
            )
            sampled_time_saved = _speedup_to_time_saved(sampled_speedup_medians)
            observed_time_saved = _speedup_to_time_saved(
                np.median(block.speedup, axis=0)
            )
            point[dataset_index, method_positions] = observed_time_saved
            pool[dataset_index, method_positions, :] = sampled_time_saved.T
            run_count[dataset_index, method_positions] = block.runs.size

        low, high = _percentile_interval(pool, confidence, axis=-1)
        for dataset_index, dataset in enumerate(datasets):
            for method_index, method_key in enumerate(methods):
                if not np.isfinite(point[dataset_index, method_index]):
                    continue
                splitter, _, variant = method_key.partition("|")
                dataset_rows.append(
                    {
                        "task": task,
                        "dataset": dataset,
                        "method_key": method_key,
                        "splitter": splitter,
                        "variant": variant,
                        "analysis_tier": analysis_tier(method_key),
                        "block_length_runs": block_length,
                        "estimate_time_saved_pct": point[
                            dataset_index, method_index
                        ],
                        "ci_low": low[dataset_index, method_index],
                        "ci_high": high[dataset_index, method_index],
                        "confidence": confidence,
                        "n_complete_paired_runs": int(
                            run_count[dataset_index, method_index]
                        ),
                        "n_bootstrap": n_bootstrap,
                        "interval_method": "circular_moving_block_percentile_bootstrap",
                        "unit": metric_unit(task, "time_saved_pct"),
                    }
                )

        resample_cache: dict[
            tuple[str, ...], tuple[np.ndarray, np.ndarray]
        ] = {}
        dataset_array = np.asarray(datasets, dtype=object)
        for method_index, method_key in enumerate(methods):
            point_values = point[:, method_index]
            method_pool = pool[:, method_index, :]
            valid = np.isfinite(point_values) & np.isfinite(method_pool).any(axis=1)
            point_values = point_values[valid]
            method_pool = method_pool[valid, :]
            valid_dataset_ids = tuple(str(value) for value in dataset_array[valid])
            outer_index, inner_index = _identity_paired_resample_indices(
                valid_dataset_ids,
                n_bootstrap=n_bootstrap,
                rng=rng,
                cache=resample_cache,
            )
            within_draws = np.mean(method_pool, axis=0)
            between_draws = np.mean(point_values[outer_index], axis=1)
            hierarchical_draws = np.mean(
                method_pool[outer_index, inner_index], axis=1
            )
            splitter, _, variant = method_key.partition("|")
            row = {
                "task": task,
                "method_key": method_key,
                "splitter": splitter,
                "variant": variant,
                "analysis_tier": analysis_tier(method_key),
                "block_length_runs": block_length,
                "estimand": "equal_dataset_weight_time_saved_centroid",
                "estimate": float(np.mean(point_values)),
                "confidence": confidence,
                "n_datasets": point_values.size,
                "n_bootstrap": n_bootstrap,
                "interval_method": "circular_moving_block_percentile_bootstrap",
                "unit": metric_unit(task, "time_saved_pct"),
            }
            for scope, draws in (
                ("within_block", within_draws),
                ("between_dataset", between_draws),
                ("hierarchical_block", hierarchical_draws),
            ):
                ci_low, ci_high = _percentile_interval(
                    draws, confidence, axis=0
                )
                row[f"{scope}_ci_low"] = float(ci_low)
                row[f"{scope}_ci_high"] = float(ci_high)
                row[f"{scope}_se"] = float(np.std(draws, ddof=1))
            centroid_rows.append(row)
        del pool

    return pd.DataFrame(dataset_rows), pd.DataFrame(centroid_rows)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(indir: Path, tasks: list[str]) -> pd.DataFrame:
    """Hash audited sources, benchmark metadata, and every selected raw CSV."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    test_path = script_path.with_name("test_analysis_inference.py")
    benchmark_metadata_path = (indir / "benchmark_metadata.json").resolve()
    audited_sources = [
        (category, (repo_root / relative_path).resolve())
        for category, relative_path in AUDITED_SOURCE_RELATIVE_PATHS
    ]
    required = [
        script_path,
        test_path,
        benchmark_metadata_path,
        *(path for _, path in audited_sources),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot build hash manifest; missing: " + ", ".join(missing))

    entries: list[tuple[str, str, Path]] = [
        ("analysis_source", "", script_path),
        ("test_source", "", test_path),
        ("benchmark_metadata", "", benchmark_metadata_path),
        *((category, "", path) for category, path in audited_sources),
    ]
    for task in tasks:
        config = TASKS[task]
        paths = sorted(indir.glob(f"{config.prefix}_run*.csv"))
        if not paths:
            raise FileNotFoundError(f"No raw input CSVs found for {task}")
        entries.extend(("raw_input_csv", task, path.resolve()) for path in paths)

    physical_paths = [path for _, _, path in entries]
    if len(set(physical_paths)) != len(physical_paths):
        raise ValueError("Hash manifest contains duplicate physical paths")
    rows = []
    resolved_indir = indir.resolve()
    for category, task, path in entries:
        try:
            logical_path = str(path.relative_to(resolved_indir))
        except ValueError:
            try:
                logical_path = str(path.relative_to(repo_root))
            except ValueError:
                logical_path = path.name
        rows.append(
            {
                "category": category,
                "task": task,
                "logical_path": logical_path,
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["category", "task", "logical_path"], kind="mergesort"
    ).reset_index(drop=True)


def source_git_provenance() -> dict[str, object]:
    """Record the audited checkout without implying archive-binary identity."""
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    relative_paths = [
        str(script_path.relative_to(repo_root)),
        str(script_path.with_name("test_analysis_inference.py").relative_to(repo_root)),
        *(path for _, path in AUDITED_SOURCE_RELATIVE_PATHS),
    ]

    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--short", "--", *relative_paths)
    return {
        "repository_root": str(repo_root),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "audited_source_dirty": bool(status.stdout.strip()),
        "audited_source_status": status.stdout.splitlines(),
        "identity_scope": (
            "This records the source audited for the revision. The April benchmark "
            "archive did not retain the loaded compiled extension or a source commit, "
            "so bitwise identity with the archived executable cannot be asserted."
        ),
    }


def analyze_task(
    *,
    indir: Path,
    task: str,
    n_bootstrap: int,
    confidence: float,
    seed_sequence: np.random.SeedSequence,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Run all inference stages for one task."""
    raw, paths = load_task_archive(indir, task)
    run_metrics = prepare_paired_run_metrics(raw, task)
    run_seed, global_seed, reliability_seed, timing_seed, family_seed = (
        seed_sequence.spawn(5)
    )
    (
        datasets,
        methods,
        observed,
        bootstrap,
        run_counts,
        diagnostics,
    ) = paired_run_bootstrap(
        run_metrics,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        rng=np.random.default_rng(run_seed),
    )
    require_confirmatory_methods(methods)
    dataset_intervals = dataset_interval_table(
        task=task,
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=bootstrap,
        run_counts=run_counts,
        confidence=confidence,
    )
    global_intervals = global_interval_table(
        task=task,
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=bootstrap,
        confidence=confidence,
        rng=np.random.default_rng(global_seed),
    )
    pairwise, omnibus = inferential_test_tables(
        task=task,
        datasets=datasets,
        methods=methods,
        observed=observed,
        global_intervals=global_intervals,
    )
    joint_reliability = joint_reliability_interval_table(
        task=task,
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=bootstrap,
        confidence=confidence,
        rng=np.random.default_rng(reliability_seed),
    )
    loss_margin_sign = loss_margin_sign_test_table(
        task=task,
        methods=methods,
        observed=observed,
        confidence=confidence,
    )
    runtime_superiority = runtime_superiority_sign_test_table(
        task=task,
        methods=methods,
        observed=observed,
        confidence=confidence,
    )
    joint_time_loss_margin = joint_time_loss_margin_test_table(
        task=task,
        methods=methods,
        observed=observed,
        confidence=confidence,
    )
    family_map = dataset_family_map_table(task, datasets)
    family_global = family_balanced_global_interval_table(
        task=task,
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=bootstrap,
        confidence=confidence,
        rng=np.random.default_rng(family_seed),
    )
    family_joint = family_balanced_joint_test_table(
        task=task,
        datasets=datasets,
        methods=methods,
        observed=observed,
        confidence=confidence,
    )
    consistency_check = point_estimate_consistency_check(
        run_metrics=run_metrics,
        datasets=datasets,
        methods=methods,
        observed=observed,
    )
    if not consistency_check["passed_atol_1e-12"]:
        raise RuntimeError(
            f"Point estimates do not reproduce manuscript conventions: {consistency_check}"
        )
    del bootstrap
    timing_run_index = timing_run_index_diagnostic_table(
        task=task,
        run_metrics=run_metrics,
    )
    timing_block_dataset, timing_block_centroid = timing_block_bootstrap_tables(
        task=task,
        run_metrics=run_metrics,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        rng=np.random.default_rng(timing_seed),
    )
    diagnostics.insert(0, "task", task)
    inventory = {
        "task": task,
        "n_input_files": len(paths),
        "run_ids": sorted(int(value) for value in raw["run"].unique()),
        "n_raw_rows": int(raw.shape[0]),
        "n_datasets": len(datasets),
        "n_methods": len(methods),
        "method_keys": methods,
        "min_complete_paired_runs": int(diagnostics["n_complete_paired_runs"].min()),
        "max_complete_paired_runs": int(diagnostics["n_complete_paired_runs"].max()),
        "incomplete_dataset_blocks": diagnostics.loc[
            diagnostics["complete_block_status"] != "complete",
            ["dataset", "n_complete_paired_runs", "excluded_run_ids"],
        ].to_dict(orient="records"),
        "point_estimate_check": consistency_check,
    }
    return (
        dataset_intervals,
        global_intervals,
        pairwise,
        omnibus,
        joint_reliability,
        loss_margin_sign,
        runtime_superiority,
        joint_time_loss_margin,
        timing_run_index,
        timing_block_dataset,
        timing_block_centroid,
        diagnostics,
        family_map,
        family_global,
        family_joint,
        inventory,
    )


def _metadata(
    *,
    indir: Path,
    tasks: list[str],
    seed: int,
    n_bootstrap: int,
    confidence: float,
    inventories: list[dict],
    file_manifest: pd.DataFrame,
) -> dict:
    benchmark_metadata_path = indir / "benchmark_metadata.json"
    benchmark_metadata = (
        json.loads(benchmark_metadata_path.read_text())
        if benchmark_metadata_path.is_file()
        else None
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(indir.resolve()),
        "tasks": tasks,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "interval_method": "nonparametric percentile bootstrap",
        "estimands": {
            "dataset_point": "median across random-seed runs after same-run pairing to exhaustive CART",
            "centroid_mean": "equal-weight mean across dataset-level median points (Figure 1 centroid)",
            "cross_dataset_median": "median across dataset-level median points (summary-table estimand)",
            "time_saved_pct": "100 * (1 - 1 / median_run(t_best / t_method))",
            "regression_predictive_loss_pct": "100 * median_run(1 - RMSE_best / RMSE_method)",
            "classification_predictive_loss_pct": "100 * median_run(F1_best - F1_method)",
            "effort_saved_pct": "100 * median_run(1 - gains_method / gains_best)",
        },
        "resampling": {
            "within": "resample run identifiers within dataset; use one shared draw across all methods and metrics",
            "between": "resample complete datasets as clusters; hold observed run medians fixed",
            "hierarchical": "resample datasets by identity, then select a paired-run bootstrap draw for every sampled dataset occurrence",
            "resample_pairing_key": "ordered tuple of actual valid dataset identities",
            "cv_folds": "not resampled because benchmark folds are fixed and fold scores were archived only as summaries",
        },
        "multiple_comparisons": {
            "confirmatory_methods": list(CONFIRMATORY_METHODS),
            "confirmatory_set_requirement": "all ten keys are mandatory in every task; analysis fails if any key is absent",
            "omnibus": "Friedman rank test across revision-designated representative methods using complete dataset blocks",
            "post_hoc": "two-sided paired Wilcoxon signed-rank tests versus exhaustive CART, limited to revision-designated representatives",
            "post_hoc_adjustment": "Holm family-wise-error correction separately within each task and metric",
            "loss_margin_sign_support": "exact one-sided sign tests of P_dataset(loss < margin | non-tie) > 0.5",
            "runtime_superiority_support": "exact one-sided sign tests of P_dataset(time saved > 0 | non-tie) > 0.5, Holm-adjusted within task",
            "joint_simultaneous_success_test": "one-sided exact binomial test of whether more than half of dataset entries simultaneously have positive time saving and loss below the margin",
            "primary_loss_margin_pct": PRIMARY_LOSS_MARGIN_PCT,
            "primary_adjustment": "Holm across the nine non-baseline representative methods within each task at the task-specific 1.0 margin",
            "sensitivity_margins_pct": list(SENSITIVITY_LOSS_MARGINS_PCT),
            "sensitivity_adjustment": "Holm across all 18 method-by-margin hypotheses within each task; sensitivity findings are not labeled primary confirmation",
            "exploratory_policy": "all non-representative variants receive descriptive intervals only and are excluded from confirmatory tests; archived S_par rows are additionally excluded because the implementation audit found calibration defects",
        },
        "s_par_archive_disposition": {
            "status": "defective archived proxy-parametric ablation; excluded from main results, rankings, recommendations, theory validation, and confirmatory inference",
            "regression_defect": "the fitted proxy includes a target-translation-dependent node constant rather than fitting the between-child gain alone",
            "classification_defect": "the implemented inverse-normal approximation reverses the quantile sign, so upper requested quantiles act as corresponding lower quantiles",
            "raw_archive_policy": "raw rows and descriptive interval outputs are retained solely for auditability; correcting either defect would define a new implementation and require a new S_par benchmark",
        },
        "timing_serial_sensitivity": {
            "run_index_diagnostics": "dataset-method Spearman trend, linear slope, and first-versus-second-half median contrast",
            "block_lengths_runs": list(TIMING_BLOCK_LENGTHS),
            "bootstrap": "paired circular moving-block bootstrap over ordered run positions for dataset medians and equal-weight global centroids",
            "interpretation": "assesses sensitivity to serial run-order dependence but cannot remove bias from a fixed method evaluation order within each run",
        },
        "joint_reliability": {
            "time_saved_thresholds_pct": list(TIME_SAVED_THRESHOLDS_PCT),
            "loss_tolerances_pct": list(LOSS_TOLERANCES_PCT),
            "experimental_unit": "dataset",
            "definition": "fraction of dataset-level run-median points meeting both constraints",
        },
        "dataset_family_sensitivity": {
            "regression_friedman_entries": "62 related fri_c entries are collapsed into five generator families c0-c4 by within-family median",
            "regression_aliases": "197_cpu_act/573_cpu_act and 227_cpu_small/562_cpu_small are exact duplicate pairs and are each collapsed to one family",
            "other_entries": "all remaining PMLB entries are singleton families",
            "estimand": "equal family weight after within-family median aggregation",
        },
        "deferred": [
            "S_par is excluded from confirmatory inference and substantive claims because its archived regression and classification calibrations contain implementation defects.",
            "Simultaneous confidence bands across every screened variant are not reported; all-variant intervals are marginal and exploratory.",
            "Uncertainty from drawing new CV partitions is not estimable from the archived fold-averaged files and would require a new nested-resampling benchmark.",
            "The primary joint claim is restricted to the revision-designated task-specific 1.0 margin; 0.5 and 2.5 form one multiplicity-adjusted sensitivity family.",
        ],
        "limitations": [
            "PMLB is a curated convenience benchmark, not a random probability sample; between-dataset intervals quantify empirical benchmark heterogeneity and only conditionally support broader generalization.",
            "CV folds were deterministic and fixed across runs. Run-level intervals describe estimator-seed and timing variability, not uncertainty from drawing new train/test partitions.",
            "Each archived run contains fold-averaged outcomes, so fold-level dependence cannot be reconstructed from these files.",
            "Percentile intervals are marginal, not simultaneous confidence bands across all methods.",
            "Wall-clock observations were collected on one hardware/software configuration and do not directly generalize to other systems.",
            "Circular run-block resampling can reveal sensitivity to serial run-order dependence but cannot remove fixed method-order bias within benchmark runs.",
            "The original all-method complete-case corpus included S_par; its influence on entry retention cannot be reconstructed from the archived successful-run files.",
            "The exact sharded benchmark invocation was not retained. Because the executable default imposes n*p <= 1,000,000, the archive cannot verify that this cap was disabled.",
        ],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy_version,
        },
        "inference_invocation": [sys.executable, *sys.argv],
        "audited_source_git": source_git_provenance(),
        "benchmark_provenance_limit": (
            "benchmark_metadata.json records runtime package versions, platform, "
            "seed range, fixed-fold protocol, and the loaded treeple path. The "
            "temporary loaded extension path and exact sharded shell command were "
            "not retained; source hashes identify the independently audited "
            "revision sources rather than proving bitwise identity with that binary."
        ),
        "benchmark_metadata": benchmark_metadata,
        "archive_inventory": inventories,
        "sha256_file_manifest": file_manifest.to_dict(orient="records"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indir",
        type=Path,
        default=script_dir / "benchmark_results",
        help="Directory containing the 100 per-run benchmark CSVs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Fresh output directory (preferably under /tmp).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=list(TASKS),
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.outdir}")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be strictly between 0 and 1")
    args.outdir.mkdir(parents=True, exist_ok=True)
    file_manifest = build_file_manifest(args.indir, args.tasks)

    root_seed = np.random.SeedSequence(args.seed)
    task_seeds = root_seed.spawn(len(args.tasks))
    collected = {
        "dataset_median_intervals": [],
        "global_intervals": [],
        "paired_vs_exhaustive": [],
        "omnibus_tests": [],
        "joint_reliability_intervals": [],
        "loss_margin_sign_tests": [],
        "runtime_superiority_sign_tests": [],
        "joint_time_loss_margin_tests": [],
        "timing_run_index_diagnostics": [],
        "timing_block_dataset_intervals": [],
        "timing_block_centroid_intervals": [],
        "archive_diagnostics": [],
        "dataset_family_map": [],
        "family_balanced_global_intervals": [],
        "family_balanced_joint_tests": [],
    }
    inventories = []
    for task, task_seed in zip(args.tasks, task_seeds):
        print(f"[{task}] loading archive and running {args.n_bootstrap} bootstraps", flush=True)
        (
            dataset_intervals,
            global_intervals,
            pairwise,
            omnibus,
            joint_reliability,
            loss_margin_sign,
            runtime_superiority,
            joint_time_loss_margin,
            timing_run_index,
            timing_block_dataset,
            timing_block_centroid,
            diagnostics,
            family_map,
            family_global,
            family_joint,
            inventory,
        ) = analyze_task(
            indir=args.indir,
            task=task,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed_sequence=task_seed,
        )
        collected["dataset_median_intervals"].append(dataset_intervals)
        collected["global_intervals"].append(global_intervals)
        collected["paired_vs_exhaustive"].append(pairwise)
        collected["omnibus_tests"].append(omnibus)
        collected["joint_reliability_intervals"].append(joint_reliability)
        collected["loss_margin_sign_tests"].append(loss_margin_sign)
        collected["runtime_superiority_sign_tests"].append(runtime_superiority)
        collected["joint_time_loss_margin_tests"].append(joint_time_loss_margin)
        collected["timing_run_index_diagnostics"].append(timing_run_index)
        collected["timing_block_dataset_intervals"].append(timing_block_dataset)
        collected["timing_block_centroid_intervals"].append(timing_block_centroid)
        collected["archive_diagnostics"].append(diagnostics)
        collected["dataset_family_map"].append(family_map)
        collected["family_balanced_global_intervals"].append(family_global)
        collected["family_balanced_joint_tests"].append(family_joint)
        inventories.append(inventory)
        print(
            f"[{task}] {inventory['n_datasets']} datasets, "
            f"{inventory['n_methods']} methods, paired runs "
            f"{inventory['min_complete_paired_runs']}-"
            f"{inventory['max_complete_paired_runs']}",
            flush=True,
        )

    for name, frames in collected.items():
        output = pd.concat(frames, ignore_index=True)
        output.to_csv(args.outdir / f"{name}.csv", index=False, float_format="%.17g")
    file_manifest.to_csv(args.outdir / "input_file_manifest.csv", index=False)

    metadata = _metadata(
        indir=args.indir,
        tasks=args.tasks,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        inventories=inventories,
        file_manifest=file_manifest,
    )
    (args.outdir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote inference bundle to {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
