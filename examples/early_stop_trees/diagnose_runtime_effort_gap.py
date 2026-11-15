"""Measure why gain-evaluation savings need not become equal time savings.

The diagnostic is deliberately small.  It samples datasets across the ordered
selection lists, fits exhaustive CART and three early-stop splitters on shared
folds, and records both fit time and native splitter operation counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from pmlb import fetch_data
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

import treeple
from benchmark_secretary_pmlb import _collect_effort_summary
from treeple.tree import EarlyStopDecisionTreeClassifier, EarlyStopDecisionTreeRegressor


METHODS = (
    "best",
    "instrumented_best",
    "secretary",
    "secretary_all",
    "prophet_1sample",
)


def _sample_names(path: Path, count: int) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if count >= len(names):
        return names
    indices = np.linspace(0, len(names) - 1, count, dtype=int)
    return [names[index] for index in indices]


def _rotated_methods(dataset: str, seed: int) -> list[str]:
    digest = hashlib.sha256(f"{dataset}|{seed}".encode()).digest()
    offset = int.from_bytes(digest[:8], "little") % len(METHODS)
    return list(METHODS[offset:] + METHODS[:offset])


def _estimator(task: str, splitter: str, criterion: str, seed: int):
    implementation = "mab_all" if splitter == "instrumented_best" else splitter
    common = dict(splitter=implementation, max_depth=20, random_state=seed)
    if task == "regression":
        return EarlyStopDecisionTreeRegressor(criterion="squared_error", **common)
    return EarlyStopDecisionTreeClassifier(criterion=criterion, **common)


def _evaluate_dataset(task, dataset, X, y, criterion, seed):
    if task == "regression":
        cv = KFold(5, shuffle=False)
    else:
        cv = StratifiedKFold(5, shuffle=False)
    rows = []
    for position, splitter in enumerate(_rotated_methods(dataset, seed)):
        estimator = _estimator(task, splitter, criterion, seed)
        result = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=1,
            return_estimator=True,
        )
        effort = _collect_effort_summary(
            result["estimator"], estimator=estimator, X=X, y=y, cv=cv
        )
        rows.append(
            {
                "task": task if task == "regression" else f"classification_{criterion}",
                "dataset": dataset,
                "seed": seed,
                "splitter": splitter,
                "evaluation_position": position,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "fit_time_mean": float(np.mean(result["fit_time"])),
                **effort,
            }
        )
    return rows


def _paired_summary(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["task", "dataset", "seed"]
    metrics = [
        "fit_time_mean",
        "gain_evaluations_mean",
        "feature_sorts_mean",
        "random_partitions_mean",
        "final_partitions_mean",
        "minmax_passes_mean",
    ]
    time_baseline = raw[raw["splitter"] == "best"][keys + ["fit_time_mean"]].rename(
        columns={"fit_time_mean": "baseline_fit_time_mean"}
    )
    operation_metrics = metrics[1:]
    operation_baseline = raw[raw["splitter"] == "instrumented_best"][
        keys + operation_metrics
    ].rename(columns={metric: f"baseline_{metric}" for metric in operation_metrics})
    paired = raw.merge(time_baseline, on=keys, how="left", validate="many_to_one")
    paired = paired.merge(
        operation_baseline, on=keys, how="left", validate="many_to_one"
    )
    paired["time_saved_pct"] = 100 * (
        1 - paired["fit_time_mean"] / paired["baseline_fit_time_mean"]
    )
    paired["gain_saved_pct"] = 100 * (
        1
        - paired["gain_evaluations_mean"]
        / paired["baseline_gain_evaluations_mean"]
    )
    paired["feature_sort_saved_pct"] = 100 * (
        1 - paired["feature_sorts_mean"] / paired["baseline_feature_sorts_mean"]
    )
    paired["extra_linear_passes"] = (
        paired["random_partitions_mean"] + paired["minmax_passes_mean"]
    )
    paired["extra_linear_passes_per_100_baseline_sorts"] = 100 * (
        paired["extra_linear_passes"] / paired["baseline_feature_sorts_mean"]
    )
    value_columns = [
        "time_saved_pct",
        "gain_saved_pct",
        "feature_sort_saved_pct",
        "extra_linear_passes",
        "extra_linear_passes_per_100_baseline_sorts",
    ]
    dataset_medians = (
        paired[paired["splitter"] != "instrumented_best"]
        .groupby(["task", "dataset", "splitter"], as_index=False)[value_columns]
        .median()
    )
    summary = (
        dataset_medians.groupby(["task", "splitter"], as_index=False)[value_columns]
        .agg(["mean", "median"])
    )
    summary.columns = [
        "task",
        "splitter",
        *[f"{metric}_{stat}" for metric in value_columns for stat in ("mean", "median")],
    ]
    return paired, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-datasets", type=Path, required=True)
    parser.add_argument("--regression-datasets", type=Path, required=True)
    parser.add_argument("--pmlb-cache-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--datasets-per-task", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    selections = {
        "regression": _sample_names(args.regression_datasets, args.datasets_per_task),
        "classification": _sample_names(
            args.classification_datasets, args.datasets_per_task
        ),
    }
    for task, names in selections.items():
        for dataset in names:
            X, y = fetch_data(
                dataset,
                return_X_y=True,
                local_cache_dir=str(args.pmlb_cache_dir),
            )
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)
            for offset in range(args.n_seeds):
                seed = args.seed_start + offset
                print(f"[{task}] {dataset}, seed {seed}", flush=True)
                rows.extend(
                    _evaluate_dataset(task, dataset, X, y, "gini", seed)
                )

    raw = pd.DataFrame(rows)
    paired, summary = _paired_summary(raw)
    raw.to_csv(args.outdir / "runtime_effort_diagnostic_raw.csv", index=False)
    paired.to_csv(args.outdir / "runtime_effort_diagnostic_paired.csv", index=False)
    summary.to_csv(args.outdir / "runtime_effort_diagnostic_summary.csv", index=False)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "invocation": list(sys.argv),
        "dataset_selection": selections,
        "folds": 5,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "treeple": treeple.__version__,
        "operation_note": (
            "feature_sorts counts sorted node-feature scans; random_partitions and "
            "minmax_passes are full-node linear passes not represented by gain counts"
        ),
    }
    (args.outdir / "runtime_effort_diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
