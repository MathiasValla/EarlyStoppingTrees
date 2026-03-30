#!/usr/bin/env python
"""Focused prophet_1sample vs ExtraTree(max_features=all) benchmark."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pmlb import classification_dataset_names, fetch_data, regression_dataset_names
from sklearn.metrics import f1_score, make_scorer, mean_squared_error

from examples.early_stop_trees import benchmark_secretary_pmlb as bench
from treeple.tree import (
    EarlyStopDecisionTreeClassifier,
    EarlyStopDecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


TASKS = ("regression", "classification_gini", "classification_entropy")


def _task_dataset_names(task):
    if task == "regression":
        source = regression_dataset_names
    else:
        source = classification_dataset_names
    return [
        name
        for name in source
        if not name.startswith("_deprecated_") and name not in set(bench.SKIP_DATASETS)
    ]


def _criterion_for_task(task):
    if task == "classification_gini":
        return "gini"
    if task == "classification_entropy":
        return "entropy"
    return None


def _make_estimators(task, *, random_state, max_depth):
    criterion = _criterion_for_task(task)
    if task == "regression":
        return (
            (
                "prophet_1sample",
                "",
                EarlyStopDecisionTreeRegressor(
                    splitter="prophet_1sample",
                    random_state=random_state,
                    max_depth=max_depth,
                ),
            ),
            (
                "extra_tree",
                "max_features=all",
                ExtraTreeRegressor(
                    random_state=random_state,
                    max_depth=max_depth,
                    max_features=None,
                ),
            ),
        )
    return (
        (
            "prophet_1sample",
            "",
            EarlyStopDecisionTreeClassifier(
                splitter="prophet_1sample",
                criterion=criterion,
                random_state=random_state,
                max_depth=max_depth,
            ),
        ),
        (
            "extra_tree",
            "max_features=all",
            ExtraTreeClassifier(
                criterion=criterion,
                random_state=random_state,
                max_depth=max_depth,
                max_features=None,
            ),
        ),
    )


def _scoring(task):
    if task == "regression":
        rmse_scorer = make_scorer(
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=True,
        )
        return {"neg_rmse": rmse_scorer}
    return {
        "accuracy": "accuracy",
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }


def _load_dataset(name, *, task, cache_dir, max_product):
    X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
    X = np.asarray(X, dtype=np.float64)
    if task == "regression":
        y = np.asarray(y, dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.intp)

    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("contains NaN")

    n_samples, n_features = X.shape
    if max_product is not None and n_samples * n_features > max_product:
        raise ValueError(f"n*p={n_samples * n_features} > {max_product}")
    return X, y


def _root_improvement(estimator):
    tree = estimator.tree_
    left = tree.children_left[0]
    right = tree.children_right[0]
    if left < 0 or right < 0:
        return 0.0
    weighted_n = tree.weighted_n_node_samples
    impurity = tree.impurity
    return float(
        weighted_n[0] * impurity[0]
        - weighted_n[left] * impurity[left]
        - weighted_n[right] * impurity[right]
    )


def _write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(all_runs):
    grouped = defaultdict(list)
    for run_rows in all_runs:
        for row in run_rows:
            key = (
                row["dataset"],
                row["splitter"],
                row.get("variant", ""),
                row.get("criterion", ""),
            )
            grouped[key].append(row)

    out = []
    for (dataset, splitter, variant, criterion), rows in sorted(grouped.items()):
        agg = {
            "dataset": dataset,
            "splitter": splitter,
            "variant": variant,
            "n_runs": len(rows),
            "n_samples": int(rows[0]["n_samples"]),
            "n_features": int(rows[0]["n_features"]),
        }
        if criterion:
            agg["criterion"] = criterion
        numeric_keys = [
            key
            for key, value in rows[0].items()
            if isinstance(value, (int, float, np.floating))
            and key not in {"run", "n_samples", "n_features"}
        ]
        for key in numeric_keys:
            values = [float(row[key]) for row in rows]
            agg[key] = float(np.mean(values))
            agg[f"{key}_run_std"] = float(np.std(values))
        out.append(agg)
    return out


def _read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def run_task(
    task,
    *,
    outdir,
    cache_dir,
    n_runs,
    random_state,
    max_depth,
    max_product,
):
    datasets = _task_dataset_names(task)
    scoring = _scoring(task)
    all_runs = []
    stump_rows = []

    for run_idx in range(n_runs):
        seed = random_state + run_idx
        run_rows = []
        succeeded = 0
        print(f"[{task}] run {run_idx + 1}/{n_runs} start (seed={seed})", flush=True)
        for dataset_idx, name in enumerate(datasets, start=1):
            try:
                X, y = _load_dataset(
                    name,
                    task=task,
                    cache_dir=cache_dir,
                    max_product=max_product,
                )
            except Exception as exc:
                print(f"[{task}] skip {name}: {exc}", flush=True)
                continue

            if dataset_idx == 1 or dataset_idx % 10 == 0:
                print(
                    f"[{task}] run {run_idx + 1}/{n_runs} dataset "
                    f"{dataset_idx}/{len(datasets)} {name} (n={X.shape[0]}, p={X.shape[1]})",
                    flush=True,
                )

            row_prefix = {
                "dataset": name,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "run": run_idx + 1,
            }
            criterion = _criterion_for_task(task)
            if criterion is not None:
                row_prefix["criterion"] = criterion

            for splitter, variant, estimator in _make_estimators(
                task, random_state=seed, max_depth=max_depth
            ):
                cv, effort = bench._cross_validate_with_effort(
                    estimator,
                    X,
                    y,
                    cv=bench.N_FOLDS,
                    scoring=scoring,
                    return_train_score=False,
                )
                row = dict(row_prefix)
                row["splitter"] = splitter
                row["variant"] = variant
                row["fit_time_mean"] = float(np.mean(cv["fit_time"]))
                if task == "regression":
                    row["rmse_mean"] = -float(np.mean(cv["test_neg_rmse"]))
                    row["rmse_std"] = float(np.std(cv["test_neg_rmse"]))
                else:
                    row["accuracy_mean"] = float(np.mean(cv["test_accuracy"]))
                    row["accuracy_std"] = float(np.std(cv["test_accuracy"]))
                    row["f1_weighted_mean"] = float(np.mean(cv["test_f1_weighted"]))
                    row["f1_weighted_std"] = float(np.std(cv["test_f1_weighted"]))
                row.update(effort)
                run_rows.append(row)

            prophet_stump = _make_estimators(task, random_state=seed, max_depth=1)[0][2]
            extra_stump = _make_estimators(task, random_state=seed, max_depth=1)[1][2]
            prophet_stump.fit(X, y)
            extra_stump.fit(X, y)
            prophet_imp = _root_improvement(prophet_stump)
            extra_imp = _root_improvement(extra_stump)

            stump_row = dict(row_prefix)
            stump_row.update(
                {
                    "prophet_root_improvement": prophet_imp,
                    "extra_root_improvement": extra_imp,
                    "prophet_minus_extra": prophet_imp - extra_imp,
                    "prophet_ge_extra": int(prophet_imp + 1e-12 >= extra_imp),
                }
            )
            stump_rows.append(stump_row)
            succeeded += 1

        all_runs.append(run_rows)
        _write_rows(outdir / f"{task}_run_{run_idx + 1:02d}.csv", run_rows)
        print(f"[{task}] run {run_idx + 1}/{n_runs} done with {succeeded} datasets", flush=True)

    agg_rows = _aggregate_rows(all_runs)
    _write_rows(outdir / f"{task}_aggregated.csv", agg_rows)
    _write_rows(outdir / f"{task}_stump_checks.csv", stump_rows)
    violations = sum(1 for row in stump_rows if not row["prophet_ge_extra"])
    print(
        f"[{task}] stump dominance violations: {violations}/{len(stump_rows)}",
        flush=True,
    )


def _pair_rows_by_dataset(rows):
    out = {}
    for row in rows:
        out[(row["dataset"], row["splitter"], row.get("variant", ""))] = row
    return out


def _safe_float(row, key):
    return float(row[key])


def summarize_task(task, *, outdir):
    agg_rows = _read_rows(outdir / f"{task}_aggregated.csv")
    stump_rows = _read_rows(outdir / f"{task}_stump_checks.csv")
    if not agg_rows:
        return {
            "task": task,
            "status": "missing",
        }

    pairs = _pair_rows_by_dataset(agg_rows)
    prophet_rows = {
        dataset: row
        for (dataset, splitter, variant), row in pairs.items()
        if splitter == "prophet_1sample"
    }
    extra_rows = {
        dataset: row
        for (dataset, splitter, variant), row in pairs.items()
        if splitter == "extra_tree" and variant == "max_features=all"
    }
    common = sorted(set(prophet_rows) & set(extra_rows))

    summary = {
        "task": task,
        "status": "ok",
        "n_common_datasets": len(common),
        "stump_checks": len(stump_rows),
        "stump_violations": sum(int(row["prophet_ge_extra"]) == 0 for row in stump_rows),
    }

    if not common:
        return summary

    if task == "regression":
        rmse_diffs = [
            _safe_float(prophet_rows[name], "rmse_mean")
            - _safe_float(extra_rows[name], "rmse_mean")
            for name in common
        ]
        fit_time_ratios = [
            _safe_float(prophet_rows[name], "fit_time_mean")
            / _safe_float(extra_rows[name], "fit_time_mean")
            for name in common
            if _safe_float(extra_rows[name], "fit_time_mean") > 0.0
        ]
        summary.update(
            {
                "prophet_better_rmse_frac": float(np.mean([diff < 0.0 for diff in rmse_diffs])),
                "prophet_worse_rmse_frac": float(np.mean([diff > 0.0 for diff in rmse_diffs])),
                "mean_rmse_diff": float(np.mean(rmse_diffs)),
                "median_rmse_diff": float(np.median(rmse_diffs)),
                "prophet_slower_frac": float(
                    np.mean(
                        [
                            _safe_float(prophet_rows[name], "fit_time_mean")
                            > _safe_float(extra_rows[name], "fit_time_mean")
                            for name in common
                        ]
                    )
                ),
                "mean_fit_time_ratio": float(np.mean(fit_time_ratios)),
                "median_fit_time_ratio": float(np.median(fit_time_ratios)),
            }
        )
    else:
        acc_diffs = [
            _safe_float(prophet_rows[name], "accuracy_mean")
            - _safe_float(extra_rows[name], "accuracy_mean")
            for name in common
        ]
        f1_diffs = [
            _safe_float(prophet_rows[name], "f1_weighted_mean")
            - _safe_float(extra_rows[name], "f1_weighted_mean")
            for name in common
        ]
        fit_time_ratios = [
            _safe_float(prophet_rows[name], "fit_time_mean")
            / _safe_float(extra_rows[name], "fit_time_mean")
            for name in common
            if _safe_float(extra_rows[name], "fit_time_mean") > 0.0
        ]
        summary.update(
            {
                "prophet_better_accuracy_frac": float(np.mean([diff > 0.0 for diff in acc_diffs])),
                "prophet_worse_accuracy_frac": float(np.mean([diff < 0.0 for diff in acc_diffs])),
                "mean_accuracy_diff": float(np.mean(acc_diffs)),
                "median_accuracy_diff": float(np.median(acc_diffs)),
                "prophet_better_f1_frac": float(np.mean([diff > 0.0 for diff in f1_diffs])),
                "prophet_worse_f1_frac": float(np.mean([diff < 0.0 for diff in f1_diffs])),
                "mean_f1_diff": float(np.mean(f1_diffs)),
                "median_f1_diff": float(np.median(f1_diffs)),
                "prophet_slower_frac": float(
                    np.mean(
                        [
                            _safe_float(prophet_rows[name], "fit_time_mean")
                            > _safe_float(extra_rows[name], "fit_time_mean")
                            for name in common
                        ]
                    )
                ),
                "mean_fit_time_ratio": float(np.mean(fit_time_ratios)),
                "median_fit_time_ratio": float(np.median(fit_time_ratios)),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "summary", *TASKS),
        default="all",
    )
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-product", type=int, default=1_000_000)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results_prophet_vs_extra_all",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results" / "pmlb_cache",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if args.task == "all" else (() if args.task == "summary" else (args.task,))

    for task in tasks:
        run_task(
            task,
            outdir=args.outdir,
            cache_dir=args.cache_dir,
            n_runs=args.n_runs,
            random_state=args.random_state,
            max_depth=args.max_depth,
            max_product=args.max_product,
        )

    summaries = [summarize_task(task, outdir=args.outdir) for task in TASKS]
    summary_path = args.outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
