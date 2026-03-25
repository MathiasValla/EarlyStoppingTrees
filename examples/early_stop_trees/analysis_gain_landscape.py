#!/usr/bin/env python
"""
Gain-landscape diagnostics under exhaustive split search.

This script fits exhaustive baseline trees on the benchmark datasets and
summarizes, at the internal-node level, how broad the near-optimal threshold
landscape is. The intended use is to support the regression/classification
asymmetry discussion in the manuscript.

Outputs:
- analysis_gain_landscape/gain_landscape_dataset_summary.csv
- analysis_gain_landscape/gain_landscape_task_summary.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "analysis_gain_landscape"
DELTA_LEVELS = (0.05, 0.10)
EPS = 1e-12


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    criterion: str | None


TASKS = {
    "regression": TaskSpec("regression", "regression", None),
    "classification_gini": TaskSpec("classification_gini", "classification", "gini"),
    "classification_entropy": TaskSpec("classification_entropy", "classification", "entropy"),
}


def _result_dataset_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    df = pd.read_csv(path, usecols=["dataset"])
    return sorted(df["dataset"].dropna().astype(str).unique().tolist())


def _run_level_dataset_names(indir: Path, prefix: str) -> list[str]:
    dataset_names: set[str] = set()
    for path in sorted(indir.glob(f"{prefix}_run*.csv")):
        df = pd.read_csv(path, usecols=["dataset"])
        dataset_names.update(df["dataset"].dropna().astype(str).tolist())
    return sorted(dataset_names)


def _task_dataset_names(indir: Path, task: str) -> list[str]:
    if task == "regression":
        run_level = _run_level_dataset_names(indir, "regression")
        if run_level:
            return run_level
        return _result_dataset_names(indir / "regression_results.csv")

    gini = set(_run_level_dataset_names(indir, "classification_gini"))
    entropy = set(_run_level_dataset_names(indir, "classification_entropy"))
    if not gini:
        gini = set(_result_dataset_names(indir / "classification_gini_results.csv"))
    if not entropy:
        entropy = set(_result_dataset_names(indir / "classification_entropy_results.csv"))
    if gini and entropy:
        return sorted(gini & entropy)
    if task == "classification_gini":
        return sorted(gini)
    return sorted(entropy)


def _class_impurity(counts: np.ndarray, criterion: str) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts.astype(float) / total
    probs = probs[probs > 0]
    if criterion == "gini":
        return 1.0 - float(np.sum(probs**2))
    if criterion == "entropy":
        return -float(np.sum(probs * np.log2(probs)))
    raise ValueError(f"Unsupported criterion: {criterion}")


def _feature_gains_regression(x: np.ndarray, y: np.ndarray, min_samples_leaf: int) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order].astype(float)
    n = y_sorted.shape[0]
    if n < 2 * min_samples_leaf:
        return np.empty(0, dtype=float)

    valid_pos = np.flatnonzero(np.diff(x_sorted) > 0) + 1
    valid_pos = valid_pos[(valid_pos >= min_samples_leaf) & ((n - valid_pos) >= min_samples_leaf)]
    if valid_pos.size == 0:
        return np.empty(0, dtype=float)

    prefix_sum = np.cumsum(y_sorted)
    prefix_sq = np.cumsum(y_sorted * y_sorted)
    total_sum = float(prefix_sum[-1])
    total_sq = float(prefix_sq[-1])
    parent_impurity = total_sq / n - (total_sum / n) ** 2

    gains = np.empty(valid_pos.size, dtype=float)
    for idx, pos in enumerate(valid_pos):
        left_n = pos
        right_n = n - pos
        left_sum = float(prefix_sum[pos - 1])
        left_sq = float(prefix_sq[pos - 1])
        right_sum = total_sum - left_sum
        right_sq = total_sq - left_sq
        left_impurity = left_sq / left_n - (left_sum / left_n) ** 2
        right_impurity = right_sq / right_n - (right_sum / right_n) ** 2
        gains[idx] = parent_impurity - (left_n / n) * left_impurity - (right_n / n) * right_impurity
    return gains


def _feature_gains_classification(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    criterion: str,
    min_samples_leaf: int,
) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order].astype(int)
    n = y_sorted.shape[0]
    if n < 2 * min_samples_leaf:
        return np.empty(0, dtype=float)

    valid_pos = np.flatnonzero(np.diff(x_sorted) > 0) + 1
    valid_pos = valid_pos[(valid_pos >= min_samples_leaf) & ((n - valid_pos) >= min_samples_leaf)]
    if valid_pos.size == 0:
        return np.empty(0, dtype=float)

    indicator = np.zeros((n, n_classes), dtype=np.int64)
    indicator[np.arange(n), y_sorted] = 1
    prefix_counts = np.cumsum(indicator, axis=0)
    total_counts = prefix_counts[-1]
    parent_impurity = _class_impurity(total_counts, criterion)

    gains = np.empty(valid_pos.size, dtype=float)
    for idx, pos in enumerate(valid_pos):
        left_counts = prefix_counts[pos - 1]
        right_counts = total_counts - left_counts
        left_n = pos
        right_n = n - pos
        left_impurity = _class_impurity(left_counts, criterion)
        right_impurity = _class_impurity(right_counts, criterion)
        gains[idx] = parent_impurity - (left_n / n) * left_impurity - (right_n / n) * right_impurity
    return gains


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _node_gain_metrics(
    X_node: np.ndarray,
    y_node: np.ndarray,
    spec: TaskSpec,
    *,
    min_samples_leaf: int,
    n_classes: int | None,
) -> dict[str, float] | None:
    all_gains = []
    feature_gains = []

    for feature_idx in range(X_node.shape[1]):
        x = X_node[:, feature_idx]
        if spec.kind == "regression":
            gains = _feature_gains_regression(x, y_node, min_samples_leaf=min_samples_leaf)
        else:
            gains = _feature_gains_classification(
                x,
                y_node,
                n_classes=n_classes or 0,
                criterion=spec.criterion or "gini",
                min_samples_leaf=min_samples_leaf,
            )
        if gains.size == 0:
            continue
        all_gains.append(gains)
        feature_gains.append((feature_idx, gains))

    if not all_gains:
        return None

    gain_values = np.concatenate(all_gains)
    winner_feature_idx, winner_gains = max(feature_gains, key=lambda item: float(np.max(item[1])))
    best_gain = float(np.max(winner_gains))
    denom = max(best_gain, EPS)
    metrics = {
        "n_threshold_candidates": float(gain_values.size),
        "best_gain": best_gain,
        "relative_median_gap": float((best_gain - np.median(gain_values)) / denom),
        "winner_feature": float(winner_feature_idx),
    }

    for delta in DELTA_LEVELS:
        threshold = (1.0 - delta) * best_gain
        suffix = str(int(round(100 * delta))).zfill(2)
        near_optimal_mask = gain_values >= threshold
        winner_mask = winner_gains >= threshold
        metrics[f"near_optimal_prop_{suffix}"] = float(np.mean(near_optimal_mask))
        metrics[f"winner_width_{suffix}"] = float(
            _longest_true_run(winner_mask) / max(winner_gains.size, 1)
        )
    return metrics


def _iter_internal_node_indices(estimator, X: np.ndarray):
    tree = estimator.tree_
    stack = [(0, np.arange(X.shape[0], dtype=np.intp))]
    while stack:
        node_id, sample_idx = stack.pop()
        feature = tree.feature[node_id]
        if feature < 0:
            continue
        yield node_id, sample_idx
        threshold = tree.threshold[node_id]
        left_mask = X[sample_idx, feature] <= threshold
        right_mask = ~left_mask
        if np.any(right_mask):
            stack.append((tree.children_right[node_id], sample_idx[right_mask]))
        if np.any(left_mask):
            stack.append((tree.children_left[node_id], sample_idx[left_mask]))


def _fit_exhaustive_tree(spec: TaskSpec, X: np.ndarray, y: np.ndarray):
    if spec.kind == "regression":
        return DecisionTreeRegressor(splitter="best", max_depth=20, random_state=0).fit(X, y)
    return DecisionTreeClassifier(
        criterion=spec.criterion or "gini",
        splitter="best",
        max_depth=20,
        random_state=0,
    ).fit(X, y)


def _analyze_dataset(
    dataset: str,
    spec: TaskSpec,
    *,
    cache_dir: Path,
    max_samples: int | None,
    random_state: int,
) -> dict[str, float] | None:
    try:
        X, y = fetch_data(dataset, return_X_y=True, local_cache_dir=str(cache_dir))
    except Exception:
        return None

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if np.any(np.isnan(X)) or np.any(pd.isna(y)):
        return None

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        keep = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[keep]
        y = y[keep]

    n_classes = None
    if spec.kind == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        n_classes = int(len(encoder.classes_))
    else:
        y = np.asarray(y, dtype=np.float64)

    estimator = _fit_exhaustive_tree(spec, X, y)
    node_rows = []
    min_samples_leaf = int(getattr(estimator, "min_samples_leaf", 1))
    for node_id, sample_idx in _iter_internal_node_indices(estimator, X):
        node_metrics = _node_gain_metrics(
            X[sample_idx],
            y[sample_idx],
            spec,
            min_samples_leaf=min_samples_leaf,
            n_classes=n_classes,
        )
        if node_metrics is None:
            continue
        node_metrics["node_id"] = float(node_id)
        node_rows.append(node_metrics)

    if not node_rows:
        return None

    node_df = pd.DataFrame(node_rows)
    summary = {
        "task": spec.name,
        "dataset": dataset,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_internal_nodes": int(node_df.shape[0]),
        "n_threshold_candidates_median": float(node_df["n_threshold_candidates"].median()),
        "relative_median_gap_median": float(node_df["relative_median_gap"].median()),
    }
    for delta in DELTA_LEVELS:
        suffix = str(int(round(100 * delta))).zfill(2)
        summary[f"near_optimal_prop_{suffix}_median"] = float(
            node_df[f"near_optimal_prop_{suffix}"].median()
        )
        summary[f"winner_width_{suffix}_median"] = float(node_df[f"winner_width_{suffix}"].median())
    return summary


def _task_summary(dataset_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "n_threshold_candidates_median",
        "relative_median_gap_median",
        "near_optimal_prop_05_median",
        "near_optimal_prop_10_median",
        "winner_width_05_median",
        "winner_width_10_median",
    ]
    for task, sub in dataset_df.groupby("task"):
        row = {"task": task, "n_datasets": int(sub["dataset"].nunique())}
        for col in metric_cols:
            values = sub[col].dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[f"{col}_median"] = np.nan
                row[f"{col}_iqr"] = np.nan
            else:
                row[f"{col}_median"] = float(np.median(values))
                row[f"{col}_iqr"] = float(np.subtract(*np.percentile(values, [75, 25])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("task").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Gain-landscape diagnostics for exhaustive trees")
    parser.add_argument(
        "--benchmark-indir",
        type=str,
        default=None,
        help="Benchmark results directory used to determine the dataset universe",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: examples/early_stop_trees/analysis_gain_landscape)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="regression,classification_gini,classification_entropy",
        help="Comma-separated tasks",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Restrict to one dataset")
    parser.add_argument("--max-datasets", type=int, default=None, help="Max datasets per task")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional subsample cap for development/smoke tests",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Seed used only for subsampling")
    args = parser.parse_args()

    indir = Path(args.benchmark_indir) if args.benchmark_indir else BENCHMARK_DIR
    outdir = Path(args.outdir) if args.outdir else OUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = indir / "pmlb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    dataset_rows = []
    for task in tasks:
        spec = TASKS[task]
        if args.dataset is not None:
            dataset_names = [args.dataset]
        else:
            dataset_names = _task_dataset_names(indir, task)
        if args.max_datasets is not None:
            dataset_names = dataset_names[: args.max_datasets]

        print(f"[{task}] {len(dataset_names)} dataset(s)")
        for index, dataset in enumerate(dataset_names, start=1):
            print(f"[{task}] {index}/{len(dataset_names)} {dataset}", flush=True)
            row = _analyze_dataset(
                dataset,
                spec,
                cache_dir=cache_dir,
                max_samples=args.max_samples,
                random_state=args.random_state,
            )
            if row is not None:
                dataset_rows.append(row)

    dataset_df = pd.DataFrame(dataset_rows)
    dataset_path = outdir / "gain_landscape_dataset_summary.csv"
    dataset_df.to_csv(dataset_path, index=False)
    print(f"Wrote {dataset_path}")

    task_df = _task_summary(dataset_df) if not dataset_df.empty else pd.DataFrame()
    task_path = outdir / "gain_landscape_task_summary.csv"
    task_df.to_csv(task_path, index=False)
    print(f"Wrote {task_path}")


if __name__ == "__main__":
    main()
