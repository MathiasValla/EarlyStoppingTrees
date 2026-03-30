#!/usr/bin/env python
"""
Benchmark secretary-style and ExtraTree splitters on PMLB datasets.

Regression: RMSE and fit time (5-fold cross-validation).
Classification: accuracy, F1-score (weighted), and fit time for Gini and entropy
(5-fold cross-validation).

Cross-validation uses ``cv=N_FOLDS`` with scikit-learn defaults, i.e. no shuffling,
so the fold partition is fixed across methods and repeated runs for a given dataset.
Secretary procedures and ExtraTrees depend on randomness; it is controlled by the
estimator's ``random_state`` hyperparameter (and by ``--random-state`` in this script).
When both classification criteria are benchmarked together, the final Gini and
entropy CSVs are restricted to the common set of datasets that succeeded under
both criteria, so downstream comparisons use matched dataset sets.

Usage:
  pip install pmlb
  python examples/early_stop_trees/benchmark_secretary_pmlb.py [--max-datasets N] [--outdir DIR]
  python examples/early_stop_trees/benchmark_secretary_pmlb.py --n-runs 10 --random-state 42  # N runs, then aggregate means
  python examples/early_stop_trees/benchmark_secretary_pmlb.py --isolate-datasets  # run each dataset in a subprocess; SIGSEGV skips that dataset automatically

Output: CSV files in outdir (default: examples/early_stop_trees/benchmark_results).
Each result row also reports effort metrics: direct splitter counters for the
early-stop families, exact post-fit counts for the exhaustive `best` splitter,
and derived post-fit counts for `extra_tree`. The script also writes a
benchmark_metadata.json file describing the timing protocol and hardware/software
environment.
"""
import argparse
import csv
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sklearn

try:
    from pmlb import (
        fetch_data,
        regression_dataset_names,
        classification_dataset_names,
    )
except ImportError:
    print("Install pmlb: pip install pmlb", file=sys.stderr)
    sys.exit(1)

from sklearn.base import is_classifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection._split import check_cv
from sklearn.metrics import mean_squared_error, f1_score, make_scorer

import treeple
from treeple.tree import (
    EarlyStopDecisionTreeClassifier,
    EarlyStopDecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


RANDOM_STATE = 42
N_FOLDS = 5
# Base splitter families; secretary_par and extra_tree are expanded into parameter variants.
SPLITTERS = (
    "best",
    "secretary",
    "secretary_par",
    "secretary_all",
    "double_secretary",
    "block_rank",
    "prophet_1sample",
    "extra_tree",
)
CRITERIA_CLF = ("gini", "entropy")
# Datasets to skip by default (e.g. known to segfault or raise in C extension)
SKIP_DATASETS = ("192_vineyard", "687_sleuth_ex1605")

# Exit code when child process is killed by SIGSEGV (128 + 11)
SIGSEGV_EXIT = 139
EFFORT_KEYS = (
    "split_calls",
    "threshold_candidates",
    "gain_evaluations",
    "threshold_candidates_per_split",
    "gain_evaluations_per_split",
    "parametric_gain_samples",
    "parametric_quantile_fits",
)
FEATURE_THRESHOLD = 1e-7


def _finalize_effort_stats(
    split_calls,
    threshold_candidates,
    gain_evaluations,
    *,
    parametric_gain_samples=0.0,
    parametric_quantile_fits=0.0,
):
    split_calls = float(split_calls)
    threshold_candidates = float(threshold_candidates)
    gain_evaluations = float(gain_evaluations)
    parametric_gain_samples = float(parametric_gain_samples)
    parametric_quantile_fits = float(parametric_quantile_fits)
    denom = split_calls if split_calls > 0 else 1.0
    return {
        "split_calls": split_calls,
        "threshold_candidates": threshold_candidates,
        "gain_evaluations": gain_evaluations,
        "threshold_candidates_per_split": threshold_candidates / denom,
        "gain_evaluations_per_split": gain_evaluations / denom,
        "parametric_gain_samples": parametric_gain_samples,
        "parametric_quantile_fits": parametric_quantile_fits,
    }


def _candidate_split_positions(values):
    values = np.sort(np.asarray(values, dtype=np.float32))
    if values.size <= 1:
        return np.empty(0, dtype=np.intp)
    return np.flatnonzero(values[1:] > values[:-1] + FEATURE_THRESHOLD) + 1


def _count_nonconstant_features(X_node):
    if X_node.size == 0:
        return 0
    mins = np.min(X_node, axis=0)
    maxs = np.max(X_node, axis=0)
    return int(np.sum(maxs > mins + FEATURE_THRESHOLD))


def _iter_internal_node_samples(tree, X_train):
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    stack = [(0, np.arange(X_train.shape[0], dtype=np.intp))]
    while stack:
        node_id, sample_idx = stack.pop()
        if sample_idx.size == 0 or features[node_id] < 0:
            continue
        yield node_id, sample_idx
        feat = features[node_id]
        thr = thresholds[node_id]
        node_values = X_train[sample_idx, feat]
        left_mask = node_values <= thr
        stack.append((children_right[node_id], sample_idx[~left_mask]))
        stack.append((children_left[node_id], sample_idx[left_mask]))


def _derive_best_effort_stats(estimator, X_train):
    tree = estimator.tree_
    min_samples_leaf = int(getattr(estimator, "min_samples_leaf_", 1))
    threshold_candidates = 0.0
    gain_evaluations = 0.0
    split_calls = 0.0
    max_features = int(getattr(estimator, "max_features_", X_train.shape[1]))

    for _, sample_idx in _iter_internal_node_samples(tree, X_train):
        split_calls += 1.0
        X_node = X_train[sample_idx]
        n_node_features = X_node.shape[1]
        feature_scale = 1.0
        if max_features < n_node_features:
            feature_scale = max_features / float(n_node_features)
        node_thresholds = 0.0
        node_gains = 0.0
        for feature_idx in range(n_node_features):
            positions = _candidate_split_positions(X_node[:, feature_idx])
            if positions.size == 0:
                continue
            node_thresholds += float(positions.size)
            valid = (positions >= min_samples_leaf) & ((sample_idx.size - positions) >= min_samples_leaf)
            node_gains += float(np.sum(valid))
        threshold_candidates += feature_scale * node_thresholds
        gain_evaluations += feature_scale * node_gains

    return _finalize_effort_stats(split_calls, threshold_candidates, gain_evaluations)


def _derive_extra_tree_effort_stats(estimator, X_train):
    tree = estimator.tree_
    split_calls = 0.0
    threshold_candidates = 0.0
    gain_evaluations = 0.0
    max_features = int(getattr(estimator, "max_features_", X_train.shape[1]))

    for _, sample_idx in _iter_internal_node_samples(tree, X_train):
        split_calls += 1.0
        X_node = X_train[sample_idx]
        n_nonconstant = _count_nonconstant_features(X_node)
        # One random threshold and one gain evaluation per sampled non-constant feature.
        evaluated = float(min(max_features, n_nonconstant))
        threshold_candidates += evaluated
        gain_evaluations += evaluated

    return _finalize_effort_stats(split_calls, threshold_candidates, gain_evaluations)


def _derive_effort_stats(estimator, X_train):
    splitter_name = getattr(estimator, "splitter", None)
    if splitter_name == "best":
        return _derive_best_effort_stats(estimator, X_train)
    if splitter_name == "random":
        return _derive_extra_tree_effort_stats(estimator, X_train)
    return None


def _collect_effort_summary(estimators, *, estimator, X, y, cv):
    estimators = estimators or []
    if not estimators:
        return {f"{key}_mean": float("nan") for key in EFFORT_KEYS}

    stats_by_fold = []
    cv_splits = None
    for fold_idx, fitted_estimator in enumerate(estimators):
        stats = getattr(fitted_estimator, "splitter_stats_", None)
        if not stats:
            if cv_splits is None:
                cv_splitter = check_cv(cv, y, classifier=is_classifier(estimator))
                cv_splits = list(cv_splitter.split(X, y))
            train_idx, _ = cv_splits[fold_idx]
            stats = _derive_effort_stats(fitted_estimator, X[train_idx])
        if stats:
            stats_by_fold.append(stats)

    summary = {}
    for key in EFFORT_KEYS:
        values = []
        for stats in stats_by_fold:
            value = stats.get(key)
            if value is None:
                continue
            values.append(float(value))
        summary[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
    return summary


def _benchmark_metadata(splitters, n_runs, random_state):
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "treeple_version": treeple.__version__,
        "n_folds": N_FOLDS,
        "splitters": list(splitters),
        "n_runs": int(n_runs),
        "random_state_start": int(random_state),
        "timing_protocol": {
            "fit_time_source": "sklearn.model_selection.cross_validate",
            "fit_time_scope": "Estimator fit only; dataset fetch, subsampling, CSV writing, and scoring happen outside the reported fit_time.",
            "cv_splitter": "Integer cv=N_FOLDS, so sklearn uses deterministic KFold / StratifiedKFold with shuffle=False.",
        },
        "effort_protocol": {
            "early_stop": "Measured directly from splitter counters.",
            "best": "Derived exactly a posteriori from fitted trees and fold-specific training data.",
            "extra_tree": "Derived a posteriori as one random-threshold gain per sampled non-constant feature at a node.",
        },
    }


def _write_benchmark_metadata(outdir, splitters, n_runs, random_state):
    path = Path(outdir) / "benchmark_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_benchmark_metadata(splitters, n_runs, random_state), f, indent=2, sort_keys=True)
    print(f"Wrote {path}")


def _cross_validate_with_effort(estimator, X, y, *, cv, scoring, return_train_score=False):
    cv_result = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=return_train_score,
        return_estimator=True,
    )
    effort = _collect_effort_summary(
        cv_result.get("estimator"),
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
    )
    return cv_result, effort


def _write_rows_csv(path, rows, fieldnames=None):
    """Write row dicts to CSV."""
    if not rows and not fieldnames:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def _dataset_names_from_rows(rows):
    """Unique dataset names present in a row list."""
    return {str(r["dataset"]) for r in rows if "dataset" in r}


def _enforce_common_classification_datasets(gini_rows, entropy_rows, *, label):
    """
    Restrict Gini and entropy rows to datasets present in both result sets.

    This guarantees matched dataset sets across criteria when a dataset fails for
    one impurity but not the other.
    """
    gini_names = _dataset_names_from_rows(gini_rows)
    entropy_names = _dataset_names_from_rows(entropy_rows)
    common_names = gini_names & entropy_names
    if gini_names == entropy_names:
        print(
            f"[classification] {label}: kept {len(common_names)} common datasets across "
            "gini and entropy",
            flush=True,
        )
        return gini_rows, entropy_rows, common_names

    dropped_gini_only = sorted(gini_names - common_names)
    dropped_entropy_only = sorted(entropy_names - common_names)
    if dropped_gini_only:
        print(
            f"[classification] {label}: dropping {len(dropped_gini_only)} dataset(s) "
            f"present only in gini: {', '.join(dropped_gini_only)}",
            file=sys.stderr,
        )
    if dropped_entropy_only:
        print(
            f"[classification] {label}: dropping {len(dropped_entropy_only)} dataset(s) "
            f"present only in entropy: {', '.join(dropped_entropy_only)}",
            file=sys.stderr,
        )
    print(
        f"[classification] {label}: using {len(common_names)} common datasets across "
        "gini and entropy",
        flush=True,
    )
    gini_rows = [r for r in gini_rows if r.get("dataset") in common_names]
    entropy_rows = [r for r in entropy_rows if r.get("dataset") in common_names]
    return gini_rows, entropy_rows, common_names


def _secretary_par_grid(n_samples: int):
    """Return list of (p_thr_par, n_gain_samples_par, sample_mode_label) for secretary_par.

    Samples modes:
      - 2 samples (cap)
      - 10 samples (cap)
      - 10% of thresholds (p_thr_par = 0.1)
      - sqrt(n) splits (p_thr_par = sqrt(n)/n, capped at 1)
      - ln(n) samples (n_gain_samples_par = min(256, max(1, round(ln(n)))))
    """
    out = []
    out.append((1.0, 2, "2"))
    out.append((1.0, 10, "10"))
    out.append((0.1, 256, "0.1n"))
    if n_samples > 0:
        n = float(n_samples)
        p_sqrt = min(1.0, np.sqrt(n) / n)
        out.append((p_sqrt, 256, "sqrt_n"))
        n_ln = min(256, max(1, int(round(np.log(n)))))
        out.append((1.0, n_ln, "ln_n"))
    return out


def _secretary_variants(n_samples: int):
    """Return list of (split_search_dict, variant_label) for base secretary: 1/e, sqrt(n), ln(n), 10%.

    split_search uses secretary_threshold: "1/e", "sqrt_n", or a float (explore fraction).
    """
    out = []
    out.append(({}, "1overe"))  # default 1/e
    out.append(({"secretary_threshold": "sqrt_n"}, "sqrt_n"))
    if n_samples > 1:
        explore_ln = 1.0 / max(1.0, math.log(n_samples))
        out.append(({"secretary_threshold": explore_ln}, "ln_n"))
    out.append(({"secretary_threshold": 0.1}, "0.1n"))
    return out


def _extra_tree_variants():
    """Return list of (max_features, variant_label) for ExtraTree baselines."""
    return (
        (1, "max_features=1"),
        (1.0 / 3.0, "max_features=1over3"),
        (2.0 / 3.0, "max_features=2over3"),
        (None, "max_features=all"),
    )


def _iter_regression_estimators(n_samples: int, random_state: int, splitters=None):
    """Yield (splitter, variant, estimator) for one regression dataset."""
    splitters = splitters if splitters is not None else SPLITTERS
    for splitter in splitters:
        if splitter == "secretary_par":
            for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
                for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                    yield (
                        "secretary_par",
                        f"samples={sample_mode},q={q_thr_par}",
                        EarlyStopDecisionTreeRegressor(
                            splitter="secretary_par",
                            random_state=random_state,
                            max_depth=20,
                            split_search={
                                "p_thr_par": p_thr_par,
                                "q_thr_par": q_thr_par,
                                "n_gain_samples_par": int(n_gain_samples_par),
                            },
                        ),
                    )
        elif splitter in ("secretary", "secretary_all", "double_secretary"):
            for split_search, variant_label in _secretary_variants(n_samples):
                yield (
                    splitter,
                    variant_label,
                    EarlyStopDecisionTreeRegressor(
                        splitter=splitter,
                        random_state=random_state,
                        max_depth=20,
                        split_search=split_search,
                    ),
                )
        elif splitter == "extra_tree":
            for max_features, variant_label in _extra_tree_variants():
                yield (
                    "extra_tree",
                    variant_label,
                    ExtraTreeRegressor(
                        random_state=random_state,
                        max_depth=20,
                        max_features=max_features,
                    ),
                )
        else:
            yield (
                splitter,
                "",
                EarlyStopDecisionTreeRegressor(
                    splitter=splitter,
                    random_state=random_state,
                    max_depth=20,
                ),
            )


def _iter_classification_estimators(n_samples: int, criterion: str, random_state: int, splitters=None):
    """Yield (splitter, variant, estimator) for one classification dataset."""
    splitters = splitters if splitters is not None else SPLITTERS
    for splitter in splitters:
        if splitter == "secretary_par":
            for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
                for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                    yield (
                        "secretary_par",
                        f"samples={sample_mode},q={q_thr_par}",
                        EarlyStopDecisionTreeClassifier(
                            splitter="secretary_par",
                            criterion=criterion,
                            random_state=random_state,
                            max_depth=20,
                            split_search={
                                "p_thr_par": p_thr_par,
                                "q_thr_par": q_thr_par,
                                "n_gain_samples_par": int(n_gain_samples_par),
                            },
                        ),
                    )
        elif splitter in ("secretary", "secretary_all", "double_secretary"):
            for split_search, variant_label in _secretary_variants(n_samples):
                yield (
                    splitter,
                    variant_label,
                    EarlyStopDecisionTreeClassifier(
                        splitter=splitter,
                        criterion=criterion,
                        random_state=random_state,
                        max_depth=20,
                        split_search=split_search,
                    ),
                )
        elif splitter == "extra_tree":
            for max_features, variant_label in _extra_tree_variants():
                yield (
                    "extra_tree",
                    variant_label,
                    ExtraTreeClassifier(
                        criterion=criterion,
                        random_state=random_state,
                        max_depth=20,
                        max_features=max_features,
                    ),
                )
        else:
            yield (
                splitter,
                "",
                EarlyStopDecisionTreeClassifier(
                    splitter=splitter,
                    criterion=criterion,
                    random_state=random_state,
                    max_depth=20,
                ),
            )


def _evaluate_regression_dataset(name, X, y, random_state, splitters=None):
    """Run the full regression estimator grid on one dataset and return CSV rows."""
    n_samples, n_features = X.shape
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )
    dataset_rows = []
    for splitter, variant, est in _iter_regression_estimators(n_samples, random_state, splitters=splitters):
        cv, effort = _cross_validate_with_effort(
            est,
            X,
            y,
            cv=N_FOLDS,
            scoring={"neg_rmse": rmse_scorer},
            return_train_score=False,
        )
        dataset_rows.append(
            {
                "dataset": name,
                "n_samples": n_samples,
                "n_features": n_features,
                "splitter": splitter,
                "variant": variant,
                "rmse_mean": -float(np.mean(cv["test_neg_rmse"])),
                "rmse_std": float(np.std(cv["test_neg_rmse"])),
                "fit_time_mean": float(np.mean(cv["fit_time"])),
                **effort,
            }
        )
    return dataset_rows


def _evaluate_classification_dataset(name, X, y, criterion, random_state, splitters=None):
    """Run the full classification estimator grid on one dataset and return CSV rows."""
    n_samples, n_features = X.shape
    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }
    dataset_rows = []
    for splitter, variant, est in _iter_classification_estimators(
        n_samples, criterion, random_state, splitters=splitters
    ):
        cv, effort = _cross_validate_with_effort(
            est,
            X,
            y,
            cv=N_FOLDS,
            scoring=scoring,
            return_train_score=False,
        )
        dataset_rows.append(
            {
                "dataset": name,
                "n_samples": n_samples,
                "n_features": n_features,
                "criterion": criterion,
                "splitter": splitter,
                "variant": variant,
                "accuracy_mean": float(np.mean(cv["test_accuracy"])),
                "accuracy_std": float(np.std(cv["test_accuracy"])),
                "f1_weighted_mean": float(np.mean(cv["test_f1_weighted"])),
                "f1_weighted_std": float(np.std(cv["test_f1_weighted"])),
                "fit_time_mean": float(np.mean(cv["fit_time"])),
                **effort,
            }
        )
    return dataset_rows


def _run_single_dataset_regression(
    name,
    outdir,
    random_state,
    pmlb_cache_dir=None,
    max_samples=None,
    max_rows=None,
    max_features=None,
    max_product=None,
    splitters=None,
):
    """Run regression for one dataset; return list of row dicts or [] if skipped (fetch/size)."""
    splitters = splitters if splitters is not None else SPLITTERS
    outdir = Path(outdir or ".")
    cache_dir = Path(pmlb_cache_dir) if pmlb_cache_dir is not None else (outdir / "pmlb_cache")
    try:
        X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
    except Exception:
        return []
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return []
    if max_rows is not None and len(y) > max_rows:
        return []
    if max_features is not None and X.shape[1] > max_features:
        return []
    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    n_samples, n_features = X.shape
    if max_product is not None and n_samples * n_features > max_product:
        return []
    return _evaluate_regression_dataset(name, X, y, random_state, splitters=splitters)


def _run_single_dataset_classification(
    name,
    criterion,
    outdir,
    random_state,
    pmlb_cache_dir=None,
    max_samples=None,
    max_rows=None,
    max_features=None,
    max_product=None,
    splitters=None,
):
    """Run classification for one dataset; return list of row dicts or [] if skipped."""
    splitters = splitters if splitters is not None else SPLITTERS
    outdir = Path(outdir or ".")
    cache_dir = Path(pmlb_cache_dir) if pmlb_cache_dir is not None else (outdir / "pmlb_cache")
    try:
        X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
    except Exception:
        return []
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.intp)
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return []
    if max_rows is not None and len(y) > max_rows:
        return []
    if max_features is not None and X.shape[1] > max_features:
        return []
    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    n_samples, n_features = X.shape
    if max_product is not None and n_samples * n_features > max_product:
        return []
    return _evaluate_classification_dataset(
        name,
        X,
        y,
        criterion,
        random_state,
        splitters=splitters,
    )


def run_regression(max_datasets=None, max_samples=None, max_rows=None, max_features=None, max_product=None, outdir=None, dataset=None, datasets=None, pmlb_cache_dir=None, exclude=None, random_state=None, isolate_datasets=False, per_run_path=None, splitters=None):
    """Run regression benchmark once. Returns (rows, path). random_state controls estimator and subsampling RNG.
    If isolate_datasets=True, each dataset runs in a subprocess; SIGSEGV (or any non-zero exit) skips that dataset automatically.
    If per_run_path is set (e.g. by run_benchmark_n_times), results are written there instead of regression_results.csv (so each run gets its own file)."""
    splitters = splitters if splitters is not None else SPLITTERS
    random_state = random_state if random_state is not None else RANDOM_STATE
    outdir = Path(outdir or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    path = Path(per_run_path) if per_run_path is not None else outdir / "regression_results.csv"
    exclude = set(exclude or [])

    if datasets is not None:
        datasets = [
            n
            for n in datasets
            if n in regression_dataset_names and not n.startswith("_deprecated_") and n not in exclude
        ]
        if max_datasets is not None:
            datasets = datasets[: max_datasets]
    elif dataset is not None:
        datasets = [dataset] if dataset in regression_dataset_names else []
    else:
        datasets = [n for n in regression_dataset_names if not n.startswith("_deprecated_") and n not in exclude]
        if max_datasets is not None:
            datasets = datasets[: max_datasets]

    rows = []
    if isolate_datasets:
        script = Path(__file__).resolve()
        for i, name in enumerate(datasets):
            cmd = [
                sys.executable,
                str(script),
                "--run-single-dataset", name,
                "--task", "regression",
                "--random-state", str(random_state),
                "--outdir", str(outdir),
            ]
            if pmlb_cache_dir is not None:
                cmd += ["--pmlb-cache-dir", str(pmlb_cache_dir)]
            if max_product is not None and max_product > 0:
                cmd += ["--max-product", str(max_product)]
            if max_samples is not None:
                cmd += ["--max-samples", str(max_samples)]
            if max_rows is not None:
                cmd += ["--max-rows", str(max_rows)]
            if max_features is not None:
                cmd += ["--max-features", str(max_features)]
            if splitters is not None:
                cmd += ["--splitters", ",".join(splitters)]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"[regression] skip {name}: subprocess timed out after 300s; adding to skip list for this run", file=sys.stderr)
                continue
            if result.returncode != 0:
                print(f"[regression] skip {name}: subprocess exited with {result.returncode} (SIGSEGV or error); adding to skip list for this run", file=sys.stderr)
                if result.stderr:
                    print(result.stderr.decode(errors="replace")[:500], file=sys.stderr)
                continue
            try:
                out = result.stdout
                if not out.strip():
                    continue
                dataset_rows = pickle.loads(out)
            except Exception as e:
                print(f"[regression] skip {name}: failed to read subprocess output ({e})", file=sys.stderr)
                continue
            if dataset_rows:
                n_s, n_f = dataset_rows[0]["n_samples"], dataset_rows[0]["n_features"]
                print(f"[regression] {i+1}/{len(datasets)} {name} (n={n_s}, p={n_f})", flush=True)
                rows.extend(dataset_rows)
    else:
        for i, name in enumerate(datasets):
            try:
                cache_dir = Path(pmlb_cache_dir) if pmlb_cache_dir is not None else (outdir / "pmlb_cache")
                X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
            except Exception as e:
                print(f"[regression] skip {name}: {e}", file=sys.stderr)
                continue
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print(f"[regression] skip {name}: contains NaN", file=sys.stderr)
                continue
            if max_rows is not None and len(y) > max_rows:
                if dataset is not None:
                    orig_n = len(y)
                    rng = np.random.default_rng(random_state)
                    idx = rng.choice(orig_n, size=max_rows, replace=False)
                    X, y = X[idx], y[idx]
                    print(f"[regression] {name}: subsampled to n={max_rows} (from {orig_n})", file=sys.stderr)
                else:
                    print(f"[regression] skip {name}: n={len(y)} > max_rows={max_rows}", file=sys.stderr)
                    continue
            if max_features is not None and X.shape[1] > max_features:
                print(f"[regression] skip {name}: p={X.shape[1]} > max_features={max_features}", file=sys.stderr)
                continue
            if max_samples is not None and len(y) > max_samples:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(len(y), size=max_samples, replace=False)
                X, y = X[idx], y[idx]
            n_samples, n_features = X.shape
            if max_product is not None and n_samples * n_features > max_product:
                print(f"[regression] skip {name}: n*p={n_samples * n_features} > max_product={max_product}", file=sys.stderr)
                continue
            print(f"[regression] {i+1}/{len(datasets)} {name} (n={n_samples}, p={n_features})", flush=True)

            try:
                dataset_rows = _evaluate_regression_dataset(name, X, y, random_state, splitters=splitters)
                rows.extend(dataset_rows)
            except Exception as e:
                print(f"[regression] skip {name}: fit error ({e})", file=sys.stderr)
                continue

    if not rows:
        print(f"No regression results to write.", file=sys.stderr)
        return [], path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")
    return rows, path


def run_classification(max_datasets=None, max_samples=None, max_rows=None, max_features=None, max_product=None, criterion="gini", outdir=None, dataset=None, datasets=None, pmlb_cache_dir=None, exclude=None, random_state=None, isolate_datasets=False, per_run_path=None, splitters=None):
    """Run classification benchmark once. Returns (rows, path). random_state controls estimator and subsampling RNG.
    If isolate_datasets=True, each dataset runs in a subprocess; SIGSEGV (or any non-zero exit) skips that dataset automatically.
    If per_run_path is set (e.g. by run_benchmark_n_times), results are written there instead of classification_{criterion}_results.csv (so each run gets its own file).
    When both criteria are run from the main driver, their final CSVs are post-filtered to the common successful dataset set."""
    splitters = splitters if splitters is not None else SPLITTERS
    random_state = random_state if random_state is not None else RANDOM_STATE
    outdir = Path(outdir or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    path = Path(per_run_path) if per_run_path is not None else outdir / f"classification_{criterion}_results.csv"
    exclude = set(exclude or [])

    if datasets is not None:
        datasets = [
            n
            for n in datasets
            if n in classification_dataset_names and not n.startswith("_deprecated_") and n not in exclude
        ]
        if max_datasets is not None:
            datasets = datasets[: max_datasets]
    elif dataset is not None:
        datasets = [dataset] if dataset in classification_dataset_names else []
    else:
        datasets = [n for n in classification_dataset_names if not n.startswith("_deprecated_") and n not in exclude]
        if max_datasets is not None:
            datasets = datasets[: max_datasets]

    rows = []
    if isolate_datasets:
        script = Path(__file__).resolve()
        for i, name in enumerate(datasets):
            cmd = [
                sys.executable,
                str(script),
                "--run-single-dataset", name,
                "--task", f"classification_{criterion}",
                "--random-state", str(random_state),
                "--outdir", str(outdir),
            ]
            if pmlb_cache_dir is not None:
                cmd += ["--pmlb-cache-dir", str(pmlb_cache_dir)]
            if max_product is not None and max_product > 0:
                cmd += ["--max-product", str(max_product)]
            if max_samples is not None:
                cmd += ["--max-samples", str(max_samples)]
            if max_rows is not None:
                cmd += ["--max-rows", str(max_rows)]
            if max_features is not None:
                cmd += ["--max-features", str(max_features)]
            if splitters is not None:
                cmd += ["--splitters", ",".join(splitters)]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"[classification {criterion}] skip {name}: subprocess timed out after 300s", file=sys.stderr)
                continue
            if result.returncode != 0:
                print(f"[classification {criterion}] skip {name}: subprocess exited with {result.returncode} (SIGSEGV or error)", file=sys.stderr)
                if result.stderr:
                    print(result.stderr.decode(errors="replace")[:500], file=sys.stderr)
                continue
            try:
                out = result.stdout
                if not out.strip():
                    continue
                dataset_rows = pickle.loads(out)
            except Exception as e:
                print(f"[classification {criterion}] skip {name}: failed to read subprocess output ({e})", file=sys.stderr)
                continue
            if dataset_rows:
                n_s, n_f = dataset_rows[0]["n_samples"], dataset_rows[0]["n_features"]
                print(f"[classification {criterion}] {i+1}/{len(datasets)} {name} (n={n_s}, p={n_f})", flush=True)
                rows.extend(dataset_rows)
    else:
        for i, name in enumerate(datasets):
            try:
                cache_dir = Path(pmlb_cache_dir) if pmlb_cache_dir is not None else (outdir / "pmlb_cache")
                X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
            except Exception as e:
                print(f"[classification {criterion}] skip {name}: {e}", file=sys.stderr)
                continue
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.intp)
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print(f"[classification {criterion}] skip {name}: contains NaN", file=sys.stderr)
                continue
            if max_rows is not None and len(y) > max_rows:
                if dataset is not None:
                    orig_n = len(y)
                    rng = np.random.default_rng(random_state)
                    idx = rng.choice(orig_n, size=max_rows, replace=False)
                    X, y = X[idx], y[idx]
                    print(f"[classification {criterion}] {name}: subsampled to n={max_rows} (from {orig_n})", file=sys.stderr)
                else:
                    print(f"[classification {criterion}] skip {name}: n={len(y)} > max_rows={max_rows}", file=sys.stderr)
                    continue
            if max_features is not None and X.shape[1] > max_features:
                print(f"[classification {criterion}] skip {name}: p={X.shape[1]} > max_features={max_features}", file=sys.stderr)
                continue
            if max_samples is not None and len(y) > max_samples:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(len(y), size=max_samples, replace=False)
                X, y = X[idx], y[idx]
            n_samples, n_features = X.shape
            if max_product is not None and n_samples * n_features > max_product:
                print(f"[classification {criterion}] skip {name}: n*p={n_samples * n_features} > max_product={max_product}", file=sys.stderr)
                continue
            print(f"[classification {criterion}] {i+1}/{len(datasets)} {name} (n={n_samples}, p={n_features})", flush=True)

            try:
                dataset_rows = _evaluate_classification_dataset(
                    name,
                    X,
                    y,
                    criterion,
                    random_state,
                    splitters=splitters,
                )
                rows.extend(dataset_rows)
            except Exception as e:
                print(f"[classification {criterion}] skip {name}: fit error ({e})", file=sys.stderr)
                continue

    if not rows:
        print(f"No classification ({criterion}) results to write.", file=sys.stderr)
        return [], path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")
    return rows, path


def _aggregate_regression_rows(all_rows):
    """Aggregate N runs: group by (dataset, splitter, variant), mean and std of metrics across runs."""
    from collections import defaultdict
    key_to_values = defaultdict(list)
    for run_rows in all_rows:
        for r in run_rows:
            key = (
                r["dataset"],
                r["n_samples"],
                r["n_features"],
                r["splitter"],
                r.get("variant", ""),
            )
            key_to_values[key].append(r)
    out = []
    for (dataset, n_s, n_f, splitter, variant), run_list in sorted(key_to_values.items()):
        rmse_means = [x["rmse_mean"] for x in run_list]
        rmse_stds = [x["rmse_std"] for x in run_list]
        fit_means = [x["fit_time_mean"] for x in run_list]
        row = {
            "dataset": dataset,
            "n_samples": n_s,
            "n_features": n_f,
            "splitter": splitter,
            "variant": variant,
            "rmse_mean": float(np.mean(rmse_means)),
            "rmse_std": float(np.std(rmse_means)) if len(rmse_means) > 1 else float(np.mean(rmse_stds)),
            "fit_time_mean": float(np.mean(fit_means)),
            "fit_time_std": float(np.std(fit_means)) if len(fit_means) > 1 else 0.0,
            "n_runs": len(run_list),
        }
        for key in EFFORT_KEYS:
            metric_key = f"{key}_mean"
            values = [float(x[metric_key]) for x in run_list if metric_key in x and not np.isnan(x[metric_key])]
            row[metric_key] = float(np.mean(values)) if values else float("nan")
        out.append(row)
    return out


def _aggregate_classification_rows(all_rows):
    """Aggregate N runs: group by (dataset, criterion, splitter, variant), mean and std of metrics across runs."""
    from collections import defaultdict
    key_to_values = defaultdict(list)
    for run_rows in all_rows:
        for r in run_rows:
            key = (
                r["dataset"],
                r["n_samples"],
                r["n_features"],
                r["criterion"],
                r["splitter"],
                r.get("variant", ""),
            )
            key_to_values[key].append(r)
    out = []
    for (dataset, n_s, n_f, criterion, splitter, variant), run_list in sorted(key_to_values.items()):
        acc_means = [x["accuracy_mean"] for x in run_list]
        f1_means = [x["f1_weighted_mean"] for x in run_list]
        fit_means = [x["fit_time_mean"] for x in run_list]
        row = {
            "dataset": dataset,
            "n_samples": n_s,
            "n_features": n_f,
            "criterion": criterion,
            "splitter": splitter,
            "variant": variant,
            "accuracy_mean": float(np.mean(acc_means)),
            "accuracy_std": float(np.std(acc_means)) if len(acc_means) > 1 else run_list[0]["accuracy_std"],
            "f1_weighted_mean": float(np.mean(f1_means)),
            "f1_weighted_std": float(np.std(f1_means)) if len(f1_means) > 1 else run_list[0]["f1_weighted_std"],
            "fit_time_mean": float(np.mean(fit_means)),
            "fit_time_std": float(np.std(fit_means)) if len(fit_means) > 1 else 0.0,
            "n_runs": len(run_list),
        }
        for key in EFFORT_KEYS:
            metric_key = f"{key}_mean"
            values = [float(x[metric_key]) for x in run_list if metric_key in x and not np.isnan(x[metric_key])]
            row[metric_key] = float(np.mean(values)) if values else float("nan")
        out.append(row)
    return out


def run_benchmark_n_times(
    n_runs,
    random_state=None,
    max_datasets=None,
    max_samples=None,
    max_rows=None,
    max_features=None,
    max_product=None,
    outdir=None,
    dataset=None,
    datasets=None,
    pmlb_cache_dir=None,
    exclude=None,
    regression_only=False,
    classification_only=False,
    isolate_datasets=False,
    splitters=None,
):
    """
    Run the full benchmark N times (each with a different seed: random_state, random_state+1, ...),
    then aggregate results so that each (dataset, splitter) has mean and std of metrics across the N runs.
    CV folds are otherwise unchanged across runs because cross_validate uses fixed, unshuffled folds.
    Each run is written to a separate file (regression_run001.csv, ..., regression_run{N}.csv, and
    classification_{criterion}_run001.csv etc.) so no run is overwritten. The aggregated summary
    is written to regression_results.csv and classification_*_results.csv.
    """
    random_state = random_state if random_state is not None else RANDOM_STATE
    outdir = Path(outdir or ".")
    outdir.mkdir(parents=True, exist_ok=True)

    all_regression = []
    all_classification_gini = []
    all_classification_entropy = []
    run_times = []  # seconds per run

    for run_idx in range(n_runs):
        seed = random_state + run_idx
        print(f"\n--- Run {run_idx + 1}/{n_runs} (random_state={seed}) ---", flush=True)
        t0 = time.perf_counter()
        run_suffix = f"_run{run_idx + 1:03d}.csv"
        if not classification_only:
            rows_reg, _ = run_regression(
                max_datasets=max_datasets,
                max_samples=max_samples,
                max_rows=max_rows,
                max_features=max_features,
                max_product=max_product,
                outdir=outdir,
                dataset=dataset,
                datasets=datasets,
                pmlb_cache_dir=pmlb_cache_dir,
                exclude=exclude,
                random_state=seed,
                isolate_datasets=isolate_datasets,
                per_run_path=outdir / f"regression{run_suffix}",
                splitters=splitters,
            )
            all_regression.append(rows_reg)
        if not regression_only:
            rows_by_criterion = {}
            per_run_paths = {}
            fieldnames_by_criterion = {}
            for criterion in CRITERIA_CLF:
                per_run_path = outdir / f"classification_{criterion}{run_suffix}"
                rows_clf, _ = run_classification(
                    max_datasets=max_datasets,
                    max_samples=max_samples,
                    max_rows=max_rows,
                    max_features=max_features,
                    max_product=max_product,
                    criterion=criterion,
                    outdir=outdir,
                    dataset=dataset,
                    datasets=datasets,
                    pmlb_cache_dir=pmlb_cache_dir,
                    exclude=exclude,
                    random_state=seed,
                    isolate_datasets=isolate_datasets,
                    per_run_path=per_run_path,
                    splitters=splitters,
                )
                rows_by_criterion[criterion] = rows_clf
                per_run_paths[criterion] = per_run_path
                fieldnames_by_criterion[criterion] = list(rows_clf[0].keys()) if rows_clf else None

            if set(CRITERIA_CLF).issubset(rows_by_criterion):
                rows_gini, rows_entropy, _ = _enforce_common_classification_datasets(
                    rows_by_criterion["gini"],
                    rows_by_criterion["entropy"],
                    label=f"run {run_idx + 1}/{n_runs}",
                )
                rows_by_criterion["gini"] = rows_gini
                rows_by_criterion["entropy"] = rows_entropy
                _write_rows_csv(
                    per_run_paths["gini"],
                    rows_gini,
                    fieldnames=fieldnames_by_criterion.get("gini") or fieldnames_by_criterion.get("entropy"),
                )
                _write_rows_csv(
                    per_run_paths["entropy"],
                    rows_entropy,
                    fieldnames=fieldnames_by_criterion.get("entropy") or fieldnames_by_criterion.get("gini"),
                )

            all_classification_gini.append(rows_by_criterion.get("gini", []))
            all_classification_entropy.append(rows_by_criterion.get("entropy", []))

        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)
        print(f"Run {run_idx + 1}/{n_runs} took {elapsed:.1f}s", flush=True)
        if run_idx + 1 < n_runs:
            remaining = n_runs - (run_idx + 1)
            avg_per_run = sum(run_times) / len(run_times)
            eta_sec = avg_per_run * remaining
            if eta_sec >= 3600:
                eta_str = f"~{eta_sec / 3600:.1f}h remaining"
            elif eta_sec >= 60:
                eta_str = f"~{eta_sec / 60:.1f}m remaining"
            else:
                eta_str = f"~{eta_sec:.0f}s remaining"
            print(f"Estimated {eta_str} ({remaining} runs × ~{avg_per_run:.1f}s/run)", flush=True)

    # Aggregate and write (aggregated summary to regression_results.csv etc.; per-run files already written above)
    if all_regression:
        agg_reg = _aggregate_regression_rows(all_regression)
        path_reg = outdir / "regression_results.csv"
        fieldnames = list(agg_reg[0].keys())
        with open(path_reg, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(agg_reg)
        print(f"Wrote aggregated regression ({n_runs} runs) to {path_reg}")
    if all_classification_gini:
        agg_gini = _aggregate_classification_rows(all_classification_gini)
        path_gini = outdir / "classification_gini_results.csv"
        fieldnames = list(agg_gini[0].keys())
        with open(path_gini, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(agg_gini)
        print(f"Wrote aggregated classification gini ({n_runs} runs) to {path_gini}")
    if all_classification_entropy:
        agg_ent = _aggregate_classification_rows(all_classification_entropy)
        path_ent = outdir / "classification_entropy_results.csv"
        fieldnames = list(agg_ent[0].keys())
        with open(path_ent, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(agg_ent)
        print(f"Wrote aggregated classification entropy ({n_runs} runs) to {path_ent}")


def main():
    p = argparse.ArgumentParser(description="Benchmark secretary splitters on PMLB")
    p.add_argument("--max-datasets", type=int, default=None, help="Max regression and classification datasets each (default: all)")
    p.add_argument("--max-samples", type=int, default=None, help="Subsample to this many rows per dataset (default: no limit)")
    p.add_argument("--max-rows", type=int, default=None, help="Skip datasets with more than this many rows (default: no limit)")
    p.add_argument("--max-features", type=int, default=None, help="Skip datasets with more than this many features (default: no limit)")
    p.add_argument("--max-product", type=int, default=1000000, help="Skip datasets with n_samples*n_features > this (default: 1000000, use 0 for no limit)")
    p.add_argument("--outdir", type=str, default=None, help="Output directory for CSVs (default: examples/early_stop_trees/benchmark_results)")
    p.add_argument("--dataset", type=str, default=None, help="Run only this dataset (by name); must be in regression and/or classification list")
    p.add_argument(
        "--datasets-file",
        type=str,
        default=None,
        help="Optional newline-separated dataset list. Names outside the selected task are ignored.",
    )
    p.add_argument(
        "--pmlb-cache-dir",
        type=str,
        default=None,
        help="Optional shared PMLB cache directory. Defaults to <outdir>/pmlb_cache.",
    )
    p.add_argument("--exclude-datasets", type=str, default=None, help="Comma-separated dataset names to skip (e.g. 192_vineyard)")
    p.add_argument("--regression-only", action="store_true", help="Run only regression")
    p.add_argument("--classification-only", action="store_true", help="Run only classification")
    p.add_argument("--random-state", type=int, default=None, help="Random seed for estimators and subsampling (default: 42). Repeated runs increment this seed while keeping the CV folds fixed.")
    p.add_argument("--n-runs", type=int, default=1, help="Run full benchmark N times and aggregate mean (and std) of metrics across runs (default: 1). Repeated runs vary estimator randomness, not the CV folds.")
    p.add_argument("--isolate-datasets", action="store_true", help="Run each dataset in a subprocess; if one crashes (e.g. SIGSEGV), skip it and continue.")
    p.add_argument("--run-single-dataset", type=str, default=None, help="(Internal) Run only this dataset and print pickle of rows to stdout.")
    p.add_argument("--task", type=str, default=None, help="(Internal) With --run-single-dataset: regression, classification_gini, or classification_entropy")
    p.add_argument(
        "--splitters",
        type=str,
        default=None,
        help="Comma-separated splitter names (default: all). E.g. best,secretary,prophet_1sample,extra_tree",
    )
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if args.outdir else script_dir / "benchmark_results"
    exclude = list(SKIP_DATASETS) + [s.strip() for s in (args.exclude_datasets or "").split(",") if s.strip()]
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE
    max_product = args.max_product if args.max_product > 0 else None
    splitters = tuple(s.strip() for s in (args.splitters or "").split(",") if s.strip()) if args.splitters else None
    if args.dataset is not None and args.datasets_file is not None:
        print("Use either --dataset or --datasets-file, not both.", file=sys.stderr)
        sys.exit(2)
    datasets = None
    if args.datasets_file is not None:
        datasets = [
            line.strip()
            for line in Path(args.datasets_file).read_text().splitlines()
            if line.strip()
        ]

    # Entry point for subprocess: run one dataset and output pickle to stdout
    if args.run_single_dataset is not None:
        name = args.run_single_dataset
        outdir.mkdir(parents=True, exist_ok=True)
        max_prod = max_product if max_product is not None else None
        try:
            if args.task == "regression":
                rows = _run_single_dataset_regression(
                    name, outdir, random_state,
                    pmlb_cache_dir=args.pmlb_cache_dir,
                    max_samples=args.max_samples,
                    max_rows=args.max_rows,
                    max_features=args.max_features,
                    max_product=max_prod,
                    splitters=splitters,
                )
            elif args.task in ("classification_gini", "classification_entropy"):
                criterion = "gini" if args.task == "classification_gini" else "entropy"
                rows = _run_single_dataset_classification(
                    name, criterion, outdir, random_state,
                    pmlb_cache_dir=args.pmlb_cache_dir,
                    max_samples=args.max_samples,
                    max_rows=args.max_rows,
                    max_features=args.max_features,
                    max_product=max_prod,
                    splitters=splitters,
                )
            else:
                print(f"Unknown task: {args.task}", file=sys.stderr)
                sys.exit(1)
            sys.stdout.buffer.write(pickle.dumps(rows))
            sys.exit(0)
        except Exception as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    isolate = args.isolate_datasets
    outdir.mkdir(parents=True, exist_ok=True)
    _write_benchmark_metadata(outdir, splitters or SPLITTERS, args.n_runs, random_state)

    if args.n_runs > 1:
        run_benchmark_n_times(
            n_runs=args.n_runs,
            random_state=random_state,
            max_datasets=args.max_datasets,
            max_samples=args.max_samples,
            max_rows=args.max_rows,
            max_features=args.max_features,
            max_product=max_product,
            outdir=outdir,
            dataset=args.dataset,
            datasets=datasets,
            pmlb_cache_dir=args.pmlb_cache_dir,
            exclude=exclude,
            regression_only=args.regression_only,
            classification_only=args.classification_only,
            isolate_datasets=isolate,
            splitters=splitters,
        )
    else:
        if not args.classification_only:
            run_regression(
                max_datasets=args.max_datasets,
                max_samples=args.max_samples,
                max_rows=args.max_rows,
                max_features=args.max_features,
                max_product=max_product,
                outdir=outdir,
                dataset=args.dataset,
                datasets=datasets,
                pmlb_cache_dir=args.pmlb_cache_dir,
                exclude=exclude,
                random_state=random_state,
                isolate_datasets=isolate,
                splitters=splitters,
            )
        if not args.regression_only:
            rows_by_criterion = {}
            path_by_criterion = {}
            fieldnames_by_criterion = {}
            for criterion in CRITERIA_CLF:
                rows_clf, path_clf = run_classification(
                    max_datasets=args.max_datasets,
                    max_samples=args.max_samples,
                    max_rows=args.max_rows,
                    max_features=args.max_features,
                    max_product=max_product,
                    criterion=criterion,
                    outdir=outdir,
                    dataset=args.dataset,
                    datasets=datasets,
                    pmlb_cache_dir=args.pmlb_cache_dir,
                    exclude=exclude,
                    random_state=random_state,
                    isolate_datasets=isolate,
                    splitters=splitters,
                )
                rows_by_criterion[criterion] = rows_clf
                path_by_criterion[criterion] = path_clf
                fieldnames_by_criterion[criterion] = list(rows_clf[0].keys()) if rows_clf else None

            if set(CRITERIA_CLF).issubset(rows_by_criterion):
                rows_gini, rows_entropy, _ = _enforce_common_classification_datasets(
                    rows_by_criterion["gini"],
                    rows_by_criterion["entropy"],
                    label="single run",
                )
                _write_rows_csv(
                    path_by_criterion["gini"],
                    rows_gini,
                    fieldnames=fieldnames_by_criterion.get("gini") or fieldnames_by_criterion.get("entropy"),
                )
                _write_rows_csv(
                    path_by_criterion["entropy"],
                    rows_entropy,
                    fieldnames=fieldnames_by_criterion.get("entropy") or fieldnames_by_criterion.get("gini"),
                )
    print("Done.")


if __name__ == "__main__":
    main()
