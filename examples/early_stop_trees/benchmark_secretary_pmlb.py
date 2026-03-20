#!/usr/bin/env python
"""
Benchmark all secretary splitters on PMLB datasets.

Regression: RMSE and fit time (5-fold CV).
Classification: accuracy, F1-score (weighted), and fit time for Gini and Entropy (5-fold CV).

Secretary procedures depend on randomness; it is controlled by the estimator's
random_state hyperparameter (and by --random-state in this script).

Usage:
  pip install pmlb
  python examples/early_stop_trees/benchmark_secretary_pmlb.py [--max-datasets N] [--outdir DIR]
  python examples/early_stop_trees/benchmark_secretary_pmlb.py --n-runs 10 --random-state 42  # N runs, then aggregate means
  python examples/early_stop_trees/benchmark_secretary_pmlb.py --isolate-datasets  # run each dataset in a subprocess; SIGSEGV skips that dataset automatically

Output: CSV files in outdir (default: examples/early_stop_trees/benchmark_results).
"""
import argparse
import csv
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from pmlb import (
        fetch_data,
        regression_dataset_names,
        classification_dataset_names,
    )
except ImportError:
    print("Install pmlb: pip install pmlb", file=sys.stderr)
    sys.exit(1)

from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, f1_score, make_scorer

from treeple.tree import EarlyStopDecisionTreeRegressor, EarlyStopDecisionTreeClassifier


RANDOM_STATE = 42
N_FOLDS = 5
# Base splitter families; secretary_par will be expanded into multiple parameter variants.
SPLITTERS = ("best", "secretary", "secretary_par", "secretary_all", "double_secretary", "block_rank", "prophet_1sample")
CRITERIA_CLF = ("gini", "entropy")
# Datasets to skip by default (e.g. known to segfault or raise in C extension)
SKIP_DATASETS = ("192_vineyard", "687_sleuth_ex1605")

# Exit code when child process is killed by SIGSEGV (128 + 11)
SIGSEGV_EXIT = 139


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


def _run_single_dataset_regression(
    name,
    outdir,
    random_state,
    max_samples=None,
    max_rows=None,
    max_features=None,
    max_product=None,
    splitters=None,
):
    """Run regression for one dataset; return list of row dicts or [] if skipped (fetch/size)."""
    splitters = splitters if splitters is not None else SPLITTERS
    outdir = Path(outdir or ".")
    try:
        X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(outdir / "pmlb_cache"))
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
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )
    dataset_rows = []
    for splitter in splitters:
        if splitter == "secretary_par":
            # SecretaryParam grid over (samples, quantile)
            for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
                for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                    est = EarlyStopDecisionTreeRegressor(
                        splitter="secretary_par",
                        random_state=random_state,
                        max_depth=20,
                        split_search={
                            "p_thr_par": p_thr_par,
                            "q_thr_par": q_thr_par,
                            "n_gain_samples_par": int(n_gain_samples_par),
                        },
                    )
                    cv = cross_validate(
                        est,
                        X,
                        y,
                        cv=N_FOLDS,
                        scoring={"neg_rmse": rmse_scorer},
                        return_train_score=False,
                    )
                    rmse_mean = -float(np.mean(cv["test_neg_rmse"]))
                    rmse_std = float(np.std(cv["test_neg_rmse"]))
                    fit_time_mean = float(np.mean(cv["fit_time"]))
                    dataset_rows.append(
                        {
                            "dataset": name,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "splitter": "secretary_par",
                            "variant": f"samples={sample_mode},q={q_thr_par}",
                            "rmse_mean": rmse_mean,
                            "rmse_std": rmse_std,
                            "fit_time_mean": fit_time_mean,
                        }
                    )
        elif splitter in ("secretary", "secretary_all", "double_secretary"):
            for split_search, variant_label in _secretary_variants(n_samples):
                est = EarlyStopDecisionTreeRegressor(
                    splitter=splitter,
                    random_state=random_state,
                    max_depth=20,
                    split_search=split_search,
                )
                cv = cross_validate(
                    est,
                    X,
                    y,
                    cv=N_FOLDS,
                    scoring={"neg_rmse": rmse_scorer},
                    return_train_score=False,
                )
                rmse_mean = -float(np.mean(cv["test_neg_rmse"]))
                rmse_std = float(np.std(cv["test_neg_rmse"]))
                fit_time_mean = float(np.mean(cv["fit_time"]))
                dataset_rows.append(
                    {
                        "dataset": name,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "splitter": splitter,
                        "variant": variant_label,
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "fit_time_mean": fit_time_mean,
                    }
                )
        else:
            # best, block_rank, prophet_1sample
            est = EarlyStopDecisionTreeRegressor(
                splitter=splitter,
                random_state=random_state,
                max_depth=20,
            )
            cv = cross_validate(
                est,
                X,
                y,
                cv=N_FOLDS,
                scoring={"neg_rmse": rmse_scorer},
                return_train_score=False,
            )
            rmse_mean = -float(np.mean(cv["test_neg_rmse"]))
            rmse_std = float(np.std(cv["test_neg_rmse"]))
            fit_time_mean = float(np.mean(cv["fit_time"]))
            dataset_rows.append(
                {
                    "dataset": name,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "",
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "fit_time_mean": fit_time_mean,
                }
            )
    return dataset_rows


def _run_single_dataset_classification(
    name,
    criterion,
    outdir,
    random_state,
    max_samples=None,
    max_rows=None,
    max_features=None,
    max_product=None,
    splitters=None,
):
    """Run classification for one dataset; return list of row dicts or [] if skipped."""
    splitters = splitters if splitters is not None else SPLITTERS
    outdir = Path(outdir or ".")
    try:
        X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(outdir / "pmlb_cache"))
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
    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }
    dataset_rows = []
    for splitter in splitters:
        if splitter == "secretary_par":
            for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
                for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                    est = EarlyStopDecisionTreeClassifier(
                        splitter="secretary_par",
                        criterion=criterion,
                        random_state=random_state,
                        max_depth=20,
                        split_search={
                            "p_thr_par": p_thr_par,
                            "q_thr_par": q_thr_par,
                            "n_gain_samples_par": int(n_gain_samples_par),
                        },
                    )
                    cv = cross_validate(
                        est,
                        X,
                        y,
                        cv=N_FOLDS,
                        scoring=scoring,
                        return_train_score=False,
                    )
                    acc_mean = float(np.mean(cv["test_accuracy"]))
                    acc_std = float(np.std(cv["test_accuracy"]))
                    f1_mean = float(np.mean(cv["test_f1_weighted"]))
                    f1_std = float(np.std(cv["test_f1_weighted"]))
                    fit_time_mean = float(np.mean(cv["fit_time"]))
                    dataset_rows.append(
                        {
                            "dataset": name,
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "criterion": criterion,
                            "splitter": "secretary_par",
                            "variant": f"samples={sample_mode},q={q_thr_par}",
                            "accuracy_mean": acc_mean,
                            "accuracy_std": acc_std,
                            "f1_weighted_mean": f1_mean,
                            "f1_weighted_std": f1_std,
                            "fit_time_mean": fit_time_mean,
                        }
                    )
        elif splitter in ("secretary", "secretary_all", "double_secretary"):
            for split_search, variant_label in _secretary_variants(n_samples):
                est = EarlyStopDecisionTreeClassifier(
                    splitter=splitter,
                    criterion=criterion,
                    random_state=random_state,
                    max_depth=20,
                    split_search=split_search,
                )
                cv = cross_validate(
                    est,
                    X,
                    y,
                    cv=N_FOLDS,
                    scoring=scoring,
                    return_train_score=False,
                )
                acc_mean = float(np.mean(cv["test_accuracy"]))
                acc_std = float(np.std(cv["test_accuracy"]))
                f1_mean = float(np.mean(cv["test_f1_weighted"]))
                f1_std = float(np.std(cv["test_f1_weighted"]))
                fit_time_mean = float(np.mean(cv["fit_time"]))
                dataset_rows.append(
                    {
                        "dataset": name,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "criterion": criterion,
                        "splitter": splitter,
                        "variant": variant_label,
                        "accuracy_mean": acc_mean,
                        "accuracy_std": acc_std,
                        "f1_weighted_mean": f1_mean,
                        "f1_weighted_std": f1_std,
                        "fit_time_mean": fit_time_mean,
                    }
                )
        else:
            est = EarlyStopDecisionTreeClassifier(
                splitter=splitter,
                criterion=criterion,
                random_state=random_state,
                max_depth=20,
            )
            cv = cross_validate(
                est,
                X,
                y,
                cv=N_FOLDS,
                scoring=scoring,
                return_train_score=False,
            )
            acc_mean = float(np.mean(cv["test_accuracy"]))
            acc_std = float(np.std(cv["test_accuracy"]))
            f1_mean = float(np.mean(cv["test_f1_weighted"]))
            f1_std = float(np.std(cv["test_f1_weighted"]))
            fit_time_mean = float(np.mean(cv["fit_time"]))
            dataset_rows.append(
                {
                    "dataset": name,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "criterion": criterion,
                    "splitter": splitter,
                    "variant": "",
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "f1_weighted_mean": f1_mean,
                    "f1_weighted_std": f1_std,
                    "fit_time_mean": fit_time_mean,
                }
            )
    return dataset_rows


def run_regression(max_datasets=None, max_samples=None, max_rows=None, max_features=None, max_product=None, outdir=None, dataset=None, exclude=None, random_state=None, isolate_datasets=False, per_run_path=None, splitters=None):
    """Run regression benchmark once. Returns (rows, path). random_state controls estimator and subsampling RNG.
    If isolate_datasets=True, each dataset runs in a subprocess; SIGSEGV (or any non-zero exit) skips that dataset automatically.
    If per_run_path is set (e.g. by run_benchmark_n_times), results are written there instead of regression_results.csv (so each run gets its own file)."""
    splitters = splitters if splitters is not None else SPLITTERS
    random_state = random_state if random_state is not None else RANDOM_STATE
    outdir = Path(outdir or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    path = Path(per_run_path) if per_run_path is not None else outdir / "regression_results.csv"
    exclude = set(exclude or [])

    if dataset is not None:
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
        rmse_scorer = make_scorer(
            lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=True,
        )
        for i, name in enumerate(datasets):
            try:
                X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(outdir / "pmlb_cache"))
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
                dataset_rows = []
                for splitter in splitters:
                    try:
                        est = EarlyStopDecisionTreeRegressor(
                            splitter=splitter,
                            random_state=random_state,
                            max_depth=20,
                        )
                        cv = cross_validate(
                            est,
                            X,
                            y,
                            cv=N_FOLDS,
                            scoring={"neg_rmse": rmse_scorer},
                            return_train_score=False,
                        )
                        rmse_mean = -float(np.mean(cv["test_neg_rmse"]))
                        rmse_std = float(np.std(cv["test_neg_rmse"]))
                        fit_time_mean = float(np.mean(cv["fit_time"]))
                    except Exception as e:
                        print(f"  {splitter}: {e}", file=sys.stderr)
                        raise
                    dataset_rows.append({
                        "dataset": name,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "splitter": splitter,
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "fit_time_mean": fit_time_mean,
                    })
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


def run_classification(max_datasets=None, max_samples=None, max_rows=None, max_features=None, max_product=None, criterion="gini", outdir=None, dataset=None, exclude=None, random_state=None, isolate_datasets=False, per_run_path=None, splitters=None):
    """Run classification benchmark once. Returns (rows, path). random_state controls estimator and subsampling RNG.
    If isolate_datasets=True, each dataset runs in a subprocess; SIGSEGV (or any non-zero exit) skips that dataset automatically.
    If per_run_path is set (e.g. by run_benchmark_n_times), results are written there instead of classification_{criterion}_results.csv (so each run gets its own file)."""
    splitters = splitters if splitters is not None else SPLITTERS
    random_state = random_state if random_state is not None else RANDOM_STATE
    outdir = Path(outdir or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    path = Path(per_run_path) if per_run_path is not None else outdir / f"classification_{criterion}_results.csv"
    exclude = set(exclude or [])

    if dataset is not None:
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
        scoring = {
            "accuracy": "accuracy",
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        }
        for i, name in enumerate(datasets):
            try:
                X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(outdir / "pmlb_cache"))
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
                dataset_rows = []
                for splitter in splitters:
                    try:
                        est = EarlyStopDecisionTreeClassifier(
                            splitter=splitter,
                            criterion=criterion,
                            random_state=random_state,
                            max_depth=20,
                        )
                        cv = cross_validate(
                            est,
                            X,
                            y,
                            cv=N_FOLDS,
                            scoring=scoring,
                            return_train_score=False,
                        )
                        acc_mean = float(np.mean(cv["test_accuracy"]))
                        acc_std = float(np.std(cv["test_accuracy"]))
                        f1_mean = float(np.mean(cv["test_f1_weighted"]))
                        f1_std = float(np.std(cv["test_f1_weighted"]))
                        fit_time_mean = float(np.mean(cv["fit_time"]))
                    except Exception as e:
                        print(f"  {splitter}: {e}", file=sys.stderr)
                        raise
                    dataset_rows.append({
                        "dataset": name,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "criterion": criterion,
                        "splitter": splitter,
                        "accuracy_mean": acc_mean,
                        "accuracy_std": acc_std,
                        "f1_weighted_mean": f1_mean,
                        "f1_weighted_std": f1_std,
                        "fit_time_mean": fit_time_mean,
                    })
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
        out.append({
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
        })
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
        out.append({
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
        })
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
    exclude=None,
    regression_only=False,
    classification_only=False,
    isolate_datasets=False,
    splitters=None,
):
    """
    Run the full benchmark N times (each with a different seed: random_state, random_state+1, ...),
    then aggregate results so that each (dataset, splitter) has mean and std of metrics across the N runs.
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
                exclude=exclude,
                random_state=seed,
                isolate_datasets=isolate_datasets,
                per_run_path=outdir / f"regression{run_suffix}",
                splitters=splitters,
            )
            all_regression.append(rows_reg)
        if not regression_only:
            for criterion in CRITERIA_CLF:
                rows_clf, _ = run_classification(
                    max_datasets=max_datasets,
                    max_samples=max_samples,
                    max_rows=max_rows,
                    max_features=max_features,
                    max_product=max_product,
                    criterion=criterion,
                    outdir=outdir,
                    dataset=dataset,
                    exclude=exclude,
                    random_state=seed,
                    isolate_datasets=isolate_datasets,
                    per_run_path=outdir / f"classification_{criterion}{run_suffix}",
                    splitters=splitters,
                )
                if criterion == "gini":
                    all_classification_gini.append(rows_clf)
                else:
                    all_classification_entropy.append(rows_clf)

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
    p.add_argument("--exclude-datasets", type=str, default=None, help="Comma-separated dataset names to skip (e.g. 192_vineyard)")
    p.add_argument("--regression-only", action="store_true", help="Run only regression")
    p.add_argument("--classification-only", action="store_true", help="Run only classification")
    p.add_argument("--random-state", type=int, default=None, help="Random seed for estimators and subsampling (default: 42). Controls all secretary randomness.")
    p.add_argument("--n-runs", type=int, default=1, help="Run full benchmark N times and aggregate mean (and std) of metrics across runs (default: 1)")
    p.add_argument("--isolate-datasets", action="store_true", help="Run each dataset in a subprocess; if one crashes (e.g. SIGSEGV), skip it and continue.")
    p.add_argument("--run-single-dataset", type=str, default=None, help="(Internal) Run only this dataset and print pickle of rows to stdout.")
    p.add_argument("--task", type=str, default=None, help="(Internal) With --run-single-dataset: regression, classification_gini, or classification_entropy")
    p.add_argument("--splitters", type=str, default=None, help="Comma-separated splitter names (default: all). E.g. best,secretary,prophet_1sample")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if args.outdir else script_dir / "benchmark_results"
    exclude = list(SKIP_DATASETS) + [s.strip() for s in (args.exclude_datasets or "").split(",") if s.strip()]
    random_state = args.random_state if args.random_state is not None else RANDOM_STATE
    max_product = args.max_product if args.max_product > 0 else None
    splitters = tuple(s.strip() for s in (args.splitters or "").split(",") if s.strip()) if args.splitters else None

    # Entry point for subprocess: run one dataset and output pickle to stdout
    if args.run_single_dataset is not None:
        name = args.run_single_dataset
        outdir.mkdir(parents=True, exist_ok=True)
        max_prod = max_product if max_product is not None else None
        try:
            if args.task == "regression":
                rows = _run_single_dataset_regression(
                    name, outdir, random_state,
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
                exclude=exclude,
                random_state=random_state,
                isolate_datasets=isolate,
                splitters=splitters,
            )
        if not args.regression_only:
            for criterion in CRITERIA_CLF:
                run_classification(
                    max_datasets=args.max_datasets,
                    max_samples=args.max_samples,
                    max_rows=args.max_rows,
                    max_features=args.max_features,
                    max_product=max_product,
                    criterion=criterion,
                    outdir=outdir,
                    dataset=args.dataset,
                    exclude=exclude,
                    random_state=random_state,
                    isolate_datasets=isolate,
                    splitters=splitters,
                )
    print("Done.")


if __name__ == "__main__":
    main()
