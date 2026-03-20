#!/usr/bin/env python
"""
Quick sanity benchmark of new early-stop strategies on SMALL PMLB datasets.

Focus:
- New splitters: prophet_1sample, block_rank.
- Secretary family with sqrt(n_node) exploration instead of 1/e.
- SecretaryParam (secretary_par) with a grid over:
  * number of gains used for the parametric fit
      - 10 samples (cap)
      - 10% of thresholds (p_thr_par = 0.1)
      - sqrt(n_samples) thresholds (approx via p_thr_par = sqrt(n)/n)
      - log(n_samples) thresholds (approx via p_thr_par = log(n)/n)
  * quantile threshold q_thr_par in {0.5, 0.75, 0.9, 0.95}.

We restrict to small datasets via max_product (n_samples * n_features) so that
the run finishes quickly.

Output:
- CSV files under examples/early_stop_trees/benchmark_results_new/:
  * regression_new_strats_small.csv
  * classification_gini_new_strats_small.csv
  * classification_entropy_new_strats_small.csv
"""

from __future__ import annotations

import csv
import pickle
import subprocess
import sys
from math import log, sqrt
from pathlib import Path

import numpy as np
from pmlb import classification_dataset_names, fetch_data, regression_dataset_names
from sklearn.metrics import f1_score, make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate

from treeple.tree import EarlyStopDecisionTreeClassifier, EarlyStopDecisionTreeRegressor


RANDOM_STATE = 42
N_FOLDS = 3  # smaller CV for quick tests
MAX_PRODUCT_SMALL = 2_500  # n_samples * n_features <= this
SIGSEGV_EXIT = 139  # 128 + 11, same convention as main benchmark

# Base secretary-style splitters we want to test with sqrt(n) exploration
SECRETARY_SPLITTERS_SQRT = (
    "secretary",
    "secretary_all",
    "double_secretary",
    "block_rank",
)


def _load_pmlb_dataset(name: str, outdir: Path, task: str):
    """Fetch dataset from PMLB; return (None, None) if not available or invalid.

    This mirrors the robust behaviour in benchmark_secretary_pmlb.py: datasets
    that cannot be loaded (missing from current PMLB version, NaNs, etc.) are
    simply skipped instead of crashing the whole benchmark.
    """
    try:
        X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(outdir / "pmlb_cache"))
    except Exception:
        # Dataset no longer present or fetch error in this PMLB version; skip.
        return None, None
    X = np.asarray(X, dtype=np.float64)
    if task == "regression":
        y = np.asarray(y, dtype=np.float64)
    else:
        y = np.asarray(y, dtype=np.intp)
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return None, None
    return X, y


def _small_enough(X, max_product: int) -> bool:
    n_samples, n_features = X.shape
    return n_samples * n_features <= max_product


def _secretary_par_grid(n_samples: int):
    """Return list of (p_thr_par, n_gain_samples_par, sample_mode_label)."""
    out = []
    # 10 samples (cap), sample_prob = 1.0
    out.append((1.0, 10, "10"))
    # 10% of thresholds (approx 0.1 fraction of splits)
    out.append((0.1, 256, "0.1n"))
    # sqrt(n_samples) thresholds: p = sqrt(n)/n
    if n_samples > 0:
        p_sqrt = min(1.0, sqrt(float(n_samples)) / float(n_samples))
        out.append((p_sqrt, 256, "sqrt_n"))
        # log(n_samples) thresholds: p = log(n)/n
        p_log = min(1.0, log(max(float(n_samples), 2.0)) / float(n_samples))
        out.append((p_log, 256, "log_n"))
    return out


def run_regression_small(outdir: Path, max_datasets: int = 40):
    out_path = outdir / "regression_new_strats_small.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rmse_scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )

    rows = []
    n_done = 0
    for name in regression_dataset_names:
        if n_done >= max_datasets:
            break
        X, y = _load_pmlb_dataset(name, outdir, task="regression")
        if X is None:
            continue
        if not _small_enough(X, MAX_PRODUCT_SMALL):
            continue
        n_samples, n_features = X.shape

        # 1) Best splitter (exhaustive)
        for splitter in ("best", "prophet_1sample"):
            est = EarlyStopDecisionTreeRegressor(
                splitter=splitter,
                random_state=RANDOM_STATE,
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
            rows.append(
                {
                    "dataset": name,
                    "task": "regression",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "",
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 2) Secretary-style with sqrt(n) exploration (including block_rank)
        for splitter in SECRETARY_SPLITTERS_SQRT:
            est = EarlyStopDecisionTreeRegressor(
                splitter=splitter,
                random_state=RANDOM_STATE,
                max_depth=20,
                split_search={"secretary_threshold": "sqrt_n"},
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
            rows.append(
                {
                    "dataset": name,
                    "task": "regression",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "sqrt_n",
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 3) SecretaryParam grid over (samples, quantile)
        for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
            for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                est = EarlyStopDecisionTreeRegressor(
                    splitter="secretary_par",
                    random_state=RANDOM_STATE,
                    max_depth=20,
                    split_search={
                        "p_thr_par": p_thr_par,
                        "q_thr_par": q_thr_par,
                        "n_gain_samples_par": int(n_gain_samples_par),
                        "secretary_threshold": "sqrt_n",
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
                rows.append(
                    {
                        "dataset": name,
                        "task": "regression",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "splitter": "secretary_par",
                        "variant": f"samples={sample_mode},q={q_thr_par}",
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "fit_time_mean": fit_time_mean,
                    }
                )

        n_done += 1

    if rows:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "task",
                    "n_samples",
                    "n_features",
                    "splitter",
                    "variant",
                    "rmse_mean",
                    "rmse_std",
                    "fit_time_mean",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


def run_classification_small(outdir: Path, criterion: str, max_datasets: int = 40):
    out_path = outdir / f"classification_{criterion}_new_strats_small.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    rows = []
    n_done = 0
    for name in classification_dataset_names:
        if n_done >= max_datasets:
            break
        X, y = _load_pmlb_dataset(name, outdir, task="classification")
        if X is None:
            continue
        if not _small_enough(X, MAX_PRODUCT_SMALL):
            continue
        n_samples, n_features = X.shape

        # 1) Best splitter and prophet_1sample
        for splitter in ("best", "prophet_1sample"):
            est = EarlyStopDecisionTreeClassifier(
                splitter=splitter,
                criterion=criterion,
                random_state=RANDOM_STATE,
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
            rows.append(
                {
                    "dataset": name,
                    "task": f"classification_{criterion}",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "",
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "f1_weighted_mean": f1_mean,
                    "f1_weighted_std": f1_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 2) Secretary-style with sqrt(n) exploration (including block_rank)
        for splitter in SECRETARY_SPLITTERS_SQRT:
            est = EarlyStopDecisionTreeClassifier(
                splitter=splitter,
                criterion=criterion,
                random_state=RANDOM_STATE,
                max_depth=20,
                split_search={"secretary_threshold": "sqrt_n"},
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
            rows.append(
                {
                    "dataset": name,
                    "task": f"classification_{criterion}",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "sqrt_n",
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "f1_weighted_mean": f1_mean,
                    "f1_weighted_std": f1_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 3) SecretaryParam grid over (samples, quantile)
        for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
            for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                est = EarlyStopDecisionTreeClassifier(
                    splitter="secretary_par",
                    criterion=criterion,
                    random_state=RANDOM_STATE,
                    max_depth=20,
                    split_search={
                        "p_thr_par": p_thr_par,
                        "q_thr_par": q_thr_par,
                        "n_gain_samples_par": int(n_gain_samples_par),
                        "secretary_threshold": "sqrt_n",
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
                rows.append(
                    {
                        "dataset": name,
                        "task": f"classification_{criterion}",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "splitter": "secretary_par",
                        "variant": f"samples={sample_mode},q={q_thr_par}",
                        "accuracy_mean": acc_mean,
                        "accuracy_std": acc_std,
                        "f1_weighted_mean": f1_mean,
                        "f1_weighted_std": f1_std,
                        "fit_time_mean": fit_time_mean,
                    }
                )

        n_done += 1

    if rows:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "task",
                    "n_samples",
                    "n_features",
                    "splitter",
                    "variant",
                    "accuracy_mean",
                    "accuracy_std",
                    "f1_weighted_mean",
                    "f1_weighted_std",
                    "fit_time_mean",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


def _run_single_dataset_small(task: str, name: str, outdir: Path):
    """Run a SINGLE dataset (reg or clf) and print pickled rows to stdout.

    This is used by the isolate-datasets driver below to avoid a single
    SIGSEGV killing the whole benchmark.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if task == "regression":
        run_regression_small_for_list(outdir, [name])
    elif task in ("classification_gini", "classification_entropy"):
        crit = "gini" if task == "classification_gini" else "entropy"
        run_classification_small_for_list(outdir, crit, [name])
    else:
        raise ValueError(f"Unknown task: {task}")


def run_regression_small_for_list(outdir: Path, dataset_names):
    """Like run_regression_small but restricted to a given list of dataset names."""
    out_path = outdir / "regression_new_strats_small.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rmse_scorer = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )

    rows = []
    for name in dataset_names:
        X, y = _load_pmlb_dataset(name, outdir, task="regression")
        if X is None:
            continue
        if not _small_enough(X, MAX_PRODUCT_SMALL):
            continue
        n_samples, n_features = X.shape

        # 1) Best splitter (exhaustive) and prophet_1sample
        for splitter in ("best", "prophet_1sample"):
            est = EarlyStopDecisionTreeRegressor(
                splitter=splitter,
                random_state=RANDOM_STATE,
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
            rows.append(
                {
                    "dataset": name,
                    "task": "regression",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "",
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 2) Secretary-style with sqrt(n) exploration (including block_rank)
        for splitter in SECRETARY_SPLITTERS_SQRT:
            est = EarlyStopDecisionTreeRegressor(
                splitter=splitter,
                random_state=RANDOM_STATE,
                max_depth=20,
                split_search={"secretary_threshold": "sqrt_n"},
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
            rows.append(
                {
                    "dataset": name,
                    "task": "regression",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "sqrt_n",
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 3) SecretaryParam grid over (samples, quantile)
        for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
            for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                est = EarlyStopDecisionTreeRegressor(
                    splitter="secretary_par",
                    random_state=RANDOM_STATE,
                    max_depth=20,
                    split_search={
                        "p_thr_par": p_thr_par,
                        "q_thr_par": q_thr_par,
                        "n_gain_samples_par": int(n_gain_samples_par),
                        "secretary_threshold": "sqrt_n",
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
                rows.append(
                    {
                        "dataset": name,
                        "task": "regression",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "splitter": "secretary_par",
                        "variant": f"samples={sample_mode},q={q_thr_par}",
                        "rmse_mean": rmse_mean,
                        "rmse_std": rmse_std,
                        "fit_time_mean": fit_time_mean,
                    }
                )

    if rows:
        mode = "w" if not out_path.exists() else "a"
        with out_path.open(mode, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "task",
                    "n_samples",
                    "n_features",
                    "splitter",
                    "variant",
                    "rmse_mean",
                    "rmse_std",
                    "fit_time_mean",
                ],
            )
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)


def run_classification_small_for_list(outdir: Path, criterion: str, dataset_names):
    out_path = outdir / f"classification_{criterion}_new_strats_small.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    }

    rows = []
    for name in dataset_names:
        X, y = _load_pmlb_dataset(name, outdir, task="classification")
        if X is None:
            continue
        if not _small_enough(X, MAX_PRODUCT_SMALL):
            continue
        n_samples, n_features = X.shape

        # 1) Best splitter and prophet_1sample
        for splitter in ("best", "prophet_1sample"):
            est = EarlyStopDecisionTreeClassifier(
                splitter=splitter,
                criterion=criterion,
                random_state=RANDOM_STATE,
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
            rows.append(
                {
                    "dataset": name,
                    "task": f"classification_{criterion}",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "",
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "f1_weighted_mean": f1_mean,
                    "f1_weighted_std": f1_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 2) Secretary-style with sqrt(n) exploration (including block_rank)
        for splitter in SECRETARY_SPLITTERS_SQRT:
            est = EarlyStopDecisionTreeClassifier(
                splitter=splitter,
                criterion=criterion,
                random_state=RANDOM_STATE,
                max_depth=20,
                split_search={"secretary_threshold": "sqrt_n"},
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
            rows.append(
                {
                    "dataset": name,
                    "task": f"classification_{criterion}",
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "splitter": splitter,
                    "variant": "sqrt_n",
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                    "f1_weighted_mean": f1_mean,
                    "f1_weighted_std": f1_std,
                    "fit_time_mean": fit_time_mean,
                }
            )

        # 3) SecretaryParam grid over (samples, quantile)
        for p_thr_par, n_gain_samples_par, sample_mode in _secretary_par_grid(n_samples):
            for q_thr_par in (0.5, 0.75, 0.9, 0.95):
                est = EarlyStopDecisionTreeClassifier(
                    splitter="secretary_par",
                    criterion=criterion,
                    random_state=RANDOM_STATE,
                    max_depth=20,
                    split_search={
                        "p_thr_par": p_thr_par,
                        "q_thr_par": q_thr_par,
                        "n_gain_samples_par": int(n_gain_samples_par),
                        "secretary_threshold": "sqrt_n",
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
                rows.append(
                    {
                        "dataset": name,
                        "task": f"classification_{criterion}",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "splitter": "secretary_par",
                        "variant": f"samples={sample_mode},q={q_thr_par}",
                        "accuracy_mean": acc_mean,
                        "accuracy_std": acc_std,
                        "f1_weighted_mean": f1_mean,
                        "f1_weighted_std": f1_std,
                        "fit_time_mean": fit_time_mean,
                    }
                )

    if rows:
        mode = "w" if not out_path.exists() else "a"
        with out_path.open(mode, newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "task",
                    "n_samples",
                    "n_features",
                    "splitter",
                    "variant",
                    "accuracy_mean",
                    "accuracy_std",
                    "f1_weighted_mean",
                    "f1_weighted_std",
                    "fit_time_mean",
                ],
            )
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parent
    outdir = root / "benchmark_results_new"

    # If invoked as a subprocess for a single dataset, run and return.
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-single-dataset":
        task = sys.argv[2]
        name = sys.argv[3]
        _run_single_dataset_small(task, name, outdir)
        return

    # Driver: run each dataset in an isolated subprocess; skip those that crash.
    # Regression
    for name in regression_dataset_names:
        cmd = [
            sys.executable,
            __file__,
            "--run-single-dataset",
            "regression",
            name,
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[regression] skip {name}: subprocess error {e}", file=sys.stderr)
            continue
        if proc.returncode == SIGSEGV_EXIT:
            print(f"[regression] skip {name}: SIGSEGV", file=sys.stderr)
            continue
        if proc.returncode != 0:
            print(f"[regression] skip {name}: returncode {proc.returncode}, stderr={proc.stderr.decode().strip()}", file=sys.stderr)
            continue

    # Classification (gini, entropy)
    for task, crit in (("classification_gini", "gini"), ("classification_entropy", "entropy")):
        for name in classification_dataset_names:
            cmd = [
                sys.executable,
                __file__,
                "--run-single-dataset",
                task,
                name,
            ]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"[{task}] skip {name}: subprocess error {e}", file=sys.stderr)
                continue
            if proc.returncode == SIGSEGV_EXIT:
                print(f"[{task}] skip {name}: SIGSEGV", file=sys.stderr)
                continue
            if proc.returncode != 0:
                print(f"[{task}] skip {name}: returncode {proc.returncode}, stderr={proc.stderr.decode().strip()}", file=sys.stderr)
                continue


if __name__ == "__main__":
    main()

