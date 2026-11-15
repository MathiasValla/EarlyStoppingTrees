"""Focused bagging validation for split-search strategies.

This experiment answers whether node-level behavior persists after aggregation.
It is intentionally smaller than the single-tree benchmark: a deterministic,
size-stratified subset of retained PMLB entries, five estimator seeds, five
fixed folds, and 25 bootstrap trees per ensemble.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from pmlb import fetch_data
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, recall_score
from sklearn.model_selection import check_cv

import treeple
from treeple.tree import (
    EarlyStopDecisionTreeClassifier,
    EarlyStopDecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


N_FOLDS = 5
DEFAULT_SEED = 42


def _ordered_methods(methods, dataset: str, seed: int, task: str):
    """Rotate method order deterministically across dataset--seed blocks."""
    items = list(methods.items())
    digest = hashlib.sha256(f"{dataset}|{seed}|{task}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little") % len(items)
    return items[offset:] + items[:offset]


def _classification_methods(seed: int):
    return {
        "exhaustive": EarlyStopDecisionTreeClassifier(
            splitter="best", criterion="gini", max_depth=20, random_state=seed
        ),
        "secretary_all_1overe": EarlyStopDecisionTreeClassifier(
            splitter="secretary_all", criterion="gini", max_depth=20, random_state=seed
        ),
        "random_feature_exact_1overe": EarlyStopDecisionTreeClassifier(
            splitter="best",
            criterion="gini",
            max_depth=20,
            max_features=1.0 / math.e,
            random_state=seed,
        ),
        "ert_all": ExtraTreeClassifier(
            criterion="gini", max_depth=20, max_features=None, random_state=seed
        ),
    }


def _regression_methods(seed: int):
    return {
        "exhaustive": EarlyStopDecisionTreeRegressor(
            splitter="best", max_depth=20, random_state=seed
        ),
        "secretary_all_1overe": EarlyStopDecisionTreeRegressor(
            splitter="secretary_all", max_depth=20, random_state=seed
        ),
        "random_feature_exact_1overe": EarlyStopDecisionTreeRegressor(
            splitter="best",
            max_depth=20,
            max_features=1.0 / math.e,
            random_state=seed,
        ),
        "ert_all": ExtraTreeRegressor(
            max_depth=20, max_features=None, random_state=seed
        ),
    }


def _load_dataset_names(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _write_metadata(path: Path, args):
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "invocation": list(sys.argv),
        "task": args.task,
        "n_estimators": args.n_estimators,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "n_folds": N_FOLDS,
        "max_product": args.max_product,
        "bootstrap": True,
        "n_jobs": 1,
        "method_order": "deterministic cyclic rotation by SHA-256(dataset, seed, task)",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "treeple_version": treeple.__version__,
        "treeple_import_path": str(Path(treeple.__file__).resolve()),
        "methods": [
            "exhaustive",
            "secretary_all_1overe",
            "random_feature_exact_1overe",
            "ert_all",
        ],
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _evaluate_classification(name, X, y, seed, n_estimators):
    rows = []
    for evaluation_position, (method, base_estimator) in enumerate(
        _ordered_methods(_classification_methods(seed), name, seed, "classification")
    ):
        fit_times = []
        accuracies = []
        f1_weighted = []
        f1_macro = []
        recall_min = []
        cv = check_cv(N_FOLDS, y=y, classifier=True)
        for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            classes = np.unique(y_train)
            class_to_col = {label: index for index, label in enumerate(classes)}
            proba_sum = np.zeros((X_test.shape[0], classes.size), dtype=float)
            rng = np.random.default_rng(seed * 10_000 + fold_index)
            fit_start = time.perf_counter()
            fitted = []
            for tree_index in range(n_estimators):
                bootstrap_idx = rng.integers(0, X_train.shape[0], X_train.shape[0])
                tree = clone(base_estimator).set_params(
                    random_state=seed * 100_000 + fold_index * n_estimators + tree_index
                )
                tree.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])
                fitted.append(tree)
            fit_times.append(time.perf_counter() - fit_start)
            for tree in fitted:
                tree_proba = tree.predict_proba(X_test)
                for source_col, label in enumerate(tree.classes_):
                    proba_sum[:, class_to_col[label]] += tree_proba[:, source_col]
            prediction = classes[np.argmax(proba_sum, axis=1)]
            accuracies.append(accuracy_score(y_test, prediction))
            f1_weighted.append(
                f1_score(y_test, prediction, average="weighted", zero_division=0)
            )
            f1_macro.append(
                f1_score(y_test, prediction, average="macro", zero_division=0)
            )
            recall_min.append(
                float(np.min(recall_score(y_test, prediction, average=None, zero_division=0)))
            )
        rows.append(
            {
                "dataset": name,
                "task": "classification_gini",
                "seed": seed,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "method": method,
                "n_estimators": n_estimators,
                "evaluation_position": evaluation_position,
                "fit_time_mean": float(np.mean(fit_times)),
                "accuracy_mean": float(np.mean(accuracies)),
                "f1_weighted_mean": float(np.mean(f1_weighted)),
                "f1_macro_mean": float(np.mean(f1_macro)),
                "recall_min_mean": float(np.mean(recall_min)),
            }
        )
    return rows


def _evaluate_regression(name, X, y, seed, n_estimators):
    rows = []
    for evaluation_position, (method, base_estimator) in enumerate(
        _ordered_methods(_regression_methods(seed), name, seed, "regression")
    ):
        fit_times = []
        rmses = []
        cv = check_cv(N_FOLDS, y=y, classifier=False)
        for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            rng = np.random.default_rng(seed * 10_000 + fold_index)
            fit_start = time.perf_counter()
            fitted = []
            for tree_index in range(n_estimators):
                bootstrap_idx = rng.integers(0, X_train.shape[0], X_train.shape[0])
                tree = clone(base_estimator).set_params(
                    random_state=seed * 100_000 + fold_index * n_estimators + tree_index
                )
                tree.fit(X_train[bootstrap_idx], y_train[bootstrap_idx])
                fitted.append(tree)
            fit_times.append(time.perf_counter() - fit_start)
            prediction = np.mean([tree.predict(X_test) for tree in fitted], axis=0)
            rmses.append(float(np.sqrt(mean_squared_error(y_test, prediction))))
        rows.append(
            {
                "dataset": name,
                "task": "regression",
                "seed": seed,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "method": method,
                "n_estimators": n_estimators,
                "evaluation_position": evaluation_position,
                "fit_time_mean": float(np.mean(fit_times)),
                "rmse_mean": float(np.mean(rmses)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--datasets-file", type=Path, required=True)
    parser.add_argument("--pmlb-cache-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=25)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-product", type=int, default=1_000_000)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_metadata(args.outdir / "metadata.json", args)
    rows = []
    exclusions = []
    names = _load_dataset_names(args.datasets_file)
    for dataset_index, name in enumerate(names, start=1):
        try:
            X, y = fetch_data(
                name, return_X_y=True, local_cache_dir=str(args.pmlb_cache_dir)
            )
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)
        except Exception as exc:
            exclusions.append({"dataset": name, "reason": f"load_error: {exc}"})
            continue
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            exclusions.append({"dataset": name, "reason": "non_finite_value"})
            continue
        if X.shape[0] * X.shape[1] > args.max_product:
            exclusions.append({"dataset": name, "reason": "max_product"})
            continue

        print(
            f"[{args.task}] {dataset_index}/{len(names)} {name} "
            f"(n={X.shape[0]}, p={X.shape[1]})",
            flush=True,
        )
        for seed in range(args.seed_start, args.seed_start + args.n_seeds):
            try:
                if args.task == "classification":
                    rows.extend(
                        _evaluate_classification(name, X, y, seed, args.n_estimators)
                    )
                else:
                    rows.extend(
                        _evaluate_regression(name, X, y, seed, args.n_estimators)
                    )
            except Exception as exc:
                exclusions.append(
                    {"dataset": name, "seed": seed, "reason": f"fit_error: {exc}"}
                )

    output_path = args.outdir / f"ensemble_{args.task}.csv"
    if rows:
        with output_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.outdir / "exclusions.json").write_text(
        json.dumps(exclusions, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
