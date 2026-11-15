"""Paired control benchmark requested during peer review.

This benchmark times the methods that are needed to isolate adaptive stopping:
exhaustive CART, ``S_all(1/e)``, exact CART over fixed random feature subsets,
and ERT.  Every method is fitted on the same folds and seeds.  A deterministic
cyclic rotation changes the evaluation order across dataset--seed blocks so
that no method systematically benefits from being run first or last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from pmlb import fetch_data
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_squared_error,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

import treeple
from treeple.tree import (
    EarlyStopDecisionTreeClassifier,
    EarlyStopDecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


N_FOLDS = 5
DEFAULT_SEED_START = 42


def _stable_offset(dataset: str, seed: int, task: str, n_methods: int) -> int:
    payload = f"{dataset}|{seed}|{task}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % n_methods


def _rotate(items, offset: int):
    items = list(items)
    if not items:
        return items
    offset %= len(items)
    return items[offset:] + items[:offset]


def _feature_budget(max_features, n_features: int) -> int:
    if max_features is None:
        return int(n_features)
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    return min(int(max_features), int(n_features))


def _regression_methods(seed: int):
    methods = [
        (
            "exhaustive",
            "all",
            None,
            EarlyStopDecisionTreeRegressor(
                splitter="best", max_depth=20, random_state=seed
            ),
        ),
        (
            "secretary_all",
            "1overe",
            None,
            EarlyStopDecisionTreeRegressor(
                splitter="secretary_all", max_depth=20, random_state=seed
            ),
        ),
    ]
    exact_budgets = (
        (1, "1"),
        (0.1, "0.1p"),
        (1.0 / math.e, "1overe"),
        (0.5, "0.5p"),
        (2.0 / 3.0, "2over3p"),
    )
    for budget, label in exact_budgets:
        methods.append(
            (
                "random_feature_exact",
                label,
                budget,
                EarlyStopDecisionTreeRegressor(
                    splitter="best",
                    max_features=budget,
                    max_depth=20,
                    random_state=seed,
                ),
            )
        )
    ert_budgets = ((1, "1"), (1.0 / 3.0, "1over3p"), (2.0 / 3.0, "2over3p"), (None, "all"))
    for budget, label in ert_budgets:
        methods.append(
            (
                "ert",
                label,
                budget,
                ExtraTreeRegressor(
                    max_features=budget, max_depth=20, random_state=seed
                ),
            )
        )
    return methods


def _classification_methods(seed: int, criterion: str):
    methods = [
        (
            "exhaustive",
            "all",
            None,
            EarlyStopDecisionTreeClassifier(
                splitter="best",
                criterion=criterion,
                max_depth=20,
                random_state=seed,
            ),
        ),
        (
            "secretary_all",
            "1overe",
            None,
            EarlyStopDecisionTreeClassifier(
                splitter="secretary_all",
                criterion=criterion,
                max_depth=20,
                random_state=seed,
            ),
        ),
    ]
    exact_budgets = (
        (1, "1"),
        (0.1, "0.1p"),
        (1.0 / math.e, "1overe"),
        (0.5, "0.5p"),
        (2.0 / 3.0, "2over3p"),
    )
    for budget, label in exact_budgets:
        methods.append(
            (
                "random_feature_exact",
                label,
                budget,
                EarlyStopDecisionTreeClassifier(
                    splitter="best",
                    criterion=criterion,
                    max_features=budget,
                    max_depth=20,
                    random_state=seed,
                ),
            )
        )
    ert_budgets = ((1, "1"), (1.0 / 3.0, "1over3p"), (2.0 / 3.0, "2over3p"), (None, "all"))
    for budget, label in ert_budgets:
        methods.append(
            (
                "ert",
                label,
                budget,
                ExtraTreeClassifier(
                    criterion=criterion,
                    max_features=budget,
                    max_depth=20,
                    random_state=seed,
                ),
            )
        )
    return methods


def _worst_class_recall(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return float(np.min(recalls)) if recalls.size else 0.0


CLASSIFICATION_SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "recall_min": make_scorer(_worst_class_recall),
}


def _evaluate_regression(dataset, X, y, seed, method_order):
    cv = list(KFold(N_FOLDS, shuffle=False).split(X, y))
    scoring = make_scorer(
        lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=True,
    )
    rows = []
    for position, (method, variant, budget, estimator) in enumerate(method_order):
        result = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )
        scores = -np.asarray(result["test_score"], dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "task": "regression",
                "criterion": "squared_error",
                "seed": seed,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "method": method,
                "variant": variant,
                "feature_budget": (
                    X.shape[1] if method in {"exhaustive", "secretary_all"}
                    else _feature_budget(budget, X.shape[1])
                ),
                "evaluation_position": position,
                "fit_time_mean": float(np.mean(result["fit_time"])),
                "fit_time_std": float(np.std(result["fit_time"])),
                "rmse_mean": float(np.mean(scores)),
                "rmse_std": float(np.std(scores)),
            }
        )
    return rows


def _evaluate_classification(dataset, X, y, seed, criterion, method_order):
    cv = list(StratifiedKFold(N_FOLDS, shuffle=False).split(X, y))
    rows = []
    for position, (method, variant, budget, estimator) in enumerate(method_order):
        result = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=CLASSIFICATION_SCORING,
            return_train_score=False,
            n_jobs=1,
        )
        row = {
            "dataset": dataset,
            "task": f"classification_{criterion}",
            "criterion": criterion,
            "seed": seed,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "method": method,
            "variant": variant,
            "feature_budget": (
                X.shape[1] if method in {"exhaustive", "secretary_all"}
                else _feature_budget(budget, X.shape[1])
            ),
            "evaluation_position": position,
            "fit_time_mean": float(np.mean(result["fit_time"])),
            "fit_time_std": float(np.std(result["fit_time"])),
        }
        for metric in CLASSIFICATION_SCORING:
            values = np.asarray(result[f"test_{metric}"], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return rows


def _load_names(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _write_metadata(path: Path, args):
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "invocation": list(sys.argv),
        "task": args.task,
        "criteria": args.criteria,
        "run_start": args.run_start,
        "run_end": args.run_end,
        "seed_start": args.seed_start,
        "n_folds": N_FOLDS,
        "max_product": args.max_product,
        "method_order": "deterministic cyclic rotation by SHA-256(dataset, seed, task)",
        "timing_scope": "estimator fit_time returned by sklearn.model_selection.cross_validate",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "treeple_version": treeple.__version__,
        "treeple_import_path": str(Path(treeple.__file__).resolve()),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("regression", "classification"), required=True)
    parser.add_argument("--datasets-file", type=Path, required=True)
    parser.add_argument("--pmlb-cache-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--run-start", type=int, default=1)
    parser.add_argument("--run-end", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--criteria", default="gini,entropy")
    parser.add_argument("--max-product", type=int, default=1_000_000)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_metadata(args.outdir / "metadata.json", args)
    names = _load_names(args.datasets_file)
    criteria = [item.strip() for item in args.criteria.split(",") if item.strip()]
    exclusions = []

    for run in range(args.run_start, args.run_end + 1):
        seed = args.seed_start + run - 1
        output = args.outdir / f"{args.task}_run{run:03d}.csv"
        if output.is_file():
            print(f"Skipping existing {output.name}", flush=True)
            continue
        rows = []
        for dataset_index, dataset in enumerate(names, start=1):
            try:
                X, y = fetch_data(
                    dataset,
                    return_X_y=True,
                    local_cache_dir=str(args.pmlb_cache_dir),
                )
                X = np.asarray(X, dtype=np.float64)
                y = np.asarray(y)
            except Exception as exc:
                exclusions.append(
                    {"run": run, "dataset": dataset, "reason": f"load_error: {exc}"}
                )
                continue
            if not np.isfinite(X).all() or pd.isna(y).any():
                exclusions.append(
                    {"run": run, "dataset": dataset, "reason": "non_finite_value"}
                )
                continue
            if args.max_product and X.shape[0] * X.shape[1] > args.max_product:
                exclusions.append(
                    {"run": run, "dataset": dataset, "reason": "max_product"}
                )
                continue

            print(
                f"[{args.task}] run {run}/{args.run_end}, "
                f"dataset {dataset_index}/{len(names)}: {dataset}",
                flush=True,
            )
            try:
                if args.task == "regression":
                    methods = _regression_methods(seed)
                    offset = _stable_offset(dataset, seed, "regression", len(methods))
                    rows.extend(
                        _evaluate_regression(dataset, X, y, seed, _rotate(methods, offset))
                    )
                else:
                    for criterion in criteria:
                        methods = _classification_methods(seed, criterion)
                        task = f"classification_{criterion}"
                        offset = _stable_offset(dataset, seed, task, len(methods))
                        rows.extend(
                            _evaluate_classification(
                                dataset,
                                X,
                                y,
                                seed,
                                criterion,
                                _rotate(methods, offset),
                            )
                        )
            except Exception as exc:
                exclusions.append(
                    {"run": run, "dataset": dataset, "reason": f"fit_error: {exc}"}
                )

        if rows:
            temporary = output.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(output)
            print(f"Wrote {output} ({len(rows)} rows)", flush=True)

    (args.outdir / "exclusions.json").write_text(
        json.dumps(exclusions, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
