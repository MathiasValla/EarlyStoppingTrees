#!/usr/bin/env python
"""
Aggregate PMLB benchmark CSVs: mean RMSE / accuracy / F1 and mean time per splitter across datasets.

Reads from benchmark_results/ (or --indir):
  - regression_results.csv
  - classification_gini_results.csv
  - classification_entropy_results.csv

Writes to the same directory:
  - regression_aggregated.csv   (splitter, rmse_mean, rmse_std, fit_time_mean, fit_time_std, n_datasets)
  - classification_aggregated.csv (criterion, splitter, accuracy_mean, accuracy_std, f1_weighted_mean, f1_weighted_std, fit_time_mean, fit_time_std, n_datasets)

Usage:
  python examples/early_stop_trees/aggregate_benchmark_results.py [--indir DIR]
"""
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def aggregate_regression(indir: Path) -> Optional[Path]:
    path = indir / "regression_results.csv"
    if not path.exists():
        print(f"Skip regression: {path} not found.", file=sys.stderr)
        return None
    # key = (splitter, variant)
    by_key: Dict[tuple, Dict[str, Any]] = defaultdict(lambda: {"rmse": [], "time": [], "datasets": set()})
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            splitter = row.get("splitter")
            if splitter is None:
                continue
            variant = row.get("variant", "")
            key = (splitter, variant)
            try:
                by_key[key]["rmse"].append(float(row["rmse_mean"]))
                by_key[key]["time"].append(float(row["fit_time_mean"]))
                by_key[key]["datasets"].add(row["dataset"])
            except (KeyError, ValueError):
                continue
    rows = []
    for splitter, variant in sorted(by_key.keys()):
        d = by_key[(splitter, variant)]
        n = len(d["datasets"])
        rows.append({
            "splitter": splitter,
            "variant": variant,
            "rmse_mean": _mean(d["rmse"]),
            "rmse_std": _std(d["rmse"]),
            "fit_time_mean": _mean(d["time"]),
            "fit_time_std": _std(d["time"]),
            "n_datasets": n,
        })
    out = indir / "regression_aggregated.csv"
    fieldnames = ["splitter", "variant", "rmse_mean", "rmse_std", "fit_time_mean", "fit_time_std", "n_datasets"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    return out


def aggregate_classification(indir: Path) -> Optional[Path]:
    gini_path = indir / "classification_gini_results.csv"
    entropy_path = indir / "classification_entropy_results.csv"
    if not gini_path.exists() and not entropy_path.exists():
        print("Skip classification: no classification_*_results.csv found.", file=sys.stderr)
        return None
    # key = (criterion, splitter, variant)
    by_key: Dict[tuple, Dict[str, Any]] = defaultdict(
        lambda: {"accuracy": [], "f1": [], "time": [], "datasets": set()}
    )
    for path in (gini_path, entropy_path):
        if not path.exists():
            continue
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                # New small-benchmark CSVs encode the task in a 'task' column
                # (e.g. 'classification_gini', 'classification_entropy') instead
                # of a separate 'criterion' column. Fall back to that when
                # 'criterion' is absent.
                criterion = row.get("criterion")
                if criterion is None:
                    # Split 'classification_gini' -> 'gini', 'classification_entropy' -> 'entropy'
                    task = row.get("task", "")
                    if task.startswith("classification_"):
                        criterion = task.split("classification_")[-1]
                    else:
                        criterion = task
                splitter = row.get("splitter")
                if splitter is None:
                    continue
                variant = row.get("variant", "")
                key = (criterion, splitter, variant)
                try:
                    by_key[key]["accuracy"].append(float(row["accuracy_mean"]))
                    by_key[key]["f1"].append(float(row["f1_weighted_mean"]))
                    by_key[key]["time"].append(float(row["fit_time_mean"]))
                    by_key[key]["datasets"].add(row["dataset"])
                except (KeyError, ValueError):
                    continue
    rows = []
    for (criterion, splitter, variant) in sorted(by_key.keys()):
        d = by_key[(criterion, splitter, variant)]
        n = len(d["datasets"])
        rows.append({
            "criterion": criterion,
            "splitter": splitter,
            "variant": variant,
            "accuracy_mean": _mean(d["accuracy"]),
            "accuracy_std": _std(d["accuracy"]),
            "f1_weighted_mean": _mean(d["f1"]),
            "f1_weighted_std": _std(d["f1"]),
            "fit_time_mean": _mean(d["time"]),
            "fit_time_std": _std(d["time"]),
            "n_datasets": n,
        })
    out = indir / "classification_aggregated.csv"
    fieldnames = [
        "criterion", "splitter", "variant",
        "accuracy_mean", "accuracy_std", "f1_weighted_mean", "f1_weighted_std",
        "fit_time_mean", "fit_time_std", "n_datasets",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate PMLB benchmark CSVs per splitter")
    p.add_argument(
        "--indir",
        type=str,
        default=None,
        help="Input directory with regression_results.csv and classification_*_results.csv (default: examples/early_stop_trees/benchmark_results)",
    )
    args = p.parse_args()
    script_dir = Path(__file__).resolve().parent
    indir = Path(args.indir) if args.indir else script_dir / "benchmark_results"
    if not indir.is_dir():
        print(f"Not a directory: {indir}", file=sys.stderr)
        sys.exit(1)
    aggregate_regression(indir)
    aggregate_classification(indir)
    print("Done.")


if __name__ == "__main__":
    main()
