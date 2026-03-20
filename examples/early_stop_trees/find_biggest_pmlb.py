#!/usr/bin/env python
"""
Find the PMLB dataset for which n_samples * n_features is maximum.

Usage:
  python examples/early_stop_trees/find_biggest_pmlb.py [--outdir DIR] [--max-datasets N]

Prints the dataset name, task (regression/classification/both), n_samples, n_features, product.
Uses outdir/pmlb_cache for fetch cache. Optional --max-datasets limits how many to scan (for quick tests).
"""
import argparse
import sys
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


def main():
    p = argparse.ArgumentParser(description="Find PMLB dataset with max n_samples * n_features")
    p.add_argument("--outdir", type=str, default=None, help="Directory for pmlb cache (default: examples/early_stop_trees/benchmark_results)")
    p.add_argument("--max-datasets", type=int, default=None, help="Limit number of datasets to scan (default: all)")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if args.outdir else script_dir / "benchmark_results"
    cache_dir = str(outdir / "pmlb_cache")

    reg_set = {n for n in regression_dataset_names if not n.startswith("_deprecated_")}
    clf_set = {n for n in classification_dataset_names if not n.startswith("_deprecated_")}
    all_names = sorted(reg_set | clf_set)
    if args.max_datasets is not None:
        all_names = all_names[: args.max_datasets]

    best_name = None
    best_product = -1
    best_n_samples = 0
    best_n_features = 0
    best_tasks = []

    for i, name in enumerate(all_names):
        try:
            X, y = fetch_data(name, return_X_y=True, local_cache_dir=cache_dir)
        except Exception as e:
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        X = np.asarray(X, dtype=np.float64)
        if np.any(np.isnan(X)):
            print(f"[skip] {name}: contains NaN", file=sys.stderr)
            continue
        n_samples, n_features = X.shape
        product = n_samples * n_features
        tasks = []
        if name in reg_set:
            tasks.append("regression")
        if name in clf_set:
            tasks.append("classification")
        if product > best_product:
            best_product = product
            best_name = name
            best_n_samples = n_samples
            best_n_features = n_features
            best_tasks = tasks
        print(f"[{i+1}/{len(all_names)}] {name}: n={n_samples}, p={n_features}, product={product}", flush=True)

    if best_name is None:
        print("No valid dataset found.", file=sys.stderr)
        sys.exit(1)
    print()
    print(f"Biggest (rows × features): {best_name}")
    print(f"  task(s): {', '.join(best_tasks)}")
    print(f"  n_samples = {best_n_samples}, n_features = {best_n_features}")
    print(f"  product = {best_product}")
    print()
    print(f"Run benchmark on it:")
    print(f"  python examples/early_stop_trees/benchmark_secretary_pmlb.py --dataset {best_name}")


if __name__ == "__main__":
    main()
