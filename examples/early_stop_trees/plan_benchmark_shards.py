#!/usr/bin/env python
"""
Plan cost-balanced benchmark shards for the full PMLB rerun.

The planner uses:
- observed per-dataset benchmark cost from a previous archive when available
- cached dataset shapes from PMLB for the runnable dataset universe
- a kNN fallback on (log n, log p) for datasets with no observed cost
- proportional task-level shard allocation
- longest-processing-time greedy packing within each task

Outputs:
- one manifest txt file per shard (dataset names)
- a JSON summary with predicted loads and counts
- a CSV with the per-dataset cost model used for planning
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pmlb import (
        classification_dataset_names,
        fetch_data,
        regression_dataset_names,
    )
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"Install pmlb first: {e}")


SKIP_DATASETS = {"192_vineyard", "687_sleuth_ex1605"}
KNN_NEIGHBORS = 7


def _raw_dataset_names() -> dict[str, list[str]]:
    return {
        "regression": [
            n for n in regression_dataset_names if not n.startswith("_deprecated_") and n not in SKIP_DATASETS
        ],
        "classification": [
            n for n in classification_dataset_names if not n.startswith("_deprecated_") and n not in SKIP_DATASETS
        ],
    }


def _load_shapes(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for task, names in _raw_dataset_names().items():
        for name in names:
            try:
                X, _ = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
            except Exception:
                continue
            X = np.asarray(X)
            rows.append(
                {
                    "task": task,
                    "dataset": name,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                }
            )
    if not rows:
        raise RuntimeError(f"No dataset shapes could be loaded from {cache_dir}")
    return pd.DataFrame(rows)


def _load_observed_costs(indir: Path) -> pd.DataFrame:
    reg = (
        pd.read_csv(indir / "regression_results.csv")
        .groupby("dataset", as_index=False)["fit_time_mean"]
        .sum()
        .rename(columns={"fit_time_mean": "cost"})
    )
    reg["task"] = "regression"

    gini = (
        pd.read_csv(indir / "classification_gini_results.csv")
        .groupby("dataset", as_index=False)["fit_time_mean"]
        .sum()
        .rename(columns={"fit_time_mean": "gini_cost"})
    )
    entropy = (
        pd.read_csv(indir / "classification_entropy_results.csv")
        .groupby("dataset", as_index=False)["fit_time_mean"]
        .sum()
        .rename(columns={"fit_time_mean": "entropy_cost"})
    )
    clf = gini.merge(entropy, on="dataset", how="outer").fillna(0.0)
    clf["cost"] = clf["gini_cost"] + clf["entropy_cost"]
    clf = clf[["dataset", "cost"]]
    clf["task"] = "classification"
    return pd.concat([reg, clf], ignore_index=True)


def _predict_missing_costs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_source"] = np.where(out["cost"].notna(), "observed", "predicted")
    for task in ("regression", "classification"):
        known = out[(out["task"] == task) & out["cost"].notna()].copy()
        missing = out[(out["task"] == task) & out["cost"].isna()].copy()
        if missing.empty:
            continue
        x_known = np.column_stack(
            [
                np.log1p(known["n_samples"].to_numpy(dtype=float)),
                np.log1p(known["n_features"].to_numpy(dtype=float)),
            ]
        )
        y_known = known["cost"].to_numpy(dtype=float)
        x_missing = np.column_stack(
            [
                np.log1p(missing["n_samples"].to_numpy(dtype=float)),
                np.log1p(missing["n_features"].to_numpy(dtype=float)),
            ]
        )
        predictions = []
        for row in x_missing:
            d2 = np.sum((x_known - row) ** 2, axis=1)
            idx = np.argsort(d2)[: min(KNN_NEIGHBORS, len(d2))]
            weights = 1.0 / (np.sqrt(d2[idx]) + 1e-6)
            predictions.append(float(np.average(y_known[idx], weights=weights)))
        out.loc[missing.index, "cost"] = predictions
    return out


def _allocate_task_shards(cost_df: pd.DataFrame, total_shards: int) -> dict[str, int]:
    task_cost = cost_df.groupby("task")["cost"].sum()
    reg_cost = float(task_cost.get("regression", 0.0))
    clf_cost = float(task_cost.get("classification", 0.0))
    if reg_cost <= 0 or clf_cost <= 0:
        raise RuntimeError("Both regression and classification must have positive predicted cost.")

    reg_share = reg_cost / (reg_cost + clf_cost)
    reg_shards = int(round(total_shards * reg_share))
    reg_shards = max(1, min(total_shards - 1, reg_shards))
    clf_shards = total_shards - reg_shards

    reg_n = int((cost_df["task"] == "regression").sum())
    clf_n = int((cost_df["task"] == "classification").sum())
    reg_shards = min(reg_shards, reg_n)
    clf_shards = min(clf_shards, clf_n)
    if reg_shards + clf_shards < total_shards:
        spare = total_shards - reg_shards - clf_shards
        if clf_n - clf_shards >= reg_n - reg_shards:
            clf_shards += spare
        else:
            reg_shards += spare
    return {"regression": reg_shards, "classification": clf_shards}


def _pack_task(cost_df: pd.DataFrame, n_shards: int) -> list[dict]:
    rows = cost_df.sort_values(["cost", "n_samples", "n_features"], ascending=False).to_dict(orient="records")
    loads = [0.0] * n_shards
    bins: list[list[dict]] = [[] for _ in range(n_shards)]
    for row in rows:
        target = min(range(n_shards), key=lambda i: loads[i])
        bins[target].append(row)
        loads[target] += float(row["cost"])
    packed = []
    for i, (load, items) in enumerate(zip(loads, bins), start=1):
        packed.append(
            {
                "shard_index": i,
                "predicted_cost": float(load),
                "dataset_count": len(items),
                "datasets": [item["dataset"] for item in items],
                "top_datasets": [
                    {
                        "dataset": item["dataset"],
                        "predicted_cost": float(item["cost"]),
                        "n_samples": int(item["n_samples"]),
                        "n_features": int(item["n_features"]),
                        "cost_source": item["cost_source"],
                    }
                    for item in items[:10]
                ],
            }
        )
    return packed


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan cost-balanced benchmark shards.")
    parser.add_argument(
        "--benchmark-indir",
        type=str,
        default=None,
        help="Directory containing regression_results.csv and classification_*_results.csv",
    )
    parser.add_argument(
        "--pmlb-cache-dir",
        type=str,
        default=None,
        help="Cache directory used to load runnable dataset shapes",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=None,
        help="Total number of benchmark shards to plan (default: os.cpu_count())",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory where shard manifests and summaries will be written",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    benchmark_indir = Path(args.benchmark_indir) if args.benchmark_indir else (script_dir / "benchmark_results")
    cache_dir = Path(args.pmlb_cache_dir) if args.pmlb_cache_dir else (benchmark_indir / "pmlb_cache")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    total_shards = int(args.total_shards or os.cpu_count() or 1)
    total_shards = max(2, total_shards)

    shapes = _load_shapes(cache_dir)
    observed = _load_observed_costs(benchmark_indir)
    full = shapes.merge(observed, on=["task", "dataset"], how="left")
    full = _predict_missing_costs(full)
    full = full.sort_values(["task", "cost", "dataset"], ascending=[True, False, True]).reset_index(drop=True)

    shard_counts = _allocate_task_shards(full, total_shards)
    plans = {
        task: _pack_task(full[full["task"] == task].copy(), n_shards)
        for task, n_shards in shard_counts.items()
    }

    for task, shards in plans.items():
        prefix = "reg" if task == "regression" else "clf"
        for shard in shards:
            path = outdir / f"{prefix}_{shard['shard_index']:02d}.txt"
            path.write_text("\n".join(shard["datasets"]) + "\n")

    full.to_csv(outdir / "dataset_costs.csv", index=False)

    summary = {
        "total_shards": total_shards,
        "shard_counts": shard_counts,
        "task_costs": {
            task: float(full.loc[full["task"] == task, "cost"].sum()) for task in ("regression", "classification")
        },
        "task_dataset_counts": {
            task: int((full["task"] == task).sum()) for task in ("regression", "classification")
        },
        "unavailable_dataset_counts": {
            task: len(_raw_dataset_names()[task]) - int((shapes["task"] == task).sum())
            for task in ("regression", "classification")
        },
        "plans": plans,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["shard_counts"], indent=2))
    for task, shards in plans.items():
        loads = [shard["predicted_cost"] for shard in shards]
        imbalance = 0.0
        if loads:
            imbalance = (max(loads) - min(loads)) / max(np.mean(loads), 1e-12)
        print(
            f"{task}: loads={[round(x, 2) for x in loads]} "
            f"counts={[shard['dataset_count'] for shard in shards]} "
            f"imbalance_pct={100.0 * imbalance:.1f}"
        )
    print(f"Wrote shard manifests and summary to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
