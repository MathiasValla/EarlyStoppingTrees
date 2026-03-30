#!/usr/bin/env python
"""
Merge a preserved benchmark archive with refreshed shard directories.

Typical use in this project:
- preserve the old 100-run archive for unchanged splitter families,
- rerun only changed/new families in sharded outdirs,
- merge both sources back into one benchmark_results directory.

This script merges run-level CSVs and rewrites aggregated summary CSVs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


REGRESSION_PREFIX = "regression"
CLASSIFICATION_PREFIXES = ("classification_gini", "classification_entropy")
EFFORT_FIELDS = [
    "split_calls_mean",
    "threshold_candidates_mean",
    "gain_evaluations_mean",
    "threshold_candidates_per_split_mean",
    "gain_evaluations_per_split_mean",
    "parametric_gain_samples_mean",
    "parametric_quantile_fits_mean",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _splitters_filter(rows: list[dict[str, str]], splitters: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("splitter", "") in splitters]


def _run_file(path_dir: Path, prefix: str, run_idx: int) -> Path:
    return path_dir / f"{prefix}_run{run_idx:03d}.csv"


def _load_shard_rows(shard_dirs: list[Path], prefix: str) -> list[list[dict[str, str]]]:
    per_run_rows: list[list[dict[str, str]]] = []
    for shard_dir in shard_dirs:
        run_files = sorted(shard_dir.glob(f"{prefix}_run*.csv"))
        for run_file in run_files:
            per_run_rows.append(_read_rows(run_file))
    return per_run_rows


def _enforce_common_classification_datasets(
    gini_rows: list[dict[str, str]],
    entropy_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    gini_names = {row["dataset"] for row in gini_rows}
    entropy_names = {row["dataset"] for row in entropy_rows}
    common = gini_names & entropy_names
    return (
        [row for row in gini_rows if row["dataset"] in common],
        [row for row in entropy_rows if row["dataset"] in common],
    )


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _aggregate_regression(indir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(indir.glob("regression_run*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "variant" not in df.columns:
        df["variant"] = ""
    df["variant"] = df["variant"].fillna("")
    metric_cols = ["rmse_mean", "rmse_std", "fit_time_mean", *EFFORT_FIELDS]
    df = _to_numeric(df, metric_cols)
    agg = (
        df.groupby(["dataset", "n_samples", "n_features", "splitter", "variant"], dropna=False)
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            fit_time_mean=("fit_time_mean", "mean"),
            fit_time_std=("fit_time_mean", "std"),
            n_runs=("rmse_mean", "size"),
            **{field: (field, "mean") for field in EFFORT_FIELDS if field in df.columns},
        )
        .reset_index()
        .sort_values(["dataset", "splitter", "variant"])
    )
    agg["rmse_std"] = agg["rmse_std"].fillna(0.0)
    agg["fit_time_std"] = agg["fit_time_std"].fillna(0.0)
    for field in EFFORT_FIELDS:
        if field not in agg.columns:
            agg[field] = math.nan
    return agg


def _aggregate_classification(indir: Path, criterion: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(indir.glob(f"classification_{criterion}_run*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "variant" not in df.columns:
        df["variant"] = ""
    df["variant"] = df["variant"].fillna("")
    metric_cols = [
        "accuracy_mean",
        "accuracy_std",
        "f1_weighted_mean",
        "f1_weighted_std",
        "fit_time_mean",
        *EFFORT_FIELDS,
    ]
    df = _to_numeric(df, metric_cols)
    agg = (
        df.groupby(
            ["dataset", "n_samples", "n_features", "criterion", "splitter", "variant"],
            dropna=False,
        )
        .agg(
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_std=("accuracy_mean", "std"),
            f1_weighted_mean=("f1_weighted_mean", "mean"),
            f1_weighted_std=("f1_weighted_mean", "std"),
            fit_time_mean=("fit_time_mean", "mean"),
            fit_time_std=("fit_time_mean", "std"),
            n_runs=("accuracy_mean", "size"),
            **{field: (field, "mean") for field in EFFORT_FIELDS if field in df.columns},
        )
        .reset_index()
        .sort_values(["dataset", "splitter", "variant"])
    )
    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    agg["f1_weighted_std"] = agg["f1_weighted_std"].fillna(0.0)
    agg["fit_time_std"] = agg["fit_time_std"].fillna(0.0)
    for field in EFFORT_FIELDS:
        if field not in agg.columns:
            agg[field] = math.nan
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge preserved and refreshed benchmark archives")
    parser.add_argument("--base-archive", type=str, required=True)
    parser.add_argument("--refreshed-reg-dirs", type=str, required=True, help="Comma-separated shard directories")
    parser.add_argument("--refreshed-clf-dirs", type=str, required=True, help="Comma-separated shard directories")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--base-splitters",
        type=str,
        default="secretary,secretary_par,block_rank,prophet_1sample",
    )
    parser.add_argument(
        "--refreshed-splitters",
        type=str,
        default="best,secretary_all,double_secretary,extra_tree",
    )
    args = parser.parse_args()

    base_archive = Path(args.base_archive)
    refreshed_reg_dirs = [Path(part) for part in args.refreshed_reg_dirs.split(",") if part]
    refreshed_clf_dirs = [Path(part) for part in args.refreshed_clf_dirs.split(",") if part]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_splitters = {part.strip() for part in args.base_splitters.split(",") if part.strip()}
    refreshed_splitters = {part.strip() for part in args.refreshed_splitters.split(",") if part.strip()}

    refreshed_reg_runs = _load_shard_rows(refreshed_reg_dirs, REGRESSION_PREFIX)
    refreshed_gini_runs = _load_shard_rows(refreshed_clf_dirs, "classification_gini")
    refreshed_entropy_runs = _load_shard_rows(refreshed_clf_dirs, "classification_entropy")

    n_runs = max(
        len(sorted(base_archive.glob("regression_run*.csv"))),
        len(refreshed_reg_runs),
    )
    if not (len(refreshed_reg_runs) == len(refreshed_gini_runs) == len(refreshed_entropy_runs) == n_runs):
        raise ValueError(
            "Refreshed shard runs do not align with the base archive run count: "
            f"base={n_runs}, reg={len(refreshed_reg_runs)}, gini={len(refreshed_gini_runs)}, "
            f"entropy={len(refreshed_entropy_runs)}"
        )

    for run_idx in range(1, n_runs + 1):
        base_reg_rows = _splitters_filter(
            _read_rows(_run_file(base_archive, REGRESSION_PREFIX, run_idx)),
            base_splitters,
        )
        merged_reg = base_reg_rows + _splitters_filter(refreshed_reg_runs[run_idx - 1], refreshed_splitters)
        _write_rows(_run_file(outdir, REGRESSION_PREFIX, run_idx), merged_reg)

        base_gini_rows = _splitters_filter(
            _read_rows(_run_file(base_archive, "classification_gini", run_idx)),
            base_splitters,
        )
        base_entropy_rows = _splitters_filter(
            _read_rows(_run_file(base_archive, "classification_entropy", run_idx)),
            base_splitters,
        )
        merged_gini = base_gini_rows + _splitters_filter(refreshed_gini_runs[run_idx - 1], refreshed_splitters)
        merged_entropy = base_entropy_rows + _splitters_filter(refreshed_entropy_runs[run_idx - 1], refreshed_splitters)
        merged_gini, merged_entropy = _enforce_common_classification_datasets(merged_gini, merged_entropy)
        _write_rows(_run_file(outdir, "classification_gini", run_idx), merged_gini)
        _write_rows(_run_file(outdir, "classification_entropy", run_idx), merged_entropy)

    reg_agg = _aggregate_regression(outdir)
    reg_agg.to_csv(outdir / "regression_results.csv", index=False)

    gini_agg = _aggregate_classification(outdir, "gini")
    entropy_agg = _aggregate_classification(outdir, "entropy")
    if not gini_agg.empty and not entropy_agg.empty:
        common = set(gini_agg["dataset"]) & set(entropy_agg["dataset"])
        gini_agg = gini_agg[gini_agg["dataset"].isin(common)].copy()
        entropy_agg = entropy_agg[entropy_agg["dataset"].isin(common)].copy()
    gini_agg.to_csv(outdir / "classification_gini_results.csv", index=False)
    entropy_agg.to_csv(outdir / "classification_entropy_results.csv", index=False)

    print(f"Merged {n_runs} runs into {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
