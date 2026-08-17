#!/usr/bin/env python
"""Generate benchmark dataset descriptors from the retained run archive.

The main table reports one row for regression and one for classification. Gini
and entropy are not duplicated because they use the same classification tasks.
A machine-readable inventory keeps one row per retained dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_results_utils import load_all


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
TABLES_DIR = SCRIPT_DIR / "tables"


def _dataset_inventory(run_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return one row per retained dataset using only archived run metadata."""
    required = {"dataset", "run", "n_samples", "n_features"}
    missing = required.difference(run_df.columns)
    if missing:
        raise ValueError(f"Missing required run columns: {sorted(missing)}")

    grouped = run_df.groupby("dataset", sort=True)
    inventory = grouped.agg(
        n_samples=("n_samples", "first"),
        n_features=("n_features", "first"),
        n_valid_runs=("run", "nunique"),
    ).reset_index()
    inventory.insert(1, "task", task)
    inventory["n_times_p"] = inventory["n_samples"] * inventory["n_features"]
    inventory["p_over_n"] = inventory["n_features"] / inventory["n_samples"]
    return inventory


def _add_target_metadata(inventory: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Add task-specific target descriptors from the local PMLB cache."""
    try:
        from pmlb import fetch_data
    except ImportError as exc:
        raise RuntimeError("pmlb is required to generate target descriptors") from exc

    rows = []
    for row in inventory.itertuples(index=False):
        try:
            _, y = fetch_data(
                row.dataset,
                return_X_y=True,
                local_cache_dir=str(cache_dir),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load retained dataset {row.dataset!r} from {cache_dir}"
            ) from exc

        y = np.asarray(y)
        values, counts = np.unique(y, return_counts=True)
        metadata = {
            "dataset": row.dataset,
            "n_unique_target": int(values.size),
            "n_classes": np.nan,
            "majority_class_proportion": np.nan,
            "imbalance_ratio": np.nan,
            "target_std": np.nan,
        }
        if row.task == "Classification":
            metadata["n_classes"] = int(values.size)
            metadata["majority_class_proportion"] = float(counts.max() / counts.sum())
            metadata["imbalance_ratio"] = float(counts.max() / counts.min())
        else:
            metadata["target_std"] = float(np.std(y.astype(float), ddof=0))
        rows.append(metadata)

    return inventory.merge(pd.DataFrame(rows), on="dataset", how="left", validate="one_to_one")


def _quantile_summary(inventory: pd.DataFrame, task: str) -> dict:
    """Summarize structural descriptors with equal weight per dataset."""
    row = {"task": task, "n_datasets": int(inventory["dataset"].nunique())}
    for source, label in (
        ("n_samples", "n"),
        ("n_features", "p"),
        ("n_times_p", "n_times_p"),
        ("p_over_n", "p_over_n"),
        ("n_valid_runs", "valid_runs"),
    ):
        values = pd.to_numeric(inventory[source], errors="coerce").dropna()
        row.update(
            {
                f"min_{label}": float(values.min()),
                f"q1_{label}": float(values.quantile(0.25)),
                f"median_{label}": float(values.median()),
                f"q3_{label}": float(values.quantile(0.75)),
                f"max_{label}": float(values.max()),
            }
        )

    for source, label in (
        ("n_classes", "classes"),
        ("majority_class_proportion", "majority_class_proportion"),
        ("imbalance_ratio", "imbalance_ratio"),
        ("target_std", "target_std"),
    ):
        values = pd.to_numeric(inventory[source], errors="coerce").dropna()
        if values.empty:
            for stat in ("min", "q1", "median", "q3", "max"):
                row[f"{stat}_{label}"] = np.nan
            continue
        row.update(
            {
                f"min_{label}": float(values.min()),
                f"q1_{label}": float(values.quantile(0.25)),
                f"median_{label}": float(values.median()),
                f"q3_{label}": float(values.quantile(0.75)),
                f"max_{label}": float(values.max()),
            }
        )
    return row


def _fmt_number(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if np.isclose(value, round(value)):
        return f"{int(round(value)):,}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _quartile_range(row: pd.Series, label: str) -> str:
    if not np.isfinite(row[f"median_{label}"]):
        return "--"
    return (
        f"{_fmt_number(row[f'q1_{label}'])}/"
        f"{_fmt_number(row[f'median_{label}'])}/"
        f"{_fmt_number(row[f'q3_{label}'])} "
        f"({_fmt_number(row[f'min_{label}'])}--{_fmt_number(row[f'max_{label}'])})"
    )


def _write_latex_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task & $N$ & Observations $n$ & Features $p$ & Classes \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        task = str(row["task"]).replace("&", r"\&")
        lines.append(
            f"{task} & {int(row['n_datasets'])} & "
            f"{_quartile_range(row, 'n')} & {_quartile_range(row, 'p')} & "
            f"{_quartile_range(row, 'classes')} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--outdir", type=Path, default=TABLES_DIR)
    parser.add_argument(
        "--pmlb-cache-dir",
        type=Path,
        default=None,
        help="Local PMLB cache (default: <indir>/pmlb_cache).",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.pmlb_cache_dir or (args.indir / "pmlb_cache")
    data = load_all(args.indir, exclude_secretary_par=False, by_variant=True)

    regression = _dataset_inventory(data["regression_run"], "Regression")
    classification_gini = _dataset_inventory(
        data["classification_gini_run"], "Classification"
    )
    classification_entropy = _dataset_inventory(
        data["classification_entropy_run"], "Classification"
    )
    comparison_cols = ["dataset", "n_samples", "n_features", "n_valid_runs"]
    pd.testing.assert_frame_equal(
        classification_gini[comparison_cols].reset_index(drop=True),
        classification_entropy[comparison_cols].reset_index(drop=True),
    )

    inventory = pd.concat([regression, classification_gini], ignore_index=True)
    inventory = _add_target_metadata(inventory, cache_dir)
    inventory = inventory.sort_values(["task", "dataset"], kind="stable").reset_index(drop=True)
    inventory.to_csv(args.outdir / "dataset_benchmark_inventory.csv", index=False)

    summary = pd.DataFrame(
        [
            _quantile_summary(inventory[inventory["task"] == task], task)
            for task in ("Regression", "Classification")
        ]
    )
    summary.to_csv(args.outdir / "dataset_benchmark_summary.csv", index=False)
    _write_latex_summary(summary, args.outdir / "dataset_benchmark_summary.tex")
    print(f"Wrote {args.outdir / 'dataset_benchmark_inventory.csv'}")
    print(f"Wrote {args.outdir / 'dataset_benchmark_summary.csv'}")
    print(f"Wrote {args.outdir / 'dataset_benchmark_summary.tex'}")


if __name__ == "__main__":
    main()
