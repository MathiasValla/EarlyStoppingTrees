#!/usr/bin/env python
"""
Summarize secretary-family variants against ExtraTree baselines from raw run CSVs.

This script is designed for focused benchmark directories produced by
``benchmark_secretary_pmlb.py`` without requiring the exhaustive ``best``
baseline to be present.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = (
    ("regression", "regression_run*.csv", "rmse_mean", False, None),
    ("classification_gini", "classification_gini_run*.csv", "f1_weighted_mean", True, "accuracy_mean"),
    ("classification_entropy", "classification_entropy_run*.csv", "f1_weighted_mean", True, "accuracy_mean"),
)
EXTRA_KEYS = (
    "extra_tree|max_features=1",
    "extra_tree|max_features=1over3",
    "extra_tree|max_features=2over3",
    "extra_tree|max_features=all",
)
REFERENCE_KEY = "extra_tree|max_features=1"
RUN_RE = re.compile(r"_run(\d+)\.csv$")


def _with_method_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["variant"] = df["variant"].fillna("").astype(str)
    df["method_key"] = df["splitter"].astype(str) + "|" + df["variant"]
    return df


def _load_task_runs(indir: Path, pattern: str) -> pd.DataFrame:
    dfs = []
    for path in sorted(indir.glob(pattern)):
        match = RUN_RE.search(path.name)
        if match is None:
            continue
        df = pd.read_csv(path)
        df["run"] = int(match.group(1))
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return _with_method_key(pd.concat(dfs, ignore_index=True))


def _dataset_summary(run_df: pd.DataFrame, metric_col: str, secondary_metric_col: str | None) -> pd.DataFrame:
    agg = {
        "fit_time_mean": ("fit_time_mean", "median"),
        metric_col: (metric_col, "median"),
        "n_runs": ("run", "nunique"),
        "n_samples": ("n_samples", "first"),
        "n_features": ("n_features", "first"),
    }
    if secondary_metric_col is not None and secondary_metric_col in run_df.columns:
        agg[secondary_metric_col] = (secondary_metric_col, "median")
    return (
        run_df.groupby(["dataset", "splitter", "variant", "method_key"], as_index=False)
        .agg(**agg)
        .rename(columns={"fit_time_mean": "fit_time_median", metric_col: "metric_median"})
    )


def _pairwise_metrics(
    dataset_summary: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    metric_higher_is_better: bool,
) -> dict:
    cols = ["dataset", "method_key", "fit_time_median", "metric_median"]
    df = dataset_summary[cols].dropna()
    da = df[df["method_key"] == method_a].drop_duplicates("dataset")
    db = df[df["method_key"] == method_b].drop_duplicates("dataset")
    da = da.rename(columns={"fit_time_median": "time_a", "metric_median": "metric_a"})
    db = db.rename(columns={"fit_time_median": "time_b", "metric_median": "metric_b"})
    merged = da[["dataset", "time_a", "metric_a"]].merge(
        db[["dataset", "time_b", "metric_b"]],
        on="dataset",
        how="inner",
    )
    if merged.empty:
        return {
            "n_common_datasets": 0,
            "frac_A_faster": np.nan,
            "frac_A_better_metric": np.nan,
            "frac_A_dominates_both": np.nan,
        }

    faster = merged["time_a"] < merged["time_b"]
    if metric_higher_is_better:
        better = merged["metric_a"] > merged["metric_b"]
    else:
        better = merged["metric_a"] < merged["metric_b"]
    both = faster & better
    return {
        "n_common_datasets": int(merged.shape[0]),
        "frac_A_faster": float(np.mean(faster)),
        "frac_A_better_metric": float(np.mean(better)),
        "frac_A_dominates_both": float(np.mean(both)),
    }


def _task_summary(
    run_df: pd.DataFrame,
    *,
    metric_col: str,
    metric_higher_is_better: bool,
    secondary_metric_col: str | None,
) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    ds = _dataset_summary(run_df, metric_col, secondary_metric_col)
    global_df = (
        ds.groupby(["splitter", "variant", "method_key"], as_index=False)
        .agg(
            global_time_median=("fit_time_median", "median"),
            global_metric_median=("metric_median", "median"),
            n_datasets=("dataset", "nunique"),
        )
    )
    if secondary_metric_col is not None and secondary_metric_col in ds.columns:
        aux = (
            ds.groupby(["splitter", "variant", "method_key"], as_index=False)
            .agg(global_secondary_metric_median=(secondary_metric_col, "median"))
        )
        global_df = global_df.merge(aux, on=["splitter", "variant", "method_key"], how="left")

    extra_metric = {
        row["method_key"]: float(row["global_metric_median"])
        for _, row in global_df[global_df["method_key"].isin(EXTRA_KEYS)].iterrows()
    }
    extra_time = {
        row["method_key"]: float(row["global_time_median"])
        for _, row in global_df[global_df["method_key"].isin(EXTRA_KEYS)].iterrows()
    }

    rows = []
    for _, row in global_df.iterrows():
        method_key = row["method_key"]
        if method_key == "best|" or method_key in EXTRA_KEYS:
            continue
        pairwise = _pairwise_metrics(
            ds,
            method_key,
            REFERENCE_KEY,
            metric_higher_is_better=metric_higher_is_better,
        )
        if metric_higher_is_better:
            beats_extra_metric = sum(
                float(row["global_metric_median"]) > extra_metric[key]
                for key in EXTRA_KEYS
                if key in extra_metric
            )
        else:
            beats_extra_metric = sum(
                float(row["global_metric_median"]) < extra_metric[key]
                for key in EXTRA_KEYS
                if key in extra_metric
            )

        reference_time = extra_time.get(REFERENCE_KEY, np.nan)
        rows.append(
            {
                "splitter": row["splitter"],
                "variant": row["variant"],
                "method_key": method_key,
                "n_datasets": int(row["n_datasets"]),
                "global_time_median": float(row["global_time_median"]),
                "global_metric_median": float(row["global_metric_median"]),
                "global_time_ratio_vs_extra_tree_max_features_1": (
                    float(row["global_time_median"]) / reference_time
                    if np.isfinite(reference_time) and reference_time > 0
                    else np.nan
                ),
                "n_extra_tree_baselines_beaten_on_global_metric": int(beats_extra_metric),
                "frac_faster_than_extra_tree_max_features_1": pairwise["frac_A_faster"],
                "frac_better_metric_than_extra_tree_max_features_1": pairwise["frac_A_better_metric"],
                "frac_dominate_time_and_metric_vs_extra_tree_max_features_1": pairwise["frac_A_dominates_both"],
                "n_common_datasets_vs_extra_tree_max_features_1": pairwise["n_common_datasets"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            [
                "n_extra_tree_baselines_beaten_on_global_metric",
                "frac_better_metric_than_extra_tree_max_features_1",
                "global_metric_median",
                "global_time_median",
            ],
            ascending=[
                False,
                False,
                not metric_higher_is_better,
                True,
            ],
        ).reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize secretary-family variants against ExtraTree baselines.")
    ap.add_argument("--indir", type=Path, required=True, help="Benchmark result directory")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory for summary CSV/JSON")
    args = ap.parse_args()

    indir = args.indir.resolve()
    outdir = (args.outdir or indir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {}
    for task_name, pattern, metric_col, metric_higher_is_better, secondary_metric_col in TASKS:
        run_df = _load_task_runs(indir, pattern)
        task_df = _task_summary(
            run_df,
            metric_col=metric_col,
            metric_higher_is_better=metric_higher_is_better,
            secondary_metric_col=secondary_metric_col,
        )
        payload[task_name] = task_df.to_dict(orient="records")
        csv_path = outdir / f"{task_name}_secretary_vs_extra_summary.csv"
        task_df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")
        if not task_df.empty:
            print(f"\n[{task_name}] top variants")
            print(task_df.head(10).to_string(index=False))

    json_path = outdir / "secretary_vs_extra_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
