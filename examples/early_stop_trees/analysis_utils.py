"""
Analysis utilities for dataset-level paired comparisons and meta-analysis.

Design principles:
- Always aggregate within each (dataset, method) first (median + IQR across runs).
- Comparisons across methods are paired at the dataset level.
- Separate within-dataset randomness (run-to-run variability) from between-dataset heterogeneity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from benchmark_results_utils import (
    SPLITTERS_NO_PAR,
    VARIANT_ORDER,
    load_all,
)


def add_method_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add method_key = 'splitter|variant' (variant may be '')."""
    out = df.copy()
    if "variant" not in out.columns:
        out["variant"] = ""
    out["variant"] = out["variant"].fillna("").astype(str)
    out["splitter"] = out["splitter"].astype(str)
    out["method_key"] = out["splitter"] + "|" + out["variant"]
    return out


def method_sort_key(method_key: str):
    """Deterministic order: best first; then SPLITTERS_NO_PAR; then VARIANT_ORDER; then lexical."""
    splitter, _, variant = method_key.partition("|")
    try:
        s_idx = SPLITTERS_NO_PAR.index(splitter)
    except ValueError:
        s_idx = 999
    if variant in VARIANT_ORDER:
        v_idx = VARIANT_ORDER.index(variant)
    elif variant == "":
        v_idx = -1
    else:
        v_idx = 999
    return (s_idx, v_idx, variant, method_key)


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))


@dataclass(frozen=True)
class TaskSpec:
    name: str
    run_key: str
    summary_key: str
    # columns in run-level DF
    speedup_col: str
    loss_col: str


TASKS = {
    "regression": TaskSpec(
        name="regression",
        run_key="regression_run",
        summary_key="regression_summary",
        speedup_col="speedup",
        loss_col="loss_rmse",
    ),
    "classification_gini": TaskSpec(
        name="classification_gini",
        run_key="classification_gini_run",
        summary_key="classification_gini_summary",
        speedup_col="speedup",
        loss_col="loss_f1",
    ),
    "classification_entropy": TaskSpec(
        name="classification_entropy",
        run_key="classification_entropy_run",
        summary_key="classification_entropy_summary",
        speedup_col="speedup",
        loss_col="loss_f1",
    ),
}


def load_task(indir: Path, task: str, *, exclude_secretary_par: bool = True, by_variant: bool = True):
    """Return (run_df, summary_df) for a task with method_key added."""
    spec = TASKS[task]
    data = load_all(Path(indir), exclude_secretary_par=exclude_secretary_par, by_variant=by_variant)
    run_df = data[spec.run_key]
    summary_df = data[spec.summary_key]
    if run_df is not None:
        run_df = add_method_key(run_df)
    if summary_df is not None:
        summary_df = add_method_key(summary_df)
    return run_df, summary_df


def paired_deltas(
    dataset_summary: pd.DataFrame,
    *,
    metric_median_col: str,
    ref_method_key: str,
    include_best_as_candidate: bool = False,
) -> pd.DataFrame:
    """
    Compute paired per-dataset deltas vs ref_method_key:
      delta(dataset, method) = metric_median(method) - metric_median(ref)

    Returns long DF with columns: dataset, method_key, ref_method_key, delta.
    """
    df = dataset_summary[["dataset", "method_key", metric_median_col]].dropna().copy()
    ref = df[df["method_key"] == ref_method_key][["dataset", metric_median_col]].rename(
        columns={metric_median_col: "ref_value"}
    )
    merged = df.merge(ref, on="dataset", how="inner")
    merged["delta"] = merged[metric_median_col] - merged["ref_value"]
    if not include_best_as_candidate:
        merged = merged[merged["method_key"] != ref_method_key].copy()
    merged["ref_method_key"] = ref_method_key
    return merged[["dataset", "method_key", "ref_method_key", "delta"]]


def within_between_decomposition(
    dataset_summary: pd.DataFrame,
    *,
    metric_median_col: str,
    metric_iqr_col: str,
) -> pd.DataFrame:
    """
    For each method_key, compute:
    - between_dataset_sd: SD across datasets of the dataset-level medians
    - typical_within_iqr: median across datasets of within-dataset IQR across runs
    - ratio_within_to_between: typical_within_iqr / between_dataset_sd (scale sense only)
    """
    rows = []
    for method_key, sub in dataset_summary.groupby("method_key"):
        med = sub[metric_median_col].to_numpy(dtype=float)
        iqr = sub[metric_iqr_col].to_numpy(dtype=float) if metric_iqr_col in sub.columns else np.full_like(med, np.nan)
        between_sd = float(np.nanstd(med, ddof=1)) if np.sum(np.isfinite(med)) >= 2 else np.nan
        typical_within_iqr = float(np.nanmedian(iqr)) if np.any(np.isfinite(iqr)) else np.nan
        ratio = typical_within_iqr / between_sd if (np.isfinite(typical_within_iqr) and np.isfinite(between_sd) and between_sd > 0) else np.nan
        rows.append(
            {
                "method_key": method_key,
                "between_dataset_sd": between_sd,
                "typical_within_iqr": typical_within_iqr,
                "ratio_within_to_between": ratio,
                "n_datasets": int(sub["dataset"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("method_key", key=lambda s: s.map(method_sort_key)).reset_index(drop=True)
    return out


def summarize_paired_deltas(
    deltas: pd.DataFrame,
    *,
    better_if_delta_positive: bool,
) -> pd.DataFrame:
    """
    Aggregate paired deltas per method_key:
    - median_delta, mean_delta
    - win_rate: fraction of datasets where the delta indicates the method is better than ref
    - n_pairs
    """
    rows = []
    if better_if_delta_positive:
        win_mask = lambda d: d > 0
    else:
        win_mask = lambda d: d < 0
    for mk, sub in deltas.groupby("method_key"):
        d = sub["delta"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        rows.append(
            {
                "method_key": mk,
                "median_delta": float(np.median(d)),
                "mean_delta": float(np.mean(d)),
                "win_rate": float(np.mean(win_mask(d))),
                "n_pairs": int(d.size),
                "ref_method_key": str(sub["ref_method_key"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("method_key", key=lambda s: s.map(method_sort_key)).reset_index(drop=True)
    return out


def task_dataset_benchmark_stats(run_df: pd.DataFrame | None, task_label: str) -> dict | None:
    """
    Compact stats for datasets **actually present** in benchmark run-level results (after skips/errors).

    Returns one dict: task, n_datasets, median_n, median_p, min_n, max_n, min_p, max_p.
    """
    if run_df is None or run_df.empty:
        return None
    if "dataset" not in run_df.columns or "n_samples" not in run_df.columns or "n_features" not in run_df.columns:
        return None
    meta = run_df[["dataset", "n_samples", "n_features"]].drop_duplicates("dataset")
    n = meta["n_samples"].astype(float).to_numpy()
    p = meta["n_features"].astype(float).to_numpy()
    return {
        "task": task_label,
        "n_datasets": int(meta.shape[0]),
        "median_n": float(np.median(n)),
        "median_p": float(np.median(p)),
        "min_n": float(np.min(n)),
        "max_n": float(np.max(n)),
        "min_p": float(np.min(p)),
        "max_p": float(np.max(p)),
    }


def pairwise_method_pair_metrics(
    dataset_summary: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    speed_median_col: str,
    loss_median_col: str,
) -> dict:
    """
    Paired comparison at dataset level using **medians across runs** per (dataset, method).

    - Faster: median speedup(A) > median speedup(B) (higher speedup = faster vs exhaustive).
    - Better loss: median loss(A) < median loss(B) (lower is better for RMSE-relative and F1 loss).

    Returns counts and fractions on the intersection of datasets where both methods exist.
    """
    cols = ["dataset", "method_key", speed_median_col, loss_median_col]
    df = dataset_summary[[c for c in cols if c in dataset_summary.columns]].dropna()
    da = df[df["method_key"] == method_a].drop_duplicates("dataset")
    db = df[df["method_key"] == method_b].drop_duplicates("dataset")
    da = da.rename(columns={speed_median_col: "speed_a", loss_median_col: "loss_a"})
    db = db.rename(columns={speed_median_col: "speed_b", loss_median_col: "loss_b"})
    m = da[["dataset", "speed_a", "loss_a"]].merge(db[["dataset", "speed_b", "loss_b"]], on="dataset", how="inner")
    if m.empty:
        return {
            "method_a": method_a,
            "method_b": method_b,
            "n_common_datasets": 0,
            "frac_A_faster": np.nan,
            "frac_A_better_loss": np.nan,
            "frac_A_dominates_both": np.nan,
            "frac_tie_speed": np.nan,
            "frac_tie_loss": np.nan,
            "wilcoxon_speed_diff_pvalue": np.nan,
            "wilcoxon_loss_diff_pvalue": np.nan,
        }
    ds = m["speed_a"] - m["speed_b"]
    dl = m["loss_a"] - m["loss_b"]
    faster = ds > 0
    better_loss = dl < 0
    both = faster & better_loss
    tie_speed = np.isclose(ds.to_numpy(dtype=float), 0.0, rtol=0, atol=1e-12)
    tie_loss = np.isclose(dl.to_numpy(dtype=float), 0.0, rtol=0, atol=1e-12)

    def _wilcoxon_p(d: np.ndarray) -> float:
        d = d[np.isfinite(d)]
        if d.size < 5:
            return np.nan
        try:
            from scipy.stats import wilcoxon

            r = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
            return float(r.pvalue)
        except Exception:
            return np.nan

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_common_datasets": int(m.shape[0]),
        "frac_A_faster": float(np.mean(faster)),
        "frac_A_better_loss": float(np.mean(better_loss)),
        "frac_A_dominates_both": float(np.mean(both)),
        "frac_tie_speed": float(np.mean(tie_speed)),
        "frac_tie_loss": float(np.mean(tie_loss)),
        "wilcoxon_speed_diff_pvalue": _wilcoxon_p(ds.to_numpy(dtype=float)),
        "wilcoxon_loss_diff_pvalue": _wilcoxon_p(dl.to_numpy(dtype=float)),
    }


def add_basic_dataset_meta_from_results(run_or_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Extract dataset-level (n, p) from existing result columns; returns one row per dataset."""
    cols = [c for c in ["dataset", "n_samples", "n_features"] if c in run_or_summary_df.columns]
    meta = run_or_summary_df[cols].drop_duplicates("dataset").copy()
    if "n_samples" in meta.columns:
        meta["n"] = meta["n_samples"].astype(float)
    if "n_features" in meta.columns:
        meta["p"] = meta["n_features"].astype(float)
    if "n" in meta.columns and "p" in meta.columns:
        meta["p_over_n"] = meta["p"] / meta["n"].replace(0, np.nan)
        meta["log_n"] = _safe_log(meta["n"].to_numpy())
        meta["log_p"] = _safe_log(meta["p"].to_numpy())
    return meta


def compute_classification_label_meta(
    datasets: Iterable[str],
    *,
    cache_dir: Path,
    max_rows_for_threshold_proxy: Optional[int] = None,
    compute_threshold_proxy: bool = False,
) -> pd.DataFrame:
    """
    Fetch y from PMLB cache and compute:
    - n_classes
    - imbalance_ratio: max_class_count / min_class_count (>=1)

    Optionally (expensive), compute a proxy for mean admissible thresholds per feature:
    - mean_unique_per_feature - 1 (after optional subsampling of rows)
    """
    try:
        from pmlb import fetch_data
    except Exception as e:
        raise RuntimeError("pmlb is required for label meta; install with `pip install pmlb`") from e

    rows = []
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        try:
            if compute_threshold_proxy:
                X, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
                X = np.asarray(X)
            else:
                _, y = fetch_data(name, return_X_y=True, local_cache_dir=str(cache_dir))
                X = None
            y = np.asarray(y)
        except Exception:
            continue
        vals, counts = np.unique(y, return_counts=True)
        if counts.size == 0:
            continue
        n_classes = int(vals.size)
        cmax = float(np.max(counts))
        cmin = float(np.min(counts))
        imbalance = (cmax / cmin) if cmin > 0 else np.nan
        out = {"dataset": name, "n_classes": n_classes, "imbalance_ratio": imbalance}
        if compute_threshold_proxy and X is not None:
            if max_rows_for_threshold_proxy is not None and X.shape[0] > max_rows_for_threshold_proxy:
                rng = np.random.default_rng(0)
                idx = rng.choice(X.shape[0], size=max_rows_for_threshold_proxy, replace=False)
                Xs = X[idx]
            else:
                Xs = X
            # mean over features of (unique values - 1), which equals admissible threshold count if all values distinct
            uniq = []
            for j in range(Xs.shape[1]):
                col = Xs[:, j]
                # robust to floats/ints/strings
                try:
                    u = np.unique(col).size
                except Exception:
                    u = len(set(col.tolist()))
                uniq.append(max(0, u - 1))
            out["thresholds_per_feature_proxy_mean"] = float(np.mean(uniq)) if uniq else np.nan
        rows.append(out)
    return pd.DataFrame(rows)

