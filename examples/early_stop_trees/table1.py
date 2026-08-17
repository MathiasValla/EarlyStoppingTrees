#!/usr/bin/env python
"""Global benchmark summaries with hierarchical confidence intervals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_results_utils import (
    load_all,
    method_display_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "tables"
ARTICLE_TABLE_DIR = SCRIPT_DIR.parents[1] / "RESEARCH_ARTICLE" / "TABLES"

MAIN_TABLE_ROWS = [
    ("Regression", "best", ""),
    ("Regression", "secretary", "1overe"),
    ("Regression", "double_secretary", "1overe"),
    ("Regression", "secretary_all", "1overe"),
    ("Regression", "block_rank", ""),
    ("Regression", "prophet_1sample", ""),
    ("Regression", "extra_tree", "max_features=1"),
    ("Regression", "extra_tree", "max_features=1over3"),
    ("Regression", "extra_tree", "max_features=2over3"),
    ("Regression", "extra_tree", "max_features=all"),
    ("Classification (Gini)", "best", ""),
    ("Classification (Gini)", "secretary", "1overe"),
    ("Classification (Gini)", "double_secretary", "1overe"),
    ("Classification (Gini)", "secretary_all", "1overe"),
    ("Classification (Gini)", "block_rank", ""),
    ("Classification (Gini)", "prophet_1sample", ""),
    ("Classification (Gini)", "extra_tree", "max_features=1"),
    ("Classification (Gini)", "extra_tree", "max_features=1over3"),
    ("Classification (Gini)", "extra_tree", "max_features=2over3"),
    ("Classification (Gini)", "extra_tree", "max_features=all"),
    ("Classification (Entropy)", "best", ""),
    ("Classification (Entropy)", "secretary", "1overe"),
    ("Classification (Entropy)", "double_secretary", "1overe"),
    ("Classification (Entropy)", "secretary_all", "1overe"),
    ("Classification (Entropy)", "block_rank", ""),
    ("Classification (Entropy)", "prophet_1sample", ""),
    ("Classification (Entropy)", "extra_tree", "max_features=1"),
    ("Classification (Entropy)", "extra_tree", "max_features=1over3"),
    ("Classification (Entropy)", "extra_tree", "max_features=2over3"),
    ("Classification (Entropy)", "extra_tree", "max_features=all"),
]


def _inference_row(
    inference: pd.DataFrame,
    *,
    task: str,
    method_key: str,
    metric: str,
    estimand: str,
) -> pd.Series:
    rows = inference[
        (inference["task"] == task)
        & (inference["method_key"] == method_key)
        & (inference["metric"] == metric)
        & (inference["estimand"] == estimand)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one inference row for {task}/{method_key}/{metric}/{estimand}, "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def _global_summary_one_task(
    summary: pd.DataFrame,
    loss_col: str,
    *,
    inference: pd.DataFrame,
    task_key: str,
) -> pd.DataFrame:
    """Summarize centroid and cross-dataset-median estimands with 95% CIs."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    summary = summary.dropna(subset=["speedup_median", loss_col]).copy()
    if "variant" not in summary.columns:
        summary["variant"] = ""
    summary["method_key"] = summary["splitter"].astype(str) + "|" + summary["variant"].fillna("").astype(str)
    rows = []
    for method_key in summary["method_key"].unique():
        sub = summary[summary["method_key"] == method_key]
        if sub.empty:
            continue
        s = sub["splitter"].iloc[0]
        v = sub["variant"].iloc[0]
        if pd.isna(v):
            v = ""
        method_label = method_display_label(s, v)
        method_key = f"{s}|{v}"
        sp = sub["speedup_median"].values
        centroid_time_saved_pct = np.nanmean(100.0 * (1.0 - 1.0 / sp))
        time_row = _inference_row(
            inference,
            task=task_key,
            method_key=method_key,
            metric="time_saved_pct",
            estimand="centroid_mean",
        )
        loss_centroid_row = _inference_row(
            inference,
            task=task_key,
            method_key=method_key,
            metric="predictive_loss_pct",
            estimand="centroid_mean",
        )
        loss_median_row = _inference_row(
            inference,
            task=task_key,
            method_key=method_key,
            metric="predictive_loss_pct",
            estimand="cross_dataset_median",
        )
        effort_median_row = _inference_row(
            inference,
            task=task_key,
            method_key=method_key,
            metric="effort_saved_pct",
            estimand="cross_dataset_median",
        )
        if not np.isclose(
            centroid_time_saved_pct, float(time_row["estimate"]), atol=1e-8
        ):
            raise ValueError(
                f"Time centroid mismatch for {task_key}/{method_key}: "
                f"summary={centroid_time_saved_pct}, inference={time_row['estimate']}"
            )
        rows.append({
            "method": method_label,
            "splitter": s,
            "variant": v,
            "centroid_time_saved_pct": float(time_row["estimate"]),
            "centroid_time_saved_ci_low": float(time_row["hierarchical_ci_low"]),
            "centroid_time_saved_ci_high": float(time_row["hierarchical_ci_high"]),
            "centroid_loss_pct": float(loss_centroid_row["estimate"]),
            "centroid_loss_ci_low": float(loss_centroid_row["hierarchical_ci_low"]),
            "centroid_loss_ci_high": float(loss_centroid_row["hierarchical_ci_high"]),
            "median_loss_pct": float(loss_median_row["estimate"]),
            "median_loss_ci_low": float(loss_median_row["hierarchical_ci_low"]),
            "median_loss_ci_high": float(loss_median_row["hierarchical_ci_high"]),
            "median_effort_saved_pct": float(effort_median_row["estimate"]),
            "median_effort_saved_ci_low": float(effort_median_row["hierarchical_ci_low"]),
            "median_effort_saved_ci_high": float(effort_median_row["hierarchical_ci_high"]),
        })
    return pd.DataFrame(rows)


def _article_method_label(splitter: str, variant: str) -> str:
    if splitter == "best":
        return "Exhaustive"
    if splitter == "secretary":
        return rf"$S$ ({'f=1/e' if variant == '1overe' else variant})"
    if splitter == "double_secretary":
        return rf"$S^2$ ({'f=1/e' if variant == '1overe' else variant})"
    if splitter == "secretary_all":
        return rf"$S_{{\mathrm{{all}}}}$ ({'f=1/e' if variant == '1overe' else variant})"
    if splitter == "secretary_par":
        fields = dict(part.split("=", 1) for part in variant.split(",") if "=" in part)
        samples = fields.get("samples", "")
        quantile = fields.get("q", "")
        sample_label = {
            "sqrt_n": r"$\rho=N^{-1/2}$",
            "ln_n": r"$B=\operatorname{round}(\log N)$",
            "1overe": r"$\rho=1/e$",
            "0.1n": r"$\rho=0.1$",
        }.get(samples, samples)
        return rf"$S_{{\mathrm{{par}}}}$ ({sample_label}, q={quantile})"
    if splitter == "block_rank":
        return "Rank-inspired"
    if splitter == "prophet_1sample":
        return "Prophet-style"
    if splitter == "extra_tree":
        mtry = {
            "max_features=1": "1",
            "max_features=1over3": "p/3",
            "max_features=2over3": "2p/3",
            "max_features=all": "p",
        }.get(variant, variant)
        return rf"ERT ($m_{{\mathrm{{try}}}}={mtry}$)"
    return method_display_label(splitter, variant)


def _write_article_main_table(raw_tables: dict[str, pd.DataFrame]) -> None:
    ARTICLE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    task_map = {
        "Regression": raw_tables["regression"],
        "Classification (Gini)": raw_tables["gini"],
        "Classification (Entropy)": raw_tables["entropy"],
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Representative-method summaries with hierarchical 95\% bootstrap intervals. "
            r"Centroids average entry-level run medians; classification losses are weighted-F1 "
            r"percentage points and all other entries are percentages.}"
        ),
        r"\label{tab:summary_results}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\scriptsize",
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Task & Method & Centroid time saved & Centroid loss & Median loss & Median effort saved \\",
        r"\midrule",
    ]

    current_task = None
    for task, splitter, variant in MAIN_TABLE_ROWS:
        df = task_map[task]
        row = df[(df["splitter"] == splitter) & (df["variant"].fillna("") == variant)]
        if row.empty:
            continue
        row = row.iloc[0]
        if current_task is not None and task != current_task:
            lines.append(r"\midrule")
        current_task = task
        lines.append(
            " & ".join(
                [
                    task,
                    _article_method_label(splitter, variant),
                    _format_interval(row, "centroid_time_saved", 2),
                    _format_interval(row, "centroid_loss", 2),
                    _format_interval(row, "median_loss", 2),
                    _format_interval(row, "median_effort_saved", 2),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
        ]
    )
    (ARTICLE_TABLE_DIR / "table1_main.tex").write_text("\n".join(lines) + "\n")


def _format_interval(row: pd.Series, stem: str, digits: int) -> str:
    estimate = row[f"{stem}_pct"]
    low = row[f"{stem}_ci_low"]
    high = row[f"{stem}_ci_high"]
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=SCRIPT_DIR / "inference_results",
        help="Directory containing global_intervals.csv.",
    )
    args = parser.parse_args()
    inference_path = args.inference_dir / "global_intervals.csv"
    if not inference_path.is_file():
        raise FileNotFoundError(f"Missing inferential results: {inference_path}")
    inference = pd.read_csv(inference_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(BENCHMARK_DIR, exclude_secretary_par=True, by_variant=True)

    regression_summary = data["regression_summary"]
    gini_summary = data["classification_gini_summary"]
    entropy_summary = data["classification_entropy_summary"]

    configs = [
        ("regression", regression_summary, "loss_rmse_bounded_median", "Regression", "regression"),
        ("gini", gini_summary, "loss_f1_median", "Classification (Gini)", "classification_gini"),
        ("entropy", entropy_summary, "loss_f1_median", "Classification (Entropy)", "classification_entropy"),
    ]

    all_tables = []
    raw_tables = {}
    for tag, summary, loss_col, task_label, task_key in configs:
        if summary is None:
            continue
        df = _global_summary_one_task(
            summary, loss_col, inference=inference, task_key=task_key
        )
        if df.empty:
            continue
        raw_tables[tag] = df.copy()
        df["task"] = task_label
        # Reorder columns: task, method, then metrics
        cols = [
            "task",
            "method",
            "centroid_time_saved_pct",
            "centroid_time_saved_ci_low",
            "centroid_time_saved_ci_high",
            "centroid_loss_pct",
            "centroid_loss_ci_low",
            "centroid_loss_ci_high",
            "median_loss_pct",
            "median_loss_ci_low",
            "median_loss_ci_high",
            "median_effort_saved_pct",
            "median_effort_saved_ci_low",
            "median_effort_saved_ci_high",
        ]
        df = df[[c for c in cols if c in df.columns]]
        df = df.rename(columns={
            "centroid_time_saved_pct": "centroid_time_saved_%",
            "centroid_time_saved_ci_low": "centroid_time_saved_ci_low",
            "centroid_time_saved_ci_high": "centroid_time_saved_ci_high",
            "centroid_loss_pct": "centroid_loss_%",
            "centroid_loss_ci_low": "centroid_loss_ci_low",
            "centroid_loss_ci_high": "centroid_loss_ci_high",
            "median_loss_pct": "median_loss_%",
            "median_loss_ci_low": "median_loss_ci_low",
            "median_loss_ci_high": "median_loss_ci_high",
            "median_effort_saved_pct": "median_effort_saved_%",
            "median_effort_saved_ci_low": "median_effort_saved_ci_low",
            "median_effort_saved_ci_high": "median_effort_saved_ci_high",
        })
        out_csv = OUT_DIR / f"table1_{tag}.csv"
        df.to_csv(out_csv, index=False, float_format="%.4g")
        all_tables.append(df)
        print(f"Saved {out_csv}")

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined.to_csv(OUT_DIR / "table1_all.csv", index=False, float_format="%.4g")
        print(f"Saved {OUT_DIR / 'table1_all.csv'}")
        # LaTeX (one table with task as first column)
        out_tex = OUT_DIR / "table1.tex"
        cols = [c for c in combined.columns if c not in ("task", "method")]
        with open(out_tex, "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n")
            f.write("\\caption{Global summary by method: speedup, effort saved relative to exhaustive gain evaluations, predictive loss, and operating-point probabilities.}\n")
            f.write("\\label{tab:global-summary}\n")
            f.write("\\begin{tabular}{ll" + "r" * len(cols) + "}\n\\toprule\n")
            f.write("Task & Method & " + " & ".join(c.replace("_", "\\_").replace("%", "\\%") for c in cols) + " \\\\\n\\midrule\n")
            for _, row in combined.iterrows():
                vals = [str(row["task"]), str(row["method"]).replace("_", " ")]
                for c in cols:
                    v = row[c]
                    vals.append(f"{v:.3g}" if pd.notna(v) and isinstance(v, (int, float)) else str(v))
                f.write(" & ".join(vals) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        print(f"Saved {out_tex}")
    if set(raw_tables) >= {"regression", "gini", "entropy"}:
        _write_article_main_table(raw_tables)
        for name in (
            "table1.tex",
            "table1_all.csv",
            "table1_regression.csv",
            "table1_gini.csv",
            "table1_entropy.csv",
        ):
            src = OUT_DIR / name
            if src.is_file():
                ARTICLE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
                ARTICLE_TABLE_DIR.joinpath(name).write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
