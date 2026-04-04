#!/usr/bin/env python
"""
Table 1. Global summary by method.

For each method and each task (Regression, Gini, Entropy), report:
- centroid time saved across datasets
- median total effort saved across datasets (% gain evaluations saved versus best)
- median predictive loss across datasets (%)
- 90th percentile predictive loss (%)
- proportion of datasets for which loss stays below a fixed tolerance
- proportion of datasets for which speedup exceeds a fixed threshold.

Output: CSV and optionally LaTeX in tables/ (table1_*.csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_results_utils import (
    SPAR_REPRESENTATIVE_KEY,
    keep_secretary_par_representative,
    load_all,
    method_display_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmark_results"
OUT_DIR = SCRIPT_DIR / "tables"
ARTICLE_TABLE_DIR = SCRIPT_DIR.parents[1] / "RESEARCH_ARTICLE" / "TABLES"

# Fixed tolerance and threshold for the proportion columns
LOSS_TOLERANCE = 0.05   # 5% loss (loss in [0,1] so 0.05)
SPEEDUP_THRESHOLD_PCT = 20  # P(speedup ≥ 20%) -> speedup >= 1.20
SPEEDUP_THRESHOLD = 1.0 + SPEEDUP_THRESHOLD_PCT / 100.0


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
    ("Classification (Gini)", "secretary_par", SPAR_REPRESENTATIVE_KEY.split("|", 1)[1]),
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
    ("Classification (Entropy)", "secretary_par", SPAR_REPRESENTATIVE_KEY.split("|", 1)[1]),
    ("Classification (Entropy)", "block_rank", ""),
    ("Classification (Entropy)", "prophet_1sample", ""),
    ("Classification (Entropy)", "extra_tree", "max_features=1"),
    ("Classification (Entropy)", "extra_tree", "max_features=1over3"),
    ("Classification (Entropy)", "extra_tree", "max_features=2over3"),
    ("Classification (Entropy)", "extra_tree", "max_features=all"),
]


def _global_summary_one_task(summary: pd.DataFrame, loss_col: str) -> pd.DataFrame:
    """Per (splitter, variant): centroid time saved, effort saved, loss (%), P90 loss (%), P(loss ≤ τ), P(speedup ≥ threshold)."""
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
        sp = sub["speedup_median"].values
        centroid_time_saved_pct = np.nanmean(100.0 * (1.0 - 1.0 / sp))
        loss = sub[loss_col].values
        loss_pct = 100.0 * loss
        effort_saved_total_pct = np.nan
        if "effort_saved_total_median" in sub.columns:
            effort_saved_total_pct = 100.0 * np.median(sub["effort_saved_total_median"].values)
        rows.append({
            "method": method_label,
            "splitter": s,
            "variant": v,
            "centroid_time_saved_pct": centroid_time_saved_pct,
            "median_effort_saved_total_pct": effort_saved_total_pct,
            "median_loss_pct": np.median(loss_pct),
            "p90_loss_pct": np.percentile(loss_pct, 90),
            "p_loss_below_tol": np.mean(loss <= LOSS_TOLERANCE),
            "p_speedup_above_thr": np.mean(sp >= SPEEDUP_THRESHOLD),
        })
    return pd.DataFrame(rows)


def _article_method_label(splitter: str, variant: str) -> str:
    if splitter == "best":
        return "Exhaustive"
    if splitter == "secretary":
        return rf"$S$ ({'n/e' if variant == '1overe' else variant})"
    if splitter == "double_secretary":
        return rf"$S^2$ ({'n/e' if variant == '1overe' else variant})"
    if splitter == "secretary_all":
        return rf"$S_{{\mathrm{{all}}}}$ ({'n/e' if variant == '1overe' else variant})"
    if splitter == "secretary_par":
        fields = dict(part.split("=", 1) for part in variant.split(",") if "=" in part)
        samples = fields.get("samples", "")
        quantile = fields.get("q", "")
        sample_label = {
            "sqrt_n": r"$\sqrt{n}$",
            "ln_n": r"$\ln(n)$",
            "1overe": "n/e",
            "0.1n": "0.1n",
        }.get(samples, samples)
        return rf"$S_{{\mathrm{{par}}}}$ ({sample_label}, q={quantile})"
    if splitter == "block_rank":
        return "Block-rank"
    if splitter == "prophet_1sample":
        return "1-sample prophet"
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
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Representative global summary by method after the final 100-run no-limit benchmark. "
            r"The $n/e$ schedule is used for the secretary families, "
            r"$S_{\mathrm{par}}(\sqrt{n},q=0.75)$ is the single parametric representative shown in the main comparison, "
            r"and ERT denotes extremely randomized trees with four $m_{\mathrm{try}}$ budgets. "
            r"Centroid time saved is the horizontal centroid of the dataset-level median points shown in Figure~1, expressed relative to the exhaustive \texttt{scikit-learn} splitter. "
            r"Effort saved is the percentage reduction in gain evaluations relative to exhaustive split search. "
            r"Loss is expressed in percent.}"
        ),
        r"\label{tab:summary_results}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Task & Method & Centroid time saved (\%) & Median effort saved (\%) & Median loss (\%) & 90th pct.\ loss (\%) & $P(\mathrm{loss}\le 5\%)$ & $P(\mathrm{speedup}\ge 20\%)$ \\",
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
                    f"{row['centroid_time_saved_pct']:.2f}",
                    f"{row['median_effort_saved_total_pct']:.2f}",
                    f"{row['median_loss_pct']:.3f}",
                    f"{row['p90_loss_pct']:.3f}",
                    f"{row['p_loss_below_tol']:.3f}",
                    f"{row['p_speedup_above_thr']:.3f}",
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
        ]
    )
    (ARTICLE_TABLE_DIR / "table1_main.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all(BENCHMARK_DIR, exclude_secretary_par=False, by_variant=True)

    regression_summary = data["regression_summary"]
    if regression_summary is not None:
        regression_summary = regression_summary[regression_summary["splitter"] != "secretary_par"].copy()

    gini_summary = keep_secretary_par_representative(data["classification_gini_summary"])
    entropy_summary = keep_secretary_par_representative(data["classification_entropy_summary"])

    configs = [
        ("regression", regression_summary, "loss_rmse_bounded_median", "Regression"),
        ("gini", gini_summary, "loss_f1_median", "Classification (Gini)"),
        ("entropy", entropy_summary, "loss_f1_median", "Classification (Entropy)"),
    ]

    all_tables = []
    raw_tables = {}
    for tag, summary, loss_col, task_label in configs:
        if summary is None:
            continue
        df = _global_summary_one_task(summary, loss_col)
        if df.empty:
            continue
        raw_tables[tag] = df.copy()
        df["task"] = task_label
        # Reorder columns: task, method, then metrics
        cols = [
            "task",
            "method",
            "centroid_time_saved_pct",
            "median_effort_saved_total_pct",
            "median_loss_pct",
            "p90_loss_pct",
            "p_loss_below_tol",
            "p_speedup_above_thr",
        ]
        df = df[[c for c in cols if c in df.columns]]
        df = df.rename(columns={
            "centroid_time_saved_pct": "centroid_time_saved_%",
            "median_effort_saved_total_pct": "median_effort_saved_%",
            "median_loss_pct": "median_loss_%",
            "p90_loss_pct": "p90_loss_%",
            "p_loss_below_tol": f"P(loss≤{int(LOSS_TOLERANCE*100)}%)",
            "p_speedup_above_thr": f"P(speedup≥{SPEEDUP_THRESHOLD_PCT}%)",
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
