"""
Load benchmark run-level CSVs and compute normalized metrics (speedup, loss) relative to the
exhaustive (best) splitter, then per-dataset medians and IQR across runs.

Formulas:
- Speedup_dmr = t_d^best / t_dmr  (>1 = secretary faster)
- Loss^RMSE_dmr = (RMSE_dmr - RMSE_d^best) / RMSE_d^best
- Loss^Acc_dmr = Acc_d^best - Acc_dmr,  Loss^F1_dmr = F1_d^best - F1_dmr
- tilde{Speedup}_dm = median_r Speedup_dmr,  tilde{Loss}_dm = median_r Loss_dmr
- Variability: IQR (or IDR) across runs for each (dataset, method).
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle


# Methods to compare (exhaustive baseline first)
SPLITTERS = (
    "best",
    "secretary",
    "secretary_par",
    "secretary_all",
    "double_secretary",
    "block_rank",
    "prophet_1sample",
    "extra_tree",
)
# Historical name kept for compatibility; this is the list of all non-best families.
SECRETARY_SPLITTERS = [s for s in SPLITTERS if s != "best"]
# For figures/tables that exclude secretary_par and show variants
SPLITTERS_NO_PAR = (
    "best",
    "secretary",
    "secretary_all",
    "double_secretary",
    "block_rank",
    "prophet_1sample",
    "extra_tree",
)
SECRETARY_SPLITTERS_NO_PAR = [s for s in SPLITTERS_NO_PAR if s != "best"]

# Distinct base colors per method (no secretary_par); variants get shades of these.
# ExtraTree intentionally avoids blue/cyan hues so it stays visually distinct from Secretary.
BASE_COLORS_NO_PAR = {
    "best": "#333333",
    "secretary": "#2166ac",
    "secretary_all": "#1b9e77",
    "double_secretary": "#d73027",
    "block_rank": "#6a3d9a",
    "prophet_1sample": "#8c564b",
    "extra_tree": "#e69f00",
}
VARIANT_ORDER = ("1overe", "sqrt_n", "ln_n", "0.1n")  # canonical order for shading
EXTRA_TREE_VARIANT_ORDER = (
    "max_features=1",
    "max_features=1over3",
    "max_features=2over3",
    "max_features=all",
)
EFFORT_RUN_COLS = (
    "split_calls_mean",
    "threshold_candidates_mean",
    "gain_evaluations_mean",
    "threshold_candidates_per_split_mean",
    "gain_evaluations_per_split_mean",
    "parametric_gain_samples_mean",
    "parametric_quantile_fits_mean",
)
EFFORT_RELATIVE_COLS = (
    "effort_speedup_total",
    "effort_saved_total",
    "effort_speedup_per_split",
    "effort_saved_per_split",
)
EARLY_STOP_SPLITTERS = {
    "secretary",
    "secretary_par",
    "secretary_all",
    "double_secretary",
    "block_rank",
    "prophet_1sample",
}
SPAR_REPRESENTATIVE_KEY = "secretary_par|samples=10,q=0.5"

# secretary_par (parametric) — kept neutral so it does not compete visually with ExtraTree.
BASE_COLOR_SECRETARY_PAR = "#7f7f7f"


def _shade_hex(hex_color: str, mix_white: float) -> str:
    """Return hex color mixed with white. mix_white=0 => original, mix_white=1 => white."""
    import re
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    r = r + (1 - r) * mix_white
    g = g + (1 - g) * mix_white
    b = b + (1 - b) * mix_white
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def add_method_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized variant and method_key columns."""
    if df is None:
        return None
    out = df.copy()
    if "variant" not in out.columns:
        out["variant"] = ""
    out["variant"] = out["variant"].fillna("").astype(str)
    out["method_key"] = out["splitter"].astype(str) + "|" + out["variant"]
    return out


def filter_method_keys(df: pd.DataFrame, allowed_method_keys=None) -> pd.DataFrame:
    """Keep only the requested method keys (e.g. one S_par representative)."""
    if df is None or allowed_method_keys is None:
        return df
    out = add_method_key(df)
    allowed = set(str(k) for k in allowed_method_keys)
    return out[out["method_key"].isin(allowed)].copy()


def keep_secretary_par_representative(
    df: pd.DataFrame,
    representative_key: str = SPAR_REPRESENTATIVE_KEY,
) -> pd.DataFrame:
    """Keep all non-S_par methods plus a single selected S_par representative."""
    if df is None:
        return None
    out = add_method_key(df)
    keep = out["splitter"] != "secretary_par"
    keep |= out["method_key"] == str(representative_key)
    return out[keep].copy()


def add_effort_relative_metrics(run_level: pd.DataFrame) -> pd.DataFrame:
    """Add effort-relative metrics versus the best splitter at the same dataset/run."""
    if run_level is None or run_level.empty or "gain_evaluations_mean" not in run_level.columns:
        return run_level

    best_cols = ["run", "dataset"]
    for col in (
        "gain_evaluations_mean",
        "gain_evaluations_per_split_mean",
        "split_calls_mean",
    ):
        if col in run_level.columns:
            best_cols.append(col)
    best = run_level[run_level["splitter"] == "best"][best_cols].copy()
    best = best.rename(
        columns={
            "gain_evaluations_mean": "gain_evaluations_best",
            "gain_evaluations_per_split_mean": "gain_evaluations_per_split_best",
            "split_calls_mean": "split_calls_best",
        }
    )
    merged = run_level.merge(best, on=["run", "dataset"], how="left")
    if "gain_evaluations_mean" in merged.columns and "gain_evaluations_best" in merged.columns:
        denom = merged["gain_evaluations_mean"].replace(0, np.nan)
        merged["effort_speedup_total"] = merged["gain_evaluations_best"] / denom
        merged["effort_saved_total"] = 1.0 - (denom / merged["gain_evaluations_best"].replace(0, np.nan))
    if (
        "gain_evaluations_per_split_mean" in merged.columns
        and "gain_evaluations_per_split_best" in merged.columns
    ):
        denom = merged["gain_evaluations_per_split_mean"].replace(0, np.nan)
        merged["effort_speedup_per_split"] = merged["gain_evaluations_per_split_best"] / denom
        merged["effort_saved_per_split"] = 1.0 - (
            denom / merged["gain_evaluations_per_split_best"].replace(0, np.nan)
        )
    return merged


def get_variant_method_order_and_colors(*summary_dfs, include_secretary_par: bool = False):
    """
    From one or more variant-level summary DataFrames (with splitter, variant), build
    method_order (list of keys "splitter|variant"), method_colors, and method_labels.
    Variants get shades of the base color (darker to lighter).

    If include_secretary_par is True, append all observed secretary_par variants (after the
    main SPLITTERS_NO_PAR block), shaded from BASE_COLOR_SECRETARY_PAR.
    """
    seen = set()
    for df in summary_dfs:
        if df is None or df.empty or "splitter" not in df.columns:
            continue
        variant_col = "variant" if "variant" in df.columns else None
        for _, row in df[["splitter", variant_col or "splitter"]].drop_duplicates().iterrows():
            s = row["splitter"]
            v = row.get("variant", "") if variant_col else ""
            v = "" if pd.isna(v) else str(v).strip()
            seen.add((s, v))
    # Build ordered list: best first, then SPLITTERS_NO_PAR order, then variant order
    method_order = []
    def _variant_sort_key(splitter: str, variant: str):
        if splitter == "extra_tree":
            if variant in EXTRA_TREE_VARIANT_ORDER:
                return (0, EXTRA_TREE_VARIANT_ORDER.index(variant), variant)
            return (1, 999, variant)
        if variant in VARIANT_ORDER:
            return (0, VARIANT_ORDER.index(variant), variant)
        return (1, 999, variant)

    for splitter in SPLITTERS_NO_PAR:
        variants_here = sorted(
            [v for (s, v) in seen if s == splitter],
            key=lambda x: _variant_sort_key(splitter, x),
        )
        for v in variants_here:
            method_order.append((splitter, v))
    if include_secretary_par:
        par_variants = sorted(
            [v for (s, v) in seen if s == "secretary_par"],
            key=lambda x: x,
        )
        for v in par_variants:
            method_order.append(("secretary_par", v))
    keys = [f"{s}|{v}" for s, v in method_order]
    labels = []
    colors = {}
    for s, v in method_order:
        key = f"{s}|{v}"
        if v:
            labels.append(f"{s.replace('_', ' ')} ({v})")
        else:
            labels.append(s.replace("_", " "))
        if s == "secretary_par":
            base = BASE_COLOR_SECRETARY_PAR
        else:
            base = BASE_COLORS_NO_PAR.get(s, "#888888")
        variants_of_s = [v_ for (s_, v_) in method_order if s_ == s]
        n = len(variants_of_s)
        if n <= 1:
            colors[key] = base
        else:
            idx = variants_of_s.index(v)
            mix = 0.15 + 0.45 * (idx / (n - 1))
            colors[key] = _shade_hex(base, mix)
    return keys, colors, labels


def plot_grouped_variant_legend(
    ax,
    method_order,
    method_colors,
    method_labels,
    *,
    fontsize=7,
    header_fontsize=8,
    legend_style="patch",
):
    """
    Legend layout for figures 1 / 3 / 4:
    - Fixed columns for Secretary, S_all, S², and ExtraTree variants.
    - Final column for all remaining methods (best, block-rank, prophet, S_par).

    legend_style:
    - ``"patch"``: small rectangles (matches bar / block legend; default for figure 4).
    - ``"point"``: filled circles (matches scatter / regime maps; use for figures 1 and 3).
    """
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if method_labels is None:
        method_labels = [str(m).replace("_", " ") for m in method_order]
    pairs = list(zip(method_order, method_labels))

    families = defaultdict(list)
    other_pairs = []
    for k, lab in pairs:
        sp = str(k).split("|", 1)[0] if "|" in str(k) else str(k)
        if sp in ("secretary", "secretary_all", "double_secretary", "extra_tree"):
            families[sp].append((k, lab))
        else:
            other_pairs.append((k, lab))

    def _var_key(key):
        splitter = str(key).split("|", 1)[0] if "|" in str(key) else str(key)
        v = str(key).split("|", 1)[1] if "|" in str(key) else ""
        if splitter == "extra_tree":
            if v in EXTRA_TREE_VARIANT_ORDER:
                return (0, EXTRA_TREE_VARIANT_ORDER.index(v), v)
            return (1, 999, v)
        if v in VARIANT_ORDER:
            return (0, VARIANT_ORDER.index(v), v)
        return (1, 999, v)

    family_order = ("secretary", "secretary_all", "double_secretary", "extra_tree")
    for sp in family_order:
        families[sp].sort(key=lambda kv: _var_key(kv[0]))

    def _other_key(item):
        key, _ = item
        sp = str(key).split("|", 1)[0] if "|" in str(key) else str(key)
        variant = str(key).split("|", 1)[1] if "|" in str(key) else ""
        if sp == "best":
            return (0, 0, "")
        if sp == "block_rank":
            return (1, 0, "")
        if sp == "prophet_1sample":
            return (2, 0, "")
        if sp == "secretary_par":
            return (3, _var_key(key))
        return (4, sp, variant)

    other_pairs.sort(key=_other_key)
    n_cols = 5
    col_w = 1.0 / n_cols

    family_headers = {
        "secretary": "Secretary",
        "secretary_all": "S_all",
        "double_secretary": "S²",
        "extra_tree": "ExtraTree",
        "others": "Others",
    }

    y_header = 0.98
    y_top = 0.90
    y_bot = 0.06

    def _display_label(key, default_label):
        splitter = str(key).split("|", 1)[0] if "|" in str(key) else str(key)
        variant = str(key).split("|", 1)[1] if "|" in str(key) else ""
        if splitter in {"secretary", "secretary_all", "double_secretary"}:
            return {
                "1overe": "n/e",
                "sqrt_n": "sqrt(n)",
                "ln_n": "ln(n)",
                "0.1n": "0.1n",
            }.get(variant, default_label)
        if splitter == "extra_tree":
            return {
                "max_features=1": "mtry=1",
                "max_features=1over3": "mtry=p/3",
                "max_features=2over3": "mtry=2p/3",
                "max_features=all": "mtry=p",
            }.get(variant, default_label)
        return default_label

    def _draw_column(ci, title, entries):
        x0 = ci * col_w
        if title:
            xc = x0 + col_w * 0.5
            ax.text(
                xc,
                y_header,
                title,
                ha="center",
                va="top",
                fontsize=header_fontsize,
                fontweight="bold",
                transform=ax.transAxes,
            )
        n = len(entries)
        if n == 0:
            return
        row_h = (y_top - y_bot) / max(n, 1)
        for ri, (key, lab) in enumerate(entries):
            y = y_top - (ri + 0.5) * row_h
            color = method_colors.get(key, "#888888")
            px = x0 + col_w * 0.06
            pw = min(col_w * 0.22, 0.12)
            ph = 0.02
            text_x = px + pw + 0.01
            if legend_style == "point":
                mr = min(0.011, pw / 2.5)
                cx = px + mr
                circ = Circle(
                    (cx, y),
                    mr,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                    clip_on=False,
                )
                ax.add_patch(circ)
                text_x = px + 2.2 * mr + 0.008
            else:
                rect = Rectangle(
                    (px, y - ph / 2),
                    pw,
                    ph,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                    clip_on=False,
                )
                ax.add_patch(rect)
                text_x = px + pw + 0.01
            ax.text(
                text_x,
                y,
                _display_label(key, lab),
                ha="left",
                va="center",
                fontsize=fontsize,
                transform=ax.transAxes,
            )

    for ci, sp in enumerate(family_order):
        _draw_column(ci, family_headers.get(sp, sp.replace("_", " ")), families[sp])
    _draw_column(len(family_order), family_headers["others"], other_pairs)


def _load_run_files(indir: Path, prefix: str, run_pattern: str = "run*.csv"):
    """Load all run CSVs with given prefix (e.g. 'regression' or 'classification_gini') into one DataFrame with run index."""
    indir = Path(indir)
    frames = []
    for p in sorted(indir.glob(f"{prefix}_{run_pattern}")):
        # e.g. regression_run001.csv -> run_id 1
        stem = p.stem  # regression_run001
        run_id = int(stem.split("run")[-1])
        df = pd.read_csv(p)
        df["run"] = run_id
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    _validate_early_stop_effort(out, prefix=prefix, indir=indir)
    return out


def _validate_early_stop_effort(df: pd.DataFrame, *, prefix: str, indir: Path) -> None:
    """Fail loudly when an archive is missing effort counters for early-stop splitters."""
    if df is None or df.empty:
        return
    if "splitter" not in df.columns or "gain_evaluations_mean" not in df.columns:
        return

    splitter = df["splitter"].astype(str)
    early = splitter.isin(EARLY_STOP_SPLITTERS)
    if not early.any():
        return

    required_cols = [
        col
        for col in (
            "split_calls_mean",
            "threshold_candidates_mean",
            "gain_evaluations_mean",
        )
        if col in df.columns
    ]
    if not required_cols:
        return

    missing_mask = early.copy()
    for col in required_cols:
        missing_mask &= pd.to_numeric(df[col], errors="coerce").isna()
    missing_count = int(missing_mask.sum())
    if missing_count == 0:
        return

    sample_cols = [col for col in ("dataset", "splitter", "variant", "run") if col in df.columns]
    sample = df.loc[missing_mask, sample_cols].head(5).to_dict(orient="records")
    raise RuntimeError(
        "Corrupted benchmark archive: early-stop splitter effort counters are missing "
        f"for {missing_count} rows in '{prefix}' under {indir}. "
        "This usually means the benchmark imported a stale or mismatched treeple build. "
        f"Example rows: {sample}"
    )


def load_regression_runs(indir: Path):
    """Load all regression_run001.csv, ... and return long DataFrame with columns dataset, splitter, run, rmse_mean, fit_time_mean."""
    return _load_run_files(indir, "regression")


def load_classification_runs(indir: Path, criterion: str):
    """Load all classification_{criterion}_run001.csv, ... and return long DataFrame."""
    return _load_run_files(indir, f"classification_{criterion}")


def compute_regression_speedup_loss(long_df: pd.DataFrame):
    """
    For each (run, dataset, splitter) compute Speedup and losses relative to best.
    long_df must have columns: dataset, splitter, run, rmse_mean, fit_time_mean.
    Returns DataFrame with run, dataset, splitter, variant, speedup, loss_rmse, loss_rmse_bounded.
    - loss_rmse: relative increase (RMSE_m - RMSE_b)/RMSE_b (unbounded above).
    - loss_rmse_bounded: 1 - (RMSE_b/RMSE_m), in (-inf, 1]; negative = method better than best.
    """
    best = long_df[long_df["splitter"] == "best"][["run", "dataset", "fit_time_mean", "rmse_mean"]].rename(
        columns={"fit_time_mean": "t_best", "rmse_mean": "rmse_best"}
    )
    merged = long_df.merge(best, on=["run", "dataset"], how="left")
    merged["speedup"] = merged["t_best"] / merged["fit_time_mean"].replace(0, np.nan)
    r = merged["rmse_best"].replace(0, np.nan)
    merged["loss_rmse"] = (merged["rmse_mean"] - merged["rmse_best"]) / r
    # Bounded loss: 0 = as good as best, <0 = better than best, 1 = infinitely worse; only clip upper
    merged["loss_rmse_bounded"] = 1.0 - (merged["rmse_best"] / merged["rmse_mean"].replace(0, np.nan))
    merged["loss_rmse_bounded"] = merged["loss_rmse_bounded"].clip(upper=1.0)
    cols = ["run", "dataset", "splitter", "n_samples", "n_features", "speedup", "loss_rmse", "loss_rmse_bounded"]
    if "variant" in merged.columns:
        cols.insert(3, "variant")
    cols.extend([c for c in EFFORT_RUN_COLS if c in merged.columns])
    return add_effort_relative_metrics(merged[cols])


def compute_classification_speedup_loss(long_df: pd.DataFrame):
    """
    For each (run, dataset, splitter) compute Speedup and Loss^Acc, Loss^F1 relative to best.
    long_df must have columns: dataset, splitter, run, accuracy_mean, f1_weighted_mean, fit_time_mean.
    Returns DataFrame with run, dataset, splitter, variant, speedup, loss_acc, loss_f1.
    """
    best = long_df[long_df["splitter"] == "best"][
        ["run", "dataset", "fit_time_mean", "accuracy_mean", "f1_weighted_mean"]
    ].rename(
        columns={
            "fit_time_mean": "t_best",
            "accuracy_mean": "acc_best",
            "f1_weighted_mean": "f1_best",
        }
    )
    merged = long_df.merge(best, on=["run", "dataset"], how="left")
    merged["speedup"] = merged["t_best"] / merged["fit_time_mean"].replace(0, np.nan)
    merged["loss_acc"] = merged["acc_best"] - merged["accuracy_mean"]
    merged["loss_f1"] = merged["f1_best"] - merged["f1_weighted_mean"]
    cols = ["run", "dataset", "splitter", "n_samples", "n_features", "speedup", "loss_acc", "loss_f1"]
    if "variant" in merged.columns:
        cols.insert(3, "variant")
    cols.extend([c for c in EFFORT_RUN_COLS if c in merged.columns])
    return add_effort_relative_metrics(merged[cols])


def _iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25)


def per_dataset_median_iqr(run_level: pd.DataFrame, value_cols: list, group_cols=None):
    """
    For each (dataset, splitter) compute median and IQR across runs for each value column.
    group_cols default: ['dataset', 'splitter'].
    """
    if group_cols is None:
        group_cols = ["dataset", "splitter"]
    grp = run_level.groupby(group_cols)
    med = grp[value_cols].median().reset_index()
    med = med.rename(columns={c: f"{c}_median" for c in value_cols})
    iqr = grp[value_cols].agg(_iqr).reset_index()
    iqr = iqr.rename(columns={c: f"{c}_iqr" for c in value_cols})
    return med.merge(iqr, on=group_cols)


def _summary_value_cols(run_level: pd.DataFrame, base_cols: list[str]) -> list[str]:
    """Return base summary columns plus any available relative-effort diagnostics."""
    value_cols = list(base_cols)
    value_cols.extend(col for col in EFFORT_RELATIVE_COLS if col in run_level.columns)
    return value_cols


def get_regression_run_level(indir: Path) -> pd.DataFrame:
    """Load regression runs and compute run-level speedup and loss_rmse."""
    long_df = load_regression_runs(indir)
    if long_df is None:
        return None
    return compute_regression_speedup_loss(long_df)


def get_regression_dataset_summary(indir: Path, exclude_par: bool = False, by_variant: bool = False) -> pd.DataFrame:
    """Per (dataset, splitter) or (dataset, splitter, variant): median and IQR. exclude_par drops secretary_par; by_variant groups by variant."""
    run_level = get_regression_run_level(indir)
    if run_level is None:
        return None
    if exclude_par:
        run_level = run_level[run_level["splitter"].isin(SPLITTERS_NO_PAR)].copy()
    value_cols = _summary_value_cols(run_level, ["speedup", "loss_rmse", "loss_rmse_bounded"])
    if by_variant:
        if "variant" not in run_level.columns:
            run_level["variant"] = ""
        run_level["variant"] = run_level["variant"].fillna("").astype(str)
        return per_dataset_median_iqr(run_level, value_cols, group_cols=["dataset", "splitter", "variant"])
    return per_dataset_median_iqr(run_level, value_cols)


def get_classification_run_level(indir: Path, criterion: str) -> pd.DataFrame:
    """Load classification runs for criterion and compute run-level speedup, loss_acc, loss_f1."""
    long_df = load_classification_runs(indir, criterion)
    if long_df is None:
        return None
    return compute_classification_speedup_loss(long_df)


def get_classification_dataset_summary(indir: Path, criterion: str, exclude_par: bool = False, by_variant: bool = False) -> pd.DataFrame:
    """Per (dataset, splitter) or (dataset, splitter, variant): median and IQR. exclude_par drops secretary_par; by_variant groups by variant."""
    run_level = get_classification_run_level(indir, criterion)
    if run_level is None:
        return None
    if exclude_par:
        run_level = run_level[run_level["splitter"].isin(SPLITTERS_NO_PAR)].copy()
    value_cols = _summary_value_cols(run_level, ["speedup", "loss_acc", "loss_f1"])
    if by_variant:
        if "variant" not in run_level.columns:
            run_level["variant"] = ""
        run_level["variant"] = run_level["variant"].fillna("").astype(str)
        return per_dataset_median_iqr(run_level, value_cols, group_cols=["dataset", "splitter", "variant"])
    return per_dataset_median_iqr(run_level, value_cols)


def load_all(indir: Path, exclude_secretary_par: bool = False, by_variant: bool = False):
    """
    Load all run-level data and per-dataset summaries.
    exclude_secretary_par: drop secretary_par from run data and summaries.
    by_variant: summaries have one row per (dataset, splitter, variant).
    Returns dict with regression_run, regression_summary, classification_*_run, classification_*_summary.
    """
    indir = Path(indir)
    out = {}
    r_run = get_regression_run_level(indir)
    if exclude_secretary_par and r_run is not None:
        r_run = r_run[r_run["splitter"].isin(SPLITTERS_NO_PAR)].copy()
    out["regression_run"] = r_run
    out["regression_summary"] = get_regression_dataset_summary(indir, exclude_par=exclude_secretary_par, by_variant=by_variant)
    for crit in ("gini", "entropy"):
        c_run = get_classification_run_level(indir, crit)
        if exclude_secretary_par and c_run is not None:
            c_run = c_run[c_run["splitter"].isin(SPLITTERS_NO_PAR)].copy()
        out[f"classification_{crit}_run"] = c_run
        out[f"classification_{crit}_summary"] = get_classification_dataset_summary(indir, crit, exclude_par=exclude_secretary_par, by_variant=by_variant)
    return out


if __name__ == "__main__":
    import sys
    indir = Path(__file__).resolve().parent / "benchmark_results"
    if len(sys.argv) > 1:
        indir = Path(sys.argv[1])
    data = load_all(indir)
    for k, v in data.items():
        print(k, v.shape if v is not None else None)
    if data["regression_summary"] is not None:
        print(data["regression_summary"].head())
