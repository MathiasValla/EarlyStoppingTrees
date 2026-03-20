#!/usr/bin/env python
"""
Meta-regression: relate dataset-level performance to dataset characteristics.

Targets (per dataset-method):
- y1 = log(speedup_median)
- y2 = loss_median

Features:
- log_n, log_p, p_over_n
- task indicators
- for classification: n_classes, imbalance_ratio
- optional thresholds_per_feature_proxy_mean (expensive; cached via PMLB cache)

Models:
- Linear regression (with standardized features)
- Spline-linear regression: SplineTransformer + Ridge

Outputs:
- CSV with out-of-sample R^2 via grouped CV by dataset (and by method)
- Coefficient tables (linear)
- Partial dependence style 1D curves for a few key covariates (spline model)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (
    TASKS,
    add_basic_dataset_meta_from_results,
    compute_classification_label_meta,
    load_task,
    method_sort_key,
)


def _ensure_dirs(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)


def _standardize(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _fit_predict_linear(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import Ridge

    Xs, mu, sd = _standardize(X)
    model = Ridge(alpha=1.0, fit_intercept=True, random_state=0)
    model.fit(Xs, y)
    return model, mu, sd


def _fit_predict_spline(X: np.ndarray, y: np.ndarray, n_knots: int = 5, degree: int = 3):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import Ridge

    Xs, mu, sd = _standardize(X)
    model = Pipeline(
        steps=[
            ("spline", SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)),
            ("ridge", Ridge(alpha=1.0, fit_intercept=True, random_state=0)),
        ]
    )
    model.fit(Xs, y)
    return model, mu, sd


def _grouped_cv_r2(df: pd.DataFrame, feat_cols: list[str], target_col: str, group_col: str, model_kind: str):
    """Grouped CV by group_col; returns per-fold and overall R^2."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score

    data = df.dropna(subset=feat_cols + [target_col, group_col]).copy()
    if data.empty or data[group_col].nunique() < 5:
        return {"r2_mean": np.nan, "r2_std": np.nan, "n": int(data.shape[0]), "n_groups": int(data[group_col].nunique())}

    X = data[feat_cols].to_numpy(dtype=float)
    y = data[target_col].to_numpy(dtype=float)
    groups = data[group_col].to_numpy()
    gkf = GroupKFold(n_splits=min(10, data[group_col].nunique()))
    r2s = []
    for tr, te in gkf.split(X, y, groups=groups):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if model_kind == "linear":
            model, mu, sd = _fit_predict_linear(Xtr, ytr)
            yhat = model.predict((Xte - mu) / sd)
        else:
            model, mu, sd = _fit_predict_spline(Xtr, ytr)
            yhat = model.predict((Xte - mu) / sd)
        r2s.append(r2_score(yte, yhat))
    r2s = np.asarray(r2s, dtype=float)
    return {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)), "n": int(data.shape[0]), "n_groups": int(data[group_col].nunique())}


def _partial_dependence_1d(model, mu, sd, X_ref: np.ndarray, feat_idx: int, grid: np.ndarray):
    Xg = np.tile(X_ref[None, :], (grid.size, 1))
    Xg[:, feat_idx] = grid
    Xg_std = (Xg - mu) / sd
    yhat = model.predict(Xg_std)
    return yhat


def _plot_pd_curves(pd_rows: pd.DataFrame, outpath: Path, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for method_key, sub in pd_rows.groupby("method_key"):
        ax.plot(sub["x"].to_numpy(), sub["yhat"].to_numpy(), label=method_key, linewidth=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(pd_rows["feature"].iloc[0])
    ax.set_ylabel(pd_rows["target"].iloc[0])
    ax.grid(True, alpha=0.25)
    # keep legend compact
    ax.legend(loc="best", fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--task", type=str, default="regression", choices=sorted(TASKS.keys()))
    ap.add_argument("--compute-threshold-proxy", action="store_true", help="Fetch X and compute thresholds-per-feature proxy (slow)")
    ap.add_argument("--threshold-proxy-max-rows", type=int, default=2000, help="Row cap when computing thresholds proxy")
    ap.add_argument("--max-methods", type=int, default=20, help="For PD plots, keep top-K methods by dataset coverage")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    indir = Path(args.indir) if args.indir else (script_dir / "benchmark_results")
    outdir = Path(args.outdir) if args.outdir else (script_dir / "analysis")
    _ensure_dirs(outdir)

    run_df, ds_sum = load_task(indir, args.task, exclude_secretary_par=True, by_variant=True)
    if ds_sum is None or ds_sum.empty:
        raise SystemExit(f"No dataset summary found for task={args.task} in {indir}")

    spec = TASKS[args.task]
    speed_med = f"{spec.speedup_col}_median"
    loss_med = f"{spec.loss_col}_median"

    # Build modeling table: one row per (dataset, method).
    # Note: ds_sum doesn't carry n_samples/n_features; we join dataset meta from run_df.
    df = ds_sum[["dataset", "method_key", speed_med, loss_med]].copy()
    df = df.dropna(subset=[speed_med, loss_med])
    df["log_speedup_median"] = np.log(np.maximum(df[speed_med].to_numpy(dtype=float), 1e-12))

    if run_df is None or run_df.empty:
        raise SystemExit(f"run_df missing/empty; cannot extract dataset meta for {args.task}")
    meta_basic = add_basic_dataset_meta_from_results(run_df)
    df = df.merge(meta_basic[["dataset", "n", "p", "p_over_n", "log_n", "log_p"]], on="dataset", how="left")
    df["task"] = args.task

    # Classification-only label meta
    if args.task.startswith("classification"):
        cache_dir = (Path(indir) / "pmlb_cache")
        label_meta = compute_classification_label_meta(
            df["dataset"].unique().tolist(),
            cache_dir=cache_dir,
            compute_threshold_proxy=args.compute_threshold_proxy,
            max_rows_for_threshold_proxy=args.threshold_proxy_max_rows,
        )
        df = df.merge(label_meta, on="dataset", how="left")
    else:
        df["n_classes"] = np.nan
        df["imbalance_ratio"] = np.nan
        df["thresholds_per_feature_proxy_mean"] = np.nan

    # Save modeling table
    df.to_csv(outdir / f"{args.task}_meta_regression_table.csv", index=False)

    # Feature set
    feat_cols = ["log_n", "log_p", "p_over_n"]
    if args.task.startswith("classification"):
        feat_cols += ["n_classes", "imbalance_ratio"]
        if args.compute_threshold_proxy:
            feat_cols += ["thresholds_per_feature_proxy_mean"]

    # Fit per method (paired across datasets) so coefficients/R2 are meaningful per method.
    cov = df.groupby("method_key")["dataset"].nunique().sort_values(ascending=False)
    keep = cov.head(args.max_methods).index.tolist()
    keep = sorted(keep, key=method_sort_key)
    df_keep = df[df["method_key"].isin(keep)].copy()

    # Grouped CV by dataset (paired structure), per method
    rows = []
    for mk in keep:
        df_m = df_keep[df_keep["method_key"] == mk].copy()
        for target in ["log_speedup_median", loss_med]:
            for model_kind in ["linear", "spline"]:
                res = _grouped_cv_r2(
                    df_m,
                    feat_cols=feat_cols,
                    target_col=target,
                    group_col="dataset",
                    model_kind=model_kind,
                )
                rows.append({"method_key": mk, "task": args.task, "target": target, "model": model_kind, **res})
    pd.DataFrame(rows).to_csv(outdir / f"{args.task}_meta_regression_cv_r2_per_method.csv", index=False)

    # Optional interpretability: store linear coefficients fitted on all data per method.
    # Coefficients are on standardized feature scale (after z-scoring).
    coeff_rows = []
    for mk in keep:
        sub = df_keep[df_keep["method_key"] == mk].copy()
        X = sub[feat_cols].to_numpy(dtype=float)
        if X.shape[0] < 10:
            continue
        for target in ["log_speedup_median", loss_med]:
            y = sub[target].to_numpy(dtype=float)
            model, mu, sd = _fit_predict_linear(X, y)
            for j, feat in enumerate(feat_cols):
                coeff_rows.append(
                    {
                        "method_key": mk,
                        "target": target,
                        "feature": feat,
                        "coef_standardized": float(model.coef_[j]),
                        "intercept_standardized": float(model.intercept_),
                    }
                )
    if coeff_rows:
        pd.DataFrame(coeff_rows).to_csv(outdir / f"{args.task}_meta_regression_linear_coeffs_per_method.csv", index=False)

    # Predicted regime map support (simple): choose best method on a grid in (log_n, p_over_n).
    # We use spline predictions; other features are fixed at each method's dataset-median value.
    try:
        # Use only methods with enough data for stable spline fitting
        regime_df = df_keep.copy()
        regime_df = regime_df.dropna(subset=feat_cols + ["log_speedup_median", loss_med]).copy()
        if regime_df.shape[0] > 30:
            qn = np.nanquantile(regime_df["log_n"].to_numpy(dtype=float), [0.05, 0.95])
            qpn = np.nanquantile(regime_df["p_over_n"].to_numpy(dtype=float), [0.05, 0.95])
            log_n_grid = np.linspace(float(qn[0]), float(qn[1]), 40)
            p_over_n_grid = np.linspace(float(qpn[0]), float(qpn[1]), 40)
            LN, PON = np.meshgrid(log_n_grid, p_over_n_grid, indexing="xy")
            LN_flat = LN.reshape(-1)
            PON_flat = PON.reshape(-1)
            grid_points = LN_flat.size

            # Allocate predictions: (n_methods, n_points)
            pred_speed = np.full((len(keep), grid_points), np.nan, dtype=float)
            pred_loss = np.full((len(keep), grid_points), np.nan, dtype=float)

            feat_idx = {c: i for i, c in enumerate(feat_cols)}
            for mi, mk in enumerate(keep):
                sub = regime_df[regime_df["method_key"] == mk].copy()
                if sub.shape[0] < 25:
                    continue
                Xsub = sub[feat_cols].to_numpy(dtype=float)

                # Fit two separate spline models (one per target) for clarity
                y_speed = sub["log_speedup_median"].to_numpy(dtype=float)
                y_loss = sub[loss_med].to_numpy(dtype=float)
                model_speed, mu_speed, sd_speed = _fit_predict_spline(Xsub, y_speed)
                model_loss, mu_loss, sd_loss = _fit_predict_spline(Xsub, y_loss)

                base = sub[feat_cols].median(numeric_only=True).to_numpy(dtype=float)
                # Build grid feature matrix
                Xg = np.tile(base[None, :], (grid_points, 1))
                if "log_n" in feat_idx:
                    Xg[:, feat_idx["log_n"]] = LN_flat
                if "p_over_n" in feat_idx:
                    Xg[:, feat_idx["p_over_n"]] = PON_flat
                # Keep log_p consistent with (n, p/n): log_p = log(n * p/n) = log_n + log(p/n)
                if "log_p" in feat_idx:
                    Xg[:, feat_idx["log_p"]] = LN_flat + np.log(np.maximum(PON_flat, 1e-12))

                Xg_speed = (Xg - mu_speed) / sd_speed
                Xg_loss = (Xg - mu_loss) / sd_loss
                pred_speed[mi, :] = model_speed.predict(Xg_speed)
                pred_loss[mi, :] = model_loss.predict(Xg_loss)

            # Choose best method per grid point (robust to NaNs)
            keep_arr = np.array(keep, dtype=object)
            valid_speed = np.isfinite(pred_speed).any(axis=0)
            valid_loss = np.isfinite(pred_loss).any(axis=0)

            best_speed_idx = np.full(grid_points, -1, dtype=int)
            best_loss_idx = np.full(grid_points, -1, dtype=int)

            if np.any(valid_speed):
                tmp_idx = np.nanargmax(pred_speed[:, valid_speed], axis=0)
                best_speed_idx[valid_speed] = tmp_idx
            if np.any(valid_loss):
                tmp_idx = np.nanargmin(pred_loss[:, valid_loss], axis=0)
                best_loss_idx[valid_loss] = tmp_idx

            best_speed_method = np.where(best_speed_idx >= 0, keep_arr[best_speed_idx], None)
            best_loss_method = np.where(best_loss_idx >= 0, keep_arr[best_loss_idx], None)

            best_speed_val = np.full(grid_points, np.nan, dtype=float)
            best_loss_val = np.full(grid_points, np.nan, dtype=float)
            if np.any(valid_speed):
                best_speed_val[valid_speed] = pred_speed[best_speed_idx[valid_speed], np.where(valid_speed)[0]]
            if np.any(valid_loss):
                best_loss_val[valid_loss] = pred_loss[best_loss_idx[valid_loss], np.where(valid_loss)[0]]

            regime_out = pd.DataFrame(
                {
                    "log_n": LN_flat,
                    "p_over_n": PON_flat,
                    "best_method_speedup": best_speed_method,
                    "best_pred_log_speedup_median": best_speed_val,
                    "best_method_loss": best_loss_method,
                    "best_pred_loss_median": best_loss_val,
                }
            )
            regime_out.to_csv(outdir / f"{args.task}_regime_map_predicted_spline.csv", index=False)
    except Exception as e:
        # Regime support should not break the full script
        print(f"[meta-regression] regime map prediction skipped due to error: {e}")

    # Partial dependence curves for log_n and p_over_n
    import matplotlib.pyplot as plt  # ensure import works early

    for target in ["log_speedup_median", loss_med]:
        pd_rows_all = []
        for mk in keep:
            sub = df_keep[df_keep["method_key"] == mk].dropna(subset=feat_cols + [target])
            if sub.shape[0] < 20:
                continue
            X = sub[feat_cols].to_numpy(dtype=float)
            y = sub[target].to_numpy(dtype=float)
            model, mu, sd = _fit_predict_spline(X, y)
            X_ref = np.nanmedian(X, axis=0)
            for feat_name in ["log_n", "p_over_n"]:
                if feat_name not in feat_cols:
                    continue
                j = feat_cols.index(feat_name)
                qlo, qhi = np.nanquantile(X[:, j], [0.05, 0.95])
                grid = np.linspace(qlo, qhi, 80)
                yhat = _partial_dependence_1d(model, mu, sd, X_ref=X_ref, feat_idx=j, grid=grid)
                for x, yh in zip(grid, yhat):
                    pd_rows_all.append({"method_key": mk, "feature": feat_name, "x": float(x), "yhat": float(yh), "target": target})

        if pd_rows_all:
            pd_rows = pd.DataFrame(pd_rows_all)
            pd_rows.to_csv(outdir / f"{args.task}_meta_regression_pd_spline_{target}.csv", index=False)
            for feat_name in pd_rows["feature"].unique():
                _plot_pd_curves(
                    pd_rows[pd_rows["feature"] == feat_name],
                    outpath=outdir / "figures" / f"{args.task}_pd_{target}_vs_{feat_name}.pdf",
                    title=f"{args.task}: spline PD of {target} vs {feat_name} (per method, others fixed at median)",
                )

    print(f"Wrote meta-regression outputs to {outdir}")


if __name__ == "__main__":
    main()

