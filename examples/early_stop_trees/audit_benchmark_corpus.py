"""Reconstruct and document the PMLB benchmark inclusion flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pmlb import classification_dataset_names, fetch_data, regression_dataset_names


EXPLICIT_SKIPS = {"192_vineyard", "687_sleuth_ex1605"}


def _retained_names(indir: Path, prefix: str) -> set[str]:
    names: set[str] = set()
    for path in sorted(indir.glob(f"{prefix}_run*.csv")):
        frame = pd.read_csv(path, usecols=["dataset"])
        names.update(frame["dataset"].dropna().astype(str))
    if names:
        return names
    aggregate = indir / f"{prefix}_results.csv"
    if aggregate.is_file():
        frame = pd.read_csv(aggregate, usecols=["dataset"])
        names.update(frame["dataset"].dropna().astype(str))
    return names


def _audit_task(task: str, names, retained: set[str], cache: Path, max_product: int):
    rows = []
    for dataset in names:
        row = {"task": task, "dataset": dataset}
        if dataset.startswith("_deprecated_"):
            row["status"] = "deprecated_name"
            rows.append(row)
            continue
        explicitly_skipped = dataset in EXPLICIT_SKIPS
        try:
            X, y = fetch_data(dataset, return_X_y=True, local_cache_dir=str(cache))
            X = np.asarray(X)
            y = np.asarray(y)
            row.update(
                {
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "n_product": int(X.shape[0] * X.shape[1]),
                }
            )
        except Exception as exc:
            row["status"] = "explicit_skip" if explicitly_skipped else "load_error"
            row["detail"] = str(exc)
            rows.append(row)
            continue

        if dataset in retained:
            row["status"] = "retained"
        elif explicitly_skipped:
            row["status"] = "explicit_skip"
        elif not np.isfinite(X).all() or pd.isna(y).any():
            row["status"] = "non_finite_value"
        elif max_product and row["n_product"] > max_product:
            row["status"] = "over_max_product"
        else:
            row["status"] = "eligible_not_retained"
            row["detail"] = "Execution failure or exclusion not recoverable from historical metadata"
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-indir", type=Path, required=True)
    parser.add_argument("--pmlb-cache-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-product", type=int, default=1_000_000)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    retained_regression = _retained_names(args.benchmark_indir, "regression")
    retained_gini = _retained_names(args.benchmark_indir, "classification_gini")
    retained_entropy = _retained_names(args.benchmark_indir, "classification_entropy")
    retained_classification = retained_gini & retained_entropy

    rows = _audit_task(
        "regression",
        regression_dataset_names,
        retained_regression,
        args.pmlb_cache_dir,
        args.max_product,
    )
    rows.extend(
        _audit_task(
            "classification",
            classification_dataset_names,
            retained_classification,
            args.pmlb_cache_dir,
            args.max_product,
        )
    )
    audit = pd.DataFrame(rows).sort_values(["task", "status", "dataset"])
    audit.to_csv(args.outdir / "benchmark_corpus_audit.csv", index=False)
    summary = (
        audit.groupby(["task", "status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    summary.to_csv(args.outdir / "benchmark_corpus_audit_summary.csv", index=False)
    metadata = {
        "benchmark_indir": str(args.benchmark_indir.resolve()),
        "pmlb_cache_dir": str(args.pmlb_cache_dir.resolve()),
        "max_product": args.max_product,
        "explicit_skips": sorted(EXPLICIT_SKIPS),
        "retained_regression": len(retained_regression),
        "retained_classification_common_to_gini_entropy": len(retained_classification),
        "status_definition": {
            "retained": "Present in the 100-run result corpus",
            "explicit_skip": "Named in the benchmark driver's built-in skip list",
            "over_max_product": "n_samples * n_features exceeds the active cap",
            "non_finite_value": "Input or target contains a non-finite/missing value",
            "eligible_not_retained": "Passes reconstructed filters but is absent; historical execution failure or unrecorded exclusion",
            "load_error": "PMLB entry could not be loaded from the configured cache/source",
        },
    }
    (args.outdir / "benchmark_corpus_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
