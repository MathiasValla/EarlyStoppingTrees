"""Select a deterministic, heterogeneous subset for ensemble validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _float(row, key, default=0.0):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--task", choices=("Classification", "Regression"), required=True)
    parser.add_argument("--n-datasets", type=int, default=20)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.n_datasets % 2:
        raise ValueError("n_datasets must be even so each size stratum contributes two entries")
    with args.inventory.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["task"] == args.task]
    rows.sort(key=lambda row: (_float(row, "n_times_p"), row["dataset"]))
    bins = np.array_split(np.asarray(rows, dtype=object), args.n_datasets // 2)
    selected = []
    secondary_key = "imbalance_ratio" if args.task == "Classification" else "p_over_n"
    for bin_index, group_array in enumerate(bins, start=1):
        group = list(group_array)
        low = min(group, key=lambda row: (_float(row, secondary_key), row["dataset"]))
        high = max(group, key=lambda row: (_float(row, secondary_key), row["dataset"]))
        if high["dataset"] == low["dataset"]:
            high = max(group, key=lambda row: (_float(row, "n_features"), row["dataset"]))
        for role, row in (("low", low), ("high", high)):
            selected.append(
                {
                    "dataset": row["dataset"],
                    "task": args.task,
                    "size_stratum": bin_index,
                    "selection_role": f"{role}_{secondary_key}",
                    "n_samples": row["n_samples"],
                    "n_features": row["n_features"],
                    "n_times_p": row["n_times_p"],
                    "p_over_n": row["p_over_n"],
                    "imbalance_ratio": row.get("imbalance_ratio", ""),
                }
            )

    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_list.write_text("".join(f"{row['dataset']}\n" for row in selected))
    with args.output_manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f"Wrote {len(selected)} datasets to {args.output_list}")


if __name__ == "__main__":
    main()
