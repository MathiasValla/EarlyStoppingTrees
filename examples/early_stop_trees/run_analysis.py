#!/usr/bin/env python
"""
Convenience runner for additional analysis:
- analysis_paired.py (dataset-level paired comparisons + within/between variability)
- analysis_meta_regression.py (meta-regression + predicted regime map support)

Runs for regression + both classification criteria by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _call(python_exe: str, script_path: Path, args: list[str]):
    cmd = [python_exe, str(script_path)] + args
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default=None, help="Benchmark results directory (default: examples/early_stop_trees/benchmark_results)")
    ap.add_argument("--outdir", type=str, default=None, help="Base output directory (default: examples/early_stop_trees/analysis)")
    ap.add_argument("--ref", type=str, default="best|", help="Reference method_key for paired comparisons, e.g. 'best|'")
    ap.add_argument(
        "--tasks",
        type=str,
        default="regression,classification_gini,classification_entropy",
        help="Comma-separated tasks: regression, classification_gini, classification_entropy",
    )
    ap.add_argument("--compute-threshold-proxy", action="store_true", help="Slow: compute thresholds-per-feature proxy via PMLB fetch")
    ap.add_argument("--threshold-proxy-max-rows", type=int, default=2000, help="Row cap for threshold-proxy proxy computation")
    ap.add_argument("--max-methods", type=int, default=20, help="Max methods for PD/regime predictions")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    indir = Path(args.indir) if args.indir else (script_dir / "benchmark_results")
    outdir = Path(args.outdir) if args.outdir else (script_dir / "analysis")

    python_exe = sys.executable
    paired_script = script_dir / "analysis_paired.py"
    meta_script = script_dir / "analysis_meta_regression.py"

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        _call(
            python_exe,
            paired_script,
            [
                "--indir",
                str(indir),
                "--outdir",
                str(outdir / "paired"),
                "--task",
                task,
                "--ref",
                args.ref,
            ],
        )
        meta_args = [
            "--indir",
            str(indir),
            "--outdir",
            str(outdir / "meta"),
            "--task",
            task,
        ]
        if args.compute_threshold_proxy:
            meta_args.append("--compute-threshold-proxy")
            meta_args += ["--threshold-proxy-max-rows", str(args.threshold_proxy_max_rows)]
        _call(python_exe, meta_script, meta_args)


if __name__ == "__main__":
    main()

