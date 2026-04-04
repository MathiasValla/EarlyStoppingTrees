#!/usr/bin/env python
"""
Wait for the optimized sharded no-limit benchmark, refresh paper assets, clean
temporary artifacts, and optionally commit/push the resulting changes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

MINIMAL_EXAMPLES_KEEP = {
    "README.md",
    "README_PAPER_EXPORT.md",
    "MAIN_FIGURES",
    "SUPP_FIGURES",
    "figures",
    "tables",
    "benchmark_results",
    "aggregate_benchmark_results.py",
    "analysis_gain_landscape.py",
    "analysis_meta_regression.py",
    "analysis_paired.py",
    "analysis_utils.py",
    "benchmark_results_utils.py",
    "benchmark_secretary_pmlb.py",
    "export_paper_figures.py",
    "figure1.py",
    "figure2.py",
    "figure3.py",
    "figure4.py",
    "finalize_opt_benchmark_refresh.py",
    "plan_benchmark_shards.py",
    "run_analysis.py",
    "supp_plot_effort_metrics.py",
    "supp_plot_gain_landscape.py",
    "supp_plot_predicted_regime_maps.py",
    "supp_plot_within_between_variability.py",
    "table1.py",
    "table2.py",
    "table_dataset_benchmark_summary.py",
    "table_pairwise_method_comparison.py",
    "wait_and_export_no_limit.py",
}


def _call(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    where = cwd if cwd is not None else Path.cwd()
    print(f"Running in {where}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"Removed {path}", flush=True)


def _cleanup_article_sidecars(article_dir: Path) -> None:
    removable_patterns = ("*.log", "*.out", "*.spl", "missfont.log")
    for pattern in removable_patterns:
        for path in article_dir.glob(pattern):
            if path.name.endswith(".aux"):
                continue
            _cleanup_path(path)


def _prune_examples_tree(examples_dir: Path) -> None:
    for path in sorted(examples_dir.iterdir()):
        if path.name in MINIMAL_EXAMPLES_KEEP:
            continue
        _cleanup_path(path)


def _git_has_changes(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Finalize optimized benchmark rerun and refresh outputs.")
    ap.add_argument("--reg-dirs", type=str, required=True, help="Comma-separated regression shard directories")
    ap.add_argument("--clf-dirs", type=str, required=True, help="Comma-separated classification shard directories")
    ap.add_argument("--wait-runs", type=int, default=100)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--article-dir", type=str, required=True)
    ap.add_argument("--pmlb-cache-dir", type=str, default=None)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--python-executable", type=str, default=sys.executable)
    ap.add_argument("--commit-message", type=str, default="Refresh optimized 100-run benchmark and manuscript outputs")
    ap.add_argument("--skip-git", action="store_true", help="Skip git add/commit after refresh.")
    ap.add_argument("--push", action="store_true", help="Push origin/main after committing.")
    ap.add_argument(
        "--cleanup-paths",
        type=str,
        default="",
        help="Comma-separated files/directories to remove after refresh succeeds.",
    )
    ap.add_argument(
        "--prune-examples",
        action="store_true",
        help="Remove non-essential files from examples/early_stop_trees after refresh succeeds.",
    )
    args = ap.parse_args()

    python_executable = args.python_executable
    article_dir = Path(args.article_dir).resolve()
    cleanup_paths = [Path(part).resolve() for part in args.cleanup_paths.split(",") if part.strip()]

    wait_cmd = [
        python_executable,
        str(SCRIPT_DIR / "wait_and_export_no_limit.py"),
        "--reg-dirs",
        args.reg_dirs,
        "--clf-dirs",
        args.clf_dirs,
        "--wait-runs",
        str(args.wait_runs),
        "--outdir",
        str(Path(args.outdir).resolve()),
        "--article-dir",
        str(article_dir),
        "--poll-seconds",
        str(args.poll_seconds),
        "--python-executable",
        python_executable,
    ]
    if args.pmlb_cache_dir:
        wait_cmd += ["--pmlb-cache-dir", str(Path(args.pmlb_cache_dir).resolve())]
    _call(wait_cmd, cwd=REPO_ROOT)

    for path in cleanup_paths:
        _cleanup_path(path)
    _cleanup_article_sidecars(article_dir)
    if args.prune_examples:
        _prune_examples_tree(SCRIPT_DIR)

    if args.skip_git:
        print("Skipping git add/commit by request.", flush=True)
    elif _git_has_changes(REPO_ROOT):
        _call(["git", "add", "-A"], cwd=REPO_ROOT)
        _call(["git", "commit", "-m", args.commit_message], cwd=REPO_ROOT)
        if args.push:
            _call(["git", "push", "origin", "main"], cwd=REPO_ROOT)
    else:
        print("No git changes to commit after refresh.", flush=True)

    print("Optimized benchmark finalization completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
