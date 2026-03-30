#!/usr/bin/env python
"""
Wait for sharded benchmark reruns, then merge results and refresh paper assets.

Typical usage:
  .venv_codex/bin/python examples/early_stop_trees/wait_and_export_refresh.py \
    --base-archive /tmp/benchmark_results_pre_extra_tree_refresh_20260325 \
    --refreshed-reg-dirs /tmp/es_refresh_shards/reg_1,/tmp/es_refresh_shards/reg_2,/tmp/es_refresh_shards/reg_3,/tmp/es_refresh_shards/reg_4 \
    --refreshed-clf-dirs /tmp/es_refresh_shards/clf_1,/tmp/es_refresh_shards/clf_2,/tmp/es_refresh_shards/clf_3,/tmp/es_refresh_shards/clf_4 \
    --wait-runs 25 \
    --outdir examples/early_stop_trees/benchmark_results \
    --article-dir RESEARCH_ARTICLE
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_DIR = SCRIPT_DIR / "MAIN_FIGURES"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"


def _call(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    where = cwd if cwd is not None else Path.cwd()
    print(f"Running in {where}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _count_outputs(path_dir: Path) -> tuple[int, int, int]:
    return (
        len(list(path_dir.glob("regression_run*.csv"))),
        len(list(path_dir.glob("classification_gini_run*.csv"))),
        len(list(path_dir.glob("classification_entropy_run*.csv"))),
    )


def _wait_for_shards(reg_dirs: list[Path], clf_dirs: list[Path], wait_runs: int, poll_seconds: int) -> None:
    while True:
        reg_counts = {path.name: _count_outputs(path) for path in reg_dirs}
        clf_counts = {path.name: _count_outputs(path) for path in clf_dirs}
        ready_reg = all(counts[0] >= wait_runs for counts in reg_counts.values())
        ready_clf = all(counts[1] >= wait_runs and counts[2] >= wait_runs for counts in clf_counts.values())

        print("Current shard counts:", flush=True)
        for name, (reg, gini, entropy) in {**reg_counts, **clf_counts}.items():
            print(f"  {name}: regression={reg}, gini={gini}, entropy={entropy}", flush=True)

        if ready_reg and ready_clf:
            print("All shard outputs are ready.", flush=True)
            return

        print(f"Waiting {poll_seconds}s before the next check...", flush=True)
        time.sleep(poll_seconds)


def _copy_matching(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in patterns:
        for src in sorted(src_dir.glob(pattern)):
            shutil.copy2(src, dst_dir / src.name)
            copied += 1
    print(f"Copied {copied} files from {src_dir} to {dst_dir}", flush=True)


def _sync_article_assets(article_dir: Path) -> None:
    _copy_matching(MAIN_DIR, article_dir / "MAIN_FIGURES", ("*.png", "*.pdf"))
    _copy_matching(SUPP_DIR, article_dir / "SUPP_FIGURES", ("*.png",))


def _compile_article(article_dir: Path) -> None:
    main_tex = article_dir / "main.tex"
    supp_tex = article_dir / "supp.tex"
    cls_file = article_dir / "elsarticle.cls"
    if not (main_tex.is_file() and supp_tex.is_file() and cls_file.is_file()):
        print("Skipping LaTeX build because main.tex, supp.tex, or elsarticle.cls is missing.", flush=True)
        return

    latex_env = os.environ.copy()
    latex_env.setdefault("TEXMFVAR", "/tmp/texlive-var")
    latex_env.setdefault("VARTEXFONTS", "/tmp/texfonts")
    pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]

    _call(pdflatex + ["main.tex"], cwd=article_dir, env=latex_env)
    _call(["bibtex", "main"], cwd=article_dir, env=latex_env)
    _call(pdflatex + ["main.tex"], cwd=article_dir, env=latex_env)
    _call(pdflatex + ["main.tex"], cwd=article_dir, env=latex_env)
    _call(pdflatex + ["supp.tex"], cwd=article_dir, env=latex_env)


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for sharded benchmark reruns and refresh paper assets.")
    ap.add_argument("--base-archive", type=str, required=True)
    ap.add_argument("--refreshed-reg-dirs", type=str, required=True, help="Comma-separated shard directories")
    ap.add_argument("--refreshed-clf-dirs", type=str, required=True, help="Comma-separated shard directories")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--article-dir", type=str, default=None)
    ap.add_argument("--wait-runs", type=int, default=25)
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--python-executable", type=str, default=sys.executable)
    args = ap.parse_args()

    base_archive = Path(args.base_archive).resolve()
    reg_dirs = [Path(part).resolve() for part in args.refreshed_reg_dirs.split(",") if part]
    clf_dirs = [Path(part).resolve() for part in args.refreshed_clf_dirs.split(",") if part]
    outdir = Path(args.outdir).resolve()
    article_dir = Path(args.article_dir).resolve() if args.article_dir else None
    python_executable = args.python_executable

    _wait_for_shards(reg_dirs, clf_dirs, args.wait_runs, args.poll_seconds)

    _call(
        [
            python_executable,
            str(SCRIPT_DIR / "merge_benchmark_archives.py"),
            "--base-archive",
            str(base_archive),
            "--refreshed-reg-dirs",
            ",".join(str(path) for path in reg_dirs),
            "--refreshed-clf-dirs",
            ",".join(str(path) for path in clf_dirs),
            "--outdir",
            str(outdir),
        ],
        cwd=SCRIPT_DIR,
    )
    _call(
        [python_executable, str(SCRIPT_DIR / "aggregate_benchmark_results.py"), "--indir", str(outdir)],
        cwd=SCRIPT_DIR,
    )
    _call(
        [
            python_executable,
            str(SCRIPT_DIR / "run_analysis.py"),
            "--indir",
            str(outdir),
            "--outdir",
            str(SCRIPT_DIR / "analysis_additional"),
        ],
        cwd=SCRIPT_DIR,
    )

    for script_name in ("figure1.py", "figure4.py", "figure2.py", "figure3.py"):
        _call([python_executable, str(SCRIPT_DIR / script_name)], cwd=SCRIPT_DIR)

    _call(
        [
            python_executable,
            str(SCRIPT_DIR / "supp_plot_predicted_regime_maps.py"),
            "--analysis-dir",
            str(SCRIPT_DIR / "analysis_additional"),
        ],
        cwd=SCRIPT_DIR,
    )

    for task in ("regression", "classification_gini", "classification_entropy"):
        _call(
            [
                python_executable,
                str(SCRIPT_DIR / "supp_plot_within_between_variability.py"),
                "--analysis-dir",
                str(SCRIPT_DIR / "analysis_additional"),
                "--task",
                task,
            ],
            cwd=SCRIPT_DIR,
        )

    _call([python_executable, str(SCRIPT_DIR / "analysis_gain_landscape.py")], cwd=SCRIPT_DIR)
    _call([python_executable, str(SCRIPT_DIR / "supp_plot_gain_landscape.py")], cwd=SCRIPT_DIR)
    _call([python_executable, str(SCRIPT_DIR / "supp_plot_effort_metrics.py")], cwd=SCRIPT_DIR)

    for script_name in (
        "table1.py",
        "table2.py",
        "table_dataset_benchmark_summary.py",
        "table_pairwise_method_comparison.py",
        "export_paper_figures.py",
    ):
        _call([python_executable, str(SCRIPT_DIR / script_name)], cwd=SCRIPT_DIR)

    if article_dir is not None:
        _sync_article_assets(article_dir)
        _compile_article(article_dir)

    print("Refresh pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
