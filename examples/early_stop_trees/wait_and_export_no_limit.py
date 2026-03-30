#!/usr/bin/env python
"""
Wait for the no-limit rerun outputs, then refresh paper assets.

Supports either a single regression/classification directory or sharded comma-
separated directory lists.

Typical usage:
  python examples/early_stop_trees/wait_and_export_no_limit.py \
    --reg-dirs /tmp/es_full_nolimit_shards/reg_1,/tmp/es_full_nolimit_shards/reg_2 \
    --clf-dirs /tmp/es_full_nolimit_shards/clf_1,/tmp/es_full_nolimit_shards/clf_2 \
    --wait-runs 10 \
    --outdir examples/early_stop_trees/benchmark_results \
    --article-dir RESEARCH_ARTICLE
"""

from __future__ import annotations

import argparse
import csv
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


def _parse_dir_args(single: str | None, multi: str | None) -> list[Path]:
    parts: list[str] = []
    if multi:
        parts.extend(p.strip() for p in multi.split(",") if p.strip())
    if single:
        parts.append(single.strip())
    dirs = [Path(part).resolve() for part in parts]
    if not dirs:
        raise ValueError("At least one directory must be provided.")
    return dirs


def _count_outputs(path_dir: Path) -> tuple[int, int, int]:
    return (
        len(list(path_dir.glob("regression_run*.csv"))),
        len(list(path_dir.glob("classification_gini_run*.csv"))),
        len(list(path_dir.glob("classification_entropy_run*.csv"))),
    )


def _results_ready(path_dir: Path, expected: tuple[str, ...]) -> bool:
    return all((path_dir / name).is_file() for name in expected)


def _wait_for_runs(reg_dirs: list[Path], clf_dirs: list[Path], wait_runs: int, poll_seconds: int) -> None:
    while True:
        reg_counts = {path.name: _count_outputs(path) for path in reg_dirs}
        clf_counts = {path.name: _count_outputs(path) for path in clf_dirs}
        ready_reg = all(
            counts[0] >= wait_runs and _results_ready(path, ("regression_results.csv",))
            for path, counts in ((p, reg_counts[p.name]) for p in reg_dirs)
        )
        ready_clf = all(
            counts[1] >= wait_runs
            and counts[2] >= wait_runs
            and _results_ready(path, ("classification_gini_results.csv", "classification_entropy_results.csv"))
            for path, counts in ((p, clf_counts[p.name]) for p in clf_dirs)
        )

        print("Current no-limit rerun counts:", flush=True)
        for name, (reg, gini, entropy) in {**reg_counts, **clf_counts}.items():
            print(f"  {name}: regression={reg}, gini={gini}, entropy={entropy}", flush=True)

        if ready_reg and ready_clf:
            print("All no-limit benchmark outputs are ready.", flush=True)
            return

        print(f"Waiting {poll_seconds}s before the next check...", flush=True)
        time.sleep(poll_seconds)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _concat_csv_group(source_paths: list[Path], out_path: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for src in source_paths:
        if not src.is_file():
            continue
        src_fieldnames, src_rows = _read_csv_rows(src)
        if src_rows and not fieldnames:
            fieldnames = src_fieldnames
        rows.extend(src_rows)
    if not rows or not fieldnames:
        return
    _write_csv_rows(out_path, rows, fieldnames)
    print(f"Wrote {out_path}", flush=True)


def _refresh_outdir(outdir: Path, reg_dirs: list[Path], clf_dirs: list[Path], wait_runs: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*.json"):
        for path in outdir.glob(pattern):
            path.unlink()

    for run_idx in range(1, wait_runs + 1):
        suffix = f"_run{run_idx:03d}.csv"
        _concat_csv_group(
            [path / f"regression{suffix}" for path in reg_dirs],
            outdir / f"regression{suffix}",
        )
        _concat_csv_group(
            [path / f"classification_gini{suffix}" for path in clf_dirs],
            outdir / f"classification_gini{suffix}",
        )
        _concat_csv_group(
            [path / f"classification_entropy{suffix}" for path in clf_dirs],
            outdir / f"classification_entropy{suffix}",
        )

    _concat_csv_group(
        [path / "regression_results.csv" for path in reg_dirs],
        outdir / "regression_results.csv",
    )
    _concat_csv_group(
        [path / "classification_gini_results.csv" for path in clf_dirs],
        outdir / "classification_gini_results.csv",
    )
    _concat_csv_group(
        [path / "classification_entropy_results.csv" for path in clf_dirs],
        outdir / "classification_entropy_results.csv",
    )

    for src_dir in list(reg_dirs) + list(clf_dirs):
        metadata = src_dir / "benchmark_metadata.json"
        if metadata.is_file():
            shutil.copy2(metadata, outdir / metadata.name)
            print(f"Copied {metadata} -> {outdir / metadata.name}", flush=True)
            break


def _sync_pmlb_cache(src_cache_dir: Path | None, outdir: Path) -> None:
    if src_cache_dir is None or not src_cache_dir.exists():
        return
    dst_cache_dir = outdir / "pmlb_cache"
    dst_cache_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in src_cache_dir.iterdir():
        dst = dst_cache_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} PMLB cache entries from {src_cache_dir} to {dst_cache_dir}", flush=True)


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
    ap = argparse.ArgumentParser(description="Wait for the no-limit rerun and refresh paper assets.")
    ap.add_argument("--reg-dir", type=str, default=None)
    ap.add_argument("--clf-dir", type=str, default=None)
    ap.add_argument("--reg-dirs", type=str, default=None, help="Comma-separated regression shard directories")
    ap.add_argument("--clf-dirs", type=str, default=None, help="Comma-separated classification shard directories")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--article-dir", type=str, default=None)
    ap.add_argument("--wait-runs", type=int, default=10)
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--python-executable", type=str, default=sys.executable)
    ap.add_argument(
        "--pmlb-cache-dir",
        type=str,
        default=None,
        help="Optional shared PMLB cache directory to copy into benchmark_results and reuse in analysis.",
    )
    args = ap.parse_args()

    reg_dirs = _parse_dir_args(args.reg_dir, args.reg_dirs)
    clf_dirs = _parse_dir_args(args.clf_dir, args.clf_dirs)
    outdir = Path(args.outdir).resolve()
    article_dir = Path(args.article_dir).resolve() if args.article_dir else None
    python_executable = args.python_executable
    pmlb_cache_dir = Path(args.pmlb_cache_dir).resolve() if args.pmlb_cache_dir else None

    analysis_env = os.environ.copy()
    analysis_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(analysis_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    _wait_for_runs(reg_dirs, clf_dirs, args.wait_runs, args.poll_seconds)
    _refresh_outdir(outdir, reg_dirs, clf_dirs, args.wait_runs)
    _sync_pmlb_cache(pmlb_cache_dir, outdir)

    _call(
        [python_executable, str(SCRIPT_DIR / "aggregate_benchmark_results.py"), "--indir", str(outdir)],
        cwd=SCRIPT_DIR,
        env=analysis_env,
    )
    run_analysis_cmd = [
        python_executable,
        str(SCRIPT_DIR / "run_analysis.py"),
        "--indir",
        str(outdir),
        "--outdir",
        str(SCRIPT_DIR / "analysis_additional"),
    ]
    if pmlb_cache_dir is not None:
        run_analysis_cmd += ["--pmlb-cache-dir", str(pmlb_cache_dir)]
    _call(
        run_analysis_cmd,
        cwd=SCRIPT_DIR,
        env=analysis_env,
    )

    for script_name in ("figure1.py", "figure4.py", "figure2.py", "figure3.py"):
        _call([python_executable, str(SCRIPT_DIR / script_name)], cwd=SCRIPT_DIR, env=analysis_env)

    _call(
        [
            python_executable,
            str(SCRIPT_DIR / "supp_plot_predicted_regime_maps.py"),
            "--analysis-dir",
            str(SCRIPT_DIR / "analysis_additional"),
        ],
        cwd=SCRIPT_DIR,
        env=analysis_env,
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
            env=analysis_env,
        )

    _call([python_executable, str(SCRIPT_DIR / "analysis_gain_landscape.py")], cwd=SCRIPT_DIR, env=analysis_env)
    _call([python_executable, str(SCRIPT_DIR / "supp_plot_gain_landscape.py")], cwd=SCRIPT_DIR, env=analysis_env)
    _call([python_executable, str(SCRIPT_DIR / "supp_plot_effort_metrics.py")], cwd=SCRIPT_DIR, env=analysis_env)

    for script_name in (
        "table1.py",
        "table2.py",
        "table_dataset_benchmark_summary.py",
        "table_pairwise_method_comparison.py",
        "export_paper_figures.py",
    ):
        _call([python_executable, str(SCRIPT_DIR / script_name)], cwd=SCRIPT_DIR, env=analysis_env)

    if article_dir is not None:
        _sync_article_assets(article_dir)
        _compile_article(article_dir)

    print("No-limit refresh pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
