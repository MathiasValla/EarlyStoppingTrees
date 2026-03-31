#!/usr/bin/env python
"""
Wait for secretary-only rerun shards, patch those rows into an existing
benchmark_results archive, then regenerate analysis assets and article outputs.
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


def _parse_dir_args(multi: str | None) -> list[Path]:
    parts = [p.strip() for p in (multi or "").split(",") if p.strip()]
    dirs = [Path(part).resolve() for part in parts]
    if not dirs:
        raise ValueError("At least one shard directory must be provided.")
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

        print("Current secretary rerun counts:", flush=True)
        for name, (reg, gini, entropy) in {**reg_counts, **clf_counts}.items():
            print(f"  {name}: regression={reg}, gini={gini}, entropy={entropy}", flush=True)

        if ready_reg and ready_clf:
            print("All secretary rerun outputs are ready.", flush=True)
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


def _combine_fieldnames(*fieldname_lists: list[str]) -> list[str]:
    fieldnames: list[str] = []
    for current in fieldname_lists:
        for name in current:
            if name not in fieldnames:
                fieldnames.append(name)
    return fieldnames


def _sort_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("criterion", "")),
        str(row.get("splitter", "")),
        str(row.get("variant", "")),
    )


def _patched_rows(base_rows: list[dict[str, str]], shard_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    kept = [row for row in base_rows if row.get("splitter") != "secretary"]
    patched = kept + shard_rows
    patched.sort(key=_sort_key)
    return patched


def _concat_rows(source_paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for path in source_paths:
        if not path.is_file():
            continue
        src_fieldnames, src_rows = _read_csv_rows(path)
        fieldnames = _combine_fieldnames(fieldnames, src_fieldnames)
        rows.extend(src_rows)
    return fieldnames, rows


def _patch_single_csv(base_path: Path, shard_paths: list[Path]) -> None:
    base_fieldnames, base_rows = _read_csv_rows(base_path)
    shard_fieldnames, shard_rows = _concat_rows(shard_paths)
    if not shard_rows:
        raise RuntimeError(f"No secretary rerun rows found for {base_path.name}")
    fieldnames = _combine_fieldnames(base_fieldnames, shard_fieldnames)
    patched_rows = _patched_rows(base_rows, shard_rows)
    _write_csv_rows(base_path, patched_rows, fieldnames)
    print(f"Patched {base_path}", flush=True)


def _patch_benchmark_archive(outdir: Path, reg_dirs: list[Path], clf_dirs: list[Path], wait_runs: int) -> None:
    for run_idx in range(1, wait_runs + 1):
        suffix = f"_run{run_idx:03d}.csv"
        _patch_single_csv(
            outdir / f"regression{suffix}",
            [path / f"regression{suffix}" for path in reg_dirs],
        )
        _patch_single_csv(
            outdir / f"classification_gini{suffix}",
            [path / f"classification_gini{suffix}" for path in clf_dirs],
        )
        _patch_single_csv(
            outdir / f"classification_entropy{suffix}",
            [path / f"classification_entropy{suffix}" for path in clf_dirs],
        )

    _patch_single_csv(
        outdir / "regression_results.csv",
        [path / "regression_results.csv" for path in reg_dirs],
    )
    _patch_single_csv(
        outdir / "classification_gini_results.csv",
        [path / "classification_gini_results.csv" for path in clf_dirs],
    )
    _patch_single_csv(
        outdir / "classification_entropy_results.csv",
        [path / "classification_entropy_results.csv" for path in clf_dirs],
    )

    for src_dir in list(reg_dirs) + list(clf_dirs):
        metadata = src_dir / "benchmark_metadata.json"
        if metadata.is_file():
            shutil.copy2(metadata, outdir / "benchmark_metadata.secretary_fix.json")
            print(f"Copied {metadata} -> {outdir / 'benchmark_metadata.secretary_fix.json'}", flush=True)
            break


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
    tables_src = SCRIPT_DIR / "tables"
    tables_dst = article_dir / "TABLES"
    if tables_src.is_dir():
        _copy_matching(tables_src, tables_dst, ("*.tex", "*.csv"))


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
    ap = argparse.ArgumentParser(description="Wait for secretary reruns, patch benchmark archive, and refresh paper assets.")
    ap.add_argument("--reg-dirs", type=str, required=True, help="Comma-separated regression shard directories")
    ap.add_argument("--clf-dirs", type=str, required=True, help="Comma-separated classification shard directories")
    ap.add_argument("--outdir", type=str, required=True, help="Existing benchmark_results directory to patch in place")
    ap.add_argument("--article-dir", type=str, default=None)
    ap.add_argument("--wait-runs", type=int, default=50)
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--python-executable", type=str, default=sys.executable)
    ap.add_argument("--pmlb-cache-dir", type=str, default=None)
    args = ap.parse_args()

    reg_dirs = _parse_dir_args(args.reg_dirs)
    clf_dirs = _parse_dir_args(args.clf_dirs)
    outdir = Path(args.outdir).resolve()
    article_dir = Path(args.article_dir).resolve() if args.article_dir else None
    python_executable = args.python_executable

    analysis_env = os.environ.copy()
    analysis_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(analysis_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    _wait_for_runs(reg_dirs, clf_dirs, args.wait_runs, args.poll_seconds)
    _patch_benchmark_archive(outdir, reg_dirs, clf_dirs, args.wait_runs)

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
    if args.pmlb_cache_dir:
        run_analysis_cmd += ["--pmlb-cache-dir", args.pmlb_cache_dir]
    _call(run_analysis_cmd, cwd=SCRIPT_DIR, env=analysis_env)

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

    print("Secretary refresh pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
