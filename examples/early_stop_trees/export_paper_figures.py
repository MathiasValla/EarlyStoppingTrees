#!/usr/bin/env python
"""
Copy benchmark figures into MAIN_FIGURES/.

Supplementary figures (SUPP_FIGURES/) are **PNG-only** and are written in-place by the
generating scripts (merged layouts, single legends — no PDF merge step).

Prerequisites (from ``examples/early_stop_trees/``):

  python figure1.py              # main + supp 01–02 (use --no-supp to skip)
  python figure4.py              # main + supp 06–07
  python figure2.py                # main + supp 03–05
  python supp_plot_predicted_regime_maps.py   # supp 08
  python supp_plot_within_between_variability.py --task regression
  python supp_plot_within_between_variability.py --task classification_gini
  python supp_plot_within_between_variability.py --task classification_entropy

  python figure3.py              # main text figure (not supp)

Optional: ``python figure1.py --variants-of secretary_par`` for per-tag variant PDFs.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES = SCRIPT_DIR / "figures"
MAIN_DIR = SCRIPT_DIR / "MAIN_FIGURES"
SUPP_DIR = SCRIPT_DIR / "SUPP_FIGURES"


def _copy_pair(stem: str, src_dir: Path, dst_dir: Path) -> None:
    for ext in (".pdf", ".png"):
        src = src_dir / f"{stem}{ext}"
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)
            print(f"Copied {src.name} -> {dst_dir.name}/")


def main() -> int:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    for stem in (
        "figure1_pareto_all",
        "figure4_success_combined",
        "figure3_regime_combined",
    ):
        _copy_pair(stem, FIGURES, MAIN_DIR)

    expected_supp = [
        "supp_figure_01_pareto_large_small.png",
        "supp_figure_02_secretary_par_all_large_small.png",
        "supp_figure_03_ridgelines_regression.png",
        "supp_figure_04_ridgelines_classification_gini.png",
        "supp_figure_05_ridgelines_classification_entropy.png",
        "supp_figure_06_success_joint_large_small.png",
        "supp_figure_07_success_loss_only_large_small.png",
        "supp_figure_08_predicted_regime_maps.png",
        "supp_figure_09_within_whiskers_loss_speedup.png",
        "supp_figure_10_within_whiskers_loss_speedup.png",
        "supp_figure_11_within_whiskers_loss_speedup.png",
        "supp_figure_12_success_joint_all.png",
    ]
    print("--- SUPP_FIGURES/ (expected supplementary PNGs) ---")
    n_ok = 0
    for name in expected_supp:
        p = SUPP_DIR / name
        if p.is_file():
            print(f"  OK   {name}")
            n_ok += 1
        else:
            print(f"  MISS {name}", file=sys.stderr)
    print(f"Found {n_ok}/{len(expected_supp)} supplementary PNGs.")
    print("Done. MAIN_FIGURES updated; SUPP_FIGURES checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
