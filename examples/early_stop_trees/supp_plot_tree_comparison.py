#!/usr/bin/env python
"""Generate Supplementary Figure 2: fold-specific complete-tree comparison.

The dataset is selected reproducibly from retained classification datasets whose
sample and feature counts both lie within the corresponding inventory IQR. The
selection is made after sorting dataset names, using a fixed NumPy RNG seed.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INVENTORY_PATH = SCRIPT_DIR / "tables" / "dataset_benchmark_inventory.csv"
PMLB_CACHE_DIR = SCRIPT_DIR / "benchmark_results" / "pmlb_cache"
LOCAL_OUTPUT_PATH = SCRIPT_DIR / "SUPP_FIGURES" / "supp_figure_02_tree_comparison.png"
OUTPUT_PATH = REPO_ROOT / "RESEARCH_ARTICLE" / "SUPP_FIGURES" / "supp_figure_02_tree_comparison.png"

# Keep plotting caches outside the repository when the default user cache is unavailable.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "early_stop_trees_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmlb import fetch_data
from sklearn.model_selection import StratifiedKFold
from treeple._lib.sklearn.tree import plot_tree
from treeple.tree import EarlyStopDecisionTreeClassifier, ExtraTreeClassifier


SELECTION_RNG_SEED = 20260817
MODEL_SEED = 42
N_FOLDS = 5
FOLD_INDEX = 0
CRITERION = "gini"
MAX_TREE_DEPTH = 20
DISPLAY_LEVELS = 3


def _load_classification_inventory(path: Path) -> list[dict[str, object]]:
    """Load retained classification dataset metadata from the paper inventory."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing retained-dataset inventory: {path}")

    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["task"] != "Classification":
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "n_samples": int(row["n_samples"]),
                    "n_features": int(row["n_features"]),
                }
            )
    if not rows:
        raise ValueError(f"No retained classification datasets found in {path}")
    return rows


def _select_medium_dataset(
    rows: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Select one dataset with fixed RNG from the joint inventory-IQR subset."""
    n_values = np.asarray([row["n_samples"] for row in rows], dtype=float)
    p_values = np.asarray([row["n_features"] for row in rows], dtype=float)
    n_q1, n_q3 = np.quantile(n_values, [0.25, 0.75])
    p_q1, p_q3 = np.quantile(p_values, [0.25, 0.75])

    eligible = []
    for row in rows:
        within_n_iqr = n_q1 <= int(row["n_samples"]) <= n_q3
        within_p_iqr = p_q1 <= int(row["n_features"]) <= p_q3
        if within_n_iqr and within_p_iqr:
            eligible.append(row)
    eligible.sort(key=lambda row: str(row["dataset"]))
    if not eligible:
        raise ValueError("No retained classification dataset lies within both inventory IQRs")

    rng = np.random.default_rng(SELECTION_RNG_SEED)
    selected = eligible[int(rng.integers(len(eligible)))]
    selection = {
        "rng_seed": SELECTION_RNG_SEED,
        "eligible_dataset_count": len(eligible),
        "n_samples_iqr": [float(n_q1), float(n_q3)],
        "n_features_iqr": [float(p_q1), float(p_q3)],
        "rule": "n_samples and n_features both within retained-classification IQRs",
    }
    return selected, selection


def _fetch_selected_dataset(
    selected: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch the selected PMLB dataset and verify it against the inventory."""
    dataset = str(selected["dataset"])
    try:
        frame = fetch_data(dataset, local_cache_dir=str(PMLB_CACHE_DIR))
    except Exception as exc:
        raise RuntimeError(
            "PMLB fetch failed for the reproducibly selected dataset "
            f"{dataset!r} using cache {PMLB_CACHE_DIR}: {exc!r}"
        ) from exc

    if "target" not in frame.columns:
        raise ValueError(f"PMLB dataset {dataset!r} has no 'target' column")
    feature_names = [str(column) for column in frame.columns if column != "target"]
    X = frame.loc[:, feature_names].to_numpy(dtype=np.float64, copy=True)
    y = frame.loc[:, "target"].to_numpy(dtype=np.intp, copy=True)

    expected_shape = (int(selected["n_samples"]), int(selected["n_features"]))
    if X.shape != expected_shape:
        raise ValueError(
            "Inventory/data shape mismatch for "
            f"{dataset!r}: expected {expected_shape}, got {X.shape}"
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Selected dataset {dataset!r} contains non-finite values")
    return X, y, feature_names


def _make_estimators() -> list[tuple[str, str, object]]:
    """Construct the four publication-facing estimators from the benchmark protocol."""
    common = {
        "criterion": CRITERION,
        "random_state": MODEL_SEED,
        "max_depth": MAX_TREE_DEPTH,
    }
    return [
        (
            "(a)",
            "Exhaustive CART",
            EarlyStopDecisionTreeClassifier(splitter="best", **common),
        ),
        (
            "(b)",
            r"$S_{\mathrm{all}}(1/e)$",
            EarlyStopDecisionTreeClassifier(splitter="secretary_all", split_search={}, **common),
        ),
        (
            "(c)",
            r"$S^2(1/e)$",
            EarlyStopDecisionTreeClassifier(splitter="double_secretary", split_search={}, **common),
        ),
        (
            "(d)",
            r"ERT ($m_{\mathrm{try}}=p$)",
            ExtraTreeClassifier(max_features=None, **common),
        ),
    ]


def _build_caption(metadata: dict[str, object]) -> str:
    """Return a manuscript-ready caption carrying the reproducibility metadata."""
    return (
        "First three levels of complete classification trees fitted on the same "
        f"cross-validation training fold for {metadata['dataset']} "
        f"(n={metadata['n_samples']}, p={metadata['n_features']}). "
        "The dataset was drawn with a fixed RNG from retained classification datasets "
        "whose n and p both lie within their inventory IQRs. All methods use the Gini "
        f"criterion, estimator seed {metadata['model_seed']}, and fold "
        f"{metadata['fold_one_based']} of {metadata['n_folds']}. Trees are grown with "
        f"maximum depth {metadata['max_tree_depth']}; only the first "
        f"{metadata['display_levels']} levels are displayed."
    )


def _render_figure(
    estimators: list[tuple[str, str, object]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, object],
) -> None:
    """Fit the four complete trees and render their first three levels."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "figure.facecolor": "#f7f5ef",
            "axes.facecolor": "#fffdf8",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 11.5), facecolor="#f7f5ef")
    title_colors = ("#263238", "#00796b", "#b23a2b", "#b26a00")
    class_names = [f"Class {value}" for value in np.unique(y_train)]

    fitted_summaries: list[dict[str, object]] = []
    for ax, (panel, label, estimator), color in zip(
        axes.flat, estimators, title_colors, strict=True
    ):
        estimator.fit(X_train, y_train)
        fitted_summaries.append(
            {
                "method": label.replace("$", ""),
                "tree_depth": int(estimator.tree_.max_depth),
                "node_count": int(estimator.tree_.node_count),
            }
        )
        artists = plot_tree(
            estimator,
            ax=ax,
            max_depth=DISPLAY_LEVELS - 1,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            impurity=False,
            precision=2,
            fontsize=7,
        )
        for artist in artists:
            patch = artist.get_bbox_patch()
            if patch is not None:
                patch.set_edgecolor("#56636b")
                patch.set_linewidth(0.65)
        ax.set_title(f"{panel}  {label}", loc="left", pad=10, fontsize=13, color=color)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#c7c1b5")
            spine.set_linewidth(0.8)

    dataset = str(metadata["dataset"])
    fig.suptitle(
        "Fold-specific complete trees: first three levels",
        x=0.5,
        y=0.987,
        fontsize=17,
        fontweight="bold",
        color="#1f2933",
    )
    fig.text(
        0.5,
        0.954,
        f"Dataset: {dataset}",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#35434b",
    )
    fig.text(
        0.5,
        0.930,
        (
            f"n={metadata['n_samples']:,}  |  p={metadata['n_features']}  |  "
            f"Gini  |  seed={metadata['model_seed']}  |  "
            f"fold={metadata['fold_one_based']}/{metadata['n_folds']}  |  "
            f"training n={metadata['n_train']:,}"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#52616b",
    )
    selection = metadata["selection"]
    fig.text(
        0.5,
        0.018,
        (
            "Dataset selected by fixed RNG seed "
            f"{selection['rng_seed']} from {selection['eligible_dataset_count']} retained "
            "classification datasets with both n and p inside their inventory IQRs."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#5f6b72",
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.875,
        bottom=0.065,
        hspace=0.24,
        wspace=0.08,
    )

    for output_path in (LOCAL_OUTPUT_PATH, OUTPUT_PATH):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    metadata["fitted_trees"] = fitted_summaries


def main() -> int:
    """Select the dataset, fit one common fold, and write the supplementary figure."""
    inventory = _load_classification_inventory(INVENTORY_PATH)
    selected, selection = _select_medium_dataset(inventory)
    X, y, feature_names = _fetch_selected_dataset(selected)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)
    train_indices, test_indices = list(cv.split(X, y))[FOLD_INDEX]
    metadata: dict[str, object] = {
        "dataset": str(selected["dataset"]),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "criterion": CRITERION,
        "model_seed": MODEL_SEED,
        "fold_zero_based": FOLD_INDEX,
        "fold_one_based": FOLD_INDEX + 1,
        "n_folds": N_FOLDS,
        "n_train": int(train_indices.size),
        "n_test": int(test_indices.size),
        "max_tree_depth": MAX_TREE_DEPTH,
        "display_levels": DISPLAY_LEVELS,
        "selection": selection,
    }

    _render_figure(
        _make_estimators(),
        X[train_indices],
        y[train_indices],
        feature_names,
        metadata,
    )
    print(f"Wrote {OUTPUT_PATH}")
    print("Caption metadata:")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("Suggested caption:")
    print(_build_caption(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
