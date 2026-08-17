#!/usr/bin/env python3
"""Export compact supplementary tables from the inferential-analysis bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_INFERENCE_DIR = Path("examples/early_stop_trees/inference_results")
DEFAULT_ARTICLE_DIR = Path("RESEARCH_ARTICLE/TABLES")
PRODUCTION_BOOTSTRAPS = 10_000
REQUIRED_CONFIDENCE = 0.95

TASK_ORDER = (
    "regression",
    "classification_gini",
    "classification_entropy",
)
TASK_LABELS = {
    "regression": "Regression",
    "classification_gini": "Classification (Gini)",
    "classification_entropy": "Classification (Entropy)",
}

METRIC_ORDER = (
    "time_saved_pct",
    "predictive_loss_pct",
    "effort_saved_pct",
)
METRIC_LABELS = {
    "time_saved_pct": "Time saved",
    "predictive_loss_pct": "Predictive loss",
    "effort_saved_pct": "Effort saved",
}

BASELINE_METHOD = "best|"
METHOD_ORDER = (
    "secretary|1overe",
    "double_secretary|1overe",
    "secretary_all|1overe",
    "block_rank|",
    "prophet_1sample|",
    "extra_tree|max_features=1",
    "extra_tree|max_features=1over3",
    "extra_tree|max_features=2over3",
    "extra_tree|max_features=all",
)
CONFIRMATORY_METHODS = (BASELINE_METHOD, *METHOD_ORDER)

METHOD_LABELS_CSV = {
    "secretary|1overe": "S (f=1/e)",
    "double_secretary|1overe": "S^2 (f=1/e)",
    "secretary_all|1overe": "S_all (f=1/e)",
    "block_rank|": "Rank-inspired",
    "prophet_1sample|": "Prophet-style",
    "extra_tree|max_features=1": "ERT (m_try=1)",
    "extra_tree|max_features=1over3": "ERT (m_try=p/3)",
    "extra_tree|max_features=2over3": "ERT (m_try=2p/3)",
    "extra_tree|max_features=all": "ERT (m_try=p)",
}
METHOD_LABELS_TEX = {
    "secretary|1overe": r"$S$ ($f=1/e$)",
    "double_secretary|1overe": r"$S^{2}$ ($f=1/e$)",
    "secretary_all|1overe": r"$S_{\mathrm{all}}$ ($f=1/e$)",
    "block_rank|": "Rank-inspired",
    "prophet_1sample|": "Prophet-style",
    "extra_tree|max_features=1": r"ERT ($m_{\mathrm{try}}=1$)",
    "extra_tree|max_features=1over3": r"ERT ($m_{\mathrm{try}}=p/3$)",
    "extra_tree|max_features=2over3": r"ERT ($m_{\mathrm{try}}=2p/3$)",
    "extra_tree|max_features=all": r"ERT ($m_{\mathrm{try}}=p$)",
}


class InputValidationError(ValueError):
    """Raised when the inference bundle cannot support the requested tables."""


def _read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise InputValidationError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(set(required_columns) - columns)
        if missing:
            raise InputValidationError(
                f"{path.name} is missing columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise InputValidationError(f"Input table is empty: {path}")
    return rows


def _as_float(row: Mapping[str, str], column: str, context: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise InputValidationError(
            f"Invalid {column!r} in {context}: {row.get(column)!r}"
        ) from error
    if not math.isfinite(value):
        raise InputValidationError(f"Non-finite {column!r} in {context}")
    return value


def _as_int(row: Mapping[str, str], column: str, context: str) -> int:
    value = _as_float(row, column, context)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise InputValidationError(f"Non-integral {column!r} in {context}: {value}")
    return rounded


def _as_bool(row: Mapping[str, str], column: str, context: str) -> bool:
    value = str(row.get(column, "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise InputValidationError(
        f"Invalid Boolean {column!r} in {context}: {row.get(column)!r}"
    )


def _load_metadata(inference_dir: Path, allow_nonproduction: bool) -> tuple[int, float]:
    path = inference_dir / "inference_metadata.json"
    if not path.is_file():
        raise InputValidationError(f"Missing required input: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        n_bootstrap = int(metadata["n_bootstrap"])
        confidence = float(metadata["confidence"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InputValidationError(f"Invalid inference metadata: {path}") from error

    tasks = set(metadata.get("tasks", ()))
    if tasks != set(TASK_ORDER):
        raise InputValidationError(
            "Inference metadata must contain exactly the three expected tasks"
        )
    if not math.isclose(confidence, REQUIRED_CONFIDENCE, rel_tol=0.0, abs_tol=1e-12):
        raise InputValidationError(
            f"Expected 95% intervals, found confidence={confidence:g}"
        )
    if n_bootstrap != PRODUCTION_BOOTSTRAPS and not allow_nonproduction:
        raise InputValidationError(
            f"Production tables require n_bootstrap={PRODUCTION_BOOTSTRAPS}; "
            f"found {n_bootstrap}. Pass --allow-nonproduction only for QA."
        )
    if n_bootstrap < 2:
        raise InputValidationError("n_bootstrap must be at least 2")

    recorded = (
        metadata.get("multiple_comparisons", {}).get("confirmatory_methods")
    )
    if not isinstance(recorded, list) or tuple(recorded) != CONFIRMATORY_METHODS:
        raise InputValidationError(
            "Metadata must record the expected ordered 10-method confirmatory family"
        )
    return n_bootstrap, confidence


def _unique_grid(
    rows: Sequence[Mapping[str, str]],
    keys: Sequence[str],
    context: str,
) -> dict[tuple[str, ...], Mapping[str, str]]:
    grid: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(str(row[column]) for column in keys)
        if key in grid:
            raise InputValidationError(f"Duplicate {context} row for {key}")
        grid[key] = row
    return grid


def _validate_bootstrap_row(
    row: Mapping[str, str], expected: int, context: str
) -> None:
    actual = _as_int(row, "n_bootstrap", context)
    if actual != expected:
        raise InputValidationError(
            f"Bootstrap mismatch in {context}: metadata={expected}, row={actual}"
        )


def _build_omnibus(inference_dir: Path) -> list[dict[str, object]]:
    rows = _read_csv(
        inference_dir / "omnibus_tests.csv",
        {
            "task",
            "metric",
            "test",
            "statistic",
            "df",
            "pvalue",
            "kendalls_w",
            "n_complete_datasets",
            "n_methods",
            "unit",
        },
    )
    grid = _unique_grid(rows, ("task", "metric"), "omnibus")
    output: list[dict[str, object]] = []
    for task in TASK_ORDER:
        for metric in METRIC_ORDER:
            key = (task, metric)
            if key not in grid:
                raise InputValidationError(f"Missing omnibus row for {key}")
            row = grid[key]
            context = f"omnibus {task}/{metric}"
            n_methods = _as_int(row, "n_methods", context)
            degrees_freedom = _as_int(row, "df", context)
            if row["test"] != "friedman" or n_methods != 10 or degrees_freedom != 9:
                raise InputValidationError(
                    f"{context} must be a 10-method Friedman test with df=9"
                )
            output.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "unit": row["unit"],
                    "n_complete_datasets": _as_int(
                        row, "n_complete_datasets", context
                    ),
                    "n_methods": n_methods,
                    "statistic": _as_float(row, "statistic", context),
                    "df": degrees_freedom,
                    "pvalue": _as_float(row, "pvalue", context),
                    "kendalls_w": _as_float(row, "kendalls_w", context),
                }
            )
    return output


def _build_joint(inference_dir: Path) -> list[dict[str, object]]:
    rows = _read_csv(
        inference_dir / "joint_time_loss_margin_tests.csv",
        {
            "task",
            "method_key",
            "loss_margin_pct",
            "is_primary_margin",
            "n_complete_datasets",
            "runtime_n_non_ties",
            "runtime_n_positive",
            "loss_n_non_ties",
            "loss_n_below_margin",
            "joint_n_success",
            "joint_n_failure",
            "joint_success_pvalue_holm",
            "multiplicity_family_size",
            "supports_primary_joint_claim_0.05",
        },
    )
    primary = [
        row
        for row in rows
        if math.isclose(
            _as_float(row, "loss_margin_pct", "joint test"),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    grid = _unique_grid(primary, ("task", "method_key"), "primary joint test")
    output: list[dict[str, object]] = []
    for task in TASK_ORDER:
        for method_key in METHOD_ORDER:
            key = (task, method_key)
            if key not in grid:
                raise InputValidationError(f"Missing primary joint-test row for {key}")
            row = grid[key]
            context = f"primary joint test {task}/{method_key}"
            if not _as_bool(row, "is_primary_margin", context):
                raise InputValidationError(f"Primary-margin flag is false in {context}")
            if _as_int(row, "multiplicity_family_size", context) != len(METHOD_ORDER):
                raise InputValidationError(f"Unexpected Holm family size in {context}")

            n_complete = _as_int(row, "n_complete_datasets", context)
            runtime_total = _as_int(row, "runtime_n_non_ties", context)
            runtime_positive = _as_int(row, "runtime_n_positive", context)
            loss_total = _as_int(row, "loss_n_non_ties", context)
            loss_below = _as_int(row, "loss_n_below_margin", context)
            joint_success = _as_int(row, "joint_n_success", context)
            joint_failure = _as_int(row, "joint_n_failure", context)
            if not (0 <= runtime_positive <= runtime_total <= n_complete):
                raise InputValidationError(f"Invalid runtime counts in {context}")
            if not (0 <= loss_below <= loss_total <= n_complete):
                raise InputValidationError(f"Invalid loss counts in {context}")
            if joint_success + joint_failure != n_complete:
                raise InputValidationError(f"Invalid simultaneous counts in {context}")

            pvalue = _as_float(row, "joint_success_pvalue_holm", context)
            if not 0.0 <= pvalue <= 1.0:
                raise InputValidationError(f"Adjusted p-value outside [0, 1] in {context}")
            supported = _as_bool(
                row, "supports_primary_joint_claim_0.05", context
            )
            if supported != (pvalue < 0.05):
                raise InputValidationError(f"Support flag disagrees with Holm p in {context}")

            output.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "method_key": method_key,
                    "method": METHOD_LABELS_CSV[method_key],
                    "loss_margin_pct": 1.0,
                    "n_complete_datasets": n_complete,
                    "runtime_n_positive": runtime_positive,
                    "runtime_n_non_ties": runtime_total,
                    "loss_n_below_margin": loss_below,
                    "loss_n_non_ties": loss_total,
                    "joint_n_success": joint_success,
                    "joint_n_failure": joint_failure,
                    "joint_success_pvalue_holm": pvalue,
                    "supports_primary_joint_claim_0.05": supported,
                }
            )
    return output


def _build_timing(
    inference_dir: Path, n_bootstrap: int, confidence: float
) -> list[dict[str, object]]:
    global_rows = _read_csv(
        inference_dir / "global_intervals.csv",
        {
            "task",
            "method_key",
            "metric",
            "estimand",
            "estimate",
            "confidence",
            "n_datasets",
            "n_bootstrap",
            "hierarchical_ci_low",
            "hierarchical_ci_high",
        },
    )
    standard_rows = [
        row
        for row in global_rows
        if row["metric"] == "time_saved_pct"
        and row["estimand"] == "centroid_mean"
        and row["task"] in TASK_ORDER
        and row["method_key"] in METHOD_ORDER
    ]
    standard = _unique_grid(
        standard_rows, ("task", "method_key"), "standard timing interval"
    )

    block_rows = _read_csv(
        inference_dir / "timing_block_centroid_intervals.csv",
        {
            "task",
            "method_key",
            "block_length_runs",
            "estimate",
            "confidence",
            "n_datasets",
            "n_bootstrap",
            "hierarchical_block_ci_low",
            "hierarchical_block_ci_high",
        },
    )
    selected_blocks = [
        row
        for row in block_rows
        if row["task"] in TASK_ORDER
        and row["method_key"] in METHOD_ORDER
        and _as_int(row, "block_length_runs", "timing block") in {5, 10}
    ]
    blocks = _unique_grid(
        selected_blocks,
        ("task", "method_key", "block_length_runs"),
        "timing block interval",
    )

    output: list[dict[str, object]] = []
    for task in TASK_ORDER:
        for method_key in METHOD_ORDER:
            key = (task, method_key)
            if key not in standard:
                raise InputValidationError(f"Missing standard timing interval for {key}")
            row = standard[key]
            context = f"standard timing interval {task}/{method_key}"
            _validate_bootstrap_row(row, n_bootstrap, context)
            row_confidence = _as_float(row, "confidence", context)
            if not math.isclose(
                row_confidence, confidence, rel_tol=0.0, abs_tol=1e-12
            ):
                raise InputValidationError(f"Confidence mismatch in {context}")
            estimate = _as_float(row, "estimate", context)
            standard_low = _as_float(row, "hierarchical_ci_low", context)
            standard_high = _as_float(row, "hierarchical_ci_high", context)
            n_datasets = _as_int(row, "n_datasets", context)
            if standard_low > standard_high:
                raise InputValidationError(f"Reversed standard interval in {context}")

            block_values: dict[int, tuple[float, float]] = {}
            for block_length in (5, 10):
                block_key = (task, method_key, str(block_length))
                if block_key not in blocks:
                    raise InputValidationError(
                        f"Missing block-{block_length} timing interval for {key}"
                    )
                block = blocks[block_key]
                block_context = (
                    f"block-{block_length} timing interval {task}/{method_key}"
                )
                _validate_bootstrap_row(block, n_bootstrap, block_context)
                block_confidence = _as_float(block, "confidence", block_context)
                if not math.isclose(
                    block_confidence, confidence, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise InputValidationError(
                        f"Confidence mismatch in {block_context}"
                    )
                block_estimate = _as_float(block, "estimate", block_context)
                if not math.isclose(
                    block_estimate, estimate, rel_tol=0.0, abs_tol=1e-8
                ):
                    raise InputValidationError(
                        f"Point estimate mismatch in {block_context}"
                    )
                if _as_int(block, "n_datasets", block_context) != n_datasets:
                    raise InputValidationError(
                        f"Dataset-count mismatch in {block_context}"
                    )
                low = _as_float(block, "hierarchical_block_ci_low", block_context)
                high = _as_float(block, "hierarchical_block_ci_high", block_context)
                if low > high:
                    raise InputValidationError(f"Reversed interval in {block_context}")
                block_values[block_length] = (low, high)

            output.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "method_key": method_key,
                    "method": METHOD_LABELS_CSV[method_key],
                    "estimate_time_saved_pct": estimate,
                    "standard_hierarchical_ci_low": standard_low,
                    "standard_hierarchical_ci_high": standard_high,
                    "block5_hierarchical_ci_low": block_values[5][0],
                    "block5_hierarchical_ci_high": block_values[5][1],
                    "block10_hierarchical_ci_low": block_values[10][0],
                    "block10_hierarchical_ci_high": block_values[10][1],
                    "confidence": confidence,
                    "n_datasets": n_datasets,
                    "n_bootstrap": n_bootstrap,
                }
            )
    return output


def _tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _tex_pvalue(value: float) -> str:
    if value == 0.0:
        return r"$<10^{-300}$"
    if value < 0.001:
        exponent = int(math.floor(math.log10(value)))
        mantissa = value / (10.0**exponent)
        return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"
    return f"{value:.3f}"


def _tex_interval(low: float, high: float) -> str:
    return rf"[{low:.1f}, {high:.1f}]"


def _table_header(caption: str, label: str, columns: str) -> list[str]:
    return [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
        r"\toprule",
    ]


def _render_omnibus(rows: Sequence[Mapping[str, object]]) -> str:
    lines = _table_header(
        "Friedman tests across the 10 revision-designated representative methods. "
        "$N$ is the number of complete datasets and $W$ is Kendall's coefficient.",
        "tab:inference-friedman",
        "llrrrrr",
    )
    lines.extend(
        [
            r"Task & Metric & $N$ & $\chi^2_F$ & df & $p$ & Kendall $W$ \\",
            r"\midrule",
        ]
    )
    for task_index, task in enumerate(TASK_ORDER):
        task_rows = [row for row in rows if row["task"] == task]
        for metric_index, row in enumerate(task_rows):
            task_cell = _tex_escape(row["task_label"]) if metric_index == 0 else ""
            lines.append(
                " & ".join(
                    (
                        task_cell,
                        _tex_escape(row["metric_label"]),
                        str(row["n_complete_datasets"]),
                        f"{float(row['statistic']):.2f}",
                        str(row["df"]),
                        _tex_pvalue(float(row["pvalue"])),
                        f"{float(row['kendalls_w']):.3f}",
                    )
                )
                + r" \\"
            )
        if task_index < len(TASK_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines)


def _render_joint(rows: Sequence[Mapping[str, object]]) -> str:
    lines = _table_header(
        "Primary simultaneous-success tests at the task-specific 1-unit margin: "
        "1\% bounded-relative RMSE for regression and one weighted-F1 percentage "
        "point for classification. $p_{\mathrm{Holm}}$ tests whether more than "
        "half of dataset entries satisfy both conditions.",
        "tab:inference-joint-primary",
        "llrrrrc",
    )
    lines.extend(
        [
            r"Task & Method & Time $>0$ & Loss $<1$ & Both & $p_{\mathrm{Holm}}$ & Support \\",
            r"\midrule",
        ]
    )
    for task_index, task in enumerate(TASK_ORDER):
        task_rows = [row for row in rows if row["task"] == task]
        for method_index, row in enumerate(task_rows):
            task_cell = _tex_escape(row["task_label"]) if method_index == 0 else ""
            method_key = str(row["method_key"])
            lines.append(
                " & ".join(
                    (
                        task_cell,
                        METHOD_LABELS_TEX[method_key],
                        f"{row['runtime_n_positive']}/{row['runtime_n_non_ties']}",
                        f"{row['loss_n_below_margin']}/{row['loss_n_non_ties']}",
                        f"{row['joint_n_success']}/{row['n_complete_datasets']}",
                        _tex_pvalue(float(row["joint_success_pvalue_holm"])),
                        "Yes"
                        if bool(row["supports_primary_joint_claim_0.05"])
                        else "No",
                    )
                )
                + r" \\"
            )
        if task_index < len(TASK_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines)


def _render_timing(rows: Sequence[Mapping[str, object]]) -> str:
    lines = _table_header(
        "Centroid training-time saving with hierarchical 95\% intervals. "
        "The block columns use circular moving-block lengths 5 and 10.",
        "tab:inference-timing-sensitivity",
        "llrrrr",
    )
    lines.extend(
        [
            r"Task & Method & Estimate & Standard & Block 5 & Block 10 \\",
            r"\midrule",
        ]
    )
    for task_index, task in enumerate(TASK_ORDER):
        task_rows = [row for row in rows if row["task"] == task]
        for method_index, row in enumerate(task_rows):
            task_cell = _tex_escape(row["task_label"]) if method_index == 0 else ""
            method_key = str(row["method_key"])
            lines.append(
                " & ".join(
                    (
                        task_cell,
                        METHOD_LABELS_TEX[method_key],
                        f"{float(row['estimate_time_saved_pct']):.1f}",
                        _tex_interval(
                            float(row["standard_hierarchical_ci_low"]),
                            float(row["standard_hierarchical_ci_high"]),
                        ),
                        _tex_interval(
                            float(row["block5_hierarchical_ci_low"]),
                            float(row["block5_hierarchical_ci_high"]),
                        ),
                        _tex_interval(
                            float(row["block10_hierarchical_ci_low"]),
                            float(row["block10_hierarchical_ci_high"]),
                        ),
                    )
                )
                + r" \\"
            )
        if task_index < len(TASK_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise InputValidationError(f"Refusing to write empty table: {path.name}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    article_dir: Path,
    omnibus: Sequence[Mapping[str, object]],
    joint: Sequence[Mapping[str, object]],
    timing: Sequence[Mapping[str, object]],
) -> list[Path]:
    article_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (
            article_dir / "inference_friedman.csv",
            article_dir / "inference_friedman.tex",
            omnibus,
            _render_omnibus(omnibus),
        ),
        (
            article_dir / "inference_joint_primary_1pct.csv",
            article_dir / "inference_joint_primary_1pct.tex",
            joint,
            _render_joint(joint),
        ),
        (
            article_dir / "inference_timing_centroid_sensitivity.csv",
            article_dir / "inference_timing_centroid_sensitivity.tex",
            timing,
            _render_timing(timing),
        ),
    )
    written: list[Path] = []
    for csv_path, tex_path, rows, tex in outputs:
        _write_csv(csv_path, rows)
        tex_path.write_text(tex, encoding="utf-8")
        written.extend((csv_path, tex_path))

    for path in written:
        content = path.read_text(encoding="utf-8")
        if "secretary_par" in content or "S_par" in content:
            raise InputValidationError(f"S_par leaked into output: {path}")
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export supplementary inferential tables from a validated bundle."
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=DEFAULT_INFERENCE_DIR,
        help=f"Inference bundle directory (default: {DEFAULT_INFERENCE_DIR})",
    )
    parser.add_argument(
        "--article-dir",
        type=Path,
        default=DEFAULT_ARTICLE_DIR,
        help=f"Output directory for TeX and CSV tables (default: {DEFAULT_ARTICLE_DIR})",
    )
    parser.add_argument(
        "--allow-nonproduction",
        action="store_true",
        help="Permit an inference bundle with fewer than 10,000 bootstrap replicates.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        n_bootstrap, confidence = _load_metadata(
            args.inference_dir, args.allow_nonproduction
        )
        omnibus = _build_omnibus(args.inference_dir)
        joint = _build_joint(args.inference_dir)
        timing = _build_timing(args.inference_dir, n_bootstrap, confidence)
        written = _write_outputs(args.article_dir, omnibus, joint, timing)
    except InputValidationError as error:
        raise SystemExit(f"table_inference.py: {error}") from error

    for path in written:
        print(path)
    if n_bootstrap != PRODUCTION_BOOTSTRAPS:
        print(
            f"WARNING: nonproduction output (n_bootstrap={n_bootstrap})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
