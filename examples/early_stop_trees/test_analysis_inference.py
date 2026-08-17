"""Focused tests for analysis_inference.py."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_inference import (  # noqa: E402
    BASELINE_METHOD,
    CONFIRMATORY_METHODS,
    PRIMARY_LOSS_MARGIN_PCT,
    _identity_paired_resample_indices,
    _one_sided_sign_summary,
    _run_number,
    build_file_manifest,
    dataset_family_id,
    dataset_interval_table,
    family_balanced_global_interval_table,
    family_balanced_joint_test_table,
    global_interval_table,
    holm_adjust,
    inferential_test_tables,
    joint_time_loss_margin_test_table,
    loss_margin_sign_test_table,
    paired_run_bootstrap,
    prepare_paired_run_metrics,
    require_confirmatory_methods,
    runtime_superiority_sign_test_table,
    sha256_file,
    timing_block_bootstrap_tables,
    timing_run_index_diagnostic_table,
)


def _regression_raw() -> pd.DataFrame:
    rows = []
    for run, best_time, method_time in [(1, 10.0, 5.0), (2, 12.0, 6.0)]:
        rows.extend(
            [
                {
                    "run": run,
                    "dataset": "toy",
                    "n_samples": 20,
                    "n_features": 2,
                    "splitter": "best",
                    "variant": "",
                    "method_key": BASELINE_METHOD,
                    "rmse_mean": 2.0,
                    "fit_time_mean": best_time,
                    "gain_evaluations_mean": 100.0,
                },
                {
                    "run": run,
                    "dataset": "toy",
                    "n_samples": 20,
                    "n_features": 2,
                    "splitter": "secretary",
                    "variant": "1overe",
                    "method_key": "secretary|1overe",
                    "rmse_mean": 2.5,
                    "fit_time_mean": method_time,
                    "gain_evaluations_mean": 25.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _full_confirmatory_observed(
    n_datasets: int = 7,
    *,
    extra_methods: tuple[str, ...] = (),
) -> tuple[list[str], np.ndarray]:
    methods = [*CONFIRMATORY_METHODS, *extra_methods]
    observed = np.zeros((n_datasets, len(methods), 3), dtype=float)
    for method_index in range(1, len(CONFIRMATORY_METHODS)):
        observed[:, method_index, 0] = 10.0 + method_index
        observed[:, method_index, 1] = 0.25
        observed[:, method_index, 2] = 20.0 + method_index
    for method_index in range(len(CONFIRMATORY_METHODS), len(methods)):
        observed[:, method_index, 0] = 50.0
        observed[:, method_index, 1] = 0.0
        observed[:, method_index, 2] = 50.0
    return methods, observed


def _timing_run_metrics() -> pd.DataFrame:
    rows = []
    for dataset in ("a", "b"):
        for run in range(1, 13):
            for method_key, speedup in (
                (BASELINE_METHOD, 1.0),
                ("method|", 1.0 / 0.9),
                ("clone|", 1.0 / 0.9),
            ):
                splitter, _, variant = method_key.partition("|")
                rows.append(
                    {
                        "run": run,
                        "dataset": dataset,
                        "splitter": splitter,
                        "variant": variant,
                        "method_key": method_key,
                        "n_samples": 20,
                        "n_features": 2,
                        "speedup": speedup,
                        "predictive_loss": 0.0,
                        "effort_saved": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_prepare_paired_run_metrics_uses_same_run_baseline():
    paired = prepare_paired_run_metrics(_regression_raw(), "regression")
    method = paired[paired["method_key"] == "secretary|1overe"]
    np.testing.assert_allclose(method["speedup"], 2.0)
    np.testing.assert_allclose(method["predictive_loss"], 0.2)
    np.testing.assert_allclose(method["effort_saved"], 0.75)


def test_classification_f1_loss_is_percentage_point_difference():
    raw = _regression_raw().rename(columns={"rmse_mean": "f1_weighted_mean"})
    raw.loc[raw["method_key"] == BASELINE_METHOD, "f1_weighted_mean"] = 0.80
    raw.loc[
        raw["method_key"] == "secretary|1overe", "f1_weighted_mean"
    ] = 0.79
    paired = prepare_paired_run_metrics(raw, "classification_gini")
    method = paired[paired["method_key"] == "secretary|1overe"]
    np.testing.assert_allclose(method["predictive_loss"], 0.01)


def test_paired_bootstrap_uses_identical_run_draws_for_methods():
    paired = prepare_paired_run_metrics(_regression_raw(), "regression")
    duplicate = paired[paired["method_key"] == "secretary|1overe"].copy()
    duplicate["splitter"] = "clone"
    duplicate["variant"] = ""
    duplicate["method_key"] = "clone|"
    paired = pd.concat([paired, duplicate], ignore_index=True)

    datasets, methods, observed, draws, counts, _ = paired_run_bootstrap(
        paired,
        n_bootstrap=50,
        confidence=0.95,
        rng=np.random.default_rng(123),
        chunk_size=13,
    )
    assert datasets == ["toy"]
    secretary = methods.index("secretary|1overe")
    clone = methods.index("clone|")
    np.testing.assert_allclose(observed[0, secretary], observed[0, clone])
    np.testing.assert_allclose(draws[0, secretary], draws[0, clone])
    assert counts[0, secretary] == 2


def test_paired_bootstrap_is_invariant_to_input_row_order():
    paired = prepare_paired_run_metrics(_regression_raw(), "regression")
    original = paired_run_bootstrap(
        paired,
        n_bootstrap=40,
        confidence=0.95,
        rng=np.random.default_rng(99),
        chunk_size=11,
    )
    shuffled = paired_run_bootstrap(
        paired.sample(frac=1.0, random_state=7),
        n_bootstrap=40,
        confidence=0.95,
        rng=np.random.default_rng(99),
        chunk_size=11,
    )
    assert original[0] == shuffled[0]
    assert original[1] == shuffled[1]
    np.testing.assert_allclose(original[2], shuffled[2])
    np.testing.assert_allclose(original[3], shuffled[3])


def test_constant_ten_percent_saving_has_degenerate_interval():
    raw = _regression_raw()
    method = raw["method_key"] == "secretary|1overe"
    best_time = raw.loc[~method, ["run", "fit_time_mean"]].set_index("run")[
        "fit_time_mean"
    ]
    raw.loc[method, "fit_time_mean"] = (
        raw.loc[method, "run"].map(best_time) * 0.9
    )
    paired = prepare_paired_run_metrics(raw, "regression")
    _, methods, observed, draws, _, _ = paired_run_bootstrap(
        paired,
        n_bootstrap=50,
        confidence=0.95,
        rng=np.random.default_rng(10),
    )
    method_index = methods.index("secretary|1overe")
    np.testing.assert_allclose(observed[0, method_index, 0], 10.0)
    np.testing.assert_allclose(draws[0, method_index, 0], 10.0)


def test_duplicate_seed_rows_are_rejected_instead_of_narrowing_ci():
    paired = prepare_paired_run_metrics(_regression_raw(), "regression")
    duplicated = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)
    try:
        paired_run_bootstrap(
            duplicated,
            n_bootstrap=20,
            confidence=0.95,
            rng=np.random.default_rng(5),
        )
    except ValueError as error:
        assert "artificially narrow" in str(error)
    else:
        raise AssertionError("Duplicated seed rows must be rejected")


def test_incomplete_dataset_uses_raw_run_intersection_without_imputation():
    paired = prepare_paired_run_metrics(_regression_raw(), "regression")
    paired = paired[
        ~(
            (paired["method_key"] == "secretary|1overe")
            & (paired["run"] == 2)
        )
    ]
    _, _, _, _, counts, diagnostics = paired_run_bootstrap(
        paired,
        n_bootstrap=20,
        confidence=0.95,
        rng=np.random.default_rng(8),
    )
    assert np.all(counts == 1)
    assert diagnostics.loc[0, "n_complete_paired_runs"] == 1
    assert diagnostics.loc[0, "excluded_run_ids"] == "2"
    assert diagnostics.loc[0, "complete_block_status"] == "incomplete_flagged"


def test_dataset_and_global_intervals_have_expected_sources():
    datasets = ["a", "b"]
    methods = [BASELINE_METHOD, "method|"]
    observed = np.zeros((2, 2, 3), dtype=float)
    observed[:, 1, 0] = [10.0, 30.0]
    observed[:, 1, 1] = [1.0, 3.0]
    observed[:, 1, 2] = [40.0, 60.0]
    draws = np.repeat(observed[..., None], 200, axis=-1)
    counts = np.full((2, 2), 100, dtype=int)

    dataset_table = dataset_interval_table(
        task="regression",
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=draws,
        run_counts=counts,
        confidence=0.95,
    )
    row = dataset_table[
        (dataset_table["dataset"] == "a")
        & (dataset_table["method_key"] == "method|")
        & (dataset_table["metric"] == "time_saved_pct")
    ].iloc[0]
    assert row["estimate"] == row["ci_low"] == row["ci_high"] == 10.0

    global_table = global_interval_table(
        task="regression",
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=draws,
        confidence=0.95,
        rng=np.random.default_rng(321),
    )
    row = global_table[
        (global_table["method_key"] == "method|")
        & (global_table["metric"] == "time_saved_pct")
        & (global_table["estimand"] == "centroid_mean")
    ].iloc[0]
    assert row["estimate"] == 20.0
    assert row["within_ci_low"] == row["within_ci_high"] == 20.0
    assert row["between_ci_low"] <= 10.0
    assert row["between_ci_high"] >= 30.0
    assert row["hierarchical_ci_low"] <= 10.0
    assert row["hierarchical_ci_high"] >= 30.0


def test_holm_adjustment_is_step_down_and_order_preserving():
    raw = np.array([0.04, 0.001, 0.02, np.nan])
    adjusted = holm_adjust(raw)
    np.testing.assert_allclose(adjusted[:3], [0.04, 0.003, 0.04])
    finite_order = np.argsort(raw[:3])
    assert np.all(np.diff(adjusted[:3][finite_order]) >= 0.0)
    assert np.isnan(adjusted[3])


def test_runtime_superiority_sign_test_uses_representatives_only():
    exploratory = "secretary_par|samples=sqrt_n,q=0.75"
    methods, observed = _full_confirmatory_observed(
        5, extra_methods=(exploratory,)
    )
    table = runtime_superiority_sign_test_table(
        task="regression",
        methods=methods,
        observed=observed,
        confidence=0.95,
    )
    assert table["method_key"].tolist() == list(CONFIRMATORY_METHODS[1:])
    assert exploratory not in set(table["method_key"])
    assert np.all(table["n_positive_time_saving"] == 5)
    assert np.all(table["analysis_tier"] == "confirmatory")


def test_confirmatory_set_is_strict_and_excludes_screened_variants():
    missing = list(CONFIRMATORY_METHODS[:-1])
    try:
        require_confirmatory_methods(missing)
    except ValueError as error:
        assert CONFIRMATORY_METHODS[-1] in str(error)
    else:
        raise AssertionError("A missing representative method must fail")

    exploratory = "secretary_par|samples=sqrt_n,q=0.75"
    selected = require_confirmatory_methods([*CONFIRMATORY_METHODS, exploratory])
    assert selected == list(CONFIRMATORY_METHODS)
    assert exploratory not in selected


def test_sign_tests_use_correct_directions_and_omit_ties():
    summary = _one_sided_sign_summary(
        np.asarray([2.0, 1.0, 0.0, -1.0]), confidence=0.95
    )
    assert summary["n_favorable"] == 2
    assert summary["n_unfavorable"] == 1
    assert summary["n_ties"] == 1
    assert summary["n_non_ties"] == 3
    assert summary["pvalue_raw"] == 0.5

    methods, observed = _full_confirmatory_observed(4)
    time_and_loss = np.asarray([0.5, 0.8, 1.0, 1.2])
    observed[:, :, 0] = time_and_loss[:, None] - 1.0
    observed[:, :, 1] = time_and_loss[:, None]
    runtime = runtime_superiority_sign_test_table(
        task="classification_gini",
        methods=methods,
        observed=observed,
        confidence=0.95,
    )
    loss = loss_margin_sign_test_table(
        task="classification_gini",
        methods=methods,
        observed=observed,
        confidence=0.95,
    )
    runtime_row = runtime.iloc[0]
    primary_loss_row = loss[
        np.isclose(loss["loss_margin_pct"], PRIMARY_LOSS_MARGIN_PCT)
    ].iloc[0]
    assert runtime_row["n_positive_time_saving"] == 1
    assert runtime_row["n_negative_time_saving"] == 2
    assert runtime_row["n_zero_ties"] == 1
    assert primary_loss_row["n_below_margin"] == 2
    assert primary_loss_row["n_above_margin"] == 1
    assert primary_loss_row["n_equal_margin"] == 1


def test_joint_test_uses_same_dataset_for_time_and_loss_success():
    methods, observed = _full_confirmatory_observed(7)
    observed[:, 1:, 0] = np.asarray([1, 1, 1, 1, 1, -1, -1])[:, None]
    observed[:, 1:, 1] = np.asarray([2, 2, 0, 0, 0, 0, 0])[:, None]
    table = joint_time_loss_margin_test_table(
        task="classification_entropy",
        methods=methods,
        observed=observed,
        confidence=0.95,
    )
    row = table[
        (table["method_key"] == CONFIRMATORY_METHODS[1])
        & np.isclose(table["loss_margin_pct"], PRIMARY_LOSS_MARGIN_PCT)
    ].iloc[0]
    assert row["runtime_n_positive"] == 5
    assert row["loss_n_below_margin"] == 5
    assert row["joint_n_success"] == 3
    assert row["joint_n_failure"] == 4
    assert np.isclose(row["joint_success_pvalue_raw"], 0.7734375)


def test_joint_primary_and_sensitivity_multiplicity_are_separate():
    methods, observed = _full_confirmatory_observed(7)
    table = joint_time_loss_margin_test_table(
        task="regression",
        methods=methods,
        observed=observed,
        confidence=0.95,
    )
    primary = table[table["is_primary_margin"]]
    sensitivity = table[~table["is_primary_margin"]]
    assert PRIMARY_LOSS_MARGIN_PCT == 1.0
    assert len(primary) == 9
    assert len(sensitivity) == 18
    assert set(primary["multiplicity_family_size"]) == {9}
    assert set(sensitivity["multiplicity_family_size"]) == {18}
    assert set(primary["evidence_role"]) == {"primary_confirmatory"}
    assert set(sensitivity["evidence_role"]) == {"sensitivity_analysis"}
    np.testing.assert_allclose(primary["joint_success_pvalue_raw"], 1.0 / 128.0)
    np.testing.assert_allclose(primary["joint_success_pvalue_holm"], 9.0 / 128.0)
    np.testing.assert_allclose(
        sensitivity["joint_success_pvalue_holm"], 18.0 / 128.0
    )
    assert not sensitivity["supports_primary_joint_claim_0.05"].any()


def test_dataset_family_mapping_groups_related_regression_entries_only():
    assert dataset_family_id("regression", "579_fri_c0_250_5") == (
        "friedman_generator_c0"
    )
    assert dataset_family_id("regression", "650_fri_c0_500_50") == (
        "friedman_generator_c0"
    )
    assert dataset_family_id("regression", "197_cpu_act") == "cpu_act_alias"
    assert dataset_family_id("regression", "573_cpu_act") == "cpu_act_alias"
    assert dataset_family_id("classification_gini", "197_cpu_act") == "197_cpu_act"


def test_family_balanced_exports_use_family_level_labels():
    methods = list(CONFIRMATORY_METHODS)
    datasets = ["579_fri_c0_250_5", "650_fri_c0_500_50", "197_cpu_act"]
    observed = np.zeros((len(datasets), len(methods), 3), dtype=float)
    bootstrap = np.zeros((len(datasets), len(methods), 3, 2), dtype=float)
    global_table = family_balanced_global_interval_table(
        task="regression",
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=bootstrap,
        confidence=0.95,
        rng=np.random.default_rng(0),
    )
    assert set(global_table["experimental_unit"]) == {"dataset_family"}
    assert set(global_table["estimand"]) == {
        "family_centroid_mean",
        "cross_family_median",
    }

    joint_table = family_balanced_joint_test_table(
        task="regression",
        datasets=datasets,
        methods=methods,
        observed=observed,
        confidence=0.95,
    )
    assert set(joint_table["experimental_unit"]) == {"dataset_family"}
    assert joint_table["joint_test_rule"].str.contains("family-level").all()
    assert joint_table["joint_null"].str.startswith("P_family").all()
    assert joint_table["joint_alternative"].str.startswith("P_family").all()


def test_identity_keyed_resampling_does_not_reuse_equal_sized_other_sets():
    cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(123)
    first = _identity_paired_resample_indices(
        ("a", "b"), n_bootstrap=30, rng=rng, cache=cache
    )
    repeated = _identity_paired_resample_indices(
        ("a", "b"), n_bootstrap=30, rng=rng, cache=cache
    )
    _identity_paired_resample_indices(
        ("c", "d"), n_bootstrap=30, rng=rng, cache=cache
    )
    assert set(cache) == {("a", "b"), ("c", "d")}
    assert first[0] is repeated[0]
    assert first[1] is repeated[1]


def test_hierarchical_global_resampling_preserves_identical_method_pairing():
    datasets = ["a", "b", "c"]
    methods = ["method_a|", "method_b|"]
    observed = np.arange(18, dtype=float).reshape(3, 2, 3)
    observed[:, 1, :] = observed[:, 0, :]
    rng = np.random.default_rng(44)
    base_pool = observed[:, 0, :, None] + rng.normal(
        scale=0.1, size=(3, 3, 80)
    )
    bootstrap = np.stack([base_pool, base_pool], axis=1)
    table = global_interval_table(
        task="regression",
        datasets=datasets,
        methods=methods,
        observed=observed,
        bootstrap=bootstrap,
        confidence=0.95,
        rng=np.random.default_rng(55),
    )
    numeric = [
        "estimate",
        "within_ci_low",
        "within_ci_high",
        "between_ci_low",
        "between_ci_high",
        "hierarchical_ci_low",
        "hierarchical_ci_high",
    ]
    first = table[table["method_key"] == methods[0]].reset_index(drop=True)
    second = table[table["method_key"] == methods[1]].reset_index(drop=True)
    np.testing.assert_allclose(first[numeric], second[numeric])


def test_friedman_and_wilcoxon_outputs_have_confirmatory_schema():
    methods, observed = _full_confirmatory_observed(12)
    dataset_shift = np.linspace(-0.2, 0.2, observed.shape[0])
    for method_index in range(len(methods)):
        observed[:, method_index, :] += dataset_shift[:, None]
    bootstrap = np.repeat(observed[..., None], 40, axis=-1)
    global_table = global_interval_table(
        task="regression",
        datasets=[f"d{i}" for i in range(observed.shape[0])],
        methods=methods,
        observed=observed,
        bootstrap=bootstrap,
        confidence=0.95,
        rng=np.random.default_rng(66),
    )
    pairwise, omnibus = inferential_test_tables(
        task="regression",
        datasets=[f"d{i}" for i in range(observed.shape[0])],
        methods=methods,
        observed=observed,
        global_intervals=global_table,
    )
    assert len(pairwise) == 27
    assert len(omnibus) == 3
    assert set(omnibus["n_methods"]) == {10}
    assert {
        "wilcoxon_pvalue_raw",
        "wilcoxon_pvalue_holm",
        "signed_rank_biserial",
        "unit",
    }.issubset(pairwise.columns)
    assert {"statistic", "pvalue", "kendalls_w", "unit"}.issubset(
        omnibus.columns
    )


def test_hash_manifest_is_complete_and_reproducible():
    with tempfile.TemporaryDirectory() as directory:
        indir = Path(directory)
        metadata = indir / "benchmark_metadata.json"
        raw = indir / "regression_run001.csv"
        metadata.write_text('{"benchmark": "toy"}\n')
        raw.write_text("dataset,value\ntoy,1\n")
        first = build_file_manifest(indir, ["regression"])
        second = build_file_manifest(indir, ["regression"])
        pd.testing.assert_frame_equal(first, second)
        assert set(first["category"]) == {
            "analysis_source",
            "test_source",
            "benchmark_metadata",
            "benchmark_source",
            "splitter_source",
            "reference_splitter_source",
            "build_source",
            "raw_input_csv",
        }
        assert {
            "examples/early_stop_trees/benchmark_secretary_pmlb.py",
            "treeple/tree/_early_stop_splitter.pyx",
            "treeple/_lib/sklearn_fork/sklearn/tree/_splitter.pyx",
            "pyproject.toml",
        }.issubset(set(first["logical_path"]))
        expected = hashlib.sha256(raw.read_bytes()).hexdigest()
        assert sha256_file(raw) == expected
        raw.write_text("dataset,value\ntoy,2\n")
        changed = build_file_manifest(indir, ["regression"])
        first_raw = first[first["category"] == "raw_input_csv"]["sha256"].iloc[0]
        changed_raw = changed[changed["category"] == "raw_input_csv"]["sha256"].iloc[0]
        assert first_raw != changed_raw


def test_timing_block_outputs_are_paired_reproducible_and_correct():
    run_metrics = _timing_run_metrics()
    diagnostics = timing_run_index_diagnostic_table(
        task="regression", run_metrics=run_metrics
    )
    assert {
        "spearman_rho_run_index_vs_time_saved",
        "linear_slope_pct_points_per_10_runs",
        "second_half_minus_first_half_median_pct_points",
    }.issubset(diagnostics.columns)
    first = timing_block_bootstrap_tables(
        task="regression",
        run_metrics=run_metrics,
        n_bootstrap=30,
        confidence=0.95,
        rng=np.random.default_rng(77),
    )
    second = timing_block_bootstrap_tables(
        task="regression",
        run_metrics=run_metrics.sample(frac=1.0, random_state=2),
        n_bootstrap=30,
        confidence=0.95,
        rng=np.random.default_rng(77),
    )
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    dataset_rows, centroid_rows = first
    assert set(dataset_rows["block_length_runs"]) == {5, 10}
    assert set(centroid_rows["block_length_runs"]) == {5, 10}
    method_rows = dataset_rows[dataset_rows["method_key"] == "method|"]
    np.testing.assert_allclose(method_rows["estimate_time_saved_pct"], 10.0)
    np.testing.assert_allclose(method_rows["ci_low"], 10.0)
    np.testing.assert_allclose(method_rows["ci_high"], 10.0)
    clone = centroid_rows[centroid_rows["method_key"] == "clone|"].reset_index(
        drop=True
    )
    method = centroid_rows[centroid_rows["method_key"] == "method|"].reset_index(
        drop=True
    )
    numeric = [column for column in centroid_rows if column.endswith(("_low", "_high", "_se"))]
    np.testing.assert_allclose(clone[numeric], method[numeric])


def test_actual_564_fried_archive_is_missing_only_run_4():
    archive = Path(__file__).resolve().parent / "benchmark_results"
    paths = sorted(archive.glob("regression_run*.csv"))
    assert len(paths) == 100
    missing = []
    for path in paths:
        datasets = pd.read_csv(path, usecols=["dataset"])["dataset"].astype(str)
        if not np.any(datasets == "564_fried"):
            missing.append(_run_number(path))
    assert missing == [4]
