# Early-stop splitters: build and run

## 1. Submodule and symlink

Ensure the sklearn tree submodule is present and visible to the build:

```bash
# From repo root
git submodule update --init

# Meson expects treeple/_lib/sklearn/ (sources under sklearn/tree/...).
# The submodule is at treeple/_lib/sklearn_fork/ with content in sklearn_fork/sklearn/.
cd treeple/_lib
rm -f sklearn
ln -s sklearn_fork/sklearn sklearn
cd ../..
```

## 2. Build and install

```bash
pip install .
```

For a clean test environment (recommended if you have other editable packages that hook into sklearn):

```bash
python -m venv .venv_early_stop
.venv_early_stop/bin/pip install numpy scipy scikit-learn
.venv_early_stop/bin/pip install .
```

## 3. Run the Iris test script

```bash
python examples/early_stop_trees/run_iris_early_stop.py
```

Or with the venv:

```bash
.venv_early_stop/bin/python examples/early_stop_trees/run_iris_early_stop.py
```

The script runs **EarlyStopDecisionTreeRegressor** and **EarlyStopDecisionTreeClassifier** with `splitter="best"` and `splitter="secretary"` on Iris (or synthetic data if sklearn.datasets is unavailable) and prints MSE/R² and accuracy.

## 4. Benchmark secretary methods on PMLB

Requires [pmlb](https://github.com/EpistasisLab/pmlb) (Penn Machine Learning Benchmark):

```bash
pip install pmlb
python examples/early_stop_trees/benchmark_secretary_pmlb.py
```

Optional arguments:

- `--max-datasets N` — limit to the first N regression and N classification datasets (for quick runs).
- `--max-samples N` — subsample each dataset to at most N rows.
- `--outdir DIR` — directory for CSV results (default: `examples/early_stop_trees/benchmark_results`).
- `--regression-only` / `--classification-only` — run only one task.
- `--dataset NAME` — run only this PMLB dataset (subsamples to `--max-rows` if given).
- `--max-rows N` / `--max-features N` — skip datasets exceeding these (or subsample to N rows when using `--dataset`).
- `--max-product N` — skip datasets with n_samples×n_features > N (default: 1000000, to avoid SIGSEGV on large data). Use `--max-product 0` for no limit.
- `--exclude-datasets "name1,name2"` — skip these datasets by name (in addition to built-in skip list). If any splitter raises during a dataset, that dataset is skipped entirely and not written to the CSV.

**High-feature datasets** (where secretary is most likely faster than `best`): e.g. `Hill_Valley_with_noise` (n=1212, p=100), `588_fri_c4_1000_100` (n=1000, p=100). Use `python examples/early_stop_trees/find_biggest_pmlb.py` to list sizes.

Outputs (in `benchmark_results/`):

- **Regression**: `regression_results.csv` — columns `dataset`, `n_samples`, `n_features`, `splitter`, `rmse_mean`, `rmse_std`, `fit_time_mean` for splitters `best`, `secretary`, `secretary_par`, `secretary_all`, `double_secretary`.
- **Classification**: `classification_gini_results.csv` and `classification_entropy_results.csv` — columns `dataset`, `n_samples`, `n_features`, `criterion`, `splitter`, `accuracy_mean`, `accuracy_std`, `f1_weighted_mean`, `f1_weighted_std`, `fit_time_mean`.

All metrics are from 5-fold cross-validation.

### Aggregate results per splitter

After the benchmark, aggregate mean RMSE / accuracy / F1 and mean time per splitter across datasets:

```bash
python examples/early_stop_trees/aggregate_benchmark_results.py
```

Optional: `--indir DIR` to point to a directory other than `examples/early_stop_trees/benchmark_results`.

This writes in the same directory:

- **regression_aggregated.csv** — columns `splitter`, `rmse_mean`, `rmse_std`, `fit_time_mean`, `fit_time_std`, `n_datasets`.
- **classification_aggregated.csv** — columns `criterion`, `splitter`, `accuracy_mean`, `accuracy_std`, `f1_weighted_mean`, `f1_weighted_std`, `fit_time_mean`, `fit_time_std`, `n_datasets`.
