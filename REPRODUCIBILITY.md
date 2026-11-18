# Reproducing Secretary-style split search for decision trees

This guide starts from a clean clone and reproduces the reported statistical
summaries from the archived per-seed outputs. It also documents how to rerun
the benchmarks. Wall-clock values depend on hardware and system load, so exact
timings are not expected to transfer across machines.

## 1. Clone and install

The recorded revision experiments used Python 3.10.18, scikit-learn 1.7.2,
treeple 0.10.3, and macOS 15.2 on an Apple M3 computer. The broad benchmark
used NumPy 1.26.4. The later counterbalanced control and ensemble experiments
used NumPy 2.2.6. The setup below installs the latter analysis environment.

```bash
git clone https://github.com/MathiasValla/EarlyStoppingTrees.git
cd EarlyStoppingTrees
python3.10 -m venv .venv_reproduction
. .venv_reproduction/bin/activate
PYTHON="$PWD/.venv_reproduction/bin/python" \
  ./examples/early_stop_trees/setup_reproduction_env.sh
```

The setup script initializes the scikit-learn source dependency at the pinned
public commit, applies the versioned compatibility patch, creates the source
symlink expected by Meson, installs the pinned analysis dependencies, and
builds this treeple fork.

## 2. Verify the archives

From the repository root:

```bash
cd RESEARCH_ARTICLE/SUPPLEMENTARY_DATA
openssl dgst -sha256 *.tar.gz
cd ../..
```

Compare the values with `RESEARCH_ARTICLE/SUPPLEMENTARY_DATA/README.md`.

## 3. Extract the reported data

```bash
export REPRO_OUT=/tmp/early_stop_reproduction
rm -rf "$REPRO_OUT"
mkdir -p "$REPRO_OUT"/benchmark "$REPRO_OUT"/revision "$REPRO_OUT"/spar

for archive in \
  benchmark_regression_runs.tar.gz \
  benchmark_classification_gini_runs.tar.gz \
  benchmark_classification_entropy_runs.tar.gz
do
  tar -xzf "RESEARCH_ARTICLE/SUPPLEMENTARY_DATA/$archive" \
    -C "$REPRO_OUT/benchmark"
done

tar -xzf RESEARCH_ARTICLE/SUPPLEMENTARY_DATA/revision_controls_and_ensemble.tar.gz \
  -C "$REPRO_OUT/revision"
tar -xzf RESEARCH_ARTICLE/SUPPLEMENTARY_DATA/spar_screening_runs.tar.gz \
  -C "$REPRO_OUT/spar"
```

## 4. Regenerate the statistical results

The commands below use the seeds and 10,000 bootstrap replicates reported in
the manuscript.

```bash
python examples/early_stop_trees/analysis_inference.py \
  --indir "$REPRO_OUT/benchmark" \
  --outdir "$REPRO_OUT/inference" \
  --n-bootstrap 10000 \
  --seed 20260810

python examples/early_stop_trees/analyze_revision_controls.py \
  --control-root "$REPRO_OUT/revision/examples/early_stop_trees/benchmark_revision_controls_20" \
  --ensemble-root "$REPRO_OUT/revision/examples/early_stop_trees/ensemble_validation" \
  --outdir "$REPRO_OUT/revision_analysis" \
  --n-bootstrap 10000 \
  --seed 20261112

mkdir -p "$REPRO_OUT/spar_article/TABLES" "$REPRO_OUT/spar_article/SUPP_FIGURES"
python examples/early_stop_trees/analysis_spar_corrected.py \
  --baseline-dir "$REPRO_OUT/benchmark" \
  --corrected-dir "$REPRO_OUT/spar/benchmark_spar_corrected_100" \
  --output-dir "$REPRO_OUT/spar_analysis" \
  --article-dir "$REPRO_OUT/spar_article"
```

The generated CSVs can be compared with the corresponding archived analysis
outputs and the tables in `RESEARCH_ARTICLE/TABLES`. Absolute source paths in
metadata manifests are machine-specific; hashes, task labels, sample counts,
and numerical outputs are the reproducibility checks.

## 5. Run the focused analysis tests

```bash
pytest -q examples/early_stop_trees/test_analysis_inference.py
```

## 6. Rerun benchmarks

The complete historical benchmark is computationally expensive. Its canonical
single-process command is:

```bash
python examples/early_stop_trees/benchmark_secretary_pmlb.py \
  --outdir /tmp/early_stop_full \
  --max-product 1000000
```

Use `--max-datasets 5` for a smoke test. The counterbalanced control and focused
ensemble scripts expose their complete command-line interfaces through
`--help`. Dataset lists used by the revision are versioned as
`revision_regression_datasets.txt` and `revision_classification_datasets.txt`.
Every campaign writes software, hardware, seed, fold, timing, and invocation
metadata beside its CSV files.

## 7. Rebuild the manuscripts

```bash
cd RESEARCH_ARTICLE
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
pdflatex supp.tex
bibtex supp
pdflatex supp.tex
pdflatex supp.tex
```

The expected outputs are a nine-page `main.pdf` and a 24-page `supp.pdf`.
