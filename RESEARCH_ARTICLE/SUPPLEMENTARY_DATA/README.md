# Raw benchmark and revision data

These archives are versioned with the repository and can also be uploaded as
separate supplementary-data files. The three historical benchmark archives
contain 100 per-seed CSV files, the corresponding aggregate CSV, and benchmark
metadata. The revision archives contain the raw fixed-budget control, focused
bagging validation, operation-counter diagnostic, strict reliability analysis,
corpus audit, and ERT-weighted gain-landscape summaries. AppleDouble metadata
files are excluded.

| Archive | SHA-256 |
|---|---|
| `benchmark_regression_runs.tar.gz` | `24fb675a74b297a4bf1aa7983f1f3834284b5ebb4448021988a8fb73d10b8af4` |
| `benchmark_classification_gini_runs.tar.gz` | `ed4d2d7858acf1969895200dee185c70c40b90e3135faf9bd4f33573115698d3` |
| `benchmark_classification_entropy_runs.tar.gz` | `abbad17eb20820f53fdc283c44aa9d715fabf20629658e942344e14d89692d02` |
| `revision_controls_and_ensemble.tar.gz` | `3c9dc36572049908ed80287450f48a53d3cfced96b7afe3a0723ec42fc77f6b5` |
| `runtime_effort_diagnostic.tar.gz` | `f198901f564ef4ec6e6b8eef3382b75117cf136957f0ad62c6ac5a5a39ca2648` |
| `inference_10pct_and_corpus_audit.tar.gz` | `1aa2ba4f0a869d10e45e503a6a46fe6b0f10abc85eff7ba8e1d8a14ff47263a3` |
| `gain_landscape_ert_weighted.tar.gz` | `15070df21c8e0d56b7ce4a8f23e446bc944ea7b0e399939f55524e27510391b0` |

The fixed-budget archive includes all 20 paired seeds for every retained entry,
the ensemble raw outputs, and the CSV summaries used in the new figures and
tables. The inference archive includes the 10,000-replicate outputs for the
10% minimum-time-saving criterion, the retained-entry manifests, and the
complete corpus inclusion audit. Every new campaign also stores its exact
invocation, software versions, method-order rule, and exclusions in JSON.
