# Raw benchmark supplementary data

Upload the three archives in this directory as separate supplementary-data
files. Each archive contains 100 per-seed CSV files, the corresponding aggregate
CSV, and the benchmark metadata JSON. AppleDouble metadata files are excluded.

| Archive | SHA-256 |
|---|---|
| `benchmark_regression_runs.tar.gz` | `24fb675a74b297a4bf1aa7983f1f3834284b5ebb4448021988a8fb73d10b8af4` |
| `benchmark_classification_gini_runs.tar.gz` | `ed4d2d7858acf1969895200dee185c70c40b90e3135faf9bd4f33573115698d3` |
| `benchmark_classification_entropy_runs.tar.gz` | `abbad17eb20820f53fdc283c44aa9d715fabf20629658e942344e14d89692d02` |

The archives are intentionally ignored by Git because they are submission data,
not source files. The checksummed input manifest and inferential exports remain
under `examples/early_stop_trees/inference_results/`.
