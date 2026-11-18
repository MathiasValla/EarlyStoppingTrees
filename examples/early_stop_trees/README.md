# Early-stop tree experiments

The authoritative clean-clone instructions are in
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md). They cover dependency setup,
archive verification, statistical regeneration, tests, benchmark commands, and
manuscript compilation.

## Main scripts

- `benchmark_secretary_pmlb.py`: broad 100-seed PMLB benchmark.
- `benchmark_revision_controls.py`: counterbalanced exact-subset control.
- `benchmark_ensemble_validation.py`: focused 25-tree bagging validation.
- `analysis_inference.py`: paired hierarchical inference for the broad benchmark.
- `analyze_revision_controls.py`: direct-control and ensemble summaries.
- `analysis_spar_corrected.py`: effort-loss analysis for the S_par screen.
- `export_paper_figures.py`: copies generated figures into the article directories.

The benchmark command accepts `--max-datasets` for smoke tests and
`--max-product` for the dense-data size rule. The reported corpus uses
`--max-product 1000000`. Each benchmark output directory contains metadata with
the seed range, fold rule, software environment, timing scope, and invocation.

Do not use an untracked demonstration script as an installation check. Run the
focused analysis test instead:

```bash
pytest -q examples/early_stop_trees/test_analysis_inference.py
```
