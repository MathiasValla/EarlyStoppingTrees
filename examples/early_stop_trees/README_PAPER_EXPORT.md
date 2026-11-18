# Paper figure and table export

Run the analysis steps in [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md)
before regenerating visual outputs.

## Main figures

- `figure1.py`: broad-benchmark time-loss Pareto summaries.
- `analyze_revision_controls.py`: fixed-budget control figure.
- `figure3.py`: regime maps.

## Supplementary figures

- `figure1.py`: broad Pareto panels and complete-tree comparison.
- `figure2.py`: task-specific ridgelines.
- `figure4.py`: size-stratified and all-entry reliability panels.
- `supp_plot_predicted_regime_maps.py`: regime maps.
- `supp_plot_within_between_variability.py`: within-dataset whiskers.
- `supp_plot_gain_landscape.py`: gain-landscape diagnostic.
- `supp_plot_effort_metrics.py`: effort-loss and time-effort panels.
- `analysis_spar_corrected.py`: S_par effort-loss screen.
- `analyze_revision_controls.py`: macro-F1 fixed-budget control.

After generating the source figures, run:

```bash
python examples/early_stop_trees/export_paper_figures.py
```

The final manuscript uses `MAIN_FIGURES/figure1_pareto_all`,
`MAIN_FIGURES/figure2_direct_control`, and
`MAIN_FIGURES/figure3_regime_combined`. Supplementary figure filenames are
listed directly in `RESEARCH_ARTICLE/supp.tex`; that source is authoritative
for numbering and inclusion.
