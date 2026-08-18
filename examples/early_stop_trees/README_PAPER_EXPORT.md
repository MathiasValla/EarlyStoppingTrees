# Paper figure export (`MAIN_FIGURES` / `SUPP_FIGURES`)

## Generate sources, then export

```bash
cd examples/early_stop_trees

# Main text + supplementary (merged PNGs written to SUPP_FIGURES/)
python figure1.py                    # supp 01–02 (add --no-supp to skip)
python figure4.py                    # supp 06–07
python figure2.py                    # supp 03–05
python figure3.py

python supp_plot_predicted_regime_maps.py   # supp 08

python supp_plot_within_between_variability.py --task regression
python supp_plot_within_between_variability.py --task classification_gini
python supp_plot_within_between_variability.py --task classification_entropy
# (add --no-supp-png to skip supp 09–11 merged PNGs)

python analysis_gain_landscape.py
python supp_plot_gain_landscape.py          # supp 13
python analysis_spar_corrected.py            # corrected S_par screening

python export_paper_figures.py
```

**Supplementary figures are PNG-only** (no PDF merge). Merged layouts and single legends are produced by the scripts above.

The corrected `S_par` campaign is analyzed separately because its wall-clock
measurements were collected later than the exhaustive reference timings.

## `MAIN_FIGURES/`

- `figure1_pareto_all` (PDF + PNG)
- `figure4_success_combined` — **loss CDF row only** (same titles as `figure4_success_loss_only`) + grouped **marker** legend (joint bar grid is supp 12 only)
- `figure3_regime_combined`

## `SUPP_FIGURES/` (PNG)

| File | Contents |
|------|----------|
| `supp_figure_01_*.png` | Pareto: large + small rows, one legend |
| `supp_figure_02_*.png` | Complete-tree comparison |
| `supp_figure_03` … `05` | Figure 2 ridgelines (regression, gini, entropy) |
| `supp_figure_06_*.png` | Figure 4 joint grid: large + small, one legend |
| `supp_figure_07_*.png` | Figure 4 loss-only CDFs: large + small, one legend |
| `supp_figure_08_*.png` | Predicted regime maps: 3 tasks × 2 metrics, one legend |
| `supp_figure_09` … `11` | Within whiskers: speedup + loss stacked per task |
| `supp_figure_12_*.png` | Joint success bar grid (all datasets), standalone |
| `supp_figure_13_*.png` | Exhaustive gain-landscape diagnostics for regression vs classification |
| `supp_figure_16_*.png` | Corrected `S_par` effort-loss screening |

**MAIN** `figure4_success_combined` matches the loss-CDF row of `figure4_success_loss_only` (same panel titles) with a **single bottom legend** using **marker swatches** (not line handles). The joint 4×3 bar grid is exported separately as `supp_figure_12_success_joint_all.png`.

Legend styles: **points** (circles) for main figures 1, 3, and 4 (CDF); **patches** for bar-based supplementary figure 4 panels where appropriate.
