#!/bin/sh

set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
article="$root/RESEARCH_ARTICLE"
out="$article/INFORMATION_SCIENCES_SUBMISSION"
source_dir="$out/SOURCE"

mkdir -p "$out"
rm -rf "$source_dir" "$out/SUPPLEMENTARY_DATA"
mkdir -p "$source_dir/MAIN_FIGURES" "$source_dir/SUPP_FIGURES" "$source_dir/TABLES"
mkdir -p "$out/SUPPLEMENTARY_DATA"

for file in \
    main.tex supp.tex references.bib elsarticle.cls elsarticle-num.bst \
    elsevier-logo.pdf dataset_benchmark_summary.tex main.bbl supp.bbl \
    highlights.tex graphical_abstract.tex
do
    cp "$article/$file" "$source_dir/"
done

for file in \
    figure1_pareto_all.pdf \
    figure2_direct_control.pdf \
    figure3_regime_combined.pdf
do
    cp "$article/MAIN_FIGURES/$file" "$source_dir/MAIN_FIGURES/"
done

for file in \
    supp_figure_01_pareto_large_small.png \
    supp_figure_02_tree_comparison.png \
    supp_figure_03_ridgelines_regression.png \
    supp_figure_04_ridgelines_classification_gini.png \
    supp_figure_05_ridgelines_classification_entropy.png \
    supp_figure_07_success_loss_only_large_small.png \
    supp_figure_08_predicted_regime_maps.png \
    supp_figure_09_within_whiskers_loss_speedup.png \
    supp_figure_10_within_whiskers_loss_speedup.png \
    supp_figure_11_within_whiskers_loss_speedup.png \
    supp_figure_12_success_joint_all.png \
    supp_figure_13_gain_landscape.png \
    supp_figure_14_effort_loss.png \
    supp_figure_15_time_effort_calibration.png \
    supp_figure_16_secretary_par_screening.png \
    revision_control_macro_f1_tradeoff.pdf
do
    cp "$article/SUPP_FIGURES/$file" "$source_dir/SUPP_FIGURES/"
done

for file in \
    table1_main.tex ensemble_validation.tex spar_corrected_summary.tex \
    direct_control_summary.tex direct_control_class_metrics.tex \
    direct_control_tail_losses.tex classification_family_sensitivity.tex \
    inference_friedman.tex inference_joint_primary_1pct.tex \
    inference_timing_centroid_sensitivity.tex runtime_effort_diagnostic.tex
do
    cp "$article/TABLES/$file" "$source_dir/TABLES/"
done

cp "$article/main.pdf" "$article/supp.pdf" "$article/highlights.pdf" \
    "$article/graphical_abstract.pdf" "$out/"
cp "$article/highlights.tex" "$article/graphical_abstract.tex" "$out/"
cp "$article/SUPPLEMENTARY_DATA/"*.tar.gz "$out/SUPPLEMENTARY_DATA/"
cp "$article/SUPPLEMENTARY_DATA/README.md" "$out/SUPPLEMENTARY_DATA/"

rm -f "$out/manuscript_source.zip"
(
    cd "$source_dir"
    zip -qr "$out/manuscript_source.zip" .
)

git -C "$root" rev-parse HEAD > "$out/REPOSITORY_COMMIT.txt"

printf '%s\n' "Submission package refreshed at $out"
