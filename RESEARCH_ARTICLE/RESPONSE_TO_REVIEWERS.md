# Response to the Editor and Reviewers

**Manuscript:** PRLETTERS-D-26-00639, "Secretary-style Split Search for Decision Trees"

All numerical results below come from the 10,000-replicate analysis export
finalized on August 17, 2026. Page references correspond to the final compiled
revision.

## Cover Response

Dear Editor and Reviewers,

Thank you for the detailed comments. We revised the presentation, rechecked the
implementation and numerical results, expanded the statistical analysis, and
narrowed the claims where the evidence required it.

The central message is now direct. Secretary-style split search can provide a
legitimate middle ground between exhaustive CART and more aggressive
randomization when interpretation is restricted to methods that are empirically
supported by the simultaneous-success analysis or Pareto-efficient in the
observed comparisons. In the two classification tasks, `S_all(f=1/e)` provides
the clearest conservative example: it reduces measured fit time while keeping
weighted-F1 loss small. `S^2` also passes the primary simultaneous-success
criterion. ERT often saves more time but usually incurs greater predictive loss.
No secretary-style method passes the primary regression criterion, so the paper
does not claim a universal speedup or a uniformly best splitter.

The main revisions are as follows.

1. We rewrote the abstract, Introduction, Results, and Conclusion around the
   middle-ground result rather than one favorable method.
2. We added entry-level hierarchical confidence intervals, paired comparisons,
   simultaneous time-and-loss tests, and a regression family-balance
   sensitivity analysis.
3. We clarified that every estimator is a complete recursively grown tree. A
   new reproducible visual comparison places exhaustive CART, `S_all`, `S^2`,
   and ERT trees side by side on the same fixed data split.
4. We expanded the Methods and Supplement with code-checked pseudocode, exact
   exploration schedules, stopping and fallback behavior, tie handling, and a
   node-level cost analysis.
5. We expanded Table 1 and added a machine-readable one-row-per-entry inventory.
   The regime analysis now relates its conclusions to the observed benchmark
   support.
6. We regenerated the figures at publication size, increased text and legend
   sizes, improved color and marker discrimination, and added uncertainty to the
   main summaries.
7. We documented the software, hardware, cross-validation, seeds, timing
   boundary, result-file completeness, and analysis provenance. A SHA-256
   manifest links the raw result files and analysis source to the inferential
   output.
8. We added keywords, data and code availability, funding, competing-interest,
   CRediT authorship, and manuscript-preparation declarations, and prepared the
   benchmark data as a separate submission file.
9. We expanded the related work to cover foundational tree induction,
   randomized trees, scalable and approximate split finding, online and dynamic
   trees, global tree optimization, multi-dataset statistical comparison, and
   recent split-construction research.
10. We corrected `S_par`, added focused tests, and reran all 20 configurations
    for 100 seeds on every retained entry. The Supplement reports this screening
    separately. No corrected configuration enters the full effort-loss Pareto
    frontier, so `S_par` is not recommended or included in the confirmatory
    family.

The responses below represent every editorial and reviewer comment. The
editorial synthesis contains the full answers; repeated reviewer items refer
back to those answers and state only the reviewer-specific point.

## Response to the Editor and Editorial Synthesis

### E1. Scope and contribution, including the reference to knowledge distillation

**Comment.** The contribution is unclear or difficult to identify. The editorial
summary describes the paper as a plausible extension to knowledge distillation.

**Response.** The manuscript studies decision-tree split search, not knowledge
distillation. We revised the abstract, the final Introduction paragraph, and the
Conclusion to state three contributions:

1. We formulate feature-ordered, secretary-inspired stopping rules for
   within-node CART split search and specify their exploration, selection, and
   fallback behavior.
2. We compare them with exhaustive CART and Extremely Randomized Trees (ERT)
   under one complete-tree protocol while measuring wall-clock fit time and
   evaluated-gain effort separately.
3. We use paired entry-level inference and Pareto comparisons to identify which
   reductions in search remain compatible with stated predictive-loss
   tolerances.

The paper now presents secretary-style splitting as a legitimate middle ground,
not as a replacement for CART in every setting. Interpretation focuses on
methods that are empirically supported by the simultaneous-success analysis or
Pareto-efficient on the observed time-loss frontier. `S_all(f=1/e)` is the
clearest conservative example: for both classification impurities it has small
estimated F1 loss, positive time saving, and support under the primary joint
criterion. `S^2` also passes that criterion. More aggressive methods can save
more time at greater loss. No regression alternative passes the primary
criterion. These results support a task-dependent middle ground, not a universal
speedup or a new CART approximation guarantee.

**Location.** Main manuscript, Abstract and Sections 1, 4, and 5.

### E2. Inferential analysis and confidence intervals

**Comment.** The work should include a full inferential statistical analysis;
confidence intervals are missing for medians and centroids.

**Response.** The analysis treats PMLB entries, not seeds, folds, trees, or
result rows, as the top-level experimental and inferential units. Within each
entry, every method remains paired with exhaustive CART by estimator seed. The
100 runs are therefore described as 100 seeded repetitions on fixed folds, not
as 100 independent experiments.

We use a paired two-stage nonparametric bootstrap with 10,000 replicates, a 95%
confidence level, and fixed analysis seed 20260810. Shared seed identifiers are
resampled jointly across methods and metrics within each entry. At the outer
level, entries are resampled with replacement and receive equal weight. The
analysis reports:

- entry-specific median intervals from paired-seed resampling;
- 95% hierarchical intervals for cross-entry medians;
- separate 95% hierarchical intervals for both coordinates of the
  equal-entry-weight centroids in Figure 1;
- 95% hierarchical intervals for entry-level reliability probabilities;
- task-wise Friedman tests and paired Wilcoxon comparisons with exhaustive CART,
  with Holm correction and paired effect sizes; and
- exact one-sided binomial tests of simultaneous entry-level time and loss
  success.

The primary criterion uses a task-specific 1-unit loss margin. An entry succeeds
only when the same paired-seed summary has positive time saving and loss within
that margin. An exact one-sided binomial test asks whether the simultaneous
success probability exceeds one half. Holm correction covers the nine
non-reference representatives within each task. Margins of 0.5 and 2.5 form one
18-hypothesis sensitivity family. Regression margins are relative RMSE loss in
percent; classification margins are absolute weighted-F1 loss in percentage
points.

Regression entry `564_fried` lacks run 4. That seed is excluded from the complete
paired block for every method and metric on that entry and is not imputed.
Circular moving-block bootstrap sensitivities with block lengths 5 and 10 assess
run-index dependence. Resampling cannot remove bias from the fixed method order
or reconstruct execution chronology across independently scheduled shards.

The regression corpus contains related entries. A sensitivity analysis collapses
62 Friedman variants into five generator families and two exact CPU alias pairs
into two further families, leaving 54 equally weighted regression families. This
analysis gives the same primary decision: no regression alternative is
supported.

For `S_all(f=1/e)`, the equal-entry mean time saving was 16.44% (95%
hierarchical interval 15.45 to 17.20) in regression, 12.74% (11.31 to 14.07)
for Gini classification, and 12.59% (11.20 to 13.85) for entropy
classification. Its corresponding predictive losses were 2.67% RMSE (1.97 to
3.46), 0.07 weighted-F1 points (-0.12 to 0.25), and 0.03 weighted-F1 points
(-0.17 to 0.20). No regression alternative passed the primary criterion after
Holm correction. In both classification tasks, `S^2`, `S_all`, and ERT with all
features passed. Their simultaneous-success counts were 97/122, 105/122, and
85/122 for Gini, and 92/122, 106/122, and 85/122 for entropy. `S` and
prophet-style search did not pass the corrected same-entry test. The Results
report effect estimates and intervals before adjusted p-values.

The intervals describe heterogeneity under resampling of the retained PMLB
entries and observed estimator seeds. Because PMLB is a curated corpus rather
than a probability sample, they are not population-wide coverage statements for
all tabular prediction tasks.

**Location.** Main manuscript, Section 3.2, Figures 1 and 2, and Table 2.
Supplement, Section S5 and Tables S3 to S5.

### E3. Why one tree rather than a full reference tree or ensemble

**Comment.** Explain more clearly why the study does not use a full reference
tree or an ensemble.

**Response.** The former wording was ambiguous. Every benchmark fit already uses
a complete reference tree. The exhaustive CART splitter is applied recursively
at every internal node under the same tree builder, stopping constraints,
cross-validation folds, and maximum depth 20 as the alternative splitters. The
experiment is not restricted to one root split or a stump.

Here, "single-tree" means one complete tree rather than a forest or boosted
ensemble. The estimand is the effect of replacing the within-node search rule
while holding the greedy tree-growing procedure fixed. Bagging introduces
resampling, aggregation, tree count, and parallel execution; random forests also
introduce feature subsampling. Those mechanisms can alter both the predictive
cost of a suboptimal split and the accumulation of per-node time differences.
The present benchmark therefore makes no ensemble-performance claim. A matched
tree-count and matched-compute ensemble study is separate future work.

We also added a reproducible visual comparison to resolve the terminology at a
glance. Supplementary Figure S2 shows the first three levels of four complete
trees: exhaustive CART, `S_all(f=1/e)`, `S^2(f=1/e)`, and ERT with all features.
All four use the same classification entry, fold, training data, Gini criterion,
maximum depth 20, and estimator seed 42. The generation script selects a
medium-size entry by a stated inventory rule with selection seed 20260817 and
uses fold 1 of the fixed fivefold partition. It records the selected entry and
tree sizes and regenerates the figure from those inputs. The comparison is
illustrative, is not used in the benchmark inference, and adds no performance
claim.

If "full reference tree" instead means a globally optimized tree, that method
answers a different question because it changes the whole topology rather than
the local split-search rule inside a fixed greedy builder. The related-work
section now cites that literature.

**Location.** Main manuscript, Sections 1, 2.2, 3.1, and 5. Supplementary Figure
S2 and `examples/early_stop_trees/supp_plot_tree_comparison.py`.

### E4. Figure readability and explanation

**Comment.** The figures should be easy to read and their explanation should be
beneficial.

**Response.** We regenerated the main and supplementary figures at their intended
publication dimensions. The revision increases type and legend sizes, uses
distinct colors and redundant marker encodings, and reduces the prominence of
background elements. The main figures are vector PDFs, and the supplementary
composites are exported at 600 dpi. We checked the compiled PDFs at final size
for overlaps, clipping, and legend placement.

Main Figures 2 and 3 are no taller than 4.35 inches before journal-width scaling,
which keeps them within the journal's half-page figure limit. Figure 2 contains
the joint operating criteria; the complementary loss-only profiles remain in
Supplementary Figure S6.

Figure 1 emphasizes entry-level centroids and shows separate 95% hierarchical
intervals for both coordinates. Figure 2 uses entries as the aggregation unit
and shows uncertainty for joint reliability. Figure 3 is a 3-by-3 regime map:
columns are regression, Gini classification, and entropy classification, and
rows use loss margins 0.5, 1, and 2.5 in their task-specific units. For each
observed entry, methods above the stated median-loss margin are removed and the
fastest remaining method is selected. Exhaustive CART is always feasible. The
pale 7-nearest-neighbor background is descriptive interpolation, not a confidence
region or a rule for unseen entries.

**Location.** Main manuscript, Section 4 and Figures 1 to 3. Supplement, Figures
S1 to S14.

### E5. Apparent reference to a previous publication

**Comment.** If the abstract refers to a previous publication by the author,
that work should be cited and the relationship made explicit.

**Response.** The phrase "earlier implementations" referred to software
iterations within this study, not to a previous publication. We removed it and
rewrote the abstract as a self-contained statement of the question, benchmark,
main findings, and scope. The abstract no longer implies a comparison with
unpublished or uncited prior work.

**Location.** Main manuscript, Abstract.

### E6. Foundational and current literature

**Comment.** The bibliography is too sparse for an old topic, and the manuscript
should review more current literature related to the results.

**Response.** We expanded the Introduction and bibliography to cover
foundational tree induction and cut-point evaluation, randomized and scalable
trees, streaming and dynamic methods, global tree optimization, secretary and
prophet theory, and multi-dataset inference. Each source supports a specific
nearby claim, and its metadata were checked against a primary publisher or
proceedings record. The main bibliography now contains 22 cited entries.

**Location.** Main manuscript, Sections 1 and 5, and References.

### E7. Pseudocode and algorithm clarity

**Comment.** Add pseudocode to clarify the algorithms.

**Response.** The main article now gives a common explore-select algorithm, and
the supplement gives method-specific pseudocode. The algorithms define the
randomized feature order, continuous-threshold draws, exploration prefix, exact
feature scans, acceptance comparisons, tie handling, stopping event, and
fallback when no later feature qualifies. The implemented schedules are
expressed as feature-prefix fractions: `f=1/e`, `f=0.1`,
`f=1/log(N_dataset)`, and `f(n)=1/sqrt(n_node)`.

The prefix target is computed from the pre-scan feature budget. Features newly
found constant at a node are skipped without reducing that target. For `S^2`,
the schedule controls the outer feature prefix while the inner
sampled-threshold fraction remains fixed at `1/e`.

We checked the pseudocode against the Cython control flow and revised the method
names where the theoretical assumptions do not transfer. `S` uses
sampled-threshold exploration followed by feature-level exact selection. The
former "block-rank" method is called "blockwise rank-inspired," and the former
"1-sample prophet" method is called "prophet-style." The implementation does
not satisfy the original online models' independence, random-order, or
irrevocability assumptions, so the paper claims no corresponding approximation
guarantees.

**Location.** Main manuscript, Section 2.1 and Algorithm 1. Supplement, Sections
S1 and S2 and Algorithms S1 to S7.

### E8. Dataset characteristics and operating scenarios

**Comment.** Include a summary of the main dataset characteristics, such as the
number of instances and features, to help identify scenarios in which each
strategy is effective; this can also inform the Conclusions.

**Response.** Table 1 now reports the 25th percentile, median, 75th percentile,
and range across retained PMLB entries. The 113 regression entries have
sample-size quartiles 250/500/1,000 (range 47 to 40,768) and feature-count
quartiles 5/10/25 (range 2 to 124). The 122 classification entries have
sample-size quartiles 202/736/3,196 (range 32 to 58,000), feature-count quartiles
6.25/13/29 (range 2 to 240), and class-count quartiles 2/2/4 (range 2 to 26).
Gini and entropy use the same classification corpus and are summarized once.

A machine-readable supplementary inventory provides one row per entry with
`n`, `p`, `n*p`, `p/n`, valid-run count, class count, majority-class proportion
for classification, and target standard deviation for regression. The
classification majority-class proportion has quartiles 33.7%/51.4%/66.4% and
range 4.1% to 98.5%.

The Results relate these descriptors to the tolerance-constrained regime maps.
Under the task-specific endpoints used here, retained classification entries
meet the displayed constraints more often than regression entries. At the
1-unit margin, ERT is the fastest admissible family for 62/122 Gini and 68/122
entropy entries. It wins 23/32 and 22/32 classification entries with
`n >= 3196`, and 22/31 entries for each impurity in the lowest `p/n` quartile
(`p/n <= 0.00769`). Regression retains exhaustive CART for 63/113 entries
overall and 24/43 entries with `n >= 1000`; `S_all` wins 13/113. These cutoffs
are corpus quartiles, not deployment thresholds. The comparison is descriptive,
not an intrinsic task-type effect, because relative RMSE and weighted-F1
percentage-point margins are different endpoints. The nearest-neighbor
background is not evidence of causal or universal boundaries.

**Location.** Main manuscript, Table 1, Sections 4.2 and 5, and Figure 3.
Supplement, Section S4 and Figure S8.

## Response to Reviewer 1

### R1.1. Contribution and practical value beyond `S_all`

**Comment.** The contribution is difficult to identify. If accuracy is preserved
but wall-clock gains are modest, is the paper justified only by an improvement
for `S_all`?

**Response.** See response E1. The paper does not depend on one favorable
`S_all` result. It now interprets secretary-style methods only where they are
empirically supported by the simultaneous-success analysis or Pareto-efficient
in the observed comparisons. The main result is that selected secretary-style
rules, especially `S_all(f=1/e)` in classification, form a legitimate middle
ground between exhaustive search and more aggressive randomization. The negative
regression result and the distinction between gain counts and wall-clock time
are part of that contribution.

**Location.** Main manuscript, Abstract and Sections 1, 4, and 5.

### R1.2. Full inferential analysis and confidence intervals

**Comment.** The benchmark effort should be used for full inferential analysis;
confidence intervals for medians and centroids are missing.

**Response.** See response E2. Figure 1 and the revised tables now report the
requested entry-level, cross-entry, and centroid intervals. The supplement gives
the paired tests, simultaneous-success analysis, and regression family-balance
sensitivity.

**Location.** Main manuscript, Section 3.2, Figures 1 and 2, and Table 2.
Supplement, Section S5 and Tables S3 to S5.

### R1.3. "Only one tree? Why not a full reference tree?"

**Comment.** Only one tree is evaluated; explain why a full reference tree is not
used.

**Response.** See response E3. Every estimator is a complete recursively grown
tree, not a stump. The new reproducible visual comparison shows the exhaustive
CART reference tree beside complete `S_all`, `S^2`, and ERT trees fitted to the
same fixed data split. The study remains a single-tree comparison and makes no
ensemble-performance claim.

**Location.** Main manuscript, Sections 1, 2.2, 3.1, and 5. Supplementary Figure
S2 and `examples/early_stop_trees/supp_plot_tree_comparison.py`.

### R1.4. Figure 3 readability and explanation

**Comment.** Figure 3 is difficult to read and would benefit from explanation.

**Response.** See response E4. Figure 3 was rebuilt at final size with larger
type, stronger observed-point outlines, lower-opacity interpolation, and a
reorganized legend. The Results now explain the loss filter and fastest-admissible
method rule in the order applied.

**Location.** Main manuscript, Section 4.2, Figure 3, and its caption.

### R1.5. Abstract and possible prior publication

**Comment.** The abstract appears to refer to an earlier paper; reformulate it
and cite that work in the body if applicable.

**Response.** See response E5. The phrase referred to software iterations within
this study, not an earlier publication. It has been removed, and the abstract is
self-contained.

**Location.** Main manuscript, Abstract.

### R1.6. Table 1 quartiles and ranges

**Comment.** Add the 25th and 75th percentiles and range to Table 1.

**Response.** See response E8. Table 1 now gives Q1/median/Q3 and ranges for
observations and features, plus class-count summaries for classification. The
supplement provides the complete entry inventory.

**Location.** Main manuscript, Table 1. Supplement, Section S4 and the
machine-readable inventory cited there.

### R1.7. Software and hardware configuration

**Comment.** Report language and package versions and the hardware used so future
comparisons are possible.

**Response.** The reproducibility subsection now reports Python 3.10.18, NumPy
1.26.4, scikit-learn 1.7.2, treeple 0.10.3, macOS 15.2 on ARM64, an Apple M3
with eight CPU cores and 16 GB memory, deterministic unshuffled fivefold KFold
or StratifiedKFold partitions, estimator seeds 42 to 141, maximum depth 20, and
the common tree-growth settings.

Fit time is the estimator fit time returned by
`sklearn.model_selection.cross_validate`; data loading, scoring, result
serialization, and figure generation are excluded. The reproducibility
materials contain benchmark metadata, reviewed source, analysis scripts,
inferential tables, and a 312-entry hash manifest covering the analysis sources
and all 300 raw run files. The fixed method order is stated as a limitation of
the timing measurements. Because the exact invocation is unavailable and the driver
defaults to an `n*p` cap, the revision states that disabling this cap cannot be
verified; the largest retained product is 834,870.

**Location.** Main manuscript, Section 3.1 and Data and code availability.
Supplement, Sections S4, S5.3, and S6.

### R1.8. Sparse bibliography

**Comment.** There are too few references for a long-established topic.

**Response.** See response E6. The related-work discussion now covers the main
foundational and current literatures relevant to split construction, and the
main article cites 22 references.

**Location.** Main manuscript, Section 1 and References.

### R1.9. Pseudocode

**Comment.** Add pseudocode to clarify the algorithm.

**Response.** See response E7. The main algorithm gives the common randomized
feature-prefix procedure, and the supplement specifies each method's exploration
score, exact selection scan, stopping comparison, tie behavior, and fallback.

**Location.** Main manuscript, Section 2.1 and Algorithm 1. Supplement, Sections
S1 and S2 and Algorithms S1 to S7.

## Response to Reviewer 2

### R2.1. Expanded strategy descriptions and reproducibility

**Comment.** Expand the description of the secretary-style strategies to improve
clarity and reproducibility.

**Response.** See response E7. The Methods and pseudocode now define `n`, the
pre-scan feature budget `m`, admissible boundaries `C_j`, feature order,
exploration prefixes, sampled thresholds, exact scans, stopping, fallback, and
ties. Newly discovered constant features are skipped without reducing the
exploration target. The text also distinguishes `1/log(N_dataset)`, fixed over a
tree, from `1/sqrt(n_node)`, recomputed at each node, and states the dense,
finite-valued, no-monotonic-constraint scope.

**Location.** Main manuscript, Section 2.1. Supplement, Sections S1 and S2 and
Algorithms S1 to S7.

### R2.2. Computational-complexity analysis

**Comment.** The manuscript lacks a computational complexity analysis that
characterizes the differences among the approaches.

**Response.** The revision adds a fixed-node cost analysis that separates
proxy-gain evaluations from sorting, min/max passes, partitioning, random-number
generation, sampled-threshold sorting, calibration, replay, memory traffic, and
final split application. If `C_j` is the number of candidate boundaries and
`C_all` their sum, exhaustive CART performs at most `C_all` gain evaluations
plus feature sorting and scanning. ERT evaluates at most one sampled threshold
per selected feature. Secretary-style costs are expressed in terms of
exploration features, visited selection features, sampled draws, and exact
boundaries scanned before acceptance.

At a fixed node, `S`, `S_all`, and the blockwise rank-inspired heuristic do not
exceed the exhaustive exact-gain count. `S^2` can exceed it
because it adds sampled gains before a full exact exploration-feature scan. The
prophet-style rule has the safe bound `m + C_all`: it first evaluates one sampled
gain per feature and then replays exact scans while skipping each sampled
partition position. Corrected `S_par` adds sampling, calibration, and replay
work. Its separate 100-run screening reduced recorded gain evaluations, but no
configuration entered the full effort-loss Pareto frontier. Different selected
splits can also create different descendants, so a fixed-node ordering need not
hold for whole-tree effort.

The effort metric is a search proxy, not a wall-clock complexity model. CART and
ERT effort is reconstructed for successful internal-node searches, whereas
direct early-stop counters include failed calls that produce leaves. The paper
therefore retains the conservative total-tree diagnostic and removes the
non-comparable per-call normalization from Figure S14.

**Location.** Main manuscript, Section 2.2. Supplement, Section S3 and Table S1.

### R2.3. Dataset characteristics and effective scenarios

**Comment.** Summarize instances, features, and other relevant dataset
properties; use them to identify where each strategy is effective and consider
this in the Conclusions.

**Response.** See response E8. Table 1 and the supplementary inventory report the
requested entry characteristics. The Results and Conclusion relate them to the
tolerance-constrained regime maps while treating corpus quartiles as descriptive
support, not deployment thresholds or causal boundaries.

**Location.** Main manuscript, Table 1, Sections 4.2 and 5, and Figure 3.
Supplement, Section S4 and Figure S8.

### R2.4. Figure quality and color-label readability

**Comment.** Improve figure quality because the text identifying the color
coding is difficult to read.

**Response.** See response E4. All figures were regenerated at final physical
size, with vector main figures, 600-dpi supplementary composites, larger
post-scaling type and legends, distinct method-family colors, redundant marker
shapes, and lower-opacity backgrounds. We checked the compiled PDFs for
overlaps, clipping, and legend placement.

**Location.** Main manuscript, Figures 1 to 3. Supplement, Figures S1 to S15.

## Response to Reviewer 3

### R3.1. Expanded references

**Comment.** The final reference section is too sparse and should be expanded.

**Response.** See response E6. The related-work discussion and bibliography now
cover foundational and current split construction, scalable and approximate
tree induction, streaming and dynamic trees, randomized trees and ensembles,
global tree optimization, statistical comparison across entries, and the
secretary and prophet literature. The main bibliography contains 22 cited
entries.

**Location.** Main manuscript, Sections 1 and 5, and References.

## Correcting and evaluating `S_par`

The implementation review found two defects in the `S_par` calibration used for
the current 100-run benchmark. In regression, the fitted working law used a proxy
with a node-dependent constant, so the stopping decision could change when every
target value was translated by the same amount. In classification, the normal
inverse-CDF helper reversed the sign convention and mapped nominal upper
quantiles to lower-tail values.

Results from that implementation are not used. The paper excludes `S_par` from
the main comparison and confirmatory inference.

A corrected implementation is now included. It evaluates the centered
between-child sum of squares for regression, uses the correct normal-quantile
direction in classification, and generates each sampled threshold vector once
before a monotone scan. Focused tests cover quantile sign and symmetry,
parametric sampling counters, and regression target-translation invariance. The
corrected implementation was then rerun for all 20 configurations, 100 seeds,
113 regression entries, and 122 classification entries under each impurity.
No configuration enters the full centroid effort-loss Pareto frontier. The
exploratory display configuration, `samples=sqrt_n,q=0.9`, saves
46.06% of recorded gain evaluations in regression and about 23% in
classification, but its regression loss is 11.19%.

The corrected timing campaign used the same host, operating system, Python,
scikit-learn, folds, and seeds, but was run later with NumPy 2.2.6 rather than
1.26.4. The Supplement therefore does not report a wall-clock comparison for
this screening. Predictive loss and gain-evaluation effort remain paired by
entry and seed. The negative screening
result is reported rather than omitted: corrected `S_par` is reproducible, but
the tested calibration does not provide a competitive compromise.

**Location.** Main manuscript, Sections 4.1 and 5. Supplement, the `S_par` status
box, Algorithm S7, corrected screening subsection, and Figure S15.

Before submission, we checked the numerical values, references, algorithms, and
figures against the analysis output and compiled manuscripts. The author will
perform the final proofread of all submission files.
