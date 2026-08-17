# Response to the Editor and Reviewers

**Manuscript:** PRLETTERS-D-26-00639, "Secretary-Style Split Search for Decision Trees"

All numerical results below come from the 10,000-replicate analysis export
finalized on August 17, 2026. Page references correspond to the final
compiled revision.

## Cover Response

Dear Editor and Reviewers,

Thank you for the detailed comments. We revised the presentation, rechecked the
implementation and numerical archive, expanded the statistical analysis, and
narrowed claims where the evidence required it.

The main revisions are as follows.

1. We rewrote the abstract and Introduction around a three-part contribution:
   a family of feature-ordered stopping
   rules for CART split search; a controlled complete-tree comparison that
   distinguishes gain-evaluation effort from wall-clock fit time; and a paired
   multi-dataset analysis of the resulting time--prediction trade-offs.
2. We added a full entry-level inferential analysis. It includes paired
   hierarchical 95% confidence intervals for cross-entry medians, both
   coordinates of the method centroids, and reliability probabilities, together
   with omnibus tests, multiplicity-adjusted paired comparisons, and joint
   time--loss tests.
3. We clarified that every fit is a complete recursively grown depth-limited
   tree, not a root split or stump. The deliberate single-tree scope isolates the
   effect of the within-node splitter before bagging, aggregation, tree count,
   feature subsampling, and parallel execution introduce separate effects.
4. We expanded the Methods and Supplement with code-checked pseudocode, exact
   exploration schedules, acceptance and fallback behavior, tie handling,
   theoretical boundaries, and a node-level computational-cost analysis.
5. We expanded Table 1 with quartiles and ranges and added a machine-readable
   one-row-per-entry inventory. We also connected entry characteristics to
   the regime analysis while keeping those conclusions explicitly conditional
   on the present benchmark.
6. We redesigned the figures at publication dimensions, enlarged their text and
   legends, improved color and marker discrimination, added uncertainty to the
   principal summaries, and rewrote the Figure 3 explanation around its exact
   tolerance-constrained winner rule.
7. We documented the software, hardware, cross-validation, random seeds, timing
   boundary, archive completeness, and analysis provenance. A SHA-256 manifest
   links the raw results and analysis source to the inferential export.
8. We added keywords, data and code availability, funding, competing-interest,
   CRediT authorship, and manuscript-preparation declarations, and prepared the
   supporting benchmark archive as a separate submission file.
9. We broadened the related work to include foundational tree induction,
   randomized trees, scalable and approximate split finding, online and dynamic
   trees, global tree optimization, multi-dataset statistical comparison, and
   recent work directly relevant to split construction.

Our implementation audit also identified two defects in the archived
parametric-calibration screen, `S_par`: its regression statistic is
target-translation dependent, and its classification normal-quantile helper has
a reversed sign. We therefore removed `S_par` from the main article, all
confirmatory inference, recommendations, and theoretical-validation claims. We
retain the archived output only as a clearly labeled supplementary exploratory
ablation, disclose both defects, and do not use it as evidence for any
conclusion. Correcting either behavior would define a different implementation
and require a dedicated rerun. The archived results are therefore disclosed only
as an ablation and are not mixed with results for a corrected method.

The point-by-point responses below reproduce or faithfully summarize every
editorial and reviewer request. Locations refer to the revised manuscript and
supplement.

## Response to the Editor and Editorial Synthesis

### E1. Scope and contribution, including the reference to knowledge distillation

**Comment.** The contribution is unclear or difficult to identify. The editorial
summary describes the paper as a plausible extension to knowledge distillation.

**Response.** The manuscript studies decision-tree split search, not knowledge
distillation. We revised the abstract, the final Introduction paragraph, and the
Conclusion to state three contributions:

1. We formulate and audit feature-ordered, secretary-inspired stopping rules for
   within-node CART split search, including their exact exploration,
   selection, and fallback behavior.
2. We compare them with exhaustive CART and Extremely Randomized Trees (ERT)
   under one complete-tree protocol while separately measuring wall-clock fit
   time and evaluated-gain effort.
3. We use paired entry-level inference to identify where reduced search is or
   is not compatible with a specified predictive-loss tolerance.

The revised framing does not depend on one favorable `S_all` result. Its
scientific contribution is the controlled characterization of a design space,
including the negative result that large reductions in recorded gain evaluations
do not necessarily become comparable wall-clock savings, and the finding that
classification and regression occupy different operating regimes. We explicitly
avoid claiming a universal speedup or a new approximation guarantee for CART.

**Location.** Main manuscript, Abstract, Sections 1 and 5.

### E2. Inferential analysis and confidence intervals

**Comment.** The work should include a full inferential statistical analysis;
confidence intervals are missing for medians and centroids.

**Response.** The revised analysis treats PMLB entries, not seeds, folds,
trees, or result rows, as the top-level experimental and inferential units. For
each entry and method, outcomes remain paired with exhaustive CART by seed.
The 100 runs are therefore described as "100 seeded repetitions on fixed folds,"
not as 100 independent experiments.

We use a paired two-stage nonparametric bootstrap with 10,000 replicates, a 95%
confidence level, and fixed analysis seed 20260810. Within each entry, shared
seed identifiers are resampled jointly across methods and metrics; at the outer
level, entries are resampled with replacement and receive equal weight. The
analysis reports:

- entry-specific median intervals from paired-seed resampling;
- 95% hierarchical intervals for cross-entry medians;
- separate 95% hierarchical intervals for the time and loss coordinates of the
  equal-entry-weight centroids in Figure 1;
- 95% hierarchical intervals for entry-level reliability probabilities;
- task-wise Friedman omnibus tests and paired Wilcoxon comparisons with
  exhaustive CART, with Holm correction and paired effect sizes; and
- exact one-sided binomial tests of simultaneous entry-level time and loss
  success.

The revision-designated primary criterion uses the task-specific 1-unit loss
margin. An entry succeeds only if the same paired-seed summary has positive time
saving and loss within that margin. An exact one-sided binomial test asks whether
the simultaneous-success probability exceeds one half. Holm correction is
applied across the nine non-reference representatives within each task. Margins
of 0.5 and 2.5 form one 18-hypothesis sensitivity family. The full
hyperparameter grid remains exploratory.
Regression margins refer to relative RMSE loss in percent; classification
margins refer to absolute weighted-F1 loss in percentage points. The revised
manuscript keeps those units explicit rather than treating them as interchangeable.

Regression dataset `564_fried` lacks archived run 4. That seed is excluded from
the complete paired block for every method and metric on that dataset; it is not
imputed. Timing diagnostics and circular moving-block bootstrap sensitivities
with block lengths 5 and 10 assess run-index dependence. We also state the
remaining limitation: resampling cannot remove bias caused by the fixed method
execution order or reproduce unavailable shard chronology.

The regression corpus contains related benchmark entries. A sensitivity analysis
therefore collapses 62 Friedman variants into five generator families and two
exact CPU alias pairs into two further families, leaving 54 equally weighted
regression families. This family-balanced analysis does not change the primary
decision: no regression alternative is supported.

For example, in regression, the equal-entry mean of median time saving for
`S_all(f=1/e)` was 16.44% (95% hierarchical interval 15.45--17.20), with
2.67% equal-entry mean RMSE loss
(1.97--3.46). In Gini and entropy classification, its centroid time savings
were 12.74% (11.31--14.07) and 12.59% (11.20--13.85), while weighted-F1 losses
were 0.07 (-0.12--0.25) and 0.03 (-0.17--0.20) percentage points. No regression
alternative supported the primary joint 1-unit criterion after Holm correction. In
both classification criteria, `S^2`, `S_all`, and ERT with all features did so.
The corresponding simultaneous-success counts were 97/122, 105/122, and 85/122
for Gini, and 92/122, 106/122, and 85/122 for entropy. `S` and prophet-style
search did not pass the corrected same-entry test. We lead the Results with effect
estimates and intervals; adjusted p-values are secondary.

These intervals describe heterogeneity under resampling of the retained PMLB
corpus and archived seeds. Because PMLB is a curated corpus rather than a
probability sample, the revised text does not present them as population-wide
coverage over all tabular prediction tasks.

**Location.** Main manuscript, Section 3.2, Figures 1--2, and Table 2.
Supplement, Section S5 and Tables S3--S5.

### E3. Why one tree rather than a full reference tree or ensemble

**Comment.** Explain more clearly why the study does not use a full reference
tree or an ensemble.

**Response.** Our former wording was ambiguous. Every benchmark
fit already uses a complete reference tree: the standard exhaustive CART
splitter is applied recursively at every internal node, under the same tree
builder, stopping constraints, cross-validation folds, and maximum depth 20 as
the alternative splitters. The experiment is not restricted to one root split
or to a stump.

By "single-tree" we mean one complete tree rather than a forest or boosted
ensemble. This is deliberate. The estimand is the consequence of replacing the
within-node search rule while holding the greedy tree-growing protocol fixed.
Bagging adds bootstrap resampling, variance reduction, aggregation, tree count,
and parallel execution; random forests also add feature subsampling. Those
mechanisms can either attenuate a suboptimal individual split or amplify small
per-node time differences across many trees, so ensemble behavior cannot be
deduced from the present benchmark.

We did not add a small post hoc forest experiment and present it as a substitute
for the controlled mechanism study. Instead, we now state the scope prominently,
limit every conclusion to complete individual trees, and identify a matched-tree-
count and matched-compute bagging study as the principal next step. If "full
reference tree" was intended to mean a globally optimized tree, we also clarify
that global topology optimization answers a different question from replacing a
local splitter inside a fixed greedy builder; the revised related work cites this
distinct literature.

**Location.** Main manuscript, Sections 1, 2.2, 3.1, and 5. Supplement,
Section S4.

### E4. Figure readability and explanation

**Comment.** The figures should be easy to read and their explanation should be
beneficial.

**Response.** The main and supplementary figures were regenerated at
their intended publication dimensions, with larger type, larger legends,
accessible and more distinct colors, and redundant marker encodings. The main
figures are supplied as vector PDFs and the supplementary composites as
high-resolution images. The reliability figure uses the revision-designated
representative set; the Pareto and regime figures retain schedule variants with
reorganized multi-column legends. We inspected the rendered manuscript at final
size rather than judging only the full-size source images.

Figure 1 now emphasizes the dataset-level centroids and shows separate 95%
hierarchical intervals for both coordinates. Figure 2 uses datasets as the unit
of aggregation and shows uncertainty for the joint reliability summaries.
Figure 3 was redesigned as a readable 3-by-3 regime map and is explained step by
step in the Results: columns are regression, Gini classification, and entropy
classification; rows are loss tolerances 0.5, 1, and 2.5 in the task-specific
units. For each observed
dataset, methods above the displayed median-loss tolerance are discarded and
the fastest remaining method is selected. Exhaustive CART is always feasible and
wins only when no qualifying alternative is faster. The pale 7-nearest-neighbor
background is explicitly described as a descriptive interpolation, not a
confidence region or a validated rule for unseen datasets.

Captions remain descriptive and concise; interpretation is in the Results and
Discussion.

**Location.** Main manuscript, Section 4 and Figures 1--3. Supplement,
Figures S1--S14.

### E5. Apparent reference to a previous publication

**Comment.** If the abstract refers to a previous publication by the author,
that work should be cited and the relationship made explicit.

**Response.** The former phrase
"earlier implementations" referred to software iterations conducted within the
present study, not to a previous publication. It was unnecessary and could
reasonably be misread. We removed it and rewrote the abstract as a self-contained
statement of the research question, benchmark, principal findings, and scope.
It contains no implicit comparison with unpublished or uncited prior work.

**Location.** Main manuscript, Abstract.

### E6. Foundational and current literature

**Comment.** The bibliography is too sparse for an old topic, and the manuscript
should review more current literature related to the results.

**Response.** We expanded the Introduction and bibliography to cover
foundational tree induction and cut-point evaluation, randomized and scalable
trees, streaming and dynamic methods, global tree optimization, secretary and
prophet theory, and multi-dataset inference. Each added source supports a
specific nearby claim, and its metadata were checked against a primary
publisher or proceedings record. The main bibliography now contains 22 cited
entries.

**Location.** Main manuscript, Sections 1 and 5, and References.

### E7. Pseudocode and algorithm clarity

**Comment.** Add pseudocode to clarify the algorithms.

**Response.** The main article now gives a compact common
explore--select algorithm, and the supplement gives method-specific pseudocode.
The algorithms define the randomized feature order, continuous-threshold draws,
exploration prefix, exact feature scans, strict or inclusive acceptance
comparisons, tie handling, early stopping event, and fallback when no later
feature qualifies. The implemented schedules are expressed as feature-prefix
fractions, including `f=1/e`, `f=0.1`, `f=1/log(N_dataset)`, and
`f(n)=1/sqrt(n_node)`, rather than being mislabeled as uniform candidate counts.
The revision also states that the prefix target is computed from the pre-scan
feature budget: features newly found constant at a node are skipped without
reducing that target. For `S^2`, the schedule controls the outer feature prefix,
while the inner sampled-threshold fraction remains fixed at `1/e`.

The pseudocode was checked against the Cython control flow. This audit also led
us to use precise labels: `S` is sampled-threshold exploration followed by
feature-level exact selection; the former "block-rank" label is now
"blockwise rank-inspired"; and "1-sample prophet" is now "prophet-style"
because the assumptions of the corresponding prophet inequality do not hold in
the tree implementation. The manuscript clearly separates theoretical
motivation from guarantees that do not transfer to the grouped and revisitable
candidate structure used here.

**Location.** Main manuscript, Section 2.1 and Algorithm 1. Supplement,
Sections S1--S2 and Algorithms S1--S7.

### E8. Dataset characteristics and operating scenarios

**Comment.** Include a summary of the main dataset characteristics, such as the
number of instances and features, to help identify scenarios in which each
strategy is effective; this can also inform the Conclusions.

**Response.** Table 1 now reports the 25th percentile, median, 75th
percentile, and range across retained PMLB entries. The 113 regression
entries have sample-size quartiles 250/500/1,000 (range 47--40,768) and feature-
count quartiles 5/10/25 (range 2--124). The 122 classification entries have
sample-size quartiles 202/736/3,196 (range 32--58,000), feature-count quartiles
6.25/13/29 (range 2--240), and class-count quartiles 2/2/4 (range 2--26). Gini
and entropy use the same classification corpus and are summarized once.

A machine-readable supplementary inventory provides one row per entry with
`n`, `p`, `n*p`, `p/n`, valid-run count, class count and majority-class
proportion for classification, and target standard deviation for regression.
For context, the classification majority-class proportion has quartiles
33.7%/51.4%/66.4% and range 4.1%--98.5%.

The revised Results and Conclusion connect these descriptors to the regime maps
cautiously. Under the task-specific endpoints used here, retained classification
entries meet the displayed constraints more often than regression entries. This
is a descriptive benchmark contrast, not an intrinsic task-type effect, because
relative RMSE and weighted-F1 percentage-point margins are different endpoints.
We do not present the interpolated regimes as universal deployment rules or
causal effects of `n` or `p`.

**Location.** Main manuscript, Table 1, Sections 4.2 and 5, and Figure 3.
Supplement, Section S4 and Figure S8.

## Response to Reviewer 1

Reviewer 1's comments prompted revisions to the contribution statement,
statistical analysis, algorithm descriptions, and experimental scope.

### R1.1. Contribution and practical value beyond `S_all`

**Comment.** The contribution is difficult to identify. If accuracy is preserved
but wall-clock gains are modest, is the paper justified only by an improvement
for `S_all`?

**Response.** Our previous presentation may have implied that the contribution
depended on `S_all`. The revision instead compares several stopping mechanisms
with CART and four ERT feature budgets under a common protocol. It distinguishes
gain-evaluation effort from wall-clock time, reports dataset-level intervals and
joint time--loss tests, and documents cases in which theoretical proxies do not
transfer. At the task-specific 1-unit margin, no regression alternative meets the primary joint
criterion; `S^2`, `S_all`, and ERT with all features meet it for both
classification impurities. `S` and prophet-style search do not pass the corrected
same-entry test. These results are task-specific and do not identify a uniformly
best method.

**Location.** Main manuscript, Abstract and Sections 1, 4, and 5.

### R1.2. Full inferential analysis and confidence intervals

**Comment.** The benchmark effort should be used for full inferential analysis;
confidence intervals for medians and centroids are missing.

**Response.** We implemented the paired entry-level analysis
described in response E2. PMLB entries are the inferential units; fixed folds are
not treated as replicates; shared seeds are resampled jointly to preserve pairing;
and 10,000 hierarchical bootstrap replicates produce 95% intervals for
entry-specific and cross-entry medians, centroid coordinates, and reliability
probabilities. Figure 1 now displays centroid intervals, and the numerical
intervals appear in the revised tables and machine-readable inferential export.

We additionally report task-wise omnibus tests, Holm-adjusted paired comparisons
with effect sizes, and a primary joint time-saving/noninferiority-style test at a
task-specific 1-unit loss margin, with 0.5- and 2.5-unit sensitivity analyses. We call this a joint
benchmark operating criterion rather than a clinical-style noninferiority trial.
For `S_all(f=1/e)`, centroid time saving was 16.44% (15.45--17.20) in
regression, 12.74% (11.31--14.07) for Gini, and 12.59% (11.20--13.85) for
entropy; the corresponding signed rank-biserial effects for time were 1.000,
0.994, and 0.988. Its centroid predictive-loss intervals were 1.97--3.46%
(regression), -0.12--0.25 F1 points (Gini), and -0.17--0.20 F1 points
(entropy). For `S_all`, the Holm-adjusted primary simultaneous-success p-value
was 1.000 in regression, 5.16 x 10^-16 for Gini, and 8.18 x 10^-17 for entropy.
The family-balanced regression sensitivity also did not support a primary claim.

**Location.** Main manuscript, Section 3.2, Figures 1--2, and Table 2.
Supplement, Section S5 and Tables S3--S5.

### R1.3. "Only one tree? Why not a full reference tree?"

**Comment.** Only one tree is evaluated; explain why a full reference tree is not
used.

**Response.** Our use of "single-tree" was ambiguous. The reference is a
complete tree, not one split or a stump: exhaustive CART is grown
recursively, subject to the common stopping rules and up to depth 20, in every
cross-validation fold under the same
tree-level conditions as every alternative. We now state this explicitly.

If "full reference" means a forest or boosted ensemble, that is an important
next experiment, but it is a different estimand. Bagging and boosting
add aggregation, resampling or reweighting, tree count, and potentially feature
subsampling and parallelism. They can change both the predictive cost of a
suboptimal split and the scaling of small per-node time savings. We therefore
retain the complete-single-tree design as the first controlled step, make no
ensemble-performance claim, and identify matched-compute bagging as the primary
follow-up. If the comment instead refers to global tree optimization, we now
cite that literature and explain why changing the whole topology is not a
drop-in baseline for a local split-search replacement.

**Location.** Main manuscript, Sections 1, 2.2, 3.1, and 5.

### R1.4. Figure 3 readability and explanation

**Comment.** Figure 3 is difficult to read and would benefit from explanation.

**Response.** We redesigned both the figure and its explanation.
Typography, contrast, markers, and the representative-method legend are larger
at final size. Columns now unambiguously represent regression, Gini, and entropy;
rows represent loss tolerances 0.5, 1, and 2.5 in the task-specific units.

The Results now explain the encoding in operational order. Each method's median
loss and median speed over the 100 seeded repetitions are computed for an
observed dataset. Methods exceeding the row's loss tolerance are discarded. The
point is colored by the fastest remaining method. Exhaustive CART has zero
relative loss and is always feasible. The faint KNN background only interpolates
the observed winner labels and is not inferential evidence. This makes the
figure a tolerance-constrained method-selection map rather than a generic speed
or loss plot.

**Location.** Main manuscript, Section 4.2, Figure 3, and its caption.

### R1.5. Abstract and possible prior publication

**Comment.** The abstract appears to refer to an earlier paper; reformulate it
and cite that work in the body if applicable.

**Response.** The phrase referred to implementation iterations within this
study, not an earlier publication. We removed it rather than requiring the reader
to infer that distinction. The revised abstract is self-contained and states the
problem, compared methods, complete-
tree protocol, benchmark, principal findings, and scope directly.

**Location.** Main manuscript, Abstract.

### R1.6. Table 1 quartiles and ranges

**Comment.** Add the 25th and 75th percentiles and range to Table 1.

**Response.** Table 1 now gives Q1/median/Q3 and range for observations
and features, plus class-count summaries for classification. The values are
generated from the exact retained benchmark inventory rather than transcribed
manually. The regression and classification values are reported in response E8.
The supplement and repository additionally provide the complete per-dataset
inventory and relevant structural descriptors.

**Location.** Main manuscript, Table 1. Supplement, Section S4 and the
machine-readable inventory cited there.

### R1.7. Software and hardware configuration

**Comment.** Report language and package versions and the hardware used so future
comparisons are possible.

**Response.** The reproducibility subsection now reports Python
3.10.18, NumPy 1.26.4, scikit-learn 1.7.2, and treeple 0.10.3; macOS 15.2 on
ARM64; an Apple M3 with eight CPU cores and 16 GB memory; deterministic unshuffled
fivefold KFold or StratifiedKFold partitions; estimator seeds 42--141; and the
common maximum depth 20 and tree-growth settings.

The timing boundary is also explicit: we report the estimator fit time returned
by `sklearn.model_selection.cross_validate`; data loading, scoring, result
serialization, and figure generation are excluded. The reproducibility
materials contain the benchmark metadata, audited source, analysis scripts,
production inferential tables, and a 312-entry hash manifest covering the
analysis sources and all 300 raw run files. The exact temporary compiled
extension and sharded shell invocation were not retained, and we state that
limitation rather than claiming bitwise executable identity. We also disclose
the fixed method-order limitation for wall-clock measurements. Because the exact
invocation is unavailable and the driver defaults to an `n*p` cap, the revision
also states that disabling this cap cannot be verified from the archive; the
largest retained product is 834,870.

**Location.** Main manuscript, Section 3.1 and Data and code availability. Supplement,
Sections S4, S5.3, and S6.

### R1.8. Sparse bibliography

**Comment.** There are too few references for a long-established topic.

**Response.** As detailed in response E6, we expanded the literature review to
cover foundational induction, cut-point evaluation, online trees, scalable and
approximate split finding, feature pruning, globally optimized trees, and recent
split-construction results. The main article now cites 22 references.

**Location.** Main manuscript, Section 1 and References.

### R1.9. Pseudocode

**Comment.** Add pseudocode to clarify the algorithm.

**Response.** The main algorithm now exposes the common randomized
feature-prefix and first-qualifying-feature framework; the supplement gives the
method-specific exploration score, exact selection scan, acceptance comparison,
tie behavior, and fallback. We also include exhaustive CART and ERT pseudocode
so the candidate-space differences are explicit. The pseudocode and terminology
were independently checked against the implementation, including the facts that
the prophet-style method makes two passes and that `S^2` fully scans each
exploration feature after sampled calibration.

**Location.** Main manuscript, Section 2.1 and Algorithm 1. Supplement,
Sections S1--S2 and Algorithms S1--S7.

## Response to Reviewer 2

Reviewer 2 requested fuller method descriptions, a cost analysis, dataset
context, and clearer figures; we address these revisions below.

### R2.1. Expanded strategy descriptions and reproducibility

**Comment.** Expand the description of the secretary-style strategies to improve
clarity and reproducibility.

**Response.** The revised Methods introduce common notation at one
node: `n` samples, `m` as the pre-scan feature budget after removing constants
already known from ancestor nodes, and `C_j` admissible adjacent-value boundaries
on feature `j`. Newly discovered constant features are skipped without reducing
the exploration target. We distinguish an exact gain, which
uses all node samples, from exhaustive enumeration of every candidate. We then
specify randomized feature order, exploration prefix, continuous-threshold
sampling, exact feature scans, stopping, fallback, and ties for every method.

The pseudocode and parameter table state the schedules exactly as implemented,
including the distinction between `1/log(N_dataset)` fixed over a tree and
`1/sqrt(n_node)` recomputed at each node. We also state the dense/no-missing/no-
monotonic-constraint scope and the 4,096-feature exhaustive fallback; the latter
was not triggered because the retained benchmark has at most 240 features.

To avoid overstating correspondence with classical theory, the revised text
uses "secretary-inspired," "rank-inspired," and "prophet-style" where the
implementation does not satisfy the original online model's independence,
random-order, or irrevocability assumptions.

**Location.** Main manuscript, Section 2.1. Supplement, Sections S1--S2 and
Algorithms S1--S7.

### R2.2. Computational-complexity analysis

**Comment.** The manuscript lacks a computational complexity analysis that
characterizes the differences among the approaches.

**Response.** The revision adds a fixed-node cost analysis that
separates proxy-gain evaluations from sorting, min/max passes, partitioning,
random-number generation, sampled-threshold sorting, calibration fitting,
replay, memory traffic, and final split application. Letting `C_j` be the number
of candidate boundaries and `C_all` their sum, exhaustive CART performs at most
`C_all` gain evaluations plus feature sorting and scanning. ERT evaluates at
most one sampled threshold per selected feature. The secretary-style costs are
expressed in terms of exploration features, visited selection features, sampled
draw counts, and exact boundaries scanned before acceptance.

The analysis identifies both upper bounds and important exceptions. At a fixed
node, `S`, `S_all`, and the blockwise rank-inspired heuristic do not exceed the
exhaustive exact-gain count absent fallback. `S^2` can exceed it because it adds
sampled gains before a full exact exploration-feature scan. The archived
`S_par` ablation can add sampled gains, a fitted quantile, and replay. The
prophet-style rule has the safe bound `m + C_all`: it first performs one sampled
gain evaluation per feature and then replays exact scans while skipping each
sampled partition position.
Different selected splits can also create different descendant topologies, so a
fixed-node ordering need not hold for whole-tree effort.

This analysis explains why the effort metric is a useful but incomplete search
proxy and why it cannot be treated as a wall-clock complexity model. CART and ERT
effort is reconstructed for successful internal-node searches, whereas direct
early-stop counters include failed calls that produce leaves. We therefore keep
the conservative total-tree diagnostic and remove the non-comparable per-call
normalization from Figure S14.

**Location.** Main manuscript, Section 2.2. Supplement, Section S3 and Table S1.

### R2.3. Dataset characteristics and effective scenarios

**Comment.** Summarize instances, features, and other relevant dataset
properties; use them to identify where each strategy is effective and consider
this in the Conclusions.

**Response.** Table 1 now reports the quartiles and ranges listed in
response E8, and the supplementary inventory provides `n`, `p`, `n*p`, `p/n`,
class count, class imbalance, regression target variability, and archive
completeness for every dataset. This allows the displayed regimes to be audited
against the actual benchmark support.

The revised Results combine these descriptors with the tolerance-constrained
regime maps. We report only benchmark-supported patterns: under their respective
loss endpoints, retained classification entries admit incomplete search more
often, regression more often retains exhaustive CART under strict margins, and
loosening the margin makes aggressive ERT and stopping rules eligible on more
entries. We explicitly state that the nearest-neighbor background is descriptive
and that the corpus does not establish causal or universal boundaries. The
Conclusion recommends reporting future ensemble results by task and entry regime
rather than only as one pooled average.

**Location.** Main manuscript, Table 1, Sections 4.2 and 5, and Figure 3.
Supplement, Section S4 and Figure S8.

### R2.4. Figure quality and color-label readability

**Comment.** Improve figure quality because the text identifying the color
coding is difficult to read.

**Response.** All figures were regenerated at final physical size,
with vector main figures, supplementary composites exported at 600 dpi, larger
post-scaling typography and legends, clearer method-family colors, redundant
marker shapes, stronger outlines around observed data, and lower-opacity
background elements. The reliability figure uses the revision-designated
representatives; the Pareto and regime maps retain schedule variants with
organized legends. We checked the compiled PDFs at 100% display size and
corrected overlaps, clipping, and legend placement. Figure 3 received the more
extensive redesign described in responses E4 and R1.4.

**Location.** Main manuscript, Figures 1--3. Supplement, Figures S1--S14.

## Response to Reviewer 3

Reviewer 3 requested a broader reference section; we expanded it as described
below.

### R3.1. Expanded references

**Comment.** The final reference section is too sparse and should be expanded.

**Response.** The revised related-work discussion and bibliography now
cover foundational and modern split construction, scalable and approximate
tree induction, streaming and dynamic trees, randomized trees and ensembles,
global tree optimization, statistical comparison across datasets, and the
secretary/prophet literature directly motivating the proposed rules. As detailed
in response E6, each new reference supports a nearby claim and its metadata was
verified against a primary record. The revised main bibliography contains 22
cited entries.

**Location.** Main manuscript, Sections 1 and 5, and References.

## Additional implementation issue identified during revision

The revision audit identified two defects in the archived `S_par`
implementation that affect interpretation. The regression implementation fits its working
law to a proxy containing a node-dependent constant, making the fitted decision
target-translation dependent. It also showed that the classification normal-
quantile helper reverses the sign convention, so nominal upper quantiles are
mapped to lower-tail values. These are substantive implementation defects.

Accordingly, `S_par` has been removed from the main method comparison,
confirmatory method family, recommendations, and theoretical-validation claims.
Its old results appear only in a separately labeled supplementary archival panel
that states both defects and that the results do not evaluate the intended
calibration. They are not used in inferential tests or conclusions. We did not
silently correct the method while retaining its old benchmark values. A corrected
parametric calibration would require a new dedicated benchmark and is outside
the evidence claimed in this revision.

**Location.** Main manuscript, Section 4.1, paragraph following Table 2.
Supplement, boxed archived-ablation disclosure, Algorithm S7, and Figure S2.

Before submission, we checked the numerical values, references, algorithms, and
figures against the archived analysis and compiled manuscripts. The author will
perform the final proofread of all submission files.
