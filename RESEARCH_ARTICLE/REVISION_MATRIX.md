# PRLETTERS-D-26-00639 Major-Revision Matrix

Revision deadline: 2026-08-21

This matrix is the controlling checklist for the revision. A row is closed only
when the manuscript change, generated evidence, response-letter entry, and an
independent verification all agree.

## Editorial Synthesis

| ID | Request | Planned evidence or change | Closure test | Status |
|---|---|---|---|---|
| E1 | Clarify the contribution. | State the same precise methodological and empirical contributions in the abstract, introduction, conclusion, and response letter; compare directly with prior split-search work. | A reader can identify the novelty without inferring it from the results for one method. | Closed |
| E2 | Add full inferential statistical analysis and confidence intervals for medians and centroids. | Use datasets as experimental units; preserve within-dataset run pairing; report effect sizes, 95% paired hierarchical bootstrap intervals, and multiplicity-aware tests for revision-designated representatives. | Analysis code passed 22 direct tests; every main-text interval maps to the 10,000-replicate production export. | Closed |
| E3 | Explain why the study uses one tree rather than a full reference tree. | State that every estimator is a complete recursively grown tree, not a stump; document the common growth settings; explain why single-tree analysis isolates split-search effects before ensemble aggregation; identify matched bagging as the principal future study rather than adding a post hoc ensemble experiment. | Main text and response cover both plausible readings of the comment without overstating ensemble implications. | Closed |
| E4 | Improve figure readability and explanations. | Redesign Figure 3, enlarge final-size typography, use accessible colors, prefer vector output, and explain the tolerance-constrained winner rule directly. Audit all main and supplementary figures at final PDF size. | The compiled main paper and all 23 supplement pages were rendered and visually checked; no clipping or unreadable labels remain. Main Figures 2 and 3 are no taller than 4.35 inches and satisfy the half-page limit. | Closed |
| E5 | Clarify the apparent reference to a previous publication. | Remove "earlier implementations" and other development-history wording. State in the response that this referred to an internal implementation iteration, not a previous publication. | Abstract is self-contained and contains no unexplained comparison with prior work. | Closed |
| E6 | Expand foundational and current literature. | Add only verified, directly relevant primary sources on CART, randomized and online trees, efficient/approximate split finding, optimal stopping, and multi-dataset statistical evaluation. | The main paper cites 22 directly relevant sources, each tied to a nearby claim. | Closed |
| E7 | Add pseudocode and expand the method descriptions. | Add one unified main-paper algorithm and detailed supplement algorithms, including candidate order, exploration budget, threshold, acceptance, fallback, ties, and randomization. | Pseudocode and complexity bounds were independently checked against the Cython control flow and corrected where needed. | Closed |
| E8 | Summarize dataset characteristics and identify favorable scenarios. | Add Q1, median, Q3, and range for n and p; add relevant classification descriptors and a machine-readable per-dataset inventory; connect regime findings cautiously to observed dataset properties. | Generated Table 1 and the machine-readable inventory agree with the retained corpus. | Closed |

## Reviewer 1

| ID | Request | Planned evidence or change | Closure test | Status |
|---|---|---|---|---|
| R1.1 | Make the contribution and publishable value clear despite modest wall-clock gains. | Frame the work as a controlled test of a plausible split-search replacement that identifies viable and non-viable rules, task-dependent regimes, and the gap between gain-evaluation effort and runtime. | Claims are supported by uncertainty-aware results and do not reduce the contribution to S_all alone. | Closed |
| R1.2 | Add confidence intervals for medians and centroids. | Report dataset-conditional median intervals in supplement/machine-readable output and dataset-resampled intervals for cross-dataset medians and both centroid coordinates. | Main tables and figures use the same point estimands and intervals. | Closed |
| R1.3 | Explain "Only one tree? Why not a full reference tree?" | Address complete-tree versus stump and single-tree versus ensemble interpretations explicitly. | Response is unambiguous and the manuscript states the scope and limitation. | Closed |
| R1.4 | Make Figure 3 readable and explain it. | Simplify the map and legend; increase typography; explain that the color is the fastest method among those satisfying the displayed loss tolerance. | Final-size PDF review passed. | Closed |
| R1.5 | Reformulate the abstract if it alludes to previous work. | Remove unexplained historical comparison; cite actual prior work only in the body. | Abstract stands alone and all novelty claims are traceable. | Closed |
| R1.6 | Improve Table 1 with Q1, Q3, and range. | Generate quartile-and-range dataset descriptors. | Recomputed values match source metadata. | Closed |
| R1.7 | Add language, package, and hardware configuration. | Report Python 3.10.18, NumPy 1.26.4, scikit-learn 1.7.2, treeple 0.10.3, macOS 15.2/ARM64, Apple M3 8-core CPU, 16 GB RAM, CV/timing scope, seeds, and source revision. | Supported details are reported; unavailable invocation and binary provenance are disclosed rather than invented. | Closed |
| R1.8 | Expand the bibliography. | Addressed jointly with E6. | Relevance and bibliographic accuracy reviewed. | Closed |
| R1.9 | Add pseudocode. | Addressed jointly with E7. | Code-review sign-off confirms the described behavior. | Closed |

## Reviewer 2

| ID | Request | Planned evidence or change | Closure test | Status |
|---|---|---|---|---|
| R2.1 | Expand strategy descriptions for clarity and reproducibility. | Add unified notation, exact algorithms, parameter table, random-order semantics, fallback, and ties. | The main and supplementary algorithms now define every validated method and the separately screened corrected parametric rule. | Closed |
| R2.2 | Add computational-complexity analysis. | Separate sorting/scanning, random threshold generation, exact gain evaluations, expected stopping, worst-case work, and implementation overhead. | Complexity statements use defined variables and match actual control flow. | Closed |
| R2.3 | Summarize dataset characteristics and relate them to effectiveness. | Addressed jointly with E8; quantify only benchmark-supported scenario associations. | Conclusions carry benchmark and task-specific endpoint qualifiers. | Closed |
| R2.4 | Improve figure quality and color-label readability. | Addressed jointly with E4. | Main figures are vector; supplementary figures are 600 dpi; final PDF review passed. | Closed |

## Reviewer 3

| ID | Request | Planned evidence or change | Closure test | Status |
|---|---|---|---|---|
| R3.1 | Expand the sparse references. | Addressed jointly with E6 while preserving the focused scope. | Reference list is broader and each source is relevant to a stated claim. | Closed |

## Statistical Non-Negotiables

- The outer experimental unit is the dataset, not a fold, seed, fitted tree, or CSV row.
- The 100 seeded repetitions use fixed folds and quantify conditional randomization and timing variability.
- All method-versus-exhaustive quantities remain paired by dataset and valid run block.
- Regression dataset `564_fried` lacks run 4; that block is excluded consistently or rerun, never imputed.
- Gini and entropy results are analyzed separately unless their dependence is explicitly preserved.
- Classification F1 loss is reported in percentage points, not as a relative percentage.
- The corrected S_par implementation was rerun separately for all 20 configurations and is reported only through predictive loss and gain-evaluation effort. It remains excluded from confirmatory inference and recommendations because no configuration entered the full effort-loss Pareto frontier.
- Effect sizes and intervals lead; p-values are secondary and multiplicity corrected.
- Inference is conditional on resampling this benchmark corpus and is not presented as population-wide coverage over all tabular problems.

## Chair Sign-Off Gates

1. Freeze and validate the benchmark inputs and estimands.
2. Approve and independently test the inferential implementation.
3. Approve implementation-faithful pseudocode and complexity claims.
4. Approve dataset and reproducibility metadata.
5. Approve every new citation and contribution claim.
6. Regenerate and numerically cross-check all tables and figures.
7. Compile and visually inspect the main paper, supplement, and response letter.
8. Re-read the complete decision letter and close every matrix row with page/line evidence.

## Final Author Gates

- Proofread `main.pdf`, `supp.pdf`, and `response_to_reviewers.pdf` before upload.
- Verify the funding-role, competing-interest, CRediT, and AI-use declarations as personal statements.
- Confirm that `mathias.valla@gmail.com` is used consistently in both the manuscript and Editorial Manager, as chosen by the author.
- Generate the official Elsevier declarations Word file from `DECLARATIONS_DRAFT.md`; the Markdown file is a checked source, not a substitute for the journal tool.
- Build and inspect Editorial Manager's merged submission PDF before approval.
