# Information Sciences transfer notes

## How serious the reviews were

Three points were publication-critical because they affected the scientific
interpretation rather than presentation alone:

1. The missing exact-search random-feature baseline left open whether adaptive
   stopping added value beyond examining fewer features.
2. The single-tree scope left the intended ensemble use insufficiently tested.
3. The unavailable raw outputs and incomplete run provenance weakened
   reproducibility.

The requests for full-tree complexity, a quantitative runtime diagnosis,
stronger reliability criteria, minority-class metrics, and closer related-work
coverage were major but could be answered without changing the splitter. The
remaining comments about figure size, the 54 regression families, conclusion
organization, and terminology were editorial.

## What was added

- A counterbalanced 20-seed benchmark compares exhaustive CART,
  \(S_{\mathrm{all}}(1/e)\), exact search over fixed random feature subsets,
  and ERT on all 113 regression and 122 classification entries.
- A focused 25-tree bagging validation covers 20 datasets per task, five seeds,
  and five folds.
- Native splitter counters quantify gain evaluations, feature sorts, min/max
  passes, and partition passes in a focused runtime diagnostic.
- Reliability is now assessed at a practically meaningful 10% minimum time
  saving and a one-percentage-point loss tolerance. Macro-F1, worst-class
  recall, tail-loss, and classification-family checks complement weighted F1.
- The supplement states node-level and complete-tree complexity, explains the
  113-to-54 regression-family mapping, and corrects the ERT gain-landscape
  weighting.
- A dedicated related-work section discusses exact scalable tree induction,
  adaptive split search including MABSplit, approximation methods, and recent
  2025--2026 work.
- Versioned archives now contain the historical benchmark runs, the new control
  and bagging outputs, inference results, corpus audit, operation diagnostic,
  and gain-landscape summaries with checksums and run metadata.

## What changed in the interpretation

The original full-benchmark measurements did not change. New experiments make
the conclusion narrower and better supported:

- \(S_{\mathrm{all}}(1/e)\) is presented as a conservative intermediate
  Pareto option, not as a generally superior splitter.
- Exact random feature subsets usually save more time, but incur larger mean
  predictive losses. The tested curve places adaptive stopping at a distinct
  low-loss operating point, but no fixed subset exactly matches its runtime, so
  the experiment does not identify a causal benefit of adaptivity at equal cost.
- Under the stricter 10%-saving reliability criterion, no prespecified method
  supports a majority-of-datasets claim after Holm correction. The manuscript
  no longer makes that broad reliability claim.
- The focused bagging experiment preserves the low-loss classification
  trade-off on average, but it is not evidence about large forests or scaling.
- Regression remains the unfavorable regime for secretary-style stopping.

## Files for the transfer

- `main.tex` and `main.pdf`: neutral Elsevier two-column manuscript formatted
  for Information Sciences.
- `supp.tex` and `supp.pdf`: algorithms, complexity details, statistical
  analyses, controls, diagnostics, and robustness figures.
- `references.bib`, `elsarticle.cls`, figure directories, and `TABLES/`: source
  dependencies.
- `SUPPLEMENTARY_DATA/*.tar.gz` and `SUPPLEMENTARY_DATA/README.md`: raw outputs,
  metadata, and checksums.
- `highlights.tex` and `highlights.pdf`: five submission highlights.

The earlier PRL response letter is historical material and should not be sent
with a fresh Information Sciences submission unless the receiving editor asks
for the previous review history.

## Final author check before upload

1. Read the abstract, contributions, fixed-budget control, discussion, and
   conclusion together and confirm that the conservative framing matches the
   intended claim.
2. Check the numerical statements in Tables 2 and 3 and Figures 1 and 2 against
   the supplementary control tables.
3. Confirm the author details, funding, competing-interest, CRediT, data/code
   availability, and generative-AI disclosure.
4. Upload the editable LaTeX sources, both PDFs, all figure and table inputs,
   the bibliography, highlights, and the seven supplementary-data archives.
5. Use a new cover letter for Information Sciences. Do not reuse the PRL
   point-by-point response as a submission document.
