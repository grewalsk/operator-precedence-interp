# FINAL_STATUS.md

5 improvement sweeps complete. `paper.pdf` compiles: **4-page body** (all of \S1-\S6 incl. the
folded conclusion on pages 1-4), 6 pages total (4 body + references + appendix), 11 references
resolving, 0 undefined citations, 0 em dashes in the compiled PDF.

## Rubric scorecard (post-fix)

| # | Rubric item | Grade | Note |
|---|---|---|---|
| 1 | Single claim survives hostile review | GREEN | anti-fishing defenses now backed by committed files; headline on the band-independent causal null |
| 2 | Every number traceable to a committed file | GREEN | fabricated dilution claim removed; prereg_gap + fewshot_control staged; C5 added to DATA.md |
| 3 | Every quantitative claim has a CI | GREEN | few-shot Wilson CIs added; body claims carry CIs or point to the appendix CI table |
| 4 | No claim exceeds its evidence level | GREEN | 14/85 reframed as within-null variance, not a causal handle; D never asserted as C |
| 5 | Positive-control magnitude beside every null | GREEN* | operand-route recovery 0.50 is stated + visible (Fig 2c); *honestly flagged as a different readout+operation (see residual 1) |
| 6 | Cross-model = replication + confound | GREEN | matched-bare replication + explicit prompt-format confound; never a dissociation |
| 7 | Related work positions vs the 3 venue papers | GREEN | Williamson delta sharpened (test-the-conjecture); Matsumoto contrast; Reusch delta added |
| 8 | Fig 1 conceptual; Fig 2 error-barred | GREEN | Fig 1 reconciled (no false swap-flip); Fig 2 is 3 panels, all 95% CIs, colorblind-safe |
| 9 | Limitations honest | GREEN | distributed locus, instruct-only controls, small/sub-floor gap, single wrapper family |
| 10 | 4-page ACL body; compiles; refs resolve | GREEN | body ends on page 4; latexmk clean; all 11 refs HTTP/Anthology-verified |
| 11 | No em dashes | GREEN | PDF U+2014 scan == 0 |

## Residuals (in OPEN_ITEMS.md)
1. **Positive control is a different metric+operation.** The steering/swap null uses full-product
   emit-P'/logprob on a counterfactual donor swap; the positive control (operand-position
   recovery 0.50) uses clean-answer denoising recovery. It is now softened and disclosed as such,
   and the answer-site swap's own +0.59 logprob shift (with emit-P' = 0) is reported. A
   same-metric positive control (an intervention that moves the full-product output at an
   answer-carrying site) would fully close attack (iii). This needs a GPU run and is not in the
   committed data; it is not fabricated.
2. **Static Makelov certification (82s) is all-null on main** (first-token-basis bug; fix
   committed but not re-run). The paper does NOT rely on it; the dormancy rests on the
   causal-share certification (82y). Future strengthener.
3. **band (2,99) is behaviorally sub-floor** (C4 = 0.71/0.73). Disclosed in \S5; the headline
   rests on the band-independent causal null, not this band's gap.
4. **A1/A2/D1 (operation/depth controls) are instruct-only** in committed data; stated as such.

## Acceptance-risk read
**Borderline / weak-accept.** The panel scored the pre-fix draft weak-reject, recoverable to
weak-accept, and every blocking and load-bearing major defect was applied. The paper's strengths
are its control discipline, calibrated claims, and honesty (a refuted assumption was caught and
kept out; a fabricated control was removed). The residual risk is a hostile reviewer pressing on
residual 1 (the positive control's metric mismatch) and on the distributed, non-minimal
mechanism. Neither is fatal given the convergent nulls (steering-dead, swap-no-flip,
decode-direction carries <=14% of a sub-flip effect) and the honest framing, but they cap the
ceiling below a clear accept.

## Single highest-value next action
Run a **same-metric positive control** for the steering null: an intervention (e.g. a donor swap
or a readout-direction inject) that provably moves the **full-product** output at an
answer-carrying site, reported in the same emit-P'/logprob units as the C1 null and placed beside
it in Figure 2. This converts residual 1 from "disclosed caveat" to "closed," and is the single
change most likely to move the paper from weak-accept to accept.

## Artifacts
- `paper.pdf` (compiled), `main.tex` + `sections/*.tex`, `refs.bib`, `build.sh`
- `figs/make_*.py` (reproducible from `paper/data/*`, itself staged from committed `origin/main`)
- `DATA.md` (single source of truth), `CLAIM.md`, `OPEN_ITEMS.md`
- `reviews/sweep_{1..5}.md` + `reviews/changelog_{1..5}.md`
