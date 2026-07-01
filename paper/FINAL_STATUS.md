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

## Round 2 — external mech-interp review (applied)

A second independent read-only reviewer (theory-vs-data audit + MathNLP venue fit) was run. All
writing/hygiene findings are now applied:
- **Desk-reject risk fixed:** the body genuinely spilled ~2 lines onto page 5 (the earlier check
  matched a mid-conclusion phrase, not the last line). The full body, through the conclusion, now
  ends on page 4; page 5 begins with the References heading (verified with column-aware extraction).
- **§5 causal-null sentence rewritten** (the most attackable): the first-token null (vs
  random/shuffled, non-discriminative) and the full-product dose-response are now correctly
  separated and attributed. The same-axis dose-response is promoted from an anonymous parenthetical
  to the discriminative control: C1 inject flat at every dose, C4 inject rises +0.08 -> +0.46 (CIs
  exclude 0) yet never flips. This narrows the positive-control residual for base using committed
  data.
- **Swap surface named** (C4 `=`) in §5 and the Fig 2b caption; the swap magnitude corrected to
  "+0.59 at L19 where we decompose it, peak +0.66 at L23."
- **"pre-registered" -> "C1-independent criterion (C4 decode peak)"** (§5 + contribution bullet);
  "85% of the variance" -> "85% of the swap's answer-effect" (effect share, with the 20% geometric
  magnitude share noted); Makelov "dormant" given a one-clause strong-sense definition.
- **Honesty/precision:** band-2 product reading hedged to "operand-nonlinear component, weak
  product signal"; B-at-`=` decode scoped to 0.96-0.97; nesting control moved to within-instruct
  (0.25 from 0.99); wrapper-heatmap position-dependence scoped to base; n_boot=300 for gap CIs
  noted in §3.
- **Repo hygiene:** DATA.md now flags the stale `inject_C1_L4=+1.99` cross-surface-bug artifact so
  a repo-reading reviewer is not misled; stale `sections/07_conclusion.tex` deleted.

## Round-2 re-check defects (S1-S6): disposition

- S1 (output vs emitted answer) — APPLIED: abstract, intro bullet 3, and Fig 1 now say "emitted
  answer" / "answer", consistent with the C4 log-probability movement in §5.
- S2 (circular §5 grammar) — APPLIED: "the swap moves only ~20% of its magnitude along it."
- S3 (stale appendix caption) — APPLIED: figA_steering now points to "the dose-response (§5) and
  Figure 2b."
- S4 (repo-only) — APPLIED: DATA.md explains the `WEAK_INSTRUMENT` label's flip-level threshold so
  a repo reader does not quote it as "instrument dead."
- S5 (availability statement, removed by the page cut) — DEFERRED to camera-ready. An anonymous
  MathNLP submission cannot carry the (deanonymizing) repo link anyway; restore the "code and data
  released" line at camera-ready when the page count relaxes.
- S6 (Fig 1 wording) — APPLIED alongside S1.

## GPU runs to do later (deferred; not run here)

In priority order (from the reviewer's "run before submission"):
1. **Same-metric answer-site positive control (highest value, ~+10pp odds).** Re-score the
   operand-position patch (and/or a donor swap at an answer-carrying site) in the SAME
   emit-P'/full-product-logprob units as the C1 null, so the null and its positive control sit in
   identical units on Fig 2c. Reuses the 82r machinery. This converts residual 1 from "disclosed
   caveat" to "closed," and rescues instruct (whose dose reference is weak). One GPU pass.
2. **One second wrapper family (~+5pp odds).** Run `( 1 * B ) * C` (base mul 0.60 behaviorally)
   through the mechanistic pipeline (selectivity gap + dose null at `=`), to show the dormant
   answer site is not an additive-identity idiosyncrasy. Cheap; pre-empts "is this one weird
   prompt?"
3. **Optional stretch:** re-run the fixed 82s static Makelov certification (all-answer-token
   basis) as an appendix strengthener; commit base A1/A2/D1 to drop the instruct-only caveat
   (behavioral, cheap); a richer nonlinear baseline (B^2, C^2, B+C features) for the band-2 gap to
   sharpen "product" vs "any nonlinearity."

After run 1, update Fig 2c and the §5 positive-control sentence; after run 2, add a one-row
robustness panel. Everything else in the paper is submission-ready.

## Artifacts
- `paper.pdf` (compiled), `main.tex` + `sections/*.tex`, `refs.bib`, `build.sh`
- `figs/make_*.py` (reproducible from `paper/data/*`, itself staged from committed `origin/main`)
- `DATA.md` (single source of truth), `CLAIM.md`, `OPEN_ITEMS.md`
- `reviews/sweep_{1..5}.md` + `reviews/changelog_{1..5}.md`
