# Changelog 3 (adversarial reviewer panel)

Applied from the 8-agent hostile-review synthesis (ranked defects). Each entry resolved a named
critic point by an actual text/figure change.

Blocking:
- Removed the unsupported context-dilution claim (figs/make_figA_decode.py + appendix caption).
- Staged prereg_gap_{base,instruct}.json + fewshot_control.csv into paper/data/ (real committed
  files the reviewers could not find); the pre-registered-layer clause is now traceable.
- \S5: disclosed band (2,99) is sub-floor (C4 = 0.71 base / 0.73 instruct), read as a stress
  test, headline tied to the causal null.
- \S5 + abstract + intro: softened the positive control to "operand positions, a clean-answer
  recovery metric ... different readout and operation," instead of "the same patching machinery."

Major:
- \S5: disclosed emit-P' saturates (stays 0 even for a CI-excluding-0 reference effect); made the
  logprob CI the discriminative null.
- \S5: reframed the 14/85 decomposition as "where the swap's weak sub-flip variance lies," not a
  "causal handle"; dropped the word handle.
- \S5: dropped the non-significant instruct 7% (w-only CI includes 0); decomposition reported for
  base; reconciled swap magnitude to +0.59 at L19 (removed the confusing early-layer -0.20).
- \S4: added the spacing-invariance rebuttal (C8/C7) and folded the format-brittleness argument
  into the operation-specific paragraph; added few-shot Wilson CIs and the fixed-seed caveat.
- \S2: sharpened the Williamson delta (domain, DLT, test-the-conjecture) and added the Reusch
  delta.
- Figure 2: added panel (c) making the positive control visible (operand recovery 0.50 vs
  answer-site swap emit-P' = 0), a dead site not a dead probe.
- Title retargeted to the answer-site representation (not "arithmetic composition").
- Intro contribution 3 split into inertness + positive control, and the causal-share why.

Minor:
- 0.968 -> 0.967 (base decode R^2); ">=0.86 of R^2 is pure B,C" -> ">=0.94 reproduced by the
  linear baseline"; C0 0.85 -> 0.848; appendix C5 .543 -> .542; added C5 to DATA.md.
