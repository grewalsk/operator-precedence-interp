# Changelog 2 (full draft with calibrated claims)

- Wrote all sections: abstract (~180 words), intro + 4 contributions + TikZ hero, thematic
  related work with the two positioning sentences, setup + 3 RQs, blind-spot hook (Table 1 with
  CIs + four controls), the dormant-subspace centerpiece (decode -> selectivity -> steering ->
  certification, Figure 2), generality + format confound + limitations, 3-sentence conclusion,
  appendix (full battery with Wilson CIs, magnitude bins, three appendix figures, cross-model).
- Tagged every claim by evidence level; attached CIs (Wilson for accuracies computed
  programmatically; paired bootstrap for deltas from confidence_intervals.json; item-bootstrap
  for the gap).
- Enforced the 4-page ACL body limit through targeted prose and figure trims (resolves the
  Sweep-2 length finding).
- Replaced em-dash table placeholders with `n/a` (resolves the guardrail-#8 finding).
- Confirmed 0 undefined references, 11 citations render, PDF compiles at 4-page body.
