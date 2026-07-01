# Sweep 4 review — figures and tables to publication quality

Theme: finalize figures, ensure every table has CIs, enforce ACL formatting and the 4-page body
limit, confirm the PDF compiles.

## Critic pass
- [PASS] Figure 1 (hero) is a clean conceptual TikZ schematic reconciled to the evidence: the
  swap does NOT flip (no false "swap flips" arrow); computation is at the operand positions.
- [PASS] Figure 2 has three panels, all error-barred (95% CI), colorblind-safe (Wong palette,
  base=blue / instruct=orange consistently after the panel-c fix), vector PDF (Type-42 fonts).
  Panel (c) now surfaces the load-bearing positive control.
- [PASS] Appendix figures (wrapper heatmap, by-layer decodability, steering nulls) regenerate
  from committed data; the decode figure carries no mechanistic claim.
- [PASS] Every table carries CIs: Table 1 has the C4-C1 delta CI; the appendix full battery has
  Wilson CIs on all twelve conditions; magnitude bins carry positive paired-bootstrap CIs.
- [FIX-APPLIED] 4-page body: the review additions pushed the body to 5 pages. Reclaimed by
  folding the conclusion into \S6 (removing a section heading), compacting Table 1, tightening
  \S2/\S3/\S5/\S6, shrinking both figures, and removing the redundant standalone
  format-brittleness paragraph. Conclusion now ends on page 4.
- [FIX-APPLIED] Reproducible build: build.sh regenerates every figure from paper/data/*.
- [PASS] No em dashes in the compiled PDF (verified via pdftotext U+2014 scan == 0).

## Exit gate: MET
Figures self-contained and colorblind-safe; PDF compiles; body within 4 pages; total 6 pages
(4 body + references + appendix).
