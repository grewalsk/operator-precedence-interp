# Changelog 4 (figures and tables to publication quality)

- Figure 2: fixed panel-c colors to the base=blue / instruct=orange convention; reshaped to a
  short-wide 3-panel layout; all bars carry 95% CIs.
- Enforced the 4-page body limit: folded the conclusion into \S6, compacted Table 1 (dropped the
  delta and few-shot rows into prose/caption), removed the redundant standalone
  format-brittleness paragraph (its content folded into the operation-specific paragraph),
  tightened related work, setup, dormant, and generality, and shrank both figures.
- Confirmed the appendix full-battery table carries Wilson CIs on all conditions and the
  magnitude bins carry positive paired-bootstrap CIs.
- Verified: latexmk compiles with 0 undefined references, 11 citations, 0 em dashes, 4-page body.
