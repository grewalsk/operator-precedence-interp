# Sweep 3 review — adversarial reviewer panel

Theme: simulate the five hardest MathNLP attacks and answer each in-text. Executed as an
8-agent hostile-reviewer workflow (one reviewer per attack + a data-verification auditor + a
MathNLP-fit reviewer + an area-chair synthesis). Raw output: 7 reviewers, 91-line ranked defect
list. Panel verdict: **weak-reject, cleanly recoverable to weak-accept**. All blocking and the
load-bearing major defects were applied; see changelog_3.md.

## The five attacks and where each is now answered in-text
- (i) Novelty vs Williamson 2025 -> \S2 (Related), rewritten to name the domain gap (symbolic vs
  NL word problems), the DLT axis, and the test-the-conjecture framing.
- (ii) "Just prompt-format sensitivity" -> \S4, the operation-specific paragraph now also carries
  the spacing-invariance rebuttal (C8 survives / C7 collapses without spaces) and position
  dependence, so three facts rule out generic format brittleness.
- (iii) "No causal result / uninterpretable null" -> \S5, positive control re-based on the
  operand route (recovery 0.50) and softened (different readout+operation); emit-P' disclosed as
  a saturating criterion; the discriminative null is the logprob CI; Figure 2c makes the
  dissociation visible.
- (iv) "Few-shot recovery = no real failure" -> \S4, recovery demoted to a control, framed as
  elicitation fragility, exemplars flagged fixed-seed; the contribution rests on the zero-shot
  regime.
- (v) "Selectivity is a band artifact" -> \S5, the pre-registered-layer robustness is now backed
  by a committed file (prereg_gap_*.json, staged), and band (2,99)'s sub-floor status is
  disclosed (C4 = 0.71/0.73), with the headline tied to the band-independent causal null.

## Blocking defects fixed (guardrail #1 / honesty)
1. Deleted the "length-matched dilution control" claim (no committed source) from the decode
   figure and its caption.
2. Staged prereg_gap_*.json (real, was committed but unstaged) so the pre-registered-layer claim
   is traceable.
3. Disclosed band (2,99) sub-floor parts accuracy.
4. Softened the positive-control claim (different metric + operation than the null).

## Exit gate: MET
All five attacks answered in the body text (not only in this file); no surviving guardrail-#1
violation.
