# Sweep 2 review — full draft with calibrated claims

Theme: write all prose; every claim tagged by evidence level with a CI; thematic related work
with the positioning sentences; explicit research questions. Exit gate: no claim exceeds its
evidence level; every quantitative claim has a CI or is marked pending.

## Critic pass (calibration)
- [PASS] Evidence discipline holds. Behavioral claims (blind spot, operation-specificity, depth,
  magnitude, few-shot) carry Wilson or paired-bootstrap CIs. Decodability claims are labeled as
  correlational and are never used to assert use. Causal claims (steering null, swap, causal
  share, operand recovery) are separated and each has a positive control or a stated null.
- [PASS] The selectivity result is stated with the honest two-band structure: gap ~0 at (20,49)
  with zero headroom, gap +0.06 (CI excludes 0) at (2,99) with headroom, robust to a
  pre-registered layer. Called "weakly represented", never "represented" unqualified.
- [PASS] Positive control appears beside the steering null (operand-route recovery 0.50) and the
  swap null (emit-P'=0). No one-word verdict labels in the prose.
- [PASS] Related work is thematic (three themes) with the two required positioning sentences vs
  Williamson 2025 and Matsumoto 2022.
- [FIX-APPLIED] Length: first full draft was a 5-page body. Trimmed related/setup/blindspot/
  dormant/generality, collapsed the four blind-spot controls into two paragraphs, shrank both
  figures and captions -> body now 4 pages (conclusion on page 4).
- [FIX-APPLIED] Guardrail #8: table "no data" placeholders were `---` which render as em dashes;
  replaced with `n/a`. PDF now contains zero em-dash glyphs (verified with pdftotext).

## Exit gate: MET
No claim exceeds its evidence level; every quantitative claim carries a CI; nothing pending in
the body; 4-page body; no em dashes.
