# FINAL_STATUS.md

Target venue: **BlackboxNLP 2026** (EMNLP 2026), archival, up to 8 pages, deadline 2026-07-17 AoE.
Working branch: `wo8-blackboxnlp` (off `wo6-operand-localization`). Prior rounds (MathNLP 4-page
short) are summarized below the Round-3 ledger.

## Round 3 — BlackboxNLP conversion (this session)

Compiles clean: numbered body ends on **page 5** (target <=6), unnumbered Limitations on p6,
References p6-7, appendix p7-9; total 9 pp. 0 em dashes, 0 "pre-registered", 0 unresolved
citations, **24 references** (all Anthology-fetched or arXiv-id title-verified).

### Landed
- **A1 intro reframe** — opens on "when does decodability license a computational claim; arithmetic
  as testbed"; Williamson demoted to one motivating cite in thread 3.
- **A2 related work rebuilt into 3 threads** (probing validity / arithmetic representations /
  elicitation), +13 verified refs, with the Pimentel delta (BC decodability is inherited from
  (B,C); our baseline answers exactly that) and the Nikankin delta (operand echo, assembled at
  readout) stated explicitly.
- **B1 Proposition 1 (headroom bound)** in body, 3-line proof in Appendix F, instantiated
  (band1 headroom 0.032 vacuous; band2 0.135, gap inside).
- **B2 causal-abstraction paragraph** (§5): the swap is an interchange intervention; interchange
  accuracy 0; the "computed-at-= " hypothesis fails as a causal-abstraction claim (Geiger cite).
- **B3 hypothesis-evidence matrix** (Table 2): H1 stored (rejected) / H2 assembled-at-readout
  (supported) / H3 sparse circuit (no locus), every cell traced to a committed file.
- **B4 falsifiable predictions** from H2 (attention-ablation kills C4; few-shot acts at operands),
  labeled as predictions.
- **C1 emit_true** — after the C4 swap the model still emits its own correct answer 0.90-0.95
  (L19-L31): one clause in §5 + Appendix Table (app:emit_true). Strongest free sentence.
- **C2 dose-response** promoted to Figure 2 panel (d) (C1 flat vs C4 monotone, emit-P'=0).
- **C3** verbatim prompt strings (app:prompts). **C4** C5 clause in the battery caption
  (parenthesization actively hurts). **C5** decodability-audit protocol box (app:audit).
  **C6** baseline-scope note (app:repro). **C7** bf16 + A100 + versions in app:repro.
- **E1** restructure: Generality / Conclusion / unnumbered Limitations (moved + expanded:
  single family, instruct dose weak, band-2 sub-floor, distributed locus, same-metric control +
  second wrapper as next experiments) / references / appendix-after-references. Anonymized
  availability statement (A3) with an anonymous.4open.science placeholder URL.

### Open (next session)
- **D1/D2 GPU notebooks** — being built next (`cells/84*.py` -> assembled `.ipynb` at repo root);
  human runs them, commits JSON to origin/main root, then integration per the WO branch protocol.
- **E2 anonymized mirror** — build the scrubbed, history-free mirror (drop REVIEW/RECHECK/
  FINAL_STATUS, scrub DATA.md, grep for author strings) and point the availability URL at it.
- **E3 full pre-submission audit** — after D integration: theory-vs-data ledger over the new
  claims, column-aware boundary re-check, metadata grep, figure-regeneration byte-check.

## Ledger status
Every number in the tex traces to a committed file at `origin/main` root, mapped in DATA.md;
the new surfaced numbers (emit_true 0.90-0.95, headroom 0.032/0.135) were added to DATA.md this
session. No new GPU numbers entered the tex (none exist yet).

## Next gate
Build D1 (`wo8_d1_same_metric_control.ipynb`) to the "runnable top-to-bottom, SMOKE-gated,
pairs_sha=db4da1 asserted" spec; hand to the human; integrate on PASS/FAIL per the protocol.

---

## Round 2 — external mech-interp review (MathNLP 4-page, prior)
(Applied; see git history on `wo6-operand-localization`.) Fixed the page-4 desk-reject risk,
rewrote the §5 causal-null sentence (metric split + dose-response promotion), named the swap
surface (C4), replaced "pre-registered", defined Makelov "dormant" strongly, plus S1-S6 polish.

## Round 1 — 5-sweep build (MathNLP 4-page, prior)
Skeleton+verify -> full calibrated draft -> 8-agent hostile-reviewer workflow -> figures/format ->
top-paper calibration. Caught + corrected: the swap does NOT flip (positive control re-based on the
operand route); band2 gap CI excludes 0; a fabricated "dilution control" removed.
