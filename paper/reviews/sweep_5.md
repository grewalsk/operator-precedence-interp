# Sweep 5 review — top-paper calibration

Theme: match a strong MathNLP short paper (Williamson et al. 2025, 2025.mathnlp-main.1); sharpen
abstract/intro/contributions; confirm the argument is reconstructable from Fig 1, Fig 2, and the
two tables; verify every reference resolves; final overclaim scan against the guardrails.

## Calibration against Williamson et al. 2025
- Structure matched: a sharp single claim, a behavioral hook demoted to motivation, a controlled
  mechanistic core, an honest generality section with a named confound, tight limitations.
- Our differentiator is now explicit (\S1, \S2): where Williamson is behavioral over NL word
  problems and conjectures a surface/representation coupling, we test that conjecture on a
  token-matched symbolic instance and show the obvious version (a computed-but-unused product)
  fails; the signature is a dormant, operand-explained subspace.

## Reference verification (guardrail #7)
All 11 entries resolve. Venue papers pulled verbatim from ACL Anthology (Williamson 2025,
Matsumoto 2022, Reusch 2022, Stolfo 2023, Hewitt-Liang 2019, Belinkov 2022). The five arXiv/ICLR
entries were HTTP-checked and title-matched:
2311.17030 (Makelov), 2410.21272 (Nikankin), 2309.16042 (Zhang-Nanda), 2404.15255 (Heimersheim),
2407.21783 (Llama 3) all return 200 with the expected title.

## Final overclaim scan (against Section 0 guardrails)
- No causal claim beyond evidence: the null is stated with a (softened, honest) positive control;
  the 14/85 decomposition is described as within-null variance, not a causal handle.
- No "swap flips" claim anywhere (the refuted assertion never entered the paper).
- No "pure illusion / zero product": the small band-(2,99) gap is reported and its sub-floor
  status disclosed; the product is called weakly represented.
- Cross-model framed as matched-format replication + a format confound, never a dissociation.
- No one-word verdict labels; no em dashes (PDF scan == 0); every quantitative claim in the body
  carries a CI or points to the appendix CI table.

## Argument reconstructable from the display items? YES
Fig 1 (the illusion), Fig 2a (operand-dominated), Fig 2b (decode direction not the causal
carrier), Fig 2c (dead site not dead probe), Table 1 (parts work, composition fails). A reader
can reconstruct the claim without the prose.

## Residual (see FINAL_STATUS.md / OPEN_ITEMS.md)
- The static Makelov dormant certification (82s) is all-null on main and is not relied on; the
  dormancy rests on the causal-share certification. Noted as future work.

## Exit gate: MET
Rubric all green or green-after-fix (see FINAL_STATUS.md); no blocking defect remains.
