# OPEN_ITEMS.md — gated / reconciled / pending items

Per guardrail #1 and Section 8: anything not backed by a committed file, or where the work
order's assumption is contradicted by committed data, is recorded here rather than fabricated.

## RESOLVED against committed data (work-order assumption corrected)

1. **[CORRECTED — load-bearing] The full-residual swap at `=` does NOT flip the output.**
   Work order §1/§2 assert "the site carries the answer (a full-residual swap flips it)" and name it
   "the primary positive control", but flag "VERIFY the number before relying on it." Verified:
   `causal_steering_lateswap_{base,instruct}.json` = PRODUCT_NOT_AT_EQUALS, best emit-P' = **0.0**;
   `causal_steering_fullproduct_*` swap emit-P' = 0.0, logprobΔ = -0.20 (base) / -0.001 (instruct).
   => We do NOT write "the swap flips." Reframed claim: the answer-site subspace is causally dormant AND
   even a full-residual swap does not move the output; the answer is re-derived diffusely from the operand
   positions. **New positive control** = operand-position patching (b_last/c_last recovery 0.50): the same
   intervention machinery DOES move the answer at the operand positions, so the answer-site null is a dead
   direction/site, not a dead instrument. (Guardrail #4 satisfied with a REAL, committed positive control.)

2. **[RESOLVED — the §1 Conditional] band2 gap CI is committed and EXCLUDES 0.**
   `band_robustness_{base,instruct}.json`: C1 `=` product gap base +0.068 CI [0.051, 0.086], instruct +0.060
   CI [0.043, 0.079]; pre-registered-layer re-check (`prereg_gap_*.json`) also excludes 0 and is robust across
   layer choices. Per §1: finding is "**weakly represented but causally dormant**; the steering null carries
   the paper." NOT "pure illusion / zero product." At band (20,49) the gap is ~0 (operand-explained, zero
   headroom); the small component only shows where there is headroom (band2). Both stated, with the nuance.

3. **[NOTE] WO#5 c4_ref counterfactual is a FAILED positive control** (Δ ~ -0.01, GT-first-token-logit metric
   is non-discriminative). Not cited as a working control. The full-product dose-response (82x) and the
   operand-route patching supersede it.

## PENDING (not blocking; noted for honesty)

4. **Static Makelov dormant certification (82s) is all-null on `main`** (`dormant_certification_*.json`:
   readout basis built from the shared first answer token -> rank-0). A fix (all-answer-token basis) is
   committed in cell 82s but needs a re-run after clearing the cached `wo_loc_eqresid_*` capture. The paper
   does NOT rely on it: the CAUSAL certification (82y causal-share) carries the dormancy claim. 82s is future
   work / an appendix strengthener. Do not state a static inert-share number.

5. **Base A1/A2/D1 not committed** (instruct-only). Operation-specific / depth controls are stated as
   instruct-only (guardrail-honest).

## STOP-CONDITION LEDGER
- Toolchain present (pdflatex/bibtex/latexmk/matplotlib), network OK -> PDF + figures + template all buildable.
- No blocker prevents a complete, compiling, submission-shaped paper. The two corrections above are folded
  into the claim; nothing is fabricated.
