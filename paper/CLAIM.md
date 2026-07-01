# CLAIM.md — the locked claim (reconciled to committed evidence)

## Title (working)
A Dormant Subspace, Not a Missing One: Decodable but Causally Inert Arithmetic Composition in Llama-3.1-8B

## Single finding (the paper)
In Llama-3.1-8B, a parenthesized additive-identity wrapper selectively breaks multiplicative
composition (`( 0 + B ) * C =` fails while the parts succeed and addition composes). At the answer
site, the operand B and the near-product B*C are linearly decodable (R^2 ~ 0.96), yet this decode is
an **interpretability illusion**:
1. **operand-dominated** — at band (20,49) it has zero margin over a linear-(B,C) baseline (gap ~ 0);
   even at band (2,99), where B*C is nonlinear in the operands, the product component is small
   (gap ~ 0.06, CI excludes 0) and behavior-tracking (C4 gap > C1 gap);
2. **causally inert** — probe-direction steering along the decodable direction leaves the output
   unchanged, and a full-residual swap at the answer site ALSO fails to move the output (emit-P' = 0);
   the answer is re-derived diffusely from the operand positions (operand-patch recovery ~0.50, no
   sparse locus). Within the small sub-flip logprob effect a swap does produce, the decodable direction
   carries ~14%; ~85% lies orthogonal to it.

**Decodability at the answer site is not evidence of computation there.**

## Evidence-level discipline
- Behavioral [B]: the blind spot, operation-specificity, depth, magnitude control, few-shot recovery.
- Decodability [D]: B and B*C decode R^2; selectivity gap vs linear-(B,C) baseline (correlational).
- Causal [C]: steering null, swap-no-flip, causal-share (~14%), operand-route recovery 0.50 (the positive control).
A [D] number NEVER implies [C].

## What we DO NOT claim (retraction firewall)
- NOT "the swap flips the answer" (refuted; emit-P'=0).
- NOT "pure illusion / zero product" (band2 gap CI excludes 0 -> weakly represented).
- NOT a clean cross-family dissociation (matched-bare replication + a format confound).
- NOT "every wrapper breaks it" / "instruct worse on every wrapper" (base `( B + 0 )` barely moves; instruct
  better on plain parens).
- NOT a single causal locus (distributed, Nikankin).

## Motivating hook (one paragraph, not the headline)
Parenthesizing a multiplicative operand selectively breaks multiplication while addition is preserved;
few-shot examples recover it. The hook motivates the mechanism; it is not the contribution.

## Positioning (exact sentences)
- Williamson et al. 2025 (Syntactic Blind Spots, MathNLP 2025) show behaviorally that math errors track
  syntactic complexity and recover under rephrasing; we localize one token-matched instance and show its
  internal signature is a dormant, operand-explained subspace, not a missing representation.
- Matsumoto et al. 2022 find decoded intermediate values are causally used in arithmetic; we find a
  decodable answer-site subspace that is causally inert.
