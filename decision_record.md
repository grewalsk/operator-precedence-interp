# Work-order decision record — operator-precedence Instruct re-run

- **Battery model:** `meta-llama/Llama-3.1-8B-Instruct` (tag `instruct`)
- **Band:** (20, 49)  ·  **N:** 400  ·  **seed:** 0  ·  **format:** bare-continuation  ·  **prepend_bos:** True
- **transformer_lens:** 3.3.0  ·  **model revision:** 0e9e39f249a16976918f6564b8830bc894c89659

## Selected branch

**NO_REPAIR** — §9.B Branch-B controls + §10.B C6->C1 salvage

> Localization invalid (hard gates failed: G_floor, G_neutral, G_symmetry, G_quantity, G_support). Symbolic-arithmetic composition is brittle across base and instruction-tuned Llama-3.1-8B; precedence is decodable but not causally used regardless of tuning. Pivot fully to brittleness (Path B), generality-strengthened. NOTE: G_floor failed — Instruct cannot do bare multiplication at the 0.90 floor; this is a capability story distinct from composition and must be reported as such before any brittleness claim.

## Validity gates (§7, evaluated on the Instruct battery)

| gate | definition | value | pass |
|---|---|---|---|
| G_floor | acc(C0) >= 0.90 | 0.848 | ❌ |
| G_neutral | acc(C1) >= 0.85 AND |acc(C1)-acc(C4)| <= 0.05 | acc_C1=0.265, abs_C1_minus_C4=0.642 | ❌ |
| G_symmetry | |acc(C1)-acc(C2)| <= 0.05 | 0.412 | ❌ |
| G_quantity | corr(C1) >= 0.80 | -0.043 | ❌ |
| G_surface | acc(C7) >= 0.70  (SCOPE FLAG, not a hard abort) | 0.190 | ❌ |
| G_support | Jaccard(C1,C2) >= 0.85 | 0.356 | ❌ |

**Localization verdict:** INVALID — localization invalid; hard gates failed: ['G_floor', 'G_neutral', 'G_symmetry', 'G_quantity', 'G_support']
**Hard-gate AND** (G_floor∧G_neutral∧G_symmetry∧G_quantity∧G_support): False
**Failed hard gates:** G_floor, G_neutral, G_symmetry, G_quantity, G_support

## Battery summary (Instruct)

| cond | surface | acc | corr(B·C) | parse-fail |
|---|---|---|---|---|
| C0 | baseline_mult | 0.848 | 0.982 | 0.000 |
| C1 | depth_left | 0.265 | -0.043 | 0.000 |
| C2 | depth_right | 0.677 | 0.230 | 0.000 |
| C3 | parens_only_out | 0.755 | 0.546 | 0.000 |
| C4 | parens_only_in | 0.907 | 0.959 | 0.000 |
| C5 | identity_no_paren | 0.542 | 0.282 | 0.030 |
| C6 | subexpr_alone | 1.000 | 1.000 | 0.000 |
| C7 | format_variant | 0.190 | 0.162 | 0.035 |
| C8 | nospace_in_bracket | 0.785 | 0.910 | 0.000 |

Derived: |acc(C1)−acc(C2)| = 0.412  ·  Jaccard(C1,C2) = 0.356

## Base 2×2 surface/compose verdict (§6 Step-1)

**COMPOSE_SPECIFIC** — ONE headline: collapse is compose-specific. No-space inside-bracket (B*C)= survives, so C7's collapse is NOT a clean second axis — fold surface fragility into the composition story.


## Downstream artifact to produce (Step 5b)

- **Protocol:** §9.B Branch-B controls + §10.B C6->C1 salvage
- **Run on:** instruct
- Branch-B selectivity controls (§9.B: addition-precedence analogue, depth control, operand-magnitude stratification) + the C6→C1 salvage patch (§10.B): decodable-but-unused causal test.
