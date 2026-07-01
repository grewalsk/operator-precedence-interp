# DATA.md — single source of truth for every number in the paper

Every value below was re-read from a committed file on `origin/main` with `git show`
(guardrail #1). Each carries its source file and an evidence-level tag:
**[B]** behavioral, **[D]** decodability (correlational), **[C]** causal.
No number enters the paper unless it is here. CIs: Wilson for accuracies, item-bootstrap
for R^2, paired bootstrap for deltas.

Verification date: 2026-07-01. Band (20,49), N=400, seed 0 unless noted.

---

## Behavioral battery [B]

### Instruct — `instruct_battery.csv`
| cond | surface | acc | corr(out,B*C) |
|---|---|---|---|
| C0 | `B * C =` | 0.8475 | 0.982 |
| C1 | `( 0 + B ) * C =` | **0.265** | -0.043 |
| C2 | `( 0 + B * C ) =` | 0.6775 | 0.230 |
| C3 | `( B ) * C =` | 0.755 | 0.546 |
| C4 | `( B * C ) =` | **0.9075** | 0.959 |
| C5 | `0 + B * C =` | 0.5425 | 0.282 |
| C6 | `( 0 + B ) =` | 1.000 | 1.000 |
| C7 | `(0+B)*C=` (nospace) | 0.19 | 0.162 |
| C8 | `(B*C)=` (nospace) | 0.785 | 0.910 |
derived: |acc(C1)-acc(C2)|=0.4125, Jaccard(C1,C2)=0.3561.

### Base — `base_2x2.csv`, `cross_model_battery.csv` (base row)
C0 0.8375, C1 **0.5075**, C4 0.89, C6 1.000, C7 0.0175, C8 0.765. corr(C1)=0.060.
verdict COMPOSE_SPECIFIC (C8 nospace-inside survives 0.765; C7 nospace-outer collapses 0.0175).
**Base A1/A2/D1 are BLANK (instruct-only). State as instruct-only.**

### Branch-B controls, instruct — `branchb_controls.csv`
A1 `( 0 + B ) + C =` = **0.995**, A2 `0 + ( B + C ) =` = 0.9275, D1 `(( 0 + B )) * C =` = **0.0375**.
SELECTIVITY(add-compose works, A1>=.80)=True.
Magnitude bins (C1 vs C4 at matched |B*C|): [440,813) .532/.962 (n=79); [813,1185) .278/.944 (126);
[1185,1558) .163/.885 (104); [1558,1930) .188/.891 (64); [1930,2303) .000/.704 (27). C1<<C4 every bin.

### CIs — `confidence_intervals.json` (n_boot=10000, paired bootstrap)
- delta(C4-C1) base = 0.3825, CI [0.3325, 0.4325]; instruct = 0.6425, CI [0.595, 0.690]. [B]
- delta(A1-C1) instruct = 0.730, CI [0.6875, 0.7725]. A1 acc 0.995 Wilson [0.982, 0.999]. [B]
- (compute any missing acc CI with Wilson: C1 instruct 0.265/400 -> [0.224, 0.311].)

### Few-shot recovery — `fewshot_control.csv`
C1 acc by shots {0,2,4}: base 0.5075 / 0.8375 / 0.8825; instruct 0.265 / 0.885 / 0.915.
C4 reference ceiling: base 0.89, instruct 0.9075. [B] (elicitation caveat, not a capability gap.)

### Wrapper map — `wrapper_map_summary.csv` (drop vs bare, mul unless noted) [B]
base: bare 0.8375; `( B )` 0.495 (drop .3425); `( 0 + B )` 0.5075 (.33); **`( B + 0 )` 0.8225 (drop .015)**;
`( B - 0 )` 0.7625 (.075); `( 1 * B )` 0.6025 (.235); `( B * 1 )` 0.6875 (.15); nest2 0.6475 (.19); nest3 0.615 (.2225).
**nesting breaks ADD too**: nest3 add 0.665 (drop .335). instruct plain `( B )` mul 0.755 (drop only .0925 — instruct is BETTER on plain parens than base).
BINDING CORRECTIONS: not "every wrapper" (base `( B + 0 )` barely moves, .015); not "instruct worse on every wrapper"
(better on plain parens); "parenthesization itself" reading is base-specific; nesting (D1) is a SEPARATE failure mode.

---

## Decodability [D] (correlational — never implies causal use)

### `position_decodability_summary.csv`, `fewshot_decodability_by_layer.csv`
- B at `)` site (C1): R^2 0.9988 best (base L3); L31 base 0.9193, instruct 0.9785. [D]
- **B*C NOT decodable at `)`**: R^2 ~0.439 (base 0.4391, instruct 0.4345). [D]
- product B*C at `=` site: C1 base 0.9669 / instruct 0.9648; C4 base 0.982 / instruct 0.977. [D]
- B at `=` (C1): 0.967.

---

## Selectivity [D] — `probe_selectivity_*.json`, `band_robustness_*.json`

- Band (20,49), C1 `=` product: R2_real **0.9677** vs linear-(B,C) baseline **0.9679**, gap **-0.0002 (~0)**,
  headroom ~0 (baseline ceiling 0.968). => at this band the decode is fully operand-explained. [D]
- Band (2,99) (nonlinear regime, headroom ~0.135): C1 `=` product gap
  base **+0.068**, CI **[0.051, 0.086]**, EXCLUDES 0; instruct **+0.060**, CI **[0.043, 0.079]**, EXCLUDES 0. [D]
- Behavior-tracking: C4 `=` gap > C1 `=` gap at band2 (C4 ~0.09 vs C1 ~0.06); c4_minus_c1 diff CI committed. [D]
- Pre-registered layer (82w, `prereg_gap_*.json`): read C1 gap at the C4-decodability-peak layer
  (base L5, instruct L6; C1 own argmax L7). C1 gap still EXCLUDES 0 (base [0.040,0.072], instruct [0.044,0.078]),
  robust across both layer choices. => the small product component is NOT a layer-selection artifact. [D]

- Band2 PARTS FLOOR: at band (2,99) the parts themselves fall BELOW the 0.80 floor (acc_C4=0.71 base /
  0.73 instruct; acc_C0=0.68/0.685; in_scope_strict=False). So band2 is a HARDER regime / stress test of
  operand-domination, NOT a comfortably in-scope measurement. Must be disclosed; the headline rests on the
  causal null (band-independent), not the band2 gap. [committed in band_robustness_*.json]

**Reconciled reading:** the answer-site product decode is operand-DOMINATED (>=0.94 of the decode R^2 is
reproduced by the linear-(B,C) baseline, ~1.00 at band (20,49)) with a
SMALL but real product component (gap ~0.06, CI excludes 0 only where there is headroom). So "weakly represented",
never "the product is represented" unqualified, and never "pure illusion / zero product".

---

## Causal [C] — steering, swap, causal-share, localization

### Probe-direction steering at `=` — `causal_steering_summary.csv` (WO#5, GT-first-token-logit metric)
inject Δ ~0 at every layer (base `=`: 0.0025 / 0.0019 / 0.0031 / -0.0006 / -0.0038; CIs through 0);
random ~0, shuffled ~0. => the decodable direction is not a causal handle at `=`. [C]
POSITIVE-CONTROL CAVEAT: the WO#5 c4_ref counterfactual is NEGATIVE ~ -0.010 to -0.016 (base) — a FAILED
positive control, because the GT-first-token-logit metric is non-discriminative (Llama chunks the leading
digits; first token shared across products). This is why WO#5 was INCONCLUSIVE. Do NOT cite c4_ref as a
working positive control.

### REPO HAZARD (not a paper number) — stale `inject_C1_L4` cell
`causal_steering_fullproduct_{base,instruct}.json` contains a cell `inject_C1_L4` = +1.99 [1.71,2.28].
This is NOT "steering works": it is a KNOWN cross-surface-baseline BUG (a C1-prompt score compared
against a C4-prompt clean baseline), documented in `cells/82x_wo_dose_response.py` and SUPERSEDED by
the within-surface `dose_response_*.json` (C1 flat at every dose). Do not cite it; the file's own
verdict is SITE_OR_METRIC_STILL_BROKEN. Flagged here so a repo-reading reviewer isn't misled.

### Full-product metric re-score — `causal_steering_fullproduct_*.json`, `causal_steering_lateswap_*.json`
- **Full-residual SWAP at `=` does NOT flip the output**: emit-P' = **0.0** at every late layer
  (lateswap PRODUCT_NOT_AT_EQUALS); swap logprobΔ(P'-true) = **-0.20 (base) / -0.001 (instruct)**. [C]
  >>> THE WORK ORDER'S "a full-residual swap flips it" IS REFUTED. See OPEN_ITEMS.md. <<<
- Inject dose-response (82x, `dose_response_*.json`, full-product metric): C1 inject flat at k in {1,2,4,8}
  (CIs bracket 0); C4 reference base rises monotonically +0.083 -> +0.464 (every CI excludes 0) but stays
  sub-threshold and never flips (emit=0); instruct near-flat. => a WEAK continuous handle at C4, none at C1. [C]
  REPO NOTE: `dose_response_*.json` carries an internal verdict label WEAK_INSTRUMENT. That label is
  thresholded on FLIP-LEVEL range (it requires the C4 reference to cross a 0.5-emit / recover_thr bar), which
  it does not; it does NOT mean the axis is inert. The paper correctly reads the LOGPROB panel (the C4
  reference moves monotonically with a CI excluding 0), because emit-P' saturates at 0. Do not quote the
  one-word label as "the instrument is dead."

### Causal-share Makelov certification — `causal_share_*.json` (82y)
Full swap at `=` (L19): logprobΔ full **+0.593** [0.513,0.673] (base); ablating the decode direction (perp)
still **+0.502** [0.429,0.578]; steering ONLY the decode direction (w-only) **+0.081** [0.062,0.100].
=> the decodable direction reproduces ~14% (base) / ~7% (instruct) of the (sub-flip) effect; ~85% lives in the
orthogonal complement (non-overlapping CIs). Decode direction is NOT the causal carrier (Makelov decode != causal).
emit-P'=0 throughout => the site carries NO answer-FLIPPING effect (whole-site dormant). [C]
NOTE: verdict labels internal only. mean w-magnitude share of the swap ~0.20.

### Localization — `operand_localization_summary.csv`, `head_path_patch_*.json` (82r/82s)
Operand-position patching recovers the answer: b_last exact-recovery **0.500** (base 0.5006 / instruct 0.5010),
c_last **0.500**; attribution ~2.0. rparen/star/equals ~0. No sparse head set (best head 0.008-0.013 < 0.4;
>8 heads needed); 0 DIRECT / 0 MEDIATED of 8 top heads. => DISTRIBUTED across operand positions (diffuse,
Nikankin bag-of-heuristics); no single locus. [C]
>>> THIS is the working positive control: the SAME patching machinery DOES move the answer when applied to the
operand positions (recovery 0.50), so the answer-site nulls are dead DIRECTIONS/SITE, not a dead instrument. <<<

---

## Cross-model — `cross_model_battery.csv`, `chat_format_control.csv`

Matched BARE format (in-scope = parts_work: C4>=.80 & C6>=.80):
- Llama-3.1-8B base C1 0.5075 / instruct 0.265; **Gemma-2-9b-it C1 0.0425** (C4 0.99, C6 1.0);
  Llama-3.2-3B C1 0.3575 (C4 0.8775). All REPLICATE the collapse. [B]
- OUT OF SCOPE (parts fail, name them): Qwen2.5-7B C4 0.0; Mistral-7B C4 0.6175; Llama-3.2-1B C4 0.3125.
Chat format: Gemma composes C1 **0.9475** (C4 0.9875) — but this is a DIFFERENT format; Llama-3.1-8B chat
C4 drops to **0.70** (out of scope), Llama-3.2-3B chat C4 0.74 (out of scope).
**FRAMING (guardrail #5): replication in matched format + a format confound. NEVER a clean cross-family dissociation.**
