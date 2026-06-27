# WO#6 — Operand-route localization + dormant-subspace certification

**Branch:** `wo6-operand-localization` (off `wo5p1-steering-calibration`).

This work order completes the MathNLP paper: it turns the WO#5 *negative* (the `=`
answer-site is causally dormant) into a *positive* localization (where and how the
product is actually computed), and certifies the negative to Makelov grade. It is the
**pre-registered design + interpretation guide**; the GPU cells (`82r`, `82s`) emit the
actual numbers to `results/` on the A100 run. Methodology below is non-negotiable and
matches a verified literature review.

## Why this experiment

WO#1–#5 established a controlled dissociation on the *failing* zero-shot C1 surface
`( 0 + B ) * C =`: the product `B·C` is **linearly decodable** at the `=` site
(CV-R² ≈ 0.96–0.98), but that decode is (i) **operand-explained** — the linear-(B,C)
baseline matches it and the Hewitt–Liang control task is undecodable (WO#5 Exp B:
`OPERANDS_ONLY`, product-R² 0.97 ≤ linear-(B,C) 0.968), and (ii) **causally inert under
the probe direction** — injecting along the fitted direction never moves the answer, and
a full-residual swap localized nothing to the decodable subspace (WO#5.1d:
`DEAD_DIRECTION`; the full-residual swap *does* flip the answer, so the site carries the
answer, but **not via the decodable direction**). This is the Makelov–Lange–Nanda
dormant-subspace / interpretability-illusion phenomenon ("Is This the Subspace You Are
Looking For?", 2311.17030).

The open question is therefore **not** "is the product represented" (a settled,
operand-explained yes) but, reframed per Nikankin et al. "Arithmetic Without Algorithms"
(bag-of-heuristics): **what computation, localized where, produces the answer?** The
literature's prior (Stolfo et al. 2023) is that the model re-derives the product from the
*operand positions* during decoding — operand tokens → last token via a few attention
heads, with late MLPs — rather than from the `=` residual.

## Methodology (mandatory, from a verified review)

- **Symmetric Token Replacement counterfactuals** (Zhang & Nanda 2309.16042): flip ONE
  operand to a digit-count-matched alternative (`C→C'` or `B→B'`), so the clean and
  corrupt surfaces are token-aligned. **NOT** Gaussian noising. We run both corruption
  targets — which operand you corrupt surfaces different heads.
- **A teacher-forced, multi-token logprob-DIFFERENCE metric** (Heimersheim & Nanda
  2404.15255): `D = logprob(clean_answer) − logprob(corrupt_answer)`, each summed over the
  FULL multi-token answer. A difference against a contrast — **not** a probability, **not**
  a bare logprob, **not** a first-token logit (robust to a shared first answer token).
  `D_clean ≫ 0`, `D_corrupt ≪ 0`.
- **An explicitly stated noising-vs-denoising direction.** DENOISING = patch CLEAN into
  CORRUPT (sufficiency: does restoring this site recover the clean answer?). NOISING =
  patch CORRUPT into CLEAN (necessity: does corrupting this site destroy it?). Both are
  run; backup/self-repair can hide necessity, so we never read necessity off denoising
  alone.
- **Cheap attribution patching** (Syed et al.): one fwd + one bwd of `D` per answer gives
  the whole `(layer × position)`, `(layer × head)`, `(layer)` map. → **exact-patch
  verification** of the role×layer grid + top-K heads/MLPs, reporting attribution-vs-exact
  agreement (Spearman, sign, top-k overlap). → **path patching** (Wang et al. IOI /
  Goldowsky-Dill et al.) to split direct-vs-mediated head effects.
- **DAS is OUT** — it finds illusory subspaces and inherits the same interpretability
  illusion we are certifying. **Makelov decompose-and-compare** is used instead for B1.

## Experiment A — localize the operand→answer computation (cell `82r`)

For each STR pair and each corruption target, attribution patching builds the cheap
whole-graph map (both directions), then exact activation patching verifies it:

- a **role × layer residual** denoising-recovery grid (roles = `plus, b_last, rparen,
  star, c_last, equals`) — *the* `(position × layer)` deliverable;
- **top-K attributed heads**: single-head exact recovery + a greedy cumulative-recovery
  curve (→ `n_heads_for_half`);
- **per-layer MLP-out** exact recovery;
- a **gold flip-rate** (greedy-decode + `wo_parse_int == B·C`) at the flipped operand's
  best-attribution layer (pre-registered from attribution, not from the exact recovery —
  no winner's curse);
- paired bootstrap CIs (`wo_paired_delta_ci`) for the flipped-operand vs `=` contrast.

### Verdict logic (`wo_localization_verdict`, pure + unit-tested)

| Verdict | Fires when |
|---|---|
| `LOCALIZED_OPERAND_ROUTE` | restoring the flipped operand's position recovers the answer (`operand_pos_recovery ≥ thr`) AND a **sparse** head set carries the operand→last-token movement (best single head `≥ thr`, or `n_heads_for_half ≤ 8`). |
| `DISTRIBUTED_NO_LOCUS` | the operand position matters but **no** sparse head set reaches threshold — the product is recomputed by a diffuse bag of heuristics (Nikankin), no single causal locus. |
| `INCONCLUSIVE` | even the flipped operand's own position fails to recover (instrument/site problem, not a result). |

### Expected outcome (stated up front)

Given Stolfo et al. and the WO#4 position map, the expected result is
`LOCALIZED_OPERAND_ROUTE`: operand token positions carry the recovery, a few mid-layer
heads move operand→last-token, late MLPs participate, and the `=` position does **not**
carry it at early layers (consistent with WO#5.1d). **Either fork completes the paper** —
a clean `DISTRIBUTED_NO_LOCUS` is the publishable bag-of-heuristics result, reported
honestly. The attribution-vs-exact agreement is reported so a divergence between the cheap
estimate and the real intervention is visible, not hidden.

### Path patching — direct vs mediated (cell `82s`, A4)

For the top operand heads, patch each head's output and read its TOTAL recovery, then
re-measure with every *later* layer's attention-head outputs and MLP-outs **frozen to
corrupt** so the head's effect can reach the logits only through the residual skip — that
is its DIRECT recovery; `MEDIATED = TOTAL − DIRECT`. A mostly-direct head writes the
answer-relevant signal straight to the unembedding; a mostly-mediated head's effect is
re-processed downstream (Stolfo's late-MLP story). This is path patching, not bare head
patching.

## Experiment B — certify the `=` negative is dormant (cell `82s`, B1/B2)

**B1 — Makelov decompose-and-compare.** Fit the `B·C` decode direction `ŵ` at the
final-layer `=` residual (`wo_fit_ridge_probe`). Decompose it into the **logit-affecting**
component `v_row` (projection onto the span of the answer-token unembedding columns — what
the unembedding/late path reads) and the **logit-inert** component `v_null` (the orthogonal
remainder). Report (i) the inert share of `ŵ` and (ii) the decode-R² recoverable from the
logit-affecting subspace (`R2_row`) vs the logit-inert complement (`R2_null`) vs full
(`R2_full`).

| Verdict (`wo_dormant_verdict`) | Fires when |
|---|---|
| `DORMANT_CERTIFIED` | `ŵ` is dominantly logit-inert AND `R2_null ≈ R2_full ≫ R2_row` — decodable but causally disconnected from the weights (the Makelov-grade certification). |
| `LOGIT_COUPLED` | `ŵ` lives in the readout row space and `R2_row ≈ R2_full` — the product is readable by the logits, not dormant. |
| `INCONCLUSIVE` | the decode is not cleanly carried by either subspace. |

Expected: `DORMANT_CERTIFIED`, certifying "decodable but causally disconnected" from the
weights. The decode + decompose are taken in **post-final-LayerNorm space** when the model
exposes `ln_final` (the exact space the unembedding reads: `logits = ln_final(resid) @ W_U`),
so "logit-affecting = span(W_U columns)" is exact rather than an isometry approximation. The
verdict **downgrades to `INCONCLUSIVE`** when under-powered (`n < 10` unique pairs) or when
`R2_full` is too weak to support a dormant claim. DAS is deliberately not used.

**B2 — the three-legged dissociation.** Combined with the full-component nulls already in
hand, the `=` site is **decodable · operand-explained · causally inert**:

1. **decodable** — `B·C` linearly decodable at `=` at CV-R² ≈ 0.96–0.98 (WO#3 / WO#5);
2. **operand-explained** — product-R² (≈0.97) does not exceed the linear-(B,C) baseline
   (≈0.968); Hewitt–Liang control task undecodable (WO#5 Exp B: `OPERANDS_ONLY`);
3. **causally inert (decodable direction)** — the probe-direction inject is
   `DEAD_DIRECTION` and the decodable subspace is logit-inert (WO#5.1d + B1 here), even
   though a full-residual swap at `=` flips the answer (the site carries the answer, just
   not via the decodable direction).

The dormant-certification JSON embeds the WO#5 deliverables (`probe_selectivity_*.json`,
`causal_steering_lateswap_*.json`) when present, so the three legs travel together.

## Deliverables

- `operand_position_patch_{base,instruct}.{json,png}` — attribution + exact role×layer
  denoising heatmaps, head/MLP recoveries, the verdict, and attribution-vs-exact agreement.
- `head_path_patch_{base,instruct}.json` — per-head TOTAL / DIRECT / MEDIATED recovery +
  classification.
- `dormant_certification_{base,instruct}.json` — per-layer inert share, `R2_full/row/null`,
  the dormant verdict, and the WO#5 references.
- `operand_localization_summary.csv` — per (tag, role) best-layer exact recovery +
  attribution + verdict.

## Reproduce

`python3 tests/test_wo_logic.py` (ALL PASS — incl. the new WO#6 tests for the STR pair
builder, `wo_patch_metric` / teacher-forced logprob, the attribution-patching math,
`wo_readout_decompose` / decode-split, and the localization/dormant verdicts) and
`python3 tests/test_wo6_mock.py` (ALL PASS — execs `82r`/`82s` unmodified end-to-end
against a torch autograd stand-in: metric signs, mover-head localization, attribution-vs-
exact agreement, path patching, the Makelov dormant certification, and checkpoint/resume).
Both are CPU and **gate** the A100 run. Then `python3 build_notebook.py cells
operator_precedence_phases_0_5.ipynb` and Run-All on an A100 for `base` + `instruct`.
