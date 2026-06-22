# G3 Decision Brief — is the operator-precedence contrast valid on base Llama-3.1-8B?

A feasibility harness for a mechanistic-interpretability study of **operator precedence** in
`meta-llama/Llama-3.1-8B` (base, bf16) reached its first real behavioral result. Two of three
behavioral checks pass cleanly; the third reveals a genuine effect that forces a design
decision. This brief states what exists, the data, the proposed change, and the open question.

---

## 1. What exists

**Goal.** Decide whether the model *represents* operator precedence, then localize where, via
activation patching. The validity protocol runs five gates in order (G0 tooling, G1 novelty,
G2 controlled stimulus, G3 behavioral, G4 patching-pipeline). Phases 6–9 (the actual probes)
run only if the gates pass.

**The controlled contrast (Factor A).** Token-identical on the real Llama tokenizer (verified):

```
depth_left  :  ( 0 + B ) * C =      # (0+B)*C  = B·C ,  '*' at paren-depth 0
depth_right :  ( 0 + B * C ) =      # 0+(B·C)  = B·C ,  '*' at paren-depth 1
```

Both evaluate to **B·C**, share the identical `( 0 + B` prefix (same token length, B at the
same index), use an **additive identity** (`0 +`, never `×1`) so the multiplication operands
stay engaged, and differ only in where the `)`/`*` sit — i.e. whether `* C` binds to the whole
`(0+B)` or `B*C` is formed inside. This is the precedence manipulation.

**The no-op premise.** The design assumes `0 +` is behaviorally *neutral* — that
`( 0 + B ) * C` is computed like a plain multiplication, so the operands are genuinely engaged
and the depth contrast is clean. **G3 CHECK 2 tests this premise.**

**Gate status.** G0 pass, G1 manual, **G2 pass** (controlled stimulus builds on the real
tokenizer), **G3 FAIL** (details below), G4 not yet reached.

---

## 2. The data (G3, base Llama-3.1-8B, greedy decode)

**CHECK 1 — accuracy on `( 0 + B ) * C =`:** overall **0.767** ≥ 0.60 floor → **PASS**
(parsed-rate 1.00, n=2000; bank is mostly 1–2 digit operands).

**CHECK 3 — must-compute (accuracy vs operand magnitude):** **PASS**, a textbook
graceful-degradation curve on the structured form:

| operand range | accuracy |   |
|---|---|---|
| 2–9   | 1.000 | memorized lookup |
| 10–19 | 0.958 | near-lookup |
| **20–49** | **0.521** | **computing — the sweet spot** |
| 50–99 | 0.146 | struggling |
| 100–199 | 0.094 | |
| 200–499 | 0.021 | collapsed |

Locked must-compute band **(10, 49)**, end-to-end drop 0.44. The model *computes* the structured
form here — not lookup, not chance.

**CHECK 2 — no-op (the failure):** evaluated on a B,C grid uniform in **[2, 99]**:

| surface | corr(output, B·C) |
|---|---|
| `B * C =` (bare) | **0.947** |
| `( 0 + B ) * C =` (structured) | **−0.024** |

Additive-identity agreement (`(0+B)*C` vs `B*C` give same answer): **0.413**. → **FAIL**.

**Reading:** at matched mid-range operands, the model tracks B·C well on **bare** multiplication
but its output on the **additive-identity-parenthesized** form is essentially *uncorrelated*
with B·C. The `(0+…)` structure disrupts the arithmetic far beyond what operand size predicts.
The kill-switch did its job: the no-op premise is **false** on base Llama.

**Gate verdict:** G3 FAIL (CHECK 1 P, CHECK 2 F, CHECK 3 P).

**Known confound:** the bare form differs from the structured form in *two* ways — no `0 +`
**and** no parens. So the disruption is not yet attributed to the additive identity vs the
parentheses vs the compositional depth. (A `( B ) * C =` parens-only and `0 + B * C =`
identity-only control would isolate it; not yet run.)

---

## 3. The reframe (why the FAIL may not break the contrast)

The no-op check compares **structured vs bare**. But the experiment never uses the bare form —
it contrasts **two structured forms** (`( 0 + B ) * C` vs `( 0 + B * C )`), *both* of which carry
the `0 +` and the parens. If the structural disruption is shared across both depth conditions, it
**cancels** in the depth contrast rather than confounding it. What the contrast actually needs is:
*does the model compute B·C in **both** depth conditions, in the band?* — and CHECK 3 shows it
does for the structured form (0.96 → 0.52 across the band). The bare-equivalence the no-op check
demanded is a **stronger** condition than the contrast requires.

So the disruption is a **real, interesting behavioral finding** (base Llama does not treat
`(0+B)*C` as equivalent to `B*C`), but arguably **not** a validity-breaker for the depth contrast.

---

## 4. Proposed change (a faithfulness fix to CHECK 2)

1. Evaluate the no-op/tracking check **in the compute band (10–49)**, not [2, 99] — you cannot
   test "tracks B·C" where the model cannot compute at all.
2. Gate on the condition the contrast actually needs: do **both** `depth_left` and `depth_right`
   predictions track B·C **in-band** (operands genuinely engaged in both conditions)?
3. Demote the **bare-vs-structured** comparison to a **reported finding**, not a gate condition.
4. Add the **confound-isolation diagnostic** (parens-only vs identity-only vs bare) so the
   mechanism is always visible.

This is a design judgment, not a mechanical fix — it would likely turn G3 green while preserving
(and reporting) the disruption finding. The alternative is `-Instruct`/few-shot, which probably
makes the structured form clean across the board but sacrifices base-model purity.

---

## 5. The open question (for an expert researcher)

> A base Llama-3.1-8B computes plain multiplication `B * C =` well (corr 0.95 with B·C at
> two-digit operands) but the additive-identity-parenthesized form `( 0 + B ) * C =` poorly
> (output uncorrelated with B·C at the same operands; ~52% exact accuracy in the 20–49 band; a
> clean graceful-degradation curve). The planned probe contrasts two **structured** forms,
> `( 0 + B ) * C =` (multiply outside the paren) vs `( 0 + B * C ) =` (multiply inside),
> token-identical, both = B·C, to localize where operator precedence is resolved via activation
> patching.
>
> **Decide:** Given that the additive-identity "no-op" premise is behaviorally false (the
> structure disrupts arithmetic), is the depth_left-vs-depth_right contrast still a valid probe
> of precedence representation? Specifically:
>
> 1. Does the shared `(0+…)` structure **cancel** in the depth contrast (so the contrast is
>    valid despite the no-op failure), or does differential difficulty between the two depth
>    conditions reintroduce a confound that must be controlled?
> 2. Is the right validity gate "**both** depth forms track B·C in the must-compute band"
>    (operands engaged), or is bare-equivalence genuinely required — and if the latter, why?
> 3. Is it meaningful to **localize** a computation the model performs correctly only ~50% of the
>    time in-band? Should patching be restricted to correctly-answered items, and does that
>    induce a selection confound?
> 4. Is the disruption itself (`(0+B)*C` ≪ `B*C`) the **more interesting result** — evidence that
>    the base model does not robustly honor precedence structure — and should the study pivot to
>    characterize *that* rather than localize a clean circuit?
> 5. Stay on **base** (preserving the finding) or switch to **-Instruct** (likely restoring the
>    no-op, enabling clean localization, at the cost of probing instruction-tuned behavior)?
>
> Recommend a path and the validity gate that should govern it. Flag any control that must be run
> first (e.g. the parens-only / identity-only isolation) before either proceeding or pivoting.

---

*Reproduce:* notebook + cells at the repo root; G3 logic in `cells/55_phase3_behavioral.py`
(CHECK 2 = the no-op check). The numbers above are from one base-model run on a high-RAM A100.
