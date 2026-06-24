# WO#4 — Cross-model generality + boundary / decodability / format (protocol & reading guide)

**Status: implemented, GPU run PENDING.** The five WO#4 cells (`cells/82g`–`82k`) are
built, CPU-unit-tested, and `py_compile`-clean, but the cross-model numbers require one
A100 `Run All` (≤ a few GPU-hours). This doc states *what each deliverable measures*, *how
to read it honestly*, and the *reporting contract* — so the numbers can be dropped in
without re-litigating the framing. It will be superseded by an `A100_run_wo4_<date>.md`
once the run completes; **do not infer any cross-model result from this file** — every
number below is a placeholder/threshold, not a measurement.

The single-checkpoint finding being generalized (Llama-3.1-8B, `results/A100_run_2026-06-24.md`):
the model evaluates `( 0 + B )` (acc 1.000) and `( B * C )` (~0.91) but fails to compose
`( 0 + B ) * C` (Instruct 0.265, base 0.507); the collapse is operation-specific (addition
composes), depth- and surface-sensitive, magnitude-controlled, unchanged by instruction
tuning, and recoverable with 2–4 in-context examples. Mechanistically only a *modest*
decodable-but-unused observation survives (B decodable at the `)` site at every layer in
the failing regime; output B-invariant); every strong causal claim died under its control.

---

## §3.1 — Cross-model replication (`cross_model_battery.csv`, cell 82g)

Re-runs the model-tag-keyed harness on the SAME `WO_PAIRS` (band (20,49), seed 0, N=400,
K=8) for instruction-following models ≤ 9B: **Qwen2.5-7B-Instruct** (ungated — start here),
**Gemma-2-9b-it**, **Mistral-7B-Instruct-v0.3** (gated), and a Llama-3.2 scale pair
(**1B**, **3B**). Per model: re-validate tokenizer parity (C1/C2); apply the degeneracy
guard (bare-continuation → minimal chat wrapper if C0 parse-fail > 0.5, with the
double-BOS strip + wrapped-parity re-assert); run C0/C1/C4/C6/C7/C8 + A1/A2/D1 + few-shot
C1 at {0,2,4}; emit `wo_replication_verdict`.

**Verdict thresholds** (`WO_REPL_THR`, all overridable):
`parts_work` = C4 ≥ 0.80 ∧ C6 ≥ 0.80 · `compose_collapses` = C4−C1 ≥ 0.20 ·
`operation_specific` = A1−C1 ≥ 0.20 · `depth_sensitive` = C1−D1 ≥ 0.15 ·
`fewshot_recovers` = (fs@4 − C1 ≥ 0.20) ∧ (fs@4 ≥ C4 − 0.15) ·
`replicates_core` = parts_work ∧ compose_collapses ·
`replicates_full` = core ∧ operation_specific ∧ fewshot_recovers.

**Reporting contract (the project's hard-won honesty rule):**
- A model that does **not** replicate is a **RESULT** — it is named in the printed summary
  (`NON-replicators`) and in the CSV `label` column, never dropped.
- A model that can't even do C4/C6 is **`OUT_OF_SCOPE`** (capability, not composition) and
  is reported **separately**, not counted as a failure-to-replicate (the cross-model
  analogue of the G_floor caveat).
- A gated/unavailable model is **`ACCESS_DENIED`** (skipped + reported, run continues).
- A tokenizer whose C1/C2 parity breaks is recorded (`parity_ok=False`); its battery is
  still reported (a different tokenizer is a finding, not a crash).
- **Acceptance:** ≥ 3 additional models batteried on the shared pairs, each with a verdict.

> _Headline to write after the run:_ "The collapse + operation-specificity + few-shot
> recovery {do / do not} replicate across families (Qwen, Gemma, Mistral) and scales
> (Llama-3.2 1B→3B→8B). Replicators: …. Non-replicators: …. Out-of-scope: …." Fill from
> the CSV; **name every non-replicator and out-of-scope model.**

## §3.2 — Multi-position decodability map (`position_decodability_*`, cell 82h)

Probes B and B·C from the **zero-shot** C1 residual at four sites (`)`, `*`, C, `=`) and
B·C from C4's `=` (positive reference), via the same dual-ridge CV-R² (folds=5, ridge=1.0),
on base + instruct. **Load-bearing contrast — entirely WITHIN zero-shot, length-matched, so
confound-free:** is B decodable at `)` while B·C is NOT decodable at the `=` in C1, yet B·C
IS decodable at the `=` in C4? If so, the breakdown **localizes to binding/routing at the
answer site, not to encoding** — stated with **no causal claim**.

**Caveat to carry:** any 0-shot vs few-shot position comparison inherits the
context-**dilution** confound WO#3 established (`ctx_share ≈ 0.94`); the headline stays
within zero-shot (C1 `)`/`=` vs C4 `=`), which is length-matched. A `)`-site B-R² that
reproduces the WO#3 0-shot best-layer value is logged as a plumbing sanity check
(esp. for any HF-fallback model).

## §3.3 — Failure boundary (`boundary_map.csv`, cell 82i)

Varies the trigger over five surfaces (auxiliary operands A,D drawn deterministically
per (B,C), seed `WO_BOUNDARY_SEED=7`): `(A+B)*C`, `A*(B+C)`, `(A*B)+C`,
`((0+B)*C)*D`, `(A+B)*(C+D)`. `wo_boundary_summary` classifies against the hypothesis
**fails iff a bracketed sub-expression is an operand of a multiplicative outer op**
(outer `+` composes; outer `*` collapses). The additive-outer surface `(A*B)+C` is the
key control: if it works while the outer-`*` surfaces fail, the asymmetry is real.

## §3.4 — Format-cued recovery (`format_recovery.csv`, cell 82j)

At fixed shots (default 4), C1 accuracy under four **length-matched** demo types drawing
the **same shot pairs** (differ only in content): `correct`, `wrong_answer` (right format,
random same-#digits answer), `scrambled_format` (same tokens, broken precedence, correct
value), `random_text` (non-arithmetic filler). `wo_format_cue_verdict`: if `wrong_answer`
recovers ≈ `correct` **and** `random_text` stays flat → **"format-primed, not
content-learned"** (pins the surprising WO#3 hint); if only `correct` recovers →
"content-driven". This is a behavioral elicitation result, **not** a mechanistic claim.

## §3.5 — Refined error distribution (`error_detail.csv`, cell 82k)

Sub-classifies C1 wrong answers (the prior ~68% "other") into
`near_product` (within 10% of B·C) / `right_magnitude` (same #digits) / `unrelated`, etc.,
for every model with C1 preds in memory. Tells whether the model is **attempting the
product and erring** (a noisy multiply) vs doing something **unrelated** (the
bag-of-heuristics line, Nikankin 2024).

---

## What this run does NOT claim

It is a **behavioral generality** study plus a within-zero-shot decodability localization.
It does **not** add a causal centerpiece (the WO#2 salvage was inconclusive; output is
B-invariant in the failing regime), and it does **not** revive the "few-shot changes use
not encoding" framing (WO#3 showed the few-shot decodability drop is context **dilution**).
The contribution stands or falls on whether the *behavioral* pattern is systematic across
families and scales — which §3.1 measures, honestly, model by model.
