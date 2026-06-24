# Work Order #4 — Cross-model generality + boundary / decodability / format mapping

**You are an expert AI researcher.** You have **read/write** access to the repository below. Assume
**no other context** — everything you need is in this document. Your job is to add a small, well-scoped
set of analyses that turn a single-checkpoint behavioral finding into a **confident, cross-model
behavioral paper**, then verify them.

- **Repo:** https://github.com/grewalsk/operator-precedence-interp (work on a branch off
  `wo3-fewshot-probe`, which has all the WO#2/WO#3 machinery).
- **Hardware:** one A100-40GB on Colab. Keep every model **≤ 9B params** (bf16 weights ≤ ~18GB). Total
  added compute is a few GPU-hours. The notebook is GPU-disconnect-resilient (everything caches to disk
  and resumes).
- **Timeline:** ~1–2 days.

---

## 1. What the project found, and why THIS work order matters

The project studies whether Llama-3.1-8B represents **operator precedence / composition** in arithmetic.
The state after three prior work orders (read `results/A100_run_2026-06-24.md` for the full, honest run):

- **The robust finding is behavioral.** Llama-3.1-8B evaluates `( 0 + B )` perfectly (acc 1.000) and
  multiplies inside a bracket `( B * C )` fine (~0.91), but **fails to compose** them — `( 0 + B ) * C =`
  collapses (Instruct 0.265, base 0.507). The failure is **operation-specific** (addition composes:
  A1 `( 0 + B ) + C` = 0.995; Δ(A1−C1) CI (0.69,0.77)), **depth-sensitive** (`( ( 0 + B ) ) * C` = 0.04),
  **magnitude-controlled** (C1 ≪ C4 at every matched |B·C| bin), **unchanged by instruction tuning**, and
  **fully recoverable with 2–4 in-context examples** (Instruct C1 0.265 → 0.915).
- **Every *mechanistic* claim died under its own control.** (a) The causal "operand discarded" patch was a
  no-op by construction, redesigned, then **inconclusive** (the model's zero-shot output is B-invariant, so
  the operand-corruption patch had no contrast). (b) "Few-shot changes use not encoding" — the by-layer
  decodability curve was **not** identical across regimes, and a length-matched non-repairing control
  showed the difference is **context dilution, not consumption** (`ctx_share ≈ 0.94`). So there is **no
  mechanistic centerpiece** — only a modest decodable-but-unused *observation* (B decodable at the `)` site
  at every layer incl. L31 in the failing regime, output B-invariant).

**The one big hole reviewers will hit:** *one model, one template.* This work order fixes that. If the
collapse + operation-specificity + few-shot recovery **replicate across model families and scales**, the
behavioral finding goes from "a quirk of one checkpoint" to "a systematic compositional-elicitation
failure of instruction-following LLMs." That is the highest-value thing left to do, and it is mostly
re-running an existing, model-tag-keyed harness.

**Governing lesson from the prior work orders — obey it:** *run the control / full curve BEFORE writing
the claim.* Best-layer decodability, flip-rate-0, and "decodable in both" all looked like results and all
dissolved under controls. In this work order that means: **a model that does NOT replicate is itself a
result — report it, never cherry-pick;** and any cross-regime / cross-length comparison must carry the
length/dilution caveat the prior run established.

---

## 2. How the harness works (read before editing anything)

- The notebook is **assembled from raw cell files**: `cells/NN_name.py` (code) / `*.md` (markdown) are
  concatenated in **lexicographic filename order** by `build_notebook.py` into
  `operator_precedence_phases_0_5.ipynb`. **Always edit `cells/*.py`, then rebuild:**
  `python3 build_notebook.py cells operator_precedence_phases_0_5.ipynb`. Never hand-edit the `.ipynb`.
- **Two-tier design (preserve it, non-negotiable):** all *pure, forward-pass-free* logic (registries,
  metric/decision math, prompt/condition builders, position finders, verdict functions) lives in
  `cells/76_wo_logic.py` and is **CPU-unit-tested** in `tests/test_wo_logic.py`
  (`python3 tests/test_wo_logic.py` must print `ALL PASS`). GPU cells are **thin orchestration** over that
  verified logic. If you write new decision/metric/parsing math in a GPU cell, STOP and move it to cell 76
  with a test.
- **New cells** sort **after `82f` and before `83`** (the repro/manifest cell must stay last): name them
  `82g_*.py`, `82h_*.py`, … Update `cells/83_wo_repro.py`'s deliverables manifest for any new
  `wo_save_result` outputs.
- **Resumable / cheap-to-rerun:** guard every expensive step with `has_artifact(...)`; cache via
  `save_json`/`load_json`; **checkpoint per (model_tag, condition)** so a disconnect resumes. Re-running a
  cached cell is instant.
- **Determinism:** every draw uses a seeded `np.random.default_rng(...)`; record the seed. No bare RNG.
- **Comparability (do NOT change):** operand band **(20,49)**, the shared **`WO_PAIRS`** (N=400, seed 0),
  greedy budget **K=8**, and the decodability probe (`wo_cv_r2`, folds=5, ridge=1.0). Every model is
  evaluated on the **same** `WO_PAIRS` so all cross-model comparisons are paired.
- **Do not touch** Phases 0–5 (`cells/00`–`75`) or the `G0..G4` gate ledger. New gates/artifacts use
  `WO_`/`wo_` prefixes only.

### Key existing functions you will reuse (do not reimplement)

```python
# cells/76_wo_logic.py  (pure)
WO_MODEL_REGISTRY                 # {"base":..., "instruct":...} -> EXTEND it
WO_BAND, WO_N, WO_SEED, WO_MAX_NEW_TOKENS
WO_CONDITIONS                     # [(key,name,render(B,C)->str, gt(B,C)->int)] C0..C8
WO_BRANCHB_CONDITIONS             # A1/A2/D1
wo_build_pairs() -> [(B,C),...]   # the canonical N=400 shared pairs, band (20,49), seed 0
wo_summarize(preds, golds)        # acc/corr/parse_fail/correct_mask
wo_battery_csv(rows, header)
wo_cv_r2(X, y, folds=5, ridge=1.0)            # held-out dual-ridge CV R^2 (decodability)
wo_fewshot_render(render, gt, shots, test_pair, pool, seed)   # few-shot prompt builder
wo_last_rparen_index(token_strs)              # LAST ')' (the TEST expr's, not a shot's)
wo_classify_wrong_output(pred, B, C)          # correct/equals_B/equals_C/equals_B_plus_C/parse_fail/other
wo_shuffle_control(values, seed)
# cells/77_wo_setup.py  (GPU orchestration)
wo_load_model(tag)                # loads WO_MODEL_REGISTRY[tag]; frees prev; TL primary, HF fallback
wo_run_battery(tag, conditions, pairs, cache_tag=None)   # cached/resumable; asserts WO_ACTIVE_TAG==tag
wo_eval(prompts, key, tag)        # resumable greedy decode, tag-namespaced
wo_assert_parity(pairs, render_left, render_right)        # C1/C2 token parity on the LIVE tokenizer
wo_save_result(filename, text)    # -> ART/results/<filename> (+ mirror ./results)
# cells/79  -> the degeneracy guard + minimal chat-wrapper pattern (bare-continuation may chat on -it models)
# cells/82d -> WO_FSPROBE_PAIRS, _fsp_rparen_last(tokens), _fsp_site_ok(tokens,pos,B), the _fsp_probe loop
# cells/82f -> _wo_gt_wrong(b,c) (random same-#digits wrong answer), the length-matched control pattern
# Phase 3 -> parse_int, _eval_prompts (resumable greedy decode)
```

---

## 3. Tasks

### 3.1 Cross-model generality — **the make-or-break; do it first**

**Registry (cell 76).** Extend `WO_MODEL_REGISTRY` with **instruction-following** models ≤ 9B, plus a
Llama scale pair. Minimum set (add base variants only if budget allows):

```python
"qwen25_7b_it":  "Qwen/Qwen2.5-7B-Instruct",        # UNGATED — start here
"gemma2_9b_it":  "google/gemma-2-9b-it",            # gated (accept license); ~18GB bf16, fits A100-40
"mistral_7b_it": "mistralai/Mistral-7B-Instruct-v0.3",  # gated
"llama32_1b_it": "meta-llama/Llama-3.2-1B-Instruct",    # scale pair (small)
"llama32_3b_it": "meta-llama/Llama-3.2-3B-Instruct",    # scale pair (mid)
```

**Driver (new GPU cell, e.g. `82g_wo_crossmodel.py`).** For each tag in a `CFG`-controlled list
(default = the new models; base/instruct already have results), on the **same `WO_PAIRS`**:
1. `wo_load_model(tag)`. **Per model, re-validate the tokenizer**: run `wo_assert_parity` on C1/C2 and
   record it; if parity is BROKEN, mark the model `tokenizer_incompatible` and **skip its patch-dependent
   parts but still report its battery** (a different tokenizer is a finding, not a crash).
2. **Per model, apply the degeneracy guard** (mirror cell 79): if bare-continuation C0 parse-fail > 0.5,
   fall back to the minimal chat wrapper (and re-assert parity on the wrapped form); **record the format
   used per model** (`bare-continuation` vs `chat-wrapped`). -it models will often need the wrapper.
3. Run the key conditions: **C0, C1, C4, C6, C7, C8** (from `WO_CONDITIONS`) + **A1, A2, D1** (from
   `WO_BRANCHB_CONDITIONS`) + **few-shot C1 at shots {0,2,4}** (reuse the `wo_fewshot_render` +
   `wo_eval` pattern from cell 82a, per-item seeds).
4. Compute the **replication verdict** (pure; see below) and write a per-model row.

**Replication verdict (cell 76, tested) — `wo_replication_verdict(acc, fewshot_c1_4, thr=...)`.** `acc`
is `{cond: accuracy}`. Returns a dict of booleans + an overall label:
- `parts_work`        = acc["C4"] ≥ 0.80 AND acc["C6"] ≥ 0.80   (model CAN multiply-in-bracket & eval the sub-expr)
- `compose_collapses` = acc["C4"] − acc["C1"] ≥ 0.20
- `operation_specific`= acc["A1"] − acc["C1"] ≥ 0.20
- `depth_sensitive`   = acc["C1"] − acc["D1"] ≥ 0.15
- `fewshot_recovers`  = (fewshot_c1_4 − acc["C1"] ≥ 0.20) AND (fewshot_c1_4 ≥ acc["C4"] − 0.15)
- `replicates_core`   = parts_work AND compose_collapses
- `replicates_full`   = replicates_core AND operation_specific AND fewshot_recovers
- If `parts_work` is False → label `"OUT_OF_SCOPE (can't even do C4/C6 — capability, not composition)"`
  (the cross-model analogue of the G_floor capability caveat; report separately, do not count as a
  failure-to-replicate). Thresholds are params; document them.

**Deliverable:** `cross_model_battery.csv` (one row per model: tag, format, parity_ok, C0,C1,C4,C6,C7,C8,
A1,A2,D1, fewshot C1@0/2/4, and every verdict flag) + a printed table. **Report every model honestly,
including non-replicators and out-of-scope models.**

**Acceptance:** ≥ 3 additional models loaded and batteried on the shared pairs; `wo_replication_verdict`
computed for each; the table states, per model, replicates_core / replicates_full / out_of_scope.

### 3.2 Multi-position decodability map (decodability ONLY — no causal claim)

Extend the WO#3 probe (cell 82d) to decode **B** *and* the product **B·C** from the **zero-shot** C1
residual at **four positions**: the post-bracket `)` (source), the `*` operator, the `C` operand, and the
final `=` (answer cue). Also probe **C4** `( B * C ) =` at its `=` as the positive reference (where B·C
*is* decodable at the answer site). New GPU cell `82h_wo_position_map.py`, on base + instruct (and any
cross-model that passes parity).

- **Position finders (cell 76, tested).** Add pure helpers that, given the per-token decoded strings of a
  C1 surface, return the indices of `)` (reuse `wo_last_rparen_index` for the bare case → it's the only
  `)`), the first `*` after `)`, the `C` operand token(s) after `*`, and the final `=`. Multi-token
  operands shift positions, so **locate by decode-and-walk on token content** (mirror the robust locator
  in Phase 2 / `_fsp_site_ok`), and **assert** the located window reads the expected role before probing.
- **Targets.** `B` (decode it), `B*C` (the product). Use the SAME `wo_cv_r2` (folds=5, ridge=1.0). Report
  a `{position: {target: {best_layer, cv_r2, r2_by_layer}}}` map per model.
- **The load-bearing, confound-FREE contrast (within zero-shot):** is **B decodable at `)`** while **B·C
  is NOT decodable at the `=`** in C1 (failing), yet **B·C IS decodable at the `=` in C4** (working)? That
  localizes the breakdown to *binding/routing at the answer site*, not encoding — without any causal claim.
- **Caveat to print:** any *cross-regime* (0-shot vs few-shot) position comparison inherits the
  length/dilution confound (WO#3 established this); keep the headline contrast WITHIN zero-shot (C1 vs C4),
  which is length-matched and confound-free.

**Deliverable:** `position_decodability_{tag}.json` + `position_decodability_summary.csv` (tag, position,
target, best_layer, r2_best) + a heatmap PNG (position × layer) per target. **Acceptance:** the four-site
map produced for ≥ base+instruct; the C1-`)`-B vs C1-`=`-(B·C) vs C4-`=`-(B·C) contrast reported.

### 3.3 Map the failure boundary

Add conditions that vary the trigger, to answer "what *exactly* makes composition fail." New entries in a
`cell-76` registry `WO_BOUNDARY_CONDITIONS` (render + gt; some need a third operand A and/or D — draw them
deterministically per (B,C) with a recorded seed, keeping them in-band):
- `(A+B)*C` real addition inside the bracket (is it the additive *identity* or any bracketed sum?)
- `A*(B+C)`  (bracketed sum on the right, multiplicative outer)
- `(A*B)+C`  (bracketed *product*, additive outer — does the asymmetry flip?)
- depth-2: `( ( 0 + B ) * C ) * D =`  (= B·C·D) and/or `( A + B ) * ( C + D ) =`
Run them as a battery (reuse `wo_run_battery`) on base + instruct (+ cross-model if cheap). Produce a
**trigger table** and a one-line characterization (e.g. "fails iff a bracketed sub-expression feeds a
*multiplicative* outer op"). Pure part: the condition registry + a small `wo_boundary_summary(acc)` that
classifies the trigger. **Deliverable:** `boundary_map.csv`. **Acceptance:** ≥ 4 new surfaces run and
tabulated with a stated trigger characterization.

### 3.4 Format-cued recovery (nail the surprising hint)

The WO#3 control showed Instruct composed at **0.63 with wrong-answer demos** → in-context recovery looks
**format-/task-cued, not content-driven.** Pin it with a length-matched demo-type sweep at fixed shots
(default 4), on base + instruct (+ cross-model if cheap). Demo types (pure builders in cell 76, tested;
all length-matched to the correct-demo prompt):
- `correct`        : `( 0 + b ) * c = (b*c)`   (reuse `wo_fewshot_render` with the real gt)
- `wrong_answer`   : right format, random same-#digits answer (reuse `_wo_gt_wrong`)
- `scrambled_format`: same tokens, permuted surface (breaks the operator-precedence structure) with the correct value
- `random_text`    : length-matched non-arithmetic filler demos (pure format-length control)
Measure C1 accuracy per demo-type per model. **Verdict (pure, tested) `wo_format_cue_verdict(acc_by_type)`:**
if `wrong_answer` recovers ≈ `correct` (within ~0.15) AND `random_text` does NOT (stays near 0-shot) →
**"format-primed, not content-learned"**; if only `correct` recovers → **"content-driven."** **Deliverable:**
`format_recovery.csv` + the verdict. **Acceptance:** 4 demo types × {base,instruct} run; verdict stated.

### 3.5 Decompose the "other" errors

68% of C1 wrong answers are "other." Sub-classify them (pure, cell 76, tested) — extend with
`wo_classify_error_detail(pred, B, C)` returning e.g. `correct / equals_B / equals_C / equals_B_plus_C /
near_product (|pred − B·C|/(B·C) ≤ 0.10) / right_magnitude (same #digits as B·C) / unrelated / parse_fail`.
Aggregate over the in-memory C1 preds (`WO_*_RES["C1"]["preds"]`, index-aligned to `WO_PAIRS`) for base +
instruct (+ cross-model). Tells you whether the model is *attempting the product and erring* vs *doing
something unrelated* — connects to the bag-of-heuristics line. **Deliverable:** `error_detail.csv` (CPU,
cheap). **Acceptance:** the refined distribution written per model.

---

## 4. Deliverables (via `wo_save_result`, mirrored to `results/`)

- `cross_model_battery.csv` — per-model battery + replication verdicts (§3.1).
- `position_decodability_{base,instruct}.json` + `position_decodability_summary.csv` (+ heatmaps) (§3.2).
- `boundary_map.csv` (§3.3).
- `format_recovery.csv` (§3.4).
- `error_detail.csv` (§3.5).
- New/updated unit tests in `tests/test_wo_logic.py` for: `wo_replication_verdict`, the position finders,
  `WO_BOUNDARY_CONDITIONS` surfaces + `wo_boundary_summary`, the demo-type builders +
  `wo_format_cue_verdict`, and `wo_classify_error_detail`.
- `cells/83_wo_repro.py` manifest updated with the new deliverables.

## 5. Acceptance criteria (done when)

1. `python3 tests/test_wo_logic.py` prints **ALL PASS** (existing + all new tests).
2. `python3 build_notebook.py cells operator_precedence_phases_0_5.ipynb` succeeds (valid JSON); cell
   count = previous + (number of new cells).
3. Every new GPU cell `py_compile`s clean and is `has_artifact`-guarded / resumable (checkpointed per
   (model, condition)).
4. A Colab A100 `Run All` produces all §4 deliverables; specifically: ≥ 3 additional models batteried on
   the shared `WO_PAIRS` with a per-model replication verdict (including honest non-replicators /
   out-of-scope); the four-position decodability map; the boundary table; the format-cue verdict; the
   refined error distribution.
5. Phases 0–5 and the `G0..G4` ledger are **untouched**; new gates/artifacts use `WO_`/`wo_` prefixes only.
6. **A short honest write-up** appended to `results/` (or a new `results/A100_run_<date>.md`): does the
   pattern replicate across families and scales? Name every non-replicator and out-of-scope model. State
   the §3.2 within-zero-shot localization. Do **not** over-claim; carry the dilution caveat where relevant.

## 6. Hazards & gotchas (cross-model specific — these will bite)

- **HF gating / auth.** Qwen2.5 is ungated (start there). Gemma-2, Mistral, Llama-3.2 are **gated** —
  the license must be accepted on each model page with the same HF account, and `HF_TOKEN` set. A model you
  can't access must **skip + report**, never crash the run (wrap the load in try/except and record
  `access_denied`).
- **VRAM.** Keep ≤ 9B (Gemma-2-9b ≈ 18GB bf16 fits A100-40 with the sub-30-token sequences here). Do **not**
  add 70B. `wo_load_model` already frees the previous model (gc + `cuda.empty_cache`) before loading the next.
- **transformer_lens architecture support.** TL `from_pretrained` covers Llama/Qwen/Mistral/Gemma-2, but
  versions vary (Gemma-2 has attention logit-softcapping; confirm the loaded model reproduces a sane
  forward). If TL fails for a model, `wo_load_model` falls back to `HFHookedWrapper` (forward + resid_post
  hooks via `model.model.layers`) — fine for batteries (§3.1,3.3,3.4,3.5); for the **decodability** probe
  (§3.2) verify the fallback's `run_with_cache` captures `blocks.{L}.hook_resid_post` correctly (sanity:
  reproduce the base/instruct WO#3 `)`-site R² before trusting a new model's map).
- **Per-model tokenizer re-validation (do NOT assume).** Each model tokenizes the operand band and the
  surface differently → C1/C2 parity, the `)`/position finders, and the few-shot `( 0 + B )` site assertion
  must be **re-checked per model**, with per-model skip counts reported. A model where parity/site-finding
  breaks is `tokenizer_incompatible` for the patch/probe parts (still report its battery).
- **Chat degeneracy.** `-it`/`-Instruct` models often chat instead of emitting a bare number under
  bare-continuation. Reuse cell 79's degeneracy guard + minimal chat wrapper, **per model**, strip the
  template's extra BOS (the double-BOS pitfall), re-assert parity on the wrapped form, and **record the
  format used per model** (numbers from different formats are still comparable as the *pattern*, but the
  format must be disclosed).
- **Comparability.** Same `WO_PAIRS`, band (20,49), seed 0, K=8, probe folds=5/ridge=1.0 for ALL models.
  Few-shot uses the same per-item seeding scheme as cell 82a.
- **Honesty (the project's hard-won lesson).** A non-replication is a result — report it. Keep the §3.2
  headline contrast WITHIN zero-shot (length-matched); flag any cross-length comparison with the dilution
  caveat. Do not let a tidy table hide a model that didn't fit.

## 7. Where to look

- Run summary + current numbers: `results/A100_run_2026-06-24.md` (+ the per-deliverable CSV/JSON files).
- The model-tag harness to reuse: `cells/77_wo_setup.py` (`wo_load_model`, `wo_run_battery`,
  `wo_assert_parity`), `cells/79_wo_instruct_battery.py` (degeneracy guard + chat wrapper).
- The battery + few-shot patterns: `cells/78`, `cells/79`, `cells/82a`.
- The decodability probe + site finders: `cells/82d_wo_fewshot_probe.py`; the length-matched control +
  `_wo_gt_wrong`: `cells/82f_wo_decodability_control.py`.
- Pure logic + tests: `cells/76_wo_logic.py`, `tests/test_wo_logic.py`.
- Methodological history (why the mechanistic claims died): `docs/wo2_redesign_note.md`,
  `docs/work_order_2_causal_hardening.md`.
