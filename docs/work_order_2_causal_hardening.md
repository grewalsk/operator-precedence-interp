# Work Order #2 — Causal-claim hardening (Tier 1 + Tier 2)

**You are an expert AI researcher.** You have read/write access to the repository described below.
Assume **no other context** — everything you need is in this document. Your job is to add a small,
well-scoped set of analyses that make one specific scientific claim survive peer review, then
verify them.

- **Repo:** https://github.com/grewalsk/operator-precedence-interp
- **Hardware:** the analyses run on a single Llama-3.1-8B / -Instruct on one A100 (Google Colab).
  Total added compute is well under one GPU-hour. The notebook is GPU-disconnect-resilient
  (everything caches to disk and resumes).
- **Timeline:** ~1 day of work.

---

## 1. What the project found (the claim you are hardening)

The project studies whether Llama-3.1-8B represents **operator precedence / composition** in
arithmetic. The lead finding (already run, in `results/A100_run_2026-06-22.md`):

> The model evaluates a parenthesized subexpression `( 0 + B )` perfectly (acc **1.000**) and
> multiplies inside a bracket `( B * C )` fine (acc **0.91**), but **fails to compose** them —
> `( 0 + B ) * C =` collapses to acc **0.27** (Instruct; 0.51 base). The failure is
> operation-specific (addition composes at 0.93–0.99), depth-sensitive (redundant nesting
> `( ( 0 + B ) ) * C` = 0.04), magnitude-controlled, and unchanged by instruction tuning.
> **Causally:** the operand `B` is linearly decodable from the post-bracket `)` position at
> **every layer** (CV-R² 0.74–0.96, ~0.80 at the mid-late layers), yet patching the correctly-
> evaluated subexpression into that site **never** makes the model emit the correct product
> (flip-rate **0.00**). The operand is **computed, carried through the network, and discarded** —
> a clean *decodable-but-causally-unused* result.

**The vulnerability this work order fixes:** the causal half rests on (a) a flip-rate of 0.00 with
**no positive control** proving the patch *can* move outputs, (b) a patch applied at **only one
layer** (layer 2), and (c) a small **n=33** subset chosen by a flawed exclusion rule. Reviewers
will attack exactly these. We also add three cheap controls that pre-empt the most common
objections.

---

## 1b. Relevant work (why this matters + where the method comes from)

- **Decodability ≠ causal use.** Probing shows a quantity is *present*, not that it is *used*.
  Hewitt & Manning 2019 (structural probe; NAACL N19-1419) established decoding structure from
  representations; Hewitt & Liang 2019 (control tasks; arXiv 1909.03368) showed probes need
  selectivity controls. Our finding is a causal counterexample — a maximally-decodable operand
  that is causally discarded. (The repo's Phase 1 novelty table cites a near-neighbor that
  dissociates decodability from causal use on toy Dyck bracket languages; we instantiate it in
  *real arithmetic on a production model*.)
- **Mechanistic interpretation of arithmetic.** Stolfo, Belinkov & Sachan 2023 (causal mediation;
  EMNLP; arXiv 2305.15054) and Hanna, Liu & Variengien 2023 (GPT-2 "greater-than"; NeurIPS; arXiv
  2305.00586) localize arithmetic via activation patching — the instrument we reproduce (gate G4)
  and extend here. Quirke & Barez 2023 (arXiv 2310.13121) analyze addition in a toy transformer.
- **Heuristics vs. algorithms.** Nikankin, Reusch, Mueller & Belinkov 2024 ("Arithmetic Without
  Algorithms: a Bag of Heuristics"; arXiv 2410.21272; incl. Llama-3-8B) argue LLM arithmetic is
  heuristic, not algorithmic. Our composition failure — it computes the parts but cannot bind them —
  is direct evidence for that view.
- **Patching methodology (ties to Task 3.1).** The denoising activation-patching idiom we use
  follows the standard line (Meng et al. 2022, ROME; arXiv 2202.05262) and its methodological
  guidance (Heimersheim & Nanda 2024, "How to use and interpret activation patching"; arXiv
  2404.15255). The **positive-control requirement in §3.1 is exactly the hygiene that line
  emphasizes**: a null patching effect is only interpretable once you show the same hook *can* move
  the output.

*(Verify arXiv ids/venues before citing in a paper; the repo's Phase 1 table in the notebook has
the fuller related-work landscape.)*

---

## 2. How the harness works (read before editing anything)

- The notebook is **assembled from raw cell files**: `cells/NN_name.py` (code) and `*.md`
  (markdown) are concatenated in **lexicographic filename order** by `build_notebook.py` into
  `operator_precedence_phases_0_5.ipynb`. **Always edit the `cells/*.py` files, then rebuild**:
  `python3 build_notebook.py cells operator_precedence_phases_0_5.ipynb`. Never hand-edit the
  `.ipynb` JSON.
- The notebook runs **top to bottom on Colab A100**. Phases 0–5 load the model + validate the
  tooling (gates G0/G2/G3/G4). **Phase 6** (`cells/76`–`83`) is the work-order code you are
  extending. Cells you will touch: `cells/76_wo_logic.py` (pure CPU logic),
  `cells/77_wo_setup.py` (model load + battery runner), `cells/82_wo_downstream.py` (the salvage +
  Branch-B downstream artifact).
- **Two-tier design (preserve it):** all *pure, forward-pass-free* logic lives in
  `cells/76_wo_logic.py` and is **unit-tested on CPU** in `tests/test_wo_logic.py`
  (`python3 tests/test_wo_logic.py`, must print `ALL PASS`). GPU cells are thin orchestration over
  that verified logic. **Any new decision/metric math goes in cell 76 with a test in
  `tests/test_wo_logic.py`** — this is non-negotiable; it is how we trust numbers before spending
  GPU time.
- **Persistence:** `ART` is the artifact dir (Colab Drive). Helpers: `save_json/load_json/
  has_artifact/save_pickle/load_pickle/get_gates/set_gate/log`. Deliverables go through
  `wo_save_result(filename, text)` → `ART/results/<filename>` (mirrored to repo `./results/` when
  present). **Guard every expensive step with `has_artifact(...)` so a disconnect resumes.**

### Key existing functions you will reuse (do not reimplement)

```python
# cells/76_wo_logic.py
WO_CONDITIONS          # [(key,name,render(B,C)->str, gt(B,C)->int)] for C0..C8
WO_BRANCHB_CONDITIONS  # A1/A2/D1
wo_build_pairs() -> [(B,C), ...]          # the canonical N=400 shared pairs, band (20,49), seed 0
wo_cv_r2(X, y, folds=5, ridge=1.0)        # held-out dual-ridge CV R^2 (decodability probe)
wo_summarize(preds, golds) -> {...}       # acc/corr/parse_fail/correct_mask
wo_battery_csv(rows, header) -> str
wo_parse_int(text) -> int|None

# cells/77_wo_setup.py
wo_load_model(tag)                        # tag in {"base","instruct"}; frees prev model first
wo_run_battery(tag, conditions, pairs, cache_tag=None) -> {key: summary}  # cached/resumable
wo_save_result(filename, text)

# cells/82_wo_downstream.py
_wo_empirical_first_tok(tokens) -> int    # model's top-1 next token at final pos
_wo_first_tok_logit(logits, pos, tok_id) -> float
_wo_salvage(tag)                          # THE function you are rewriting (see §3.3)
WO_PAIRS                                  # the shared pair list (set in cell 78 at run time)
WO_INSTRUCT_RES, WO_BASE_RES              # battery results incl. per-condition 'correct_mask','preds'
model, tokenizer                          # the live transformer_lens HookedTransformer
```

### Current baseline numbers (so you can sanity-check)

- Instruct battery: C0=0.8475, C1=0.265, C2=0.6775, C3=0.755, C4=0.9075, C5=0.5425, C6=1.000,
  C7=0.19, C8=0.785.
- Salvage (Instruct): CV-R²(B)=0.96 @ layer 2 (0.74–0.96 across all 32 layers), patch flip-rate
  0.00, n_used=33, n_skipped=0, n_c1_already_correct_excluded=95.
- Patching idiom (validated in `cells/75_phase5_patching.py`, gate G4): cache CLEAN activations,
  run with a forward hook on `blocks.{L}.hook_resid_post` that overwrites `resid_post[:,pos,:]`,
  read the final-position logits. `model.run_with_cache(...)` / `model.run_with_hooks(...)`.

---

## 3. Tasks

> **Coordination note.** Tasks **3.1, 3.2, 3.3, 3.7** all modify the salvage path in
> `cells/82_wo_downstream.py` (`_wo_salvage`) and share its per-example activation-collection loop —
> implement them as **one coordinated rewrite** of `_wo_salvage`, not four passes. Tasks **3.4 and
> 3.6** are pure logic (cell 76 + tests). Task **3.5** is a new GPU cell.

### Tier 1 — needed for the causal claim to survive review

#### 3.1 Positive control for the patch  *(GPU; the single most important gap)*
**Problem.** `flip-rate = 0.00` is only meaningful if the patching hook demonstrably *can* change
C1's output. Add a positive control.

**Build.** In `_wo_salvage`, after the experimental C6→C1 patch, run a **positive-control patch**
that should flip C1 to the correct product:
- Donor = the same `(B,C)` rendered as **C4** `( B * C ) =` (the model evaluates this correctly).
  Cache the donor's `resid_post`.
- Patch the donor's residual at the **final (answer-cue) position** into C1 at the **same final
  position**, at a late layer `L_pc = floor(0.75 * n_layers)`.
- Measure `pos_ctrl_flip_rate` = fraction where C1's argmax becomes the correct first token.

**Acceptance.** `pos_ctrl_flip_rate ≥ 0.5` (proves the hook moves outputs) **while** the
experimental post-bracket `pos_ctrl`-vs-experimental contrast is reported side by side in
`salvage_c6_to_c1.json`. If the positive control is *not* ≥ 0.5, **stop and report** — the patching
machinery is suspect and the whole salvage is untrustworthy.

#### 3.2 Patch-layer sweep  *(GPU)*
**Problem.** The C6→C1 patch was applied at only layer 2.

**Build.** Sweep the experimental C6→C1 patch over a layer set `L_sweep = sorted({best_decode_layer}
∪ range(0, n_layers, 2))` (i.e. every other layer plus the best-decode layer). Report
`flip_rate_by_layer: {L: rate}`. Checkpoint per layer (`has_artifact`) so a disconnect resumes.

**Acceptance.** `flip_rate_by_layer` present in the output; the claim is supported iff flip-rate
stays ≈ 0 across the **mid-late** layers (≈ 0.6·n_layers … n_layers−1), i.e. the consumption zone.
Report the max flip-rate over that zone explicitly.

#### 3.3 Fix the over-exclusion + run on base  *(GPU)*
**Problem.** The salvage keeps an example only if C1's **first token** differs from the correct
first token — this over-excludes (n dropped from ~290 to 33). The intended subset is "examples
where C1 is **actually wrong**."

**Build.**
- Change the keep-criterion to: keep `(B,C)` iff **C1's full parsed answer ≠ B*C**. Reuse the
  already-computed battery predictions: `WO_INSTRUCT_RES["C1"]["correct_mask"]` /
  `["preds"]` are index-aligned to `WO_PAIRS`. (Decode C1 only if a needed pred is absent.)
- Raise the sample size knob `CFG["wo_salvage_n"]` default to 256 (still cheap).
- Run the salvage on **both** tags: call `_wo_salvage("instruct")` **and** `_wo_salvage("base")`
  (base is the cleanest demonstration — C6=1.000, C1=0.51). Namespace outputs per tag
  (`salvage_c6_to_c1_{tag}.json` and `wo_salvage_{tag}`).

**Acceptance.** `n_used ≥ 80` on each tag; both base and instruct salvage files written.

#### 3.4 Bootstrap confidence intervals  *(PURE; cell 76 + tests)*
**Build (cell 76).**
```python
def wo_bootstrap_ci(mask, n_boot=10000, alpha=0.05, seed=0) -> (lo, hi)
    # mask: list[0/1]. Percentile bootstrap CI for mean(mask). Deterministic (seeded).
def wo_paired_delta_ci(mask_a, mask_b, n_boot=10000, alpha=0.05, seed=0) -> (lo, hi)
    # CI for mean(a)-mean(b), resampling PAIRED indices (a,b are index-aligned).
def wo_wilson_ci(k, n, alpha=0.05) -> (lo, hi)   # closed-form binomial CI (no RNG).
```
**Tests (tests/test_wo_logic.py).**
- CI brackets the point estimate; width shrinks ~1/√n (compare n=50 vs n=5000).
- `wo_paired_delta_ci(m, m)` contains 0; on disjoint masks (a all-1 where b all-0) excludes 0.
- Deterministic given seed; `wo_wilson_ci` matches a known value (e.g. k=27,n=400 ≈ (0.047,0.097)).

**Apply (GPU cell, after the batteries — masks must be in memory).** Add a small cell that computes
CIs for the headline deltas using the in-memory `correct_mask`s: **C4 vs C1**, **A1 vs C1**, and
each operand-magnitude bin (C1 vs C4). Write `results/confidence_intervals.json`.
*Note:* the saved battery JSON strips `correct_mask`; this cell must run in the **same session** as
the batteries (re-running the battery cells is instant — they are cached).

### Tier 2 — defuses the obvious objections

#### 3.5 Few-shot / robustness control  *(GPU; new cell)*
**Problem.** "Did you just prompt it badly?"

**Build.** Add `wo_fewshot_render(render, gt, shots)` (pure, in cell 76 — testable) that prepends
`shots` worked examples of the **same surface** (`( 0 + B ) * C = <answer>\n`), with operands drawn
deterministically (separate seed) and **excluding the test pair**, then the test prompt. New cell
runs C1 under `shots ∈ {0, 2, 4}` few-shot on `WO_PAIRS`, both tags. Write
`results/fewshot_control.csv`.

**Acceptance.** Few-shot C1 accuracy reported per shot count; the claim holds iff it stays well
below C4 (does **not** jump to ~0.9). The prompt-builder test asserts the shots are distinct from
the test pair and correctly formatted.

#### 3.6 "What does C1 output instead?" classifier  *(PURE; cell 76 + tests)*
**Build (cell 76).**
```python
def wo_classify_wrong_output(pred, B, C) -> str
    # returns one of: 'correct' (pred==B*C), 'equals_B', 'equals_C', 'equals_B_plus_C',
    #                 'parse_fail' (pred is None), 'other'
```
**Test.** Label a table of synthetic `(pred,B,C)` → expected category (pred=B*C→'correct';
pred=B→'equals_B'; pred=C→'equals_C'; pred=B+C→'equals_B_plus_C'; pred=None→'parse_fail';
pred=99999→'other'). Resolve ties by priority order as listed.

**Apply (GPU/CPU cell).** Aggregate the category distribution over **C1's parsed preds**
(`WO_INSTRUCT_RES["C1"]["preds"]` + the pairs) for base and instruct. Write
`results/wrong_output_distribution.csv` (category, count, fraction, per tag).

#### 3.7 Decodability baseline  *(GPU activations; PURE logic)*
**Problem.** Is `B` *specifically* sitting unused, or is everything decodable at that site?

**Build.** In the salvage's activation-collection loop, at the post-bracket `)` site, fit
`wo_cv_r2` for **multiple targets**: `B` (have), the **product `B*C`** (the thing the model should
be computing toward), `C`, and a **shuffled-B control** (B values permuted → expect ≈ 0). Report
`decodability_by_target: {target: cv_r2_best_layer_and_value}` in the salvage output.

**Acceptance.** Expect `B` high (~0.9), `B*C` **low** at this early/site position, control ≈ 0.
This shows `B` is present where the product is not — strengthening "the operand sits there unused."
(`wo_cv_r2` is already tested; the only new logic is target construction — add a one-line test that
the shuffled-B target decorrelates.)

---

## 4. Deliverables

Written via `wo_save_result(...)` to `ART/results/` (mirror to repo `results/`):

- `salvage_c6_to_c1_instruct.json` and `salvage_c6_to_c1_base.json` — each containing, in addition
  to the existing fields: `pos_ctrl_flip_rate`, `flip_rate_by_layer`, `n_used` (≥80),
  `decodability_by_target`.
- `confidence_intervals.json` — CIs for C4-vs-C1, A1-vs-C1, and each magnitude bin.
- `fewshot_control.csv` — C1 acc at shots ∈ {0,2,4}, base + instruct.
- `wrong_output_distribution.csv` — C1 output-category distribution, base + instruct.
- New/updated unit tests in `tests/test_wo_logic.py` for: `wo_bootstrap_ci`, `wo_paired_delta_ci`,
  `wo_wilson_ci`, `wo_classify_wrong_output`, `wo_fewshot_render`, and the shuffled-B target.

## 5. Acceptance criteria (this work order is done when)

1. `python3 tests/test_wo_logic.py` prints **ALL PASS** (existing + all new tests).
2. `python3 build_notebook.py cells operator_precedence_phases_0_5.ipynb` succeeds (valid JSON).
3. Every new GPU cell `py_compile`s clean and is guarded/resumable (`has_artifact`).
4. A Colab A100 `Run All` produces all §4 deliverables, and specifically:
   - `pos_ctrl_flip_rate ≥ 0.5` on both tags (else a documented STOP);
   - `flip_rate_by_layer` ≈ 0 across the mid-late layers;
   - salvage `n_used ≥ 80` on both tags;
   - CIs reported for the three headline contrasts;
   - few-shot C1 acc reported and the wrong-output distribution written.
5. The validated Phases 0–5 and the `G0..G4` gate ledger are **untouched**; new gates/artifacts use
   `WO_`/`wo_` prefixes only.

## 6. Risks & gotchas

- **Hook closure late-binding.** When you build a forward hook inside a loop, the hook must capture
  the current layer/position/donor by **argument** (factory function) or be **consumed synchronously**
  in the same iteration — never collect hooks in a list and call them later. (See `_mk_hook` in
  `cells/82` for the correct pattern.)
- **Device/dtype.** Donor/patch residuals stored on CPU must be `.to(model.cfg.device)` before the
  hook writes them, and `.to(resid_post.dtype)` inside the hook (bf16). See the existing salvage
  code for the pattern.
- **`correct_mask` is stripped from saved JSON.** Anything needing per-item masks (CIs, the
  exclusion fix, the wrong-output distribution) must read the **in-memory** `WO_*_RES` in the same
  session, not the saved files. Re-running the battery cells is instant (cached).
- **Determinism.** Every sampling draw (few-shot operands, bootstrap, salvage sample) uses a fixed
  seeded `np.random.default_rng(...)`; record the seeds. Do **not** introduce `Math.random`/
  unseeded RNG.
- **Do not change** the operand band (20,49), `WO_PAIRS`, or the greedy budget (K=8) — comparability
  with the existing run depends on them.
- **GPU cost guard.** The layer sweep is `len(L_sweep) × n_used` forward passes per tag (~16 × 90 ×
  2 ≈ 3k forwards); fine, but checkpoint per layer so a disconnect never restarts the sweep.

## 7. Where to look

- Run summary + current numbers: `results/A100_run_2026-06-22.md`,
  `results/salvage_c6_to_c1.json`, `results/instruct_battery.csv`.
- Validated patching idiom to mirror: `cells/75_phase5_patching.py`.
- The function you are rewriting + the pure-logic cell + the tests:
  `cells/82_wo_downstream.py`, `cells/76_wo_logic.py`, `tests/test_wo_logic.py`.
- Harness/runbook: `docs/work_order_runbook.md`.
