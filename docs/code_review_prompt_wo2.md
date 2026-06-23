# Code-review prompt — Work Order #2 (causal-claim hardening)

> Paste everything below the line into a fresh agent / hand it to a reviewer. It is
> self-contained: it assumes no prior context beyond read access to the repository.

---

You are an **expert AI researcher and mechanistic-interpretability practitioner**. You have
**read access** to the repository below and are conducting a **rigorous, adversarial code review** of
a recently-landed change. Your job is to decide whether this change is correct, reproducible, and
strong enough that the scientific claim it supports would **survive peer review at a top ML venue**
(NeurIPS/ICML/ACL). Be skeptical. Cite exact `file:line`. Separate genuine defects from taste.

## 1. The scientific claim being hardened

The project asks whether **Llama-3.1-8B represents operator precedence / composition** in arithmetic.
The lead, already-run finding: the model evaluates a bracketed subexpression `( 0 + B )` perfectly
(acc 1.000) and multiplies inside a bracket `( B * C )` fine (~0.91), but **fails to compose** them —
`( 0 + B ) * C =` collapses (acc 0.27 Instruct / 0.51 base). The causal centerpiece: the operand `B`
is **linearly decodable** from the post-bracket `)` residual at essentially every layer (CV-R² 0.74–0.96),
yet **patching the correctly-evaluated subexpression into that site never makes the model emit the
correct product** (flip-rate 0.00) → a clean **decodable-but-causally-unused** result.

**The change under review** (Work Order #2) hardens the *causal half* against the three attacks a
reviewer will make: (a) flip-rate 0.00 had **no positive control** proving the patch *can* move outputs;
(b) the patch was applied at **only one layer** (layer 2); (c) the subset was **n=33**, chosen by a
flawed exclusion rule. It also adds cheap pre-emptive controls (few-shot robustness, output-category
distribution, bootstrap CIs, a multi-target decodability baseline).

**The authoritative spec is `docs/work_order_2_causal_hardening.md`. Read it first** — every claim of
correctness below must be judged against that document (tasks §3, deliverables §4, acceptance §5,
gotchas §6).

## 2. How the harness works (read before judging anything)

- The notebook is **assembled from raw cell files**: `cells/NN_name.py` (code) / `*.md` (markdown) are
  concatenated in **lexicographic filename order** by `build_notebook.py` into
  `operator_precedence_phases_0_5.ipynb`. Edits are made to `cells/*.py`, never to the `.ipynb` JSON.
- **Two-tier design (the project's trust model):** all *pure, forward-pass-free* decision/metric math
  lives in `cells/76_wo_logic.py` and is **CPU-unit-tested** in `tests/test_wo_logic.py`; GPU cells are
  thin orchestration over that verified logic. Any new numeric logic that is *not* in cell 76 with a
  test is a finding.
- It runs **top-to-bottom on one Colab A100**, must be **GPU-disconnect-resilient** (everything caches
  to disk via `save_json/has_artifact/load_json`; expensive steps are guarded so a reconnect resumes),
  and loads two models in one session (`base`, `instruct`) namespaced by tag.
- The **validated patching idiom** to mirror is `cells/75_phase5_patching.py` (gate G4): cache CLEAN
  activations, register a forward hook on `blocks.{L}.hook_resid_post` that overwrites
  `resid_post[:, pos, :]`, read final-position logits. `model.run_with_cache` / `model.run_with_hooks`.

## 3. What changed (scope your review here)

Review the **complete change**. It is committed as `b312884` on branch `wo2-causal-hardening`.
See it with `git show --stat b312884` and read cells with `git show b312884 -- cells/82_wo_downstream.py`
(or `git diff main...wo2-causal-hardening`). **Do not** use `git diff HEAD~1` — the history contains
unrelated "Add files via upload" commits. If reviewing an uncommitted working tree instead, run **both**
`git status --short` and `git ls-files --others --exclude-standard` (the new cells `82a/82b/82c` may be
untracked and invisible to a plain `git diff`). The change touches **only** these files:

- `cells/76_wo_logic.py` — new section "8b": `wo_bootstrap_ci`, `wo_paired_delta_ci`, `wo_wilson_ci`,
  `wo_classify_wrong_output`, `wo_fewshot_render`, `wo_shuffle_control`, plus added inline self-test asserts.
- `cells/82_wo_downstream.py` — **coordinated rewrite of `_wo_salvage`** (tasks §3.1/§3.2/§3.3/§3.7),
  new helpers `_wo_battery_res_for_salvage` / `_wo_mk_patch_hook` / `_wo_print_salvage`, a
  `WO_BRANCHB_RES_{tag}` global exposed in `_wo_run_branchb`, and the dispatch now runs the salvage on
  **both** tags.
- `cells/82a_wo_fewshot.py` (new) — few-shot robustness control (§3.5).
- `cells/82b_wo_wrong_output.py` (new) — C1 output-category distribution (§3.6).
- `cells/82c_wo_confidence.py` (new) — bootstrap/Wilson confidence intervals (§3.4).
- `cells/83_wo_repro.py` — deliverables manifest updated.
- `tests/test_wo_logic.py` — new tests for the six functions + the shuffled-B control.

**Phases 0–5 (`cells/00`–`75`) and the `G0..G4` gate ledger must be untouched.** New artifacts/gates
must use `WO_`/`wo_` prefixes only.

## 4. What you must verify

### A. Pure-logic correctness (`cells/76_wo_logic.py` + `tests/test_wo_logic.py`)
- Does each function match its spec in work order §3.4/§3.6 and the pure parts of §3.5/§3.7?
- `wo_wilson_ci`: is the Wilson score formula correct (z from the normal quantile), clamped to [0,1],
  and does it reproduce the documented value `wo_wilson_ci(27,400) ≈ (0.047, 0.097)`?
- `wo_paired_delta_ci`: does it resample **one** set of bootstrap indices and apply it to **both** masks
  (a genuine *paired* bootstrap), not two independent draws?
- `wo_classify_wrong_output`: correct categories and **tie-break priority** (correct > equals_B >
  equals_C > equals_B_plus_C > parse_fail > other)?
- `wo_fewshot_render`: lines are `render(b,c) <gt(b,c)>`, the test prompt is appended bare, **no trailing
  space** (Llama tokenizer pitfall — see cell 75), shots **exclude the test pair** and are distinct, and
  output is **deterministic** given the seed?
- `wo_shuffle_control`: a seeded permutation that actually **decorrelates** the target?
- Are the **tests adequate** — do they actually pin the spec (CI brackets the estimate, width shrinks
  ~1/√n, determinism, paired-delta excludes/contains 0, the classifier ties, the few-shot format), or
  are they shallow enough to let a bug through? Run `python3 tests/test_wo_logic.py` — it must print
  `ALL PASS`. Try to construct an input that breaks a function but passes the tests.
- **Domain guards:** for each new public function, feed an **out-of-domain** input (e.g.
  `wo_wilson_ci(k>n)`, empty masks, `shots > len(pool)`) and confirm it degrades gracefully rather than
  raising — the tests only check documented in-range values, so a missing `0 ≤ k ≤ n` guard can hide.

### B. The salvage GPU rewrite (`cells/82_wo_downstream.py`, `_wo_salvage`)
This is the load-bearing change. Verify each task and **every runtime hazard**:
- **§3.1 positive control:** donor = C4 `( B * C ) =` final-position resid, patched into C1's **final
  position** at `L_pc = floor(0.75·n_layers)`; `pos_ctrl_flip_rate` = fraction C1 argmax becomes the
  correct first token; a **STOP** path triggers if `pos_ctrl_flip_rate < 0.5`. Is this a *valid* positive
  control — i.e. does a pass actually license interpreting the experimental null? Is patching a single
  final-position residual at one late layer a strong enough mover, or could a sub-0.5 result be a
  false alarm? Is the readout target (`correct_first`, derived empirically from C4) the right thing,
  and is it applied **consistently** to both the positive control and the experimental patch?
- **§3.2 layer sweep:** `L_sweep = sorted({best_decode_layer} ∪ range(0, n_layers, 2))`; is
  `best_decode_layer` guaranteed to be in the sweep? Is `flip_rate_by_layer` reported and the **mid-late
  zone (≥0.6·n_layers)** max computed? Is the sweep **checkpointed per layer** and does it resume
  correctly (stale-checkpoint guard on `best_layer`/`n`; deterministic example re-collection)?
- **§3.3 exclusion fix:** keep `(B,C)` iff C1's **full parsed answer ≠ B·C**, read from the
  **in-memory** `WO_*_RES["C1"]["correct_mask"]` index-aligned to `WO_PAIRS` (the saved JSON strips
  masks — confirm the in-memory path is used, not a reload of a stripped file). `wo_salvage_n` default
  256; runs on **both** tags; outputs `salvage_c6_to_c1_{tag}.json`. (`WO_PAIRS` is N=400; C1 acc
  ≈0.27 instruct / 0.51 base ⇒ ~292 / ~196 wrong candidates, so `n_used ≥ 80` is comfortably
  achievable absent heavy alignment-guard skipping.) Does the new criterion correctly avoid the old
  first-token-match over-exclusion **without** silently re-introducing it elsewhere? **Is `n_used ≥ 80`
  merely reported, or actually gated?** (The bar for this work order: `pos_ctrl < 0.5` must STOP;
  `n_used < 80` must at minimum emit a loud WARNING / set an `*_ok` flag — verify which the code does.)
- **Net vs absolute flip-rate:** is the flip counted **net of the unpatched baseline**? Check whether
  the flip loop excludes examples whose *unpatched* C1 argmax was already `correct_first` (the code
  tracks `n_unpatched_correct` / `unpatched_argmax_correct_rate` but verify the `causally_used` verdict
  actually subtracts it). An absolute flip-rate inflates the verdict near its threshold.
- **Magic thresholds:** are all verdict thresholds named constants with a rationale and a test, per the
  two-tier rule? Specifically flag `causally_used = flip_rate_midlate_max ≥ 0.20` — 0.20 is hard-coded,
  not a `CFG` knob, and stricter than the spec's "flip-rate ≈ 0" (§3.2/§5.4). Should it be defined
  relative to `unpatched_rate` / `pos_ctrl` rather than a bare point estimate?
- **§3.7 decodability baseline:** CV-R² for **B, B·C, C, shuffled-B** at the `)` site. Are the targets
  constructed correctly (especially: is `C` expectedly ≈0 because it is *causally future* at the `)`
  position, and is the shuffled-B null actually a null)?
- **Runtime hazards (work order §6):** hook **closure late-binding** in the sweep / positive-control
  loops (must capture loop vars by argument or consume synchronously); **device/dtype** of the CPU-stored
  donor residuals (`.to(model.cfg.device)` then `.to(resid_post.dtype)` in the hook); the `)` **alignment
  guard** (`rp1 == rp6` + token-identical `( 0 + B )` prefix before transplant); any **use-before-assign**
  when `n_used == 0` or `best_layer is None`; any tensor left on the wrong device; GPU/CPU memory blow-up.

### C. New cells + manifest (`cells/82a`, `82b`, `82c`, `83`)
- Do they execute **after** the cells that define their inputs? Confirm lexicographic order
  `82 < 82a < 82b < 82c < 83` and that `WO_PAIRS`, `WO_BASE_RES`, `WO_INSTRUCT_RES`,
  `WO_BRANCHB_RES_instruct` are in memory at that point.
- Few-shot cell: decodes C1 at shots ∈ {0,2,4} on both tags via the resumable `wo_eval`, reuses the
  battery for 0-shot, writes `fewshot_control.csv`. Is the live model tag correct when it decodes
  (`wo_run_battery` asserts `WO_ACTIVE_TAG == tag`)? Any unseeded RNG?
- Wrong-output cell: uses **in-memory** C1 `preds` (index-aligned to `WO_PAIRS`) → `wrong_output_distribution.csv`.
- CI cell: in-memory masks; C4-vs-C1 (both tags), A1-vs-C1 (instruct), magnitude bins → `confidence_intervals.json`;
  is the A1 mask sourced correctly (with a working fallback if Branch-B didn't run)?
- Do the **deliverable filenames exactly match work order §4**? Any `NameError`/`KeyError` if a battery
  dict lacks an expected key?

### D. Acceptance-criteria mapping (work order §4 + §5)
For **each** §4 deliverable and **each** §5 criterion, state satisfied / partial / missing and cite where.
Criterion §5.4 is GPU-run-only (`pos_ctrl ≥ 0.5`; `flip_rate_by_layer ≈ 0` mid-late; `n_used ≥ 80`) —
judge whether the code is correctly *set up to produce and gate on* those, even though they can't run on CPU.

### E. Scientific validity (the part a code-only review misses)
Step back from the code: **is the hardened argument sound?** Audit these in order —
- **CRITICAL — is the C6→C1 experimental patch a no-op by construction?** The alignment guard requires
  C1 (`( 0 + B ) * C =`) and C6 (`( 0 + B ) =`) to be token-identical through `)`. Under causal
  attention, `resid_post[L][')']` is a deterministic function of *only* those identical tokens, so the
  C6 donor residual at `)` is ≈ the residual **already present** in C1 at `)`. If so, patching one into
  the other is an **identity operation**, a near-zero flip-rate is **mathematically guaranteed**
  regardless of whether the model composes, and the causal null carries almost no information. Verify
  by having the code report the per-layer `‖c6_resid[')'] − c1_resid[')']‖`; if it is ≈0, this is the
  load-bearing weakness and the entire layer sweep merely re-measures the unpatched output.
- **Is the positive control too STRONG (tautological) and not site-matched?** It overwrites the
  final-position late-layer residual — one unembed step from the logits — so it almost always flips and
  proves only that the hook *writes tensors*. It does **not** prove that an intervention at the
  experimental `)` site / swept layers can move the output. Is there a **site-matched** positive control
  (e.g. denoising a clean `)` residual into a B-corrupted C1, the `cells/75` operand-corruption idiom)
  that would actually license interpreting the experimental null?
- Is "B decodable, B·C/C not, at the `)` site" honest evidence, given the causal mask makes `C` (and
  thus `B·C`) unavailable there **by construction**? Does the "B is present where the product is not"
  framing *overstate* a trivially-true gap?
- What is the **single strongest objection** a reviewer would raise that this change does **not** yet
  answer? Name it, and say whether it is fixable cheaply (e.g. by reusing the validated operand-corruption
  recovery idiom at the `)` site instead of the identical-prefix transplant).

### F. Non-regression & determinism invariants
- `git diff --stat`: only the files in §3 changed; **no** edits to `cells/00`–`75` or the G0..G4 ledger.
- Operand band `(20,49)`, `wo_build_pairs`/`WO_PAIRS`, and greedy budget `K=8` are unchanged.
- Every sampling draw (salvage subset, few-shot operands, bootstrap, shuffle) uses a **seeded**
  `np.random.default_rng` with a recorded seed; no unseeded RNG.

## 5. How to run the mechanical checks

```bash
cd <repo>
python3 tests/test_wo_logic.py                                   # must print: ALL PASS
python3 build_notebook.py cells operator_precedence_phases_0_5.ipynb   # must print: wrote ... with 30 cells (validated JSON)
python3 -m py_compile cells/76_wo_logic.py cells/82_wo_downstream.py \
    cells/82a_wo_fewshot.py cells/82b_wo_wrong_output.py cells/82c_wo_confidence.py cells/83_wo_repro.py
git diff --stat                                                  # confirm change scope
git diff HEAD -- cells/                                          # eyeball every edited cell
```

(The salvage / batteries require an A100 and cannot run on CPU — review them by reading, against the
`cells/75` idiom.)

## 6. Deliverable — your review

Return a structured review:

1. **Verdict:** `APPROVE` / `APPROVE-WITH-NITS` / `REQUEST-CHANGES`, one sentence.
2. **Findings table**, each row: `severity` (blocker | major | minor | nit) · `file:line` · `title` ·
   `evidence` (quote the code) · `why it bites at runtime or in review` · `concrete fix`. Default to
   *not* reporting style; report real defects, spec violations, resume/determinism breakage, and
   scientific weaknesses.
3. **Acceptance-criteria checklist** (§4/§5): one line each, satisfied/partial/missing + where.
4. **The single most important weakness** of the change as a whole (engineering *or* scientific).
5. **Mechanical-check results** (the §5 commands) and whether each passed.

Be concrete. A finding without a `file:line` and a quoted line of evidence does not count.
