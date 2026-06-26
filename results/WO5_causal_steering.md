# WO#5 — Contrast-free causal steering + probe selectivity

**Branch:** `wo5-causal-steering` (off `wo4-crossmodel-generality`).
**Cells:** `82l_wo_steering.py` (Experiment A, GPU), `82m_wo_selectivity.py` (Experiment B, CPU),
pure logic + tests in `76_wo_logic.py` / `tests/test_wo_logic.py` / `tests/test_wo_steering_mock.py`.
**Status:** CPU-verified (325 checks across the pure-logic + mock-execution suites, all green);
awaiting an A100 Run-All to fill the numbers below. This document is the **pre-registered design
+ interpretation guide**; the cells emit the actual numbers to `results/` — *no number here is
hand-entered.*

## Why this experiment

The decodability result is **correlational**: `B` is linearly decodable at the `)` site and `B·C`
at the `=` site of the *failing* zero-shot C1 surface `( 0 + B ) * C =` (R²≈0.96–0.98). That is
exactly the pattern Hewitt–Liang taught reviewers to distrust, and the only thing separating us
from "Syntactic Blind Spots." The prior causal salvage returned **n_used=0** because it relied on a
first-token *argmax contrast*, and zero-shot C1 output is `B`-invariant — there was nothing to
denoise. WO#5 drops the argmax contrast and scores the **ground-truth product token's logit**, so
every test item contributes a Δ regardless of its argmax.

## Experiment A — what it does (cell 82l)

For C1 at the `)` site (inject operand `B`) and the `=` site (inject product `B·C`):

1. **Capture** `hook_resid_post` at both sites over the shared pairs (all layers), plus the C4
   `( B * C ) =` `=` residual; cache to one pickle per tag (`wo_steer_resid_{tag}.pkl`) so
   Experiment B and every re-run are **CPU-only**.
2. **TRAIN/TEST split** (seeded). Per `(site, target, layer)` fit the *same* dual-ridge probe as
   `wo_cv_r2` on the **TRAIN** half → a unit direction ŵ + a value↔coordinate map. The steered
   items are never in the probe's fit set.
3. On the held-out **TEST** half, **activation-steer the clean run** along ŵ via `run_with_hooks`:
   `resid' = resid + (target_coord − ŵ·resid)·ŵ` (pure `wo_inject_to_target`). Because we steer the
   *clean* run, the live residual at the site equals the cached clean residual, so the whole delta is
   precomputed on CPU and the hook is a pure additive patch (mirrors the validated cell-75 idiom).
   - **Inject** (headline): write the correct value in. The layer is **pre-registered** by
     TRAIN cross-validated decodability (the layer where the probe fits best on the *train* half),
     **not** by the test-set argmax-Δ — so the verdict's Δ and CI are scored at a layer chosen
     without peeking at the tested statistic (no winner's-curse). The full sweep is reported, but
     as *exploratory*. The C4 reference layer is pre-registered the same way.
   - **Erase** (LEACE-ish): steer the probe coordinate to the train mean. At the pre-registered layer.
4. **Metric:** Δ = `logit_GT(intervened) − logit_GT(clean-C1 baseline)`, GT = the product's first
   answer token (leading-space convention, cell 75). Mean Δ over TEST with a **paired bootstrap CI**
   (`wo_paired_delta_ci`) + flip-rate-to-GT-product.
5. **Controls** (each its own row, swept across layers): norm-matched **random direction** (Δ≈0 ⇒
   effect is direction-specific), **shuffled target** (inject a wrong product ⇒ GT logit must not
   rise), and the **C4 positive reference** — the inject mechanism + per-pair logit metric *do* move
   a routed-product logit at a `=` site. NB the literal "inject the true product at C4's `=`" has a
   **ceiling** (C4 already emits it, Δ≈0 would falsely read as a dead instrument), so the reference
   injects a **counterfactual** product `P'` at C4's `=` and shows *its* logit rises — ceiling-free.

### Verdict logic (`wo_steering_verdict`, pure + unit-tested)

| Verdict | Meaning | Fires when |
|---|---|---|
| **RECOVERS** | product present and causally *sufficient when routed*, unused by default | inject Δ ≥ threshold, CI excludes 0, and it beats both random + shuffled controls |
| **CLEAN_NULL** | operand/product genuinely *ignored downstream*, not merely mis-decoded | \|Δ\| within the null band with a tight CI **and** the C4 reference confirms the metric works |
| **INCONCLUSIVE** | the instrument is unproven | the C4 reference itself fails |

**Governing rule:** run the control before the claim; a clean null with the C4 reference passing
**is the finding** — do not over-claim. An n-driven "inconclusive" is not a result; a CLEAN_NULL is.

### Expected outcome (stated up front, to be confirmed by the run)

Given the prior evidence — `B·C` decodable at the `=` site yet C1 fails, and the WO#4 position map
localizing the breakdown to **binding/routing at the answer site** — the expected result is
**CLEAN_NULL with the C4 reference passing**: injecting the decodable product along the C1-`=` probe
direction does **not** route to the output, i.e. *the product is decodable at the answer site but
causally ignored downstream — a routing failure, not a mis-decode.* If instead injection raises the
GT logit (RECOVERS), the product is present and sufficient-when-routed but unused by default. Either
way the contrast-free metric removes the n=0 failure mode. The run fills it in:

> **Steering verdict (base):** `<see results/causal_steering_base.json → verdict.label>`
> **Steering verdict (instruct):** `<see results/causal_steering_instruct.json → verdict.label>`

Deliverables: `causal_steering_{base,instruct}.json`, `causal_steering_summary.csv`,
`causal_steering_layersweep_{base,instruct}.png`.

## Experiment B — probe selectivity (cell 82m, pure CPU)

Protects the decodability claim against "R²=0.96 just means B and C are linearly present and ridge
approximates their product." Three controls, fixed folds=5/ridge=1.0:

1. **Hewitt–Liang control task** — random label per unique `(B,C)`; selectivity = R²(real) − R²(control).
2. **Shuffled-product target** — must collapse to ~0.
3. **The decisive linear-(B,C) baseline** — a synthetic matrix carrying *only* `B` and `C` linearly
   (+ noise). If the real residual's product-R² **exceeds** this baseline, the residual genuinely
   contains the product.

> **BAND CAVEAT (reported honestly, not hidden).** Over the positive band `[20,49]`, `B·C` is
> *already* ~linearly predictable from `B` and `C` (the main-effect approximation captures ≈97% of
> Var(B·C); the interaction term is small). So the linear-(B,C) baseline need **not** collapse — the
> load-bearing quantity is the **contrast** R²(real) − R²(baseline), not R²(real) in isolation. The
> verdict reflects this: `REPRESENTED` only if real exceeds the baseline by a margin; otherwise
> `OPERANDS_ONLY` ("the product-R² is explained by the linear presence of B and C — the reviewer's
> rebuttal holds at this band"). This nuance is the whole point of the control; a naive "baseline
> collapses ⇒ product represented" would have over-claimed.

Deliverable: `probe_selectivity.csv` (tag, site, target, layer, R2_real, R2_control_task,
R2_shuffled, R2_linearBC_baseline, selectivity, baseline_gap, verdict) +
`probe_selectivity_{tag}.json`.

> **Selectivity (C1 `=`, product):** `<see probe_selectivity.csv / probe_selectivity_{tag}.json>`

## WO#5.1 — steering-instrument calibration (follow-up, cell `82n`)

The WO#5 A100 run came back **INCONCLUSIVE on both tags**: every Δ (inject, random,
shuffled, **and the C4 positive reference**) was ~0 at every layer, so the steering
instrument was never validated and the C1 result is uninterpretable (the verdict
logic correctly refused a clean-null claim). Experiment B explained part of it —
`OPERANDS_ONLY`: the product-R²=0.97 at `=` does **not** exceed the linear-(B,C)
baseline (0.968, = the analytic ~97% main-effect ceiling for band [20,49]), so the
probe direction is ~99% operand-reconstructible and isn't a product-specific axis.

**Cell `82n` (WO#5.1) disambiguates** whether WO#5 was a *dead instrument* or a real
null, by climbing a causal ladder at the C4 `=` site and **reusing the cached WO#5
residuals (no re-capture)**:
1. **Zero-ablation** — zero the whole C4 `=` residual; the GT logit must collapse, else
   the hook/site/metric is broken (`INSTRUMENT_BROKEN`).
2. **Full-residual swap** — replace the C4 `=` residual with a donor (product P′); does
   the model emit P′? (a guaranteed-causal positive control).
3. **Magnitude sweep** — scale the probe-direction counterfactual inject by k∈{1…32}
   until P′'s logit crosses threshold; record `k*` and `‖δ‖/‖resid‖`.
4. **C1 re-eval @k\*** — if calibrated, inject a counterfactual at C1's `=` at k* and ask
   whether the C1 answer site is drivable.

Verdict (`wo_steer_calibration_verdict`): `INSTRUMENT_BROKEN` / `METRIC_OR_SITE_SUSPECT`
/ `CALIBRATED@k*` (WO#5 was under-powered → re-run C1 at k*) / `DEAD_DIRECTION` (the
operand-aligned probe direction is genuinely not a causal handle — *decoding ≠ causal*).
Deliverables: `causal_steering_calibration_{base,instruct}.json` +
`causal_steering_calibration_summary.csv`. Compute: ~2k forward passes/tag at one layer
each (≈15–20 min A100 total) since the expensive WO#5 capture + full sweep are reused.

### WO#5.1 run result + WO#5.1b re-metric (cell `82o`)

The WO#5.1 A100 run returned **`METRIC_OR_SITE_SUSPECT`** on both tags — but the raw
ladder showed the instrument *works*: zero-ablation hammered the GT logit (Δ≈−7.7), and
a **full-residual swap at C4's `=` flipped the model's answer to the donor's product on
100% of items** (`flip_to_donor=1.0`, both tags). Yet the *absolute first-token logit*
moved by ~0 (Δ≈0.13 / −0.006). Diagnosis: the metric is compressed — many leading-digit
chunk tokens (`" 108"`, `" 176"`, …) sit at similar logits, so the argmax can flip
completely while the absolute value stays flat. **So the WO#5 null was (at least partly)
a metric artifact, not "product unused."**

Cell `82o` (WO#5.1b) re-scores the ladder with **flip-rate-to-target** + **logit-diff
(target − true token)**, and sweeps the inject across **(layer × k)** including a *late*
layer (WO#5/5.1 pre-registered the early decodability-peak ~L4, not the composition site
~L0.85·n). It reuses the cached residuals. Verdict `wo_steer_flip_verdict`: `CALIBRATED@
(layer,k)` if the rank-1 probe-direction inject flips the answer somewhere (⇒ WO#5 was
under-powered/under-metriced; re-run C1 there), else `DEAD_DIRECTION` (the swap flips but
the operand-aligned probe axis never does ⇒ *decoding ≠ causal direction*; switch to
DAS/gradient directions or a centered operand band). Deliverables:
`causal_steering_remetric_{base,instruct}.json` + `_summary.csv`. ~15–20 min A100 (cache
reused).

## Reproduce

`python3 tests/test_wo_logic.py` (ALL PASS) and `python3 tests/test_wo_steering_mock.py` (ALL PASS)
first — these are CPU and gate the GPU run. Then `python3 build_notebook.py cells
operator_precedence_phases_0_5.ipynb`, set `HF_TOKEN`, Run-All. Every forward pass is cached per
`(model-tag, site, layer)` under `ART`; a fresh runtime resumes from disk, and Experiment B reuses
the capture pickle with no model. Cost: a few A100-hours, all cached. Phases 0–5 and gates G0–G4 are
untouched; all new artifacts use `WO_`/`wo_` prefixes.
