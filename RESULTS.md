# RESULTS — Operator-Precedence Feasibility (Phases 0–5)

**Model:** `meta-llama/Llama-3.1-8B` (base, bf16) · **HW:** single A100 (Colab) · **seed:** 0 · **last run:** 2026-06-23.

---

## ⏱️ STATE CHECKPOINT (read this first)

| | |
|---|---|
| **Phase reached** | 0–5 complete (the feasibility gauntlet). Phases 6–9 (novel probing) **not started** — intentionally gated off. |
| **Gates** | G0 ✅ · G1 ✅(manual) · G2 ✅ · **G3 ❌ (the finding)** · G4 ✅ |
| **Dashboard verdict** | "NOT YET PLAUSIBLE — unresolved core gate: G3" |
| **What that means** | The **tooling is sound** (G0/G2/G4 green). G3's red is **a real behavioral result, not a bug**: base Llama-3.1-8B does not treat the additive-identity wrapper as a no-op. |
| **Decision pending** | ONE open choice — see [§ Next decision](#next-decision). Options: (A) run `-Instruct`, (B) pivot to the brittleness/composition paper, (C) redesign the contrast. |
| **Blocking rule** | Do **NOT** run novel precedence patching (Phases 6–9) until the Phase 3.5 differential-difficulty confound is resolved (an `-Instruct` run or a redesigned contrast that makes the two arms equally solvable). |
| **How to resume** | Re-open the notebook from GitHub → Run All. Everything is cached on Google Drive (`ART = /content/drive/MyDrive/opprec_interp`); a fresh runtime reloads gates/datasets/sweeps in seconds and only the model reloads (~1–2 min from the Drive HF cache). |

**Persisted artifacts on Drive** (`ART`): `gate_status.json` (G0..G4 ledger), `dataset_phase2.pkl` + `phase2_stimuli.json` (controlled stimuli), `g3_accuracy_result/g3_noop_result/g3_operand_curve` (G3), `locked_band_spec` only if G3 passes (it didn't, so absent), `controls_report.txt` + `p35_decision.json` (Phase 3.5), `phase4_token_map.json`, `g4_baselines/g4_clean_resid_stack/g4_patch_sweep` (G4), plus PNGs `g3_operand_curve.png`, `p35_controls.png`, `g4_patch_heatmap.png`.

---

## Gate-by-gate

| Gate | Verdict | Key number(s) |
|------|---------|---------------|
| **G0** tooling | ✅ PASS | model loads + hooks; `resid_post`(1,seq,4096), `attn.hook_pattern` rows≈1, decode pipeline OK. (Check 5 scoped to "decoding works", not "base model answers 12+7=" — that's G3's job.) |
| **G1** novelty | ✅ PASS-repositionable (manual) | No prior runs the token-identical additive-identity depth contrast + padding-invariance on Llama-3; position against the decodability-vs-causal-use line (Sharma–Dawes–Raval 2026; Nikankin 2024). |
| **G2** controlled stimulus | ✅ PASS | 1000 token-identical Factor-A pairs, 1000 padding series, **parity holds**, 0 drops (after anchoring `depth_right` to `( 0 + B * C )`). |
| **G3** model computes | ❌ FAIL — **the finding** | CHECK1 acc **0.767** ≥ 0.60 (P); CHECK3 must-compute band **(10,49)**, drop 0.44 (P); **CHECK2 no-op FAIL** — `( 0 + B ) * C` corr-with-B·C = **−0.02** vs bare `B * C` = **0.95**. |
| **G4** patching instrument | ✅ PASS | addition activation-patching reproduces the known localization: final-token recovery **1.00 @ layer 30**, mid-to-late band (Stolfo 2023). Direction/sign correct. |

---

## The core result — Phase 3.5 control battery

Eight conditions, one shared operand-pair list, band (20,49), N=400, forward-pass only.

| cond | surface | acc | corr(B·C) | reading |
|------|---------|-----|-----------|---------|
| C0 | `B * C =` (bare) | **0.838** | 0.948 | baseline — it can multiply |
| C4 | `( B * C ) =` (mult **inside** bracket) | **0.890** | 0.960 | **fine** — brackets per se are not the problem |
| C2 | `( 0 + B * C ) =` (depth_right) | 0.710 | 0.282 | mostly fine |
| C5 | `0 + B * C =` (identity, no parens) | 0.583 | 0.109 | disrupted |
| C1 | `( 0 + B ) * C =` (depth_left) | 0.507 | 0.060 | disrupted |
| C3 | `( B ) * C =` (bracket **then** `* C`) | 0.495 | 0.234 | disrupted |
| C6 | `( 0 + B ) =` (sub-expr alone, ans=B) | **1.000** | 0.685 | computes the bracket *perfectly* |
| C7 | `(0+B)*C=` (no spaces) | **0.018** | −0.053 | collapses |

**Decision gate:** acc(C1)=0.507 vs acc(C2)=0.710, **|Δ|=0.203 > 0.10**; correct-subset overlap (Jaccard)=0.697 → **small_confound = False → DROP localization; PIVOT to brittleness.**
**Brittleness gate:** disruption replicates (True); survives surface variant (C7≈C1) = **False** (no-space collapse) → strict gate `brittleness_stands = False`.

### Interpretation
1. **Composition asymmetry (cleanest finding).** `( B * C )`=0.89 ≈ baseline but `( B ) * C`=0.49 — the model handles **multiplication inside a bracket** yet fails **multiplying a bracketed value by a following operand** (`(…) * C`). C6=1.00 makes it explicit: it evaluates `( 0 + B )` → B *perfectly*, then fails to compose that into the outer `* C`. Not explained by the identity alone (C3 has none) or by spacing (C3/C4 both spaced). The depth_left/right gap (0.51 vs 0.71) is the same effect.
2. **Severe surface fragility.** No-spaces `(0+B)*C=` → **0.02**. The model's symbolic arithmetic is extremely tokenization-sensitive — a real second finding, and a confound for single-surface claims.
3. **Differential difficulty kills clean localization.** The two precedence parses aren't equally solvable (Δ=0.20), so a patch contrasting them would be confounded by difficulty, not structure — exactly the artifact G3 + Phase 3.5 exist to catch *before* GPU-heavy probing.

**Bottom line:** two real behavioral findings on base Llama-3.1-8B (composition asymmetry + surface fragility), but neither supports the originally-planned clean precedence *localization* as-is.

> Figures: `docs/figures/g3_operand_curve.png`, `docs/figures/p35_controls.png`, `docs/figures/g4_patch_heatmap.png` (drop the PNGs the notebook writes to `ART`).

---

## Next decision

See [`docs/decision_prompt.md`](docs/decision_prompt.md) for a self-contained brief an agent (or you) can act on.

> **UPDATE — option 1 is now IMPLEMENTED as Phase 6 (the "work order").** The notebook now contains a self-contained Phase 6 section (`cells/76`–`83`) that runs the base 2×2 surface/compose completion, the `-Instruct` re-run (G2/G3/G3.5), the **six validity gates** (§7), the branch decision tree (§8), and the first downstream artifact (localization, or Branch-B controls + the C6→C1 salvage). It emits all deliverables to `results/`. The decision logic is CPU-unit-tested (`tests/test_wo_logic.py`); the gated numbers need one A100 `Run All`. See [`docs/work_order_runbook.md`](docs/work_order_runbook.md). Predicted outcome: **PARTIAL_REPAIR** (compose repaired, surface fragility persists → localization valid conditional on spaced format).

1. **Run `-Instruct` (recommended first move — now wired as Phase 6).** Run the notebook top-to-bottom on an A100; Phase 6 swaps to `meta-llama/Llama-3.1-8B-Instruct`, re-runs the battery, evaluates the gates, and selects the branch. Either Instruct makes C1≈C2≈C0 + surface-robust → **localization valid + a base-vs-Instruct story**, or it's also fragile → **a strong generality result**.
2. **Pivot to the composition-asymmetry paper** on base Llama — no more model runs to start writing; add operand-magnitude / addition-precedence controls.
3. **Redesign the contrast** so both arms keep the op inside a bracket (equal difficulty) → localization may become valid on base Llama; reworks Phase 2 templates.

---

## Reproduce
Notebook: `operator_precedence_phases_0_5.ipynb` (assembled from `cells/` via `build_notebook.py`). G3 logic: `cells/55_phase3_behavioral.py`; control battery: `cells/57_phase35_controls.py`. Companion docs: `docs/methods_standards_buildout.md`, `docs/g3_decision_brief.md`, `docs/results_phase0_5.md`.
