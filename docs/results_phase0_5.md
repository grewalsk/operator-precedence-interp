# Results — Operator-Precedence Feasibility (Phases 0–5), base Llama-3.1-8B

**Run:** `meta-llama/Llama-3.1-8B` (base, bf16), single A100, seed 0, 2026-06-22.
**One-line verdict:** the tooling is sound (G0/G2/G4 green); the **science gate G3 fails in an informative way** — base Llama does *not* treat the additive-identity wrapper as a no-op, and the controls battery shows *why*. The feasibility harness did its job: it caught, before any expensive probing, that **base Llama-3.1-8B cannot support a clean precedence *localization* on this contrast.**

> Figures (added separately): `figures/g3_operand_curve.png`, `figures/p35_controls.png`, `figures/g4_patch_heatmap.png`.

---

## Gate summary

| Gate | Phase | Verdict | Evidence |
|------|-------|---------|----------|
| G0 | 0 | **PASS** (tooling) | model loads + hooks: device✓, finite logits✓, `resid_post` (1,seq,4096)✓, `attn.hook_pattern` rows≈1✓, decode pipeline✓. (Check 5 rescoped: it verifies decoding works, not that the *base* model answers a bare "12 + 7 =" — that's G3's job.) |
| G1 | 1 | **PASS-repositionable** (manual) | No prior runs the token-identical additive-identity depth contrast + padding-invariance on Llama-3; nearest hit Sharma–Dawes–Raval 2026 (toy Dyck). Frame against the decodability-vs-causal-use line. |
| G2 | 2 | **PASS** | 1000 token-identical Factor-A pairs, 1000 padding series, parity holds on the real Llama tokenizer (after anchoring depth_right to `( 0 + B * C )`). 0 drops. |
| G3 | 3 | **FAIL — the finding** | CHECK1 acc 0.767 ≥ 0.60 (P); CHECK3 must-compute band (10,49), drop 0.44 (P); **CHECK2 no-op FAIL** — `( 0 + B ) * C` corr-with-B·C = **−0.02** vs bare `B * C` = **0.95**. The additive identity is NOT a behavioral no-op. |
| G4 | 5 | **PASS** | Addition activation-patching reproduces the known localization: final-token recovery **1.00 @ layer 30**, mid-to-late band (Stolfo 2023). Patch direction/sign correct. |

**Plausibility verdict:** NOT a clean green — but not because the tooling is broken. The block is **G3**, and G3's failure is a *result*, adjudicated by the Phase 3.5 control battery below.

---

## Phase 3.5 — behavioral control battery (the core result)

Eight conditions, one shared operand-pair list, band (20,49), N=400, all forward-pass only.

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

**Diagnostic verdicts:**
- Q1 parens alone disrupt? **True** — but note the asymmetry: `( B ) * C` (C3=0.49) disrupts while `( B * C )` (C4=0.89) does not.
- Q2 identity alone disrupt? **True** (C5=0.58).
- Q3 where does C1 fail? **computes `(0+B)=B`, then fails the multiply** (C6=1.00 high, C1=0.51 low).
- Q4 surface artifact? **differs** — no-space `(0+B)*C=` craters to 0.02 vs spaced 0.51.
- Ingredient: parens AND identity each disrupt independently.

**Decision gate (localization validity):** acc(C1)=0.507 vs acc(C2)=0.710, **|Δ|=0.203 > 0.10**; correct-subset overlap (Jaccard)=0.697. → **small_confound = False → DROP precedence localization to future work; PIVOT to the brittleness characterization.**

**Brittleness gate:** disruption replicates (C1/C3/C5 ≪ C0) = True; survives surface variant (C7 ≈ C1) = **False** (no-space collapse); → brittleness_stands = **False** by the strict gate.

---

## Interpretation (what these numbers mean)

**1. The composition asymmetry (the cleanest finding).** Compare C4 `( B * C )` = 0.89 against C3 `( B ) * C` = 0.49 — both spaced, both one structural step. The model handles **multiplication *inside* a bracket** but fails **multiplying a bracketed value *by* a following operand** (`(…) * C`). C6 makes the mechanism explicit: it evaluates `( 0 + B )` → B **perfectly (1.00)**, then fails to compose that result into the outer `* C`. This is a crisp *bounded-composition* statement, and it is **not** explained by the additive identity alone (C3 has no identity) or by spacing (C3/C4 are both spaced). The depth_left vs depth_right gap (0.51 vs 0.71) is the same effect: "multiply inside" (depth_right) is easier than "multiply the bracket" (depth_left).

**2. Severe surface/tokenization fragility (the confound + a second finding).** Removing spaces (`(0+B)*C=`) collapses accuracy to **0.02**. The model's symbolic arithmetic is extremely sensitive to the exact token boundaries — a real phenomenon, but one that entangles with the precedence story and undercuts strong claims from a single surface form.

**3. Differential difficulty kills clean localization.** Because the two precedence parses are not equally solvable (Δ=0.20), an activation-patching contrast between them would be confounded by difficulty rather than by structure. This is exactly the artifact G3 + Phase 3.5 exist to catch *before* GPU-heavy Phases 6–9.

**Bottom line:** there are two real behavioral findings on base Llama-3.1-8B — a **composition asymmetry** (`op` inside a bracket is fine; composing a bracketed result into an outer `op` is not) and **surface fragility** — but neither supports the originally-planned clean precedence *localization* as-is.

---

## What to do next (the decision)

Three viable directions; they are not mutually exclusive, but pick the primary thrust:

1. **Run the `-Instruct` second experiment (recommended first move).** Now strongly motivated, not an escape: swap `CFG["model_name"]="meta-llama/Llama-3.1-8B-Instruct"`, raise `g3_accuracy_floor` to ~0.80, re-run G2/G3/G3.5 (~20 min GPU). Two clean outcomes:
   - Instruct makes C1≈C2≈C0 and surface-robust → **localization becomes valid** and you get a *base-vs-Instruct "tuning installs precedence robustness"* story.
   - Instruct is *also* fragile → a strong **generality** claim about precedence brittleness in 8B models.

2. **Pivot the paper to the composition-asymmetry characterization on base Llama.** Make the result the result: `( B * C )` fine vs `( B ) * C` broken, the perfect-subexpr-but-failed-compose pattern (C6 vs C1), and the surface fragility, with the controls battery as the core evidence. Needs more controls (addition-precedence, operand-magnitude sweeps), not heavy patching.

3. **Redesign the contrast** so both arms keep the operation *inside* a bracket (avoiding the `(…)*C` asymmetry), making the two conditions equally solvable → localization on base Llama may become valid. Requires reworking the Phase 2 templates.

**Operationally, right now:**
- Re-run the notebook once more to pick up the **G0 fix** (tooling gate now correctly green) — after that, G0/G2/G4 are all green and the dashboard's only red is G3, which is the documented finding.
- Decide direction 1/2/3 above. If 1, it's a one-line model swap + re-run. If 2, no more model runs are strictly required to start writing. If 3, the Phase 2 templates change.
- The novel precedence patching (Phases 6–9) stays **gated off** until the decision-gate confound is resolved (i.e. until an `-Instruct` run or a redesigned contrast makes the two arms equally solvable).
