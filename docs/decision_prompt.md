# Decision prompt — next step for the operator-precedence interpretability project

*Paste this to an expert mech-interp agent/model. It is self-contained: all numbers needed to decide are below.*

---

You are advising on a mechanistic-interpretability project. A **feasibility study (Phases 0–5)** on **base `meta-llama/Llama-3.1-8B`** has completed. Decide the single next step and justify it.

## The original plan
Probe whether the model represents **operator precedence** by contrasting two **token-identical** arithmetic surfaces, then localize where precedence is resolved via activation patching:
- `depth_left  = "( 0 + B ) * C ="`  → (0+B)·C,  `*` at paren-depth 0
- `depth_right = "( 0 + B * C ) ="`  → 0+(B·C),  `*` at paren-depth 1

Both evaluate to **B·C**, share the `( 0 + B` prefix (equal token length, B at the same index), use an additive identity (`0 +`, never `×1`). The lead result was to be a distance/padding-invariance probe; activation patching corroborates.

## What the gates found
- **G0 (tooling) PASS** — loads + hooks correctly.
- **G2 (controlled stimulus) PASS** — 1000 token-identical pairs, parity holds on the real tokenizer.
- **G4 (patching instrument) PASS** — reproduces the known single-addition localization (final-token recovery 1.00 @ layer 30, mid-to-late layers; Stolfo 2023).
- **G3 (model computes the task) FAIL** — specifically the **no-op check**: the additive-identity wrapper is **not** behaviorally neutral.

## The decisive data (Phase 3.5 control battery, band 20–49, N=400, exact accuracy)
| cond | surface | acc | corr(B·C) |
|---|---|---|---|
| C0 | `B * C =` | 0.838 | 0.948 |
| C4 | `( B * C ) =` | 0.890 | 0.960 |
| C2 | `( 0 + B * C ) =` (depth_right) | 0.710 | 0.282 |
| C5 | `0 + B * C =` | 0.583 | 0.109 |
| C1 | `( 0 + B ) * C =` (depth_left) | 0.507 | 0.060 |
| C3 | `( B ) * C =` | 0.495 | 0.234 |
| C6 | `( 0 + B ) =` (ans=B) | 1.000 | 0.685 |
| C7 | `(0+B)*C=` (no spaces) | 0.018 | −0.053 |

Derived: **|acc(C1)−acc(C2)| = 0.203** (> 0.10 threshold); correct-subset Jaccard overlap = **0.697**; the project's pre-registered decision gate therefore returned **"DROP localization, PIVOT to brittleness."**

## What the data establishes
1. **Composition asymmetry:** multiplication *inside* a bracket is fine (C4=0.89), but multiplying a *bracketed value by a following operand* `(…) * C` fails (C3=0.49, C1=0.51). C6=1.00 shows the model evaluates `( 0 + B )` perfectly but cannot compose it into the outer `* C`.
2. **Surface fragility:** removing spaces collapses accuracy to 0.02 — symbolic arithmetic is highly tokenization-sensitive.
3. **Differential difficulty:** the two precedence parses are not equally solvable (Δ=0.20), confounding any patch that contrasts them.

## The options (pick one as primary; they can be sequenced)
- **A. Run `-Instruct`.** Swap to `meta-llama/Llama-3.1-8B-Instruct`, raise the accuracy floor, re-run G2/G3/G3.5 (~20 min GPU). If it makes C1≈C2≈C0 and surface-robust → clean localization becomes valid AND yields a base-vs-instruct "tuning installs precedence robustness" story; if it's also fragile → a strong generality claim. The pre-registered "second experiment."
- **B. Pivot to a brittleness/composition-asymmetry paper on base Llama.** Characterize C4-vs-C3, the perfect-subexpr-but-failed-compose pattern (C6 vs C1), and surface fragility; add operand-magnitude and addition-precedence controls. No clean localization; no more model runs needed to start.
- **C. Redesign the contrast** so both arms keep the operation *inside* a bracket (avoid the `(…)*C` asymmetry) → equal difficulty → localization may become valid on base Llama. Reworks the Phase 2 templates.

## What I need from you
1. **Recommend a primary path (A/B/C)** and the reasoning, weighing: scientific value, novelty (position against Nikankin 2024 bag-of-heuristics and the decodability-vs-causal-use line), confound risk, and effort.
2. **State the validity gate** that should govern the chosen path (e.g., for A: what acc symmetry / surface-robustness threshold makes localization valid?).
3. **Flag any control that must run first** before committing (e.g., is the surface-fragility finding (C7) strong enough to be a co-headline, or must it be disentangled from the composition asymmetry first?).
4. **If A:** predict the most likely Instruct outcome and what each branch implies for the paper.
5. **If B:** outline the minimal additional controls that make the composition-asymmetry claim publishable (selectivity/baseline analogues), and whether *any* localization (e.g., of the failed-compose step) is salvageable.

Be concrete and decisive. Assume a single 8B model on one A100, a few-week timeline, and a short-paper target.
