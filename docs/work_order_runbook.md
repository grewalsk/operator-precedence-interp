# Work-order runbook — Instruct re-run + surface/compose disentangling (Phase 6)

This documents the **Phase 6 / Work Order** section added to the operator-precedence
feasibility notebook: the `-Instruct` re-run, the base 2×2 surface/compose
completion, the six validity gates, the branch decision, and the first downstream
artifact. It implements the work order in full.

> **Status:** harness **built and CPU-verified**; the gated numbers require one
> `Run All` on an A100 (no GPU in the authoring environment). Everything that
> governs the publishable decision — the six gates (§7), the branch tree (§8), the
> 2×2 verdict (§6), the metrics, and the decodability probe (§10.B) — is **pure
> Python and unit-tested locally** (`tests/test_wo_logic.py`, all pass), so the
> decision logic is correct *before* any GPU time is spent. The model-dependent
> numbers are produced by the notebook itself.

---

## What was added

Eight cells (`cells/76`–`83`, assembled by `build_notebook.py` into
`operator_precedence_phases_0_5.ipynb`), inserted **after** the validated Phase 5
(G4) and **before** the summary/dashboard, so a fresh `Run All` executes the
validated Phases 0–5 first and then the work order. The validated Phases 0–5 and
the `G0..G4` gate ledger are **not touched** (WO gates are written under `WO_*`
keys).

| cell | step | work order |
|---|---|---|
| `76_wo_logic.py` | **Pure CPU logic** — pairs, conditions C0–C8 + Branch-B (A1/A2/D1), metrics, the six gates, branch tree, 2×2 verdict, recovery math, `wo_cv_r2` decodability, CSV/MD builders. Inline self-test runs every notebook execution. | §5–§10 |
| `77_wo_setup.py` | Model registry, memory-safe `wo_load_model`, **tag-namespaced** eval (`wo_{tag}_{key}` caches so base + Instruct never collide), `wo_assert_parity`, `wo_save_result`. Sets greedy `K=8` (§5). | §11 |
| `78_wo_base_2x2.py` | **Step 0** instrument sanity (G4 PASS check) + **Step 1** base 2×2: runs the missing no-space `(B*C)=` cell (C8) + the paired base battery; emits the §6 verdict → `base_2x2.csv`. | §6 |
| `79_wo_instruct_battery.py` | **Steps 2–3**: Instruct C1/C2 parity + the C0–C8 battery (bare-continuation primary; degeneracy guard + chat fallback with re-parity & single-BOS check) → `instruct_battery.csv`. | §3, §11 |
| `80_wo_gates.py` | **Step 4**: the six validity gates → `gate_evaluation.json`. | §7 |
| `81_wo_branch.py` | **Step 5a**: branch selection + `decision_record.md`. | §8 |
| `82_wo_downstream.py` | **Step 5b**: the selected branch's first artifact — C1/C2 localization (§9/§10) **or** Branch-B controls + C6→C1 salvage (§9.B/§10.B). | §9, §10 |
| `83_wo_repro.py` | `repro.txt` + deliverables manifest. | §12, §13.7 |

## How to run (A100)

1. Open `operator_precedence_phases_0_5.ipynb` (Colab/A100), set `HF_TOKEN`
   (gated Llama repos), **Run All**. Phases 0–5 run first (model load, G2/G3/G4),
   then Phase 6.
2. Everything is cached per `(model-tag, condition)` under `ART`
   (`/content/drive/MyDrive/opprec_interp` on Colab, else `~/opprec_interp_artifacts`);
   a disconnect resumes from disk. The two model runs total well under one GPU-hour.
3. Deliverables land in `ART/results/` (mirrored to a repo-local `./results/` when
   the notebook runs from the checked-out repo). Drop them into `results/` for the
   write-up, as is already done for the PNG figures.

**Knobs** (`CFG`, all optional): `wo_run_chat_secondary` (default off — the §11
chat fallback), `wo_localize_sample` (default 8 — localization examples per parse),
`wo_salvage_n` (default 128 — salvage decodability sample), `wo_force_branch`
(override the branch to exercise a specific downstream path).

## Deliverables (§12)

`base_2x2.csv` · `instruct_battery.csv` · `gate_evaluation.json` ·
`decision_record.md` · the branch artifact (`localization_sites.csv` +
`localization_*.png`, **or** `branchb_controls.csv` + `salvage_c6_to_c1.json`) ·
`repro.txt`.

## The metric subtleties (read before trusting Step 5b)

- **C1/C2 share the same answer** (B·C), so a clean-vs-corrupted *answer* logit-diff
  between the two parses is degenerate, and on a *repaired* model both parses
  already succeed. The localization therefore uses the **G4 idiom within each
  parse** via operand corruption (`C` → `C'`, same digit count → equal length),
  scores recovery of the **first answer token** (§10), and **differences the two
  maps only over token-aligned positions** (the shared `( 0 + B` prefix and `=`);
  the divergent middle holds different tokens and is excluded. The instrument is
  **verified on a C0 control** (must reproduce Stolfo mid-late final-token
  localization) before C1/C2 is trusted.
- **§10.B salvage decodability** uses a **held-out k-fold dual-ridge CV-R²**
  (`wo_cv_r2`), *not* in-sample least squares — with `n ≪ d_model` an in-sample fit
  interpolates to R²=1.0 on pure noise. The readout target is the **correct
  product's** first token, and "causally used" is the scale-free **argmax-flip
  rate**, not an arbitrary logit cutoff. `decodable ∧ ¬used ⇒ decodable-but-unused`.

## Predicted outcome (§8)

**PARTIAL_REPAIR** is the work order's most-likely prediction: Instruct/chat math
SFT repairs composition (C1 → ~0.80–0.90, corr → ~0.7+) but not the low-level
no-space tokenization collapse (C7 stays ~0.2–0.5 < 0.70), so all hard gates pass
except `G_surface` → localization valid **conditional on spaced format**, run on
Instruct. The harness implements all three branches; the runtime selects from the
actual Instruct numbers.

## Verification status

- Pure decision logic (gates, branch, 2×2, metrics, recovery, `wo_cv_r2`):
  **unit-tested locally, all pass** (`python3 tests/test_wo_logic.py`). Includes a
  regression test that `wo_cv_r2` is ~0.03 on pure noise (not 1.0) and ~0.96 on a
  prominently-encoded signal.
- All cells `py_compile`-clean; notebook JSON round-trips (27 cells).
- Adversarially reviewed against the spec (8 lenses); all findings fixed.
- **Not** verifiable here: the GPU forward-pass numbers themselves — produced by
  the A100 `Run All`.
