# WO#2 follow-up — why the salvage was redesigned (the no-op confound)

**Status:** the `_wo_salvage` implementation in `cells/82_wo_downstream.py` was redesigned after a
multi-agent code review of the first WO#2 pass. This note records the reason so the methodological
history is explicit. (The original brief is `work_order_2_causal_hardening.md`; this supersedes its
§3.1/§10.B intervention design.)

## The confound the first design had

The first salvage tested "is the operand B at the post-bracket `)` causally used?" by **copying C6's
`)` residual into C1's `)`** (C6 = `( 0 + B ) =`, C1 = `( 0 + B ) * C =`).

The alignment guard required C1 and C6 to be **token-identical through `)`**. Under causal attention,
`resid_post[L][')']` is a deterministic function of *only* the tokens at positions ≤ `)` — which are
identical (`( 0 + B )`) in both prompts. Therefore:

> **C6's `)` residual ≡ C1's `)` residual, at every layer.**

So the patch overwrote the target with a copy of itself — an **identity operation**. A flip-rate ≈ 0
was *mathematically guaranteed regardless of whether the model composes*, carrying no causal
information. WO#2's three hardenings (positive control, layer sweep, n-fix) were all correctly
implemented but did not address this deeper issue; in fact the layer sweep just re-measured the
unpatched output at every layer. The original final-position **C4 positive control** also passed
trivially (overwriting the residual one unembed step from the logits) but was **not site-matched** to
the `)` experiment, so it could not license interpreting the `)` null.

Three independent reviewers converged on this; it is the kind of objection a top-venue reviewer would
reject the paper on.

## The fix: operand-corruption denoising at the `)` site

The redesign makes the donor and target **genuinely differ** by corrupting the operand, reusing the
already-validated G4 / `_wo_localize_parse` denoising idiom:

- **Experiment (C1):** clean = `( 0 + B ) * C =`, corrupt = `( 0 + B' ) * C =` (`B'` same #digits).
  The `)` residuals now differ (B vs B'). Patch the **clean** `)` residual into the **corrupted** run
  at `)`; measure recovery of the logit-diff(`F_clean − F_corr`) at the final position, swept over
  layers. Recovery ≈ 0 across the mid-late consumption zone ⇒ restoring B at `)` does not move C1's
  output ⇒ decoded-but-causally-unused.
- **Positive control (C6) — site-matched:** the *same* operand-corruption patch at the *same* `)`
  position, but in C6 where the bracketed value *is* the answer. Recovery should be high (≥ threshold),
  proving a `)`-site patch *can* move the output. High C6 recovery beside ≈0 C1 recovery is the
  airtight contrast; if C6 also fails, the null is uninterpretable (STOP).
- **Net by construction:** `clean_first ≠ corrupt_first` is required and the corrupt run's *unpatched*
  argmax is `corrupt_first`, so an unpatched example can never count as a flip — recovery/flip are
  inherently baseline-netted (fixing the absolute-flip-rate nit).
- **Decodability (§3.7)** still probes B / B·C / C / shuffled-B from the clean C1 `)` residual, but the
  reading states honestly that B·C and C decode low *partly because C is causally future at `)`*, so
  the load-bearing evidence is the experiment-vs-control contrast, not the decodability gap.

Deliverable filenames are unchanged (`salvage_c6_to_c1_{instruct,base}.json`); the JSON now carries
`recovery_by_layer`, `pos_ctrl_recovery_by_layer`, `pos_ctrl_recovery_max`, `recovery_midlate_max`,
the four-target decodability, `n_no_contrast`, and `n_used_ok`.

## Residual risk to watch on the GPU run

The experiment needs `clean_first ≠ corrupt_first` for C1 (its output must depend on B). If the model
*largely ignores* B in C1 — which is itself the hypothesis — many examples become `no_contrast` and
`n_used` could approach the floor. `n_no_contrast` is reported (it is corroborating evidence), and an
`n_used < wo_salvage_min_n` (80) warning fires. From the C1-wrong pools (~292 instruct / ~196 base of
N=400) `n_used ≥ 80` should hold, but verify on the live run.
