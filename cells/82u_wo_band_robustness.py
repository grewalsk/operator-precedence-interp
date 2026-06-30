# ============================================================================
# Phase 6 / WORK ORDER #6 (Tier 1.1) — BAND-ROBUSTNESS OF THE OPERANDS_ONLY ILLUSION.
# ----------------------------------------------------------------------------
# The WO#5 selectivity verdict (OPERANDS_ONLY: B·C decodable at '=' is explained by
# the linear presence of B,C) has ZERO HEADROOM at band (20,49): R2_real≈0.968 vs the
# linear-(B,C) ceiling≈0.968, because B·C is ~97% linear in B,C at that band. A reviewer
# kills it as a band artifact in one sentence. This cell re-runs selectivity at a SECOND
# band where B·C is meaningfully NONLINEAR in the operands, so a genuine product
# representation COULD show a margin:
#   band2 = (2,99): the linear-(B,C) ceiling drops to ≈0.85 (verified on CPU), leaving
#   ≈0.15 of headroom above the operand baseline. The linear-(B,C) baseline is recomputed
#   FRESH at band2 (wo_probe_selectivity builds it from the actual band2 operands).
# Outcomes (both answer the reviewer):
#   OPERANDS_ONLY with headroom available -> the illusion is BAND-ROBUST (strong result).
#   REPRESENTED (R2_real exceeds the fresh baseline by the margin) -> band-SPECIFIC;
#                we narrow the claim.
# In-scope check: report C0/C4 accuracy at band2 (the parts floor) so the band isn't
# dismissed as out-of-capability. Reuses the parameterized _st_capture (no new code path)
# + the validated wo_probe_selectivity; a fresh band2 residual capture is the only GPU cost.
# Band (20,49) stays PRIMARY (WO_BAND, cross-phase comparability) — this is a robustness panel.
# ============================================================================
import json
import numpy as np

assert "_st_capture" in globals() and "wo_build_pairs" in globals() \
    and "wo_probe_selectivity" in globals() and "wo_selectivity_verdict" in globals(), (
    "WO#6 band-robustness needs cells 82l (param _st_capture) + 76 (selectivity helpers).")

WO_BAND2 = tuple(CFG.get("wo_band2", (2, 99)))                 # robustness band (nonlinear B·C).
CFG.setdefault("wo_br_tags", list(CFG.get("wo_steer_tags", ["base", "instruct"])))
CFG.setdefault("wo_br_seed", 919)
_BR_RIDGE = globals().get("WO_FSPROBE_RIDGE", 1.0)
_BR_FOLDS = globals().get("WO_FSPROBE_FOLDS", 5)
WO_PAIRS_B2 = wo_build_pairs(n=WO_N, band=WO_BAND2, seed=WO_SEED)
WO_PAIRS_B2_SHA = wo_stim_hash(WO_PAIRS_B2)
_BR_SITES = [("C1_equals", "resid_equals", ["B_times_C", "B", "C"]),
             ("C1_rparen", "resid_rparen", ["B", "B_times_C"]),
             ("C4_equals", "resid_c4_eq", ["B_times_C"])]
log(f"WO#6 band-robustness: band2={WO_BAND2}, {len(WO_PAIRS_B2)} pairs (sha {WO_PAIRS_B2_SHA[:12]}); "
    f"primary band (20,49) unchanged.")


def _br_run(tag):
    # ---- in-scope: C0/C4 accuracy at band2 (the parts floor); skip if the battery
    #      evaluator isn't present (e.g. a probe-only kernel) so the cell still runs. ----
    acc_c0 = acc_c4 = None
    if "_eval_prompts" in globals() and "wo_run_battery" in globals():
        wo_load_model(tag)
        parts = wo_run_battery(tag, [c for c in WO_CONDITIONS if c[0] in ("C0", "C4")],
                               WO_PAIRS_B2, cache_tag=f"{tag}_b2")
        acc_c0 = parts.get("C0", {}).get("exact_acc")
        acc_c4 = parts.get("C4", {}).get("exact_acc")

    # ---- band2 residual capture (reuses the parameterized _st_capture; cached) ----
    bundle = _st_capture(tag, pairs=WO_PAIRS_B2, ck=f"wo_steer_resid_b2_{tag}")
    items = bundle["items"]
    nL = bundle["n_layers"]
    B = np.array([it["B"] for it in items], dtype=float)
    C = np.array([it["C"] for it in items], dtype=float)

    rows = []
    for site, rkey, targets in _BR_SITES:
        if not items or rkey not in items[0]:
            continue
        R = np.stack([it[rkey] for it in items])
        for tgt in targets:
            y = {"B": B, "C": C, "B_times_C": B * C}[tgt]
            curve = {L: wo_cv_r2(R[:, L, :].astype(np.float32), y, folds=_BR_FOLDS, ridge=_BR_RIDGE)
                     for L in range(nL)}
            cand = [(L, v) for L, v in curve.items() if v is not None]
            bestL = max(cand, key=lambda t: t[1])[0] if cand else (nL // 2)
            row = wo_probe_selectivity(R[:, bestL, :].astype(np.float32), B, C, target=tgt,
                                       folds=_BR_FOLDS, ridge=_BR_RIDGE, seed=int(CFG["wo_br_seed"]))
            v = wo_selectivity_verdict(row["R2_real"], row["R2_control_task"],
                                       row["R2_shuffled"], row["R2_linearBC_baseline"])
            headroom = (None if row["R2_linearBC_baseline"] is None else 1.0 - row["R2_linearBC_baseline"])
            rows.append({"tag": tag, "band": str(WO_BAND2), "site": site, "target": tgt,
                         "layer": int(bestL), "n": len(items),
                         "R2_real": row["R2_real"], "R2_control_task": row["R2_control_task"],
                         "R2_shuffled": row["R2_shuffled"], "R2_linearBC_baseline": row["R2_linearBC_baseline"],
                         "selectivity": row["selectivity"], "baseline_gap": row["baseline_gap"],
                         "headroom_above_baseline": headroom, "verdict": v["label"]})
            log(f"WO#6 band2[{tag}] {site}/{tgt} @L{bestL}: R2_real={_brf(row['R2_real'])} "
                f"linBC={_brf(row['R2_linearBC_baseline'])} gap={_brf(row['baseline_gap'])} "
                f"(headroom {_brf(headroom)}) -> {v['label']}")

    head = next((r for r in rows if r["site"] == "C1_equals" and r["target"] == "B_times_C"), None)
    out = {"tag": tag, "band2": list(WO_BAND2), "primary_band": list(WO_BAND),
           "pairs_sha": WO_PAIRS_B2_SHA, "n_used": bundle["n_used"],
           "acc_C0_band2": acc_c0, "acc_C4_band2": acc_c4,
           "in_scope": (None if acc_c4 is None else bool(acc_c4 >= 0.5 or (acc_c0 or 0) >= 0.5)),
           "headline_C1_equals_product": head, "rows": rows,
           "ridge": _BR_RIDGE, "folds": _BR_FOLDS, "seed": int(CFG["wo_br_seed"]),
           "note": ("Band-robustness panel for the OPERANDS_ONLY illusion. band2 linear-(B,C) ceiling "
                    "≈0.85 (vs ≈0.97 at (20,49)) -> ≈0.15 headroom. OPERANDS_ONLY with headroom available "
                    "= band-robust illusion; REPRESENTED = band-specific (narrow the claim). The linear-(B,C) "
                    "baseline is recomputed fresh at band2 from the actual operands.")}
    wo_save_result(f"band_robustness_{tag}.json", json.dumps(out, indent=2, default=str))
    return out


def _brf(x):
    return "n/a" if x is None else f"{float(x):.3f}"


WO_BAND_ROBUST = {}
for _tag in CFG["wo_br_tags"]:
    WO_BAND_ROBUST[_tag] = _br_run(_tag)

_br_rows = []
for _tag in CFG["wo_br_tags"]:
    _br_rows.extend(WO_BAND_ROBUST[_tag]["rows"])
wo_save_result("band_robustness_summary.csv",
               wo_battery_csv(_br_rows, ["tag", "band", "site", "target", "layer", "n",
                                         "R2_real", "R2_control_task", "R2_shuffled",
                                         "R2_linearBC_baseline", "selectivity", "baseline_gap",
                                         "headroom_above_baseline", "verdict"]))

print("\n========== WO#6 (Tier 1.1) — BAND-ROBUSTNESS OF THE OPERANDS_ONLY ILLUSION ==========")
print(f"  band2={WO_BAND2} (linear-(B,C) ceiling ~0.85; ~0.15 headroom)  vs primary band {WO_BAND} (~0.97; ~0 headroom)")
for _tag in CFG["wo_br_tags"]:
    o = WO_BAND_ROBUST[_tag]; h = o["headline_C1_equals_product"]
    _scope = (f"in-scope C0={_brf(o['acc_C0_band2'])} C4={_brf(o['acc_C4_band2'])}"
              if o["acc_C4_band2"] is not None else "in-scope: (battery skipped)")
    print(f"\n  [{_tag}]  {_scope}")
    if h:
        print(f"     C1 '=' product @L{h['layer']}: R2_real={_brf(h['R2_real'])}  "
              f"linear-(B,C) baseline={_brf(h['R2_linearBC_baseline'])}  gap={_brf(h['baseline_gap'])}  "
              f"(headroom available {_brf(h['headroom_above_baseline'])})  ->  {h['verdict']}")
        if h["verdict"] == "OPERANDS_ONLY":
            print("        => BAND-ROBUST illusion: even with ~0.15 headroom, the product adds no margin "
                  "over the operand baseline. The decodability is operand-explained, not band-specific.")
        elif h["verdict"] == "REPRESENTED":
            print("        => BAND-SPECIFIC: at this band the product DOES exceed the operand baseline. "
                  "Narrow the claim to band (20,49) where it's operand-explained.")
print("=====================================================================================")
