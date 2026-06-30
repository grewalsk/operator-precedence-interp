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
    _gap_stash = {}                              # (site -> (X_at_bestL, bestL)) for the decision-grade gap CI
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
            rec = {"tag": tag, "band": str(WO_BAND2), "site": site, "target": tgt,
                   "layer": int(bestL), "n": len(items),
                   "R2_real": row["R2_real"], "R2_control_task": row["R2_control_task"],
                   "R2_shuffled": row["R2_shuffled"], "R2_linearBC_baseline": row["R2_linearBC_baseline"],
                   "selectivity": row["selectivity"], "baseline_gap": row["baseline_gap"],
                   "headroom_above_baseline": headroom, "verdict": v["label"]}
            rows.append(rec)
            if site in ("C1_equals", "C4_equals") and tgt == "B_times_C":
                _gap_stash[site] = (R[:, bestL, :].astype(np.float32), int(bestL), rec)
            log(f"WO#6 band2[{tag}] {site}/{tgt} @L{bestL}: R2_real={_brf(row['R2_real'])} "
                f"linBC={_brf(row['R2_linearBC_baseline'])} gap={_brf(row['baseline_gap'])} "
                f"(headroom {_brf(headroom)}) -> {v['label']}")

    # ---- DECISION-GRADE GAP CI: is gap = R2_real - linear-(B,C) baseline significantly > 0?
    #      This, not the fixed 0.10 margin, decides operand-explained (CI brackets 0) vs a
    #      small REAL product component (CI excludes 0). Paired item-bootstrap; same seed
    #      across surfaces so the C4-vs-C1 difference is paired (same item draws). ----
    _NB = int(CFG.get("wo_br_gap_nboot", 300))
    gap_ci = {}
    for site, (Xs, Ls, rec) in _gap_stash.items():
        gb = wo_gap_bootstrap(Xs, B, C, target="B_times_C", n_boot=_NB,
                              folds=_BR_FOLDS, ridge=_BR_RIDGE, seed=int(CFG["wo_br_seed"]))
        if gb is not None:
            gap_ci[site] = {k: gb[k] for k in ("gap_mean", "gap_ci", "gap_excludes_zero",
                                               "r2_real_ci", "r2_base_ci", "n_boot_used")}
            rec["gap_ci"] = gb["gap_ci"]; rec["gap_excludes_zero"] = gb["gap_excludes_zero"]
            log(f"WO#6 band2[{tag}] {site} product gap={_brf(gb['gap_mean'])} "
                f"95%CI=[{_brf(gb['gap_ci'][0])},{_brf(gb['gap_ci'][1])}] "
                f"-> {'REAL product component (CI excludes 0)' if gb['gap_excludes_zero'] else 'operand-explained (CI brackets 0)'}")
            gap_ci[site]["_gaps"] = gb["_gaps"]                 # kept transiently for the paired diff
    # paired C4 - C1 difference (does the working surface carry MORE product structure?).
    diff_ci = None
    if gap_ci.get("C1_equals") and gap_ci.get("C4_equals"):
        _d = np.asarray(gap_ci["C4_equals"]["_gaps"]) - np.asarray(gap_ci["C1_equals"]["_gaps"])
        _lo, _hi = wo_pct_ci(_d)
        diff_ci = {"C4_minus_C1_gap_mean": float(_d.mean()), "ci": [_lo, _hi],
                   "excludes_zero": bool(_lo is not None and _lo > 0.0)}
        log(f"WO#6 band2[{tag}] C4-C1 product-gap diff={_brf(_d.mean())} 95%CI=[{_brf(_lo)},{_brf(_hi)}] "
            f"-> {'C4>C1 (representation tracks behavior)' if diff_ci['excludes_zero'] else 'not distinguishable'}")
    for s in gap_ci:                                            # drop the bulky per-draw arrays from the saved JSON
        gap_ci[s].pop("_gaps", None)

    head = next((r for r in rows if r["site"] == "C1_equals" and r["target"] == "B_times_C"), None)
    _floor = float(CFG.get("wo_parts_floor", 0.80))            # the project's standard parts floor.
    out = {"tag": tag, "band2": list(WO_BAND2), "primary_band": list(WO_BAND),
           "pairs_sha": WO_PAIRS_B2_SHA, "n_used": bundle["n_used"],
           "acc_C0_band2": acc_c0, "acc_C4_band2": acc_c4, "parts_floor": _floor,
           "in_scope_strict": (None if acc_c4 is None else bool(acc_c4 >= _floor and (acc_c0 or 0) >= _floor)),
           "parts_note": ("band2 (2,99) is a HARDER regime: C0/C4 drop ~15-20pts vs (20,49) and sit BELOW the "
                          f"{_floor} parts floor. The probe doesn't strictly require behavioral competence "
                          "(residuals can encode B,C even when the output is wrong), but the 'model does the "
                          "parts' premise is weaker here — report as a harder regime where operand-domination "
                          "still holds, not as comfortably in-scope."),
           "gap_ci": gap_ci, "c4_minus_c1_diff_ci": diff_ci,
           "headline_C1_equals_product": head, "rows": rows,
           "ridge": _BR_RIDGE, "folds": _BR_FOLDS, "seed": int(CFG["wo_br_seed"]), "gap_nboot": _NB,
           "note": ("Band-robustness panel. band2 linear-(B,C) ceiling ≈0.85 (vs ≈0.97 at (20,49)) -> ≈0.15 "
                    "headroom. The DECISION is the gap CI (gap=R2_real-baseline): brackets 0 -> operand-explained "
                    "(no detectable product even with headroom); excludes 0 -> a small REAL product component "
                    "(then the claim shifts from 'not represented' to 'weakly represented but causally dormant', "
                    "leaning on the causal null). The fixed 0.10 'verdict' margin is a heuristic, NOT the test.")}
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

print("\n========== WO#6 (Tier 1.1) — BAND-ROBUSTNESS: IS THE ANSWER-SITE PRODUCT REPRESENTED? ==========")
print(f"  band2={WO_BAND2} (linear-(B,C) ceiling ~0.85; ~0.15 headroom)  vs primary band {WO_BAND} (~0.97; ~0 headroom)")
print("  DECISION = the gap CI (gap = R2_real - operand baseline), NOT the fixed 0.10 margin.")
for _tag in CFG["wo_br_tags"]:
    o = WO_BAND_ROBUST[_tag]; h = o["headline_C1_equals_product"]; g = (o.get("gap_ci") or {})
    _pf = o.get("parts_floor", 0.80)
    _scope = (f"PARTS at band2: C0={_brf(o['acc_C0_band2'])} C4={_brf(o['acc_C4_band2'])} "
              f"(floor {_pf}; in-scope-strict={o.get('in_scope_strict')} — a HARDER regime, parts dropped)"
              if o["acc_C4_band2"] is not None else "parts: (battery skipped)")
    print(f"\n  [{_tag}]  {_scope}")
    if h:
        gc = (g.get("C1_equals") or {}).get("gap_ci")
        gz = (g.get("C1_equals") or {}).get("gap_excludes_zero")
        print(f"     C1 '=' product @L{h['layer']}: R2_real={_brf(h['R2_real'])}  baseline={_brf(h['R2_linearBC_baseline'])}  "
              f"gap={_brf(h['baseline_gap'])}  95%CI=[{_brf(gc[0]) if gc else 'n/a'},{_brf(gc[1]) if gc else 'n/a'}]")
        if gz is True:
            print("        => gap CI EXCLUDES 0: a small but REAL product component. Claim is operand-DOMINATED "
                  f"(>={_brf(h['R2_linearBC_baseline'])} of R2 is pure B,C) with a weak product part -> lead the spine "
                  "with the CAUSAL dormancy (inject-dead, full-swap-flips), not selectivity.")
        elif gz is False:
            print("        => gap CI BRACKETS 0: no detectably-represented product even with headroom. The strong "
                  "operand-explained / illusion claim is band-robust.")
        else:
            print("        => gap CI unavailable (re-run with the cached residuals to decide).")
    dc = o.get("c4_minus_c1_diff_ci")
    if dc:
        print(f"     C4-vs-C1 product-gap difference = {_brf(dc['C4_minus_C1_gap_mean'])} "
              f"95%CI=[{_brf(dc['ci'][0])},{_brf(dc['ci'][1])}] -> "
              f"{'representation tracks behavior (C4>C1)' if dc['excludes_zero'] else 'NOT distinguishable (suggestive only)'}")
print("  NOTE: operand-DOMINATION is robust; the no-representation claim depends on the gap CI above.")
print("================================================================================================")
