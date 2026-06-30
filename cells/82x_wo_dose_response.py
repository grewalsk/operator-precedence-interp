# ============================================================================
# Phase 6 / WORK ORDER #6 (Tier 1.2) — INJECT DOSE-RESPONSE POSITIVE CONTROL (GPU; REUSES 82p).
# ----------------------------------------------------------------------------
# The headline is the CAUSAL NULL: probe-direction inject at the C1 '=' site does not
# move the output. A reviewer's first objection is "weak INSTRUMENT, not dead DIRECTION."
# This cell answers it with a DOSE-RESPONSE positive control: inject the SAME instrument
#   delta = k * (target_coord - w_hat·resid) * w_hat        (write P' along the decode axis)
# at doses k in {1,2,4,8} and measure, with the FULL-PRODUCT metric Δ(P'-true) (the only
# metric that survived the Llama leading-token chunking, §5), at TWO sites:
#   C4 '=' (REFERENCE) — the surface the model succeeds on: does the inject move it, and
#                        MORE with dose (dynamic range)?
#   C1 '=' (CLAIM)     — the failing surface: flat at every dose -> dead direction.
# Both surfaces use a CORRECT WITHIN-SURFACE paired baseline (the committed 82p inject_C1
# compared a C1-prompt score against a C4-prompt clean baseline — a cross-surface bug that
# manufactured a spurious +1.99; fixed here).
#
# Verdict (wo_dose_response_verdict; the decision is the CI, not a fixed margin):
#   DEAD_DIRECTION_CONFIRMED — ref moves monotonically with dose + crosses threshold while
#                              C1 is flat at every k -> the causal null is a real dead
#                              direction. The load-bearing positive control for the headline.
#   WEAK_INSTRUMENT          — NEITHER site moves -> no dynamic range on this inject axis;
#                              the C1 null is uninterpretable from this control (committed
#                              data: the answer-site swap/inject is inert even at C4 — the
#                              product is re-derived from the operands, Stolfo). Then the
#                              positive control must come from the OPERAND route (82r/82s).
#   DIRECTION_IS_CAUSAL      — C1 itself moves -> refutes dormancy.
# Reuses the cached residuals + 82p's _fp_logprob/_fp_generate (no re-capture). Checkpointed.
# ============================================================================
import json
import numpy as np
import torch

assert "_st_capture" in globals() and "_st_split" in globals() and "_cal_derange_perm" in globals() \
    and "_st_mk_add_hook" in globals() and "_fp_logprob" in globals() and "_fp_generate" in globals() \
    and "wo_fit_ridge_probe" in globals() and "wo_inject_to_target" in globals() \
    and "wo_probe_coord_for_value" in globals() and "wo_paired_delta_ci" in globals() \
    and "wo_parse_int" in globals() and "wo_dose_response_verdict" in globals(), (
    "WO#6 82x needs cells 82l + 82n + 82p (for _fp_logprob/_fp_generate/_st_mk_add_hook) and "
    "the cell-76 wo_dose_response_verdict. Run cells 76, 82l, 82n, 82p first.")

CFG.setdefault("wo_dr_tags", list(CFG.get("wo_fp_tags", ["base", "instruct"])))
CFG.setdefault("wo_dr_k_grid", [1, 2, 4, 8])              # injection doses (k=1 is the WO#5 minimal edit).
CFG.setdefault("wo_dr_gen_sample", 40)                   # #items for the greedy-decode emit-P' companion.
CFG.setdefault("wo_dr_gen_k", int(globals().get("WO_MAX_NEW_TOKENS", 8)))
CFG.setdefault("wo_dr_recover_thr", float(globals().get("WO_STEER_RECOVER_THR", 0.5)))
CFG.setdefault("wo_dr_null_tol", float(globals().get("WO_STEER_NULL_TOL", 0.25)))
CFG.setdefault("wo_dr_min_rise", 0.1)
_DR_SEED = int(CFG.get("wo_steer_seed", 707))
_DR_STRIDE = int(CFG.get("wo_steer_layer_stride", 2))


@torch.no_grad()
def _dr_run(tag):
    ck = f"wo_dr_{tag}"
    prog = load_json(ck) if has_artifact(ck, "json") else {}
    bundle = _st_capture(tag)                                  # REUSED (no re-capture)
    items = bundle["items"]
    nL = bundle["n_layers"]
    train_idx, test_idx = _st_split(len(items), _DR_SEED)
    K = [int(k) for k in CFG["wo_dr_k_grid"]]

    def _stack(idxs, rkey, L):
        return np.stack([items[i][rkey][L].astype(np.float32) for i in idxs])
    yP_tr = np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float)

    # candidate inject layers: the C4 decodability peak (where the product is most READABLE)
    # AND a band of LATE layers (where Stolfo says the answer is WRITTEN — the decodability
    # peak and the causal layer differ; that gap is the whole point). We pick the layer with
    # the most REFERENCE dynamic range as the instrument layer, then lock C1 to it.
    L_grid = sorted(set(range(0, nL, _DR_STRIDE)) | {nL - 1})

    def _c4_r2(L):
        r = wo_cv_r2(_stack(train_idx, "resid_c4_eq", L), yP_tr, folds=5, ridge=1.0)
        return -1e9 if r is None else float(r)        # explicit: R^2==0.0 is NOT falsy
    reg_c4 = max(L_grid, key=_c4_r2)
    late_band = sorted(set([int(round(f * (nL - 1))) for f in (0.6, 0.75, 0.85)]) | {nL - 1, reg_c4})
    log(f"WO#6 dose[{tag}]: candidate inject layers {late_band} (C4 reg-peak {reg_c4}); doses k={K}.")

    perm = _cal_derange_perm(len(test_idx), _DR_SEED + 3)
    pref = [int(items[test_idx[perm[j]]]["B"] * items[test_idx[perm[j]]]["C"]) for j in range(len(test_idx))]
    true = [int(items[i]["B"] * items[i]["C"]) for i in test_idx]
    sites = {"C4": ("resid_c4_eq", "eq4", "tok4"), "C1": ("resid_equals", "equals", "tok1")}
    gsamp = list(range(min(int(CFG["wo_dr_gen_sample"]), len(test_idx))))

    # CORRECT within-surface clean baselines: logprob_S(P') - logprob_S(true), per surface.
    if "clean" not in prog:
        cl = {}
        for s, (_rk, _pk, _tk) in sites.items():
            cl[s] = [_fp_logprob(items[i][_tk], pref[j]) - _fp_logprob(items[i][_tk], true[j])
                     for j, i in enumerate(test_idx)]
        prog["clean"] = cl
        save_json(ck, prog)
    clean = prog["clean"]

    def _fit(rkey, L):
        return wo_fit_ridge_probe(_stack(train_idx, rkey, L), yP_tr)

    def _dose_cell(surface, L, k, fit):
        """mean full-product Δ(P'-true) and paired CI at dose k (within-surface baseline)."""
        rkey, poskey, tokkey = sites[surface]
        d = []
        emit = []
        for j, i in enumerate(test_idx):
            clean_r = items[i][rkey][L].astype(np.float32)
            tgt = wo_probe_coord_for_value(fit, pref[j])
            dv = k * (wo_inject_to_target(clean_r, fit["direction"], tgt) - clean_r)
            hook = _st_mk_add_hook(torch.tensor(dv, device=model.cfg.device), items[i][poskey])
            sc = _fp_logprob(items[i][tokkey], pref[j], L, hook) - _fp_logprob(items[i][tokkey], true[j], L, hook)
            d.append(sc - clean[surface][j])
            if j in gsamp:
                emit.append(1.0 if wo_parse_int(_fp_generate(items[i][tokkey], L, hook,
                                                             int(CFG["wo_dr_gen_k"]))) == pref[j] else 0.0)
        d = np.array(d, dtype=float)
        lo, hi = wo_paired_delta_ci(d, np.zeros(len(d)), n_boot=2000, seed=_DR_SEED)
        return {"surface": surface, "layer": int(L), "k": int(k), "mean_delta": float(d.mean()),
                "ci": [lo, hi], "emit_Pprime_rate": (float(np.mean(emit)) if emit else None),
                "n": len(d), "gen_n": len(emit)}

    # ---- C4 REFERENCE dose-response across candidate layers; pick the layer with most range ----
    refcells = prog.get("ref", {})
    for L in late_band:
        fit = _fit("resid_c4_eq", L)
        if fit is None:
            continue
        for k in K:
            key = f"L{L}_k{k}"
            if key in refcells:
                continue
            refcells[key] = _dose_cell("C4", L, k, fit)
            prog["ref"] = refcells
            save_json(ck, prog)
            log(f"WO#6 dose[{tag}] C4 ref @L{L} k={k}: Δ(P'-true)={refcells[key]['mean_delta']:+.3f} "
                f"emit-P'={refcells[key]['emit_Pprime_rate']}")

    def _topk_delta(L):
        c = refcells.get(f"L{L}_k{K[-1]}")
        return (c["mean_delta"] if c and c["mean_delta"] is not None else -1e9)
    inst_L = max(late_band, key=_topk_delta)                  # the instrument layer (most ref range)
    log(f"WO#6 dose[{tag}]: instrument layer L*={inst_L} (max top-dose C4 reference Δ).")

    # ---- C1 CLAIM dose-response at the SAME instrument layer ----
    claimcells = prog.get("claim", {})
    fit_c1 = _fit("resid_equals", inst_L)
    if fit_c1 is not None:
        for k in K:
            key = f"L{inst_L}_k{k}"
            if key in claimcells:
                continue
            claimcells[key] = _dose_cell("C1", inst_L, k, fit_c1)
            prog["claim"] = claimcells
            save_json(ck, prog)
            log(f"WO#6 dose[{tag}] C1 claim @L{inst_L} k={k}: Δ(P'-true)={claimcells[key]['mean_delta']:+.3f} "
                f"emit-P'={claimcells[key]['emit_Pprime_rate']}")

    ref_row = [refcells.get(f"L{inst_L}_k{k}") for k in K]
    claim_row = [claimcells.get(f"L{inst_L}_k{k}") for k in K]
    ref_d = [(c["mean_delta"] if c else None) for c in ref_row]
    ref_ci = [(c["ci"] if c else None) for c in ref_row]
    claim_d = [(c["mean_delta"] if c else None) for c in claim_row]
    claim_ci = [(c["ci"] if c else None) for c in claim_row]
    verdict = wo_dose_response_verdict(K, ref_d, ref_ci, claim_d, claim_ci,
                                       recover_thr=float(CFG["wo_dr_recover_thr"]),
                                       null_tol=float(CFG["wo_dr_null_tol"]),
                                       min_rise=float(CFG["wo_dr_min_rise"]))
    out = {"tag": tag, "experiment": "WO6_T1.2_dose_response", "reused_capture": True,
           "k_grid": K, "instrument_layer": int(inst_L), "reg_c4_layer": int(reg_c4),
           "candidate_layers": late_band, "ref_site": "C4_equals", "claim_site": "C1_equals",
           "ref_dose": {f"k{K[i]}": ref_row[i] for i in range(len(K))},
           "claim_dose": {f"k{K[i]}": claim_row[i] for i in range(len(K))},
           "verdict": verdict, "pair_sha": bundle.get("pair_sha"),
           "note": ("Inject dose-response positive control for the C1 '=' causal null. Full-product "
                    "metric Δ(P'-true) with a CORRECT within-surface paired baseline. DEAD_DIRECTION_"
                    "CONFIRMED needs the C4 reference to move monotonically with dose; WEAK_INSTRUMENT "
                    "(no range anywhere) means the answer-site axis is inert even where the model "
                    "succeeds (Stolfo re-derivation) and the positive control must come from the "
                    "operand route. The CI is the result.")}
    wo_save_result(f"dose_response_{tag}.json", json.dumps(wo_jsonsafe(out), indent=2))
    return out


WO_DOSE = {}
for _tag in CFG["wo_dr_tags"]:
    WO_DOSE[_tag] = _dr_run(_tag)

_dr_rows = []
for _tag, o in WO_DOSE.items():
    if o is None:
        continue
    for site, key in (("C4_ref", "ref_dose"), ("C1_claim", "claim_dose")):
        for kk, c in o[key].items():
            if c is None:
                continue
            _dr_rows.append({"tag": _tag, "site": site, "layer": c["layer"], "k": c["k"],
                             "mean_delta": c["mean_delta"], "ci_lo": c["ci"][0], "ci_hi": c["ci"][1],
                             "emit_Pprime_rate": c["emit_Pprime_rate"], "verdict": o["verdict"]["label"]})
if _dr_rows:
    wo_save_result("dose_response_summary.csv",
                   wo_battery_csv(_dr_rows, ["tag", "site", "layer", "k", "mean_delta",
                                             "ci_lo", "ci_hi", "emit_Pprime_rate", "verdict"]))

print("\n========== WO#6 (Tier 1.2) — INJECT DOSE-RESPONSE: is the C1 null a DEAD DIRECTION or a WEAK INSTRUMENT? ==========")
print("  inject k*(P' - w_hat·resid)*w_hat at k in {1,2,4,8}; full-product Δ(P'-true); within-surface baseline.")
for _tag, o in WO_DOSE.items():
    if o is None:
        continue
    v = o["verdict"]; L = o["instrument_layer"]; K = o["k_grid"]
    print(f"\n  [{_tag}]  instrument layer L*={L}")
    print(f"     {'k':>3} | {'C4 ref Δ(P-true)':>18} {'95%CI':>20} | {'C1 claim Δ':>14} {'95%CI':>20}")
    for i, k in enumerate(K):
        rc = o["ref_dose"].get(f"k{k}"); cc = o["claim_dose"].get(f"k{k}")
        def _f(x):
            return "n/a" if x is None else f"{x:+.3f}"
        def _ci(c):
            return "n/a" if not c else f"[{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]"
        print(f"     {k:>3} | {_f(rc['mean_delta'] if rc else None):>18} {_ci(rc):>20} | "
              f"{_f(cc['mean_delta'] if cc else None):>14} {_ci(cc):>20}")
    print(f"     ref monotone={v['ref_monotone']}  ref has-range={v['ref_has_range']}  C1 flat={v['claim_flat']}")
    print(f"     >>> {v['label']}: {v['reason']}")
print("=================================================================================================================")
