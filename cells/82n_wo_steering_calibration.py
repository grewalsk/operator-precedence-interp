# ============================================================================
# Phase 6 / WORK ORDER #5.1 — STEERING-INSTRUMENT CALIBRATION (GPU; REUSES WO#5).
# ----------------------------------------------------------------------------
# WO#5 came back INCONCLUSIVE on both tags because the C4 positive reference did
# not move (Δ_C4 ~ 0). Two explanations were inseparable from that run: (i) the
# minimal rank-1 probe-direction edit is causally too weak / operand-aligned (a
# DEAD INSTRUMENT), or (ii) the product genuinely is not steerable there (a real
# null). This cell disambiguates by climbing a ladder of ever-more-causal edits at
# the C4 '=' site (where the product IS used and the model succeeds), reusing the
# cached WO#5 residual capture (wo_steer_resid_{tag}.pkl) — NO re-capture:
#   1. HOOK SANITY (zero-ablation): zero the whole C4 '=' residual — the GT logit
#      MUST collapse, else the hook/site/metric is broken.
#   2. FULL-RESIDUAL SWAP (guaranteed-causal positive control): replace the whole
#      C4 '=' residual with a DONOR residual from a different pair (product P') — does
#      the model now emit P'? This proves the site+metric can be driven at all.
#   3. MAGNITUDE SWEEP: scale the probe-direction counterfactual inject of P' by
#      k ∈ {1,2,4,8,16,32} until P''s logit crosses recover_thr; record k* and the
#      relative edit size ||δ||/||resid|| (settles the WO#5 magnitude question).
#   4. C1 RE-EVAL @k*: if calibrated, inject a counterfactual product at C1's '='
#      at k* — can the C1 answer site be driven once the edit is large enough?
# Verdict (pure, tested) wo_steer_calibration_verdict: INSTRUMENT_BROKEN /
# METRIC_OR_SITE_SUSPECT / CALIBRATED@k / DEAD_DIRECTION.
#
# Deliverable: causal_steering_calibration_{base,instruct}.json +
# causal_steering_calibration_summary.csv. Checkpointed per (tag, stage), resumable.
# Cheap: ~2k forward passes/tag at one layer each — far under the WO#5 full sweep.
# ============================================================================
import json
import numpy as np
import torch

assert "_st_capture" in globals() and "_st_split" in globals() and "_st_derange" in globals() \
    and "_st_first_tok_id" in globals() and "wo_steer_calibration_verdict" in globals() \
    and "_wo_mk_patch_hook" in globals() and "_st_mk_add_hook" in globals() \
    and "wo_fit_ridge_probe" in globals() and "wo_inject_to_target" in globals(), (
    "WO#5.1 calibration needs cell 82l (capture/split/derange/hooks) + cell 82 "
    "(_wo_mk_patch_hook) + cell-76 WO#5/5.1 helpers. Run them first.")

CFG.setdefault("wo_cal_tags", list(CFG.get("wo_steer_tags", ["base", "instruct"])))
CFG.setdefault("wo_cal_k_grid", [1, 2, 4, 8, 16, 32])     # injection magnitude multipliers.
CFG.setdefault("wo_cal_zeroabl_sample", 40)               # #items for the cheap zero-ablation sanity.
CFG.setdefault("wo_cal_recover_thr", float(CFG.get("wo_steer_recover_thr", 0.5)))
CFG.setdefault("wo_cal_zeroabl_floor", float(globals().get("WO_STEER_ZEROABL_FLOOR", 1.0)))
CFG.setdefault("wo_cal_nboot", int(CFG.get("wo_steer_nboot", 10000)))
_CAL_SEED = int(CFG.get("wo_steer_seed", 707))
_CAL_STRIDE = int(CFG.get("wo_steer_layer_stride", 2))


@torch.no_grad()
def _cal_logits(tokens_list, layer=None, hook=None):
    """Final-position logit row (on device) for a clean (hook=None) or hooked pass."""
    tok = torch.tensor([tokens_list], device=model.cfg.device, dtype=torch.long)
    if hook is None:
        return model(tok)[0, -1]
    return model.run_with_hooks(tok, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)])[0, -1]


def _cal_derange_perm(n, seed):
    """Permutation of range(n) with NO fixed point (each test item gets a DIFFERENT
    donor pair). Deterministic given seed."""
    rng = np.random.default_rng(int(seed))
    for _ in range(16):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p
    return (np.arange(n) + 1) % n


@torch.no_grad()
def _cal_run(tag):
    ck = f"wo_cal_{tag}"
    prog = load_json(ck) if has_artifact(ck, "json") else {}
    bundle = _st_capture(tag)                                   # REUSED from WO#5 (no re-capture)
    items = bundle["items"]
    nL = bundle["n_layers"]
    n = len(items)
    train_idx, test_idx = _st_split(n, _CAL_SEED)
    L_grid = sorted(set(range(0, nL, _CAL_STRIDE)) | {nL - 1})

    def _stack(idxs, rkey, L):
        return np.stack([items[i][rkey][L].astype(np.float32) for i in idxs])

    def _reg_layer(rkey):
        best, bestL = None, None
        yP = np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float)
        for L in L_grid:
            r = wo_cv_r2(_stack(train_idx, rkey, L), yP, folds=5, ridge=1.0)
            if r is not None and (best is None or r > best):
                best, bestL = r, int(L)
        return (bestL if bestL is not None else int(L_grid[len(L_grid) // 2])), best

    regL_c4, regR2_c4 = _reg_layer("resid_c4_eq")              # C4 '=' product layer (pre-registered)
    regL_c1, regR2_c1 = _reg_layer("resid_equals")            # C1 '=' product layer (pre-registered)

    # per-test-item donor pairs (deranged) -> counterfactual product P' + its token.
    perm = _cal_derange_perm(len(test_idx), _CAL_SEED + 3)
    pref_vals = [int(items[test_idx[perm[j]]]["B"] * items[test_idx[perm[j]]]["C"]) for j in range(len(test_idx))]
    pref_ids = [_st_first_tok_id(v) for v in pref_vals]
    gt_ids = [items[i]["gt_id"] for i in test_idx]
    eq4 = [items[i]["eq4"] for i in test_idx]
    d_model = items[0]["resid_c4_eq"].shape[1]

    # ---- clean C4 baselines (one forward/item; reused by stages 1-3) ----
    if "clean" in prog:
        clean_pref = prog["clean"]["pref"]; clean_true = prog["clean"]["true"]
    else:
        clean_pref, clean_true = [], []
        for j, i in enumerate(test_idx):
            row = _cal_logits(items[i]["tok4"])
            clean_pref.append(float(row[pref_ids[j]].item()))
            clean_true.append(float(row[gt_ids[j]].item()))
        prog["clean"] = {"pref": clean_pref, "true": clean_true}
        save_json(ck, prog)
    log(f"WO#5.1[{tag}]: regL_c4={regL_c4} (R2={_cal_fmt(regR2_c4)}) regL_c1={regL_c1}; clean C4 baselines ready.")

    # ---- Stage 1: zero-ablation hook sanity (the GT logit MUST collapse) ----
    if "zeroabl" in prog:
        zero_abl_delta = prog["zeroabl"]["delta"]
    else:
        zero = torch.zeros(d_model, device=model.cfg.device)
        sample = list(range(min(int(CFG["wo_cal_zeroabl_sample"]), len(test_idx))))
        zd, zflip = [], []
        for j in sample:
            i = test_idx[j]
            row = _cal_logits(items[i]["tok4"], regL_c4, _wo_mk_patch_hook(zero, eq4[j]))
            zd.append(float(row[gt_ids[j]].item()) - clean_true[j])
            zflip.append(0.0 if int(row.argmax().item()) == gt_ids[j] else 1.0)
        zero_abl_delta = float(np.mean(zd))
        prog["zeroabl"] = {"delta": zero_abl_delta, "flip_away": float(np.mean(zflip)), "n": len(sample)}
        save_json(ck, prog)
    log(f"WO#5.1[{tag}]: zero-ablation Δ_GT={zero_abl_delta:+.3f} (must be strongly negative).")

    # ---- Stage 2: full-residual SWAP positive control (donor product P') ----
    if "swap" in prog:
        swap_delta = prog["swap"]["delta"]
    else:
        sp, sflip = [], []
        for j, i in enumerate(test_idx):
            donor = items[test_idx[perm[j]]]["resid_c4_eq"][regL_c4].astype(np.float32)
            dv = torch.tensor(donor, device=model.cfg.device)
            row = _cal_logits(items[i]["tok4"], regL_c4, _wo_mk_patch_hook(dv, eq4[j]))
            sp.append(float(row[pref_ids[j]].item()))
            sflip.append(1.0 if int(row.argmax().item()) == pref_ids[j] else 0.0)
        swap_delta = float(np.mean(sp) - np.mean(clean_pref))
        ci = wo_paired_delta_ci(np.array(sp), np.array(clean_pref), n_boot=int(CFG["wo_cal_nboot"]), seed=_CAL_SEED)
        prog["swap"] = {"delta": swap_delta, "ci": [ci[0], ci[1]], "flip_to_donor": float(np.mean(sflip)),
                        "n": len(test_idx)}
        save_json(ck, prog)
    log(f"WO#5.1[{tag}]: full-swap Δ_P'={swap_delta:+.3f} (guaranteed-causal positive control).")

    # ---- Stage 3: magnitude sweep of the probe-direction counterfactual inject ----
    k_grid = list(CFG["wo_cal_k_grid"])
    if "sweep" in prog:
        k_deltas = prog["sweep"]["k_deltas"]
    else:
        fit = wo_fit_ridge_probe(_stack(train_idx, "resid_c4_eq", regL_c4),
                                 np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float))
        what = fit["direction"]
        # precompute per-item minimal edits δ_min and ||resid||.
        dmins, rnorms = [], []
        for j, i in enumerate(test_idx):
            clean = items[i]["resid_c4_eq"][regL_c4].astype(np.float32)
            coord = wo_probe_coord_for_value(fit, pref_vals[j])
            dmins.append(wo_inject_to_target(clean, what, coord) - clean)
            rnorms.append(float(np.linalg.norm(clean)))
        k_deltas, rel_norm = [], []
        for k in k_grid:
            kp = []
            for j, i in enumerate(test_idx):
                dv = torch.tensor((k * dmins[j]).astype(np.float32), device=model.cfg.device)
                row = _cal_logits(items[i]["tok4"], regL_c4, _st_mk_add_hook(dv, eq4[j]))
                kp.append(float(row[pref_ids[j]].item()))
            k_deltas.append(float(np.mean(kp) - np.mean(clean_pref)))
            rel_norm.append(float(np.mean([k * np.linalg.norm(dmins[j]) / (rnorms[j] + 1e-9)
                                           for j in range(len(test_idx))])))
            save_json(ck, {**prog, "sweep": {"k_grid": k_grid, "k_deltas": k_deltas, "rel_norm": rel_norm}})
            log(f"WO#5.1[{tag}]: k={k:<3} Δ_P'={k_deltas[-1]:+.3f}  (||δ||/||resid||~{rel_norm[-1]:.2f})")
        prog["sweep"] = {"k_grid": k_grid, "k_deltas": k_deltas, "rel_norm": rel_norm}
        save_json(ck, prog)

    verdict = wo_steer_calibration_verdict(zero_abl_delta, swap_delta, k_grid, k_deltas,
                                           recover_thr=float(CFG["wo_cal_recover_thr"]),
                                           zeroabl_floor=float(CFG["wo_cal_zeroabl_floor"]))

    # ---- Stage 4: C1 re-eval at k* (only if CALIBRATED) ----
    c1_reeval = None
    if verdict["label"] == "CALIBRATED" and "c1" not in prog:
        kstar = verdict["k_star"]
        fitc1 = wo_fit_ridge_probe(_stack(train_idx, "resid_equals", regL_c1),
                                   np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float))
        wc1 = fitc1["direction"]
        c1p, c1clean, c1flip = [], [], []
        for j, i in enumerate(test_idx):
            cln = _cal_logits(items[i]["tok1"])                 # clean C1 row
            c1clean.append(float(cln[pref_ids[j]].item()))
            clean = items[i]["resid_equals"][regL_c1].astype(np.float32)
            dmin = wo_inject_to_target(clean, wc1, wo_probe_coord_for_value(fitc1, pref_vals[j])) - clean
            dv = torch.tensor((kstar * dmin).astype(np.float32), device=model.cfg.device)
            row = _cal_logits(items[i]["tok1"], regL_c1, _st_mk_add_hook(dv, items[i]["equals"]))
            c1p.append(float(row[pref_ids[j]].item()))
            c1flip.append(1.0 if int(row.argmax().item()) == pref_ids[j] else 0.0)
        c1_delta = float(np.mean(c1p) - np.mean(c1clean))
        ci = wo_paired_delta_ci(np.array(c1p), np.array(c1clean), n_boot=int(CFG["wo_cal_nboot"]), seed=_CAL_SEED)
        c1_reeval = {"k_star": kstar, "layer": regL_c1, "delta": c1_delta, "ci": [ci[0], ci[1]],
                     "flip_to_Pprime": float(np.mean(c1flip)),
                     "drives_c1": bool(c1_delta >= float(CFG["wo_cal_recover_thr"]))}
        prog["c1"] = c1_reeval
        save_json(ck, prog)
        log(f"WO#5.1[{tag}]: C1 re-eval @k={kstar} Δ_P'={c1_delta:+.3f} -> "
            f"{'C1 answer site IS drivable' if c1_reeval['drives_c1'] else 'C1 site NOT drivable even at k*'}.")
    elif "c1" in prog:
        c1_reeval = prog["c1"]

    out = {
        "tag": tag, "experiment": "WO5.1_steering_instrument_calibration",
        "reused_capture": True, "pair_sha": bundle.get("pair_sha"),
        "n_test": len(test_idx), "regL_c4": regL_c4, "regL_c1": regL_c1,
        "regR2_c4": regR2_c4, "regR2_c1": regR2_c1,
        "zeroabl": prog.get("zeroabl"), "swap": prog.get("swap"), "sweep": prog.get("sweep"),
        "c1_reeval": c1_reeval, "verdict": verdict,
        "recover_thr": float(CFG["wo_cal_recover_thr"]),
        "note": ("Diagnoses WO#5's INCONCLUSIVE: climbs zero-ablation -> full donor swap -> magnitude "
                 "sweep at C4 '=' (reusing the cached residuals). CALIBRATED@k => WO#5 was under-powered; "
                 "DEAD_DIRECTION => the probe (operand-aligned) direction is not a causal handle."),
    }
    wo_save_result(f"causal_steering_calibration_{tag}.json", json.dumps(out, indent=2, default=str))
    return out


def _cal_fmt(x):
    return "n/a" if x is None else f"{float(x):.3f}"


WO_STEER_CAL = {}
for _tag in CFG["wo_cal_tags"]:
    WO_STEER_CAL[_tag] = _cal_run(_tag)

# ---- flat summary CSV --------------------------------------------------------
_cal_rows = []
for _tag in CFG["wo_cal_tags"]:
    o = WO_STEER_CAL[_tag]
    sw = o.get("sweep") or {"k_grid": [], "k_deltas": [], "rel_norm": []}
    base = {"tag": _tag, "regL_c4": o["regL_c4"], "verdict": o["verdict"]["label"],
            "k_star": o["verdict"].get("k_star"),
            "zeroabl_delta": (o["zeroabl"] or {}).get("delta"),
            "swap_delta": (o["swap"] or {}).get("delta")}
    for k, d, rn in zip(sw["k_grid"], sw["k_deltas"], sw.get("rel_norm", [None] * len(sw["k_grid"]))):
        _cal_rows.append({**base, "k": k, "k_delta": d, "rel_edit_norm": rn})
    if not sw["k_grid"]:
        _cal_rows.append({**base, "k": None, "k_delta": None, "rel_edit_norm": None})
wo_save_result("causal_steering_calibration_summary.csv",
               wo_battery_csv(_cal_rows, ["tag", "verdict", "k_star", "regL_c4", "zeroabl_delta",
                                         "swap_delta", "k", "k_delta", "rel_edit_norm"]))

print("\n================= WO#5.1 — STEERING-INSTRUMENT CALIBRATION (reused WO#5 capture) =================")
for _tag in CFG["wo_cal_tags"]:
    o = WO_STEER_CAL[_tag]; v = o["verdict"]; sw = o.get("sweep") or {}
    print(f"\n[{_tag}]  regL_c4={o['regL_c4']}  zero-abl Δ_GT={_cal_fmt((o['zeroabl'] or {}).get('delta'))}  "
          f"full-swap Δ_P'={_cal_fmt((o['swap'] or {}).get('delta'))}")
    if sw.get("k_grid"):
        print("   magnitude sweep (Δ_P' @ k):  " +
              "  ".join(f"k{k}:{d:+.3f}" for k, d in zip(sw["k_grid"], sw["k_deltas"])))
    if o.get("c1_reeval"):
        print(f"   C1 re-eval @k={o['c1_reeval']['k_star']}: Δ_P'={_cal_fmt(o['c1_reeval']['delta'])} -> "
              f"drives C1 = {o['c1_reeval']['drives_c1']}")
    print(f"   VERDICT: {v['label']}{('@k='+str(v['k_star'])) if v.get('k_star') else ''} — {v['reason']}")
print("==================================================================================================")
