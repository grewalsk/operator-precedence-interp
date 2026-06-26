# ============================================================================
# Phase 6 / WORK ORDER #5.1b — RE-METRIC THE STEERING LADDER (GPU; REUSES WO#5).
# ----------------------------------------------------------------------------
# WO#5.1 (cell 82n) returned METRIC_OR_SITE_SUSPECT, but its raw numbers showed the
# instrument actually WORKS: a full-residual SWAP at C4's '=' flipped the model's
# answer to the donor's product on 100% of items (flip_to_donor=1.0, both tags) —
# yet the absolute first-token LOGIT moved by ~0. The metric, not the instrument,
# was the problem: many leading-digit-chunk tokens (' 108',' 176',...) share a
# similar logit, so the argmax can flip completely while the absolute value is flat.
#
# This cell re-scores the ladder with the RIGHT metrics, reusing the cached WO#5
# residuals (no re-capture):
#   • FLIP-RATE-to-target (robust to logit compression — the primary read), and
#   • LOGIT-DIFF (target_first_tok − true_first_tok; scale-controlled, the standard
#     activation-patching metric).
# It also sweeps the inject across (LAYER × k) — including a LATE layer, since WO#5/
# 5.1 pre-registered the early decodability-peak layer (~L4), not the composition
# site (~L0.85·n). The decisive question: does the rank-1 probe-direction INJECT flip
# the answer at ANY (layer,k), or only the full swap?
#   CALIBRATED  -> WO#5's null was a metric/magnitude artifact; re-run C1 there.
#   DEAD_DIRECTION -> the operand-aligned probe axis is not a causal handle even where
#                     the SITE is causal (decoding != causal) -> switch to DAS / a
#                     centered operand band.
# Deliverable: causal_steering_remetric_{base,instruct}.json + _summary.csv.
# Checkpointed per (tag, stage); reuses cell-82n helpers (_cal_logits, _cal_derange_perm).
# ============================================================================
import json
import numpy as np
import torch

assert "_st_capture" in globals() and "_cal_logits" in globals() and "_cal_derange_perm" in globals() \
    and "wo_steer_flip_verdict" in globals() and "_st_mk_add_hook" in globals() \
    and "_wo_mk_patch_hook" in globals() and "wo_fit_ridge_probe" in globals(), (
    "WO#5.1b re-metric needs cells 82l + 82n (capture/_cal_logits/_cal_derange_perm/hooks) "
    "+ cell-76 wo_steer_flip_verdict. Run them first.")

CFG.setdefault("wo_ro_tags", list(CFG.get("wo_cal_tags", ["base", "instruct"])))
CFG.setdefault("wo_ro_k_grid", [1, 4, 16])                # inject magnitude multipliers (32 was destructive).
CFG.setdefault("wo_ro_flip_thr", 0.5)                     # flip-rate that counts as causal control.
CFG.setdefault("wo_ro_zeroabl_sample", 24)
CFG.setdefault("wo_ro_nboot", int(CFG.get("wo_steer_nboot", 10000)))
_RO_SEED = int(CFG.get("wo_steer_seed", 707))
_RO_STRIDE = int(CFG.get("wo_steer_layer_stride", 2))


def _ro_score(row, target_id, ref_id):
    """(logit-diff target−true, flip-to-target) from a final-position logit row."""
    return (float(row[int(target_id)].item()) - float(row[int(ref_id)].item()),
            1.0 if int(row.argmax().item()) == int(target_id) else 0.0)


@torch.no_grad()
def _ro_run(tag):
    ck = f"wo_ro_{tag}"
    prog = load_json(ck) if has_artifact(ck, "json") else {}
    bundle = _st_capture(tag)                                  # REUSED from WO#5 (no re-capture)
    items = bundle["items"]
    nL = bundle["n_layers"]
    n = len(items)
    train_idx, test_idx = _st_split(n, _RO_SEED)
    L_grid = sorted(set(range(0, nL, _RO_STRIDE)) | {nL - 1})

    def _stack(idxs, rkey, L):
        return np.stack([items[i][rkey][L].astype(np.float32) for i in idxs])

    yP_tr = np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float)

    def _reg_layer(rkey):
        best, bestL = None, None
        for L in L_grid:
            r = wo_cv_r2(_stack(train_idx, rkey, L), yP_tr, folds=5, ridge=1.0)
            if r is not None and (best is None or r > best):
                best, bestL = r, int(L)
        return bestL if bestL is not None else int(L_grid[len(L_grid) // 2])

    regL_c4 = _reg_layer("resid_c4_eq")
    late_L = int(round(0.85 * (nL - 1)))                      # composition zone (~late)
    mid_L = int(round(0.5 * (nL - 1)))
    inj_layers = sorted({regL_c4, mid_L, late_L})
    k_grid = list(CFG["wo_ro_k_grid"])

    # donor pairs (deranged) -> counterfactual product P' + tokens; true tokens = ref.
    perm = _cal_derange_perm(len(test_idx), _RO_SEED + 3)
    pref_vals = [int(items[test_idx[perm[j]]]["B"] * items[test_idx[perm[j]]]["C"]) for j in range(len(test_idx))]
    pref_ids = [_st_first_tok_id(v) for v in pref_vals]
    gt_ids = [items[i]["gt_id"] for i in test_idx]            # true product token = ref
    eq4 = [items[i]["eq4"] for i in test_idx]

    # ---- clean C4 baselines (logit-diff P'−true, one forward/item; reused) ----
    if "clean" in prog:
        clean_ld = prog["clean"]
    else:
        clean_ld = []
        for j, i in enumerate(test_idx):
            row = _cal_logits(items[i]["tok4"])
            clean_ld.append(_ro_score(row, pref_ids[j], gt_ids[j])[0])
        prog["clean"] = clean_ld
        save_json(ck, prog)

    # ---- Stage A: zero-ablation moves the output? (logit + argmax change) ----
    if "zeroabl" in prog:
        zero_abl_moves = prog["zeroabl"]["moves"]
    else:
        zero = torch.zeros(items[0]["resid_c4_eq"].shape[1], device=model.cfg.device)
        samp = list(range(min(int(CFG["wo_ro_zeroabl_sample"]), len(test_idx))))
        dgt, away = [], []
        for j in samp:
            i = test_idx[j]
            cl = _cal_logits(items[i]["tok4"])
            rw = _cal_logits(items[i]["tok4"], regL_c4, _wo_mk_patch_hook(zero, eq4[j]))
            dgt.append(float(rw[gt_ids[j]].item()) - float(cl[gt_ids[j]].item()))
            away.append(0.0 if int(rw.argmax().item()) == gt_ids[j] else 1.0)
        zero_abl_moves = bool(abs(float(np.mean(dgt))) >= float(globals().get("WO_STEER_ZEROABL_FLOOR", 1.0))
                              or float(np.mean(away)) > 0.0)
        prog["zeroabl"] = {"moves": zero_abl_moves, "dgt": float(np.mean(dgt)), "flip_away": float(np.mean(away))}
        save_json(ck, prog)

    # ---- Stage B: full-residual SWAP (donor P') with flip + logit-diff ----
    if "swap" in prog:
        swap_flip = prog["swap"]["flip"]
    else:
        sf, sld = [], []
        for j, i in enumerate(test_idx):
            donor = torch.tensor(items[test_idx[perm[j]]]["resid_c4_eq"][regL_c4].astype(np.float32),
                                 device=model.cfg.device)
            row = _cal_logits(items[i]["tok4"], regL_c4, _wo_mk_patch_hook(donor, eq4[j]))
            ld, fl = _ro_score(row, pref_ids[j], gt_ids[j])
            sld.append(ld); sf.append(fl)
        swap_flip = float(np.mean(sf))
        prog["swap"] = {"flip": swap_flip, "layer": regL_c4,
                        "logit_diff": float(np.mean(sld)),
                        "logit_diff_delta": float(np.mean(sld) - np.mean(clean_ld))}
        save_json(ck, prog)
    log(f"WO#5.1b[{tag}]: full-swap flip-to-donor={swap_flip:.3f} @L{regL_c4} "
        f"(logit-diff Δ={prog['swap']['logit_diff_delta']:+.2f}).")

    # ---- Stage C: probe-direction INJECT across (layer × k): flip + logit-diff ----
    if "sweep" in prog:
        layer_k_flips = [tuple(t) for t in prog["sweep"]["layer_k_flips"]]
        sweep_rows = prog["sweep"]["rows"]
    else:
        sweep_rows, layer_k_flips = [], []
        for L in inj_layers:
            fit = wo_fit_ridge_probe(_stack(train_idx, "resid_c4_eq", L), yP_tr)
            what = fit["direction"]
            dmins = []
            for j, i in enumerate(test_idx):
                clean = items[i]["resid_c4_eq"][L].astype(np.float32)
                dmins.append(wo_inject_to_target(clean, what, wo_probe_coord_for_value(fit, pref_vals[j])) - clean)
            for k in k_grid:
                fl, ld = [], []
                for j, i in enumerate(test_idx):
                    dv = torch.tensor((k * dmins[j]).astype(np.float32), device=model.cfg.device)
                    row = _cal_logits(items[i]["tok4"], L, _st_mk_add_hook(dv, eq4[j]))
                    d, f = _ro_score(row, pref_ids[j], gt_ids[j])
                    ld.append(d); fl.append(f)
                flip = float(np.mean(fl))
                layer_k_flips.append((int(L), int(k), flip))
                sweep_rows.append({"layer": int(L), "k": int(k), "flip_to_Pprime": flip,
                                   "logit_diff": float(np.mean(ld)),
                                   "logit_diff_delta": float(np.mean(ld) - np.mean(clean_ld))})
                save_json(ck, {**prog, "sweep": {"rows": sweep_rows, "layer_k_flips": layer_k_flips}})
                log(f"WO#5.1b[{tag}]: inject L{L} k={k:<3} flip→P'={flip:.3f} "
                    f"(logit-diff Δ={sweep_rows[-1]['logit_diff_delta']:+.2f})")
        prog["sweep"] = {"rows": sweep_rows, "layer_k_flips": layer_k_flips}
        save_json(ck, prog)

    verdict = wo_steer_flip_verdict(zero_abl_moves, swap_flip, layer_k_flips,
                                    flip_thr=float(CFG["wo_ro_flip_thr"]))

    # ---- Stage D: C1 re-eval at the winning (layer*, k*) — inject P' at C1's '=' ----
    c1 = None
    if verdict["label"] == "CALIBRATED" and "c1" not in prog:
        Ls, ks = verdict["layer_star"], verdict["k_star"]
        fitc1 = wo_fit_ridge_probe(_stack(train_idx, "resid_equals", Ls), yP_tr)
        wc1 = fitc1["direction"]
        fl, ld, cld = [], [], []
        for j, i in enumerate(test_idx):
            cl = _cal_logits(items[i]["tok1"])
            cld.append(_ro_score(cl, pref_ids[j], gt_ids[j])[0])
            clean = items[i]["resid_equals"][Ls].astype(np.float32)
            dmin = wo_inject_to_target(clean, wc1, wo_probe_coord_for_value(fitc1, pref_vals[j])) - clean
            dv = torch.tensor((ks * dmin).astype(np.float32), device=model.cfg.device)
            row = _cal_logits(items[i]["tok1"], Ls, _st_mk_add_hook(dv, items[i]["equals"]))
            d, f = _ro_score(row, pref_ids[j], gt_ids[j])
            ld.append(d); fl.append(f)
        c1 = {"layer": Ls, "k": ks, "flip_to_Pprime": float(np.mean(fl)),
              "logit_diff_delta": float(np.mean(ld) - np.mean(cld)),
              "drives_c1": bool(float(np.mean(fl)) >= float(CFG["wo_ro_flip_thr"]))}
        prog["c1"] = c1
        save_json(ck, prog)
        log(f"WO#5.1b[{tag}]: C1 inject @L{Ls} k={ks} flip→P'={c1['flip_to_Pprime']:.3f} -> "
            f"drives C1 = {c1['drives_c1']}")
    elif "c1" in prog:
        c1 = prog["c1"]

    out = {
        "tag": tag, "experiment": "WO5.1b_steering_remetric", "reused_capture": True,
        "pair_sha": bundle.get("pair_sha"), "n_test": len(test_idx),
        "regL_c4": regL_c4, "inj_layers": inj_layers, "k_grid": k_grid,
        "zeroabl": prog.get("zeroabl"), "swap": prog.get("swap"), "sweep": prog.get("sweep"),
        "c1_reeval": c1, "verdict": verdict, "flip_thr": float(CFG["wo_ro_flip_thr"]),
        "note": ("Re-scores the WO#5.1 ladder with FLIP-RATE-to-target + LOGIT-DIFF (the absolute "
                 "first-token logit was compressed). Full swap flip + the inject (layer×k) flip sweep "
                 "decide CALIBRATED (metric/magnitude artifact) vs DEAD_DIRECTION (wrong causal axis)."),
    }
    wo_save_result(f"causal_steering_remetric_{tag}.json", json.dumps(out, indent=2, default=str))
    return out


WO_STEER_RO = {}
for _tag in CFG["wo_ro_tags"]:
    WO_STEER_RO[_tag] = _ro_run(_tag)

# ---- flat summary CSV --------------------------------------------------------
_ro_rows = []
for _tag in CFG["wo_ro_tags"]:
    o = WO_STEER_RO[_tag]
    base = {"tag": _tag, "verdict": o["verdict"]["label"],
            "layer_star": o["verdict"].get("layer_star"), "k_star": o["verdict"].get("k_star"),
            "swap_flip": (o["swap"] or {}).get("flip")}
    for r in (o.get("sweep") or {}).get("rows", []):
        _ro_rows.append({**base, "inj_layer": r["layer"], "k": r["k"],
                         "inject_flip": r["flip_to_Pprime"], "inject_logit_diff_delta": r["logit_diff_delta"]})
wo_save_result("causal_steering_remetric_summary.csv",
               wo_battery_csv(_ro_rows, ["tag", "verdict", "layer_star", "k_star", "swap_flip",
                                         "inj_layer", "k", "inject_flip", "inject_logit_diff_delta"]))

print("\n================= WO#5.1b — STEERING RE-METRIC (flip-rate + logit-diff; reused WO#5) =================")
for _tag in CFG["wo_ro_tags"]:
    o = WO_STEER_RO[_tag]; v = o["verdict"]
    print(f"\n[{_tag}]  full-swap flip→donor = {(o['swap'] or {}).get('flip')}  (the guaranteed-causal control)")
    print("   probe-direction inject flip→P' by (layer,k):")
    for r in (o.get("sweep") or {}).get("rows", []):
        print(f"       L{r['layer']:<3} k={r['k']:<3} flip={r['flip_to_Pprime']:.3f}  (logit-diff Δ={r['logit_diff_delta']:+.2f})")
    if o.get("c1_reeval"):
        c = o["c1_reeval"]
        print(f"   C1 inject @L{c['layer']} k={c['k']}: flip→P'={c['flip_to_Pprime']:.3f} -> drives C1 = {c['drives_c1']}")
    print(f"   >>> VERDICT: {v['label']}"
          f"{(' @L'+str(v['layer_star'])+' k='+str(v['k_star'])) if v.get('k_star') else ''}")
    print(f"       {v['reason']}")
print("======================================================================================================")
