# ============================================================================
# Phase 6 / WORK ORDER #5.1d — LATE-LAYER FULL-SWAP SWEEP (GPU; REUSES WO#5).
# ----------------------------------------------------------------------------
# WO#5.1c showed: with the full-product metric, NOTHING emits P' — even the full
# residual SWAP (emit-P'=0). But that swap ran at the early decodability-peak layer
# (~L4), where the '=' position hasn't composed the product and the true operands
# downstream recompute it. The product is written into the '=' / last-token residual
# LATE (Stolfo 2023: operand info -> last token in late layers, MLP writes the answer).
#
# So this cell swaps the donor's C4 '=' residual across a LATE-layer sweep and asks,
# with the FULL-PRODUCT metric, WHERE transplanting it makes the model emit the
# donor's product P'. That localizes where the product is causally load-bearing at
# the answer site — the working positive control WO#5 never had. Reuses the cached
# residuals (no re-capture). Verdict:
#   LATE_SWAP_WORKS@L  -> the product IS at the '=' site by layer L; the WO#5 inject
#                         null was a layer/magnitude (+ first-token-metric) artifact.
#   PRODUCT_NOT_AT_EQUALS -> no late swap emits the donor -> the product is NOT
#                         localized to the '=' residual (distributed / in operand heads);
#                         steering the '=' is the wrong site, not 'product unused'.
# Deliverable: causal_steering_lateswap_{tag}.json + _summary.csv. Checkpointed/layer.
# ============================================================================
import json
import numpy as np
import torch

assert "_st_capture" in globals() and "_st_split" in globals() and "_cal_derange_perm" in globals() \
    and "_wo_mk_patch_hook" in globals() and "_fp_logprob" in globals() and "_fp_generate" in globals() \
    and "wo_parse_int" in globals(), (
    "WO#5.1d needs cells 82l + 82n + 82p (for _fp_logprob/_fp_generate). Run them first.")

CFG.setdefault("wo_ls_tags", list(CFG.get("wo_fp_tags", ["base", "instruct"])))
CFG.setdefault("wo_ls_gen_sample", 60)        # #items for greedy-decode emit-P' (the gold metric).
CFG.setdefault("wo_ls_gen_k", int(globals().get("WO_MAX_NEW_TOKENS", 8)))
CFG.setdefault("wo_ls_emit_thr", 0.5)         # emit-P' rate that counts as a working causal swap.
_LS_SEED = int(CFG.get("wo_steer_seed", 707))


@torch.no_grad()
def _ls_run(tag):
    ck = f"wo_ls_{tag}"
    prog = load_json(ck) if has_artifact(ck, "json") else {}
    bundle = _st_capture(tag)                                  # REUSED (no re-capture)
    items = bundle["items"]
    nL = bundle["n_layers"]
    _, test_idx = _st_split(len(items), _LS_SEED)
    # LATE sweep: the back ~40% of the network (composition/answer-write zone) + the last layer.
    late_layers = sorted(set(range(int(round(0.6 * (nL - 1))), nL, 2)) | {nL - 1})

    perm = _cal_derange_perm(len(test_idx), _LS_SEED + 3)
    pref = [int(items[test_idx[perm[j]]]["B"] * items[test_idx[perm[j]]]["C"]) for j in range(len(test_idx))]
    true = [int(items[i]["B"] * items[i]["C"]) for i in test_idx]
    eq4 = [items[i]["eq4"] for i in test_idx]
    gsamp = list(range(min(int(CFG["wo_ls_gen_sample"]), len(test_idx))))

    # clean full-product baseline logprob(P')-logprob(true) (reused from 82p if present).
    if "clean" not in prog:
        if has_artifact(f"wo_fp_{tag}", "json") and "clean" in load_json(f"wo_fp_{tag}"):
            prog["clean"] = load_json(f"wo_fp_{tag}")["clean"]
        else:
            prog["clean"] = [_fp_logprob(items[i]["tok4"], pref[j]) - _fp_logprob(items[i]["tok4"], true[j])
                             for j, i in enumerate(test_idx)]
        save_json(ck, prog)
    clean = prog["clean"]

    rows = prog.get("rows", {})
    for L in late_layers:
        if str(L) in rows:
            continue
        dscore, emit, emit_true = [], [], []
        for j, i in enumerate(test_idx):
            donor = torch.tensor(items[test_idx[perm[j]]]["resid_c4_eq"][L].astype(np.float32),
                                 device=model.cfg.device)
            hook = _wo_mk_patch_hook(donor, eq4[j])
            dscore.append((_fp_logprob(items[i]["tok4"], pref[j], L, hook)
                           - _fp_logprob(items[i]["tok4"], true[j], L, hook)) - clean[j])
            if j in gsamp:
                gen = wo_parse_int(_fp_generate(items[i]["tok4"], L, hook, int(CFG["wo_ls_gen_k"])))
                emit.append(1.0 if gen == pref[j] else 0.0)
                emit_true.append(1.0 if gen == true[j] else 0.0)
        rows[str(L)] = {"layer": int(L), "fullprod_logprob_delta": float(np.mean(dscore)),
                        "emit_Pprime_rate": float(np.mean(emit)), "emit_true_rate": float(np.mean(emit_true)),
                        "gen_n": len(emit), "n": len(dscore)}
        prog["rows"] = rows
        save_json(ck, prog)
        log(f"WO#5.1d[{tag}]: SWAP @L{L}/{nL-1}  emit-P'={rows[str(L)]['emit_Pprime_rate']:.3f} "
            f"emit-true={rows[str(L)]['emit_true_rate']:.3f}  logprobΔ={rows[str(L)]['fullprod_logprob_delta']:+.2f}")

    thr = float(CFG["wo_ls_emit_thr"])
    best = max(rows.values(), key=lambda r: r["emit_Pprime_rate"]) if rows else None
    if best and best["emit_Pprime_rate"] >= thr:
        verdict = f"LATE_SWAP_WORKS@L{best['layer']}"
    else:
        verdict = "PRODUCT_NOT_AT_EQUALS"
    out = {"tag": tag, "experiment": "WO5.1d_lateswap", "reused_capture": True,
           "late_layers": late_layers, "rows": [rows[str(L)] for L in late_layers if str(L) in rows],
           "best_layer": (best["layer"] if best else None),
           "best_emit_Pprime": (best["emit_Pprime_rate"] if best else None),
           "verdict": verdict, "pair_sha": bundle.get("pair_sha"),
           "note": "Localizes where transplanting the donor's '=' residual emits the donor product "
                   "(full-product metric). LATE_SWAP_WORKS => '=' site is causal there; "
                   "PRODUCT_NOT_AT_EQUALS => product not localized to the '=' residual."}
    wo_save_result(f"causal_steering_lateswap_{tag}.json", json.dumps(out, indent=2, default=str))
    return out


WO_STEER_LS = {}
for _tag in CFG["wo_ls_tags"]:
    WO_STEER_LS[_tag] = _ls_run(_tag)

_ls_rows = []
for _tag in CFG["wo_ls_tags"]:
    o = WO_STEER_LS[_tag]
    for r in o["rows"]:
        _ls_rows.append({"tag": _tag, "verdict": o["verdict"], "layer": r["layer"],
                         "emit_Pprime_rate": r["emit_Pprime_rate"], "emit_true_rate": r["emit_true_rate"],
                         "fullprod_logprob_delta": r["fullprod_logprob_delta"]})
wo_save_result("causal_steering_lateswap_summary.csv",
               wo_battery_csv(_ls_rows, ["tag", "verdict", "layer", "emit_Pprime_rate",
                                         "emit_true_rate", "fullprod_logprob_delta"]))

print("\n========== WO#5.1d — LATE-LAYER FULL-SWAP SWEEP at C4 '=' (full-product metric; reused WO#5) ==========")
for _tag in CFG["wo_ls_tags"]:
    o = WO_STEER_LS[_tag]
    print(f"\n[{_tag}]  (emit-P' = donor product generated; emit-true = original product generated)")
    for r in o["rows"]:
        print(f"   L{r['layer']:<3} emit-P'={r['emit_Pprime_rate']:.3f}  emit-true={r['emit_true_rate']:.3f}  "
              f"logprobΔ(P'-true)={r['fullprod_logprob_delta']:+.2f}")
    print(f"   >>> VERDICT: {o['verdict']}  (best emit-P'={o['best_emit_Pprime']} @L{o['best_layer']})")
print("=========================================================================================================")
