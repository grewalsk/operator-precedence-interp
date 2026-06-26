# ============================================================================
# Phase 6 / WORK ORDER #5.1c — FULL-PRODUCT METRIC (GPU; REUSES WO#5 capture).
# ----------------------------------------------------------------------------
# WO#5/5.1/5.1b all scored the FIRST answer token — but Llama-3 makes that token
# the leading space / coarse leading chunk, SHARED across products. Proof from the
# 82o run: logit-diff(P'−true) ≡ 0.0 on all 200 items ⇒ pref_id == gt_id always.
# So every flip/logit metric was blind to WHICH product. This cell re-scores the
# decisive interventions with a FULL-PRODUCT metric:
#   • teacher-forced sum-logprob of the WHOLE product string (all answer tokens):
#       score = logprob(" P'") − logprob(" true");  Δ = score(intervened) − score(clean)
#   • greedy-decode-and-PARSE (on a sample): does the model actually EMIT P' (full int)?
# Interventions (reusing the cached residuals, no re-capture):
#   SWAP  : full-residual swap at C4 '=' with a donor (product P')  — guaranteed-causal
#   INJECT: probe-direction counterfactual at C4 '=' (early reg layer + a LATE layer), k=1
#   C1    : probe-direction counterfactual at C1 '='                — the headline site
# Verdict (inline): does a guaranteed-causal SWAP move the full product? If yes, does
# the rank-1 INJECT? If the swap moves it but inject never does -> wrong axis.
# Deliverable: causal_steering_fullproduct_{tag}.json + _summary.csv. Checkpointed.
# ============================================================================
import json
import numpy as np
import torch

assert "_st_capture" in globals() and "_st_split" in globals() and "_cal_derange_perm" in globals() \
    and "_st_mk_add_hook" in globals() and "_wo_mk_patch_hook" in globals() \
    and "wo_fit_ridge_probe" in globals() and "wo_parse_int" in globals(), (
    "WO#5.1c needs cells 82l + 82n + cell-76. Run them first.")

CFG.setdefault("wo_fp_tags", list(CFG.get("wo_cal_tags", ["base", "instruct"])))
CFG.setdefault("wo_fp_gen_sample", 40)        # #items for the (costlier) greedy-decode flip check.
CFG.setdefault("wo_fp_gen_k", int(globals().get("WO_MAX_NEW_TOKENS", 8)))
CFG.setdefault("wo_fp_inject_k", 1)           # inject magnitude (k=1 = the WO#5 minimal edit).
_FP_SEED = int(CFG.get("wo_steer_seed", 707))
_FP_STRIDE = int(CFG.get("wo_steer_layer_stride", 2))


@torch.no_grad()
def _fp_run_with(tokens_list, hook_layer=None, hook=None):
    tok = torch.tensor([tokens_list], device=model.cfg.device, dtype=torch.long)
    if hook is None:
        return model(tok)
    return model.run_with_hooks(tok, fwd_hooks=[(f"blocks.{hook_layer}.hook_resid_post", hook)])


def _fp_ans_ids(value):
    return tokenizer(" " + str(int(value)), add_special_tokens=False)["input_ids"]


@torch.no_grad()
def _fp_logprob(prompt_ids, value, hook_layer=None, hook=None):
    """Teacher-forced sum log-prob of the FULL ' <value>' answer string appended to
    prompt_ids, under an optional (hook_layer, hook) intervention at the '=' site."""
    ans = _fp_ans_ids(value)
    full = list(prompt_ids) + ans
    lg = _fp_run_with(full, hook_layer, hook)
    logp = torch.log_softmax(lg[0].float(), dim=-1)
    start = len(prompt_ids)
    return float(sum(logp[start + t - 1, a].item() for t, a in enumerate(ans)))


@torch.no_grad()
def _fp_generate(prompt_ids, hook_layer=None, hook=None, K=8):
    """Greedy-decode K tokens (hook re-applied each step at the fixed '=' site); return
    the decoded continuation string."""
    ids = list(prompt_ids)
    for _ in range(K):
        lg = _fp_run_with(ids, hook_layer, hook)
        ids.append(int(lg[0, -1].argmax().item()))
    return tokenizer.decode(ids[len(prompt_ids):])


@torch.no_grad()
def _fp_run(tag):
    ck = f"wo_fp_{tag}"
    prog = load_json(ck) if has_artifact(ck, "json") else {}
    bundle = _st_capture(tag)                                  # REUSED (no re-capture)
    items = bundle["items"]
    nL = bundle["n_layers"]
    train_idx, test_idx = _st_split(len(items), _FP_SEED)
    L_grid = sorted(set(range(0, nL, _FP_STRIDE)) | {nL - 1})

    def _stack(idxs, rkey, L):
        return np.stack([items[i][rkey][L].astype(np.float32) for i in idxs])
    yP_tr = np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float)

    def _reg(rkey):
        best, bestL = None, None
        for L in L_grid:
            r = wo_cv_r2(_stack(train_idx, rkey, L), yP_tr, folds=5, ridge=1.0)
            if r is not None and (best is None or r > best):
                best, bestL = r, int(L)
        return bestL if bestL is not None else int(L_grid[len(L_grid) // 2])
    regL_c4 = _reg("resid_c4_eq")
    late_L = int(round(0.85 * (nL - 1)))
    kk = int(CFG["wo_fp_inject_k"])

    perm = _cal_derange_perm(len(test_idx), _FP_SEED + 3)
    pref = [int(items[test_idx[perm[j]]]["B"] * items[test_idx[perm[j]]]["C"]) for j in range(len(test_idx))]
    true = [int(items[i]["B"] * items[i]["C"]) for i in test_idx]
    eq4 = [items[i]["eq4"] for i in test_idx]

    # --- DIAGNOSTIC: confirm the first answer token is shared (P' vs true) ---
    if "diag" not in prog:
        shared = sum(1 for j in range(len(test_idx)) if _fp_ans_ids(pref[j])[0] == _fp_ans_ids(true[j])[0])
        prog["diag"] = {"first_tok_shared_frac": shared / len(test_idx)}
        save_json(ck, prog)
    log(f"WO#5.1c[{tag}]: first-answer-token shared(P',true) = {prog['diag']['first_tok_shared_frac']:.3f} "
        f"(≈1.0 ⇒ the WO#5 first-token metric was non-discriminative).")

    # --- clean full-product baselines: logprob(P')−logprob(true), no hook ---
    if "clean" not in prog:
        cl = []
        for j, i in enumerate(test_idx):
            cl.append(_fp_logprob(items[i]["tok4"], pref[j]) - _fp_logprob(items[i]["tok4"], true[j]))
        prog["clean"] = cl
        save_json(ck, prog)
    clean = prog["clean"]

    # --- fit C4 probes for the inject cells ---
    fit_c4 = {L: wo_fit_ridge_probe(_stack(train_idx, "resid_c4_eq", L), yP_tr) for L in {regL_c4, late_L}}

    def _delta_vec(rkey, L, fit, j):
        clean_r = items[test_idx[j]][rkey][L].astype(np.float32)
        return kk * (wo_inject_to_target(clean_r, fit["direction"], wo_probe_coord_for_value(fit, pref[j])) - clean_r)

    # interventions: (name, builds a per-item hook on the C4 OR C1 prompt + which prompt)
    def _swap_hook(j):
        donor = torch.tensor(items[test_idx[perm[j]]]["resid_c4_eq"][regL_c4].astype(np.float32),
                             device=model.cfg.device)
        return regL_c4, _wo_mk_patch_hook(donor, eq4[j]), items[test_idx[j]]["tok4"]

    def _inj_c4(L):
        def f(j):
            dv = torch.tensor(_delta_vec("resid_c4_eq", L, fit_c4[L], j), device=model.cfg.device)
            return L, _st_mk_add_hook(dv, eq4[j]), items[test_idx[j]]["tok4"]
        return f

    fit_c1 = wo_fit_ridge_probe(_stack(train_idx, "resid_equals", regL_c4), yP_tr)

    def _inj_c1(j):
        dv = torch.tensor(_delta_vec("resid_equals", regL_c4, fit_c1, j), device=model.cfg.device)
        return regL_c4, _st_mk_add_hook(dv, items[test_idx[j]]["equals"]), items[test_idx[j]]["tok1"]

    cells = [("swap_C4", _swap_hook), (f"inject_C4_L{regL_c4}", _inj_c4(regL_c4)),
             (f"inject_C4_L{late_L}", _inj_c4(late_L)), (f"inject_C1_L{regL_c4}", _inj_c1)]

    results = prog.get("cells", {})
    gsamp = list(range(min(int(CFG["wo_fp_gen_sample"]), len(test_idx))))
    for name, mk in cells:
        if name in results:
            continue
        dscore, gen_flip = [], []
        for j, i in enumerate(test_idx):
            L, hook, ptoks = mk(j)
            sc = _fp_logprob(ptoks, pref[j], L, hook) - _fp_logprob(ptoks, true[j], L, hook)
            dscore.append(sc - clean[j])
            if j in gsamp:                                    # greedy-decode flip on a sample
                gen_flip.append(1.0 if wo_parse_int(_fp_generate(ptoks, L, hook, int(CFG["wo_fp_gen_k"]))) == pref[j] else 0.0)
        ci = wo_paired_delta_ci(np.array(dscore), np.zeros(len(dscore)), n_boot=2000, seed=_FP_SEED)
        results[name] = {"fullprod_logprob_delta": float(np.mean(dscore)), "ci": [ci[0], ci[1]],
                         "emit_Pprime_rate": float(np.mean(gen_flip)) if gen_flip else None,
                         "n": len(dscore), "gen_n": len(gen_flip)}
        prog["cells"] = results
        save_json(ck, prog)
        log(f"WO#5.1c[{tag}] {name}: full-product logprob Δ(P'−true)={results[name]['fullprod_logprob_delta']:+.3f} "
            f"emit-P' rate={results[name]['emit_Pprime_rate']}")

    swap = results.get("swap_C4", {})
    inj_any = max((results[c].get("emit_Pprime_rate") or 0.0)
                  for c in results if c.startswith("inject_C4")) if results else 0.0
    swap_ok = (swap.get("emit_Pprime_rate") or 0.0) >= 0.5 or swap.get("fullprod_logprob_delta", 0) > 1.0
    if not swap_ok:
        verdict = "SITE_OR_METRIC_STILL_BROKEN"
    elif inj_any >= 0.5:
        verdict = "INJECT_WORKS"
    else:
        verdict = "DEAD_DIRECTION"
    out = {"tag": tag, "experiment": "WO5.1c_fullproduct", "reused_capture": True,
           "regL_c4": regL_c4, "late_L": late_L, "inject_k": kk,
           "first_tok_shared_frac": prog["diag"]["first_tok_shared_frac"],
           "cells": results, "verdict": verdict, "pair_sha": bundle.get("pair_sha"),
           "note": "Full-product metric (teacher-forced logprob + greedy-decode parse) — fixes the "
                   "non-discriminative first-token metric that produced WO#5/5.1/5.1b's degenerate nulls."}
    wo_save_result(f"causal_steering_fullproduct_{tag}.json", json.dumps(out, indent=2, default=str))
    return out


WO_STEER_FP = {}
for _tag in CFG["wo_fp_tags"]:
    WO_STEER_FP[_tag] = _fp_run(_tag)

_fp_rows = []
for _tag in CFG["wo_fp_tags"]:
    o = WO_STEER_FP[_tag]
    for name, r in o["cells"].items():
        _fp_rows.append({"tag": _tag, "verdict": o["verdict"], "intervention": name,
                         "first_tok_shared": round(o["first_tok_shared_frac"], 3),
                         "fullprod_logprob_delta": r["fullprod_logprob_delta"],
                         "emit_Pprime_rate": r["emit_Pprime_rate"]})
wo_save_result("causal_steering_fullproduct_summary.csv",
               wo_battery_csv(_fp_rows, ["tag", "verdict", "intervention", "first_tok_shared",
                                         "fullprod_logprob_delta", "emit_Pprime_rate"]))

print("\n========== WO#5.1c — FULL-PRODUCT METRIC (teacher-forced logprob + greedy-decode; reused WO#5) ==========")
for _tag in CFG["wo_fp_tags"]:
    o = WO_STEER_FP[_tag]
    print(f"\n[{_tag}]  first-answer-token shared(P',true) = {o['first_tok_shared_frac']:.3f}  "
          f"(≈1.0 confirms the old first-token metric was blind)")
    for name, r in o["cells"].items():
        print(f"   {name:<18} full-product logprob Δ(P'−true)={r['fullprod_logprob_delta']:+.3f}  "
              f"emit-P' rate={r['emit_Pprime_rate']}")
    print(f"   >>> VERDICT: {o['verdict']}")
print("============================================================================================================")
