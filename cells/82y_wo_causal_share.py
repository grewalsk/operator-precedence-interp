# ============================================================================
# Phase 6 / WORK ORDER #6 (Tier 1.3) — CAUSAL-SHARE DORMANT CERTIFICATION (GPU; REUSES 82p/82q).
# ----------------------------------------------------------------------------
# The Makelov CAUSAL half (2311.17030). The static decompose (82s) splits the decode
# direction w-hat by the UNEMBEDDING subspace; that broke here because the answer's first
# token is shared across products (first_tok_shared=1.0) so the readout basis collapsed to
# rank 0 -> all-null. This cell certifies dormancy CAUSALLY instead, with no unembedding
# basis: take the guaranteed-causal full-residual SWAP at the C4 '=' site, decompose its
# delta (donor - clean) into the component ALONG the decode direction w-hat and the
# orthogonal remainder, and apportion the swap's ANSWER-EFFECTIVE effect (full-product
# emit-P' / logprob Δ) between them:
#   effect_full  = apply the whole swap delta            (total causal effect)
#   effect_wonly = apply ONLY the w-hat component         (decodable direction alone)
#   effect_perp  = apply the swap delta with w-hat ABLATED (does the effect survive?)
# Verdict (wo_causal_share_verdict):
#   NO_CAUSAL_HANDLE            — effect_full ~ 0: the swap itself doesn't move the output.
#       The '=' site carries NO answer-effective effect; the whole site is dormant (the
#       product is re-derived from the operands; Stolfo). This is the committed reality
#       (lateswap emit-P'=0 @ every layer) and a STRONGER dormancy statement than a dormant
#       direction. Pair with the operand-route positive control (82r/82s path patch).
#   DORMANT_DIRECTION_CERTIFIED — swap moves the output, w-hat carries ~none of it, effect
#       survives ablating w-hat: the DECODABLE direction is causally dormant (decoding !=
#       causal direction) — the precise Makelov certification.
#   DIRECTION_CARRIES_EFFECT    — w-hat carries most of the effect: logit-coupled, not dormant.
# Primary metric = greedy-decode emit-P' (the gold metric, §5); full-product logprob Δ is a
# secondary panel. Reuses cached residuals + 82p's _fp_logprob/_fp_generate. Checkpointed.
# ============================================================================
import json
import numpy as np
import torch

assert "_st_capture" in globals() and "_st_split" in globals() and "_cal_derange_perm" in globals() \
    and "_st_mk_add_hook" in globals() and "_fp_logprob" in globals() and "_fp_generate" in globals() \
    and "wo_fit_ridge_probe" in globals() and "wo_direction_split" in globals() \
    and "wo_causal_share_verdict" in globals() and "wo_parse_int" in globals() \
    and "wo_paired_delta_ci" in globals(), (
    "WO#6 82y needs cells 82l + 82n + 82p (for _fp_logprob/_fp_generate/_st_mk_add_hook) and "
    "the cell-76 wo_direction_split / wo_causal_share_verdict. Run cells 76, 82l, 82n, 82p first.")

CFG.setdefault("wo_cs_tags", list(CFG.get("wo_fp_tags", ["base", "instruct"])))
CFG.setdefault("wo_cs_gen_sample", 60)                   # #items for greedy emit-P' (the primary metric).
CFG.setdefault("wo_cs_gen_k", int(globals().get("WO_MAX_NEW_TOKENS", 8)))
CFG.setdefault("wo_cs_emit_floor", 0.5)                  # full swap must flip >= this to be a causal handle.
CFG.setdefault("wo_cs_logprob_floor", 1.0)              # nats: full swap logprob Δ floor (secondary panel).
CFG.setdefault("wo_cs_share_dormant", 0.2)
CFG.setdefault("wo_cs_share_coupled", 0.6)
_CS_SEED = int(CFG.get("wo_steer_seed", 707))
_CS_STRIDE = int(CFG.get("wo_steer_layer_stride", 2))


def _cs_ratio_ci(num_mask, den_mask, n_boot=2000, seed=0):
    """Paired bootstrap CI for mean(num)/mean(den) (the w_share = effect_wonly/effect_full),
    resampling items jointly. (None,None) if den mean ~ 0 on too many draws."""
    a = np.asarray(num_mask, dtype=float); b = np.asarray(den_mask, dtype=float)
    n = a.size
    if n == 0 or b.size != n:
        return (None, None)
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        db = b[idx].mean()
        if abs(db) > 1e-9:
            vals.append(a[idx].mean() / db)
    if len(vals) < max(10, int(0.25 * n_boot)):
        return (None, None)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


@torch.no_grad()
def _cs_run(tag):
    ck = f"wo_cs_{tag}"
    prog = load_json(ck) if has_artifact(ck, "json") else {}
    bundle = _st_capture(tag)                                  # REUSED (no re-capture)
    items = bundle["items"]
    nL = bundle["n_layers"]
    train_idx, test_idx = _st_split(len(items), _CS_SEED)
    yP_tr = np.array([items[i]["B"] * items[i]["C"] for i in train_idx], float)

    def _stack(idxs, rkey, L):
        return np.stack([items[i][rkey][L].astype(np.float32) for i in idxs])

    # swap layer: reuse 82q's best lateswap layer if present, else the late band peak by
    # full-swap emit; default to ~0.85 depth. (The committed lateswap best_layer is where
    # the swap has the most chance to move the answer.)
    L_swap = None
    if has_artifact(f"wo_ls_{tag}", "json"):
        ls = load_json(f"wo_ls_{tag}")
        rows = ls.get("rows", {})
        if rows:
            L_swap = int(max(rows.values(), key=lambda r: r.get("emit_Pprime_rate", 0.0))["layer"])
    if L_swap is None and has_artifact(f"causal_steering_lateswap_{tag}", "json"):
        bl = load_json(f"causal_steering_lateswap_{tag}").get("best_layer")
        L_swap = int(bl) if bl is not None else None
    if L_swap is None:
        L_swap = int(round(0.85 * (nL - 1)))
    log(f"WO#6 causal-share[{tag}]: swap layer L_swap={L_swap}.")

    fit_c4 = wo_fit_ridge_probe(_stack(train_idx, "resid_c4_eq", L_swap), yP_tr)
    if fit_c4 is None:
        log(f"WO#6 causal-share[{tag}]: C4 probe degenerate at L{L_swap} — cannot define w-hat; skipping.")
        return None
    what = fit_c4["direction"]

    perm = _cal_derange_perm(len(test_idx), _CS_SEED + 3)
    pref = [int(items[test_idx[perm[j]]]["B"] * items[test_idx[perm[j]]]["C"]) for j in range(len(test_idx))]
    true = [int(items[i]["B"] * items[i]["C"]) for i in test_idx]
    gsamp = list(range(min(int(CFG["wo_cs_gen_sample"]), len(test_idx))))

    # within-surface (C4) clean baseline for the logprob panel.
    if "clean" not in prog:
        prog["clean"] = [_fp_logprob(items[i]["tok4"], pref[j]) - _fp_logprob(items[i]["tok4"], true[j])
                         for j, i in enumerate(test_idx)]
        save_json(ck, prog)
    clean = prog["clean"]

    # per-item swap delta and its w-hat split (geometry).
    def _delta(j):
        i = test_idx[j]
        donor = items[test_idx[perm[j]]]["resid_c4_eq"][L_swap].astype(np.float32)
        cleanr = items[i]["resid_c4_eq"][L_swap].astype(np.float32)
        return donor - cleanr

    # three interventions on the SAME swap delta: full / w-only / perp.
    def _vec(j, mode):
        d = _delta(j)
        sp = wo_direction_split(d, what)
        if sp is None:
            return None, None
        if mode == "full":
            return d, sp["w_frac"]
        if mode == "wonly":
            return sp["w_comp"], sp["w_frac"]
        return sp["perp"], sp["w_frac"]                       # perp = w-hat ablated

    cells = prog.get("cells", {})
    wfracs = prog.get("wfracs", [])
    for mode in ("full", "wonly", "perp"):
        if mode in cells:
            continue
        dlog, emit = [], []
        wf = []
        for j, i in enumerate(test_idx):
            vec, frac = _vec(j, mode)
            if vec is None:
                dlog.append(0.0)
                continue
            if mode == "full":
                wf.append(frac)
            hook = _st_mk_add_hook(torch.tensor(vec, device=model.cfg.device), items[i]["eq4"])
            sc = _fp_logprob(items[i]["tok4"], pref[j], L_swap, hook) \
                - _fp_logprob(items[i]["tok4"], true[j], L_swap, hook)
            dlog.append(sc - clean[j])
            if j in gsamp:
                emit.append(1.0 if wo_parse_int(_fp_generate(items[i]["tok4"], L_swap, hook,
                                                            int(CFG["wo_cs_gen_k"]))) == pref[j] else 0.0)
        d = np.array(dlog, dtype=float)
        lo, hi = wo_paired_delta_ci(d, np.zeros(len(d)), n_boot=2000, seed=_CS_SEED)
        cells[mode] = {"logprob_delta": float(d.mean()), "logprob_ci": [lo, hi],
                       "emit_Pprime_rate": (float(np.mean(emit)) if emit else None),
                       "emit_mask": emit, "n": len(d), "gen_n": len(emit)}
        if mode == "full":
            wfracs = [float(x) for x in wf]
            prog["wfracs"] = wfracs
        prog["cells"] = cells
        save_json(ck, prog)
        log(f"WO#6 causal-share[{tag}] {mode}: emit-P'={cells[mode]['emit_Pprime_rate']} "
            f"logprobΔ={cells[mode]['logprob_delta']:+.3f}")

    full, wonly, perp = cells["full"], cells["wonly"], cells["perp"]
    # --- primary verdict on emit-P' (the gold metric) ---
    ef, ew, ep = full["emit_Pprime_rate"], wonly["emit_Pprime_rate"], perp["emit_Pprime_rate"]
    wsh_ci = (None if (ef is None or full["gen_n"] == 0)
              else _cs_ratio_ci(wonly["emit_mask"], full["emit_mask"], seed=_CS_SEED + 5))
    verdict = wo_causal_share_verdict(ef, ep, ew, effect_floor=float(CFG["wo_cs_emit_floor"]),
                                      share_dormant=float(CFG["wo_cs_share_dormant"]),
                                      share_coupled=float(CFG["wo_cs_share_coupled"]), w_share_ci=wsh_ci)
    # --- secondary panel on full-product logprob Δ ---
    verdict_lp = wo_causal_share_verdict(full["logprob_delta"], perp["logprob_delta"], wonly["logprob_delta"],
                                         ci_full=full["logprob_ci"], ci_perp=perp["logprob_ci"],
                                         ci_wonly=wonly["logprob_ci"], effect_floor=float(CFG["wo_cs_logprob_floor"]),
                                         share_dormant=float(CFG["wo_cs_share_dormant"]),
                                         share_coupled=float(CFG["wo_cs_share_coupled"]))
    mean_wfrac = (float(np.mean(wfracs)) if wfracs else None)
    for c in (full, wonly, perp):                            # drop the per-item emit mask from the saved JSON
        c.pop("emit_mask", None)
    out = {"tag": tag, "experiment": "WO6_T1.3_causal_share", "reused_capture": True,
           "swap_layer": int(L_swap), "ref_site": "C4_equals",
           "decode_direction": "C4 '=' B*C ridge probe (the decodable product direction)",
           "mean_w_frac_geometry": mean_wfrac,
           "full": full, "wonly": wonly, "perp": perp,
           "verdict_emit": verdict, "verdict_logprob": verdict_lp,
           "headline_verdict": verdict["label"], "pair_sha": bundle.get("pair_sha"),
           "note": ("Causal-share Makelov certification: apportion the full-residual swap's "
                    "answer-effective effect between the decode direction w-hat and its orthogonal "
                    "complement. Primary metric = greedy emit-P'; effect_floor guards against "
                    "certifying when the swap itself is inert (-> NO_CAUSAL_HANDLE, the whole site "
                    "dormant). mean_w_frac_geometry is the magnitude share of the swap along w-hat "
                    "(descriptive); the CAUSAL share is effect_wonly/effect_full.")}
    wo_save_result(f"causal_share_{tag}.json", json.dumps(wo_jsonsafe(out), indent=2))
    return out


WO_CAUSAL_SHARE = {}
for _tag in CFG["wo_cs_tags"]:
    _r = _cs_run(_tag)
    if _r is not None:
        WO_CAUSAL_SHARE[_tag] = _r

_cs_rows = []
for _tag, o in WO_CAUSAL_SHARE.items():
    for mode in ("full", "wonly", "perp"):
        c = o[mode]
        _cs_rows.append({"tag": _tag, "swap_layer": o["swap_layer"], "component": mode,
                         "emit_Pprime_rate": c["emit_Pprime_rate"], "logprob_delta": c["logprob_delta"],
                         "headline_verdict": o["headline_verdict"]})
if _cs_rows:
    wo_save_result("causal_share_summary.csv",
                   wo_battery_csv(_cs_rows, ["tag", "swap_layer", "component",
                                             "emit_Pprime_rate", "logprob_delta", "headline_verdict"]))

print("\n========== WO#6 (Tier 1.3) — CAUSAL-SHARE DORMANT CERTIFICATION (apportion the swap effect onto w-hat) ==========")
print("  effect_full = whole swap; effect_wonly = decode-direction component only; effect_perp = w-hat ablated.")
for _tag, o in WO_CAUSAL_SHARE.items():
    v = o["verdict_emit"]; vl = o["verdict_logprob"]
    print(f"\n  [{_tag}]  swap layer L={o['swap_layer']}  (mean |w| share of swap = {o['mean_w_frac_geometry']})")
    print(f"     emit-P':  full={o['full']['emit_Pprime_rate']}  w-only={o['wonly']['emit_Pprime_rate']}  "
          f"perp(w-ablated)={o['perp']['emit_Pprime_rate']}  -> w_share={v.get('w_share')}")
    print(f"     logprobΔ: full={o['full']['logprob_delta']:+.3f}  w-only={o['wonly']['logprob_delta']:+.3f}  "
          f"perp={o['perp']['logprob_delta']:+.3f}")
    print(f"     >>> [emit] {v['label']}: {v['reason']}")
    print(f"     >>> [logprob] {vl['label']}")
print("==================================================================================================================")
