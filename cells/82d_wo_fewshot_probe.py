# ============================================================================
# Phase 6 / WORK ORDER #3 — FEW-SHOT DECODABILITY PROBE (GPU; decodability ONLY).
# ----------------------------------------------------------------------------
# Question (decodability contrast, NO causal claim): is the operand B linearly
# decodable from the residual at the TEST expression's post-bracket ')' site of
# C1 '( 0 + B ) * C =' UNDER a few-shot prefix (shots in {0,2,4})? We know few-shot
# RECOVERS C1 accuracy to the C4 ceiling; the question is whether B's ENCODING also
# changes, or only its downstream use.
#
# THE LOAD-BEARING CONTRAST IS WITHIN THIS CELL: this cell re-derives its OWN 0-shot
# baseline (bare C1) on the shared probe sample, then compares 2-/4-shot to that
# 0-shot arm — same sample, same probe, same target, only the prefix differs. So
# the 0->2->4 trend (what wo_fsprobe_trend consumes) is apples-to-apples by
# construction. (The prior salvage figure ~0.74-0.96 was measured on the C1-WRONG
# subset, a different population — treat it as a loose sanity reference, NOT the
# baseline the trend is judged against.)
#   - R^2 ~ unchanged 0->4  => few-shot changes USE, not encoding (paper-strengthening).
#   - R^2 rises materially   => few-shot improves the REPRESENTATION (reframe; flagged).
#   - R^2 collapses          => the probe SITE is wrong (Hazard #1) — re-check first.
#
# This is a THIN orchestration over verified cell-76 logic:
#   - wo_fewshot_render  : the few-shot prompt builder (excludes the test pair).
#   - wo_last_rparen_index : Hazard #1 fix — the TEST ')' is the LAST ')', not the
#                            first (the first closes a SHOT's bracket).
#   - wo_cv_r2           : the SAME dual-ridge probe as the zero-shot run
#                          (folds=5, ridge=1.0) — apples-to-apples.
#   - wo_fsprobe_trend   : the 0->2->4 trend classifier (decision logic).
# Same probe / same target (B) / same site semantics / same layers / same pair
# sample across all conditions; ONLY the prefix differs (Hazard #3). One prompt per
# forward pass (variable few-shot lengths — no batching). Checkpointed per
# (tag, shots) so a disconnect resumes; runs on base AND instruct.
# ============================================================================
import json
import numpy as np
import torch

assert "WO_PAIRS" in globals() and "wo_fewshot_render" in globals() and "wo_cv_r2" in globals() \
    and "wo_last_rparen_index" in globals() and "wo_fsprobe_trend" in globals(), (
    "WO#3 probe needs WO_PAIRS (cell 78) + wo_fewshot_render/wo_cv_r2/"
    "wo_last_rparen_index/wo_fsprobe_trend (cell 76).")

CFG.setdefault("wo_fsprobe_seed", 404)
CFG.setdefault("wo_fsprobe_n", len(WO_PAIRS))     # probe ALL shared pairs by default.
WO_FSPROBE_SHOTS = [0, 2, 4]
WO_FSPROBE_RIDGE = 1.0    # MATCH the zero-shot salvage probe EXACTLY (Hazard #3).
WO_FSPROBE_FOLDS = 5      # MATCH the zero-shot salvage probe EXACTLY (Hazard #3).
WO_FSPROBE_MIN_N = 80

_fsp_renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
def _fsp_gt(b, c):
    return b * c

# SAME pair sample across every (tag, shots) so per-pair pairing is possible.
_fsp_rng = np.random.default_rng(int(CFG["wo_fsprobe_seed"]))
_fsp_n = min(len(WO_PAIRS), int(CFG["wo_fsprobe_n"]))
_fsp_idx = sorted(int(i) for i in _fsp_rng.choice(len(WO_PAIRS), size=_fsp_n, replace=False))
WO_FSPROBE_PAIRS = [WO_PAIRS[i] for i in _fsp_idx]
WO_FSPROBE_PAIR_SHA = wo_stim_hash(WO_FSPROBE_PAIRS)
log(f"WO#3 few-shot probe: {_fsp_n} pairs (sha {WO_FSPROBE_PAIR_SHA[:12]}), "
    f"shots {WO_FSPROBE_SHOTS}, probe ridge={WO_FSPROBE_RIDGE} folds={WO_FSPROBE_FOLDS}.")


def _fsp_rparen_last(tokens):
    """Index of the TEST expression's ')' = the LAST ')' (Hazard #1)."""
    strs = [tokenizer.decode([t]).strip() for t in tokens[0].tolist()]
    return wo_last_rparen_index(strs)


def _fsp_site_ok(tokens, pos, B):
    """Correctness check for Hazard #1: walk back from the ')' at `pos`, skipping
    pure-space tokens, accumulating the integer that precedes it, and require it to
    equal the TEST B (i.e. the ')' really closes '( 0 + B )' for THIS test pair,
    not a shot's). Tokenization-robust (B may be one or several digit tokens)."""
    if pos is None:
        return False
    ids = tokens[0].tolist()
    digits = ""
    j = pos - 1
    while j >= 0:
        s = tokenizer.decode([ids[j]]).strip()
        if s == "":
            j -= 1; continue
        if s.isdigit():
            digits = s + digits; j -= 1; continue
        break
    return digits == str(int(B))


@torch.no_grad()
def _fsp_probe(tag, shots):
    """Probe B-decodability at the TEST ')' site for one (tag, shots). Cached per
    (tag, shots) so a disconnect resumes by skipping completed units."""
    ck = f"wo_fsprobe_{tag}_{shots}"
    if has_artifact(ck, "json"):
        log(f"WO#3 probe [{tag}/{shots}-shot]: cached — reused.")
        return load_json(ck)

    wo_load_model(tag)
    n_layers = model.cfg.n_layers
    feats = {L: [] for L in range(n_layers)}
    Bvals = []
    n_skip = 0
    seed_base = int(CFG["wo_fsprobe_seed"]) + 1000 * int(shots)
    for i, (B, C) in enumerate(WO_FSPROBE_PAIRS):
        if shots == 0:
            prompt = _fsp_renderC1(B, C)                   # 0-shot baseline = bare C1.
        else:
            # PER-ITEM demos (deterministic), excluding the test pair.
            prompt = wo_fewshot_render(_fsp_renderC1, _fsp_gt, shots, (B, C),
                                       WO_PAIRS, seed=seed_base + i)
        # Hazard #2 (real guard, both arms): the prompt must end at '=' with NO
        # trailing space, else the scored next-token id shifts on Llama's tokenizer.
        assert prompt.endswith("=") and not prompt.endswith(" "), (
            f"prompt must end at '=' with no trailing space (Hazard #2): {prompt!r}")
        tok = model.to_tokens(prompt)
        pos = _fsp_rparen_last(tok)
        if not _fsp_site_ok(tok, pos, B):                 # Hazard #1 abort-per-item.
            n_skip += 1; continue
        _, cache = model.run_with_cache(
            tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        for L in range(n_layers):
            # fp32 capture (vs the salvage's fp16 memory round-trip) — more accurate
            # and CONSISTENT across all shot levels here, which is what the within-cell
            # 0->2->4 trend requires; the per-layer R^2 only differs by ~<=0.01 from fp16.
            feats[L].append(cache[f"blocks.{L}.hook_resid_post"][0, pos, :].float().cpu().numpy())
        Bvals.append(int(B))
        del cache

    Bv = np.array(Bvals, dtype=float)
    decod = {L: wo_cv_r2(np.array(feats[L]), Bv, folds=WO_FSPROBE_FOLDS, ridge=WO_FSPROBE_RIDGE)
             for L in range(n_layers) if len(feats[L]) >= 7}
    cand = [(L, v) for L, v in decod.items() if v is not None]
    best_layer, r2_best = (max(cand, key=lambda t: t[1]) if cand else (None, None))
    out = {
        "tag": tag, "shots": int(shots), "n_used": len(Bvals), "n_skipped": n_skip,
        "n_used_ok": bool(len(Bvals) >= WO_FSPROBE_MIN_N),
        "r2_by_layer": {str(L): v for L, v in decod.items()},
        "best_layer": (int(best_layer) if best_layer is not None else None),
        "r2_best": (float(r2_best) if r2_best is not None else None),
        "seed": seed_base, "pair_sha": WO_FSPROBE_PAIR_SHA,
        "ridge": WO_FSPROBE_RIDGE, "folds": WO_FSPROBE_FOLDS,
    }
    save_json(ck, out)
    log(f"WO#3 probe [{tag}/{shots}-shot]: n_used={len(Bvals)} (skip={n_skip}) "
        f"R^2_best={out['r2_best']} @ layer {out['best_layer']}")
    return out


# --- run both tags x all shot counts -------------------------------------
WO_FSPROBE = {"base": {}, "instruct": {}}
for _tag in ("base", "instruct"):
    for _shots in WO_FSPROBE_SHOTS:
        WO_FSPROBE[_tag][_shots] = _fsp_probe(_tag, _shots)

# --- per-tag write + 0->2->4 trend classification ------------------------
_fsp_summary_rows = []
for _tag in ("base", "instruct"):
    r = WO_FSPROBE[_tag]
    _r0, _r2, _r4 = r[0]["r2_best"], r[2]["r2_best"], r[4]["r2_best"]
    _label, _detail = wo_fsprobe_trend(_r0, _r2, _r4)
    _out = {
        "tag": _tag,
        "r2_best_by_shots": {str(s): r[s]["r2_best"] for s in WO_FSPROBE_SHOTS},
        "best_layer_by_shots": {str(s): r[s]["best_layer"] for s in WO_FSPROBE_SHOTS},
        "n_used_by_shots": {str(s): r[s]["n_used"] for s in WO_FSPROBE_SHOTS},
        "n_skipped_by_shots": {str(s): r[s]["n_skipped"] for s in WO_FSPROBE_SHOTS},
        "by_shots": {str(s): r[s] for s in WO_FSPROBE_SHOTS},
        "trend": _label, "reading": _detail,
        "pair_sha": WO_FSPROBE_PAIR_SHA, "ridge": WO_FSPROBE_RIDGE, "folds": WO_FSPROBE_FOLDS,
        "note": ("Decodability contrast ONLY (no causal claim). 0-shot here is the bare-C1 "
                 "probe on the shared sample; the only thing that varies across conditions is "
                 "the few-shot prefix."),
    }
    wo_save_result(f"fewshot_decodability_{_tag}.json", json.dumps(_out, indent=2, default=str))
    save_json(f"wo_fsprobe_summary_{_tag}", _out)
    for s in WO_FSPROBE_SHOTS:
        _fsp_summary_rows.append({"tag": _tag, "shots": s, "best_layer": r[s]["best_layer"],
                                  "r2_best": r[s]["r2_best"], "n_used": r[s]["n_used"],
                                  "n_used_ok": r[s]["n_used_ok"]})

_fsp_csv = wo_battery_csv(
    _fsp_summary_rows, ["tag", "shots", "best_layer", "r2_best", "n_used", "n_used_ok"])
wo_save_result("fewshot_decodability_summary.csv", _fsp_csv)

print("\n================= WO#3 — FEW-SHOT B-DECODABILITY @ TEST ')' SITE =================")
print(f"{'tag':<9}{'shots':>6}{'n_used':>8}{'best_L':>8}{'R^2_best':>10}")
for r in _fsp_summary_rows:
    _r2s = "n/a" if r["r2_best"] is None else f"{r['r2_best']:.3f}"
    _warn = "" if r.get("n_used_ok", True) else f"  ⚠ n<{WO_FSPROBE_MIN_N} (under-powered)"
    print(f"{r['tag']:<9}{r['shots']:>6}{r['n_used']:>8}{str(r['best_layer']):>8}{_r2s:>10}{_warn}")
print("---------------------------------------------------------------------------------")
for _tag in ("base", "instruct"):
    print(f"  [{_tag}] {WO_FSPROBE[_tag][0].get('r2_best')}(0) -> "
          f"{WO_FSPROBE[_tag][2].get('r2_best')}(2) -> {WO_FSPROBE[_tag][4].get('r2_best')}(4): "
          f"{globals().get('wo_fsprobe_trend')(WO_FSPROBE[_tag][0]['r2_best'], WO_FSPROBE[_tag][2]['r2_best'], WO_FSPROBE[_tag][4]['r2_best'])[0]}")
print("=================================================================================")
