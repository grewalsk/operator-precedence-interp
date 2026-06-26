# ============================================================================
# Phase 6 / WORK ORDER #5 — EXPERIMENT B: PROBE SELECTIVITY (pure CPU).
# ----------------------------------------------------------------------------
# Protects the decodability claim ("B·C is decodable at the '=' site, R^2≈0.96")
# against the Hewitt–Liang rebuttal a sharp reviewer WILL raise: does R^2=0.96 mean
# the product is REPRESENTED, or just that B and C are linearly present and ridge
# approximates their product over 400 items? Three controls, all with the FIXED
# folds=5 / ridge=1.0 instrument (wo_cv_r2), all CPU on the residual matrices Exp A
# cached to disk (NO model needed here):
#   1. Hewitt–Liang CONTROL TASK — a fixed random label per unique (B,C). selectivity
#      = R^2(real) − R^2(control); high selectivity ⇒ the probe reads STRUCTURE, not
#      its raw fitting capacity at this (n, d).
#   2. SHUFFLED-product target — permute y; R^2 must collapse to ~0 (certifies the
#      real R^2 is signal, not a probe/dimensionality artifact).
#   3. The DECISIVE linear-(B,C) baseline — a synthetic matrix with ONLY B and C
#      LINEARLY present (+ noise). If the real residual's product-R^2 EXCEEDS this
#      baseline, the residual genuinely contains the PRODUCT; if not, the product-R^2
#      is explained by the linear presence of B and C.
#      ── BAND CAVEAT (reported honestly, not hidden): over the POSITIVE band [20,49]
#         B·C is ALREADY ~linearly predictable from B and C (main effects dominate the
#         interaction), so this baseline need NOT collapse — the load-bearing quantity
#         is the CONTRAST real − baseline. The cell prints both and the verdict
#         (REPRESENTED / OPERANDS_ONLY / NOT_SELECTIVE) reflects it.
#
# Reads wo_steer_resid_{tag} (the Experiment-A capture pickle). If that artifact is
# absent (Exp A not run yet), the cell logs and skips — it never fabricates numbers.
# Deliverable: probe_selectivity.csv (tag, site, target, layer, R2_real,
# R2_control_task, R2_shuffled, R2_linearBC_baseline, selectivity, baseline_gap,
# verdict) + probe_selectivity_{tag}.json (with the per-layer R^2_real curves).
# ============================================================================
import json
import numpy as np

assert "wo_probe_selectivity" in globals() and "wo_selectivity_verdict" in globals() \
    and "wo_cv_r2" in globals(), (
    "WO#5 selectivity needs cell-76 WO#5 helpers (wo_probe_selectivity / "
    "wo_selectivity_verdict).")

CFG.setdefault("wo_sel_tags", list(CFG.get("wo_steer_tags", ["base", "instruct"])))
CFG.setdefault("wo_sel_seed", int(CFG.get("wo_steer_seed", 707)) + 11)
CFG.setdefault("wo_sel_min_n", 60)
WO_SEL_RIDGE = globals().get("WO_FSPROBE_RIDGE", 1.0)   # MATCH every other probe (comparability).
WO_SEL_FOLDS = globals().get("WO_FSPROBE_FOLDS", 5)

# (site, residual key, targets to probe). The HEADLINE decodability claims are
# B @ ')' and B·C @ '='; we also report the off-target probes for context, and the
# C4 '=' reference (where B·C IS used) as the decodability positive anchor.
WO_SEL_SPEC = [
    ("C1_rparen", "resid_rparen", ["B", "B_times_C", "C"]),
    ("C1_equals", "resid_equals", ["B_times_C", "B", "C"]),
    ("C4_equals", "resid_c4_eq", ["B_times_C"]),
]


def _sel_r2_real_curve(R, B, C, target, nL):
    """R^2_real by layer (real target only) to locate the best layer cheaply before
    running the full 4-control selectivity there."""
    y = {"B": B, "C": C, "B_times_C": B * C}[target]
    out = {}
    for L in range(nL):
        out[L] = wo_cv_r2(R[:, L, :].astype(np.float32), y, folds=WO_SEL_FOLDS, ridge=WO_SEL_RIDGE)
    return out


def _sel_for_tag(tag):
    ck = f"wo_steer_resid_{tag}"
    if not has_artifact(ck, "pickle"):
        log(f"WO#5 selectivity[{tag}]: capture artifact '{ck}' absent — run Experiment A (82l) "
            "first. Skipping (no fabricated numbers).")
        return None
    bundle = load_pickle(ck)
    _exp_sha = globals().get("WO_STEER_PAIR_SHA")
    if _exp_sha is not None and bundle.get("pair_sha") != _exp_sha:
        log(f"WO#5 selectivity[{tag}]: capture pair_sha {str(bundle.get('pair_sha'))[:12]} != "
            f"steering run {_exp_sha[:12]} — Experiment B is scoring a MISMATCHED capture; "
            "re-run Experiment A (82l) to refresh the residual cache.")
    items = bundle["items"]
    nL = bundle["n_layers"]
    n = len(items)
    if n < int(CFG["wo_sel_min_n"]):
        log(f"WO#5 selectivity[{tag}]: only n={n} captured items (< {int(CFG['wo_sel_min_n'])}) — "
            "under-powered; reporting with a warning.")
    B = np.array([it["B"] for it in items], dtype=float)
    C = np.array([it["C"] for it in items], dtype=float)
    seed = int(CFG["wo_sel_seed"])

    rows, curves = [], {}
    for site, rkey, targets in WO_SEL_SPEC:
        if not items or rkey not in items[0]:
            log(f"WO#5 selectivity[{tag}]: residual key '{rkey}' missing in capture — skip {site}.")
            continue
        R = np.stack([it[rkey] for it in items])            # [n, nL, d] fp16
        for tgt in targets:
            curve = _sel_r2_real_curve(R, B, C, tgt, nL)
            cand = [(L, v) for L, v in curve.items() if v is not None]
            bestL = max(cand, key=lambda t: t[1])[0] if cand else (nL // 2)
            X = R[:, bestL, :].astype(np.float32)
            row = wo_probe_selectivity(X, B, C, target=tgt, folds=WO_SEL_FOLDS,
                                       ridge=WO_SEL_RIDGE, seed=seed)
            verdict = wo_selectivity_verdict(row["R2_real"], row["R2_control_task"],
                                             row["R2_shuffled"], row["R2_linearBC_baseline"])
            rec = {"tag": tag, "site": site, "target": tgt, "layer": int(bestL), "n": int(n),
                   "layer_basis": "argmax CV-R2_real over all layers",
                   "R2_real": row["R2_real"], "R2_control_task": row["R2_control_task"],
                   "R2_shuffled": row["R2_shuffled"], "R2_linearBC_baseline": row["R2_linearBC_baseline"],
                   "selectivity": row["selectivity"], "baseline_gap": row["baseline_gap"],
                   "verdict": verdict["label"], "verdict_reason": verdict["reason"]}
            rows.append(rec)
            curves[f"{site}|{tgt}"] = {str(L): v for L, v in curve.items()}
            log(f"WO#5 selectivity[{tag}] {site}/{tgt} @L{bestL}: R2_real={_sfmt(row['R2_real'])} "
                f"ctrl={_sfmt(row['R2_control_task'])} shuf={_sfmt(row['R2_shuffled'])} "
                f"linBC={_sfmt(row['R2_linearBC_baseline'])} -> sel={_sfmt(row['selectivity'])} "
                f"gap={_sfmt(row['baseline_gap'])} [{verdict['label']}]")
    return {"tag": tag, "n": n, "n_layers": nL, "rows": rows, "r2_real_curves": curves,
            "ridge": WO_SEL_RIDGE, "folds": WO_SEL_FOLDS, "seed": seed,
            "pair_sha": bundle.get("pair_sha"),
            "note": ("bestL per (site,target) is argmax CV-R2_real over all layers, so R2_real (and "
                     "hence baseline_gap, since the linear-(B,C) baseline is layer-independent) carries "
                     "a mild layer-selection optimism. The FULL per-layer R2_real curve is in "
                     "r2_real_curves; read the gap against that curve, not the single max, and lean on "
                     "the band caveat. selectivity = R2_real - R2_control_task is computed at the SAME "
                     "layer so it is not differentially inflated.")}


def _sfmt(x):
    return "n/a" if x is None else f"{x:.3f}"


WO_SELECTIVITY = {}
_sel_all_rows = []
for _tag in CFG["wo_sel_tags"]:
    _r = _sel_for_tag(_tag)
    if _r is None:
        continue
    WO_SELECTIVITY[_tag] = _r
    _sel_all_rows.extend(_r["rows"])
    wo_save_result(f"probe_selectivity_{_tag}.json", json.dumps(_r, indent=2, default=str))

if _sel_all_rows:
    _sel_header = ["tag", "site", "target", "layer", "n", "R2_real", "R2_control_task",
                   "R2_shuffled", "R2_linearBC_baseline", "selectivity", "baseline_gap", "verdict"]
    wo_save_result("probe_selectivity.csv", wo_battery_csv(_sel_all_rows, _sel_header))

    print("\n================= WO#5 EXPERIMENT B — PROBE SELECTIVITY (CPU on cached residuals) =================")
    print(f"{'tag':<9}{'site':<11}{'tgt':<10}{'L':>4}{'R2real':>8}{'ctrl':>7}{'shuf':>7}{'linBC':>7}{'sel':>7}{'gap':>7}  verdict")
    for r in _sel_all_rows:
        print(f"{r['tag']:<9}{r['site']:<11}{r['target']:<10}{r['layer']:>4}"
              f"{_sfmt(r['R2_real']):>8}{_sfmt(r['R2_control_task']):>7}{_sfmt(r['R2_shuffled']):>7}"
              f"{_sfmt(r['R2_linearBC_baseline']):>7}{_sfmt(r['selectivity']):>7}{_sfmt(r['baseline_gap']):>7}"
              f"  {r['verdict']}")
    print("-------------------------------------------------------------------------------------------------")
    # headline reading: the product @ '=' on C1 (the claim Exp A causally tests).
    for _tag in WO_SELECTIVITY:
        head = [r for r in WO_SELECTIVITY[_tag]["rows"]
                if r["site"] == "C1_equals" and r["target"] == "B_times_C"]
        if head:
            r = head[0]
            print(f"  [{_tag}] C1 '=' product: {r['verdict']} — {r['verdict_reason']}")
    print("  NB BAND CAVEAT: over band [20,49], B·C is ~linearly predictable from B,C; if the linear-(B,C)")
    print("     baseline is also high, a high product-R^2 alone is NOT evidence of a represented product —")
    print("     read the selectivity (vs control task) and the real−baseline gap, not R2_real in isolation.")
    print("=================================================================================================")
else:
    print("\nWO#5 EXPERIMENT B — no capture artifacts found (run cell 82l first); nothing to report.")
