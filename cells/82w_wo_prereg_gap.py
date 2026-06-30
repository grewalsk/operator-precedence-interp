# ============================================================================
# Phase 6 / WORK ORDER #6 (Tier 1.2) — PRE-REGISTERED-LAYER GAP CHECK (GPU; REUSES 82u).
# ----------------------------------------------------------------------------
# The WO#6 band2 gap CI (cell 82u) reports the C1 '=' product gap (R2_real - linear-(B,C)
# baseline) at C1's OWN argmax-decodability layer. A reviewer kills it in one line:
# "you searched layers for C1 and read the gap at its most favourable one — multiple
# comparisons." This cell removes that degree of freedom: PRE-REGISTER the evaluation
# layer by an INDEPENDENT criterion — the C4 (working surface) B*C decodability PEAK,
# fixed across surfaces — and read the C1 product gap CI THERE. C4 is chosen because it
# is the surface the model SUCCEEDS on, so its decodability peak is a behavior-anchored,
# C1-independent layer. We also report the gap at C1's own argmax for transparency, so
# robustness-across-layer-choice is explicit rather than hidden.
#
# Outcomes (decision = the gap CI, not a fixed margin; §8 "the CI is the result"):
#   PREREG_GAP_EXCLUDES_ZERO -> the small product component survives layer pre-registration
#                               (the layer-selection objection is killed).
#   PREREG_GAP_BRACKETS_ZERO -> the component does NOT survive a pre-registered layer ->
#                               it was a layer-selection artifact; narrow the claim.
#
# Zero extra GPU cost: reuses 82u's CACHED band2 residual capture (ck wo_steer_resid_b2_*)
# and the validated cell-76 helpers (wo_prereg_layer, wo_gap_bootstrap, wo_prereg_gap_verdict).
# Band (20,49) stays PRIMARY (WO_BAND); band2 (2,99) is the headroom band 82u introduced.
# ============================================================================
import json
import numpy as np

assert "_st_capture" in globals() and "wo_build_pairs" in globals() \
    and "wo_cv_r2" in globals() and "wo_gap_bootstrap" in globals() \
    and "wo_prereg_layer" in globals() and "wo_prereg_gap_verdict" in globals(), (
    "WO#6 82w needs cell 82l (param _st_capture) + cell 76 (wo_prereg_layer / "
    "wo_gap_bootstrap / wo_prereg_gap_verdict). Run cells 76, 82l, 82u first.")

WO_BAND2 = tuple(CFG.get("wo_band2", (2, 99)))                 # same headroom band as 82u.
CFG.setdefault("wo_pg_tags", list(CFG.get("wo_br_tags", CFG.get("wo_steer_tags", ["base", "instruct"]))))
CFG.setdefault("wo_pg_seed", int(CFG.get("wo_br_seed", 919)))
CFG.setdefault("wo_pg_gap_nboot", int(CFG.get("wo_br_gap_nboot", 300)))
_PG_RIDGE = globals().get("WO_FSPROBE_RIDGE", 1.0)
_PG_FOLDS = globals().get("WO_FSPROBE_FOLDS", 5)
# Reuse 82u's band2 pairs if present (identical sha); else rebuild deterministically.
WO_PAIRS_PG = globals().get("WO_PAIRS_B2") or wo_build_pairs(n=WO_N, band=WO_BAND2, seed=WO_SEED)
WO_PAIRS_PG_SHA = wo_stim_hash(WO_PAIRS_PG)


def _pgf(x):
    return "n/a" if x is None else f"{float(x):.3f}"


def _pg_curve(R, y, nL):
    """Per-layer CV-R^2 decodability curve {layer: R^2 or None} (the pre-registration
    criterion is computed from this curve via wo_prereg_layer)."""
    return {L: wo_cv_r2(R[:, L, :].astype(np.float32), y, folds=_PG_FOLDS, ridge=_PG_RIDGE)
            for L in range(nL)}


def _pg_run(tag):
    ck = f"wo_pg_{tag}"
    fp = {"pairs_sha": WO_PAIRS_PG_SHA, "band2": list(WO_BAND2), "nboot": int(CFG["wo_pg_gap_nboot"]),
          "seed": int(CFG["wo_pg_seed"]), "ridge": _PG_RIDGE, "folds": _PG_FOLDS}
    if has_artifact(ck, "json"):
        prev = load_json(ck)
        if prev.get("fp") == fp:
            log(f"WO#6 prereg-gap[{tag}]: cached result reused.")
            return prev["out"]
        log(f"WO#6 prereg-gap[{tag}]: stale ckpt fingerprint — recompute.")

    # ---- REUSE 82u's cached band2 capture (no re-capture) ----
    bundle = _st_capture(tag, pairs=WO_PAIRS_PG, ck=f"wo_steer_resid_b2_{tag}")
    items = bundle["items"]
    nL = bundle["n_layers"]
    if not items or "resid_c4_eq" not in items[0] or "resid_equals" not in items[0]:
        log(f"WO#6 prereg-gap[{tag}]: capture missing C1/C4 '=' residuals — skipping.")
        return None
    B = np.array([it["B"] for it in items], dtype=float)
    C = np.array([it["C"] for it in items], dtype=float)
    y = B * C
    R_c1 = np.stack([it["resid_equals"] for it in items])      # [n, nL, d]  (failing surface)
    R_c4 = np.stack([it["resid_c4_eq"] for it in items])       # [n, nL, d]  (working surface)

    # ---- PRE-REGISTERED layer = C4 '=' B*C decodability peak (C1-independent) ----
    curve_c4 = _pg_curve(R_c4, y, nL)
    curve_c1 = _pg_curve(R_c1, y, nL)
    prereg_L = wo_prereg_layer(curve_c4)                       # the fixed evaluation layer
    c1_argmax_L = wo_prereg_layer(curve_c1)                    # C1's own favourable layer (transparency)
    if prereg_L is None:
        log(f"WO#6 prereg-gap[{tag}]: C4 decodability curve degenerate — cannot pre-register a layer.")
        return None
    log(f"WO#6 prereg-gap[{tag}]: pre-registered L*={prereg_L} (C4 peak R2={_pgf(curve_c4.get(prereg_L))}); "
        f"C1 own argmax L={c1_argmax_L} (R2={_pgf(curve_c1.get(c1_argmax_L))}).")

    _NB = int(CFG["wo_pg_gap_nboot"])
    seed = int(CFG["wo_pg_seed"])

    def _gap_at(R, L):
        if L is None:
            return None
        return wo_gap_bootstrap(R[:, L, :].astype(np.float32), B, C, target="B_times_C",
                                n_boot=_NB, folds=_PG_FOLDS, ridge=_PG_RIDGE, seed=seed)

    c1_gap_prereg = _gap_at(R_c1, prereg_L)                    # the headline: C1 gap at the fixed layer
    c4_gap_prereg = _gap_at(R_c4, prereg_L)                    # C4 gap at the same fixed layer
    c1_gap_argmax = _gap_at(R_c1, c1_argmax_L) if c1_argmax_L != prereg_L else c1_gap_prereg

    def _strip(g):                                            # drop the bulky per-draw arrays before save
        if not g:
            return None
        return {k: g[k] for k in ("gap_mean", "gap_ci", "gap_excludes_zero",
                                  "r2_real_ci", "r2_base_ci", "n_boot_used")}

    verdict = wo_prereg_gap_verdict(prereg_L, c1_argmax_L, c1_gap_prereg,
                                    c4_gap=c4_gap_prereg, c1_argmax_gap=c1_gap_argmax)
    out = {"tag": tag, "band2": list(WO_BAND2), "primary_band": list(WO_BAND),
           "pairs_sha": WO_PAIRS_PG_SHA, "n_used": bundle["n_used"],
           "prereg_layer": prereg_L, "c1_argmax_layer": c1_argmax_L,
           "prereg_criterion": "argmax C4 '=' B*C CV-R^2 (working surface, C1-independent)",
           "c4_peak_r2": _wo_num(curve_c4.get(prereg_L)) if "_wo_num" in globals() else (
               None if curve_c4.get(prereg_L) is None else float(curve_c4.get(prereg_L))),
           "c1_gap_at_prereg": _strip(c1_gap_prereg), "c4_gap_at_prereg": _strip(c4_gap_prereg),
           "c1_gap_at_c1_argmax": _strip(c1_gap_argmax), "verdict": verdict,
           "gap_nboot": _NB, "seed": seed, "ridge": _PG_RIDGE, "folds": _PG_FOLDS,
           "note": ("Pre-registered-layer gap check: the evaluation layer is FIXED at the C4 '=' B*C "
                    "decodability peak (independent of C1), then the C1 '=' product gap CI is read there. "
                    "Kills the 'you read C1 at its own argmax layer' objection to the 82u band2 gap CI. "
                    "Decision = the gap CI (excludes vs brackets 0), not a fixed margin.")}
    save_json(ck, {"fp": fp, "out": out})
    wo_save_result(f"prereg_gap_{tag}.json", json.dumps(wo_jsonsafe(out), indent=2))
    g = out["c1_gap_at_prereg"] or {}
    log(f"WO#6 prereg-gap[{tag}]: C1 '=' product gap @L*{prereg_L} = {_pgf(g.get('gap_mean'))} "
        f"95%CI=[{_pgf((g.get('gap_ci') or [None, None])[0])},{_pgf((g.get('gap_ci') or [None, None])[1])}] "
        f"-> {verdict['label']}")
    return out


WO_PREREG_GAP = {}
for _tag in CFG["wo_pg_tags"]:
    _r = _pg_run(_tag)
    if _r is not None:
        WO_PREREG_GAP[_tag] = _r

_pg_rows = []
for _tag, o in WO_PREREG_GAP.items():
    g = o.get("c1_gap_at_prereg") or {}
    gc = g.get("gap_ci") or [None, None]
    _pg_rows.append({"tag": _tag, "prereg_layer": o["prereg_layer"], "c1_argmax_layer": o["c1_argmax_layer"],
                     "c1_gap_at_prereg": g.get("gap_mean"), "ci_lo": gc[0], "ci_hi": gc[1],
                     "excludes_zero": g.get("gap_excludes_zero"), "verdict": o["verdict"]["label"]})
if _pg_rows:
    wo_save_result("prereg_gap_summary.csv",
                   wo_battery_csv(_pg_rows, ["tag", "prereg_layer", "c1_argmax_layer",
                                             "c1_gap_at_prereg", "ci_lo", "ci_hi",
                                             "excludes_zero", "verdict"]))

print("\n========== WO#6 (Tier 1.2) — PRE-REGISTERED-LAYER GAP CHECK (kills the layer-selection objection) ==========")
print("  L* = argmax C4 '=' B*C decodability (working surface, chosen INDEPENDENT of C1); read the C1 gap THERE.")
for _tag, o in WO_PREREG_GAP.items():
    v = o["verdict"]; g = o.get("c1_gap_at_prereg") or {}; gc = g.get("gap_ci") or [None, None]
    ga = o.get("c1_gap_at_c1_argmax") or {}
    print(f"\n  [{_tag}]  pre-registered L*={o['prereg_layer']} (C4 peak R2={_pgf(o.get('c4_peak_r2'))}); "
          f"C1 own argmax L={o['c1_argmax_layer']}")
    print(f"     C1 '=' product gap @L*={o['prereg_layer']}: {_pgf(g.get('gap_mean'))}  "
          f"95%CI=[{_pgf(gc[0])},{_pgf(gc[1])}]  excludes0={g.get('gap_excludes_zero')}")
    if o["prereg_layer"] != o["c1_argmax_layer"]:
        print(f"     (for transparency) C1 gap @ its OWN argmax L={o['c1_argmax_layer']}: "
              f"{_pgf(ga.get('gap_mean'))}  excludes0={ga.get('gap_excludes_zero')}  "
              f"-> robust across both layer choices: {v.get('robust_to_layer_choice')}")
    print(f"     >>> {v['label']}: {v['reason']}")
print("=============================================================================================================")
