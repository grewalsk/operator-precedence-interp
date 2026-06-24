# ============================================================================
# Phase 6 / WORK ORDER #4 — §3.2 MULTI-POSITION DECODABILITY MAP (GPU; DECODABILITY
# ONLY — no causal claim).
# ----------------------------------------------------------------------------
# Extends the WO#3 ')'-site probe to FOUR sites on the ZERO-SHOT C1 surface
# '( 0 + B ) * C =':   the post-bracket ')' (source) · the '*' operator · the 'C'
# operand · the final '=' (answer cue). Targets: B and the product B*C. Same
# dual-ridge CV-R^2 (folds=5, ridge=1.0) as every other probe (comparability). C4
# '( B * C ) =' is probed at ITS '=' as the POSITIVE REFERENCE — the site where
# B*C IS decodable at the answer cue.
#
# THE LOAD-BEARING, CONFOUND-FREE CONTRAST (entirely WITHIN zero-shot, length-
# matched): is B decodable at ')' while B*C is NOT decodable at the '=' in C1
# (failing), yet B*C IS decodable at the '=' in C4 (working)? That localizes the
# breakdown to BINDING/ROUTING at the answer site, not encoding — with NO causal
# claim. The C1 vs C4 contrast is length-matched, so it carries no dilution
# confound (unlike any 0-shot vs few-shot comparison — WO#3's lesson).
#
# Site finders are the unit-tested cell-76 helpers (wo_locate_c1_sites locates
# ')'/'*'/C/'=' by DECODE-AND-WALK on token content and ASSERTS the roles; a pair
# whose window doesn't verify is SKIPPED, with the skip count reported). HF-fallback
# sanity: the ')'-site B R^2 here must reproduce the WO#3 0-shot best-layer R^2
# before a new model's map is trusted. Checkpointed per tag; runs on base+instruct
# by default (CFG['wo_posmap_tags']; add cross-models that passed parity).
# ============================================================================
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

assert "WO_FSPROBE_PAIRS" in globals() and "wo_locate_c1_sites" in globals() \
    and "wo_cv_r2" in globals() and "wo_last_index" in globals(), (
    "WO#4 position map needs WO_FSPROBE_PAIRS (cell 82d) + cell-76 site finders.")

CFG.setdefault("wo_posmap_tags", ["base", "instruct"])
WO_PM_POSITIONS = ["rparen", "star", "c_operand", "equals"]
WO_PM_TARGETS = ["B", "BC"]
WO_PM_RIDGE = globals().get("WO_FSPROBE_RIDGE", 1.0)   # MATCH every other probe.
WO_PM_FOLDS = globals().get("WO_FSPROBE_FOLDS", 5)
WO_PM_MIN_N = 7

_pm_renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
_pm_renderC4 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C4"]


@torch.no_grad()
def _pm_probe(tag):
    """B and B*C decodability at the four C1 sites + the C4 '=' reference, per layer.
    Cached per tag so a disconnect resumes."""
    ck = f"wo_posmap_{tag}"
    if has_artifact(ck, "json"):
        log(f"WO#4 position map [{tag}]: cached — reused.")
        return load_json(ck)

    wo_load_model(tag)
    nL = model.cfg.n_layers
    feats = {p: {L: [] for L in range(nL)} for p in WO_PM_POSITIONS}
    feats_c4eq = {L: [] for L in range(nL)}
    Bvals, BCvals, BCvals_c4 = [], [], []
    n_skip_c1, n_skip_c4 = 0, 0

    for (B, C) in WO_FSPROBE_PAIRS:
        # ---- C1: locate the four sites by content-walk; skip if roles don't verify ----
        p1 = _pm_renderC1(B, C)
        assert p1.endswith("=") and not p1.endswith(" ")          # Hazard #2 (Llama tok).
        tok1 = model.to_tokens(p1)
        strs1 = [tokenizer.decode([t]).strip() for t in tok1[0].tolist()]
        loc = wo_locate_c1_sites(strs1, B, C)
        if not loc["ok"]:
            n_skip_c1 += 1
        else:
            idxmap = {"rparen": loc["rparen"], "star": loc["star"],
                      "c_operand": loc["c_operand"], "equals": loc["equals"]}
            _, cache1 = model.run_with_cache(
                tok1, names_filter=lambda nm: nm.endswith("hook_resid_post"))
            for L in range(nL):
                h = cache1[f"blocks.{L}.hook_resid_post"][0]
                for p in WO_PM_POSITIONS:
                    feats[p][L].append(h[idxmap[p], :].float().cpu().numpy())
            del cache1
            Bvals.append(int(B)); BCvals.append(int(B * C))

        # ---- C4 positive reference: B*C at the answer '=' ----
        p4 = _pm_renderC4(B, C)
        assert p4.endswith("=") and not p4.endswith(" ")
        tok4 = model.to_tokens(p4)
        strs4 = [tokenizer.decode([t]).strip() for t in tok4[0].tolist()]
        eq4 = wo_last_index(strs4, "=")
        if eq4 is None:
            n_skip_c4 += 1
        else:
            _, cache4 = model.run_with_cache(
                tok4, names_filter=lambda nm: nm.endswith("hook_resid_post"))
            for L in range(nL):
                feats_c4eq[L].append(cache4[f"blocks.{L}.hook_resid_post"][0, eq4, :].float().cpu().numpy())
            del cache4
            BCvals_c4.append(int(B * C))

    def _probe_target(feat_by_layer, target):
        tv = np.asarray(target, dtype=float)
        r2 = {L: wo_cv_r2(np.asarray(feat_by_layer[L]), tv, folds=WO_PM_FOLDS, ridge=WO_PM_RIDGE)
              for L in range(nL) if len(feat_by_layer[L]) >= WO_PM_MIN_N}
        cand = [(L, v) for L, v in r2.items() if v is not None]
        bl, rb = (max(cand, key=lambda t: t[1]) if cand else (None, None))
        return {"best_layer": (int(bl) if bl is not None else None),
                "cv_r2": (float(rb) if rb is not None else None),
                "r2_by_layer": {str(L): v for L, v in r2.items()}}

    Bv, BCv = np.asarray(Bvals, float), np.asarray(BCvals, float)
    positions = {}
    for p in WO_PM_POSITIONS:
        positions[p] = {"B": _probe_target(feats[p], Bv),
                        "BC": _probe_target(feats[p], BCv)}
    c4_eq = _probe_target(feats_c4eq, np.asarray(BCvals_c4, float))

    out = {
        "tag": tag, "n_layers": nL,
        "n_used_c1": len(Bvals), "n_skipped_c1": n_skip_c1,
        "n_used_c4": len(BCvals_c4), "n_skipped_c4": n_skip_c4,
        "positions": positions, "c4_equals_BC": c4_eq,
        "ridge": WO_PM_RIDGE, "folds": WO_PM_FOLDS, "pair_sha": WO_FSPROBE_PAIR_SHA,
        "note": ("Decodability ONLY (no causal claim). Headline contrast is WITHIN zero-shot "
                 "(C1 ')' vs C1 '=' vs C4 '=') — length-matched, no dilution confound."),
    }
    save_json(ck, out)
    log(f"WO#4 position map [{tag}]: n_c1={len(Bvals)} (skip {n_skip_c1}), "
        f"n_c4={len(BCvals_c4)} (skip {n_skip_c4}); "
        f"B@) R^2={positions['rparen']['B']['cv_r2']}, "
        f"BC@= (C1) R^2={positions['equals']['BC']['cv_r2']}, "
        f"BC@= (C4) R^2={c4_eq['cv_r2']}")
    return out


WO_POSMAP = {}
for _tag in CFG["wo_posmap_tags"]:
    WO_POSMAP[_tag] = _pm_probe(_tag)

# --- HF-fallback / probe-plumbing sanity: ')'-site B R^2 must reproduce WO#3 ----
for _tag in CFG["wo_posmap_tags"]:
    _here = WO_POSMAP[_tag]["positions"]["rparen"]["B"]["cv_r2"]
    _wo3 = None
    try:
        _wo3 = globals().get("WO_FSPROBE", {}).get(_tag, {}).get(0, {}).get("r2_best")
    except Exception:
        _wo3 = None
    if _here is not None and _wo3 is not None:
        _ok = abs(_here - _wo3) <= 0.20
        log(f"WO#4 posmap sanity [{_tag}]: ')'-site best-layer B R^2={_here:.3f} vs "
            f"WO#3 0-shot best={_wo3:.3f} -> {'OK' if _ok else 'DIVERGENT (check plumbing!)'}")
    else:
        log(f"WO#4 posmap sanity [{_tag}]: ')'-site B R^2={_here} (no WO#3 ref in memory to compare).")

# --- per-tag JSON + a flat summary CSV ---------------------------------------
_pm_rows = []
for _tag in CFG["wo_posmap_tags"]:
    r = WO_POSMAP[_tag]
    wo_save_result(f"position_decodability_{_tag}.json", json.dumps(r, indent=2, default=str))
    for p in WO_PM_POSITIONS:
        for tgt in WO_PM_TARGETS:
            cell = r["positions"][p][tgt]
            _pm_rows.append({"tag": _tag, "surface": "C1", "position": p, "target": tgt,
                             "best_layer": cell["best_layer"], "r2_best": cell["cv_r2"]})
    _pm_rows.append({"tag": _tag, "surface": "C4", "position": "equals", "target": "BC",
                     "best_layer": r["c4_equals_BC"]["best_layer"],
                     "r2_best": r["c4_equals_BC"]["cv_r2"]})
wo_save_result("position_decodability_summary.csv",
               wo_battery_csv(_pm_rows, ["tag", "surface", "position", "target",
                                        "best_layer", "r2_best"]))

# --- heatmaps: position x layer, one panel per (tag, target) -----------------
try:
    _tags = CFG["wo_posmap_tags"]
    fig, axes = plt.subplots(len(_tags), len(WO_PM_TARGETS),
                             figsize=(6.4 * len(WO_PM_TARGETS), 3.0 * len(_tags)),
                             squeeze=False)
    for ti, _tag in enumerate(_tags):
        r = WO_POSMAP[_tag]
        nL = r["n_layers"]
        for gi, tgt in enumerate(WO_PM_TARGETS):
            ax = axes[ti][gi]
            rows = WO_PM_POSITIONS + (["C4:equals"] if tgt == "BC" else [])
            M = np.full((len(rows), nL), np.nan)
            for pi, p in enumerate(WO_PM_POSITIONS):
                rbl = r["positions"][p][tgt]["r2_by_layer"]
                for L in range(nL):
                    v = rbl.get(str(L))
                    if v is not None:
                        M[pi, L] = float(v)
            if tgt == "BC":
                rbl = r["c4_equals_BC"]["r2_by_layer"]
                for L in range(nL):
                    v = rbl.get(str(L))
                    if v is not None:
                        M[len(WO_PM_POSITIONS), L] = float(v)
            im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
            ax.set_xlabel("layer"); ax.set_title(f"{_tag}: decode {tgt}  (CV-R^2)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(WO_RESULTS / "position_decodability_heatmap.png"), dpi=130)
    plt.show()
except Exception as e:
    log(f"(position-map heatmap save skipped: {e})")

# --- printed within-zero-shot localization contrast --------------------------
print("\n================= WO#4 §3.2 — MULTI-POSITION DECODABILITY (zero-shot C1) =================")
print(f"{'tag':<10}{'B@)':>8}{'B@*':>8}{'B@C':>8}{'B@=':>8} | {'BC@)':>8}{'BC@=C1':>9}{'BC@=C4':>9}")
for _tag in CFG["wo_posmap_tags"]:
    r = WO_POSMAP[_tag]["positions"]
    c4 = WO_POSMAP[_tag]["c4_equals_BC"]["cv_r2"]

    def s(x):
        return "  n/a" if x is None else f"{x:.3f}"
    print(f"{_tag:<10}{s(r['rparen']['B']['cv_r2']):>8}{s(r['star']['B']['cv_r2']):>8}"
          f"{s(r['c_operand']['B']['cv_r2']):>8}{s(r['equals']['B']['cv_r2']):>8} | "
          f"{s(r['rparen']['BC']['cv_r2']):>8}{s(r['equals']['BC']['cv_r2']):>9}{s(c4):>9}")
print("-----------------------------------------------------------------------------------------")
print("READ (within zero-shot, length-matched -> confound-FREE):")
for _tag in CFG["wo_posmap_tags"]:
    r = WO_POSMAP[_tag]["positions"]
    b_rp = r["rparen"]["B"]["cv_r2"]
    bc_eq1 = r["equals"]["BC"]["cv_r2"]
    bc_eq4 = WO_POSMAP[_tag]["c4_equals_BC"]["cv_r2"]
    if None in (b_rp, bc_eq1, bc_eq4):
        print(f"  [{_tag}] incomplete map (a site had too few usable pairs).")
        continue
    localizes = (b_rp >= 0.70) and (bc_eq4 - bc_eq1 >= 0.20)
    _nc1 = WO_POSMAP[_tag]["n_used_c1"]
    _nc4 = WO_POSMAP[_tag]["n_used_c4"]
    print(f"  [{_tag}] B@)={b_rp:.2f} (source present) · B*C@=(C1)={bc_eq1:.2f} (NOT bound at "
          f"answer) · B*C@=(C4)={bc_eq4:.2f} (bound)  [n_C1={_nc1}, n_C4={_nc4}] -> "
          f"{'breakdown localizes to BINDING/ROUTING at the answer site (no causal claim)' if localizes else 'pattern not clean — report as-is'}.")
print("=========================================================================================")
