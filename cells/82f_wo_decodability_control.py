# ============================================================================
# Phase 6 / WO#3 control — length-matched NON-REPAIRING few-shot (GPU).
# ----------------------------------------------------------------------------
# Disambiguates the few-shot mid-late decodability drop seen in 82e:
#   R1 CONSUMPTION  — when the model composes, B is transformed into the product,
#                     so raw-B linear decodability falls (the interesting reading).
#   R2 DILUTION     — the longer few-shot prefix merely crowds the ')' residual,
#                     lowering decodability regardless of whether it composes.
# Control prefix: SAME '( 0 + b ) * c =' surface with answers of the SAME #digits
# (=> length-matched), but the demo answers are RANDOM and WRONG, so the prefix is
# present/attended yet does NOT teach composition. We confirm it does NOT repair
# (ctrl_acc stays low), then compare its by-layer curve to 0-shot and real few-shot:
#   drop reproduced WITHOUT repair  -> R2 dilution (demote the consumption story).
#   drop ABSENT (stays like 0-shot) -> R1 consumption (the drop needs successful
#                                       composition -> a genuine consumption signature).
# Thin orchestration over the WO#3 machinery (cell 82d) + cell-76 logic; GPU,
# checkpointed per (tag, shots). Requires 82d to have run (WO_FSPROBE* in memory).
# ============================================================================
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

assert "WO_FSPROBE_PAIRS" in globals() and "_fsp_rparen_last" in globals() and "WO_FSPROBE" in globals(), (
    "WO#3 control needs the WO#3 probe machinery — run cell 82d first.")
WO_FSCTRL_SHOTS = [2, 4]
WO_FSCTRL_S = 4   # the shot count the R1/R2 verdict is read at (strongest real drop)


def _wo_gt_wrong(b, c):
    """Deterministic RANDOM wrong answer with the SAME #digits as b*c (length-matched,
    uncorrelated with the true product) — makes the demos non-repairing."""
    p = int(b) * int(c); d = len(str(p)); lo, hi = 10 ** (d - 1), 10 ** d - 1
    r = np.random.default_rng(int(b) * 100003 + int(c))
    for _ in range(32):
        w = int(r.integers(lo, hi + 1))
        if w != p:
            return w
    return lo if lo != p else lo + 1


def _wo_ctrl_seed(shots):
    return int(CFG["wo_fsprobe_seed"]) + 5000 + 1000 * int(shots)


@torch.no_grad()
def _wo_ctrl_probe(tag, shots):
    """B-decodability by layer at the test ')' site under the NON-REPAIRING prefix.
    Cached per (tag, shots) so a disconnect resumes."""
    ck = f"wo_fsctrl_{tag}_{shots}"
    if has_artifact(ck, "json"):
        log(f"WO#3 control [{tag}/{shots}]: cached — reused.")
        return load_json(ck)
    wo_load_model(tag)
    n_layers = model.cfg.n_layers
    feats = {L: [] for L in range(n_layers)}
    Bvals, n_skip = [], 0
    sb = _wo_ctrl_seed(shots)
    for i, (B, C) in enumerate(WO_FSPROBE_PAIRS):
        prompt = wo_fewshot_render(_fsp_renderC1, _wo_gt_wrong, shots, (B, C), WO_PAIRS, seed=sb + i)
        assert prompt.endswith("=") and not prompt.endswith(" ")
        tok = model.to_tokens(prompt)
        pos = _fsp_rparen_last(tok)
        if not _fsp_site_ok(tok, pos, B):
            n_skip += 1; continue
        _, cache = model.run_with_cache(tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        for L in range(n_layers):
            feats[L].append(cache[f"blocks.{L}.hook_resid_post"][0, pos, :].float().cpu().numpy())
        Bvals.append(int(B)); del cache
    Bv = np.array(Bvals, dtype=float)
    decod = {L: wo_cv_r2(np.array(feats[L]), Bv, folds=WO_FSPROBE_FOLDS, ridge=WO_FSPROBE_RIDGE)
             for L in range(n_layers) if len(feats[L]) >= 7}
    out = {"tag": tag, "shots": int(shots), "n_used": len(Bvals), "n_skipped": n_skip,
           "r2_by_layer": {str(L): (None if v is None else float(v)) for L, v in decod.items()}}
    save_json(ck, out)
    log(f"WO#3 control [{tag}/{shots}]: n_used={len(Bvals)} skip={n_skip}")
    return out


def _wo_ctrl_acc(tag, shots):
    """C1 accuracy UNDER the wrong-answer prefix — must stay LOW (non-repairing) for
    the control to be valid. Cached/batched via wo_eval."""
    sb = _wo_ctrl_seed(shots)
    prompts = [wo_fewshot_render(_fsp_renderC1, _wo_gt_wrong, shots, (B, C), WO_PAIRS, seed=sb + i)
               for i, (B, C) in enumerate(WO_FSPROBE_PAIRS)]
    golds = [B * C for (B, C) in WO_FSPROBE_PAIRS]
    preds = [parse_int(c) for c in wo_eval(prompts, f"fsctrl_C1_{shots}", tag)]
    return float(np.mean([p is not None and p == g for p, g in zip(preds, golds)]))


WO_FSCTRL, WO_FSCTRL_ACC = {}, {}
for _tag in ("base", "instruct"):
    wo_load_model(_tag)
    for _s in WO_FSCTRL_SHOTS:
        WO_FSCTRL[(_tag, _s)] = _wo_ctrl_probe(_tag, _s)
        WO_FSCTRL_ACC[(_tag, _s)] = _wo_ctrl_acc(_tag, _s)

# ---- compare 0-shot vs real few-shot vs control, mid-late zone, + verdict ----
_NL = model.cfg.n_layers
_ML = int(np.floor(0.6 * _NL))
_S = WO_FSCTRL_S


def _ml_mean(curve):
    v = np.array([curve.get(str(L), curve.get(L, np.nan)) for L in range(_ML, _NL)], dtype=float)
    v = v[np.isfinite(v)]
    return float(np.nanmean(v)) if v.size else float("nan")


print(f"\n=== WO#3 CONTROL — does the few-shot decodability drop need REPAIR? "
      f"(mid-late L{_ML}..{_NL - 1}, {_S}-shot) ===")
print(f"{'tag':<9}{'ctrl_acc':>9}{'d0(0sh)':>9}{'dr(real)':>9}{'dc(ctrl)':>9}{'ctx_share':>11}  verdict")
_ctrl_rows = []
for _tag in ("base", "instruct"):
    d0 = _ml_mean(WO_FSPROBE[_tag][0]["r2_by_layer"])
    dr = _ml_mean(WO_FSPROBE[_tag][_S]["r2_by_layer"])
    dc = _ml_mean(WO_FSCTRL[(_tag, _S)]["r2_by_layer"])
    acc = WO_FSCTRL_ACC[(_tag, _S)]
    ctx = (d0 - dc) / (d0 - dr) if (d0 - dr) > 1e-6 else float("nan")   # ~1 dilution, ~0 consumption
    if acc > 0.6:
        v = f"INVALID control — wrong demos still repaired (acc={acc:.2f}); use random-text demos"
    elif np.isnan(ctx):
        v = "no real few-shot drop to explain"
    elif ctx >= 0.7:
        v = "R2 DILUTION — drop happens WITHOUT repair => context-length artifact; DEMOTE"
    elif ctx <= 0.3:
        v = "R1 CONSUMPTION — drop needs successful composition => signature; CENTERPIECE"
    else:
        v = "MIXED — both context and consumption contribute"
    print(f"{_tag:<9}{acc:>9.3f}{d0:>9.3f}{dr:>9.3f}{dc:>9.3f}{ctx:>11.2f}  {v}")
    _ctrl_rows.append({"tag": _tag, "shots": _S, "ctrl_acc": acc, "d0_zeroshot": d0,
                       "dr_real_fewshot": dr, "dc_control": dc, "context_share": ctx, "verdict": v})

wo_save_result("fewshot_decodability_control.csv",
               wo_battery_csv(_ctrl_rows, ["tag", "shots", "ctrl_acc", "d0_zeroshot",
                                           "dr_real_fewshot", "dc_control", "context_share", "verdict"]))
save_json("wo_fsctrl_summary", {"rows": _ctrl_rows, "shots_read": _S, "midlate_lo": _ML,
                                "note": "ctx_share=(d0-dc)/(d0-dr): ~1 => R2 dilution, ~0 => R1 consumption; "
                                        "valid only if ctrl_acc is LOW (prefix did not repair)."})

# ---- plot: 0-shot / real few-shot / control overlay, per tag ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, _tag in zip(axes, ("base", "instruct")):
    xs = range(_NL)
    g0 = WO_FSPROBE[_tag][0]["r2_by_layer"]
    gr = WO_FSPROBE[_tag][_S]["r2_by_layer"]
    gc = WO_FSCTRL[(_tag, _S)]["r2_by_layer"]
    for g, lab in ((g0, "0-shot (fails)"), (gr, f"{_S}-shot real (repairs)"),
                   (gc, f"{_S}-shot control (no repair)")):
        ax.plot(xs, [g.get(str(L), g.get(L, np.nan)) for L in xs], "o-", ms=3, label=lab)
    ax.axvspan(_ML, _NL - 1, color="orange", alpha=0.12)
    ax.axvline(30, color="k", ls="--", lw=1)
    ax.set_title(f"{_tag}: B decodability @ ')' — control overlay"); ax.set_xlabel("layer")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
axes[0].set_ylabel("CV-R^2 (B from ')' residual)"); fig.tight_layout()
try:
    fig.savefig(str(WO_RESULTS / "fewshot_decodability_control.png"), dpi=130)
except Exception as e:
    log(f"(control plot save skipped: {e})")
plt.show()
