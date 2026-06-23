# ============================================================================
# Phase 6 / WO#2 — §3.4 CONFIDENCE INTERVALS on the headline contrasts (CPU).
# Bootstrap + Wilson CIs (cell 76, unit-tested) for:
#   * C4 vs C1            (the lead surface contrast)  — base + instruct
#   * A1 vs C1            (the operation-specific contrast: add composes, mult doesn't)
#   * each operand-magnitude bin, C1 vs C4 (matched |B*C|)
# Deltas use the PAIRED bootstrap (same items under two conditions) so the CI
# reflects per-item pairing. Reads IN-MEMORY correct_masks (saved JSON strips
# them), so this runs in the same session as the batteries (all cached).
# ============================================================================
import json
import numpy as np

assert "wo_bootstrap_ci" in globals() and "WO_PAIRS" in globals(), (
    "CI cell needs wo_bootstrap_ci (cell 76) and WO_PAIRS (cell 78).")
CFG.setdefault("wo_ci_boot", 10000)
CFG.setdefault("wo_ci_seed", 303)
_NB = int(CFG["wo_ci_boot"]); _SEED = int(CFG["wo_ci_seed"])


def _mean(m):
    return float(np.mean(m)) if len(m) else None


def _contrast(name, mask_better, mask_worse, label_better, label_worse):
    """Paired contrast block: per-condition acc + bootstrap CI + Wilson CI, plus
    the PAIRED delta (better - worse) with a paired-bootstrap CI."""
    return {
        "contrast": name, "n": len(mask_better),
        label_better: {"acc": _mean(mask_better),
                       "bootstrap_ci": wo_bootstrap_ci(mask_better, _NB, seed=_SEED),
                       "wilson_ci": wo_wilson_ci(int(sum(mask_better)), len(mask_better))},
        label_worse: {"acc": _mean(mask_worse),
                      "bootstrap_ci": wo_bootstrap_ci(mask_worse, _NB, seed=_SEED),
                      "wilson_ci": wo_wilson_ci(int(sum(mask_worse)), len(mask_worse))},
        "delta_%s_minus_%s" % (label_better, label_worse): (_mean(mask_better) - _mean(mask_worse))
            if (mask_better and mask_worse) else None,
        "delta_paired_bootstrap_ci": wo_paired_delta_ci(mask_better, mask_worse, _NB, seed=_SEED),
    }


def _ensure_a1_mask():
    """A1 ('( 0 + B ) + C =') correct_mask for instruct. Prefer the in-memory
    Branch-B result (cell 82 exposes WO_BRANCHB_RES_instruct); else recompute from
    the cached A1 battery (model loaded on demand)."""
    g = globals().get("WO_BRANCHB_RES_instruct")
    if g is not None and "A1" in g and "correct_mask" in g["A1"]:
        return g["A1"]["correct_mask"]
    try:
        wo_load_model("instruct")
        r = wo_run_battery("instruct", [c for c in WO_BRANCHB_CONDITIONS if c[0] == "A1"], WO_PAIRS)
        return r["A1"]["correct_mask"]
    except Exception as e:
        log(f"WO CI: could not obtain A1 mask ({e}); skipping A1-vs-C1 CI.")
        return None


_ci = {"meta": {"n_boot": _NB, "seed": _SEED, "alpha": 0.05,
                "method": "percentile bootstrap (paired for deltas) + Wilson cross-check"},
       "C4_vs_C1": {}, "A1_vs_C1": {}, "magnitude_bins_C1_vs_C4": []}

# --- C4 vs C1 on both tags -------------------------------------------------
for _tag in ("base", "instruct"):
    _res = globals().get("WO_INSTRUCT_RES" if _tag == "instruct" else "WO_BASE_RES")
    if _res is None or "C1" not in _res or "C4" not in _res:
        log(f"WO CI: {_tag} battery not in memory — skipping C4-vs-C1 for {_tag}.")
        continue
    _ci["C4_vs_C1"][_tag] = _contrast(
        f"{_tag}: C4 vs C1", _res["C4"]["correct_mask"], _res["C1"]["correct_mask"], "C4", "C1")

# --- A1 vs C1 (instruct): operation-specificity ----------------------------
_a1 = _ensure_a1_mask()
if _a1 is not None and globals().get("WO_INSTRUCT_RES") is not None:
    _ci["A1_vs_C1"]["instruct"] = _contrast(
        "instruct: A1 (add-compose) vs C1 (mult-compose)",
        _a1, WO_INSTRUCT_RES["C1"]["correct_mask"], "A1", "C1")

# --- operand-magnitude bins, C1 vs C4 (instruct) ---------------------------
if globals().get("WO_INSTRUCT_RES") is not None:
    _mC1 = WO_INSTRUCT_RES["C1"]["correct_mask"]
    _mC4 = WO_INSTRUCT_RES["C4"]["correct_mask"]
    for _b in wo_operand_magnitude_bins(WO_PAIRS, n_bins=5):
        _idx = _b["idx"]
        if not _idx:
            continue
        _bc1 = [_mC1[j] for j in _idx]
        _bc4 = [_mC4[j] for j in _idx]
        _blk = _contrast(f"|B*C| in [{_b['lo']:.0f},{_b['hi']:.0f}) n={_b['n']}",
                         _bc4, _bc1, "C4", "C1")
        _blk["bin_lo"] = _b["lo"]; _blk["bin_hi"] = _b["hi"]
        _ci["magnitude_bins_C1_vs_C4"].append(_blk)

wo_save_result("confidence_intervals.json", json.dumps(_ci, indent=2, default=str))
save_json("wo_confidence_intervals", _ci)

print("\n================= WO#2 §3.4 — CONFIDENCE INTERVALS (95%) =================")
for _tag, _blk in _ci["C4_vs_C1"].items():
    _d = _blk.get("delta_C4_minus_C1")
    _dci = _blk.get("delta_paired_bootstrap_ci")
    print(f"  [{_tag}] C4={_blk['C4']['acc']:.3f} {_blk['C4']['bootstrap_ci']}  "
          f"C1={_blk['C1']['acc']:.3f} {_blk['C1']['bootstrap_ci']}  "
          f"Δ(C4-C1)={_d:.3f} {_dci}")
if _ci["A1_vs_C1"]:
    _blk = _ci["A1_vs_C1"]["instruct"]
    print(f"  [instruct] A1={_blk['A1']['acc']:.3f}  C1={_blk['C1']['acc']:.3f}  "
          f"Δ(A1-C1)={_blk.get('delta_A1_minus_C1'):.3f} {_blk.get('delta_paired_bootstrap_ci')}")
print(f"  magnitude bins (instruct, C1 vs C4): {len(_ci['magnitude_bins_C1_vs_C4'])} bins with CIs")
print("=========================================================================")
