# ============================================================================
# Phase 6 / WO#2 — §3.5 FEW-SHOT / ROBUSTNESS CONTROL (GPU; defuses "you just
# prompted it badly"). Run C1 '( 0 + B ) * C =' under shots ∈ {0, 2, 4} few-shot
# on BOTH tags, from the SHARED WO_PAIRS, and compare to the inside-bracket C4
# ceiling. The claim holds iff few-shot C1 stays WELL BELOW C4 (does NOT jump to
# ~0.9): a few worked examples of the same surface don't repair the composition.
#
# The few-shot prompts come from wo_fewshot_render (cell 76, unit-tested): `shots`
# correctly-worked examples of the SAME surface, operands drawn deterministically
# (seed = wo_fewshot_seed + shots), EXCLUDING the test pair, then the bare test
# prompt. Decoding reuses Phase 3's fingerprinted, resumable _eval_prompts via
# wo_eval (cached per (tag, key)), so a disconnect resumes. 0-shot reuses the
# already-computed battery C1 (no recompute).
# ============================================================================
import numpy as np

assert "WO_PAIRS" in globals() and "wo_fewshot_render" in globals(), (
    "Few-shot control needs WO_PAIRS (cell 78) and wo_fewshot_render (cell 76).")
CFG.setdefault("wo_fewshot_seed", 202)
WO_FEWSHOT_SHOTS = [0, 2, 4]

_renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
_gtC1 = dict((c[0], c[3]) for c in WO_CONDITIONS)["C1"]
_golds = [_gtC1(B, C) for (B, C) in WO_PAIRS]


def _wo_battery_for_fewshot(tag):
    """In-memory battery (for the C4 ceiling + 0-shot C1). Falls back to a cached
    re-run of C1+C4 if a battery isn't in globals (model must be live)."""
    g = globals().get("WO_INSTRUCT_RES" if tag == "instruct" else "WO_BASE_RES")
    if g is not None and "C1" in g and "C4" in g:
        return g
    return wo_run_battery(tag, [c for c in WO_CONDITIONS if c[0] in ("C1", "C4")], WO_PAIRS)


_fs_rows = []
for _tag in ("base", "instruct"):
    wo_load_model(_tag)
    _bat = _wo_battery_for_fewshot(_tag)
    _c4_ref = float(_bat["C4"]["exact_acc"])
    for _shots in WO_FEWSHOT_SHOTS:
        if _shots == 0:
            # 0-shot C1 IS the bare-continuation battery C1 (already computed/cached).
            _acc = float(_bat["C1"]["exact_acc"])
            _pf = float(_bat["C1"]["parse_fail_rate"])
        else:
            _seed = int(CFG["wo_fewshot_seed"]) + _shots
            _prompts = [wo_fewshot_render(_renderC1, _gtC1, _shots, (B, C), WO_PAIRS, seed=_seed)
                        for (B, C) in WO_PAIRS]
            _conts = wo_eval(_prompts, f"fewshot_C1_{_shots}shot", _tag)
            _preds = [parse_int(c) for c in _conts]
            _summ = wo_summarize(_preds, _golds)
            _acc = float(_summ["exact_acc"])
            _pf = float(_summ["parse_fail_rate"])
        _fs_rows.append({"tag": _tag, "shots": _shots, "c1_acc": _acc,
                         "c4_reference": _c4_ref, "c1_minus_c4": _acc - _c4_ref,
                         "parse_fail": _pf, "n": len(WO_PAIRS)})
        log(f"WO few-shot [{_tag}] {_shots}-shot C1 acc={_acc:.3f} (C4 ref={_c4_ref:.3f}, "
            f"parse_fail={_pf:.3f})")

_fs_csv = wo_battery_csv(
    _fs_rows, ["tag", "shots", "c1_acc", "c4_reference", "c1_minus_c4", "parse_fail", "n"])
# verdict line: does ANY shot count lift C1 to within 0.10 of C4 (i.e. "fixed")?
_fixed = any((r["c1_acc"] >= r["c4_reference"] - 0.10) for r in _fs_rows)
_fs_csv += (f"\n# verdict,{'FEWSHOT_FIXES_C1' if _fixed else 'FEWSHOT_DOES_NOT_FIX_C1'}\n")
_fs_csv += ("# reading,\"Composition failure is robust to few-shot prompting: a few worked "
            "examples of the same surface do NOT lift C1 to the C4 ceiling.\"\n" if not _fixed else
            "# reading,\"Few-shot prompting recovers C1 toward C4 — the failure is (partly) a "
            "prompting artifact; revisit the claim.\"\n")
wo_save_result("fewshot_control.csv", _fs_csv)
save_json("wo_fewshot_control", {"rows": _fs_rows, "fewshot_fixes_c1": bool(_fixed),
                                 "shots": WO_FEWSHOT_SHOTS, "seed_base": int(CFG["wo_fewshot_seed"])})

print("\n================= WO#2 §3.5 — FEW-SHOT CONTROL (C1 vs C4 ceiling) =================")
print(f"{'tag':<9}{'shots':>6}{'C1 acc':>9}{'C4 ref':>9}{'C1-C4':>9}{'parse_fail':>12}")
for r in _fs_rows:
    print(f"{r['tag']:<9}{r['shots']:>6}{r['c1_acc']:>9.3f}{r['c4_reference']:>9.3f}"
          f"{r['c1_minus_c4']:>9.3f}{r['parse_fail']:>12.3f}")
print("---------------------------------------------------------------------------------")
print(f"  VERDICT: {'few-shot FIXES C1 (revisit claim)' if _fixed else 'few-shot does NOT fix C1 — composition failure is robust'}")
print("=================================================================================")
