# ============================================================================
# Phase 6 / WORK ORDER #4 — §3.4 FORMAT-CUED RECOVERY (GPU; nail the surprising hint).
# ----------------------------------------------------------------------------
# WO#3 found Instruct composed at ~0.63 with WRONG-answer demos => in-context
# recovery looks FORMAT-/task-cued, not content-driven. Pin it with a length-matched
# demo-type sweep at FIXED shots (default 4), measuring C1 accuracy per demo type:
#   correct          ( 0 + b ) * c = (b*c)      (real worked example)
#   wrong_answer     ( 0 + b ) * c = <wrong>    right format, random same-#digits answer
#   scrambled_format <perm of tokens> = (b*c)   same tokens, broken precedence, correct value
#   random_text      <length-matched filler>    NO arithmetic / no '=' (pure length control)
# ALL four are length-matched and (crucially) draw the SAME shot pairs (shared
# _wo_select_shot_pairs) — they differ ONLY in demo content/structure, so the sweep
# is a clean confound control. The TEST line is always the canonical bare C1.
#
# wo_format_cue_verdict (cell 76, unit-tested): wrong_answer ~= correct AND
# random_text flat -> "format-primed, not content-learned"; only correct recovers
# -> "content-driven". Demo builders are the unit-tested cell-76 wo_demo_render.
#
# Deliverable: format_recovery.csv + the verdict. Checkpointed per (model, demo_type)
# via wo_eval; 'correct' reuses the cell-82a few-shot cache (identical prompts).
# ============================================================================
import numpy as np

assert "wo_demo_render" in globals() and "wo_format_cue_verdict" in globals(), (
    "WO#4 format recovery needs wo_demo_render + wo_format_cue_verdict (cell 76).")

CFG.setdefault("wo_format_tags", ["base", "instruct"])
CFG.setdefault("wo_format_shots", 4)
CFG.setdefault("wo_fewshot_seed", 202)
WO_FMT_SHOTS = int(CFG["wo_format_shots"])

_fmt_renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
_fmt_gtC1 = dict((c[0], c[3]) for c in WO_CONDITIONS)["C1"]
_fmt_golds = [B * C for (B, C) in WO_PAIRS]


def _fmt_wrap_for_live_tag(tag):
    """(cache_suffix, wrap) reusing the 82g-chosen format (chat vs bare)."""
    if "_wo_wrap_for_live_tag" in globals():     # defined by cell 82i; reuse it.
        return _wo_wrap_for_live_tag(tag)
    return "", (lambda s: s)


def _fmt_zeroshot_c1(tag):
    """0-shot C1 accuracy = the (format-matched) battery C1 already computed."""
    src = {"base": globals().get("WO_BASE_RES"),
           "instruct": globals().get("WO_INSTRUCT_RES")}.get(tag) \
        or globals().get("WO_CROSSMODEL_RES", {}).get(tag)
    if src and "C1" in src and "exact_acc" in src["C1"]:
        return float(src["C1"]["exact_acc"])
    return None


def _fmt_acc(cache_tag, wrap, demo_type, shots):
    """C1 accuracy under `shots` demos of `demo_type`. 'correct' reuses 82a's cache
    key (its prompts are identical to wo_fewshot_render's)."""
    base = int(CFG["wo_fewshot_seed"]) + 1000 * int(shots)
    prompts = [wrap(wo_demo_render(demo_type, _fmt_renderC1, _fmt_gtC1, shots, (B, C),
                                   WO_PAIRS, seed=base + i))
               for i, (B, C) in enumerate(WO_PAIRS)]
    key = (f"fewshot_C1_{shots}shot" if demo_type == "correct"
           else f"fmtcue_{demo_type}_{shots}")
    preds = [parse_int(c) for c in wo_eval(prompts, key, cache_tag)]
    return float(np.mean([p is not None and p == g for p, g in zip(preds, _fmt_golds)]))


WO_FORMAT_REC = {}
_fmt_rows = []
for _tag in CFG["wo_format_tags"]:
    try:
        wo_load_model(_tag)
    except Exception as e:
        log(f"WO#4 format recovery [{_tag}]: load/access failed ({type(e).__name__}) — skipped.")
        continue
    _suffix, _wrap = _fmt_wrap_for_live_tag(_tag)
    _cache_tag = _tag + _suffix
    _zs = _fmt_zeroshot_c1(_tag)
    _acc_by_type = {}
    for _dt in WO_DEMO_TYPES:
        _acc_by_type[_dt] = _fmt_acc(_cache_tag, _wrap, _dt, WO_FMT_SHOTS)
        log(f"WO#4 format [{_tag}] {WO_FMT_SHOTS}-shot {_dt}: C1 acc={_acc_by_type[_dt]:.3f} "
            f"(0-shot={_zs})")
    _verdict = wo_format_cue_verdict(_acc_by_type, _zs)
    WO_FORMAT_REC[_tag] = {"acc_by_type": _acc_by_type, "zeroshot": _zs, "verdict": _verdict}
    for _dt in WO_DEMO_TYPES:
        _fmt_rows.append({
            "tag": _tag, "shots": WO_FMT_SHOTS, "demo_type": _dt,
            "c1_acc": _acc_by_type[_dt], "zeroshot_c1": _zs,
            "delta_vs_zeroshot": (None if _zs is None else _acc_by_type[_dt] - _zs),
            "recovers": _verdict["recovers"][_dt], "verdict": _verdict["label"],
        })

_fmt_csv = wo_battery_csv(
    _fmt_rows, ["tag", "shots", "demo_type", "c1_acc", "zeroshot_c1",
                "delta_vs_zeroshot", "recovers", "verdict"])
for _tag, _r in WO_FORMAT_REC.items():
    _fmt_csv += f"# verdict[{_tag}],{_r['verdict']['label']},\"{_r['verdict']['reading']}\"\n"
wo_save_result("format_recovery.csv", _fmt_csv)
save_json("wo_format_recovery", {"by_tag": {t: {"acc_by_type": r["acc_by_type"],
                                                "zeroshot": r["zeroshot"],
                                                "label": r["verdict"]["label"]}
                                            for t, r in WO_FORMAT_REC.items()},
                                 "shots": WO_FMT_SHOTS})

print(f"\n================= WO#4 §3.4 — FORMAT-CUED RECOVERY ({WO_FMT_SHOTS}-shot) =================")
print(f"{'tag':<10}{'0-shot':>8}{'correct':>9}{'wrong':>9}{'scram':>9}{'random':>9}   verdict")
for _tag, _r in WO_FORMAT_REC.items():
    a = _r["acc_by_type"]

    def s(x):
        return "  n/a" if x is None else f"{x:.3f}"
    print(f"{_tag:<10}{s(_r['zeroshot']):>8}{s(a.get('correct')):>9}{s(a.get('wrong_answer')):>9}"
          f"{s(a.get('scrambled_format')):>9}{s(a.get('random_text')):>9}   {_r['verdict']['label']}")
print("-----------------------------------------------------------------------------------")
for _tag, _r in WO_FORMAT_REC.items():
    print(f"  [{_tag}] {_r['verdict']['reading']}")
print("===================================================================================")
