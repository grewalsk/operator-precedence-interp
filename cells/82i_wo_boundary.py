# ============================================================================
# Phase 6 / WORK ORDER #4 — §3.3 MAP THE FAILURE BOUNDARY (GPU battery).
# ----------------------------------------------------------------------------
# Vary the TRIGGER to find exactly what collapses. New surfaces (cell-76
# WO_BOUNDARY_CONDITIONS, unit-tested; auxiliary operands A,D drawn deterministically
# per (B,C), in-band, recorded seed WO_BOUNDARY_SEED — so every model sees the SAME
# A,D for a given (B,C)):
#   BD1 ( A + B ) * C =     real bracketed SUM feeding outer '*'  (identity? or any sum?)
#   BD2 A * ( B + C ) =     bracketed sum on the RIGHT, outer '*'
#   BD3 ( A * B ) + C =     bracketed PRODUCT, outer '+'  (does the asymmetry flip?)
#   BD4 ( ( 0 + B ) * C ) * D =   depth-2 nest feeding outer '*'  (= B*C*D)
#   BD5 ( A + B ) * ( C + D ) =   two bracketed sums feeding outer '*'
# Run them as a battery on base + instruct (+ cross-model via CFG, reusing the
# 82g-chosen format). wo_boundary_summary (cell 76) classifies the trigger against
# the hypothesis: FAILS iff a bracketed sub-expression is an operand of a
# MULTIPLICATIVE outer op (outer '+' composes; outer '*' collapses).
#
# Deliverable: boundary_map.csv (per-surface acc + observed + the one-line trigger
# characterization per model). Checkpointed per (model, condition) via wo_eval.
# ============================================================================
import numpy as np

assert "WO_BOUNDARY_CONDITIONS" in globals() and "wo_boundary_summary" in globals(), (
    "WO#4 boundary map needs WO_BOUNDARY_CONDITIONS + wo_boundary_summary (cell 76).")

CFG.setdefault("wo_boundary_tags", ["base", "instruct"])


def _wo_wrap_for_live_tag(tag):
    """(cache_suffix, wrap) reusing the format 82g chose for a cross-model (chat vs
    bare); base/instruct are bare-continuation. The model must already be live."""
    info = globals().get("WO_XM_RESULTS", {}).get(tag, {})
    fmt = info.get("format", "") if isinstance(info, dict) else ""
    if isinstance(fmt, str) and fmt.startswith("chat") and hasattr(tokenizer, "apply_chat_template"):
        _bos = getattr(tokenizer, "bos_token", None)

        def wrap(content):
            msg = [{"role": "user",
                    "content": "Compute and reply with ONLY the integer:\n" + content}]
            try:
                s = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            except Exception:
                return content
            if _bos and s.startswith(_bos):
                s = s[len(_bos):]
            return s
        return "_chat", wrap
    return "", (lambda s: s)


def _bd_wrap_conditions(conds, wrap):
    return [(k, n, (lambda B, C, _r=r: wrap(_r(B, C))), g) for (k, n, r, g) in conds]


def _bd_ref_acc(tag, key):
    """Reference C1/C4 accuracy for context (in-memory battery results)."""
    src = {"base": globals().get("WO_BASE_RES"),
           "instruct": globals().get("WO_INSTRUCT_RES")}.get(tag) \
        or globals().get("WO_CROSSMODEL_RES", {}).get(tag)
    if src and key in src and "exact_acc" in src[key]:
        return float(src[key]["exact_acc"])
    return None


WO_BOUNDARY_RES = {}
_bd_rows = []
_bd_chars = {}
for _tag in CFG["wo_boundary_tags"]:
    try:
        wo_load_model(_tag)
    except Exception as e:
        log(f"WO#4 boundary [{_tag}]: load/access failed ({type(e).__name__}) — skipped.")
        continue
    _suffix, _wrap = _wo_wrap_for_live_tag(_tag)
    _res = wo_run_battery(_tag, _bd_wrap_conditions(WO_BOUNDARY_CONDITIONS, _wrap),
                          WO_PAIRS, cache_tag=_tag + _suffix)
    WO_BOUNDARY_RES[_tag] = _res
    _acc = {k: _res[k]["exact_acc"] for k in _res}
    _summ = wo_boundary_summary(_acc)
    _bd_chars[_tag] = _summ["characterization"]
    _refC1, _refC4 = _bd_ref_acc(_tag, "C1"), _bd_ref_acc(_tag, "C4")
    for row in _summ["rows"]:
        _bd_rows.append({
            "tag": _tag, "key": row["key"], "surface": row["surface"],
            "outer_op": row["outer_op"], "predict": row["predict"],
            "acc": row["acc"], "observed": row["observed"],
            "corr_BC": _res[row["key"]]["corr"], "parse_fail": _res[row["key"]]["parse_fail_rate"],
            "ref_C1": _refC1, "ref_C4": _refC4,
        })
    log(f"WO#4 boundary [{_tag}]: " + " ".join(
        f"{r['key']}={(r['acc'] if r['acc'] is None else round(r['acc'],3))}({r['observed']})"
        for r in _summ["rows"]) + f"  -> {_summ['characterization']}")

_bd_csv = wo_battery_csv(
    _bd_rows, ["tag", "key", "surface", "outer_op", "predict", "acc", "observed",
               "corr_BC", "parse_fail", "ref_C1", "ref_C4"])
for _tag, _ch in _bd_chars.items():
    _bd_csv += f"# trigger[{_tag}],\"{_ch}\"\n"
wo_save_result("boundary_map.csv", _bd_csv)
save_json("wo_boundary_map", {"rows": _bd_rows, "characterizations": _bd_chars,
                              "aux_seed": WO_BOUNDARY_SEED})

print("\n================= WO#4 §3.3 — FAILURE-BOUNDARY MAP =================")
print(f"{'tag':<10}{'key':<5}{'surface':<26}{'outer':>6}{'acc':>8}{'observed':>10}")
for r in _bd_rows:
    _a = "  n/a" if r["acc"] is None else f"{r['acc']:.3f}"
    print(f"{r['tag']:<10}{r['key']:<5}{r['surface']:<26}{r['outer_op']:>6}{_a:>8}{r['observed']:>10}")
print("-------------------------------------------------------------------")
for _tag, _ch in _bd_chars.items():
    print(f"  TRIGGER [{_tag}]: {_ch}")
print("===================================================================")
