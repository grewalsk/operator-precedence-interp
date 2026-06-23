# ============================================================================
# Phase 6 / WO#2 — §3.6 "WHAT DOES C1 OUTPUT INSTEAD?" distribution (CPU only).
# Aggregate C1's PARSED predictions into diagnostic buckets (wo_classify_wrong_
# output, cell 76, unit-tested): correct / equals_B / equals_C / equals_B_plus_C
# / parse_fail / other. Makes the failure mode legible — if C1 disproportionately
# returns B (the bracketed operand) or B+C, that is direct evidence about what the
# model does with the un-composed pieces.
#
# Uses the IN-MEMORY battery preds (WO_*_RES['C1']['preds'], index-aligned to
# WO_PAIRS) — the saved battery JSON strips preds/masks, so this must run in the
# same session as the batteries (they are cached, so re-running cells 78/79 is
# instant if needed).
# ============================================================================
assert "WO_PAIRS" in globals() and "wo_classify_wrong_output" in globals(), (
    "wrong-output distribution needs WO_PAIRS (cell 78) + wo_classify_wrong_output (cell 76).")

_WO_CATS = ["correct", "equals_B", "equals_C", "equals_B_plus_C", "parse_fail", "other"]
_wo_src = {"base": globals().get("WO_BASE_RES"), "instruct": globals().get("WO_INSTRUCT_RES")}

_wd_rows = []
for _tag in ("base", "instruct"):
    _res = _wo_src.get(_tag)
    if _res is None or "C1" not in _res or "preds" not in _res["C1"]:
        log(f"WO wrong-output [{_tag}]: C1 preds not in memory — skipped (re-run battery cell).")
        continue
    _preds = _res["C1"]["preds"]
    if len(_preds) != len(WO_PAIRS):
        log(f"WO wrong-output [{_tag}]: preds len {len(_preds)} != {len(WO_PAIRS)} — skipped.")
        continue
    _counts = {c: 0 for c in _WO_CATS}
    for (B, C), p in zip(WO_PAIRS, _preds):
        _counts[wo_classify_wrong_output(p, B, C)] += 1
    _total = sum(_counts.values())
    for c in _WO_CATS:
        _wd_rows.append({"tag": _tag, "category": c, "count": _counts[c],
                         "fraction": (_counts[c] / _total) if _total else 0.0})
    log(f"WO wrong-output [{_tag}] (n={_total}): "
        + ", ".join(f"{c}={_counts[c]}" for c in _WO_CATS))

_wd_csv = wo_battery_csv(_wd_rows, ["tag", "category", "count", "fraction"])
wo_save_result("wrong_output_distribution.csv", _wd_csv)
save_json("wo_wrong_output_distribution", {"rows": _wd_rows, "categories": _WO_CATS})

print("\n================= WO#2 §3.6 — C1 WRONG-OUTPUT DISTRIBUTION =================")
print(f"{'tag':<9}{'category':<18}{'count':>7}{'fraction':>10}")
for r in _wd_rows:
    print(f"{r['tag']:<9}{r['category']:<18}{r['count']:>7}{r['fraction']:>10.3f}")
print("===========================================================================")
