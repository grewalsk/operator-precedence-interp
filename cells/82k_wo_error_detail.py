# ============================================================================
# Phase 6 / WORK ORDER #4 — §3.5 DECOMPOSE THE "OTHER" ERRORS (CPU only; cheap).
# ----------------------------------------------------------------------------
# 68% of C1 wrong answers were bucketed "other" (cell 82b). Sub-classify them with
# wo_classify_error_detail (cell 76, unit-tested):
#   correct / equals_B / equals_C / equals_B_plus_C / near_product (within 10% of
#   B*C) / right_magnitude (same #digits as B*C) / unrelated / parse_fail.
# This tells whether the model is ATTEMPTING the product and erring (near_product /
# right_magnitude — a noisy multiply) vs doing something UNRELATED — the
# bag-of-heuristics line (Nikankin 2024).
#
# Aggregates over the IN-MEMORY C1 preds (WO_*_RES['C1']['preds'], index-aligned to
# WO_PAIRS) for base + instruct + every cross-model. The saved battery JSONs strip
# preds, so base/instruct must be in memory (cells 78/79 cached -> instant re-run);
# cross-model preds come from the in-memory WO_CROSSMODEL_RES or, on a resumed run,
# the c1_preds stored in each wo_crossmodel_<tag> summary (cell 82g).
# ============================================================================
assert "wo_classify_error_detail" in globals() and "WO_PAIRS" in globals(), (
    "WO#4 error detail needs wo_classify_error_detail (cell 76) + WO_PAIRS (cell 78).")

_ED_CATS = WO_ERROR_DETAIL_CATS


def _ed_c1_preds(tag):
    """C1 preds (index-aligned to WO_PAIRS) for `tag`, from memory or the cached
    cross-model summary; None if unavailable this session."""
    src = {"base": globals().get("WO_BASE_RES"),
           "instruct": globals().get("WO_INSTRUCT_RES")}.get(tag) \
        or globals().get("WO_CROSSMODEL_RES", {}).get(tag)
    if src and "C1" in src and "preds" in src["C1"]:
        return src["C1"]["preds"]
    if has_artifact(f"wo_crossmodel_{tag}", "json"):
        blob = load_json(f"wo_crossmodel_{tag}")
        if isinstance(blob, dict) and blob.get("c1_preds") is not None:
            return blob["c1_preds"]
    return None


# tags: base, instruct, then every cross-model that produced a battery.
_ed_tags = ["base", "instruct"] + list(CFG.get("wo_crossmodel_tags", []))
_ed_rows = []
for _tag in _ed_tags:
    _preds = _ed_c1_preds(_tag)
    if _preds is None:
        log(f"WO#4 error detail [{_tag}]: C1 preds not available this session — skipped.")
        continue
    if len(_preds) != len(WO_PAIRS):
        log(f"WO#4 error detail [{_tag}]: preds len {len(_preds)} != {len(WO_PAIRS)} — skipped.")
        continue
    _counts = {c: 0 for c in _ED_CATS}
    for (B, C), p in zip(WO_PAIRS, _preds):
        _counts[wo_classify_error_detail(p, B, C)] += 1
    _total = sum(_counts.values())
    # fraction of the WRONG answers (excludes 'correct') that are product-attempts.
    _wrong = _total - _counts["correct"]
    _attempt = _counts["near_product"] + _counts["right_magnitude"]
    for c in _ED_CATS:
        _ed_rows.append({"tag": _tag, "category": c, "count": _counts[c],
                         "fraction": (_counts[c] / _total) if _total else 0.0})
    _ed_rows.append({"tag": _tag, "category": "_product_attempt_of_wrong",
                     "count": _attempt,
                     "fraction": (_attempt / _wrong) if _wrong else 0.0})
    log(f"WO#4 error detail [{_tag}] (n={_total}): "
        + ", ".join(f"{c}={_counts[c]}" for c in _ED_CATS)
        + f" | product-attempt of wrong={_attempt}/{_wrong}")

_ed_csv = wo_battery_csv(_ed_rows, ["tag", "category", "count", "fraction"])
wo_save_result("error_detail.csv", _ed_csv)
save_json("wo_error_detail", {"rows": _ed_rows, "categories": _ED_CATS})

print("\n================= WO#4 §3.5 — C1 REFINED ERROR DISTRIBUTION =================")
print(f"{'tag':<14}{'category':<26}{'count':>7}{'fraction':>10}")
for r in _ed_rows:
    print(f"{r['tag']:<14}{r['category']:<26}{r['count']:>7}{r['fraction']:>10.3f}")
print("============================================================================")
