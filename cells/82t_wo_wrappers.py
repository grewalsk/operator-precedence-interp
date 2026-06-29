# ============================================================================
# Phase 6 / WORK ORDER #7 — VACUOUS-WRAPPER BLIND-SPOT MAP (GPU; behavioral).
# ----------------------------------------------------------------------------
# The paper's hook: '( 0 + B )' is the ADDITIVE IDENTITY — it provably does not
# change B — yet it selectively breaks multiplication and not addition. This cell
# measures accuracy over identity-preserving wrappers W(B)==B crossed with the
# outer operation (* C / + C), on the SHARED WO_PAIRS, and reads off WHICH property
# drives the blind spot (parens / additive-identity / inner-additive-under-outer-
# multiplicative mismatch / nesting depth). Every ground truth is the no-op value
# (B*C or B+C), so any accuracy drop is a pure syntactic artifact.
#
# Thin orchestration over the validated wo_run_battery (greedy decode + parse_int),
# cached per (tag, condition) so a re-run is instant. Runs on base + instruct by
# default; add cross-models via CFG['wo_wrap_tags'] for the generality claim.
# ============================================================================
import json
import numpy as np

assert "wo_build_wrapper_conditions" in globals() and "wo_wrapper_verdict" in globals() \
    and "wo_run_battery" in globals() and "WO_PAIRS" in globals(), (
    "WO#7 wrapper map needs wo_build_wrapper_conditions/wo_wrapper_verdict (cell 76) "
    "+ wo_run_battery (cell 77) + WO_PAIRS (cell 78).")

CFG.setdefault("wo_wrap_tags", ["base", "instruct"])
WO_WRAP_CONDS = wo_build_wrapper_conditions()
_WRAP_KEYS = [w[0] for w in WO_WRAPPERS]          # wrapper order for the heatmap rows
log(f"WO#7 vacuous-wrapper map: {len(WO_WRAP_CONDS)} conditions "
    f"({len(WO_WRAPPERS)} wrappers x {len(WO_WRAP_OPS)} ops) on {len(WO_PAIRS)} shared pairs.")


def _wrap_run(tag):
    res = wo_run_battery(tag, WO_WRAP_CONDS, WO_PAIRS, cache_tag=tag)   # cached per (tag,key)
    acc = {k: res[k]["exact_acc"] for k in res}
    verdict = wo_wrapper_verdict(acc)
    out = {"tag": tag, "experiment": "WO7_vacuous_wrapper_map",
           "acc": acc, "verdict": verdict, "pairs_sha": WO_PAIRS_HASH,
           "note": "Identity-preserving wrappers W(B)==B x outer op (*C/+C); every gt is the no-op "
                   "value, so any accuracy drop is a pure syntactic artifact of a semantically-null rewrite."}
    wo_save_result(f"wrapper_map_{tag}.json", json.dumps(out, indent=2, default=str))
    return out


WO_WRAP = {}
for _tag in CFG["wo_wrap_tags"]:
    wo_load_model(_tag)
    WO_WRAP[_tag] = _wrap_run(_tag)

# ---- flat summary CSV (tag, wrapper, op, acc, drop-vs-bare-same-op) ----------
_wrap_rows = []
for _tag in CFG["wo_wrap_tags"]:
    acc = WO_WRAP[_tag]["acc"]
    for wk, wname, _ in WO_WRAPPERS:
        for ok, osym, _ in WO_WRAP_OPS:
            a = acc.get(f"W_{wk}_{ok}")
            base = acc.get(f"W_bare_{ok}")
            _wrap_rows.append({"tag": _tag, "wrapper": wk, "surface": wname, "op": ok,
                               "acc": a, "drop_vs_bare": (None if (a is None or base is None) else base - a)})
wo_save_result("wrapper_map_summary.csv",
               wo_battery_csv(_wrap_rows, ["tag", "wrapper", "surface", "op", "acc", "drop_vs_bare"]))

# ---- heatmap: wrappers (rows) x op (cols), accuracy, one panel per tag -------
try:
    import matplotlib.pyplot as plt
    _tags = CFG["wo_wrap_tags"]
    fig, axes = plt.subplots(1, len(_tags), figsize=(3.2 * len(_tags) + 1.5, 0.5 * len(_WRAP_KEYS) + 1.5),
                             squeeze=False)
    for ti, _tag in enumerate(_tags):
        acc = WO_WRAP[_tag]["acc"]
        M = np.array([[acc.get(f"W_{wk}_{ok}", np.nan) for ok, _, _ in WO_WRAP_OPS] for wk in _WRAP_KEYS])
        ax = axes[0][ti]
        im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(WO_WRAP_OPS))); ax.set_xticklabels([o[0] for o in WO_WRAP_OPS])
        ax.set_yticks(range(len(_WRAP_KEYS))); ax.set_yticklabels([w[1] for w in WO_WRAPPERS], fontsize=8)
        for yi in range(M.shape[0]):
            for xi in range(M.shape[1]):
                if np.isfinite(M[yi, xi]):
                    ax.text(xi, yi, f"{M[yi, xi]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{_tag}: acc by wrapper x op\n[{WO_WRAP[_tag]['verdict']['driver']}]", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(WO_RESULTS / "wrapper_map_heatmap.png"), dpi=130)
    plt.show()
except Exception as e:
    log(f"(WO#7 wrapper heatmap skipped: {e})")

# ---- printed verdict ---------------------------------------------------------
print("\n================= WO#7 — VACUOUS-WRAPPER BLIND-SPOT MAP =================")
print(f"{'tag':<10}{'wrapper':<14}{'* C (mul)':>10}{'+ C (add)':>10}")
for _tag in CFG["wo_wrap_tags"]:
    acc = WO_WRAP[_tag]["acc"]
    for wk, wname, _ in WO_WRAPPERS:
        am = acc.get(f"W_{wk}_mul"); aa = acc.get(f"W_{wk}_add")
        print(f"{_tag:<10}{wname:<14}{('n/a' if am is None else f'{am:.3f}'):>10}{('n/a' if aa is None else f'{aa:.3f}'):>10}")
    print("-" * 44)
for _tag in CFG["wo_wrap_tags"]:
    v = WO_WRAP[_tag]["verdict"]
    print(f"  [{_tag}] blindspot={v['mult_blindspot']} op_specific={v['operation_specific']} "
          f"driver={v['driver']} depth_sensitive={v['depth_sensitive']}")
    print(f"          {v['reason']}")
print("========================================================================")
