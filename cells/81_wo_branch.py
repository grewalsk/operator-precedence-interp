# ============================================================================
# Phase 6 / WO — STEP 5a : select the branch (§8) + write the decision record.
# ----------------------------------------------------------------------------
# wo_select_branch maps the gate verdict to CLEAN_REPAIR / PARTIAL_REPAIR /
# NO_REPAIR (faithful to the §8 tree; the NO_REPAIR leaf is the superset of any
# hard-gate failure). Emits results/decision_record.md with the gate table and a
# written justification, and persists the branch for the conditional Step 5b cell.
# ============================================================================
assert "WO_GATE_EVAL" in globals(), "Run the gates cell (80) first."

WO_BRANCH = wo_select_branch(WO_GATE_EVAL)

# repro fields surfaced in the record (full repro.txt is written in cell 83).
def _wo_pkg_ver(mod):
    """Robust version lookup: some transformer_lens builds don't expose
    __version__ as a module attribute, so fall back to importlib.metadata."""
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", None)
        if v:
            return v
    except Exception:
        pass
    try:
        import importlib.metadata as _im
        return _im.version(mod)
    except Exception:
        return "n/a"

_repro = {
    "transformer_lens": _wo_pkg_ver("transformer_lens"),
    "model_revision": WO_MODEL_REVISIONS.get("instruct"),
    "prepend_bos": WO_PREPEND_BOS,
    "format": WO_INSTRUCT_FORMAT,
}

# battery summary (acc/corr/parse-fail per condition) for the record table.
_summary = {k: {"exact_acc": WO_INSTRUCT_RES[k]["exact_acc"],
                "corr": WO_INSTRUCT_RES[k]["corr"],
                "parse_fail_rate": WO_INSTRUCT_RES[k]["parse_fail_rate"]}
            for k in [c[0] for c in WO_CONDITIONS]}

_md = wo_decision_record_md(
    model_tag="instruct", gate_eval=WO_GATE_EVAL, branch=WO_BRANCH,
    battery_summary=_summary, twobytwo=WO_BASE_2X2_VERDICT,
    jaccard_c1c2=WO_JACCARD_C1C2, acc_delta_c1c2=WO_ACC_DELTA_C1C2, repro=_repro)

# append the downstream protocol the next cell will execute.
_md += "\n## Downstream artifact to produce (Step 5b)\n\n"
_md += f"- **Protocol:** {WO_BRANCH['protocol']}\n"
_md += f"- **Run on:** {', '.join(WO_BRANCH['run_on'])}\n"
if WO_BRANCH["branch"] in ("CLEAN_REPAIR", "PARTIAL_REPAIR"):
    _md += ("- C1/C2 activation-patching localization (§9), first-answer-token recovery "
            "metric (§10), per-(layer,position) heatmaps + `localization_sites.csv`.\n")
else:
    _md += ("- Branch-B selectivity controls (§9.B: addition-precedence analogue, depth "
            "control, operand-magnitude stratification) + the C6→C1 salvage patch "
            "(§10.B): decodable-but-unused causal test.\n")

wo_save_result("decision_record.md", _md)
save_json("wo_branch", WO_BRANCH)

print("\n================= WO STEP 5a — BRANCH SELECTED =================")
print(f"  BRANCH   : {WO_BRANCH['branch']}")
print(f"  PROTOCOL : {WO_BRANCH['protocol']}")
print(f"  RUN ON   : {', '.join(WO_BRANCH['run_on'])}")
print(f"  WHY      : {WO_BRANCH['rationale'][:300]}...")
print("===============================================================")
