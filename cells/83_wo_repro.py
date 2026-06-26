# ============================================================================
# Phase 6 / WO — repro.txt + deliverables manifest (work order §12, §13.7).
# Complete enough to reproduce every number: seed, model revisions, TL version,
# prepend_bos, stimulus hashes, band/N, the gated format, and which deliverables
# were produced this run.
# ============================================================================
import sys

def _ver(m):
    # some packages (e.g. certain transformer_lens builds) lack __version__ as a
    # module attribute -> fall back to installed-package metadata.
    try:
        v = getattr(__import__(m), "__version__", None)
        if v:
            return v
    except Exception:
        pass
    try:
        import importlib.metadata as _im
        return _im.version(m)
    except Exception:
        return "n/a"

_lines = []
_lines.append("# repro — operator-precedence Instruct re-run + surface/compose disentangling")
_lines.append("")
_lines.append(f"seed                 : {WO_SEED}")
_lines.append(f"operand band         : {WO_BAND}  (FIXED — comparability with Phase 3.5 base)")
_lines.append(f"N (shared pairs)     : {WO_N}")
_lines.append(f"pairs sha1           : {WO_PAIRS_HASH}")
_lines.append(f"prepend_bos          : {WO_PREPEND_BOS}  (inherited from G4/Phase-0 pipeline)")
_lines.append(f"greedy max_new_tokens: {CFG.get('g3_max_answer_tokens')}  "
              f"(effective value used by _eval_prompts; §5 target K={WO_MAX_NEW_TOKENS})")
_lines.append(f"gated Instruct format: {globals().get('WO_INSTRUCT_FORMAT', 'n/a')}")
_lines.append("")
_lines.append("models:")
for tag, name in WO_MODEL_REGISTRY.items():
    _lines.append(f"  {tag:<9}: {name}  (revision={WO_MODEL_REVISIONS.get(tag)})")
_lines.append("")
_lines.append("versions:")
_lines.append(f"  transformer_lens : {_ver('transformer_lens')}")
_lines.append(f"  torch            : {_ver('torch')}")
_lines.append(f"  transformers     : {_ver('transformers')}")
_lines.append(f"  numpy            : {_ver('numpy')}")
_lines.append(f"  python           : {sys.version.split()[0]}")
_lines.append(f"  using_fallback   : {globals().get('USING_FALLBACK')}")
_lines.append("")
_lines.append("condition surfaces (rendered for B=23,C=47):")
for k, name, render, gt in WO_CONDITIONS:
    _lines.append(f"  {k:<3} {name:<20} {render(23,47):<22} -> gt {gt(23,47)}")
_lines.append("")
_lines.append("branch selected      : " + str(globals().get('WO_BRANCH', {}).get('branch')))
_lines.append("localization verdict : " + str(globals().get('WO_GATE_EVAL', {}).get('verdict')))
_lines.append("base 2x2 verdict     : " + str(globals().get('WO_BASE_2X2_VERDICT', {}).get('verdict')))
_lines.append("")

# WORK ORDER #4 — cross-model replication labels + boundary aux seed (honest record).
_xm_tbl = globals().get("WO_XM_RESULTS", {})
if _xm_tbl:
    _lines.append("WO#4 cross-model replication (shared WO_PAIRS):")
    for _t, _r in _xm_tbl.items():
        _lines.append(f"  {_t:<14}: {str(_r.get('status')):<22} {_r.get('label')}")
    _lines.append(f"  boundary aux-operand seed : {globals().get('WO_BOUNDARY_SEED')}")
    _lines.append("  WO#4 boundary surfaces (rendered for B=23,C=47):")
    for k, name, render, gt in globals().get("WO_BOUNDARY_CONDITIONS", []):
        _lines.append(f"    {k:<4} {name:<18} {render(23,47):<30} -> gt {gt(23,47)}")
    _lines.append("")

# manifest of produced deliverables (present-on-disk check).
_deliverables = [
    "base_2x2.csv", "instruct_battery.csv", "gate_evaluation.json",
    "decision_record.md", "localization_sites.csv", "branchb_controls.csv",
    # WO#2 causal-hardening deliverables (§4):
    "salvage_c6_to_c1_instruct.json", "salvage_c6_to_c1_base.json",
    "confidence_intervals.json", "fewshot_control.csv",
    "wrong_output_distribution.csv",
    # WO#3 few-shot decodability probe (§ decodability contrast):
    "fewshot_decodability_instruct.json", "fewshot_decodability_base.json",
    "fewshot_decodability_summary.csv",
    # WO#3 follow-ups: by-layer curve + non-repairing control (R1 vs R2):
    "fewshot_decodability_by_layer.csv", "fewshot_decodability_control.csv",
    # WORK ORDER #4 — cross-model generality + boundary / decodability / format (§4):
    "cross_model_battery.csv",
    "position_decodability_base.json", "position_decodability_instruct.json",
    "position_decodability_summary.csv", "position_decodability_heatmap.png",
    "boundary_map.csv", "format_recovery.csv", "error_detail.csv",
    # WORK ORDER #5 — contrast-free causal steering (A) + probe selectivity (B):
    "causal_steering_base.json", "causal_steering_instruct.json",
    "causal_steering_summary.csv",
    "causal_steering_layersweep_base.png", "causal_steering_layersweep_instruct.png",
    "probe_selectivity.csv",
    "probe_selectivity_base.json", "probe_selectivity_instruct.json",
    # WORK ORDER #5.1 — steering-instrument calibration (reuses the WO#5 capture):
    "causal_steering_calibration_base.json", "causal_steering_calibration_instruct.json",
    "causal_steering_calibration_summary.csv",
]
_lines.append("deliverables produced (ART/results):")
for d in _deliverables:
    present = (WO_RESULTS / d).exists()
    _lines.append(f"  [{'x' if present else ' '}] {d}")
_lines.append("")
_lines.append("# Reproduce: open the notebook, set HF_TOKEN, Run All. All forward passes are")
_lines.append("# cached per (model-tag, condition) under ART; a fresh runtime resumes from disk.")

wo_save_result("repro.txt", "\n".join(_lines) + "\n")
print("\n".join(_lines))
log("Phase 6 / WO complete — all §12 deliverables emitted to ART/results (mirrored to ./results).")
