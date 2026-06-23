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

# manifest of produced deliverables (present-on-disk check).
_deliverables = [
    "base_2x2.csv", "instruct_battery.csv", "gate_evaluation.json",
    "decision_record.md", "localization_sites.csv", "branchb_controls.csv",
    "salvage_c6_to_c1.json",
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
