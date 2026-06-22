# ============================================================================
# Plausibility dashboard — consolidate G0..G4 from the on-disk gate ledger.
# Reconstructs from ART/gate_status.json, so it reports correctly even after a
# GPU disconnect + kernel restart (no in-memory state required).
# ============================================================================

def _read_gates():
    try:
        return get_gates()                       # checkpoint cell's ledger reader
    except Exception:
        for nm in ("gate_status", "gates"):
            try:
                if has_artifact(nm, "json"):
                    return load_json(nm)
            except Exception:
                pass
    return {}

_g = _read_gates()

def _ok(name):
    e = _g.get(name)
    if not e:
        return None
    return bool(e.get("passed", e.get("pass")))

def _detail(name):
    e = _g.get(name) or {}
    return e.get("detail", "")

_ROWS = [
    ("G_INFRA", "Infra  — checkpoint/resume + artifact round-trip", False),
    ("G0",      "Phase 0 — model loads + hooks (smoke test)",       True),
    ("G1",      "Phase 1 — novelty (MANUAL: see Phase 1 table)",    True),
    ("G2",      "Phase 2 — controlled stimulus (token-identical)",  True),
    ("G3",      "Phase 3 — model COMPUTES, engages operand, in-band",True),
    ("G4",      "Phase 5 — patching reproduces a KNOWN result",     True),
]

def _mark(v):
    return {True: "PASS", False: "FAIL", None: " -- "}[v]

print("=" * 74)
print("  OPERATOR-PRECEDENCE INTERPRETABILITY — PLAUSIBILITY DASHBOARD")
print("=" * 74)
for name, label, _core in _ROWS:
    v = _ok(name)
    d = _detail(name)
    d = (d[:60] + "...") if len(d) > 63 else d
    print(f"  [{_mark(v):>4}]  {label:<52}")
    if d:
        print(f"          └ {d}")
print("-" * 74)

# G1 is a manual/markdown gate (no set_gate call): treat 'not recorded' as a
# reminder to read the Phase 1 related-work table, not as a failure.
_core = ["G0", "G2", "G3", "G4"]
_core_vals = {g: _ok(g) for g in _core}
_core_pass = all(_core_vals[g] is True for g in _core)
_g1 = _ok("G1")
_g1_note = {True: "confirmed", False: "FAILED", None: "MANUAL — confirm via Phase 1 table"}[_g1]

# Surface the locked operand band + dataset sizes if those artifacts exist.
def _maybe(name, kind="json"):
    try:
        return load_json(name) if has_artifact(name, kind) else None
    except Exception:
        return None

_band = _maybe("locked_band_spec")
if _band:
    print(f"  Locked must-compute band : operands [{_band.get('operand_lo')}, "
          f"{_band.get('operand_hi')}]  (overall acc={_band.get('overall_accuracy')})")
_p2 = _maybe("phase2_stimuli")
if _p2 is not None:
    print(f"  Phase 2 experimental set : {len(_p2)} token-controlled records on disk")
print("-" * 74)

print(f"  Core gates (G0,G2,G3,G4) : {'ALL PASS' if _core_pass else 'NOT ALL PASS'}")
print(f"  Novelty gate (G1)        : {_g1_note}")
print()
if _core_pass and _g1 is not False:
    print("  VERDICT:  PLAUSIBLE  — G0/G2/G3/G4 green"
          + ("" if _g1 is True else " (pending manual G1 confirmation).") )
    print("            Phase 6 (depth-1 padding-invariance) is the publishable spine.")
else:
    missing = [g for g in _core if _core_vals[g] is not True]
    print(f"  VERDICT:  NOT YET PLAUSIBLE — unresolved core gate(s): {missing or ['G1']}")
    print("            Resolve the earliest failing/again-unrun gate before proceeding.")
    print("            (G2 & G3 failing = the SCIENCE is broken; G0/G4 = the TOOLING.)")
print("=" * 74)
