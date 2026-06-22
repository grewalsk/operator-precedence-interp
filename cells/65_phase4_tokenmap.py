# ============================================================================
# Phase 4 — Token-boundary mapping (NOT a gate, but it silently corrupts every
#           patch if wrong). Pure tokenizer / CPU.
# ----------------------------------------------------------------------------
# Reconciled to Phase 2's PARENTHESIZED additive-identity templates + SUFFIX
# padding (must match Phase 2's surface EXACTLY or every patch is mis-indexed):
#   depth_left  : "( 0 + B ) * C ="   -> (0+B)*C   ; '*' at paren-depth 0
#   depth_right : "0 + ( B * C ) ="   -> 0+(B*C)   ; '*' at paren-depth 1
#   suffix pad k: append " + 0" * k before "=" (grows length, not '*'-depth).
#
# Two layers of mapping:
#   (1) token_map(template, condition, pad_len) -> CANONICAL indices for the
#       hand-verifiable SINGLE-TOKEN-operand probe (B='3', C='5'). Shared across
#       all single-token-operand examples of a template (the spec's "patch fixed
#       indices" payoff). eq index shifts by +2k under suffix padding; the core
#       operand/operator indices are INVARIANT to k (padding is suffixed).
#   (2) token_map_for_record(rec) -> the EXACT per-example indices Phase 2 already
#       stored, the robust path when operands are multi-token (their token length,
#       hence '*'/C positions, varies example to example).
#
# ADVERSARIAL-REVIEW-style safeguards kept from the prior version:
#   * BOS offset prefers Phase 2's persisted value (CFG['bos_offset'] / artifact);
#     detection is only a fallback and disagreement is logged loudly.
#   * Unit tests locate operator/operands by TOKEN CONTENT in an INDEPENDENT
#     re-tokenization (not by reusing the implementation's offsets), so an
#     off-by-one in the core layout OR the +2k suffix shift actually fails.
# ============================================================================

import json

# Single-token probe operands (single digits are 1 token in Llama BPE) so the
# canonical map is well-defined and shared. Distinct from each other and from the
# pad filler '0' so content-location is unambiguous even under padding.
_PROBE = {"B": "3", "C": "5"}
_SEP = " "
_CUE = "="
_PAD_UNIT = ["+", "0"]   # one suffix identity op; MUST match Phase 2's PAD_UNIT.

# ---- cross-check Phase 2's surface convention so the two cells cannot drift ----
if has_artifact("phase2_surface_spec", "json"):
    _spec = load_json("phase2_surface_spec")
    if _spec.get("separator") != _SEP or _spec.get("answer_cue") != _CUE \
            or _spec.get("pad_style") != "suffix_before_eq":
        log(f"Phase 4 WARNING: phase2_surface_spec {_spec} disagrees with this cell's "
            f"render convention (sep={_SEP!r}, cue={_CUE!r}, suffix pad). Indices may be wrong.")
    else:
        log("Phase 4: surface convention matches Phase 2 (sep/cue/suffix-pad OK).")
else:
    log("Phase 4: no phase2_surface_spec on disk; using built-in render convention.")

# ---------------------------------------------------------------- render (== Phase 2) ----
def _render(template, pad_len=0, B=None, C=None):
    B = _PROBE["B"] if B is None else str(int(B))
    C = _PROBE["C"] if C is None else str(int(C))
    if template == "depth_left":
        toks = ["(", "0", "+", B, ")", "*", C]
    elif template == "depth_right":
        # ( 0 + B * C ) -- anchored to the same "( 0 + B" prefix as depth_left so the two
        # tokenize to equal length on real Llama (must match Phase 2's _segments exactly).
        toks = ["(", "0", "+", B, "*", C, ")"]
    else:
        raise ValueError(f"unknown template {template!r}")
    for _ in range(int(pad_len)):
        toks += _PAD_UNIT[:]            # " + 0" suffix identity op
    toks += [_CUE]
    return _SEP.join(toks)

# ---------------------------------------------------------------- BOS handling ----
def _detect_bos_offset():
    a = tokenizer("0", add_special_tokens=True)["input_ids"]
    b = tokenizer("0", add_special_tokens=False)["input_ids"]
    return max(0, len(a) - len(b))

def _bos_offset():
    declared = None
    if isinstance(CFG, dict) and "bos_offset" in CFG:
        declared = int(CFG["bos_offset"])
    elif has_artifact("phase2_bos_offset", "json"):
        declared = int(load_json("phase2_bos_offset"))
    detected = _detect_bos_offset()
    if declared is not None:
        if declared != detected:
            log(f"Phase 4 WARNING: detected BOS offset {detected} != Phase 2 declared "
                f"{declared}; using Phase 2's {declared} (its tokenization is source of truth).")
        return declared
    return detected

# ---------------------------------------------------------------- content locator ----
def _toks_with_specials(text):
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return ids, [tokenizer.decode([i]).strip() for i in ids]

def _first(toks, sym, start=0):
    for i in range(start, len(toks)):
        if toks[i] == sym:
            return i
    return None

def _resolve_canonical(template):
    """Locate canonical single-token-operand indices BY CONTENT in the BOS-prefixed
    pad_len=0 surface. Returns absolute indices (BOS already included)."""
    text = _render(template, pad_len=0)
    ids, toks = _toks_with_specials(text)
    op0 = _first(toks, "0")                      # first '0' is the additive identity op0
    B   = _first(toks, _PROBE["B"])
    C   = _first(toks, _PROBE["C"])
    star = _first(toks, "*")
    eq  = _first(toks, "=")
    for nm, v in (("op0", op0), ("B", B), ("C", C), ("star", star), ("eq", eq)):
        assert v is not None, f"{template}: could not locate {nm} in {toks!r}"
    return {"op0": op0, "B": B, "C": C, "star": star, "eq": eq,
            "core_len": len(ids), "toks": toks}

# ---------------------------------------------------------------- build & cache ----
if has_artifact("phase4_token_map", "json"):
    _MAP = load_json("phase4_token_map")
    log("Phase 4: loaded cached token-boundary map.")
else:
    _bos = _bos_offset()
    _MAP = {
        "bos_offset": _bos,
        "pad_style": "suffix_before_eq",     # eq shifts +len(PAD_UNIT)*k ; core invariant
        "pad_unit_len": len(_PAD_UNIT),
        "templates": {t: {k: v for k, v in _resolve_canonical(t).items() if k != "toks"}
                      for t in ("depth_left", "depth_right")},
        "notes": "Absolute indices into the BOS-prefixed sequence for SINGLE-TOKEN operands "
                 "(B='3',C='5'). Suffix pad k appends k*'+ 0' before '='; core indices are "
                 "pad-invariant, eq index += pad_unit_len*k. Multi-token operands: use "
                 "token_map_for_record(rec).",
    }
    save_json("phase4_token_map", _MAP)
    log(f"Phase 4: token-boundary map saved (bos_offset={_bos}).")

# ---------------------------------------------------------------- exported API ----
def token_map(template, condition=None, pad_len=0):
    """Canonical single-token-operand indices for a locked template, into the
    BOS-prefixed, suffix-padded sequence:
      - probed_operand          : B index (held FIXED across the depth pair).
      - critical_operator       : '*' index (its binding differs between conditions).
      - intermediate_decodable  : C index (last operand; B*C / held value decodable here).
      - role_flip               : op0 index (the additive-identity '0').
      - answer_cue              : '=' index (shifts by pad_unit_len*pad_len).
    `condition` is accepted for call-site symmetry; the canonical map is condition-free."""
    if template not in _MAP["templates"]:
        raise ValueError(f"unknown template {template!r}; expected {list(_MAP['templates'])}")
    L = _MAP["templates"][template]
    k = int(pad_len)
    shift = _MAP["pad_unit_len"] * k
    return {
        "probed_operand":        L["B"],
        "critical_operator":     L["star"],
        "intermediate_decodable": L["C"],
        "role_flip":             L["op0"],
        "answer_cue":            L["eq"] + shift,
        "core_len":              L["core_len"],
        "bos_offset":            _MAP["bos_offset"],
        "pad_len":               k,
    }

def token_map_for_record(rec):
    """Exact per-example indices straight from the Phase 2 record (robust for
    MULTI-token operands, whose '*'/C positions vary by operand token length)."""
    oi = rec.get("operand_token_indices", {})
    return {
        "probed_operand":         oi.get("B"),
        "critical_operator":      rec.get("operator_token_index"),
        "intermediate_decodable": oi.get("C"),
        "role_flip":              rec.get("op0_token_index"),
        "answer_cue":             rec.get("eq_token_index"),
        "pad_len":                rec.get("pad_len", 0),
    }

# ---------------------------------------------------------------- UNIT TESTS ----
def _reference_indices(template, pad_len):
    """INDEPENDENT ground truth: re-tokenize the padded surface and locate by CONTENT,
    WITHOUT using token_map's stored offsets -> a real off-by-one fails here."""
    ids, toks = _toks_with_specials(_render(template, pad_len=pad_len))
    star = _first(toks, "*")
    op0  = _first(toks, "0")
    B    = _first(toks, _PROBE["B"])
    C    = _first(toks, _PROBE["C"])
    eq   = len(toks) - 1                          # '=' is the final token of the surface
    assert toks[eq] == "=", f"{template} k={pad_len}: last token not '=': {toks!r}"
    return {"probed_operand": B, "critical_operator": star,
            "intermediate_decodable": C, "role_flip": op0, "answer_cue": eq}

def _test_token_map():
    fails = []
    def ck(name, got, want):
        ok = (got == want)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, want {want}")
        if not ok:
            fails.append(name)

    pad_lens = sorted(set([0] + [int(k) for k in CFG.get("pad_lengths", [2, 4, 8])]))

    # (1) NON-TAUTOLOGICAL: token_map must equal the independent content-anchored
    #     reference for every template and every pad length.
    for tmpl in ("depth_left", "depth_right"):
        for k in pad_lens:
            ref = _reference_indices(tmpl, k)
            got = token_map(tmpl, tmpl.split("_")[1], k)
            for key in ("probed_operand", "critical_operator",
                        "intermediate_decodable", "role_flip", "answer_cue"):
                ck(f"{tmpl}.{key}(k={k})", got[key], ref[key])

    # (2) The spec's core invariants of the parenthesized contrast:
    #     B (probed_operand) index EQUAL across conditions; '*'-depth differs so the
    #     critical_operator index differs; answer_cue equal at k=0.
    l0, r0 = token_map("depth_left", "left", 0), token_map("depth_right", "right", 0)
    ck("B index equal across conditions (k=0)", l0["probed_operand"], r0["probed_operand"])
    ck("answer_cue equal across conditions (k=0)", l0["answer_cue"], r0["answer_cue"])
    print(f"  [INFO] critical_operator differs across conditions: "
          f"left={l0['critical_operator']} vs right={r0['critical_operator']} "
          f"({'OK' if l0['critical_operator'] != r0['critical_operator'] else 'UNEXPECTED-SAME'})")

    # (3) Pure suffix-shift property: only answer_cue moves, by pad_unit_len*k.
    base = token_map("depth_right", "right", 0)
    for k in [x for x in pad_lens if x > 0]:
        m = token_map("depth_right", "right", k)
        ck(f"core invariant under pad (B,k={k})", m["probed_operand"], base["probed_operand"])
        ck(f"core invariant under pad (*,k={k})", m["critical_operator"], base["critical_operator"])
        ck(f"answer_cue shift == pad_unit_len*k (k={k})",
           m["answer_cue"] - base["answer_cue"], _MAP["pad_unit_len"] * k)

    # (4) Tokenizer-agnostic ORDERING invariant. (Absolute indices are NOT hardcoded:
    #     real Llama emits standalone bare-space tokens, so literal positions depend on the
    #     tokenizer. The content-anchored ref in (1) already pins exact indices; here we just
    #     assert the structural order, which holds on any tokenizer.)
    m = token_map("depth_right", "right", 0)
    ck("order op0<B",   m["role_flip"]          < m["probed_operand"],        True)
    ck("order B<*",     m["probed_operand"]      < m["critical_operator"],     True)
    ck("order *<C",     m["critical_operator"]   < m["intermediate_decodable"], True)
    ck("order C<eq",    m["intermediate_decodable"] < m["answer_cue"],         True)

    # (5) token_map_for_record agrees with Phase 2 records (if dataset present).
    if has_artifact("phase2_stimuli", "json"):
        recs = load_json("phase2_stimuli")
        depth_lefts = [r for r in recs if r.get("condition") == "depth_left"][:1]
        if depth_lefts:
            r = depth_lefts[0]
            tm = token_map_for_record(r)
            ck("record.probed_operand matches stored B index",
               tm["probed_operand"], r["operand_token_indices"]["B"])
            ck("record.critical_operator matches stored * index",
               tm["critical_operator"], r["operator_token_index"])

    # (6) unknown template raises
    raised = False
    try:
        token_map("depth_middle", "x", 0)
    except ValueError:
        raised = True
    ck("unknown_template_raises", raised, True)

    print(f"Phase 4 token_map unit tests: {'ALL PASS' if not fails else 'FAIL -> ' + ', '.join(fails)}")
    assert not fails, f"token_map unit tests failed: {fails}"

_test_token_map()
log("Phase 4: token-boundary map ready; later phases import token_map / "
    "token_map_for_record (never recompute boundaries ad hoc).")
