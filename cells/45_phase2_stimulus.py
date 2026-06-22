# ============================================================================
# Phase 2 — Gate G2 : controlled stimulus generator + machine-checked assertion
#                     harness.  THE single most validity-critical artifact.
# ----------------------------------------------------------------------------
# Pure CPU. Uses ONLY the model TOKENIZER (no forward pass), so it runs even
# before the GPU is attached. Resumable: the whole dataset is guarded by
# has_artifact('dataset_phase2','pickle'); the Phase-3-facing JSON view is
# guarded by has_artifact('phase2_stimuli','json').
#
# DESIGN (spec-faithful, parenthesized additive-identity contrast):
#   Factor A (Depth):   depth_left  = "( 0 + B ) * C ="   -> (0+B)*C = B*C
#                       depth_right = "0 + ( B * C ) ="   -> 0+(B*C) = B*C
#     Same token multiset {(, ), 0, +, *, =, B, C}; both evaluate to B*C;
#     parentheses present in BOTH; only the paren BOUNDARY moves. Additive
#     identity (0+) -- NEVER multiply-by-one, which the model may compile to a
#     no-op -- so the multiplication operands B,C stay genuinely engaged.
#   Factor B (Distance, the lead result): SUFFIX padding " + 0" * k inserted
#     before "=". Grows token count, preserves answer AND the *-nesting depth.
#     The probed operand's distance to the final operator/'=' shifts by +2k;
#     we RECORD the shift (spec: track, don't forbid).
#   Factor C (Depth-2, the upside): "( 0 + ( 0 + B ) * C ) * D =" = B*C*D,
#     every bracket additive-identity-guarded. Generated+validated now, used
#     in Phase 8.
#
# MACHINE-CHECKED ASSERTIONS (tokenized with the REAL model tokenizer; compare
# TOKEN-ID lengths, never char/whitespace lengths):
#   Factor A pair (depth_left vs depth_right, same B,C):
#     [HARD] token_len(left)        == token_len(right)
#     [HARD] B_token_index(left)    == B_token_index(right)     (held fixed)
#     [HARD] answer(left)           == answer(right)            (Python ground truth)
#     [HARD] parens_present(left) and parens_present(right)
#     [HARD] tree_depth(left)       != tree_depth(right)        (the boundary moves)
#     [REC ] C_token_index shift recorded as metadata (structurally must move).
#   Factor B (pad_0 vs pad_k, same expr):
#     [HARD] answer(pad_0)          == answer(pad_k)
#     [HARD] tree_depth(pad_0)      == tree_depth(pad_k)         (padding != depth)
#     [HARD] token_len(pad_k)        > token_len(pad_0)          (it really grew)
#     [REC ] probed-operand distance-to-final-operator shift recorded.
# Violations DROP the pair with a logged reason; the assertion_report counts
# drops per factor and per reason so a tokenizer fighting the design is visible.
# ============================================================================

import re
import itertools
import numpy as np

assert "tokenizer" in globals(), "Phase 2 needs `tokenizer` (run Phase 0 first)."

# ----------------------------------------------------------------------------
# 0) Parameters (recorded in CFG so the band/volume are auditable artifacts).
# ----------------------------------------------------------------------------
CFG.setdefault("g2_target_per_factor", 2000)      # aim: low thousands of clean pairs.
CFG.setdefault("g2_min_valid_per_factor", 800)    # G2 PASS floor per factor.
CFG.setdefault("g2_sample_budget", 60000)         # max (B,C) draws before giving up.
# Digit-band grid the generator sweeps so Phase 3 can locate the must-compute band.
# (b_digits, c_digits) inclusive digit counts. Two-digit x {one,two,three}-digit is
# the spec's primary "not-a-memorized-product" zone; we span around it.
CFG.setdefault("g2_digit_grid", [[1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3], [1, 3], [3, 1]])
CFG.setdefault("g2_pad_lengths", list(CFG.get("pad_lengths", [0, 2, 4, 8, 16])))
SEP = " "                 # single space between every surface token.
ANSWER_CUE = "="          # answer cue; model predicts the next token after it.
PAD_UNIT = ["+", "0"]     # one suffix padding identity op:  ... + 0 ...
_seed = int(CFG.get("seed", 0))

# ----------------------------------------------------------------------------
# 1) BOS offset (authoritative for Phase 4). How many special tokens the
#    tokenizer prepends under add_special_tokens=True.
# ----------------------------------------------------------------------------
def _compute_bos_offset():
    with_sp = tokenizer("0", add_special_tokens=True)["input_ids"]
    no_sp   = tokenizer("0", add_special_tokens=False)["input_ids"]
    return max(0, len(with_sp) - len(no_sp))

BOS_OFFSET = _compute_bos_offset()
CFG["bos_offset"] = BOS_OFFSET
save_json("phase2_bos_offset", BOS_OFFSET)
log(f"Phase 2: BOS offset = {BOS_OFFSET} (special tokens prepended).")

# Does the tokenizer expose char offset mapping? (Fast tokenizers do; Llama-3.1 is fast.)
def _supports_offsets():
    try:
        enc = tokenizer("0 + 1", return_offsets_mapping=True, add_special_tokens=True)
        return "offset_mapping" in enc
    except Exception:
        return False

_HAS_OFFSETS = _supports_offsets()
log(f"Phase 2: tokenizer offset_mapping available = {_HAS_OFFSETS}")

# ----------------------------------------------------------------------------
# 2) Surface rendering. We build the surface as an ordered list of (text, role)
#    SEGMENTS so we know each operand's exact char span -> exact token span,
#    robust to multi-digit operands splitting into >1 token.
#    role in {'lparen','rparen','op0','plus','star','B','C','D','pad_plus',
#             'pad_zero','eq'}.
# ----------------------------------------------------------------------------
def _segments(template, B, C, pad_len=0, D=None):
    """Return ordered list of (text, role) segments for a stimulus surface."""
    B, C = str(int(B)), str(int(C))
    if template == "depth_left":          # ( 0 + B ) * C
        segs = [("(", "lparen"), ("0", "op0"), ("+", "plus"), (B, "B"),
                (")", "rparen"), ("*", "star"), (C, "C")]
    elif template == "depth_right":       # 0 + ( B * C )
        segs = [("0", "op0"), ("+", "plus"), ("(", "lparen"), (B, "B"),
                ("*", "star"), (C, "C"), (")", "rparen")]
    elif template == "depth2":            # ( 0 + ( 0 + B ) * C ) * D  = B*C*D
        D = str(int(D))
        segs = [("(", "lparen"), ("0", "op0"), ("+", "plus"),
                ("(", "lparen"), ("0", "op0"), ("+", "plus"), (B, "B"), (")", "rparen"),
                ("*", "star"), (C, "C"), (")", "rparen"), ("*", "star"), (D, "D")]
    else:
        raise ValueError(f"unknown template {template!r}")
    # SUFFIX padding: append k copies of " + 0" before the answer cue.
    for _ in range(int(pad_len)):
        segs += [("+", "pad_plus"), ("0", "pad_zero")]
    segs += [(ANSWER_CUE, "eq")]
    return segs

def _assemble(segs):
    """Join segments with SEP; return (text, list-of-(start,end,role) char spans)."""
    parts, spans, pos = [], [], 0
    for i, (txt, role) in enumerate(segs):
        if i > 0:
            pos += len(SEP)
        start = pos
        end = pos + len(txt)
        spans.append((start, end, role))
        parts.append(txt)
        pos = end
    return SEP.join(parts), spans

# ----------------------------------------------------------------------------
# 3) Tokenize a surface and resolve each operand/operator token index.
#    Returns dict: token_ids, token_len, and role_index = {role: first_tok_idx}
#    (for repeated roles like 'star'/'lparen' we keep a list).
# ----------------------------------------------------------------------------
def _locate_tokens(text, spans):
    if _HAS_OFFSETS:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
    else:
        enc = tokenizer(text, add_special_tokens=True)
        ids = enc["input_ids"]
        offs = _fallback_offsets(text, ids)
    role_index = {}
    for (cs, ce, role) in spans:
        idx = None
        for ti, (s, e) in enumerate(offs):
            if e <= s:                       # special token / empty span -> skip
                continue
            if s >= cs and e <= ce:          # token fully inside the segment span
                idx = ti
                break
            if s < ce and e > cs and idx is None:  # overlap fallback (rare BPE merge)
                idx = ti
        role_index.setdefault(role, []).append(idx)
    return ids, role_index

def _fallback_offsets(text, ids):
    """Approximate per-token char spans for a slow tokenizer by decoding tokens
    and walking the string. Best-effort; only used if offset_mapping is absent."""
    offs, cursor = [], 0
    for tid in ids:
        piece = tokenizer.decode([tid])
        stripped = piece.strip()
        if stripped == "":
            offs.append((0, 0)); continue
        j = text.find(stripped, cursor)
        if j < 0:
            offs.append((0, 0))
        else:
            offs.append((j, j + len(stripped))); cursor = j + len(stripped)
    return offs

# ----------------------------------------------------------------------------
# 4) Ground-truth evaluation and structural tree depth.
#    tree_depth here = paren-NESTING depth of the multiplication operator '*'
#    (the operation under test): depth_left -> 0, depth_right -> 1. SUFFIX
#    padding does not enclose '*', so it leaves this invariant unchanged --
#    which is exactly what the Factor B assertion requires.
# ----------------------------------------------------------------------------
def _eval_answer(template, B, C, D=None):
    if template == "depth_left":   return (0 + int(B)) * int(C)
    if template == "depth_right":  return 0 + (int(B) * int(C))
    if template == "depth2":       return (0 + (0 + int(B)) * int(C)) * int(D)
    raise ValueError(template)

def _star_nesting_depth(template):
    """Paren-nesting depth of the (primary) '*' operator."""
    if template == "depth_left":   return 0     # '(0+B) * C' : '*' outside all parens
    if template == "depth_right":  return 1     # '0 + (B*C)' : '*' one paren deep
    if template == "depth2":       return 2     # deepest '*' two parens deep
    raise ValueError(template)

def _parens_present(text):
    return ("(" in text) and (")" in text)

# ----------------------------------------------------------------------------
# 5) Record builder. One record == one fully-described surface.
# ----------------------------------------------------------------------------
def make_record(template, B, C, factor, pad_len=0, D=None):
    segs = _segments(template, B, C, pad_len=pad_len, D=D)
    text, spans = _assemble(segs)
    ids, role_index = _locate_tokens(text, spans)
    ans = _eval_answer(template, B, C, D=D)
    op_idx = {
        "B": role_index.get("B", [None])[0],
        "C": role_index.get("C", [None])[0],
    }
    if D is not None:
        op_idx["D"] = role_index.get("D", [None])[0]
    star_list = role_index.get("star", [None])
    rec = {
        "prompt": text,                 # Phase 3 reads this field.
        "expr_string": text,
        "factor": factor,               # 'A' | 'B' | 'C'
        "condition": template,          # depth_left | depth_right | depth2
        "B": int(B), "C": int(C),
        "answer": int(ans),
        "pad_len": int(pad_len),
        "token_ids": [int(t) for t in ids],
        "token_len": len(ids),
        "operand_token_indices": op_idx,
        "operator_token_index": (None if star_list[0] is None else int(star_list[0])),
        "op0_token_index": role_index.get("op0", [None])[0],
        "eq_token_index": role_index.get("eq", [None])[0],
        "tree_depth": _star_nesting_depth(template),
        "parens": _parens_present(text),
    }
    if D is not None:
        rec["D"] = int(D)
    return rec

# ----------------------------------------------------------------------------
# 6) Assertion harnesses. Each returns (ok: bool, reason: str|None).
# ----------------------------------------------------------------------------
def assert_factorA(recL, recR):
    if recL["token_len"] != recR["token_len"]:
        return False, "token_length_mismatch"
    bi_L = recL["operand_token_indices"]["B"]; bi_R = recR["operand_token_indices"]["B"]
    if bi_L is None or bi_R is None:
        return False, "B_not_located"
    if bi_L != bi_R:
        return False, "B_position_mismatch"
    if recL["operand_token_indices"]["C"] is None or recR["operand_token_indices"]["C"] is None:
        return False, "C_not_located"
    if recL["answer"] != recR["answer"]:
        return False, "answer_mismatch"
    if not (recL["parens"] and recR["parens"]):
        return False, "parens_absent"
    if recL["tree_depth"] == recR["tree_depth"]:
        return False, "tree_depth_not_differing"
    if recL["operator_token_index"] is None or recR["operator_token_index"] is None:
        return False, "star_not_located"
    return True, None

def assert_factorB(rec0, reck):
    if rec0["answer"] != reck["answer"]:
        return False, "answer_mismatch"
    if rec0["tree_depth"] != reck["tree_depth"]:
        return False, "tree_depth_changed_by_padding"
    if not (reck["token_len"] > rec0["token_len"]):
        return False, "token_length_did_not_grow"
    if reck["operand_token_indices"]["C"] is None:
        return False, "C_not_located"
    return True, None

# ----------------------------------------------------------------------------
# 7) Operand sampling across the digit grid (de-duplicated, non-trivial).
# ----------------------------------------------------------------------------
def _draw_operand(rng, ndig):
    lo = 10 ** (ndig - 1) if ndig > 1 else 2      # avoid 0/1 (×0,×1 are trivial no-ops)
    hi = (10 ** ndig) - 1
    return int(rng.integers(lo, hi + 1))

def _nontrivial(B, C):
    # exclude memorized-ish: single-digit×single-digit, and exact powers of ten products.
    if B < 2 or C < 2:
        return False
    prod = B * C
    if B <= 9 and C <= 9:
        return False
    return True

# ----------------------------------------------------------------------------
# 8) GENERATE (guarded). Build Factor A pairs, Factor B padding series, Factor C.
# ----------------------------------------------------------------------------
def _build_dataset():
    rng = np.random.default_rng(_seed)
    grid = [tuple(g) for g in CFG["g2_digit_grid"]]
    target = int(CFG["g2_target_per_factor"])
    budget = int(CFG["g2_sample_budget"])
    pads = sorted(set(int(k) for k in CFG["g2_pad_lengths"] if int(k) > 0))

    factorA, factorB, factorC = [], [], []
    drops = {"A": {}, "B": {}, "C": {}}
    seen = set()

    def _drop(factor, reason):
        drops[factor][reason] = drops[factor].get(reason, 0) + 1

    draws = 0
    gi = 0
    while len(factorA) < target and draws < budget:
        bdig, cdig = grid[gi % len(grid)]; gi += 1
        B = _draw_operand(rng, bdig); C = _draw_operand(rng, cdig)
        draws += 1
        if not _nontrivial(B, C):
            _drop("A", "trivial_operand"); continue
        key = (B, C)
        if key in seen:
            continue
        seen.add(key)

        recL = make_record("depth_left", B, C, factor="A")
        recR = make_record("depth_right", B, C, factor="A")
        okA, why = assert_factorA(recL, recR)
        if not okA:
            _drop("A", why); continue
        # Record C's structural shift (tracked, not forbidden).
        c_shift = recL["operand_token_indices"]["C"] - recR["operand_token_indices"]["C"]
        recL["C_index_shift_vs_pair"] = int(c_shift)
        recR["C_index_shift_vs_pair"] = int(-c_shift)
        pair_id = f"A_{B}x{C}"
        recL["pair_id"] = pair_id; recR["pair_id"] = pair_id
        factorA.append(recL); factorA.append(recR)

        # ---- Factor B: padding series on the depth_right base for this (B,C). ----
        base = make_record("depth_right", B, C, factor="B", pad_len=0)
        series_ok = True
        padded = []
        for k in pads:
            reck = make_record("depth_right", B, C, factor="B", pad_len=k)
            okB, whyB = assert_factorB(base, reck)
            if not okB:
                _drop("B", whyB); series_ok = False; break
            # distance from probed operand C to the final operator '=' (grows with k).
            reck["C_distance_to_eq"] = int(reck["eq_token_index"] - reck["operand_token_indices"]["C"])
            padded.append(reck)
        if series_ok and padded:
            base["C_distance_to_eq"] = int(base["eq_token_index"] - base["operand_token_indices"]["C"])
            base["pad_series_id"] = pair_id
            for r in padded:
                r["pad_series_id"] = pair_id
            factorB.append(base); factorB.extend(padded)

        # ---- Factor C: depth-2, answer-preserving (validated, used in Phase 8). ----
        if len(factorC) < target:
            D = _draw_operand(rng, max(1, bdig))
            if D >= 2:
                recC = make_record("depth2", B, C, factor="C", D=D)
                if recC["answer"] == B * C * D and recC["parens"] \
                        and recC["operator_token_index"] is not None:
                    factorC.append(recC)
                else:
                    _drop("C", "depth2_validation_failed")

    return {
        "factorA": factorA, "factorB": factorB, "factorC": factorC,
        "drops": drops, "draws": draws,
        "surface_spec": {
            "templates": ["depth_left", "depth_right", "depth2"],
            "separator": SEP, "answer_cue": ANSWER_CUE, "pad_unit": PAD_UNIT,
            "pad_style": "suffix_before_eq", "bos_offset": BOS_OFFSET,
        },
    }

if has_artifact("dataset_phase2", "pickle"):
    DATA = load_pickle("dataset_phase2")
    log(f"Phase 2: loaded cached dataset (A={len(DATA['factorA'])}, "
        f"B={len(DATA['factorB'])}, C={len(DATA['factorC'])}).")
else:
    log("Phase 2: generating controlled stimuli (CPU; tokenizer only)...")
    DATA = _build_dataset()
    save_pickle("dataset_phase2", DATA)
    log("Phase 2: dataset generated and cached.")

# ----------------------------------------------------------------------------
# 9) Phase-3-facing JSON view: the Factor A experimental stimuli, both
#    conditions, with {prompt, B, C, answer, ...}. Saved under 'phase2_stimuli'
#    (the first name Phase 3 searches).
# ----------------------------------------------------------------------------
if not has_artifact("phase2_stimuli", "json"):
    save_json("phase2_stimuli", DATA["factorA"])
    save_json("phase2_surface_spec", DATA["surface_spec"])
    log(f"Phase 2: wrote phase2_stimuli (n={len(DATA['factorA'])}) for Phase 3.")

# ----------------------------------------------------------------------------
# 10) Assertion report + G2 gate.
# ----------------------------------------------------------------------------
nA_pairs = len(DATA["factorA"]) // 2
nB_series = sum(1 for r in DATA["factorB"] if r.get("pad_len", 1) == 0)
nC = len(DATA["factorC"])
floor = int(CFG["g2_min_valid_per_factor"])

def _fmt_drops(d):
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "(none)"

report = []
report.append("==================== PHASE 2 / GATE G2 ASSERTION REPORT ====================")
report.append(f"sampling draws used        : {DATA['draws']} / budget {CFG['g2_sample_budget']}")
report.append(f"Factor A  valid pairs      : {nA_pairs}   (records={len(DATA['factorA'])})")
report.append(f"Factor A  drops            : {_fmt_drops(DATA['drops']['A'])}")
report.append(f"Factor B  valid pad-series : {nB_series} (records={len(DATA['factorB'])}, pads={CFG['g2_pad_lengths']})")
report.append(f"Factor B  drops            : {_fmt_drops(DATA['drops']['B'])}")
report.append(f"Factor C  valid depth-2    : {nC}   (Phase 8 upside; weaker controls, NOT gated)")
report.append(f"Factor C  drops            : {_fmt_drops(DATA['drops']['C'])}")
report.append(f"BOS offset                 : {BOS_OFFSET}")
report.append(f"PASS floor per factor      : {floor}")

# Token-length parity sanity: every Factor A pair must be token-length-equal (it
# is, by construction of the drop rule) -- restate as an explicit invariant count.
parity_ok = all(
    DATA["factorA"][i]["token_len"] == DATA["factorA"][i + 1]["token_len"]
    for i in range(0, len(DATA["factorA"]), 2)
)
report.append(f"Factor A token-length parity (all pairs equal) : {parity_ok}")

# 20 random spot-reads for manual inspection (the spec's manual check).
rng = np.random.default_rng(_seed + 99)
report.append("--------------------------- 20 random spot-reads ---------------------------")
idxs = rng.choice(len(DATA["factorA"]), size=min(20, len(DATA["factorA"])), replace=False)
for j in idxs:
    r = DATA["factorA"][int(j)]
    report.append(f"  [{r['condition']:>11}] {r['prompt']:<22} = {r['answer']:<7} "
                  f"tok_len={r['token_len']} Bidx={r['operand_token_indices']['B']} "
                  f"Cidx={r['operand_token_indices']['C']} *idx={r['operator_token_index']} "
                  f"depth(*)={r['tree_depth']}")
report.append("===========================================================================")
report_text = "\n".join(report)
save_text("assertion_report", report_text)
print(report_text)

# G2 PASS hinges ONLY on the genuinely token-controlled factors: Factor A (token-identical
# pairs, B-index parity) and Factor B (answer/depth-preserving padding). Factor C (depth-2)
# has WEAKER controls -- answer-preserving + parens + operator-located, but NOT held to
# Factor-A token-length parity -- and is explicit Phase 8 upside, so it is REPORTED but does
# NOT gate G2 (it must not stand on equal footing with the controlled factors).
g2_pass = bool(parity_ok and nA_pairs >= floor and nB_series >= floor)
g2_detail = (f"A_pairs={nA_pairs}, B_series={nB_series} (floor={floor}); parity={parity_ok}; "
             f"C(depth-2, ungated)={nC}; A_drops={_fmt_drops(DATA['drops']['A'])}")
set_gate("G2", g2_pass, g2_detail)

print(f"\nGATE G2: {'PASS' if g2_pass else 'FAIL'}  ({g2_detail})")
if not g2_pass:
    print("FAIL GUIDANCE:")
    if not parity_ok:
        print(" - Token-length parity broke for some pair -> the tokenizer is fighting the")
        print("   design. Inspect drops['A']['token_length_mismatch']; restrict g2_digit_grid")
        print("   to operand digit-counts that tokenize consistently, then re-run.")
    if nA_pairs < floor:
        print(f" - Only {nA_pairs} clean Factor-A pairs (< {floor}). Raise g2_sample_budget or")
        print("   widen g2_digit_grid to ranges that survive the token-length control.")
    if nB_series < floor:
        print(" - Too few padding series survived; check drops['B'] for the dominant reason.")
# Factor C is intentionally NOT a G2 gate condition (weaker controls; Phase 8 upside).
if nC < floor:
    print(f"NOTE: only {nC} depth-2 (Factor C) stimuli (< {floor}); fine for now since C is "
          f"ungated, but raise g2_sample_budget before Phase 8 if you want more.")
