# ============================================================================
# Phase 6 / WORK ORDER — PURE-LOGIC SUBSTRATE (CPU only; NO model, NO torch).
# ----------------------------------------------------------------------------
# Implements the "Instruct re-run + surface/compose disentangling" work order.
# This cell defines ONLY deterministic, forward-pass-FREE logic: stimulus pair
# draws, the condition registry (C0..C8 + Branch-B analogues), metric math,
# the SIX validity gates (work order §7), the branch decision tree (§8), the
# 2x2 surface/compose verdict (§6 Step-1), the §10 recovery-normalisation math,
# and the CSV / decision-record builders.
#
# WHY A SEPARATE PURE CELL: every number this logic emits governs a *publishable
# decision* (localization VALID/INVALID -> which paper gets written) and an
# unattended GPU run. So the logic is isolated here, imports only numpy/json/
# stdlib, and is unit-tested on CPU (tests/test_wo_logic.py) BEFORE any A100
# time is spent. The GPU cells (77..82) are thin orchestration over these
# verified functions plus the already-validated _eval_prompts / G4 instrument.
#
# Self-contained-notebook convention (matches the rest of cells/): no repo
# import; everything is inlined so the assembled .ipynb runs standalone on Colab.
# ============================================================================

import json
import math
import hashlib
import re
import numpy as np

# ----------------------------------------------------------------------------
# 0) Model registry + run constants (work order §5, §7, §11).
# ----------------------------------------------------------------------------
WO_MODEL_REGISTRY = {
    "base":     "meta-llama/Llama-3.1-8B",
    "instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # WORK ORDER #4 (§3.1) — cross-model generality. Instruction-following models
    # <= 9B (bf16 weights <= ~18GB, fits A100-40 at the sub-30-token seqs here) plus a
    # Llama-3.2 scale pair. Qwen2.5 is UNGATED (start there); Gemma-2/Mistral/Llama-3.2
    # are gated (accept the license per model page; set HF_TOKEN). A model you can't
    # access must SKIP + report (access_denied), never crash the run (§6 hazard).
    "qwen25_7b_it":  "Qwen/Qwen2.5-7B-Instruct",            # UNGATED — start here
    "gemma2_9b_it":  "google/gemma-2-9b-it",                # gated; ~18GB bf16
    "mistral_7b_it": "mistralai/Mistral-7B-Instruct-v0.3",  # gated
    "llama32_1b_it": "meta-llama/Llama-3.2-1B-Instruct",    # scale pair (small)
    "llama32_3b_it": "meta-llama/Llama-3.2-3B-Instruct",    # scale pair (mid)
}
WO_BAND = (20, 49)        # DO NOT CHANGE (work order §11: comparability w/ Phase 3.5 base).
WO_N = 400                # N=400 shared (B,C) pairs (§5.1).
WO_SEED = 0               # canonical seed; recorded in repro.txt.
WO_MAX_NEW_TOKENS = 8     # greedy budget K=8 (§5: max product 2401 <= 4 digits) (§5).
# prepend_bos is enforced to MATCH the G4/Phase-0 pipeline in the GPU setup cell;
# the value actually used is recorded into repro.txt at run time (§5.5, §11).

# Tunable thresholds for the §6 Step-1 2x2 verdict (kept explicit, not magic).
WO_C8_SURVIVE_ACC = 0.70   # no-space (B*C)= "survives" if acc >= this (§6: "~0.7+").
WO_C8_COLLAPSE_MARGIN = 0.15  # "collapses" if acc(C8) <= acc(C7) + this (stays near C7's ~0.02).


# ----------------------------------------------------------------------------
# 1) Shared (B,C) pair draws — BYTE-FOR-BYTE the recipe in cell 57 (Phase 3.5),
#    so the base battery reproduces the published RESULTS.md numbers and every
#    condition is rendered from ONE pair list (paired deltas + Jaccard valid).
# ----------------------------------------------------------------------------
def wo_build_pairs(n=WO_N, band=WO_BAND, seed=WO_SEED):
    """Deterministic shared operand pairs. Identical RNG recipe to Phase 3.5's
    _build_pairs (np.random.default_rng(seed); reject B<2/C<2 and single-digit x
    single-digit; dedup). With band (20,49) only the dedup ever fires, but the
    trivial-pair guards are kept so a widened band stays consistent."""
    rng = np.random.default_rng(int(seed))
    lo, hi = band
    pairs, seen, tries = [], set(), 0
    while len(pairs) < int(n) and tries < 500000:
        tries += 1
        B = int(rng.integers(lo, hi + 1)); C = int(rng.integers(lo, hi + 1))
        if B < 2 or C < 2:        # trivial (x0 / x1) — never fires for band>=2
            continue
        if B <= 9 and C <= 9:     # single x single (memorized-ish)
            continue
        if (B, C) in seen:
            continue
        seen.add((B, C)); pairs.append((B, C))
    return pairs


def wo_stim_hash(items):
    """Stable hash of a stimulus list (pairs or prompt strings) for repro.txt."""
    if items and isinstance(items[0], (tuple, list)):
        payload = ";".join(f"{int(a)}x{int(b)}" for a, b in items)
    else:
        payload = "\x00".join(str(s) for s in items)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# 2) Condition registry. Surfaces are EXACTLY as written in work order §2/§6
#    (mind the spaces; C7/C8 have NONE). Each entry: (key, name, render, gt).
#    gt(B,C) is the ground-truth integer the greedy continuation must match.
# ----------------------------------------------------------------------------
WO_CONDITIONS = [
    ("C0", "baseline_mult",       lambda B, C: f"{B} * {C} =",         lambda B, C: B * C),
    ("C1", "depth_left",          lambda B, C: f"( 0 + {B} ) * {C} =", lambda B, C: B * C),
    ("C2", "depth_right",         lambda B, C: f"( 0 + {B} * {C} ) =", lambda B, C: B * C),
    ("C3", "parens_only_out",     lambda B, C: f"( {B} ) * {C} =",     lambda B, C: B * C),
    ("C4", "parens_only_in",      lambda B, C: f"( {B} * {C} ) =",     lambda B, C: B * C),
    ("C5", "identity_no_paren",   lambda B, C: f"0 + {B} * {C} =",     lambda B, C: B * C),
    ("C6", "subexpr_alone",       lambda B, C: f"( 0 + {B} ) =",       lambda B, C: B),
    ("C7", "format_variant",      lambda B, C: f"(0+{B})*{C}=",        lambda B, C: B * C),
    # NEW (work order §6 Step-1): the missing 2x2 cell — no-space, inside-bracket.
    ("C8", "nospace_in_bracket",  lambda B, C: f"({B}*{C})=",          lambda B, C: B * C),
]

# Work order §6 2x2: {spaces,no-space} x {inside-bracket, outer-compose}.
#   spaces:   C4 ( B * C )   |  C1 ( 0 + B ) * C
#   no-space: C8 (B*C)       |  C7 (0+B)*C
WO_2X2 = {
    ("spaces",   "inside"):  "C4",
    ("spaces",   "outer"):   "C1",
    ("nospace",  "inside"):  "C8",
    ("nospace",  "outer"):   "C7",
}

# Branch-B (§9.B) selectivity controls — additive-precedence analogue + depth control.
#   A1/A2: if compose FAILS for '*' but SUCCEEDS for '+', the asymmetry is
#          operation-specific (the key selectivity baseline).
#   D1   : redundant nesting, same parse as C1 — isolates paren-depth vs compose-op.
WO_BRANCHB_CONDITIONS = [
    ("A1", "add_compose_left",  lambda B, C: f"( 0 + {B} ) + {C} =",     lambda B, C: B + C),
    ("A2", "add_compose_right", lambda B, C: f"0 + ( {B} + {C} ) =",     lambda B, C: B + C),
    ("D1", "depth_redundant",   lambda B, C: f"( ( 0 + {B} ) ) * {C} =", lambda B, C: B * C),
]


# ----------------------------------------------------------------------------
# 3) Answer parsing + metrics (§5 "Answer extraction & metrics"). Pure.
#    wo_parse_int mirrors Phase 3's parse_int exactly so local tests exercise the
#    SAME parser the GPU cells use (the GPU cells reuse Phase 3's parse_int).
# ----------------------------------------------------------------------------
_WO_NUM_RE = re.compile(r"-?\d[\d,]*")


def wo_parse_int(text):
    """First integer in a greedy continuation; handles leading spaces, commas
    (1,234), multi-token splits (already merged by decode). None on parse failure."""
    if text is None:
        return None
    m = _WO_NUM_RE.search(text.strip())
    if not m:
        return None
    s = m.group(0).replace(",", "").rstrip("-")
    if s in ("", "-"):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def wo_pearson(xs, ys):
    """Pearson r over paired finite values; None if < 3 points or zero variance."""
    xs = [float(x) for x in xs]; ys = [float(y) for y in ys]
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    if np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def wo_summarize(preds, golds):
    """Per-condition stats from PARSED predictions (None == parse failure).
       exact_acc counts a parse failure as incorrect; corr excludes parse fails."""
    n = len(preds)
    correct = [bool(p is not None and p == g) for p, g in zip(preds, golds)]
    parsed = [p is not None for p in preds]
    xs = [float(p) for p, ok in zip(preds, parsed) if ok]
    ys = [float(g) for g, ok in zip(golds, parsed) if ok]
    finite = [abs(float(p)) for p, ok in zip(preds, parsed) if ok]
    return {
        "n": n,
        "exact_acc": float(np.mean(correct)) if correct else 0.0,
        "corr": wo_pearson(xs, ys),
        "parse_fail_rate": float(1.0 - (np.mean(parsed) if parsed else 0.0)),
        "n_parsed": int(sum(parsed)),
        "mean_abs_output": float(np.mean(finite)) if finite else None,
        "correct_mask": correct,
    }


def wo_jaccard(mask_a, mask_b):
    """Jaccard over correct-item index sets: |A∩B| / |A∪B| (§5)."""
    inter = sum(1 for a, b in zip(mask_a, mask_b) if a and b)
    union = sum(1 for a, b in zip(mask_a, mask_b) if a or b)
    return (inter / union) if union else 0.0


def wo_cv_r2(X, y, folds=5, ridge=1.0):
    """Held-out k-fold CV R^2 of a LINEAR probe y~X via DUAL ridge (linear kernel).
    Used by the §10.B salvage to test whether B is linearly DECODABLE from C1's
    post-bracket activations.

    WHY NOT in-sample lstsq: with n << d_model (e.g. n=128, d=4096) an in-sample
    least-squares fit interpolates exactly -> R^2=1.0 even on PURE NOISE, so it
    cannot establish decodability. WHY DUAL RIDGE (not PCA-then-regress): PCA keeps
    HIGH-VARIANCE directions, which need not be the PREDICTIVE ones. Dual ridge
    regresses in the FULL feature space (solving an n×n system, cheap even at
    d=4096) and is scored on a HELD-OUT fold. WHY MEAN-CENTER ONLY (no unit-variance
    scaling): rescaling each dim to unit variance ERASES the prominence of the dims
    that actually encode B (it shrinks the signal needles down to the noise floor),
    which is exactly the structure a decodability probe must keep. Verified on
    synthetic data (tests/test_wo_logic.py): pure noise -> ~0.03, a prominently-
    encoded operand -> ~0.96. lambda is scaled to the kernel trace (feature-scale
    invariant). Pure numpy. Returns CV R^2 (may be negative) or None if too few /
    degenerate."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        return None
    n, d = X.shape
    if n < folds + 2 or len(y) != n or np.std(y) == 0:
        return None
    order = np.random.default_rng(0).permutation(n)
    fold_sizes = np.full(folds, n // folds, dtype=int)
    fold_sizes[: n % folds] += 1
    preds = np.zeros(n)
    start = 0
    for fs in fold_sizes:
        te = order[start:start + fs]
        tr = np.setdiff1d(order, te)
        start += fs
        if len(tr) < 3:
            return None
        Xtr, Xte, ytr = X[tr], X[te], y[tr]
        mu = Xtr.mean(0)                                          # mean-center on TRAIN only
        Xtr_c, Xte_c = Xtr - mu, Xte - mu
        ybar = ytr.mean()
        K = Xtr_c @ Xtr_c.T                                       # [m, m] linear kernel
        lam = ridge * (np.trace(K) / K.shape[0] + 1e-8)          # scale-invariant lambda
        alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), ytr - ybar)
        preds[te] = (Xte_c @ Xtr_c.T) @ alpha + ybar             # dual prediction
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


# ----------------------------------------------------------------------------
# 4) The SIX validity gates (work order §7). Evaluated on the INSTRUCT battery.
#    Thresholds are the §7 table verbatim. Returns each gate's value+pass plus
#    the localization VALID/INVALID verdict and the G_surface scope flag.
# ----------------------------------------------------------------------------
WO_GATE_SPEC = {
    "G_floor":    "acc(C0) >= 0.90",
    "G_neutral":  "acc(C1) >= 0.85 AND |acc(C1)-acc(C4)| <= 0.05",
    "G_symmetry": "|acc(C1)-acc(C2)| <= 0.05",
    "G_quantity": "corr(C1) >= 0.80",
    "G_surface":  "acc(C7) >= 0.70  (SCOPE FLAG, not a hard abort)",
    "G_support":  "Jaccard(C1,C2) >= 0.85",
}


_WO_EPS = 1e-9   # absorbs float-repr noise at INCLUSIVE thresholds (>=/<=). Genuine
#                  accuracy gaps are multiples of 1/N=0.0025 >> _WO_EPS, so this can
#                  never flip a real decision — only the exact-boundary FP artifact
#                  (e.g. 0.90-0.85 == 0.05000000000000004 > 0.05).


def wo_evaluate_gates(acc, corr, jaccard_c1c2):
    """acc, corr: dicts keyed by condition ('C0'..'C8') -> float/None.
       jaccard_c1c2: float. Returns the §7 gate ledger + localization verdict.

    Decision rule (§7): localization VALID iff
        G_floor ∧ G_neutral ∧ G_symmetry ∧ G_quantity ∧ G_support.
    G_surface is a SCOPE FLAG: if everything else passes but G_surface fails,
    localization is valid CONDITIONAL ON SPACED FORMAT (not aborted).
    Thresholds are INCLUSIVE; comparisons carry _WO_EPS so an exact-boundary
    value passes despite float representation error."""
    def a(k):
        v = acc.get(k)
        return None if v is None else float(v)
    def ge(x, thr):   # inclusive >=
        return x is not None and x >= thr - _WO_EPS
    def le(x, thr):   # inclusive <=
        return x is not None and x <= thr + _WO_EPS
    c1c4 = (None if a("C1") is None or a("C4") is None else abs(a("C1") - a("C4")))
    c1c2 = (None if a("C1") is None or a("C2") is None else abs(a("C1") - a("C2")))
    corrC1 = corr.get("C1")

    gates = {}
    gates["G_floor"] = {
        "value": a("C0"), "threshold": 0.90, "op": ">=",
        "pass": bool(ge(a("C0"), 0.90)),
    }
    gates["G_neutral"] = {
        "value": {"acc_C1": a("C1"), "abs_C1_minus_C4": c1c4},
        "threshold": {"acc_C1": 0.85, "abs_C1_minus_C4": 0.05},
        "pass": bool(ge(a("C1"), 0.85) and le(c1c4, 0.05)),
    }
    gates["G_symmetry"] = {
        "value": c1c2, "threshold": 0.05, "op": "<=",
        "pass": bool(le(c1c2, 0.05)),
    }
    gates["G_quantity"] = {
        "value": corrC1, "threshold": 0.80, "op": ">=",
        "pass": bool(ge(corrC1, 0.80)),
    }
    gates["G_surface"] = {
        "value": a("C7"), "threshold": 0.70, "op": ">=",
        "pass": bool(ge(a("C7"), 0.70)),
        "scope_flag": True,
    }
    gates["G_support"] = {
        "value": float(jaccard_c1c2), "threshold": 0.85, "op": ">=",
        "pass": bool(ge(float(jaccard_c1c2), 0.85)),
    }

    hard = ["G_floor", "G_neutral", "G_symmetry", "G_quantity", "G_support"]
    localization_valid = all(gates[g]["pass"] for g in hard)
    failed = [g for g in hard if not gates[g]["pass"]]
    surface_pass = gates["G_surface"]["pass"]

    if localization_valid and surface_pass:
        verdict = "VALID"
        scope = "unconditional (spaced + no-space)"
    elif localization_valid and not surface_pass:
        verdict = "VALID"
        scope = "CONDITIONAL on spaced format (G_surface failed -> scope flag, not abort)"
    else:
        verdict = "INVALID"
        scope = f"localization invalid; hard gates failed: {failed}"

    return {
        "gates": gates,
        "hard_gates": hard,
        "localization_valid": bool(localization_valid),
        "failed_hard_gates": failed,
        "g_surface_pass": bool(surface_pass),
        "verdict": verdict,
        "scope": scope,
        "spec": WO_GATE_SPEC,
    }


# ----------------------------------------------------------------------------
# 5) Branch decision tree (work order §8). Maps the gate verdict to one of three
#    branches and the downstream protocol to run.
# ----------------------------------------------------------------------------
def wo_select_branch(gate_eval):
    """Returns {branch, protocol, rationale, run_on}. Faithful to the §8 tree:
        ALL pass incl. G_surface          -> CLEAN REPAIR   (§9 on base+instruct)
        all pass EXCEPT G_surface          -> PARTIAL REPAIR (§9 on instruct, spaced scope)
        localization invalid               -> NO REPAIR      (§9.B controls + §10.B salvage)
    The third leaf is the superset of the §8 'G_neutral/G_symmetry/G_quantity
    fail' case: ANY hard-gate failure routes to Branch B (incl. the special
    G_floor='Instruct can't multiply' sub-case, which is surfaced in rationale)."""
    valid = gate_eval["localization_valid"]
    surface = gate_eval["g_surface_pass"]
    failed = gate_eval["failed_hard_gates"]

    if valid and surface:
        return {
            "branch": "CLEAN_REPAIR",
            "protocol": "§9 localization on BASE + INSTRUCT",
            "run_on": ["base", "instruct"],
            "rationale": ("All six gates pass incl. G_surface. Precedence is decodable in "
                          "base but not causally composed; tuning installs the compositional "
                          "circuit. Localize the failed-compose step in base, show it repaired "
                          "in Instruct (strongest version; developmental contrast vs "
                          "bag-of-heuristics)."),
        }
    if valid and not surface:
        return {
            "branch": "PARTIAL_REPAIR",
            "protocol": "§9 localization on INSTRUCT (spaced-format scope)",
            "run_on": ["instruct"],
            "rationale": ("All hard gates pass; G_surface fails (predicted). Tuning installs "
                          "precedence composition but not surface robustness. Localization "
                          "valid for the compose step with G_surface as an explicit scope "
                          "condition; cleanly separates two failure modes."),
        }
    special = ""
    if "G_floor" in failed:
        special = (" NOTE: G_floor failed — Instruct cannot do bare multiplication at the "
                   "0.90 floor; this is a capability story distinct from composition and must "
                   "be reported as such before any brittleness claim.")
    return {
        "branch": "NO_REPAIR",
        "protocol": "§9.B Branch-B controls + §10.B C6->C1 salvage",
        "run_on": ["instruct"],   # base already characterised; salvage probes the failing run
        "rationale": ("Localization invalid (hard gates failed: " + ", ".join(failed) + "). "
                      "Symbolic-arithmetic composition is brittle across base and "
                      "instruction-tuned Llama-3.1-8B; precedence is decodable but not "
                      "causally used regardless of tuning. Pivot fully to brittleness "
                      "(Path B), generality-strengthened." + special),
    }


# ----------------------------------------------------------------------------
# 6) The §6 Step-1 2x2 surface/compose verdict. Decides whether the paper has
#    ONE headline (compose-specific) or TWO (surface fragility is independent).
# ----------------------------------------------------------------------------
def wo_2x2_verdict(acc_c4, acc_c7, acc_c8,
                   survive_acc=WO_C8_SURVIVE_ACC, collapse_margin=WO_C8_COLLAPSE_MARGIN):
    """Interpretation rule (§6):
       - no-space (B*C)=  [C8] ALSO collapses (near C7's ~0.02) -> C7 is PURE
         TOKENIZATION -> surface fragility is an independent CO-HEADLINE.
       - no-space (B*C)=  [C8] SURVIVES (acc ~0.7+) -> the collapse is
         COMPOSE-SPECIFIC -> fold C7 into the composition story (one headline)."""
    collapses = (acc_c8 <= acc_c7 + collapse_margin)
    survives = (acc_c8 >= survive_acc)
    if survives and not collapses:
        verdict = "COMPOSE_SPECIFIC"
        headline = ("ONE headline: collapse is compose-specific. No-space inside-bracket "
                    "(B*C)= survives, so C7's collapse is NOT a clean second axis — fold "
                    "surface fragility into the composition story.")
    elif collapses and not survives:
        verdict = "PURE_TOKENIZATION"
        headline = ("TWO headlines: surface fragility is independent. No-space (B*C)= also "
                    "collapses to ~C7 even WITHOUT outer-compose, so C7 is pure tokenization "
                    "sensitivity — a co-headline finding alongside composition asymmetry.")
    else:
        verdict = "AMBIGUOUS"
        headline = ("AMBIGUOUS: no-space (B*C)= sits between collapse and survival "
                    f"(acc={acc_c8:.3f}; C7={acc_c7:.3f}, survive>={survive_acc}). Report the "
                    "number and treat the surface axis as partial, not clean.")
    return {
        "verdict": verdict,
        "headline": headline,
        "acc": {"C4_spaces_inside": acc_c4, "C1_via_caller": None,
                "C7_nospace_outer": acc_c7, "C8_nospace_inside": acc_c8},
        "collapses": bool(collapses),
        "survives": bool(survives),
        "thresholds": {"survive_acc": survive_acc, "collapse_margin": collapse_margin},
    }


# ----------------------------------------------------------------------------
# 7) §10 recovery-normalisation math (pure). The GPU cell supplies the raw
#    first-answer-token logits/log-probs; this normalises to recovery in [0,1].
#    recovery = (patched - corrupted_baseline) / (clean_baseline - corrupted_baseline).
# ----------------------------------------------------------------------------
def wo_recovery(patched, corrupted_baseline, clean_baseline, eps=1e-8):
    """0 at corrupted baseline (failing parse), 1 at clean baseline (working parse).
       For the C1/C2 patch BOTH parses target the SAME product B*C, so 'clean' is
       the higher-accuracy parse (C2/depth_right) and 'corrupted' is the failing
       parse (C1/depth_left); the metric target is the FIRST answer-token score
       (§10). Direction & target are documented at the call site."""
    denom = (clean_baseline - corrupted_baseline) + eps
    return (patched - corrupted_baseline) / denom


def wo_operand_magnitude_bins(pairs, n_bins=5):
    """§9.B operand-magnitude control: bin pair indices by |B·C| into n_bins
       equal-width magnitude bins. Returns list of {lo,hi,idx:[...]}.
       Lets acc(C1) vs acc(C4) be compared at MATCHED product magnitude, so a
       C1 failure cannot be dismissed as 'products got bigger'."""
    prods = np.array([B * C for (B, C) in pairs], dtype=float)
    lo, hi = float(prods.min()), float(prods.max())
    edges = np.linspace(lo, hi + 1e-6, n_bins + 1)
    out = []
    for i in range(n_bins):
        idx = [j for j, p in enumerate(prods) if edges[i] <= p < edges[i + 1]]
        out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "idx": idx, "n": len(idx)})
    return out


# ----------------------------------------------------------------------------
# 8) CSV + decision-record builders (pure string assembly -> §12 deliverables).
# ----------------------------------------------------------------------------
def wo_battery_csv(rows, header):
    """rows: list of dicts; header: list of column names in order. Returns CSV text."""
    out = [",".join(header)]
    for r in rows:
        cells = []
        for h in header:
            v = r.get(h, "")
            if v is None:
                v = ""
            if isinstance(v, float):
                v = f"{v:.6g}"
            s = str(v)
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            cells.append(s)
        out.append(",".join(cells))
    return "\n".join(out) + "\n"


def wo_decision_record_md(model_tag, gate_eval, branch, battery_summary,
                          twobytwo, jaccard_c1c2, acc_delta_c1c2, repro):
    """Assemble results/decision_record.md (§12). Pure markdown from the verified
       gate ledger + branch + battery numbers."""
    g = gate_eval["gates"]
    def fmt(x, nd=3):
        if x is None:
            return "n/a"
        if isinstance(x, dict):
            return ", ".join(f"{k}={fmt(v, nd)}" for k, v in x.items())
        try:
            return f"{float(x):.{nd}f}"
        except (TypeError, ValueError):
            return str(x)
    L = []
    L.append("# Work-order decision record — operator-precedence Instruct re-run\n")
    L.append(f"- **Battery model:** `{WO_MODEL_REGISTRY.get(model_tag, model_tag)}` "
             f"(tag `{model_tag}`)")
    L.append(f"- **Band:** {WO_BAND}  ·  **N:** {WO_N}  ·  **seed:** {WO_SEED}  "
             f"·  **format:** {repro.get('format', 'bare-continuation')}  "
             f"·  **prepend_bos:** {repro.get('prepend_bos')}")
    L.append(f"- **transformer_lens:** {repro.get('transformer_lens')}  ·  "
             f"**model revision:** {repro.get('model_revision')}\n")

    L.append("## Selected branch\n")
    L.append(f"**{branch['branch']}** — {branch['protocol']}\n")
    L.append(f"> {branch['rationale']}\n")

    L.append("## Validity gates (§7, evaluated on the Instruct battery)\n")
    L.append("| gate | definition | value | pass |")
    L.append("|---|---|---|---|")
    for k in ["G_floor", "G_neutral", "G_symmetry", "G_quantity", "G_surface", "G_support"]:
        gg = g[k]
        L.append(f"| {k} | {WO_GATE_SPEC[k]} | {fmt(gg['value'])} | "
                 f"{'✅' if gg['pass'] else '❌'} |")
    L.append("")
    L.append(f"**Localization verdict:** {gate_eval['verdict']} — {gate_eval['scope']}")
    L.append(f"**Hard-gate AND** (G_floor∧G_neutral∧G_symmetry∧G_quantity∧G_support): "
             f"{gate_eval['localization_valid']}")
    if gate_eval["failed_hard_gates"]:
        L.append(f"**Failed hard gates:** {', '.join(gate_eval['failed_hard_gates'])}")
    L.append("")

    L.append("## Battery summary (Instruct)\n")
    L.append("| cond | surface | acc | corr(B·C) | parse-fail |")
    L.append("|---|---|---|---|---|")
    surf = {k: r for (k, r) in battery_summary.items()}
    name_by_key = {c[0]: c[1] for c in WO_CONDITIONS}
    for k in [c[0] for c in WO_CONDITIONS]:
        if k not in surf:
            continue
        r = surf[k]
        L.append(f"| {k} | {name_by_key.get(k, '')} | {fmt(r.get('exact_acc'))} | "
                 f"{fmt(r.get('corr'))} | {fmt(r.get('parse_fail_rate'))} |")
    L.append("")
    L.append(f"Derived: |acc(C1)−acc(C2)| = {fmt(acc_delta_c1c2)}  ·  "
             f"Jaccard(C1,C2) = {fmt(jaccard_c1c2)}")
    L.append("")

    if twobytwo is not None:
        L.append("## Base 2×2 surface/compose verdict (§6 Step-1)\n")
        L.append(f"**{twobytwo['verdict']}** — {twobytwo['headline']}")
        L.append("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# 8b) WORK ORDER #2 — causal-claim hardening pure logic. Forward-pass-FREE math
#     for: bootstrap / Wilson confidence intervals (§3.4), the "what did C1 emit
#     instead" classifier (§3.6), the few-shot prompt builder (pure part of §3.5),
#     and the decodability NULL-control target (pure part of §3.7). Each is unit-
#     tested in tests/test_wo_logic.py BEFORE any A100 time; the GPU cells (82*)
#     are thin orchestration over these verified functions. All RNG is seeded
#     (np.random.default_rng) so every CI / draw is reproducible (work order §6).
# ----------------------------------------------------------------------------
import statistics as _wo_stats


def wo_bootstrap_ci(mask, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of a 0/1 mask (e.g. an accuracy).
    Resamples WITH replacement n_boot times and returns the central (1-alpha)
    percentile interval (lo, hi). Deterministic given `seed`. (None, None) on an
    empty mask. WHY bootstrap (not just Wilson): the headline numbers are means of
    correlated per-item correctness, and the same machinery gives the PAIRED delta
    CI below; the closed-form wo_wilson_ci is provided too as a cross-check."""
    arr = np.asarray(mask, dtype=float).ravel()
    n = arr.size
    if n == 0:
        return (None, None)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def wo_paired_delta_ci(mask_a, mask_b, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for mean(a) - mean(b) where a, b are index-ALIGNED
    0/1 masks (the SAME items evaluated under two conditions, e.g. C4 vs C1 on the
    shared pairs). Resamples ONE set of bootstrap indices and applies it to BOTH
    masks (paired bootstrap), so the per-item pairing is preserved and the CI is
    tighter/correct vs. resampling the two independently. Deterministic.
    (None, None) if lengths differ or empty."""
    a = np.asarray(mask_a, dtype=float).ravel()
    b = np.asarray(mask_b, dtype=float).ravel()
    n = a.size
    if n == 0 or b.size != n:
        return (None, None)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_boot), n))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo = float(np.percentile(deltas, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def wo_wilson_ci(k, n, alpha=0.05):
    """Closed-form Wilson score interval for a binomial proportion k/n (NO RNG).
    More accurate than the normal approximation at extreme p / small n, and a
    deterministic cross-check on the bootstrap. z from the stdlib normal quantile
    (statistics.NormalDist) so cell 76 keeps its numpy/json/stdlib-only contract.
    (None, None) if n == 0. Verified: wo_wilson_ci(27, 400) ~ (0.047, 0.097)."""
    if n is None or int(n) == 0:
        return (None, None)
    n = float(n)
    k = min(max(float(k), 0.0), n)   # clamp to [0,n] so an out-of-domain k never sqrt(neg).
    z = _wo_stats.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def wo_classify_wrong_output(pred, B, C):
    """Bucket C1's PARSED prediction into one diagnostic category (§3.6), so the
    failure mode is legible ('does it return B? C? B+C? garbage?'). Priority order
    (ties resolved top-down, as the work order lists them):
        correct (==B*C) > equals_B > equals_C > equals_B_plus_C > parse_fail > other
    parse_fail (pred is None) is detected first because None equals nothing."""
    if pred is None:
        return "parse_fail"
    if pred == B * C:
        return "correct"
    if pred == B:
        return "equals_B"
    if pred == C:
        return "equals_C"
    if pred == B + C:
        return "equals_B_plus_C"
    return "other"


def wo_fewshot_render(render, gt, shots, test_pair, pool, seed=0):
    """Few-shot prompt for the SAME surface as `render` (pure part of §3.5).
    Prepends `shots` worked examples 'render(b,c) <gt(b,c)>' (one per line), the
    operands (b,c) drawn deterministically (seeded) from `pool` EXCLUDING the test
    pair, then appends the bare test prompt render(*test_pair). Returns the full
    prompt string. It ends at '=' with NO trailing space — the Llama tiktoken
    pitfall cell 75 documents (a trailing space becomes its own token and shifts
    the scored next-token id). Deterministic given `seed`; shots are guaranteed
    distinct from each other (replace=False) and from the test pair (excluded)."""
    B, C = int(test_pair[0]), int(test_pair[1])
    # Shot-pair selection is shared with the WO#4 demo-type builders (cell §8d,
    # _wo_select_shot_pairs — late-bound: cell 76 runs fully before any caller) so
    # a 'correct' demo set and a length-matched 'wrong_answer'/'scrambled' set draw
    # the IDENTICAL pairs and differ only in demo content (the §3.4 confound control).
    chosen = _wo_select_shot_pairs(shots, test_pair, pool, seed)
    lines = [f"{render(b, c)} {gt(b, c)}" for (b, c) in chosen]
    lines.append(render(B, C))
    return "\n".join(lines)


def wo_shuffle_control(values, seed=0):
    """Deterministic permutation of `values` for a decodability NULL control
    (pure part of §3.7): pairing the SAME activations with a SHUFFLED target must
    collapse CV-R^2 to ~0, certifying that a high CV-R^2 for the true target is
    signal and not an artifact of the probe / dimensionality. Returns a numpy
    array; deterministic given `seed`."""
    v = np.asarray(values)
    rng = np.random.default_rng(int(seed))
    return v[rng.permutation(v.shape[0])]


# ----------------------------------------------------------------------------
# 8c) WORK ORDER #3 — few-shot decodability probe pure logic. The probe SITE under
#     a few-shot prefix is the LAST ')' (the test expression is appended last; the
#     FIRST ')' belongs to a SHOT — Hazard #1). And the 0->2->4-shot CV-R^2 trend
#     classifier (decision logic, so it lives here with a test, per the two-tier
#     rule). Both forward-pass-FREE; the GPU cell (82d) is thin orchestration.
# ----------------------------------------------------------------------------
def wo_last_rparen_index(token_strs):
    """Index of the LAST ')' in a list of per-token decoded strings (already
    stripped). Under a few-shot prefix the prompt is
        ( 0 + b1 ) * c1 = a1 \\n ... \\n ( 0 + B ) * C =
    so the FIRST ')' closes a SHOT's bracket; the TEST expression's ')' is the LAST
    one (the test line is appended last, with no answer after its ')'). Returns
    None if there is no ')'. (GPU cell builds token_strs via tokenizer.decode.)"""
    last = None
    for i, t in enumerate(token_strs):
        if (t.strip() if isinstance(t, str) else t) == ")":
            last = i
    return last


def wo_fsprobe_trend(r2_0, r2_2, r2_4, stable_tol=0.05, rise_thr=0.10,
                     collapse_thr=0.10, low_floor=0.30):
    """Classify the 0->2->4-shot best-layer CV-R^2 trend for B decodability at the
    ')' site. Returns (label, detail). DECISION LOGIC ONLY — no causal claim.
        PROBE_SITE_SUSPECT  : few-shot R^2 collapses (drops > collapse_thr below
                              0-shot, or falls under low_floor) -> likely the
                              LAST-')' finder is wrong (Hazard #1); re-check before
                              concluding anything.
        REPRESENTATION_IMPROVES : few-shot raises R^2 by > rise_thr -> few-shot
                              changes the ENCODING, not only its use -> reframe.
        DECODABLE_IN_BOTH   : R^2 stays within stable_tol of 0-shot at 2 AND 4 shot
                              -> representation present in both regimes; few-shot
                              changes USE, not encoding (the paper-strengthening case).
        MIXED               : intermediate; report the numbers as-is.
        INCONCLUSIVE        : a level is missing (None)."""
    vals = [r2_0, r2_2, r2_4]
    if any(v is None for v in vals):
        return ("INCONCLUSIVE", "missing R^2 at one or more shot levels")
    fs = [float(r2_2), float(r2_4)]
    r0 = float(r2_0)
    msg = f"0-shot={r0:.3f}, 2-shot={fs[0]:.3f}, 4-shot={fs[1]:.3f}"
    if any(v < r0 - collapse_thr for v in fs) or min(fs) < low_floor:
        return ("PROBE_SITE_SUSPECT",
                f"few-shot R^2 collapses ({msg}); re-check the LAST-')' finder "
                "(Hazard #1) before concluding.")
    if any(v - r0 > rise_thr for v in fs):
        return ("REPRESENTATION_IMPROVES",
                f"few-shot raises B decodability ({msg}); few-shot changes the "
                "REPRESENTATION, not only its downstream use — reframe the claim.")
    if all(abs(v - r0) <= stable_tol for v in fs):
        return ("DECODABLE_IN_BOTH",
                f"B stays decodable across regimes ({msg}); few-shot changes USE, "
                "not encoding.")
    return ("MIXED", f"intermediate trend ({msg}); report as-is.")


# ----------------------------------------------------------------------------
# 8d) WORK ORDER #4 — cross-model generality + boundary / decodability / format.
#     Forward-pass-FREE math for: the per-model replication verdict (§3.1), the
#     multi-position probe-site finders (§3.2), the boundary-surface registry +
#     trigger classifier (§3.3), the length-matched demo-type builders + format-cue
#     verdict (§3.4), and the refined error classifier (§3.5). Each is unit-tested
#     in tests/test_wo_logic.py BEFORE any A100 time; the GPU cells (82g..82k) are
#     thin orchestration over these verified functions. Governing lesson (§1): a
#     model that does NOT replicate is a RESULT — these functions label it honestly
#     (non-replicator / out-of-scope), never silently drop it.
# ----------------------------------------------------------------------------

# --- §3.1 cross-model replication verdict ------------------------------------
# Thresholds are PARAMS (documented), not magic. Each mirrors the single-checkpoint
# finding the cross-model run must reproduce (A100_run_2026-06-24.md):
#   parts_floor   — C4 (inside-bracket mult) AND C6 (bracket eval) must clear this,
#                   else the model can't even do the PARTS -> capability, not a
#                   composition failure (the cross-model analogue of G_floor).
#   collapse_gap  — C4 - C1: the compose collapse (instruct 0.91-0.27=0.64).
#   opspecific_gap— A1 - C1: addition composes where multiplication doesn't (0.73).
#   depth_gap     — C1 - D1: one redundant paren layer crashes it (0.27-0.04=0.23).
#   fewshot_gain  — fewshot@4 - C1: 2-4 in-context examples recover it (+0.65).
#   fewshot_ceiling_slack — fewshot@4 must reach within this of the C4 ceiling.
WO_REPL_THR = {
    "parts_floor": 0.80,
    "collapse_gap": 0.20,
    "opspecific_gap": 0.20,
    "depth_gap": 0.15,
    "fewshot_gain": 0.20,
    "fewshot_ceiling_slack": 0.15,
}


def wo_replication_verdict(acc, fewshot_c1_4, thr=None):
    """Per-model replication verdict (§3.1). `acc` is {cond: accuracy} with keys
    among C1/C4/C6/A1/D1 (None or missing => that flag can't be established).
    `fewshot_c1_4` is C1 accuracy at 4 shots (None if not run). Returns the boolean
    flags + an overall `label`. Thresholds are `thr` overrides on WO_REPL_THR.

    label hierarchy:
      INCOMPLETE          — a CORE input (C1/C4/C6) is missing; can't decide.
      OUT_OF_SCOPE (...)  — parts_work is False: the model can't do C4/C6, so a low
                            C1 is a capability gap, NOT a composition failure
                            (report separately; do NOT count as failure-to-replicate).
      REPLICATES_FULL     — core + operation_specific + fewshot_recovers.
      REPLICATES_CORE     — parts_work + compose_collapses only.
      DOES_NOT_REPLICATE  — parts work but the collapse pattern is absent (a RESULT)."""
    t = dict(WO_REPL_THR)
    if thr:
        t.update(thr)

    def a(k):
        v = acc.get(k) if acc else None
        return None if v is None else float(v)

    cC1, cC4, cC6, cA1, cD1 = a("C1"), a("C4"), a("C6"), a("A1"), a("D1")
    fs4 = None if fewshot_c1_4 is None else float(fewshot_c1_4)
    missing = [k for k in ("C1", "C4", "C6") if a(k) is None]

    parts_work = bool(cC4 is not None and cC6 is not None
                      and cC4 >= t["parts_floor"] and cC6 >= t["parts_floor"])
    compose_collapses = bool(cC4 is not None and cC1 is not None
                             and (cC4 - cC1) >= t["collapse_gap"])
    operation_specific = bool(cA1 is not None and cC1 is not None
                              and (cA1 - cC1) >= t["opspecific_gap"])
    depth_sensitive = bool(cD1 is not None and cC1 is not None
                           and (cC1 - cD1) >= t["depth_gap"])
    fewshot_recovers = bool(fs4 is not None and cC1 is not None and cC4 is not None
                            and (fs4 - cC1) >= t["fewshot_gain"]
                            and fs4 >= (cC4 - t["fewshot_ceiling_slack"]))

    replicates_core = bool(parts_work and compose_collapses)
    replicates_full = bool(replicates_core and operation_specific and fewshot_recovers)
    out_of_scope = bool((not missing) and (not parts_work))

    if missing:
        label = f"INCOMPLETE (missing {','.join(missing)})"
    elif not parts_work:
        label = "OUT_OF_SCOPE (can't even do C4/C6 — capability, not composition)"
    elif replicates_full:
        label = "REPLICATES_FULL"
    elif replicates_core:
        label = "REPLICATES_CORE"
    else:
        label = "DOES_NOT_REPLICATE"

    return {
        "parts_work": parts_work,
        "compose_collapses": compose_collapses,
        "operation_specific": operation_specific,
        "depth_sensitive": depth_sensitive,
        "fewshot_recovers": fewshot_recovers,
        "replicates_core": replicates_core,
        "replicates_full": replicates_full,
        "out_of_scope": out_of_scope,
        "label": label,
        "missing": missing,
        "thresholds": t,
    }


# --- §3.4/§3.2 shared: deterministic wrong answer + shot-pair selection -------
def wo_gt_wrong(b, c):
    """Deterministic RANDOM wrong answer with the SAME #digits as b*c (length-matched,
    uncorrelated with the true product) — for the non-repairing / wrong-answer demos.
    Moved here from the GPU control cell (82f) per the two-tier rule; byte-identical
    to that logic so cached prompts are unchanged. Pure; seeded by (b,c)."""
    p = int(b) * int(c)
    d = len(str(p))
    lo, hi = 10 ** (d - 1), 10 ** d - 1
    r = np.random.default_rng(int(b) * 100003 + int(c))
    for _ in range(32):
        w = int(r.integers(lo, hi + 1))
        if w != p:
            return w
    return lo if lo != p else lo + 1


def _wo_select_shot_pairs(shots, test_pair, pool, seed=0):
    """Deterministically choose `shots` distinct demo pairs from `pool`, EXCLUDING the
    test pair (no answer leakage). Shared by wo_fewshot_render and the WO#4 demo-type
    builders so length-matched demo variants draw the IDENTICAL pairs. Byte-for-byte
    the selection wo_fewshot_render used before the refactor (same RNG call order)."""
    B, C = int(test_pair[0]), int(test_pair[1])
    cand = [(int(b), int(c)) for (b, c) in pool if (int(b), int(c)) != (B, C)]
    s = int(shots)
    if s <= 0 or not cand:
        return []
    rng = np.random.default_rng(int(seed))
    sel = rng.choice(len(cand), size=min(s, len(cand)), replace=False)
    return [cand[int(i)] for i in sel]


# --- §3.2 multi-position probe-site finders ----------------------------------
# Locate the four probe sites in a tokenized C1 surface '( 0 + B ) * C =' by
# DECODE-AND-WALK on token CONTENT (multi-token operands shift raw indices — mirror
# the robust Phase-2 / _fsp_site_ok locator), and ASSERT the located window reads
# the expected role before any caller probes it. Decodability ONLY (no causal claim).
def wo_last_index(token_strs, target):
    """Index of the LAST per-token decoded string == `target` (already stripped),
    else None. Used for the final '=' (and as the only-')' bare case)."""
    last = None
    for i, t in enumerate(token_strs):
        if (t.strip() if isinstance(t, str) else t) == target:
            last = i
    return last


def wo_first_index_after(token_strs, target, after):
    """Index of the FIRST per-token decoded string == `target` strictly after index
    `after` (e.g. the '*' after the test ')'), else None."""
    if after is None:
        return None
    for i in range(int(after) + 1, len(token_strs)):
        t = token_strs[i]
        if (t.strip() if isinstance(t, str) else t) == target:
            return i
    return None


def _wo_walk_back_int(strs, before):
    """Concatenate the contiguous digit tokens ending just before index `before`
    (skipping pure-space tokens). Returns the integer string (may be '')."""
    digits = ""
    j = int(before) - 1
    while j >= 0:
        s = strs[j].strip() if isinstance(strs[j], str) else strs[j]
        if s == "":
            j -= 1
            continue
        if s.isdigit():
            digits = s + digits
            j -= 1
            continue
        break
    return digits


def wo_locate_c1_sites(token_strs, B, C):
    """Locate the four probe positions in a tokenized C1 surface '( 0 + B ) * C ='.
    token_strs: per-token decoded strings (stripped or not). Returns a dict with
    single representative indices the GPU probe reads the residual at:
        rparen        — the TEST ')' (LAST ')'; reuse wo_last_rparen_index semantics)
        star          — first '*' after ')'
        c_operand     — LAST token of the C operand (it has 'seen' all of C)
        c_operand_span— all C-operand token indices (multi-token operands)
        equals        — the final '='
    plus 'roles' (B sits before ')', C sits after '*') and 'ok' (all four found AND
    roles verify). A caller MUST check 'ok' before probing (Hazard: a different
    tokenizer/format can break the walk -> mark the model tokenizer_incompatible)."""
    strs = [(t.strip() if isinstance(t, str) else t) for t in token_strs]
    n = len(strs)
    out = {"rparen": None, "star": None, "c_operand": None, "c_operand_span": [],
           "equals": None, "roles": {}, "ok": False}
    rp = wo_last_rparen_index(strs)
    out["rparen"] = rp
    if rp is None:
        return out
    star = wo_first_index_after(strs, "*", rp)
    out["star"] = star
    eq = wo_last_index(strs, "=")
    out["equals"] = eq
    if star is None:
        return out
    # C operand = contiguous digit tokens after '*' (up to '=' or end-of-sequence).
    end = eq if (eq is not None and eq > star) else n
    span, digits = [], ""
    for i in range(star + 1, end):
        s = strs[i]
        if s == "":
            if span:
                break
            continue
        if s.isdigit():
            span.append(i)
            digits += s
        elif span:
            break
    out["c_operand_span"] = span
    out["c_operand"] = span[-1] if span else None
    bdig = _wo_walk_back_int(strs, rp)
    out["roles"] = {
        "B_at_rparen": bool(bdig == str(int(B))),
        "C_after_star": bool(digits == str(int(C))),
        "has_star": star is not None,
        "has_equals": eq is not None,
    }
    out["ok"] = bool(out["roles"]["B_at_rparen"] and out["roles"]["C_after_star"]
                     and span and eq is not None)
    return out


# --- §3.3 failure-boundary surfaces + trigger classifier ---------------------
# Vary the trigger: real bracketed sums, swapped inner/outer ops, and depth-2 nests.
# Surfaces need auxiliary operands A (and D) beyond (B,C) — drawn deterministically
# per (B,C), in-band, so EVERY model/condition sees the SAME A,D for a given (B,C)
# (paired across the battery, like WO_PAIRS). Drawn from a per-(B,C) seeded RNG.
WO_BOUNDARY_SEED = 7


def wo_aux_operands(B, C, seed=WO_BOUNDARY_SEED, band=WO_BAND):
    """Deterministic in-band auxiliary operands (A, D) for the boundary surfaces,
    keyed by (B,C). Same (B,C) -> same (A,D) for every model and condition (paired)."""
    lo, hi = band
    r = np.random.default_rng(int(seed) * 1_000_003 + int(B) * 1009 + int(C))
    A = int(r.integers(lo, hi + 1))
    D = int(r.integers(lo, hi + 1))
    return A, D


# (key, name, render(B,C)->str, gt(B,C)->int). Surfaces spaced like C1 (mind spaces).
WO_BOUNDARY_CONDITIONS = [
    # real addition inside the bracket: is the trigger the additive IDENTITY or ANY
    # bracketed sum feeding the outer '*'?
    ("BD1", "addsum_times",
     lambda B, C: f"( {wo_aux_operands(B, C)[0]} + {B} ) * {C} =",
     lambda B, C: (wo_aux_operands(B, C)[0] + B) * C),
    # bracketed sum on the RIGHT, multiplicative outer.
    ("BD2", "outer_times_sum",
     lambda B, C: f"{wo_aux_operands(B, C)[0]} * ( {B} + {C} ) =",
     lambda B, C: wo_aux_operands(B, C)[0] * (B + C)),
    # bracketed PRODUCT, additive outer — does the asymmetry flip? (predicted: works)
    ("BD3", "prod_plus",
     lambda B, C: f"( {wo_aux_operands(B, C)[0]} * {B} ) + {C} =",
     lambda B, C: (wo_aux_operands(B, C)[0] * B) + C),
    # depth-2 nest feeding a multiplicative outer: ( ( 0 + B ) * C ) * D = B*C*D.
    ("BD4", "depth2_times_d",
     lambda B, C: f"( ( 0 + {B} ) * {C} ) * {wo_aux_operands(B, C)[1]} =",
     lambda B, C: B * C * wo_aux_operands(B, C)[1]),
    # two bracketed sums feeding a multiplicative outer: ( A + B ) * ( C + D ) =.
    ("BD5", "sum_times_sum",
     lambda B, C: f"( {wo_aux_operands(B, C)[0]} + {B} ) * ( {C} + {wo_aux_operands(B, C)[1]} ) =",
     lambda B, C: (wo_aux_operands(B, C)[0] + B) * (C + wo_aux_operands(B, C)[1])),
]

# Structural read of each surface for the trigger classifier. The hypothesis under
# test (from the single-checkpoint finding): the model FAILS iff a PARENTHESIZED
# sub-expression is an operand of a MULTIPLICATIVE outer op (outer '*'); a bracket
# feeding an ADDITIVE outer op (outer '+') is fine.
WO_BOUNDARY_STRUCT = {
    "BD1": {"surface": "( A + B ) * C =",            "outer_op": "*", "predict": "fail"},
    "BD2": {"surface": "A * ( B + C ) =",            "outer_op": "*", "predict": "fail"},
    "BD3": {"surface": "( A * B ) + C =",            "outer_op": "+", "predict": "pass"},
    "BD4": {"surface": "( ( 0 + B ) * C ) * D =",    "outer_op": "*", "predict": "fail"},
    "BD5": {"surface": "( A + B ) * ( C + D ) =",    "outer_op": "*", "predict": "pass_or_fail"},
}


def wo_boundary_summary(acc, fail_thr=0.50, pass_thr=0.70):
    """Classify the failure trigger from boundary-surface accuracies. `acc` is
    {key: accuracy} over WO_BOUNDARY_CONDITIONS keys (None/missing -> 'n/a'). An
    accuracy < fail_thr is 'fails', >= pass_thr is 'works', between is 'partial'.
    Returns per-surface rows + whether the data is CONSISTENT with the 'bracketed
    sub-expression feeding a multiplicative outer op' trigger, and a one-line
    characterization. Decision logic only (no model call)."""
    def obs(v):
        if v is None:
            return "n/a"
        return "fails" if v < fail_thr else ("works" if v >= pass_thr else "partial")

    rows = []
    for k, meta in WO_BOUNDARY_STRUCT.items():
        v = None if not acc else acc.get(k)
        rows.append({"key": k, "surface": meta["surface"], "outer_op": meta["outer_op"],
                     "predict": meta["predict"], "acc": v, "observed": obs(v)})

    # Consistency with the outer-'*' => fail / outer-'+' => works rule, evaluated only
    # on the surfaces with a definite prediction and a definite (non-partial) read.
    decided = [r for r in rows if r["predict"] in ("fail", "pass")
               and r["observed"] in ("fails", "works")]
    consistent = bool(decided) and all(
        (r["predict"] == "fail" and r["observed"] == "fails") or
        (r["predict"] == "pass" and r["observed"] == "works")
        for r in decided)
    mult_fail = all(r["observed"] == "fails"
                    for r in rows if r["outer_op"] == "*" and r["observed"] != "n/a")
    add_ok = all(r["observed"] == "works"
                 for r in rows if r["outer_op"] == "+" and r["observed"] != "n/a")

    if consistent and mult_fail and add_ok:
        characterization = ("fails iff a bracketed sub-expression is an operand of a "
                            "multiplicative outer op (outer '+' composes; outer '*' collapses)")
    elif mult_fail and not add_ok:
        characterization = ("bracketed sub-expressions feeding '*' collapse, but the additive "
                            "outer control did not cleanly pass — report per-surface")
    else:
        characterization = "trigger pattern not clean across these surfaces — report per-surface"

    return {"rows": rows, "consistent": consistent, "mult_outer_fails": mult_fail,
            "add_outer_works": add_ok, "characterization": characterization,
            "thresholds": {"fail_thr": fail_thr, "pass_thr": pass_thr}}


# --- §3.4 length-matched demo-type builders + format-cue verdict -------------
# Pin down WHY few-shot recovers C1: is it the demos' arithmetic CONTENT, or just the
# task FORMAT/length? Four demo types at fixed shots (default 4), all length-matched
# to the correct demo (same whitespace-token count -> the only thing that varies is
# demo content/structure, not prefix length). The TEST line is always the canonical
# bare C1; only the DEMOS differ.
WO_DEMO_TYPES = ["correct", "wrong_answer", "scrambled_format", "random_text"]
_WO_FILLER = ["the", "cat", "sat", "on", "a", "mat", "by", "door", "when", "sun",
              "rose", "over", "hill", "and", "far", "away", "bird", "sang", "soft", "now"]


def _wo_demo_line(demo_type, render, gt, b, c, dseed):
    """One length-matched demo line of a given type for operands (b,c). 'correct' is
    byte-identical to wo_fewshot_render's line so wo_demo_render('correct', ...) ==
    wo_fewshot_render(...) for the same seed (asserted in tests)."""
    if demo_type == "correct":
        return f"{render(b, c)} {gt(b, c)}"
    if demo_type == "wrong_answer":
        return f"{render(b, c)} {wo_gt_wrong(b, c)}"
    if demo_type == "scrambled_format":
        # same tokens, permuted LHS (breaks operator-precedence structure), CORRECT value.
        toks = render(b, c).split()           # e.g. ['(','0','+','b',')','*','c','=']
        body = toks[:-1] if toks and toks[-1] == "=" else toks
        r = np.random.default_rng(int(dseed))
        perm = r.permutation(len(body))
        scrambled = " ".join(body[int(i)] for i in perm)
        return f"{scrambled} = {gt(b, c)}"
    if demo_type == "random_text":
        # length-matched NON-arithmetic filler (pure format/length control): same
        # whitespace-token count as a 'correct' line, no numbers, no '=' structure.
        n_tok = len(render(b, c).split()) + 1
        r = np.random.default_rng(int(dseed))
        words = [_WO_FILLER[int(r.integers(0, len(_WO_FILLER)))] for _ in range(n_tok)]
        return " ".join(words)
    raise ValueError(f"unknown demo_type {demo_type!r}; expected {WO_DEMO_TYPES}")


def wo_demo_render(demo_type, render, gt, shots, test_pair, pool, seed=0):
    """Few-shot prompt whose `shots` DEMOS are of `demo_type` (length-matched) and
    whose TEST line is the canonical bare render(test_pair). Shot pairs are the SAME
    across demo types (shared _wo_select_shot_pairs), so the types differ only in demo
    content — the §3.4 confound control. Deterministic given `seed`."""
    B, C = int(test_pair[0]), int(test_pair[1])
    chosen = _wo_select_shot_pairs(shots, test_pair, pool, seed)
    lines = [_wo_demo_line(demo_type, render, gt, b, c, int(seed) + 7919 * (di + 1))
             for di, (b, c) in enumerate(chosen)]
    lines.append(render(B, C))
    return "\n".join(lines)


def wo_format_cue_verdict(acc_by_type, zeroshot_acc, tol=0.15, recover_margin=0.20):
    """Decide whether few-shot recovery is FORMAT-primed or CONTENT-driven (§3.4).
    `acc_by_type`: {demo_type: C1 accuracy at the fixed shot count}. `zeroshot_acc`:
    C1 accuracy at 0 shots. A type 'recovers' if it clears 0-shot by >= recover_margin.
        FORMAT_PRIMED  — wrong_answer recovers ~= correct (within tol) AND random_text
                         does NOT recover (stays near 0-shot) -> in-context recovery is
                         cued by the task FORMAT, not the demos' arithmetic content.
        CONTENT_DRIVEN — only correct recovers (wrong_answer does not) -> the model
                         learns from the demos' VALUES.
        MIXED          — anything else; report the numbers as-is."""
    def g(k):
        v = acc_by_type.get(k) if acc_by_type else None
        return None if v is None else float(v)

    z = None if zeroshot_acc is None else float(zeroshot_acc)
    correct, wrong = g("correct"), g("wrong_answer")
    rand, scr = g("random_text"), g("scrambled_format")

    def recovers(x):
        return x is not None and z is not None and (x - z) >= recover_margin

    def near(x, y):
        return x is not None and y is not None and abs(x - y) <= tol

    def flat(x):  # stays near the 0-shot floor (does not recover)
        return x is not None and z is not None and (not recovers(x))

    wrong_like_correct = bool(near(wrong, correct) and recovers(wrong))
    random_flat = bool(flat(rand))

    if wrong_like_correct and random_flat:
        label = "FORMAT_PRIMED"
        reading = ("format-primed, not content-learned: wrong-answer demos recover C1 about "
                   "as well as correct demos, while length-matched random-text demos do not — "
                   "the cue is the task FORMAT, not the demos' arithmetic values.")
    elif recovers(correct) and not recovers(wrong):
        label = "CONTENT_DRIVEN"
        reading = ("content-driven: only correct demos recover C1; wrong-answer demos do not — "
                   "the model uses the demos' VALUES, not just their format.")
    else:
        label = "MIXED"
        reading = ("mixed/ambiguous: the recovery pattern across demo types is not clean — "
                   "report the per-type accuracies as-is.")

    return {"label": label, "reading": reading,
            "recovers": {k: recovers(g(k)) for k in WO_DEMO_TYPES},
            "acc_by_type": {k: g(k) for k in WO_DEMO_TYPES},
            "zeroshot_acc": z,
            "thresholds": {"tol": tol, "recover_margin": recover_margin}}


# --- §3.5 refined error classifier -------------------------------------------
def wo_classify_error_detail(pred, B, C):
    """Finer-grained bucket for a C1 PARSED prediction (§3.5), to tell 'attempting
    the product and erring' from 'doing something unrelated' (the bag-of-heuristics
    line). Priority order (ties resolved top-down):
        parse_fail (None) > correct (==B*C) > equals_B > equals_C > equals_B_plus_C
        > near_product (|pred - B*C|/(B*C) <= 0.10, i.e. close but wrong)
        > right_magnitude (same #digits as B*C) > unrelated."""
    if pred is None:
        return "parse_fail"
    prod = int(B) * int(C)
    if pred == prod:
        return "correct"
    if pred == int(B):
        return "equals_B"
    if pred == int(C):
        return "equals_C"
    if pred == int(B) + int(C):
        return "equals_B_plus_C"
    if prod != 0 and abs(pred - prod) / abs(prod) <= 0.10:
        return "near_product"
    if len(str(abs(int(pred)))) == len(str(abs(prod))):
        return "right_magnitude"
    return "unrelated"


WO_ERROR_DETAIL_CATS = ["correct", "equals_B", "equals_C", "equals_B_plus_C",
                        "near_product", "right_magnitude", "unrelated", "parse_fail"]


# ============================================================================
# 8e) WORK ORDER #5 — CONTRAST-FREE CAUSAL STEERING + PROBE SELECTIVITY (pure).
# ----------------------------------------------------------------------------
# Forward-pass-FREE math for the two WO#5 experiments. The GPU/CPU cells (82l/
# 82m) are thin orchestration over these verified functions; each is unit-tested
# in tests/test_wo_logic.py and inline below BEFORE any A100 time.
#
#   EXPERIMENT A (cell 82l, GPU) — a contrast-free causal test at the answer site.
#     Fit a ridge probe (the SAME dual-ridge instrument as wo_cv_r2) on a TRAIN
#     half to get a UNIT direction w-hat and a value<->coordinate mapping; on the
#     held-out TEST half, activation-steer the residual along w-hat to write a
#     target VALUE in, then score the GROUND-TRUTH product's first-answer-token
#     logit. Every item yields a logit Δ regardless of argmax — this is what kills
#     the n=0 failure mode of argmax/flip-only metrics on the failing regime.
#
#   EXPERIMENT B (cell 82m, CPU) — probe SELECTIVITY controls that protect the
#     "the product is represented" reading against the obvious reviewer rebuttal
#     ("ridge just approximates B*C from a linearly-present B and C").
#
# All RNG is seeded (np.random.default_rng); numpy/json/stdlib only (cell 76's
# contract) — NO torch, NO model here. The torch steering hook in 82l MIRRORS
# wo_inject_to_target exactly (documented there).
# ----------------------------------------------------------------------------
def wo_fit_ridge_probe(X, y, ridge=1.0):
    """Fit a linear probe y~X by DUAL ridge (linear kernel) and return the PRIMAL
    weight so a single steering DIRECTION + a value<->coordinate mapping can be
    read off. The kernel math is BYTE-FOR-BYTE wo_cv_r2's per-fold solve (mean-
    center on TRAIN only; lambda scaled to the kernel trace), so the steering
    probe IS the decodability probe — no instrument drift between the two claims.

    Returns a dict (or None if degenerate: ndim!=2, n<3, len(y)!=n, zero-variance
    y, or a zero weight):
        w          — primal weight vector (d,)  [predict(x) = w·(x-mu) + ybar]
        mu, ybar   — train feature mean (d,) and train target mean (scalar)
        w_norm     — ||w||   (the probe's value-per-unit-coordinate SLOPE along w-hat)
        direction  — w / ||w||  (the UNIT steering direction w-hat)
        wmu        — w·mu  (cached so the value<->coord maps are O(1))
    The maps (pure functions below) satisfy, for x' obtained by steering x so that
    direction·x' = wo_probe_coord_for_value(fit, v):   predict(x') == v  (exactly,
    because w ∥ direction so steering only moves the value-bearing coordinate)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        return None
    n, d = X.shape
    if n < 3 or len(y) != n or np.std(y) == 0:
        return None
    mu = X.mean(0)
    Xc = X - mu                                              # mean-center on TRAIN only
    ybar = float(y.mean())
    K = Xc @ Xc.T                                            # [n, n] linear kernel
    lam = ridge * (np.trace(K) / K.shape[0] + 1e-8)         # scale-invariant lambda (== wo_cv_r2)
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y - ybar)
    w = Xc.T @ alpha                                        # primal weight (d,)
    w_norm = float(np.linalg.norm(w))
    if not np.isfinite(w_norm) or w_norm == 0.0:
        return None
    return {"w": w, "mu": mu, "ybar": ybar, "w_norm": w_norm,
            "direction": w / w_norm, "wmu": float(w @ mu)}


def wo_probe_predict(fit, X):
    """Probe value-readout predict(x) = w·(x - mu) + ybar for a row or a [m,d]
    batch (the same linear map wo_cv_r2 scores). Returns a float or a (m,) array."""
    X = np.asarray(X, dtype=float)
    out = (X - fit["mu"]) @ fit["w"] + fit["ybar"]
    return float(out) if np.ndim(out) == 0 else out


def wo_probe_coord_for_value(fit, value):
    """The coordinate s along the unit direction w-hat such that a residual with
    direction·x = s is read by the probe as `value`:  s = (value - ybar + w·mu)/||w||.
    (Inverse of predict restricted to the value-bearing axis.) Used to convert a
    target VALUE — e.g. the correct product B·C — into the coordinate the steer
    writes in."""
    return float((float(value) - fit["ybar"] + fit["wmu"]) / fit["w_norm"])


def wo_probe_mean_coord(fit):
    """The coordinate of the TRAIN mean along w-hat (= direction·mu = w·mu/||w||).
    Steering a residual to THIS coordinate is the LEACE-flavoured ERASE: it removes
    the predictive variance along the probe axis while preserving the mean (the
    probe then reads ybar for every item)."""
    return float(fit["wmu"] / fit["w_norm"])


def wo_inject_to_target(resid, direction, target_coord):
    """Activation-steer: write `target_coord` into the residual's coordinate along
    the UNIT probe `direction` —
        resid' = resid + (target_coord - direction·resid) * direction
    so that direction·resid' == target_coord while every orthogonal component is
    untouched (a rank-1 oblique write along one axis). Pure numpy; the last axis is
    the feature axis, so `resid` may be a (d,) vector or a [...,d] batch. `direction`
    must be unit-norm (the caller passes fit['direction']). Returns a NEW array.

    This is the single primitive behind ALL of 82l's interventions:
      • INJECT  : target_coord = wo_probe_coord_for_value(fit, correct_value)
      • ERASE   : target_coord = wo_probe_mean_coord(fit)
      • SHUFFLED: target_coord = wo_probe_coord_for_value(fit, permuted_value)
      • RANDOM  : same call with a norm-matched RANDOM unit `direction`
    The torch hook in 82l reproduces this formula on the device tensor at one
    (layer, position); because 82l steers the CLEAN run, direction·resid there
    equals direction·(cached clean residual), so the whole delta is precomputable
    on CPU from the cached residual and the hook is a pure additive patch."""
    resid = np.asarray(resid, dtype=float)
    direction = np.asarray(direction, dtype=float)
    coord = np.tensordot(resid, direction, axes=([-1], [-1]))   # direction·resid (last axis)
    delta = (float(target_coord) - coord)
    if np.ndim(delta) > 0:
        delta = delta[..., None]                                # broadcast over feature axis
    return resid + delta * direction


def wo_gt_logit(logits_row, gt_tok_id):
    """The logit of the ground-truth first-answer-token id from a final-position
    logit vector (a numpy row or any indexable). Pure: just logits_row[gt_tok_id]
    as a float — isolated so the metric is unit-testable without a model."""
    return float(np.asarray(logits_row, dtype=float)[int(gt_tok_id)])


def wo_logit_diff_gt(logit_gt_intervened, logit_gt_baseline):
    """The headline metric Δ: GT first-answer-token logit (intervened) minus the
    clean-C1 baseline GT logit. Both are scalars (the GPU cell indexes the row on
    device; tests pass scalars / wo_gt_logit outputs). EVERY test item contributes
    a Δ regardless of its argmax — this is the contrast-free property that makes the
    test work on the FAILING regime where flip-rate would be ~0/n."""
    return float(logit_gt_intervened) - float(logit_gt_baseline)


def wo_argmax_is(logits_row, tok_id):
    """True iff argmax of the final-position logits == tok_id (one flip-to-GT
    event). flip-rate-to-GT-product = mean of this over the test items."""
    return bool(int(np.asarray(logits_row, dtype=float).argmax()) == int(tok_id))


# Documented WO#5 steering thresholds (tunable, passed through from the GPU cell).
WO_STEER_RECOVER_THR = 0.5    # mean GT-logit Δ (nats) that counts as a real lift.
WO_STEER_CTRL_MARGIN = 0.0    # inject must EXCEED random & shuffled by at least this.
WO_STEER_NULL_TOL = 0.25      # |Δ| <= this (with a CI bracketing 0) reads as a null.


def wo_steering_verdict(delta_inject, ci_inject, delta_random, delta_shuffled,
                        delta_c4_ref, recover_thr=WO_STEER_RECOVER_THR,
                        ctrl_margin=WO_STEER_CTRL_MARGIN, null_tol=WO_STEER_NULL_TOL,
                        ci_halfwidth_tol=None):
    """Decision logic for Experiment A (§A verdict). Pure; consumes only summary
    scalars so it is fully unit-testable. Arguments:
        delta_inject   — mean GT-logit Δ for PRODUCT injection at the '=' site (headline).
        ci_inject      — (lo, hi) paired-bootstrap CI for delta_inject.
        delta_random   — mean Δ for the norm-matched RANDOM-direction control.
        delta_shuffled — mean Δ for the SHUFFLED-target (wrong product) control.
        delta_c4_ref   — mean Δ for injecting a COUNTERFACTUAL permuted product P' at
                         C4's '=' and scoring P''s OWN first-token logit (ceiling-free
                         positive reference: injecting the TRUE product at C4 has no
                         headroom, so the reference steers a wrong product and shows ITS
                         logit rises — the metric+hook MUST move a routed-product logit here).
    Verdict (matches the work order):
        INCONCLUSIVE  — iff the C4 positive reference itself fails (delta_c4_ref <
                        recover_thr or missing): the instrument can't be shown to move
                        the GT logit, so neither RECOVERS nor CLEAN_NULL is supportable.
        RECOVERS      — 'present and causally sufficient when routed, unused by default':
                        delta_inject >= recover_thr AND its CI excludes 0 (lo > 0) AND it
                        beats BOTH controls by > ctrl_margin (direction- & value-specific).
        CLEAN_NULL    — 'operand/product genuinely ignored downstream, not merely mis-
                        decoded': |delta_inject| <= null_tol AND the whole CI is CONTAINED
                        in the null band [-null_tol, +null_tol] (tightly: half-width <=
                        ci_halfwidth_tol) — AND C4 confirms the metric works. (We require
                        CONTAINMENT, not bracketing-0: a tiny systematic probe bias can put
                        a practically-null CI just off 0, and a bounded-small effect with a
                        tight CI beside a large, working C4 reference IS a clean null.)
        INCONCLUSIVE  — anything else (a positive-but-not-significant / fails-controls
                        middle), reported with a reason (the thresholds are chosen so this
                        is unlikely when C4 works, but it is handled, never silently coerced).
    Returns a rich dict (label + every sub-flag + a human reason)."""
    if ci_halfwidth_tol is None:
        ci_halfwidth_tol = null_tol
    lo, hi = (ci_inject if ci_inject is not None else (None, None))
    have_ci = lo is not None and hi is not None
    c4_ok = (delta_c4_ref is not None and delta_c4_ref >= recover_thr)

    out = {"c4_ref_ok": bool(c4_ok), "delta_inject": delta_inject, "ci_inject": ci_inject,
           "delta_random": delta_random, "delta_shuffled": delta_shuffled,
           "delta_c4_ref": delta_c4_ref, "recover_thr": recover_thr,
           "beats_random": None, "beats_shuffled": None, "ci_excludes_zero": None,
           "ci_brackets_zero": None}

    if not c4_ok:
        out["label"] = "INCONCLUSIVE"
        out["reason"] = ("C4 positive reference FAILED (product-injection at C4's '=' did not "
                         f"raise the GT logit: Δ_C4={_wo_fmt(delta_c4_ref)} < {recover_thr}); the "
                         "steering instrument cannot be shown to move the GT logit, so the C1 "
                         "result is uninterpretable — fix the instrument first.")
        return out

    beats_rand = (delta_inject is not None and delta_random is not None
                  and delta_inject > delta_random + ctrl_margin)
    beats_shuf = (delta_inject is not None and delta_shuffled is not None
                  and delta_inject > delta_shuffled + ctrl_margin)
    ci_excl_zero = bool(have_ci and lo > 0.0)
    ci_brackets_zero = bool(have_ci and lo <= 0.0 <= hi)
    ci_in_null_band = bool(have_ci and lo >= -null_tol and hi <= null_tol)
    ci_halfwidth = (float(hi - lo) / 2.0) if have_ci else None
    out.update({"beats_random": bool(beats_rand), "beats_shuffled": bool(beats_shuf),
                "ci_excludes_zero": ci_excl_zero, "ci_brackets_zero": ci_brackets_zero,
                "ci_in_null_band": ci_in_null_band, "ci_halfwidth": ci_halfwidth})

    recovers = (delta_inject is not None and delta_inject >= recover_thr
                and ci_excl_zero and beats_rand and beats_shuf)
    clean_null = (delta_inject is not None and abs(delta_inject) <= null_tol
                  and ci_in_null_band and ci_halfwidth is not None
                  and ci_halfwidth <= ci_halfwidth_tol)

    if recovers:
        out["label"] = "RECOVERS"
        out["reason"] = (f"Injecting the correct product at '=' raises the GT-logit by "
                         f"{_wo_fmt(delta_inject)} (CI {_wo_fmt(lo)}..{_wo_fmt(hi)} excludes 0), "
                         f"beating the random-direction ({_wo_fmt(delta_random)}) and shuffled-"
                         f"target ({_wo_fmt(delta_shuffled)}) controls. The product is present and "
                         "causally SUFFICIENT when routed to the answer site — unused by default.")
    elif clean_null:
        out["label"] = "CLEAN_NULL"
        out["reason"] = (f"Injecting the product at '=' moves the GT-logit by only "
                         f"{_wo_fmt(delta_inject)} (CI {_wo_fmt(lo)}..{_wo_fmt(hi)} within ±{null_tol}) while "
                         f"the C4 reference confirms the same injection DOES move it ({_wo_fmt(delta_c4_ref)} "
                         f">= {recover_thr}). The product is genuinely IGNORED downstream at this site — "
                         "not merely mis-decoded.")
    else:
        out["label"] = "INCONCLUSIVE"
        bits = []
        if delta_inject is not None and delta_inject < recover_thr and not clean_null:
            bits.append(f"Δ_inject={_wo_fmt(delta_inject)} is between the null tol ({null_tol}) "
                        f"and the recover thr ({recover_thr})")
        if not ci_excl_zero and not ci_in_null_band:
            bits.append("CI is neither clear of 0 (a recovery) nor contained in the null band")
        if not beats_rand:
            bits.append("does not beat the random-direction control")
        if not beats_shuf:
            bits.append("does not beat the shuffled-target control")
        out["reason"] = ("C4 reference works, but the C1 result is ambiguous: "
                         + ("; ".join(bits) if bits else "fails the RECOVERS / CLEAN_NULL criteria")
                         + ".")
    return out


def _wo_fmt(x, nd=3):
    """Compact float formatter for verdict/reason strings ('n/a' on None)."""
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def wo_control_task_labels(keys, seed=0):
    """Hewitt–Liang CONTROL TASK target: a FIXED random real label per UNIQUE key
    (e.g. each unique (B,C) pair), assigned in sorted-key order so it is independent
    of input order and fully deterministic. Same key -> same label (a memorization
    target with NO linguistic/arithmetic structure). A probe's CV-R^2 on this
    measures the probe's CAPACITY to fit arbitrary labels at this n,d; the
    SELECTIVITY = R^2(real target) - R^2(control task) isolates structure the
    residual actually carries from raw fitting power. Returns a float array aligned
    to `keys` (keys may be tuples, lists, or scalars)."""
    def _norm(k):
        return tuple(k) if isinstance(k, (tuple, list, np.ndarray)) else k
    rng = np.random.default_rng(int(seed))
    uniq = {}
    for k in sorted({_norm(k) for k in keys}):
        uniq[k] = float(rng.standard_normal())
    return np.array([uniq[_norm(k)] for k in keys], dtype=float)


def wo_linear_bc_baseline(Bvals, Cvals, n_noise=64, signal_scale=2.0,
                          noise_scale=1.0, seed=0):
    """The DECISIVE selectivity control's synthetic feature matrix: columns that
    carry ONLY B and C LINEARLY (prominently encoded, like a written-in residual
    feature) plus Gaussian noise dims — and NOTHING bilinear. Feeding this to
    wo_cv_r2 with the B*C target answers 'can a linear probe FORM the product from
    raw operands alone?'. A probe can read B and C from it but cannot construct the
    interaction B*C beyond the linear (main-effect) approximation, so its product-
    R^2 is the NEGATIVE baseline / linear ceiling. If the real residual's product-
    R^2 EXCEEDS this baseline, the residual genuinely contains product structure;
    if it does NOT, the residual's high product-R^2 is explained by the linear
    presence of B and C — the honest reading the control exists to expose.

    NOTE (band caveat, surfaced by 82m): over a narrow POSITIVE operand band the
    main-effect approximation to B*C is already strong (the interaction term is a
    small share of Var(B*C)), so this baseline need NOT collapse to ~0; the load-
    bearing quantity is the CONTRAST real - baseline, not the baseline alone.
    Deterministic given `seed`. Returns X with shape [n, n_noise + 4]."""
    B = np.asarray(Bvals, dtype=float)
    C = np.asarray(Cvals, dtype=float)
    n = B.shape[0]
    rng = np.random.default_rng(int(seed))
    X = noise_scale * rng.standard_normal((n, int(n_noise) + 4))
    Bc = B - B.mean()
    Cc = C - C.mean()
    X[:, 0] = signal_scale * Bc + 0.5 * rng.standard_normal(n)   # B prominently, two dims
    X[:, 1] = signal_scale * Bc + 0.5 * rng.standard_normal(n)
    X[:, 2] = signal_scale * Cc + 0.5 * rng.standard_normal(n)   # C prominently, two dims
    X[:, 3] = signal_scale * Cc + 0.5 * rng.standard_normal(n)
    return X


def wo_probe_selectivity(X, Bvals, Cvals, target="B_times_C", folds=5, ridge=1.0,
                         seed=0, n_noise=128):
    """Compute ONE Experiment-B selectivity row from a cached residual matrix X
    (n×d) and the operands, with the FIXED folds=5/ridge=1.0 instrument (wo_cv_r2)
    for ALL four probes so they are directly comparable. target in {'B','C',
    'B_times_C'} picks y. Returns a dict:
        target, n,
        R2_real             — wo_cv_r2(X, y)                       (the headline)
        R2_control_task     — wo_cv_r2(X, Hewitt–Liang labels)     (capacity baseline)
        R2_shuffled         — wo_cv_r2(X, permuted y)              (must collapse ~0)
        R2_linearBC_baseline— wo_cv_r2(linear-(B,C) synth, y)      (the decisive control)
        selectivity         — R2_real - R2_control_task
        baseline_gap        — R2_real - R2_linearBC_baseline
    Pure (numpy) — the GPU cell hands it the cached residuals so Experiment B is
    CPU-only. (For target 'B'/'C' the linearBC baseline is EXPECTED to match — the
    operand IS linearly present — so baseline_gap≈0 there is correct, not a bug; the
    decisive contrast is for target 'B_times_C'.)"""
    B = np.asarray(Bvals, dtype=float)
    C = np.asarray(Cvals, dtype=float)
    ymap = {"B": B, "C": C, "B_times_C": B * C}
    if target not in ymap:
        raise ValueError(f"target must be one of {sorted(ymap)}, got {target!r}")
    y = ymap[target]
    keys = list(zip([int(b) for b in B], [int(c) for c in C]))
    r2_real = wo_cv_r2(X, y, folds=folds, ridge=ridge)
    r2_ctrl = wo_cv_r2(X, wo_control_task_labels(keys, seed=seed), folds=folds, ridge=ridge)
    r2_shuf = wo_cv_r2(X, wo_shuffle_control(y, seed=seed + 1), folds=folds, ridge=ridge)
    r2_base = wo_cv_r2(wo_linear_bc_baseline(B, C, n_noise=n_noise, seed=seed + 2),
                       y, folds=folds, ridge=ridge)
    sel = None if (r2_real is None or r2_ctrl is None) else float(r2_real - r2_ctrl)
    gap = None if (r2_real is None or r2_base is None) else float(r2_real - r2_base)
    return {"target": target, "n": int(len(y)), "R2_real": r2_real,
            "R2_control_task": r2_ctrl, "R2_shuffled": r2_shuf,
            "R2_linearBC_baseline": r2_base, "selectivity": sel, "baseline_gap": gap}


def wo_selectivity_verdict(r2_real, r2_control_task, r2_shuffled, r2_linearBC,
                           sel_margin=0.30, shuffle_floor=0.30, baseline_margin=0.10):
    """Decision logic for Experiment B. Pure. Classifies whether a high product-R^2
    at the '=' site is genuine PRODUCT structure or an artifact:
        REPRESENTED  — real beats the control task by >= sel_margin (selective),
                       the shuffled-target collapses (< shuffle_floor), AND real
                       EXCEEDS the linear-(B,C) baseline by >= baseline_margin
                       (structure beyond the linear main-effect ceiling).
        OPERANDS_ONLY— selective + shuffle collapses, but real does NOT exceed the
                       linear-(B,C) baseline: the product-R^2 is explained by the
                       linear presence of B and C (the reviewer's rebuttal HOLDS at
                       this band) — report honestly, the product claim is unsupported.
        NOT_SELECTIVE— real does not beat the control task / shuffle doesn't collapse:
                       the probe is reading capacity/artifact, not structure.
        INCONCLUSIVE — a required R^2 is missing.
    Returns a rich dict (label + sub-flags + selectivity + baseline_gap + reason)."""
    vals = {"r2_real": r2_real, "r2_control_task": r2_control_task,
            "r2_shuffled": r2_shuffled, "r2_linearBC": r2_linearBC}
    missing = [k for k, v in vals.items() if v is None]
    out = dict(vals)
    out["selectivity"] = (None if r2_real is None or r2_control_task is None
                          else float(r2_real - r2_control_task))
    out["baseline_gap"] = (None if r2_real is None or r2_linearBC is None
                           else float(r2_real - r2_linearBC))
    if missing:
        out["label"] = "INCONCLUSIVE"
        out["reason"] = f"missing R^2 for {missing}; cannot judge selectivity."
        return out
    selective = out["selectivity"] >= sel_margin
    shuffle_collapses = r2_shuffled < shuffle_floor
    exceeds_baseline = out["baseline_gap"] >= baseline_margin
    out.update({"selective": bool(selective), "shuffle_collapses": bool(shuffle_collapses),
                "exceeds_linearBC": bool(exceeds_baseline)})
    if selective and shuffle_collapses and exceeds_baseline:
        out["label"] = "REPRESENTED"
        out["reason"] = (f"R^2_real={_wo_fmt(r2_real)} is selective over the control task "
                         f"(Δ={_wo_fmt(out['selectivity'])}>={sel_margin}), the shuffled target "
                         f"collapses ({_wo_fmt(r2_shuffled)}<{shuffle_floor}), AND it exceeds the "
                         f"linear-(B,C) baseline by {_wo_fmt(out['baseline_gap'])}>={baseline_margin} "
                         "— genuine product structure beyond the linear operand ceiling.")
    elif selective and shuffle_collapses and not exceeds_baseline:
        out["label"] = "OPERANDS_ONLY"
        out["reason"] = (f"R^2_real={_wo_fmt(r2_real)} is selective and the shuffled target "
                         f"collapses, but it does NOT exceed the linear-(B,C) baseline "
                         f"({_wo_fmt(r2_linearBC)}; gap {_wo_fmt(out['baseline_gap'])}<{baseline_margin}): "
                         "the product-R^2 is explained by the LINEAR presence of B and C, not a "
                         "represented product. The reviewer's rebuttal holds at this operand band.")
    else:
        out["label"] = "NOT_SELECTIVE"
        bits = []
        if not selective:
            bits.append(f"selectivity {_wo_fmt(out['selectivity'])} < {sel_margin} (control task "
                        f"R^2={_wo_fmt(r2_control_task)} nearly as high)")
        if not shuffle_collapses:
            bits.append(f"shuffled-target R^2={_wo_fmt(r2_shuffled)} did not collapse (>= {shuffle_floor})")
        out["reason"] = "probe reads capacity/artifact, not structure: " + "; ".join(bits) + "."
    return out


# WORK ORDER #5.1 — steering-instrument calibration thresholds (tunable).
WO_STEER_ZEROABL_FLOOR = 1.0   # zeroing the answer residual must move |Δ GT-logit| >= this.


def wo_steer_calibration_verdict(zero_abl_delta, swap_delta, k_grid, k_deltas,
                                 recover_thr=WO_STEER_RECOVER_THR,
                                 zeroabl_floor=WO_STEER_ZEROABL_FLOOR):
    """Decision logic for WO#5.1 (pure; tested). Adjudicates whether WO#5's
    INCONCLUSIVE steering result is a DEAD INSTRUMENT or a genuine null, by climbing
    a ladder of ever-more-causal interventions at the C4 '=' site (where the product
    IS used). Arguments:
      zero_abl_delta — mean Δ GT-logit when the WHOLE C4 '=' residual is ZEROED (a
                       maximal edit; must move the logit hard, else the hook/site/
                       token convention is broken).
      swap_delta     — mean Δ at the DONOR product's token when the whole C4 '='
                       residual is SWAPPED for a real donor residual emitting P' (a
                       guaranteed-causal activation patch; must raise the donor logit).
      k_grid,k_deltas— parallel: injection magnitude multipliers k and the mean Δ (at
                       P') for the SCALED probe-direction counterfactual (k·δ_min).
    Verdict ladder:
      INSTRUMENT_BROKEN      — |zero_abl_delta| < zeroabl_floor: even zeroing the
                               answer residual barely moves the GT logit → the hook/
                               site/metric is broken; fix that before any causal claim.
      METRIC_OR_SITE_SUSPECT — zero-ablation works but the full donor SWAP does not
                               raise the donor logit (swap_delta < recover_thr): the
                               site is reachable but the donor/metric design is off.
      CALIBRATED@k_star      — the scaled probe-direction inject crosses recover_thr
                               at the smallest k_star: WO#5 was UNDER-POWERED; the
                               probe direction IS causal at magnitude k_star → re-run
                               the C1 test there.
      DEAD_DIRECTION         — swap works (the site IS causal) but NO tested k makes
                               the probe-direction inject cross threshold: the operand-
                               reconstructible probe direction is genuinely not a causal
                               handle (decoding != causal direction) → WO#5's C1 null
                               reflects the WRONG steering axis, not 'product unused'.
    Returns {label, k_star, reason, + the inputs}."""
    out = {"zero_abl_delta": zero_abl_delta, "swap_delta": swap_delta,
           "k_grid": list(k_grid), "k_deltas": list(k_deltas),
           "recover_thr": recover_thr, "zeroabl_floor": zeroabl_floor, "k_star": None}
    if zero_abl_delta is None or abs(zero_abl_delta) < zeroabl_floor:
        out["label"] = "INSTRUMENT_BROKEN"
        out["reason"] = (f"Zeroing the whole C4 '=' residual moved the GT logit by only "
                         f"{_wo_fmt(zero_abl_delta)} (|Δ| < {zeroabl_floor}); the hook/site/token "
                         "convention does not move the output at all — fix the instrument first; "
                         "the WO#5 null is uninterpretable.")
        return out
    if swap_delta is None or swap_delta < recover_thr:
        out["label"] = "METRIC_OR_SITE_SUSPECT"
        out["reason"] = (f"Zero-ablation works (|Δ|={_wo_fmt(zero_abl_delta)}) but a FULL donor swap "
                         f"at C4 '=' only moved the donor logit by {_wo_fmt(swap_delta)} (< {recover_thr}): "
                         "the site is reachable but the donor/metric design is off — investigate before "
                         "trusting the magnitude sweep.")
        return out
    k_star = None
    for k, d in zip(k_grid, k_deltas):
        if d is not None and d >= recover_thr:
            k_star = k
            break
    if k_star is not None:
        out["k_star"] = k_star
        out["label"] = "CALIBRATED"
        out["reason"] = (f"A full donor swap moves the output (Δ={_wo_fmt(swap_delta)}), and the scaled "
                         f"probe-direction inject crosses {recover_thr} at k={k_star}: the WO#5 run was "
                         f"UNDER-POWERED (its k=1 edit was too small/operand-aligned). Re-evaluate the C1 "
                         f"steering test at k={k_star}.")
    else:
        out["label"] = "DEAD_DIRECTION"
        out["reason"] = (f"A full donor swap DOES move the output (Δ={_wo_fmt(swap_delta)} >= {recover_thr}), "
                         f"so the C4 '=' site is causal and the metric works — yet NO tested magnitude "
                         f"(k up to {max(k_grid) if k_grid else 'n/a'}) makes the probe-direction inject cross "
                         f"{recover_thr}. The operand-reconstructible probe direction is genuinely not a causal "
                         "handle (decoding != causal direction); WO#5's C1 null reflects the wrong steering axis, "
                         "not the product being unused downstream.")
    return out


# ----------------------------------------------------------------------------
# 9) Inline self-test (runs on every notebook execution; CPU only, ~instant).
#    Mirrors tests/test_wo_logic.py so a notebook run also fails loudly if the
#    decision logic is wrong. Uses the PUBLISHED base numbers as fixtures.
# ----------------------------------------------------------------------------
def _wo_selftest():
    # base RESULTS.md numbers as a fixture for the gate/branch logic.
    base_acc = {"C0": 0.838, "C1": 0.507, "C2": 0.710, "C3": 0.495, "C4": 0.890,
                "C5": 0.583, "C6": 1.000, "C7": 0.018}
    base_corr = {"C1": 0.060, "C2": 0.282}
    ge = wo_evaluate_gates(base_acc, base_corr, jaccard_c1c2=0.697)
    assert ge["verdict"] == "INVALID", ge
    assert not ge["gates"]["G_symmetry"]["pass"]      # |0.507-0.710|=0.203 > 0.05
    assert not ge["gates"]["G_quantity"]["pass"]      # corr(C1)=0.06 < 0.80
    br = wo_select_branch(ge)
    assert br["branch"] == "NO_REPAIR", br

    # predicted PARTIAL_REPAIR: all hard gates pass, G_surface fails.
    part_acc = {"C0": 0.95, "C1": 0.88, "C2": 0.86, "C4": 0.89, "C7": 0.30}
    part_corr = {"C1": 0.85}
    ge2 = wo_evaluate_gates(part_acc, part_corr, jaccard_c1c2=0.90)
    assert ge2["localization_valid"] and not ge2["g_surface_pass"], ge2
    assert wo_select_branch(ge2)["branch"] == "PARTIAL_REPAIR"

    # CLEAN_REPAIR: everything passes.
    clean_acc = {"C0": 0.96, "C1": 0.90, "C2": 0.89, "C4": 0.91, "C7": 0.80}
    ge3 = wo_evaluate_gates(clean_acc, {"C1": 0.9}, jaccard_c1c2=0.9)
    assert wo_select_branch(ge3)["branch"] == "CLEAN_REPAIR"

    # 2x2 verdicts (predicted: C8 survives -> compose-specific OR collapses -> pure tok).
    assert wo_2x2_verdict(0.89, 0.018, 0.85)["verdict"] == "COMPOSE_SPECIFIC"
    assert wo_2x2_verdict(0.89, 0.018, 0.03)["verdict"] == "PURE_TOKENIZATION"

    # metrics: parsing + summarize + jaccard + recovery.
    assert wo_parse_int(" 1,234 foo") == 1234
    assert wo_parse_int("no digits") is None
    s = wo_summarize([6, None, 8, 9], [6, 7, 8, 0])
    assert abs(s["exact_acc"] - 0.5) < 1e-9 and abs(s["parse_fail_rate"] - 0.25) < 1e-9
    assert abs(wo_jaccard([1, 1, 0], [1, 0, 0]) - 0.5) < 1e-9
    assert abs(wo_recovery(0.5, 0.0, 1.0) - 0.5) < 1e-6

    # WO#2 causal-hardening pure logic (§3.4/§3.6 + pure parts of §3.5/§3.7).
    _lo, _hi = wo_bootstrap_ci([1] * 30 + [0] * 70, n_boot=1000, seed=0)
    assert _lo <= 0.30 <= _hi, (_lo, _hi)
    _dlo, _dhi = wo_paired_delta_ci([1, 0, 1, 0], [1, 0, 1, 0], n_boot=500, seed=0)
    assert _dlo <= 0.0 <= _dhi, (_dlo, _dhi)
    _wlo, _whi = wo_wilson_ci(27, 400)
    assert abs(_wlo - 0.047) < 0.003 and abs(_whi - 0.097) < 0.003, (_wlo, _whi)
    assert wo_classify_wrong_output(23 * 47, 23, 47) == "correct"
    assert wo_classify_wrong_output(23, 23, 47) == "equals_B"
    assert wo_classify_wrong_output(47, 23, 47) == "equals_C"
    assert wo_classify_wrong_output(70, 23, 47) == "equals_B_plus_C"
    assert wo_classify_wrong_output(None, 2, 3) == "parse_fail"
    assert wo_classify_wrong_output(99999, 23, 47) == "other"
    _rC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
    _gC1 = dict((c[0], c[3]) for c in WO_CONDITIONS)["C1"]
    _pool = [(20, 21), (22, 23), (24, 25), (26, 27), (28, 29)]
    assert wo_fewshot_render(_rC1, _gC1, 0, (20, 21), _pool) == _rC1(20, 21)
    _fs = wo_fewshot_render(_rC1, _gC1, 2, (20, 21), _pool, seed=1)
    assert len(_fs.splitlines()) == 3 and _fs.splitlines()[-1] == _rC1(20, 21)
    assert wo_fewshot_render(_rC1, _gC1, 2, (20, 21), _pool, 1) == \
        wo_fewshot_render(_rC1, _gC1, 2, (20, 21), _pool, 1)
    assert not np.array_equal(
        wo_shuffle_control(np.arange(50), 0), np.arange(50))

    # WO#3 few-shot probe pure logic.
    assert wo_last_rparen_index(["(", "0", "+", "22", ")", "*", "33", "=",
                                 "(", "0", "+", "23", ")", "*", "47", "="]) == 12
    assert wo_last_rparen_index(["(", "0", "+", "23", ")", "=", "="]) == 4
    assert wo_last_rparen_index(["a", "b"]) is None
    assert wo_fsprobe_trend(0.90, 0.90, 0.90)[0] == "DECODABLE_IN_BOTH"
    assert wo_fsprobe_trend(0.70, 0.85, 0.90)[0] == "REPRESENTATION_IMPROVES"
    assert wo_fsprobe_trend(0.90, 0.20, 0.20)[0] == "PROBE_SITE_SUSPECT"
    assert wo_fsprobe_trend(0.90, None, 0.9)[0] == "INCONCLUSIVE"

    # WORK ORDER #4 pure logic (§3.1-§3.5).
    # §3.1 replication verdict on the single-checkpoint instruct fixture -> FULL.
    _rv = wo_replication_verdict(
        {"C1": 0.265, "C4": 0.9075, "C6": 1.0, "A1": 0.995, "D1": 0.0375}, 0.915)
    assert _rv["replicates_full"] and _rv["label"] == "REPLICATES_FULL", _rv
    # a model that can't do the parts -> OUT_OF_SCOPE (not a failure-to-replicate).
    _oos = wo_replication_verdict({"C1": 0.1, "C4": 0.4, "C6": 0.5}, 0.2)
    assert _oos["out_of_scope"] and _oos["label"].startswith("OUT_OF_SCOPE"), _oos
    # parts work but no collapse -> DOES_NOT_REPLICATE (an honest non-replicator).
    _nr = wo_replication_verdict({"C1": 0.88, "C4": 0.90, "C6": 0.95}, 0.90)
    assert _nr["parts_work"] and not _nr["compose_collapses"] and _nr["label"] == "DOES_NOT_REPLICATE"
    # missing core input -> INCOMPLETE.
    assert wo_replication_verdict({"C4": 0.9, "C6": 0.9}, 0.9)["label"].startswith("INCOMPLETE")

    # §3.4 demo builders: 'correct' must equal wo_fewshot_render; types length-matched.
    _dr = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
    _dg = dict((c[0], c[3]) for c in WO_CONDITIONS)["C1"]
    _dpool = wo_build_pairs()
    _tp = _dpool[0]
    assert wo_demo_render("correct", _dr, _dg, 4, _tp, _dpool, 3) == \
        wo_fewshot_render(_dr, _dg, 4, _tp, _dpool, 3)
    for _dt in WO_DEMO_TYPES:
        _p = wo_demo_render(_dt, _dr, _dg, 4, _tp, _dpool, 5)
        assert len(_p.splitlines()) == 5 and _p.splitlines()[-1] == _dr(*_tp)
    assert wo_gt_wrong(23, 47) != 23 * 47 and len(str(wo_gt_wrong(23, 47))) == len(str(23 * 47))
    # §3.4 verdict: wrong~=correct & random flat -> FORMAT_PRIMED; only correct -> CONTENT.
    assert wo_format_cue_verdict(
        {"correct": 0.92, "wrong_answer": 0.88, "scrambled_format": 0.80,
         "random_text": 0.30}, 0.27)["label"] == "FORMAT_PRIMED"
    assert wo_format_cue_verdict(
        {"correct": 0.92, "wrong_answer": 0.30, "random_text": 0.30}, 0.27)["label"] == "CONTENT_DRIVEN"

    # §3.2 position finders on a whitespace-token analog of '( 0 + 23 ) * 47 ='.
    _c1toks = ["(", "0", "+", "23", ")", "*", "47", "="]
    _loc = wo_locate_c1_sites(_c1toks, 23, 47)
    assert _loc["ok"] and _loc["rparen"] == 4 and _loc["star"] == 5 \
        and _loc["c_operand"] == 6 and _loc["equals"] == 7, _loc
    # multi-token C operand ('4','7') shifts indices but content-walk still locates it.
    _c1split = ["(", "0", "+", "23", ")", "*", "4", "7", "="]
    _loc2 = wo_locate_c1_sites(_c1split, 23, 47)
    assert _loc2["ok"] and _loc2["c_operand_span"] == [6, 7] and _loc2["equals"] == 8, _loc2
    assert wo_last_index(["=", "x", "="], "=") == 2
    assert wo_first_index_after(["*", "a", "*"], "*", 0) == 2

    # §3.3 boundary surfaces + trigger classifier.
    _br = dict((c[0], c[2]) for c in WO_BOUNDARY_CONDITIONS)
    _bg = dict((c[0], c[3]) for c in WO_BOUNDARY_CONDITIONS)
    _A, _D = wo_aux_operands(23, 47)
    assert _br["BD1"](23, 47) == f"( {_A} + 23 ) * 47 =" and _bg["BD1"](23, 47) == (_A + 23) * 47
    assert _bg["BD4"](23, 47) == 23 * 47 * _D
    assert wo_aux_operands(23, 47) == wo_aux_operands(23, 47)   # deterministic
    _bs = wo_boundary_summary({"BD1": 0.1, "BD2": 0.15, "BD3": 0.85, "BD4": 0.05, "BD5": 0.1})
    assert _bs["consistent"] and "multiplicative outer" in _bs["characterization"], _bs

    # §3.5 refined error classifier (priority + the new fuzzy buckets).
    _ed = wo_classify_error_detail
    assert _ed(23 * 47, 23, 47) == "correct" and _ed(None, 23, 47) == "parse_fail"
    assert _ed(23, 23, 47) == "equals_B" and _ed(70, 23, 47) == "equals_B_plus_C"
    assert _ed(1081 - 50, 23, 47) == "near_product"          # |Δ|/1081 ~ 0.046 <= 0.10
    assert _ed(1500, 23, 47) == "right_magnitude"            # 4 digits like 1081, not near
    assert _ed(7, 23, 47) == "unrelated"                     # 1 digit, far

    # WORK ORDER #5 — steering probe + injection + metric + verdicts (§A/§B).
    _s5_rng = np.random.default_rng(0)
    _s5_n, _s5_d = 60, 12
    _s5_B = _s5_rng.integers(20, 50, _s5_n).astype(float)
    _s5_C = _s5_rng.integers(20, 50, _s5_n).astype(float)
    _s5_X = _s5_rng.standard_normal((_s5_n, _s5_d))
    _s5_X[:, 0] = (_s5_B - _s5_B.mean()) * 2.0 + 0.3 * _s5_rng.standard_normal(_s5_n)
    _s5_fit = wo_fit_ridge_probe(_s5_X, _s5_B)
    assert _s5_fit is not None and abs(np.linalg.norm(_s5_fit["direction"]) - 1.0) < 1e-9
    # value<->coord round trip: steer a row to be READ as an arbitrary value, exactly.
    _s5_coord = wo_probe_coord_for_value(_s5_fit, 999.0)
    _s5_xs = wo_inject_to_target(_s5_X[0], _s5_fit["direction"], _s5_coord)
    assert abs(wo_probe_predict(_s5_fit, _s5_xs) - 999.0) < 1e-5
    # injection is rank-1 along the unit direction (delta ∥ direction).
    _s5_delta = _s5_xs - _s5_X[0]
    assert np.allclose(_s5_delta - (_s5_delta @ _s5_fit["direction"]) * _s5_fit["direction"], 0.0, atol=1e-8)
    # erase (steer to the mean coordinate) -> probe reads the train mean ybar.
    _s5_xe = wo_inject_to_target(_s5_X[0], _s5_fit["direction"], wo_probe_mean_coord(_s5_fit))
    assert abs(wo_probe_predict(_s5_fit, _s5_xe) - _s5_fit["ybar"]) < 1e-5
    # inject formula, explicit: direction=e0, resid=[5,9,2], target 7 -> [7,9,2].
    assert np.allclose(wo_inject_to_target(np.array([5.0, 9.0, 2.0]),
                                           np.array([1.0, 0.0, 0.0]), 7.0), [7.0, 9.0, 2.0])
    # metric helpers (contrast-free: a number per item regardless of argmax).
    _s5_row = np.array([0.1, 2.0, -1.0, 5.0])
    assert abs(wo_gt_logit(_s5_row, 3) - 5.0) < 1e-9
    assert abs(wo_logit_diff_gt(5.0, 2.0) - 3.0) < 1e-9
    assert wo_argmax_is(_s5_row, 3) and not wo_argmax_is(_s5_row, 0)
    # steering verdict: RECOVERS / CLEAN_NULL / INCONCLUSIVE(on failed C4 ref).
    assert wo_steering_verdict(2.0, (0.5, 3.0), 0.1, 0.05, 3.0)["label"] == "RECOVERS"
    assert wo_steering_verdict(0.02, (-0.1, 0.12), 0.0, 0.0, 3.0)["label"] == "CLEAN_NULL"
    _s5_inc = wo_steering_verdict(2.0, (0.5, 3.0), 0.1, 0.05, 0.1)
    assert _s5_inc["label"] == "INCONCLUSIVE" and not _s5_inc["c4_ref_ok"]
    # selectivity controls.
    _s5_keys = [(1, 2), (1, 2), (3, 4)]
    _s5_lab = wo_control_task_labels(_s5_keys, seed=0)
    assert _s5_lab[0] == _s5_lab[1] and _s5_lab[0] != _s5_lab[2]
    assert np.array_equal(wo_control_task_labels(_s5_keys, 0), wo_control_task_labels(_s5_keys, 0))
    _s5_Xbc = wo_linear_bc_baseline(_s5_B, _s5_C, n_noise=20, seed=1)
    assert _s5_Xbc.shape == (_s5_n, 24)
    assert (wo_cv_r2(_s5_Xbc, _s5_B) or 0.0) > 0.5          # B is LINEARLY present in the baseline
    assert wo_selectivity_verdict(0.96, 0.20, 0.05, 0.50)["label"] == "REPRESENTED"
    assert wo_selectivity_verdict(0.96, 0.20, 0.05, 0.95)["label"] == "OPERANDS_ONLY"
    assert wo_selectivity_verdict(0.30, 0.28, 0.25, 0.10)["label"] == "NOT_SELECTIVE"
    # wo_probe_selectivity row on a synthetic product-encoding residual -> selective.
    _s5_Xp = _s5_rng.standard_normal((_s5_n, 40))
    _s5_prod = _s5_B * _s5_C
    for _j in range(3):
        _s5_Xp[:, _j] = (_s5_prod - _s5_prod.mean()) / (_s5_prod.std() + 1e-9) * 2.0 \
            + 0.4 * _s5_rng.standard_normal(_s5_n)
    _s5_sel = wo_probe_selectivity(_s5_Xp, _s5_B, _s5_C, target="B_times_C", seed=3)
    assert _s5_sel["R2_real"] is not None and _s5_sel["selectivity"] is not None
    assert _s5_sel["R2_real"] > (_s5_sel["R2_control_task"] or 0.0)
    # WO#5.1 calibration verdict ladder.
    assert wo_steer_calibration_verdict(-5.0, 2.0, [1, 2, 4, 8], [0.1, 0.3, 0.8, 1.2])["label"] == "CALIBRATED"
    assert wo_steer_calibration_verdict(-5.0, 2.0, [1, 2, 4, 8], [0.1, 0.3, 0.8, 1.2])["k_star"] == 4
    assert wo_steer_calibration_verdict(-5.0, 2.0, [1, 2, 4], [0.1, 0.2, 0.3])["label"] == "DEAD_DIRECTION"
    assert wo_steer_calibration_verdict(-5.0, 0.1, [1, 2], [0.1, 0.2])["label"] == "METRIC_OR_SITE_SUSPECT"
    assert wo_steer_calibration_verdict(-0.2, 2.0, [1, 2], [0.8, 0.9])["label"] == "INSTRUMENT_BROKEN"
    return True


_WO_SELFTEST_OK = _wo_selftest()
try:
    log(f"Phase 6 / WO logic: pure-logic self-test {'PASS' if _WO_SELFTEST_OK else 'FAIL'}.")
except NameError:
    print(f"[wo-logic] self-test {'PASS' if _WO_SELFTEST_OK else 'FAIL'} (no log() — standalone exec).")
