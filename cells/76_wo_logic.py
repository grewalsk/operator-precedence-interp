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


# ----------------------------------------------------------------------------
# 8g) WORK ORDER #6 (Tier 2) — CONFIDENCE-INTERVAL + REPRO HYGIENE (pure logic).
#     Apply a CI to EVERY reported number and a run-record to every experiment, and
#     package verdicts as raw-table+explicit-rule instead of a single averaged label
#     (the failure mode that mislabeled the wrapper map + produced the false Gemma
#     'REPLICATES'). All numpy/json/stdlib; unit-tested.
# ----------------------------------------------------------------------------
def wo_acc_ci(acc, n, alpha=0.05):
    """Wilson CI for an accuracy given as a PROPORTION over n items (k=round(acc*n)).
    Attaches a CI to every committed accuracy (wrapper drops, cross-model rows, C0/C4)
    with no re-run — n is the item count (e.g. WO_N=400). (None,None) on missing input."""
    if acc is None or n is None or int(n) == 0:
        return (None, None)
    return wo_wilson_ci(int(round(float(acc) * int(n))), int(n), alpha=alpha)


def wo_r2_bootstrap_ci(X, y, folds=5, ridge=1.0, n_boot=200, alpha=0.05, seed=0):
    """Item (case) bootstrap CI for the dual-ridge CV-R^2 (wo_cv_r2): resample the n
    items WITH replacement n_boot times, recompute CV-R^2 on each draw, return the
    percentile (1-alpha) interval — the item-sampling variability of a decodability R^2
    that reviewers ask for on every probe number. Pure numpy + wo_cv_r2; n_boot modest
    (each draw is a full CV). NOTE: resample-then-CV mildly under-states width via
    duplicate items spanning folds — a conventional, slightly-optimistic bootstrap.
    (None,None) if degenerate."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(y) != X.shape[0]:
        return (None, None)
    n = X.shape[0]
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        r = wo_cv_r2(X[idx], y[idx], folds=folds, ridge=ridge)
        if r is not None:
            vals.append(r)
    if len(vals) < max(10, int(0.5 * int(n_boot))):
        return (None, None)
    lo = float(np.percentile(vals, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def wo_run_meta(model_tag=None, model_revision=None, tl_version=None, torch_version=None,
                transformers_version=None, prepend_bos=None, band=None, n=None, seed=None,
                pairs_sha=None, extra=None):
    """A reproducibility run-record (seed, band, N, model revision, library versions,
    parse/answer-extraction rule) to emit as run_meta.json beside each result CSV — the
    camera-ready repro record. Pure: the GPU cell passes the live versions/revision;
    band/N/seed default to the canonical WO constants."""
    meta = {"band": list(band) if band is not None else list(WO_BAND),
            "N": int(n) if n is not None else WO_N,
            "seed": int(seed) if seed is not None else WO_SEED,
            "pairs_sha": pairs_sha, "model_tag": model_tag, "model_revision": model_revision,
            "parse_rule": "wo_parse_int: first signed integer; strips commas; tolerates leading "
                          "space / multi-token; parse-fail counts as incorrect",
            "answer_extraction": "greedy decode K=WO_MAX_NEW_TOKENS; exact_acc = parsed == ground truth",
            "transformer_lens": tl_version, "torch": torch_version,
            "transformers": transformers_version, "prepend_bos": prepend_bos}
    if extra:
        meta.update(extra)
    return meta


def wo_decision_record(table, rule, label=None,
                       label_caveat="heuristic summary; the table + rule are the source of truth"):
    """Package a verdict as RAW TABLE + the EXPLICIT decision rule it applied, instead
    of one averaged label. `table` = list of per-condition dicts (the numbers + their
    CIs); `rule` = the threshold rule string actually applied. The one-word `label` is
    kept but DEMOTED to a clearly-marked heuristic. This is the fix for the verdict-
    compression failure (wrapper mislabel, false Gemma 'REPLICATES'). Pure."""
    return {"label_heuristic": label, "label_caveat": label_caveat,
            "decision_rule": str(rule), "table": list(table)}


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


def wo_steer_flip_verdict(zero_abl_moves, swap_flip, layer_k_flips, flip_thr=0.5):
    """FLIP-RATE version of the WO#5.1 ladder (pure; tested), for cell 82o. The
    WO#5.1 run showed the absolute first-token LOGIT is compressed: a full-residual
    swap FLIPPED the answer to the donor product on 100% of items yet moved the donor
    logit by ~0 (many leading-digit-chunk tokens share a similar logit). So re-score
    the ladder with FLIP-RATE-to-target instead:
      zero_abl_moves : bool — did zeroing the C4 '=' residual change the output at all
                       (|Δlogit| past the floor OR any argmax change); the hook must fire.
      swap_flip      : flip-rate-to-donor for the full-residual swap (guaranteed-causal
                       positive control; must be high).
      layer_k_flips  : list of (layer, k, inject_flip_rate) for the scaled probe-
                       direction counterfactual across layers and magnitudes.
    Verdict:
      INSTRUMENT_BROKEN      — zeroing does nothing (hook dead).
      METRIC_OR_SITE_SUSPECT — zero works but the full SWAP does not flip the answer.
      CALIBRATED             — some (layer,k) makes the probe-direction inject flip
                               >= flip_thr (records the SMALLEST-k winner): WO#5 was
                               under-powered/under-metriced; re-run the C1 test there.
      DEAD_DIRECTION         — the swap flips (the site IS causal) but NO (layer,k)
                               makes the inject flip: the operand-aligned probe
                               direction is not a causal handle (decoding != causal).
    Returns {label, layer_star, k_star, swap_flip, best_inject_flip, reason}."""
    out = {"swap_flip": swap_flip, "flip_thr": flip_thr, "layer_star": None, "k_star": None,
           "best_inject_flip": None}
    crossing = [(L, k, f) for (L, k, f) in layer_k_flips if f is not None and f >= flip_thr]
    if layer_k_flips:
        out["best_inject_flip"] = max((f for (_, _, f) in layer_k_flips if f is not None), default=None)
    if not zero_abl_moves:
        out["label"] = "INSTRUMENT_BROKEN"
        out["reason"] = ("Zeroing the whole C4 '=' residual did not change the output at all — the "
                         "hook/site/token convention is dead; fix it before any causal claim.")
        return out
    if swap_flip is None or swap_flip < flip_thr:
        out["label"] = "METRIC_OR_SITE_SUSPECT"
        out["reason"] = (f"Zero-ablation fires but the full donor SWAP only flips the answer at rate "
                         f"{_wo_fmt(swap_flip)} (< {flip_thr}): the guaranteed-causal control does not "
                         "control the output — investigate the donor/site before the inject sweep.")
        return out
    if crossing:
        # smallest k wins; tie-break on the highest flip-rate at that k.
        kmin = min(k for (_, k, _) in crossing)
        best = max((c for c in crossing if c[1] == kmin), key=lambda c: c[2])
        out["layer_star"], out["k_star"] = int(best[0]), best[1]
        out["label"] = "CALIBRATED"
        out["reason"] = (f"The full swap flips the answer (rate {_wo_fmt(swap_flip)}), AND the scaled "
                         f"probe-direction inject flips it at rate {_wo_fmt(best[2])} at layer "
                         f"{best[0]}, k={best[1]}: WO#5's null was a metric/magnitude artifact, not "
                         f"'product unused'. Re-run the C1 steering test at (layer {best[0]}, k={best[1]}).")
    else:
        out["label"] = "DEAD_DIRECTION"
        out["reason"] = (f"The full swap flips the answer (rate {_wo_fmt(swap_flip)}) so the C4 '=' site "
                         f"IS causal, but NO tested (layer,k) makes the probe-direction inject flip "
                         f">= {flip_thr} (best {_wo_fmt(out['best_inject_flip'])}). The operand-aligned "
                         "probe direction is genuinely not a causal handle (decoding != causal direction); "
                         "WO#5's C1 null is a wrong-axis artifact — use DAS/gradient-fit directions or a "
                         "centered operand band for a real product-representation test.")
    return out


# ----------------------------------------------------------------------------
# 8b) WORK ORDER #6 — operand-route localization + dormant-subspace certification.
# ----------------------------------------------------------------------------
# Pure substrate for WO#6 (the GPU cells 82r/82s are thin orchestration over
# these). Two arcs:
#   EXP A (positive half) — localize the operand->answer computation with
#     Symmetric-Token-Replacement (STR) counterfactuals (Zhang & Nanda 2309.16042;
#     NOT Gaussian noising), a teacher-forced multi-token logprob-DIFFERENCE metric
#     (Heimersheim & Nanda 2404.15255; NOT a first-token logit/probability), cheap
#     gradient attribution patching (Syed et al.) verified against exact patching.
#   EXP B (rigorous negative half) — Makelov decompose-and-compare (2311.17030):
#     split the linearly-decodable product direction at the '=' site into its
#     logit-affecting vs logit-inert components; if the decodable signal lives in
#     the logit-inert subspace it is a DORMANT subspace (decodable but causally
#     disconnected), the interpretability-illusion certification.
# Every function here is numpy/stdlib-only and unit-tested in tests/test_wo_logic.py
# BEFORE any A100 time is spent.
# ----------------------------------------------------------------------------

def wo_corrupt_operand(x, rng, lo=20, hi=49):
    """A different operand x' with the SAME digit count as x (so the corrupt
    surface is token-length-matched to the clean surface and every downstream
    position stays aligned for patching). Generalizes cell-82's _wo_corrupt_C /
    _wo_corrupt_B to either operand. `rng` is a np.random.Generator (deterministic
    given its seed). Returns an int x' (x' != x, len(str(x'))==len(str(x))) or None
    if 64 draws fail to find a digit-count match in [lo, hi]."""
    nd = len(str(int(x)))
    for _ in range(64):
        xp = int(rng.integers(lo, hi + 1))
        if xp != int(x) and len(str(xp)) == nd:
            return xp
    return None


def wo_build_str_counterfactual(B, C, which, rng, lo=20, hi=49):
    """Build ONE Symmetric-Token-Replacement counterfactual for the C1 surface
    '( 0 + B ) * C ='. `which` in {'C','B'} picks the operand to flip to a
    digit-count-matched alternative (C->C' or B->B'); the OTHER operand is left
    intact, so the clean and corrupt surfaces differ in exactly one operand's
    digits and are token-aligned (STR, not noising). Returns a dict:
        which, B, C            — clean operands
        Bp, Cp                 — operands after the flip (one == clean, one flipped)
        clean_answer  = B*C    — the product the CLEAN surface should produce
        corrupt_answer= Bp*Cp  — the product the CORRUPT surface should produce
        digit_aligned          — True iff the flipped operand kept its digit count
    The clean/corrupt ANSWERS are the two teacher-forced targets the logprob-
    difference metric contrasts (wo_patch_metric). Returns None if the flip fails
    (no digit-count match found) or `which` is invalid."""
    which = str(which).upper()
    if which not in ("B", "C"):
        return None
    B, C = int(B), int(C)
    if which == "C":
        Cp = wo_corrupt_operand(C, rng, lo=lo, hi=hi)
        if Cp is None:
            return None
        Bp = B
    else:
        Bp = wo_corrupt_operand(B, rng, lo=lo, hi=hi)
        if Bp is None:
            return None
        Cp = C
    flipped_clean, flipped_corr = (C, Cp) if which == "C" else (B, Bp)
    return {
        "which": which, "B": B, "C": C, "Bp": int(Bp), "Cp": int(Cp),
        "clean_answer": int(B * C), "corrupt_answer": int(Bp * Cp),
        "digit_aligned": bool(len(str(flipped_clean)) == len(str(flipped_corr))),
    }


def wo_locate_operand_spans(token_strs, B=None, C=None):
    """Locate the B- and C-operand digit-token spans (plus the structural sites)
    in a tokenized C1 surface '( 0 + B ) * C ='. Companion to wo_locate_c1_sites,
    but returns BOTH operands' full token spans (attribution patching reads/writes
    the whole operand, not a single representative index). Walk:
        plus    — first '+'
        rparen  — LAST ')'                        (reuse wo_last_rparen_index)
        star    — first '*' at/after rparen
        equals  — LAST '='
        b_span  — contiguous digit tokens strictly between '+' and ')'
        c_span  — contiguous digit tokens strictly between '*' and '='
    Returns a dict with plus/rparen/star/equals/b_span/c_span/roles/ok. If B,C are
    given, roles verify the recovered digit strings match; `ok` requires both spans
    non-empty, all sites found, and (when B,C given) the roles verify. A caller MUST
    check 'ok' before probing (a different tokenizer/format breaks the walk)."""
    strs = [(t.strip() if isinstance(t, str) else t) for t in token_strs]
    n = len(strs)
    out = {"plus": None, "rparen": None, "star": None, "equals": None,
           "b_span": [], "c_span": [], "roles": {}, "ok": False}
    plus = wo_first_index_after(strs, "+", 0)
    rparen = wo_last_rparen_index(strs)
    out["plus"], out["rparen"] = plus, rparen
    if plus is None or rparen is None or rparen <= plus:
        return out
    star = wo_first_index_after(strs, "*", rparen)
    eq = wo_last_index(strs, "=")
    out["star"], out["equals"] = star, eq
    if star is None or eq is None or eq <= star:
        return out

    def _digit_span(start_excl, end_excl):
        span, digits = [], ""
        for i in range(start_excl + 1, end_excl):
            s = strs[i]
            if s == "":
                if span:
                    break
                continue
            if isinstance(s, str) and s.isdigit():
                span.append(i)
                digits += s
            elif span:
                break
        return span, digits

    b_span, b_dig = _digit_span(plus, rparen)
    c_span, c_dig = _digit_span(star, eq)
    out["b_span"], out["c_span"] = b_span, c_span
    roles = {
        "has_plus": True, "has_rparen": True, "has_star": True, "has_equals": True,
        "B_matches": (None if B is None else bool(b_dig == str(int(B)))),
        "C_matches": (None if C is None else bool(c_dig == str(int(C)))),
    }
    out["roles"] = roles
    structural_ok = bool(b_span and c_span)
    if B is not None and C is not None:
        structural_ok = structural_ok and roles["B_matches"] and roles["C_matches"]
    out["ok"] = bool(structural_ok)
    return out


def wo_teacher_forced_logprob(logprob_rows, answer_ids, start):
    """Teacher-forced sum log-prob of an answer token sequence appended at index
    `start` of a full prompt+answer sequence. `logprob_rows` is a [seq, vocab]
    log-softmax matrix (the GPU cell applies log_softmax on device and hands the
    rows to this pure summer). The token predicting answer position t lives at row
    `start + t - 1` (causal LM), so D = sum_t logprob_rows[start + t - 1, ans[t]].
    BYTE-FOR-BYTE the indexing of cell-82p's _fp_logprob, isolated here so the
    multi-token metric is unit-testable without a model. Robust to a SHARED first
    answer token (it sums the WHOLE answer, never reads just t=0). Returns a float,
    or None if start < 1, answer is empty, or an index is out of range."""
    lp = np.asarray(logprob_rows, dtype=float)
    ids = [int(a) for a in answer_ids]
    if lp.ndim != 2 or start < 1 or len(ids) == 0:
        return None
    seq, vocab = lp.shape
    total = 0.0
    for t, a in enumerate(ids):
        r = start + t - 1
        if r < 0 or r >= seq or a < 0 or a >= vocab:
            return None
        total += float(lp[r, a])
    return float(total)


def wo_patch_metric(lp_clean_answer, lp_corrupt_answer):
    """The WO#6 headline metric D = logprob(clean_answer) - logprob(corrupt_answer),
    each a teacher-forced sum over its FULL multi-token answer (wo_teacher_forced_
    logprob). A logprob-DIFFERENCE against a contrast (best practice; NOT a bare
    logprob, NOT a probability, NOT a first-token logit). On the CLEAN surface the
    clean answer B*C is the continuation so D_clean >> 0; on the CORRUPT surface the
    corrupt answer Bp*Cp is the continuation so D_corrupt << 0. Pure scalar diff,
    kept as a named metric so the GPU cell and the tests score identically."""
    return float(lp_clean_answer) - float(lp_corrupt_answer)


def wo_denoise_recovery(d_patched, d_corrupt, d_clean, eps=1e-9):
    """Denoising recovery fraction = (D_patched - D_corrupt)/(D_clean - D_corrupt):
    0 at the corrupt baseline, 1 when the patch fully restores the clean metric.
    For DENOISING we patch CLEAN activations into the CORRUPT run (sufficiency: does
    restoring this site recover the clean answer?). The eps guards a degenerate
    (D_clean == D_corrupt) contrast. (Same formula as wo_recovery, named for the
    denoising direction so the call site reads unambiguously.)"""
    denom = (float(d_clean) - float(d_corrupt)) + eps
    return (float(d_patched) - float(d_corrupt)) / denom


def wo_attribution_score(a_clean, a_corrupt, grad):
    """First-order attribution-patching estimate (Syed et al.) of the effect on the
    metric D of replacing the CORRUPT activation with the CLEAN one at a site:
        attribution = sum( (a_clean - a_corrupt) * grad )
    where `grad` = dD/d(activation) evaluated on the CORRUPT run (one backward of D).
    Sums over ALL element axes, so the caller passes the matching slice for the site
    being scored — a (d_model,) residual at one (layer,position), a (d_head,) or
    (n_head,d_head) head slice at hook_z, or a (d_model,) mlp_out. Returns a float
    (the linear estimate of ΔD, in the SAME units as D); 0.0 on a shape mismatch."""
    ac = np.asarray(a_clean, dtype=float)
    aq = np.asarray(a_corrupt, dtype=float)
    g = np.asarray(grad, dtype=float)
    if ac.shape != aq.shape or ac.shape != g.shape:
        return 0.0
    return float(np.sum((ac - aq) * g))


def _wo_rank(vals):
    """Average ranks of `vals` (ties share the mean rank). Pure numpy helper for
    Spearman correlation."""
    a = np.asarray(vals, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # resolve ties to the average rank (stable, deterministic)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def wo_attribution_exact_agreement(attrib, exact, top_k=5):
    """How well the cheap attribution-patching estimate agrees with EXACT activation
    patching over a set of aligned sites — the Syed-et-al. verification step (a
    first-order estimate must be checked against the real intervention). `attrib`
    and `exact` are aligned per-site arrays. Returns a dict:
        n, pearson         — linear agreement of magnitudes
        spearman           — rank agreement (robust to attribution's known scaling bias)
        sign_agreement     — fraction of sites where sign(attrib)==sign(exact)
        top_k, top_k_overlap — |top-k by |attrib| ∩ top-k by |exact|| / k
                               (does attribution surface the SAME causal sites?)
    None-valued fields where a quantity is undefined (n<3 / zero variance / n<k)."""
    a = np.asarray(attrib, dtype=float).ravel()
    e = np.asarray(exact, dtype=float).ravel()
    n = a.size
    out = {"n": int(n), "pearson": None, "spearman": None,
           "sign_agreement": None, "top_k": int(top_k), "top_k_overlap": None}
    if n == 0 or e.size != n:
        return out
    out["sign_agreement"] = float(np.mean(np.sign(a) == np.sign(e)))
    out["pearson"] = wo_pearson(a.tolist(), e.tolist())
    if n >= 3 and np.std(a) > 0 and np.std(e) > 0:
        out["spearman"] = wo_pearson(_wo_rank(a).tolist(), _wo_rank(e).tolist())
    k = int(top_k)
    if 0 < k <= n:
        ta = set(np.argsort(-np.abs(a), kind="mergesort")[:k].tolist())
        te = set(np.argsort(-np.abs(e), kind="mergesort")[:k].tolist())
        out["top_k_overlap"] = float(len(ta & te) / k)
    return out


def wo_topk_sites(values, k, by_abs=True):
    """Indices of the top-k sites by value (|value| if by_abs), descending. Pure
    selector the GPU cell uses to pick which attribution-ranked sites to verify with
    exact patching. Deterministic (stable sort). Returns at most k indices."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return []
    key = -np.abs(v) if by_abs else -v
    order = np.argsort(key, kind="mergesort")
    return [int(i) for i in order[: max(0, int(k))]]


def _wo_orthonormal_basis(basis, tol=1e-9):
    """Orthonormal columns spanning the column space of `basis` (d, r), dropping
    near-zero singular directions. Returns Q (d, r') or None if degenerate/empty.

    The rank threshold is dtype-aware (numpy.linalg.matrix_rank's convention):
    s_i is kept iff s_i > s_0 * max(s, tol*) where tol* = max(d, r) * eps_float32.
    The WO#6 readout basis originates as float32 (the unembedding columns), so a
    rank-deficient basis (e.g. answer-token columns lying in a low-dim subspace)
    carries ~1e-7 relative numerical noise; the float32-eps floor drops those noise
    directions so the logit-affecting subspace is the TRUE column space, not the
    whole ambient space (a too-tight tol would let noise absorb the inert subspace
    and spuriously inflate R2_row -> a false LOGIT_COUPLED)."""
    M = np.asarray(basis, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    if M.ndim != 2 or M.size == 0 or M.shape[1] == 0:
        return None
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return None
    thresh = s[0] * max(float(tol), max(M.shape) * float(np.finfo(np.float32).eps))
    keep = s > thresh
    if not np.any(keep):
        return None
    return U[:, keep]


def wo_readout_decompose(w, readout_basis, tol=1e-9):
    """Makelov decompose-and-compare (2311.17030), the vector half: split a decode
    direction `w` (d,) into the LOGIT-AFFECTING component v_row (its projection onto
    the readout subspace spanned by `readout_basis`, e.g. the answer-token
    unembedding columns) and the LOGIT-INERT component v_null (the orthogonal
    remainder, which the unembedding/late path cannot read). Returns a dict:
        row_share   = ||v_row||^2 / ||w||^2     (share of w the logits CAN read)
        inert_share = ||v_null||^2 / ||w||^2    (share that is causally disconnected)
        r_dim       = dim of the readout subspace actually used
        v_row, v_null (lists)
    A decode direction that is dominantly INERT (inert_share ~ 1) is decodable but
    logit-disconnected — the dormant-subspace signature. Returns None on a zero `w`
    or a degenerate basis."""
    wv = np.asarray(w, dtype=float).ravel()
    wn2 = float(wv @ wv)
    if wn2 <= 0.0:
        return None
    Q = _wo_orthonormal_basis(readout_basis, tol=tol)
    if Q is None or Q.shape[0] != wv.shape[0]:
        return None
    v_row = Q @ (Q.T @ wv)
    v_null = wv - v_row
    row2 = float(v_row @ v_row)
    null2 = float(v_null @ v_null)
    return {
        "row_share": row2 / wn2,
        "inert_share": null2 / wn2,
        "r_dim": int(Q.shape[1]),
        "v_row": v_row.tolist(),
        "v_null": v_null.tolist(),
    }


def wo_readout_decode_split(X, y, readout_basis, folds=5, ridge=1.0, tol=1e-9):
    """Makelov decompose-and-compare, the decodability half: how much of the linear
    decode of `y` (the product B*C) from residuals `X` (n,d) survives when the
    features are restricted to the LOGIT-AFFECTING subspace vs the LOGIT-INERT
    complement. Project X onto the readout subspace (X_row = X Q, an (n,r) feature
    set the unembedding can read) and onto its orthogonal complement (X_null =
    X - (X Q) Q^T, rank d-r, logit-inert), and score CV-R^2 (wo_cv_r2) of each plus
    full X. Returns a dict:
        R2_full  — decode-R^2 from the whole residual           (the headline ~0.96)
        R2_row   — decode-R^2 from the logit-affecting subspace
        R2_null  — decode-R^2 from the logit-inert complement
        r_dim    — readout subspace dimension
    If R2_null ~ R2_full >> R2_row, the decodable product lives in the dormant
    (logit-inert) subspace — decodable but causally disconnected from the weights.
    None-valued R^2 fields where wo_cv_r2 is degenerate; None if the basis is bad."""
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim != 2:
        return None
    Q = _wo_orthonormal_basis(readout_basis, tol=tol)
    if Q is None or Q.shape[0] != Xm.shape[1]:
        return None
    Xrow = Xm @ Q                       # (n, r) coords in the logit-affecting subspace
    Xnull = Xm - (Xm @ Q) @ Q.T         # (n, d) residual in the logit-inert complement
    return {
        "R2_full": wo_cv_r2(Xm, y, folds=folds, ridge=ridge),
        "R2_row": wo_cv_r2(Xrow, y, folds=folds, ridge=ridge),
        "R2_null": wo_cv_r2(Xnull, y, folds=folds, ridge=ridge),
        "r_dim": int(Q.shape[1]),
    }


def wo_jsonsafe(obj):
    """Recursively coerce a payload to STRICT-JSON-safe values: NaN/Inf -> None,
    numpy scalars -> python scalars, numpy arrays/tuples -> lists. json.dumps emits
    a bare `NaN` token for float('nan') (rejected by strict parsers); the WO#6
    deliverables are consumed downstream, so we sanitize before writing. Pure."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [wo_jsonsafe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): wo_jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [wo_jsonsafe(x) for x in obj]
    return obj


def wo_dormant_verdict(inert_share, r2_row, r2_null, r2_full,
                       inert_thr=0.5, recover_frac=0.7, row_margin=0.2,
                       r2_full_min=0.3, n=None, min_n=10):
    """Certify whether the linearly-decodable product at the '=' site is a DORMANT
    subspace (Makelov interpretability-illusion), combining the two decompose-and-
    compare reads. Fires:
      DORMANT_CERTIFIED — the decode direction is dominantly logit-inert
        (inert_share >= inert_thr) AND the logit-inert complement recovers most of
        the decode (R2_null >= recover_frac * R2_full) AND the logit-affecting
        subspace carries little of it (R2_row <= R2_full - row_margin). Decodable
        but causally disconnected: the dormant-subspace certification.
      LOGIT_COUPLED — the decode direction is mostly in the readout row space
        (inert_share < inert_thr) and the logit-affecting subspace carries the
        decode (R2_row >= R2_full - row_margin): the representation IS readable by
        the logits, so NOT dormant on this evidence.
      INCONCLUSIVE — neither pattern is clean (e.g. signal split across both
        subspaces, or an R^2 is undefined).
    Returns {label, reason, inert_share, R2_row, R2_null, R2_full}."""
    out = {"label": "INCONCLUSIVE", "reason": "", "inert_share": _wo_num(inert_share),
           "R2_row": _wo_num(r2_row), "R2_null": _wo_num(r2_null), "R2_full": _wo_num(r2_full),
           "n": (None if n is None else int(n))}
    if inert_share is None or r2_full is None or r2_null is None or r2_row is None:
        out["reason"] = "a required quantity is undefined (inert_share / R^2 None)."
        return out
    if n is not None and int(n) < int(min_n):
        out["reason"] = (f"under-powered (n={int(n)} < {int(min_n)}): the held-out decode-R^2 "
                         "estimates are too variable to support a dormant claim.")
        return out
    if float(r2_full) < float(r2_full_min):
        out["reason"] = (f"decodability too weak (R^2_full={_wo_fmt(r2_full)} < {r2_full_min:g}): "
                         "the product is poorly represented everywhere, so 'dormant' is not "
                         "meaningful — neither subspace carries a strong signal.")
        return out
    inert_share = float(inert_share); r2_row = float(r2_row)
    r2_null = float(r2_null); r2_full = float(r2_full)
    null_recovers = (r2_full > 0) and (r2_null >= recover_frac * r2_full)
    row_weak = r2_row <= (r2_full - row_margin)
    row_strong = r2_row >= (r2_full - row_margin)
    if inert_share >= inert_thr and null_recovers and row_weak:
        out["label"] = "DORMANT_CERTIFIED"
        out["reason"] = (
            f"the product decode direction is {_wo_fmt(inert_share)} logit-inert and the "
            f"logit-inert complement recovers R^2={_wo_fmt(r2_null)} (>= {recover_frac:g}*"
            f"{_wo_fmt(r2_full)}) while the logit-affecting subspace recovers only "
            f"R^2={_wo_fmt(r2_row)} — decodable but causally disconnected from the weights "
            "(Makelov dormant subspace).")
    elif inert_share < inert_thr and row_strong:
        out["label"] = "LOGIT_COUPLED"
        out["reason"] = (
            f"the decode direction is {_wo_fmt(1.0 - inert_share)} inside the readout row "
            f"space and that subspace carries the decode (R^2_row={_wo_fmt(r2_row)} ~ "
            f"R^2_full={_wo_fmt(r2_full)}): the product is readable by the logits, not dormant.")
    else:
        out["reason"] = (
            f"mixed: inert_share={_wo_fmt(inert_share)}, R^2_row={_wo_fmt(r2_row)}, "
            f"R^2_null={_wo_fmt(r2_null)}, R^2_full={_wo_fmt(r2_full)} — the decode is not "
            "cleanly carried by either subspace; no certification.")
    return out


def _wo_num(x):
    """float(x) or None (json-safe numeric coercion for verdict payloads)."""
    return None if x is None else float(x)


def wo_localization_verdict(operand_pos_recovery, best_head_recovery, n_heads_for_half,
                            recover_thr=0.4, sparse_max=8):
    """Classify WHERE the operand->answer computation lives, from EXACT denoising-
    patch recoveries (fraction of the clean metric D restored by patching a CLEAN
    site into the CORRUPT run). The honest fork the paper turns on:
      LOCALIZED_OPERAND_ROUTE — restoring the flipped operand's position recovers
        the clean answer (operand_pos_recovery >= recover_thr) AND the operand->last-
        token movement runs through a SPARSE set of heads (best_head_recovery >=
        recover_thr, or only n_heads_for_half <= sparse_max heads are needed to
        restore half of D): the Stolfo et al. operand->answer route exists.
      DISTRIBUTED_NO_LOCUS — the operand position matters, but NO sparse head set
        reaches threshold (best_head_recovery < recover_thr AND n_heads_for_half >
        sparse_max or never reached): the answer is recomputed by a diffuse bag of
        heuristics (Nikankin) with no single causal locus — a publishable negative.
      INCONCLUSIVE — even the flipped operand's own position fails to recover
        (operand_pos_recovery < recover_thr): an instrument/site problem, not a
        result. Returns {label, reason, ...inputs}. n_heads_for_half=None means the
        cumulative head recovery never reached half (treated as > sparse_max)."""
    nfh = (10 ** 9) if n_heads_for_half is None else int(n_heads_for_half)
    out = {"label": "INCONCLUSIVE", "reason": "",
           "operand_pos_recovery": _wo_num(operand_pos_recovery),
           "best_head_recovery": _wo_num(best_head_recovery),
           "n_heads_for_half": (None if n_heads_for_half is None else int(n_heads_for_half))}
    if operand_pos_recovery is None or best_head_recovery is None:
        out["reason"] = "a required recovery is undefined (None)."
        return out
    opr = float(operand_pos_recovery); bhr = float(best_head_recovery)
    if opr < recover_thr:
        out["reason"] = (f"the flipped operand's own position recovers only "
                         f"{_wo_fmt(opr)} (< {recover_thr:g}) of the clean metric — the "
                         "patch/metric/site is not validated; not a localization result.")
        return out
    sparse_heads = (bhr >= recover_thr) or (nfh <= sparse_max)
    if sparse_heads:
        out["label"] = "LOCALIZED_OPERAND_ROUTE"
        out["reason"] = (f"restoring the flipped operand position recovers {_wo_fmt(opr)} of D "
                         f"and a sparse head set carries the operand->last-token movement "
                         f"(best head recovery {_wo_fmt(bhr)}, {out['n_heads_for_half']} heads "
                         f"for half D) — the Stolfo operand->answer route.")
    else:
        out["label"] = "DISTRIBUTED_NO_LOCUS"
        out["reason"] = (f"the operand position matters (recovery {_wo_fmt(opr)}) but no sparse "
                         f"head set recovers the answer (best head {_wo_fmt(bhr)} < {recover_thr:g}, "
                         f">{sparse_max} heads needed for half D) — the product is recomputed by a "
                         "diffuse bag of heuristics (Nikankin), no single causal locus.")
    return out


# ============================================================================
# 8g) WORK ORDER #7 — VACUOUS-WRAPPER BLIND-SPOT MAP (pure logic; behavioral).
#     (Distinct from WO#6 operand-localization; this is a cheap behavioral map of
#      WHICH semantically-null syntax breaks WHICH operation — the paper's hook.)
# ----------------------------------------------------------------------------
# The striking, under-exploited fact: '( 0 + B )' is the ADDITIVE IDENTITY — it
# provably does not change B's value — yet it selectively breaks multiplication and
# not addition. This maps identity-preserving rewrites W(B)==B crossed with the outer
# operation (* C / + C) to pin down WHICH property triggers the blind spot:
#   parens?  additive-identity?  inner-additive-under-outer-multiplicative mismatch?
#   nesting depth?  Every surface's ground truth is just B*C (mul) or B+C (add),
# because every wrapper is a no-op, so any accuracy drop is a pure syntactic artifact.
# All forward-pass-FREE; the GPU cell is thin orchestration over wo_run_battery.
# ----------------------------------------------------------------------------
WO_WRAPPERS = [
    ("bare",  "B",              lambda B: f"{B}"),
    ("paren", "( B )",          lambda B: f"( {B} )"),
    ("add0L", "( 0 + B )",      lambda B: f"( 0 + {B} )"),     # == C1's wrapper (additive identity)
    ("add0R", "( B + 0 )",      lambda B: f"( {B} + 0 )"),
    ("sub0",  "( B - 0 )",      lambda B: f"( {B} - 0 )"),
    ("mul1L", "( 1 * B )",      lambda B: f"( 1 * {B} )"),     # multiplicative identity (inner '*')
    ("mul1R", "( B * 1 )",      lambda B: f"( {B} * 1 )"),
    ("nest2", "(( 0 + B ))",    lambda B: f"( ( 0 + {B} ) )"),  # == D1 (depth-2 additive identity)
    ("nest3", "((( 0 + B )))",  lambda B: f"( ( ( 0 + {B} ) ) )"),
]
WO_WRAP_OPS = [("mul", "*", lambda B, C: int(B) * int(C)),
               ("add", "+", lambda B, C: int(B) + int(C))]
# property groups (for the driver classifier).
WO_WRAP_ADDITIVE = ["add0L", "add0R", "sub0"]   # inner additive identity (inner '+'/'-')
WO_WRAP_MULTID = ["mul1L", "mul1R"]             # inner multiplicative identity (inner '*')
WO_WRAP_DEPTH = ["add0L", "nest2", "nest3"]     # additive identity at nesting depth 1/2/3


def wo_build_wrapper_conditions():
    """Identity-preserving wrappers W(B)==B crossed with outer op (* C / + C), as
    (key, name, render, gt) tuples for wo_run_battery. gt is always B*C (mul) / B+C
    (add) since every wrapper is a no-op. Surfaces reuse the exact battery spacing
    (W_add0L_mul == C1, W_bare_mul == C0, W_nest2_mul == D1) so this ties back to the
    main battery. Default-arg binding avoids the loop late-binding gotcha (§6)."""
    conds = []
    for wk, wname, wfn in WO_WRAPPERS:
        for ok, osym, gt in WO_WRAP_OPS:
            def render(B, C, wfn=wfn, osym=osym):
                return f"{wfn(B)} {osym} {C} ="
            def gtf(B, C, gt=gt):
                return gt(B, C)
            conds.append((f"W_{wk}_{ok}", f"{wname} {osym} C", render, gtf))
    return conds


def wo_wrapper_verdict(acc, break_thr=0.30, keep_thr=0.15):
    """Classify the vacuous-wrapper blind spot from per-condition accuracy `acc`
    (keys 'W_<wrapper>_<op>'). Pure decision logic. Returns a dict:
        mult_blindspot   — additive-identity wrappers break MULT (mean drop >= break_thr)
        operation_specific — those same wrappers do NOT break ADD (mean drop < keep_thr)
        driver — what triggers it:
            OP_MISMATCH          : inner-additive breaks mult BUT inner-multiplicative
                                   (mul1*) and bare-parens do NOT -> the trigger is an
                                   additive sub-expression under a multiplicative outer op
            PARENS               : bare '( B ) * C' also breaks -> parentheses themselves
            ADDITIVE_IDENTITY    : additive-identity breaks mult but the mismatch/parens
                                   tests are ambiguous (e.g. mul1* also drops)
            NONE                 : no mult blind spot
        depth_sensitive — drop grows monotonically with nesting depth (add0L<nest2<nest3)
    plus the per-wrapper drops and a human reason. Drops are vs the bare baseline of
    the SAME operation, so 'drop' is the pure cost of the no-op wrapper."""
    bm, ba = acc.get("W_bare_mul"), acc.get("W_bare_add")

    def drop(w, op, base):
        v = acc.get(f"W_{w}_{op}")
        return None if (v is None or base is None) else float(base) - float(v)

    dm = {w: drop(w, "mul", bm) for w in (WO_WRAP_ADDITIVE + WO_WRAP_MULTID + ["paren", "nest2", "nest3"])}
    da = {w: drop(w, "add", ba) for w in (WO_WRAP_ADDITIVE + WO_WRAP_MULTID + ["paren", "nest2", "nest3"])}

    def _mean(d, ws):
        xs = [d[w] for w in ws if d.get(w) is not None]
        return (sum(xs) / len(xs)) if xs else None

    add_mul = _mean(dm, WO_WRAP_ADDITIVE)
    mul_mul = _mean(dm, WO_WRAP_MULTID)
    paren_mul = dm.get("paren")
    add_add = _mean(da, WO_WRAP_ADDITIVE)

    blind = add_mul is not None and add_mul >= break_thr
    op_spec = bool(blind and add_add is not None and add_add < keep_thr)
    if not blind:
        driver = "NONE"
    elif paren_mul is not None and paren_mul >= break_thr:
        driver = "PARENS"
    elif mul_mul is not None and mul_mul < keep_thr:
        driver = "OP_MISMATCH"
    else:
        driver = "ADDITIVE_IDENTITY"
    depth_ok = all(dm.get(w) is not None for w in WO_WRAP_DEPTH)
    depth_sensitive = bool(depth_ok and dm["nest3"] >= dm["nest2"] - 1e-9 >= dm["add0L"] - 2e-9
                           and dm["nest3"] > dm["add0L"] + keep_thr)

    out = {"mult_blindspot": bool(blind), "operation_specific": op_spec, "driver": driver,
           "depth_sensitive": depth_sensitive,
           "drop_mul": {w: dm[w] for w in dm}, "drop_add": {w: da[w] for w in da},
           "add_identity_mul_drop": add_mul, "mul_identity_mul_drop": mul_mul,
           "paren_mul_drop": paren_mul, "add_identity_add_drop": add_add}
    if driver == "OP_MISMATCH":
        out["reason"] = (f"A semantically-null additive wrapper breaks MULTIPLICATION "
                         f"(mean acc drop {_wo_fmt(add_mul)}) while a multiplicative-identity wrapper "
                         f"({_wo_fmt(mul_mul)}) and bare parentheses ({_wo_fmt(paren_mul)}) do NOT, and the "
                         f"SAME wrappers leave ADDITION intact ({_wo_fmt(add_add)}): the blind spot is an "
                         "inner-additive / outer-multiplicative precedence-binding conflict, not parens or "
                         "identity per se.")
    elif driver == "PARENS":
        out["reason"] = (f"Even bare parentheses '( B ) * C' break multiplication "
                         f"(drop {_wo_fmt(paren_mul)}): parenthesization itself triggers the blind spot.")
    elif driver == "ADDITIVE_IDENTITY":
        out["reason"] = (f"Additive-identity wrappers break multiplication (drop {_wo_fmt(add_mul)}) but "
                         f"the parens/mismatch contrasts are ambiguous (mul-identity drop {_wo_fmt(mul_mul)}).")
    else:
        out["reason"] = "No multiplicative blind spot: no vacuous wrapper drops mult accuracy past threshold."
    if depth_sensitive:
        out["reason"] += f" Depth-sensitive: drop grows with nesting (add0L {_wo_fmt(dm['add0L'])} -> nest3 {_wo_fmt(dm['nest3'])})."
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
    # WO#5.1 re-metric (flip-rate) verdict ladder.
    _fv = wo_steer_flip_verdict(True, 1.0, [(4, 1, 0.0), (4, 4, 0.0), (30, 4, 0.8)])
    assert _fv["label"] == "CALIBRATED" and _fv["layer_star"] == 30 and _fv["k_star"] == 4
    assert wo_steer_flip_verdict(True, 1.0, [(4, 1, 0.0), (30, 32, 0.1)])["label"] == "DEAD_DIRECTION"
    assert wo_steer_flip_verdict(True, 0.0, [(4, 1, 0.9)])["label"] == "METRIC_OR_SITE_SUSPECT"
    assert wo_steer_flip_verdict(False, 1.0, [(4, 1, 0.9)])["label"] == "INSTRUMENT_BROKEN"

    # ----- WORK ORDER #6 — operand-route localization + dormant certification -----
    _w6_rng = np.random.default_rng(0)
    # STR counterfactual: digit-count-matched flip, correct clean/corrupt answers.
    _w6_cf = wo_build_str_counterfactual(23, 47, "C", _w6_rng)
    assert _w6_cf["clean_answer"] == 23 * 47 and _w6_cf["corrupt_answer"] == 23 * _w6_cf["Cp"]
    assert _w6_cf["Cp"] != 47 and len(str(_w6_cf["Cp"])) == 2 and _w6_cf["Bp"] == 23
    assert wo_build_str_counterfactual(23, 47, "B", _w6_rng)["Cp"] == 47   # B-flip leaves C
    assert wo_build_str_counterfactual(23, 47, "Q", _w6_rng) is None       # bad operand key
    assert wo_corrupt_operand(31, np.random.default_rng(1)) != 31
    # Operand span locator on the C1 surface "( 0 + 23 ) * 47 =" (digits split).
    _w6_toks = ["(", "0", "+", "2", "3", ")", "*", "4", "7", "="]
    _w6_loc = wo_locate_operand_spans(_w6_toks, 23, 47)
    assert _w6_loc["ok"] and _w6_loc["b_span"] == [3, 4] and _w6_loc["c_span"] == [7, 8]
    assert _w6_loc["rparen"] == 5 and _w6_loc["star"] == 6 and _w6_loc["equals"] == 9
    assert not wo_locate_operand_spans(_w6_toks, 99, 47)["ok"]             # wrong B -> not ok
    # Teacher-forced multi-token logprob + the patch metric D.
    _w6_lp = np.log(np.full((6, 5), 0.2))                                  # uniform -> log 0.2 each
    _w6_lp[2, 3] = np.log(0.9); _w6_lp[3, 1] = np.log(0.8)                 # boost ans tokens
    _w6_tf = wo_teacher_forced_logprob(_w6_lp, [3, 1], start=3)            # rows 2 and 3
    assert abs(_w6_tf - (np.log(0.9) + np.log(0.8))) < 1e-9
    assert wo_teacher_forced_logprob(_w6_lp, [3], start=0) is None         # start<1 invalid
    assert wo_patch_metric(-2.0, -9.0) == 7.0
    assert abs(wo_denoise_recovery(0.0, -10.0, 10.0) - 0.5) < 1e-9
    assert wo_denoise_recovery(10.0, -10.0, 10.0) > 0.99
    # Attribution math + attribution-vs-exact agreement.
    _w6_ac = np.array([1.0, 2.0, 3.0]); _w6_aq = np.array([0.0, 0.0, 0.0])
    _w6_g = np.array([1.0, 1.0, 1.0])
    assert wo_attribution_score(_w6_ac, _w6_aq, _w6_g) == 6.0
    assert wo_attribution_score(_w6_ac, _w6_aq, np.zeros(2)) == 0.0        # shape mismatch -> 0
    _w6_attr = np.array([5.0, -3.0, 0.1, 2.0, -4.0])
    _w6_exact = np.array([4.0, -2.5, 0.2, 1.5, -3.5])                      # same ranks/signs
    _w6_ag = wo_attribution_exact_agreement(_w6_attr, _w6_exact, top_k=2)
    assert _w6_ag["sign_agreement"] == 1.0 and _w6_ag["top_k_overlap"] == 1.0
    assert _w6_ag["spearman"] is not None and _w6_ag["spearman"] > 0.99
    assert wo_topk_sites(_w6_attr, 2) == [0, 4]                            # |5| then |-4|
    # Makelov decompose: a direction inside the readout span is logit-AFFECTING;
    # orthogonal to it is logit-INERT.
    _w6_basis = np.eye(6)[:, :2]                                           # span of axes 0,1
    _w6_dec_row = wo_readout_decompose(np.array([1.0, 1.0, 0, 0, 0, 0]), _w6_basis)
    assert _w6_dec_row["inert_share"] < 1e-9 and _w6_dec_row["row_share"] > 0.999
    _w6_dec_null = wo_readout_decompose(np.array([0, 0, 0, 1.0, 1.0, 0]), _w6_basis)
    assert _w6_dec_null["inert_share"] > 0.999
    assert wo_readout_decompose(np.zeros(6), _w6_basis) is None
    # decode-split: y readable from a logit-inert axis -> R2_null carries it, R2_row ~ 0.
    _w6_n = 60
    _w6_rng2 = np.random.default_rng(2)
    _w6_y = _w6_rng2.standard_normal(_w6_n)
    _w6_X = 0.05 * _w6_rng2.standard_normal((_w6_n, 6))
    _w6_X[:, 3] += 3.0 * _w6_y                                             # y lives on INERT axis 3
    _w6_split = wo_readout_decode_split(_w6_X, _w6_y, _w6_basis)
    assert _w6_split["R2_null"] > 0.5 and (_w6_split["R2_row"] or 0.0) < 0.3
    _w6_vd = wo_dormant_verdict(0.95, _w6_split["R2_row"], _w6_split["R2_null"], _w6_split["R2_full"])
    assert _w6_vd["label"] == "DORMANT_CERTIFIED"
    assert wo_dormant_verdict(0.05, 0.95, 0.10, 0.96)["label"] == "LOGIT_COUPLED"
    assert wo_dormant_verdict(0.5, 0.5, 0.5, 0.96)["label"] == "INCONCLUSIVE"
    assert wo_dormant_verdict(0.95, 0.05, 0.20, 0.25)["label"] == "INCONCLUSIVE"        # weak R2_full
    assert wo_dormant_verdict(0.95, 0.05, 0.90, 0.95, n=5)["label"] == "INCONCLUSIVE"   # under-powered
    # json sanitizer: NaN/Inf -> None, numpy -> python.
    assert wo_jsonsafe(float("nan")) is None and wo_jsonsafe(float("inf")) is None
    assert wo_jsonsafe({"a": np.int64(3), "b": [np.float64(1.5), float("nan")]}) == {"a": 3, "b": [1.5, None]}
    assert wo_jsonsafe(np.array([1.0, np.nan])) == [1.0, None]
    # localization verdict fork.
    assert wo_localization_verdict(0.8, 0.6, 3)["label"] == "LOCALIZED_OPERAND_ROUTE"
    assert wo_localization_verdict(0.8, 0.2, 20)["label"] == "DISTRIBUTED_NO_LOCUS"
    assert wo_localization_verdict(0.8, 0.2, None)["label"] == "DISTRIBUTED_NO_LOCUS"
    assert wo_localization_verdict(0.1, 0.9, 1)["label"] == "INCONCLUSIVE"
    assert wo_localization_verdict(0.8, 0.1, 4)["label"] == "LOCALIZED_OPERAND_ROUTE"   # sparse via n_heads

    # WORK ORDER #7 — vacuous-wrapper conditions + blind-spot driver verdict.
    _w7 = dict((c[0], (c[2], c[3])) for c in wo_build_wrapper_conditions())
    assert _w7["W_add0L_mul"][0](23, 47) == "( 0 + 23 ) * 47 =" and _w7["W_add0L_mul"][1](23, 47) == 1081
    assert _w7["W_bare_mul"][0](23, 47) == "23 * 47 =" and _w7["W_bare_add"][1](23, 47) == 70
    assert _w7["W_nest2_mul"][0](23, 47) == "( ( 0 + 23 ) ) * 47 =" and _w7["W_mul1L_mul"][0](23, 47) == "( 1 * 23 ) * 47 ="
    _w7acc = {"W_bare_mul": 0.85, "W_bare_add": 0.97, "W_paren_mul": 0.82, "W_paren_add": 0.96,
              "W_add0L_mul": 0.30, "W_add0R_mul": 0.33, "W_sub0_mul": 0.31,
              "W_add0L_add": 0.95, "W_add0R_add": 0.96, "W_sub0_add": 0.95,
              "W_mul1L_mul": 0.80, "W_mul1R_mul": 0.81, "W_mul1L_add": 0.96, "W_mul1R_add": 0.96,
              "W_nest2_mul": 0.18, "W_nest3_mul": 0.07, "W_nest2_add": 0.95, "W_nest3_add": 0.94}
    _w7v = wo_wrapper_verdict(_w7acc)
    assert _w7v["mult_blindspot"] and _w7v["operation_specific"] and _w7v["driver"] == "OP_MISMATCH" \
        and _w7v["depth_sensitive"], _w7v
    _w7p = dict(_w7acc); _w7p["W_paren_mul"] = 0.30
    assert wo_wrapper_verdict(_w7p)["driver"] == "PARENS"
    _w7n = {k: (0.9 if k.endswith("mul") else 0.95) for k in _w7acc}
    assert wo_wrapper_verdict(_w7n)["driver"] == "NONE" and not wo_wrapper_verdict(_w7n)["mult_blindspot"]
    # WORK ORDER #6 Tier 2 — CI + repro hygiene helpers.
    _ac = wo_acc_ci(27 / 400, 400)
    assert abs(_ac[0] - 0.047) < 0.003 and abs(_ac[1] - 0.097) < 0.003, _ac
    _rng2 = np.random.default_rng(2); _n2, _d2 = 120, 64
    _ys = _rng2.integers(20, 50, _n2).astype(float); _Xs = _rng2.standard_normal((_n2, _d2))
    for _j in range(3):
        _Xs[:, _j] = (_ys - _ys.mean()) * 2 + 0.4 * _rng2.standard_normal(_n2)
    _r2lo, _r2hi = wo_r2_bootstrap_ci(_Xs, _ys, n_boot=40, seed=0)
    assert _r2lo is not None and _r2lo > 0.3, (_r2lo, _r2hi)        # signal -> CI well above 0
    _nlo, _nhi = wo_r2_bootstrap_ci(_rng2.standard_normal((_n2, _d2)), _rng2.standard_normal(_n2), n_boot=40, seed=0)
    assert _nhi is not None and _nhi < 0.5                          # noise -> CI not high
    _rm = wo_run_meta()
    assert _rm["band"] == list(WO_BAND) and _rm["N"] == WO_N and "parse_rule" in _rm
    _dr = wo_decision_record([{"cond": "x", "acc": 0.3}], "drop>=0.30 vs bare", label="OP_MISMATCH")
    assert _dr["label_heuristic"] == "OP_MISMATCH" and _dr["table"][0]["acc"] == 0.3 and "0.30" in _dr["decision_rule"]
    return True


_WO_SELFTEST_OK = _wo_selftest()
try:
    log(f"Phase 6 / WO logic: pure-logic self-test {'PASS' if _WO_SELFTEST_OK else 'FAIL'}.")
except NameError:
    print(f"[wo-logic] self-test {'PASS' if _WO_SELFTEST_OK else 'FAIL'} (no log() — standalone exec).")
