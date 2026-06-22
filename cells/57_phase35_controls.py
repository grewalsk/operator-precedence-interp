# ============================================================================
# Phase 3.5 — Behavioral control battery: isolate the additive-identity disruption.
# Forward-pass ONLY. NO activation patching here. Reuses Phase 3 helpers
# (_eval_prompts, parse_int -> resumable greedy decode + integer parse) and the
# checkpoint/artifact helpers. Answers (a) WHICH structural ingredient disrupts and
# (b) WHETHER the two depth conditions are equally disrupted (the differential-
# difficulty confound that decides whether precedence LOCALIZATION is valid).
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt

assert "_eval_prompts" in globals() and "parse_int" in globals(), \
    "Phase 3.5 needs Phase 3 helpers (_eval_prompts, parse_int) -- run Phase 3 first."

# ---- 0) knobs (recorded to CFG) -------------------------------------------------------
CFG.setdefault("controls_n", 400)               # >= 300 shared operand pairs
CFG.setdefault("controls_disrupt_drop", 0.20)   # acc drop from C0 that counts as 'disrupted'
CFG.setdefault("controls_similar_tol", 0.10)    # acc gap within which two conditions are 'similar'
_seed = int(CFG.get("controls_seed", CFG.get("seed", 0)))
DROP = float(CFG["controls_disrupt_drop"]); TOL = float(CFG["controls_similar_tol"])

# ---- 1) operand band: prefer G3's locked band, else derive from the curve, else 20-49 --
def _resolve_band():
    if "controls_band" in CFG:
        return tuple(CFG["controls_band"])
    if has_artifact("locked_band_spec", "json"):
        s = load_json("locked_band_spec"); return (int(s["operand_lo"]), int(s["operand_hi"]))
    if has_artifact("g3_operand_curve", "json"):                 # computing-not-lookup bins
        bins = load_json("g3_operand_curve")["bins"]
        comp = [b for b in bins if 0.30 <= b["accuracy"] < 0.90]
        if comp:
            return (int(comp[0]["lo"]), int(comp[-1]["hi"]))
    return (20, 49)
BAND = _resolve_band()
log(f"Phase 3.5: control band={BAND}, N={CFG['controls_n']}, seed={_seed}, drop={DROP}, tol={TOL}")

# ---- 2) ONE shared (B,C) pair list reused across ALL conditions (deterministic) -------
def _build_pairs():
    rng = np.random.default_rng(_seed); lo, hi = BAND; pairs, seen, tries = [], set(), 0
    while len(pairs) < int(CFG["controls_n"]) and tries < 500000:
        tries += 1
        B = int(rng.integers(lo, hi + 1)); C = int(rng.integers(lo, hi + 1))
        if B < 2 or C < 2:            continue      # exclude trivial
        if B <= 9 and C <= 9:         continue      # exclude single x single (memorized)
        if (B, C) in seen:            continue
        seen.add((B, C)); pairs.append((B, C))
    return pairs
PAIRS = _build_pairs()
GOLD_BC = [B * C for (B, C) in PAIRS]
log(f"Phase 3.5: {len(PAIRS)} shared operand pairs.")

# ---- 3) conditions (all share PAIRS). gt = ground-truth answer for exact-accuracy ------
#   C6 ground truth is B (the sub-expression), NOT B*C.
CONDITIONS = [
    ("C0", "baseline_mult",     lambda B, C: f"{B} * {C} =",         lambda B, C: B * C),
    ("C1", "depth_left",        lambda B, C: f"( 0 + {B} ) * {C} =", lambda B, C: B * C),
    ("C2", "depth_right",       lambda B, C: f"( 0 + {B} * {C} ) =", lambda B, C: B * C),
    ("C3", "parens_only_out",   lambda B, C: f"( {B} ) * {C} =",     lambda B, C: B * C),
    ("C4", "parens_only_in",    lambda B, C: f"( {B} * {C} ) =",     lambda B, C: B * C),
    ("C5", "identity_no_paren", lambda B, C: f"0 + {B} * {C} =",     lambda B, C: B * C),
    ("C6", "subexpr_alone",     lambda B, C: f"( 0 + {B} ) =",       lambda B, C: B),
    ("C7", "format_variant",    lambda B, C: f"(0+{B})*{C}=",        lambda B, C: B * C),
]

def _corr_with_bc(preds):
    xs, ys = [], []
    for p, g in zip(preds, GOLD_BC):
        if p is not None:
            xs.append(float(p)); ys.append(float(g))
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])

# ---- 4) evaluate each condition (forward passes cached/resumable via _eval_prompts) ----
RES = {}
for key, name, render, gt in CONDITIONS:
    prompts = [render(B, C) for (B, C) in PAIRS]
    golds = [gt(B, C) for (B, C) in PAIRS]
    conts = _eval_prompts(prompts, f"p35_pred_{key}")           # resumable, prompt-fingerprinted
    preds = [parse_int(c) for c in conts]
    correct = [bool(p is not None and p == g) for p, g in zip(preds, golds)]
    finite = [p for p in preds if p is not None]
    RES[key] = {
        "name": name,
        "exact_accuracy": float(np.mean(correct)) if correct else 0.0,
        "corr_with_BC": _corr_with_bc(preds),
        "parsed_rate": float(np.mean([p is not None for p in preds])) if preds else 0.0,
        "mean_abs_output": float(np.mean(np.abs(finite))) if finite else None,
        "correct_mask": correct,
    }
    log(f"  [{key}] {name}: acc={RES[key]['exact_accuracy']:.3f} "
        f"corr={RES[key]['corr_with_BC']} parsed={RES[key]['parsed_rate']:.2f}")

def _acc(k): return RES[k]["exact_accuracy"]
def _disrupted(k): return (_acc("C0") - _acc(k)) >= DROP        # acc dropped >= DROP from bare

# ---- 5) the five diagnostic questions -------------------------------------------------
q1 = _disrupted("C3") or _disrupted("C4")                       # parens alone disrupt?
q2 = _disrupted("C5")                                           # additive identity (no paren) disrupts?
if _acc("C6") >= 0.70 and (_acc("C6") - _acc("C1")) >= DROP:
    q3 = "computes (0+B)=B, then FAILS the multiply (C6 high, C1 low)"
elif _acc("C6") < 0.50:
    q3 = "FAILS inside the paren (C6 low)"
else:
    q3 = "ambiguous (C6 mid)"
q4_replicates = abs(_acc("C7") - _acc("C1")) <= TOL             # surface variant ~ C1?
q4 = ("replicates -> NOT a spacing/tokenization artifact" if q4_replicates
      else "differs -> spacing/tokenization matters (possible artifact)")

# attribute the ingredient (descriptive; the controls always yield a pattern when disruption is real)
if q1 and q2:        ingredient = "parens AND identity each disrupt independently"
elif q1:             ingredient = "parentheses (identity alone is fine)"
elif q2:             ingredient = "additive identity (parens alone are fine)"
elif _disrupted("C1") or _disrupted("C2"):
    ingredient = "the COMBINATION (neither parens nor identity alone reproduces the drop)"
else:                ingredient = "no clear disruption to attribute"

# ---- 6) DECISION GATE (precedence-localization validity) ------------------------------
acc1, acc2 = _acc("C1"), _acc("C2")
m1, m2 = RES["C1"]["correct_mask"], RES["C2"]["correct_mask"]
inter = sum(1 for a, b in zip(m1, m2) if a and b)
union = sum(1 for a, b in zip(m1, m2) if a or b)
overlap = (inter / union) if union else 0.0
small_confound = (abs(acc1 - acc2) <= 0.10) and (overlap >= 0.60)
gate_branch = ("LOCALIZATION MAY PROCEED -- on the matched correct-only intersection only, "
               "reported with the selection caveat, with a check that patch-signal magnitude "
               "is comparable across C1,C2."
               if small_confound else
               "CONFOUND LARGE -> DROP precedence localization to future work; PIVOT primary "
               "contribution to the brittleness characterization.")

# ---- 7) brittleness validity gate (the likely primary path) ---------------------------
disruption_replicates = _disrupted("C1") or _disrupted("C2")
brittleness_stands = bool(disruption_replicates and q4_replicates)

# ---- 8) report ------------------------------------------------------------------------
def _f(x, nd=3): return "  n/a" if x is None else f"{x:.{nd}f}"
L = []
L.append("================= PHASE 3.5 -- BEHAVIORAL CONTROL BATTERY =================")
L.append(f"band={BAND}  N={len(PAIRS)}  seed={_seed}  disrupt_drop={DROP}  similar_tol={TOL}")
L.append("-------------------------------------------------------------------------")
L.append(f"{'cond':<4} {'name':<18} {'acc':>6} {'corr(B*C)':>10} {'parsed':>7} {'mean|out|':>9}")
for key, name, *_ in CONDITIONS:
    r = RES[key]
    mag = "  n/a" if r["mean_abs_output"] is None else f"{r['mean_abs_output']:.0f}"
    L.append(f"{key:<4} {name:<18} {r['exact_accuracy']:>6.3f} {_f(r['corr_with_BC']):>10} "
             f"{r['parsed_rate']:>7.2f} {mag:>9}")
L.append("-------------------------------------------------------------------------")
L.append("DIAGNOSTIC VERDICTS:")
L.append(f"  Q1 parens alone disrupt?   {q1}   (C3={_acc('C3'):.2f}, C4={_acc('C4'):.2f} vs C0={_acc('C0'):.2f})")
L.append(f"  Q2 identity alone disrupt? {q2}   (C5={_acc('C5'):.2f} vs C0={_acc('C0'):.2f})")
L.append(f"  Q3 where C1 fails:         {q3}")
L.append(f"  Q4 surface artifact?       {q4}   (C7={_acc('C7'):.2f} vs C1={_acc('C1'):.2f})")
L.append(f"  -> disruption ingredient:  {ingredient}")
L.append("-------------------------------------------------------------------------")
L.append("DECISION GATE (does precedence LOCALIZATION stay valid?):")
L.append(f"  acc(C1 depth_left)={acc1:.3f}  acc(C2 depth_right)={acc2:.3f}  |delta|={abs(acc1-acc2):.3f}")
L.append(f"  correct-subset overlap (Jaccard) = {overlap:.3f}  (inter={inter}, union={union})")
L.append(f"  small confound = {small_confound}")
L.append(f"  -> {gate_branch}")
L.append("-------------------------------------------------------------------------")
L.append("BRITTLENESS GATE (likely primary path):")
L.append(f"  disruption replicates (C1 or C2 << C0): {disruption_replicates}")
L.append(f"  survives surface variant (C7 ~ C1):     {q4_replicates}")
L.append(f"  ingredient localized:                   {ingredient}")
L.append(f"  -> brittleness finding STANDS: {brittleness_stands}")
L.append("=========================================================================")
L.append("NOTE: do NOT run novel precedence patching (Phases 6-9) until the DECISION GATE")
L.append("above is read. -Instruct is a SECOND experiment, not a substitute.")
report = "\n".join(L)
save_text("controls_report", report)
print(report)

# persist the gate outcome for downstream phases
save_json("p35_decision", {
    "band": list(BAND), "n": len(PAIRS), "seed": _seed,
    "acc": {k: _acc(k) for k, *_ in CONDITIONS},
    "corr": {k: RES[k]["corr_with_BC"] for k, *_ in CONDITIONS},
    "acc_depth_left": acc1, "acc_depth_right": acc2, "overlap": overlap,
    "small_confound": bool(small_confound), "gate_branch": gate_branch,
    "disruption_replicates": disruption_replicates, "brittleness_stands": brittleness_stands,
    "q1_parens_disrupt": bool(q1), "q2_identity_disrupt": bool(q2),
    "q3_fail_location": q3, "q4_surface": q4, "ingredient": ingredient,
})

# ---- 9) optional bar chart: exact accuracy by condition, with C0 reference -------------
try:
    keys = [c[0] for c in CONDITIONS]
    labels = [f"{c[0]}\n{c[1]}" for c in CONDITIONS]
    accs = [_acc(k) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, accs, color="#4C78A8")
    bars[0].set_color("#888888")                      # C0 reference bar
    ax.axhline(_acc("C0"), ls="--", c="grey", lw=1, label=f"C0 baseline ({_acc('C0'):.2f})")
    ax.set_ylabel("exact accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Phase 3.5 control battery  (band {BAND}, N={len(PAIRS)})")
    ax.legend(fontsize=8); plt.xticks(fontsize=7); fig.tight_layout()
    fig.savefig(str(ART / "p35_controls.png"), dpi=120); plt.show()
except Exception as e:
    log(f"(bar chart skipped: {e})")
