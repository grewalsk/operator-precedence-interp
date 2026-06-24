"""CPU unit tests for the work-order pure-logic cell (cells/76_wo_logic.py).

Exec's the cell in an isolated namespace (with a stub `log`) and exercises the
six validity gates (§7), the branch tree (§8), the 2x2 verdict (§6), metrics,
and the recovery math (§10) against the work-order spec and the published base
numbers. No GPU / no model required.

Run:  python3 tests/test_wo_logic.py     (also importable by pytest)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = os.path.join(HERE, "..", "cells", "76_wo_logic.py")


def _load():
    ns = {"log": lambda *a, **k: None}
    with open(CELL) as fh:
        exec(compile(fh.read(), CELL, "exec"), ns)
    return ns


WO = _load()
_fails = []


def check(name, cond):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        _fails.append(name)


# ---------------------------------------------------------------- stimuli ----
def test_pairs_deterministic_and_band():
    p1 = WO["wo_build_pairs"]()
    p2 = WO["wo_build_pairs"]()
    check("pairs deterministic", p1 == p2)
    check("pairs count == N", len(p1) == WO["WO_N"])
    lo, hi = WO["WO_BAND"]
    check("all operands in band", all(lo <= B <= hi and lo <= C <= hi for B, C in p1))
    check("no duplicate pairs", len(set(p1)) == len(p1))
    check("stim hash stable", WO["wo_stim_hash"](p1) == WO["wo_stim_hash"](p2))


def test_condition_surfaces_exact():
    # surfaces must match work order §2/§6 EXACTLY (spaces matter; C7/C8 have none).
    rend = {c[0]: c[2] for c in WO["WO_CONDITIONS"]}
    gt = {c[0]: c[3] for c in WO["WO_CONDITIONS"]}
    B, C = 23, 47
    expected = {
        "C0": "23 * 47 =", "C1": "( 0 + 23 ) * 47 =", "C2": "( 0 + 23 * 47 ) =",
        "C3": "( 23 ) * 47 =", "C4": "( 23 * 47 ) =", "C5": "0 + 23 * 47 =",
        "C6": "( 0 + 23 ) =", "C7": "(0+23)*47=", "C8": "(23*47)=",
    }
    for k, want in expected.items():
        check(f"surface {k} exact", rend[k](B, C) == want)
    check("C6 ground truth is B (not B*C)", gt["C6"](B, C) == B)
    check("C0 ground truth is B*C", gt["C0"](B, C) == B * C)
    # 2x2 mapping wired to the right keys.
    check("2x2 spaces/inside == C4", WO["WO_2X2"][("spaces", "inside")] == "C4")
    check("2x2 nospace/inside == C8", WO["WO_2X2"][("nospace", "inside")] == "C8")
    check("2x2 nospace/outer == C7", WO["WO_2X2"][("nospace", "outer")] == "C7")


def test_branchb_surfaces():
    rend = {c[0]: c[2] for c in WO["WO_BRANCHB_CONDITIONS"]}
    gt = {c[0]: c[3] for c in WO["WO_BRANCHB_CONDITIONS"]}
    check("A1 add-compose-left surface", rend["A1"](23, 47) == "( 0 + 23 ) + 47 =")
    check("A2 add-compose-right surface", rend["A2"](23, 47) == "0 + ( 23 + 47 ) =")
    check("D1 depth-redundant surface", rend["D1"](23, 47) == "( ( 0 + 23 ) ) * 47 =")
    check("A1 gt is B+C", gt["A1"](23, 47) == 70)
    check("D1 gt is B*C (same parse as C1)", gt["D1"](23, 47) == 23 * 47)


# ---------------------------------------------------------------- metrics ----
def test_parse_int():
    p = WO["wo_parse_int"]
    check("parse leading space + comma", p(" 1,234 rest") == 1234)
    check("parse plain", p("42") == 42)
    check("parse failure -> None", p("the answer") is None)
    check("parse None -> None", p(None) is None)
    check("parse negative", p("-7x") == -7)
    check("parse trailing dash stripped", p("12-") == 12)


def test_summarize_and_jaccard():
    s = WO["wo_summarize"]
    r = s([6, None, 8, 9], [6, 7, 8, 0])
    check("acc counts parse-fail as wrong", abs(r["exact_acc"] - 0.5) < 1e-9)
    check("parse_fail_rate", abs(r["parse_fail_rate"] - 0.25) < 1e-9)
    check("n_parsed excludes None", r["n_parsed"] == 3)
    # perfectly tracking -> corr ~1
    rt = s([2, 4, 6, 8], [2, 4, 6, 8])
    check("corr 1.0 on perfect track", rt["corr"] is not None and rt["corr"] > 0.999)
    j = WO["wo_jaccard"]
    check("jaccard half", abs(j([1, 1, 0, 0], [1, 0, 0, 0]) - 0.5) < 1e-9)
    check("jaccard empty union -> 0", j([0, 0], [0, 0]) == 0.0)


def test_recovery_math():
    rec = WO["wo_recovery"]
    check("recovery 0 at corrupted", abs(rec(0.0, 0.0, 1.0) - 0.0) < 1e-6)
    check("recovery 1 at clean", abs(rec(1.0, 0.0, 1.0) - 1.0) < 1e-6)
    check("recovery 0.5 midway", abs(rec(0.5, 0.0, 1.0) - 0.5) < 1e-6)
    check("recovery works negative baselines", abs(rec(-1.0, -2.0, 0.0) - 0.5) < 1e-6)


def test_cv_r2_not_vacuous_on_noise():
    # THE regression test for the reviewer-caught bug: in-sample lstsq returns
    # R^2=1.0 on pure noise at n<<d. wo_cv_r2 (held-out + PCA) must NOT.
    import numpy as np
    cv = WO["wo_cv_r2"]
    rng = np.random.default_rng(7)
    n, d = 128, 4096
    Xnoise = rng.standard_normal((n, d))
    ynoise = rng.standard_normal(n)
    r2_noise = cv(Xnoise, ynoise)
    check("CV-R^2 on pure noise is low (<0.3), NOT ~1.0",
          r2_noise is not None and r2_noise < 0.3)
    # realistic decodability: the target (an operand value) is PROMINENTLY encoded
    # in a few directions with real amplitude (as a written-in resid feature would
    # be), with the rest noise. This is the scenario the salvage probe must recover.
    Bvals = rng.integers(20, 50, n).astype(float)
    Xsig = Xnoise.copy()
    for j in range(3):
        Xsig[:, j] = (Bvals - Bvals.mean()) * 2.0 + 0.5 * rng.standard_normal(n)
    r2_signal = cv(Xsig, Bvals)
    check("CV-R^2 on prominently-encoded signal is high (>0.6)",
          r2_signal is not None and r2_signal > 0.6)
    check("CV-R^2 separates signal from noise", r2_signal > r2_noise + 0.3)
    check("CV-R^2 None on too-few-samples", cv(rng.standard_normal((4, d)), rng.standard_normal(4)) is None)


# ---------------------------------------------------------------- gates §7 ----
def test_gates_on_published_base_numbers():
    # base RESULTS.md — the gate that the work order says must FAIL pre-Instruct.
    acc = {"C0": 0.838, "C1": 0.507, "C2": 0.710, "C3": 0.495, "C4": 0.890,
           "C5": 0.583, "C6": 1.000, "C7": 0.018}
    corr = {"C1": 0.060, "C2": 0.282}
    ge = WO["wo_evaluate_gates"](acc, corr, jaccard_c1c2=0.697)
    g = ge["gates"]
    check("G_floor fails (acc C0=0.838<0.90)", not g["G_floor"]["pass"])
    check("G_neutral fails (acc C1=0.507<0.85)", not g["G_neutral"]["pass"])
    check("G_symmetry fails (|.507-.710|=.203>.05)", not g["G_symmetry"]["pass"])
    check("G_quantity fails (corr C1=.06<.80)", not g["G_quantity"]["pass"])
    check("G_surface fails (acc C7=.018<.70)", not g["G_surface"]["pass"])
    check("G_support fails (jaccard .697<.85)", not g["G_support"]["pass"])
    check("localization INVALID on base", ge["verdict"] == "INVALID")
    check("branch NO_REPAIR on base", WO["wo_select_branch"](ge)["branch"] == "NO_REPAIR")


def test_gates_boundary_values():
    # exact-threshold boundaries must PASS (>= / <=).
    acc = {"C0": 0.90, "C1": 0.85, "C2": 0.80, "C4": 0.90, "C7": 0.70}
    #   |C1-C4| = 0.05 (<=0.05 pass); |C1-C2| = 0.05 (<=0.05 pass)
    corr = {"C1": 0.80}
    ge = WO["wo_evaluate_gates"](acc, corr, jaccard_c1c2=0.85)
    for k in ["G_floor", "G_neutral", "G_symmetry", "G_quantity", "G_surface", "G_support"]:
        check(f"{k} passes exactly at threshold", ge["gates"][k]["pass"])
    check("all-at-threshold -> VALID + CLEAN", ge["localization_valid"]
          and WO["wo_select_branch"](ge)["branch"] == "CLEAN_REPAIR")


def test_gates_neutral_compound():
    # G_neutral needs BOTH acc(C1)>=0.85 AND |C1-C4|<=0.05.
    ev = WO["wo_evaluate_gates"]
    a = {"C0": 0.95, "C1": 0.95, "C2": 0.95, "C4": 0.80, "C7": 0.8}  # |C1-C4|=0.15 -> fail
    check("G_neutral fails when |C1-C4|>0.05 even if acc high",
          not ev(a, {"C1": 0.9}, 0.9)["gates"]["G_neutral"]["pass"])
    b = {"C0": 0.95, "C1": 0.80, "C2": 0.80, "C4": 0.82, "C7": 0.8}  # acc C1=0.80<0.85 -> fail
    check("G_neutral fails when acc(C1)<0.85 even if |C1-C4| small",
          not ev(b, {"C1": 0.9}, 0.9)["gates"]["G_neutral"]["pass"])


def test_partial_repair_predicted():
    # work order's PREDICTED outcome: hard gates pass, only G_surface fails.
    acc = {"C0": 0.95, "C1": 0.88, "C2": 0.86, "C4": 0.89, "C7": 0.30}
    ge = WO["wo_evaluate_gates"](acc, {"C1": 0.85}, jaccard_c1c2=0.90)
    check("PARTIAL: localization VALID", ge["localization_valid"])
    check("PARTIAL: G_surface fails", not ge["g_surface_pass"])
    check("PARTIAL: scope conditional on spaced", "CONDITIONAL" in ge["scope"])
    br = WO["wo_select_branch"](ge)
    check("PARTIAL: branch == PARTIAL_REPAIR", br["branch"] == "PARTIAL_REPAIR")
    check("PARTIAL: run_on instruct only", br["run_on"] == ["instruct"])


def test_floor_failure_surfaced():
    # G_floor failing routes to NO_REPAIR with the capability caveat surfaced.
    acc = {"C0": 0.70, "C1": 0.9, "C2": 0.9, "C4": 0.9, "C7": 0.9}
    ge = WO["wo_evaluate_gates"](acc, {"C1": 0.9}, 0.9)
    br = WO["wo_select_branch"](ge)
    check("G_floor fail -> NO_REPAIR", br["branch"] == "NO_REPAIR")
    check("G_floor fail caveat surfaced", "G_floor" in br["rationale"])


# ---------------------------------------------------------------- 2x2 §6 ----
def test_2x2_verdict():
    v = WO["wo_2x2_verdict"]
    check("C8 survives -> compose-specific", v(0.89, 0.018, 0.82)["verdict"] == "COMPOSE_SPECIFIC")
    check("C8 collapses -> pure tokenization", v(0.89, 0.018, 0.03)["verdict"] == "PURE_TOKENIZATION")
    check("C8 mid -> ambiguous", v(0.89, 0.018, 0.40)["verdict"] == "AMBIGUOUS")


# ---------------------------------------------------------------- builders ----
def test_csv_and_record_builders():
    rows = [{"cond": "C0", "acc": 0.838, "note": "a,b"}, {"cond": "C1", "acc": None}]
    csv = WO["wo_battery_csv"](rows, ["cond", "acc", "note"])
    check("csv has header", csv.splitlines()[0] == "cond,acc,note")
    check("csv quotes comma field", '"a,b"' in csv)
    check("csv None -> empty", csv.splitlines()[2].split(",")[1] == "")
    # decision record builds without error on the base fixture.
    acc = {"C0": 0.838, "C1": 0.507, "C2": 0.710, "C4": 0.890, "C7": 0.018}
    ge = WO["wo_evaluate_gates"](acc, {"C1": 0.06}, 0.697)
    br = WO["wo_select_branch"](ge)
    summ = {c[0]: {"exact_acc": acc.get(c[0]), "corr": None, "parse_fail_rate": 0.0}
            for c in WO["WO_CONDITIONS"]}
    md = WO["wo_decision_record_md"]("instruct", ge, br, summ,
                                     WO["wo_2x2_verdict"](0.89, 0.018, 0.85),
                                     0.203, 0.203,
                                     {"transformer_lens": "2.x", "model_revision": "abc",
                                      "prepend_bos": True, "format": "bare-continuation"})
    check("decision record mentions branch", br["branch"] in md)
    check("decision record has gate table", "G_symmetry" in md and "Localization verdict" in md)


def test_operand_magnitude_bins():
    pairs = WO["wo_build_pairs"]()
    bins = WO["wo_operand_magnitude_bins"](pairs, n_bins=5)
    check("5 magnitude bins", len(bins) == 5)
    total = sum(b["n"] for b in bins)
    check("all pairs binned", total == len(pairs))


# ------------------------------------------------- WO#2 §3.4 confidence intervals
def test_bootstrap_ci():
    import numpy as np
    boot = WO["wo_bootstrap_ci"]
    mask = [1] * 30 + [0] * 70           # point estimate 0.30
    lo, hi = boot(mask, n_boot=2000, seed=0)
    check("bootstrap CI brackets the point estimate", lo <= 0.30 <= hi)
    check("bootstrap CI is a proper interval", lo < hi)
    # deterministic given seed
    check("bootstrap CI deterministic", boot(mask, n_boot=2000, seed=0) == boot(mask, n_boot=2000, seed=0))
    # width shrinks ~1/sqrt(n): same proportion (0.5), 100x more samples -> ~10x tighter.
    small = boot([1] * 25 + [0] * 25, n_boot=2000, seed=1)
    large = boot([1] * 2500 + [0] * 2500, n_boot=2000, seed=1)
    w_small, w_large = small[1] - small[0], large[1] - large[0]
    check("CI width shrinks with n", w_large < w_small)
    check("CI width shrinks ~1/sqrt(n) (n x100 -> width < 0.3x)", w_large < 0.3 * w_small)
    check("empty mask -> (None, None)", boot([]) == (None, None))


def test_paired_delta_ci():
    pd = WO["wo_paired_delta_ci"]
    m = [1, 0, 1, 1, 0, 1, 0, 0]
    lo, hi = pd(m, m, n_boot=2000, seed=0)
    check("paired delta of identical masks contains 0", lo <= 0.0 <= hi)
    check("paired delta of identical masks IS [0,0]", lo == 0.0 and hi == 0.0)
    n = 40
    a, b = [1] * n, [0] * n              # disjoint: a always right, b always wrong
    dlo, dhi = pd(a, b, n_boot=2000, seed=0)
    check("disjoint masks: delta CI excludes 0", dlo > 0.0)
    check("disjoint masks: delta ~ 1.0", abs(dlo - 1.0) < 1e-9 and abs(dhi - 1.0) < 1e-9)
    check("paired delta deterministic", pd(m, a[:8], n_boot=1000, seed=3) == pd(m, a[:8], n_boot=1000, seed=3))
    check("length mismatch -> (None, None)", pd([1, 0], [1, 0, 1]) == (None, None))


def test_wilson_ci():
    w = WO["wo_wilson_ci"]
    lo, hi = w(27, 400)
    check("wilson(27,400) lo ~ 0.047", abs(lo - 0.047) < 0.003)
    check("wilson(27,400) hi ~ 0.097", abs(hi - 0.097) < 0.003)
    check("wilson brackets phat=27/400", lo <= 27 / 400 <= hi)
    check("wilson n=0 -> (None,None)", w(0, 0) == (None, None))
    lo0, hi0 = w(0, 100)
    check("wilson k=0 lo clamped to ~0 (non-negative)", 0.0 <= lo0 < 1e-6 and hi0 > 0.0)
    lon, hin = w(100, 100)
    check("wilson k=n hi clamped to 1", hin == 1.0 and lon < 1.0)
    # out-of-domain k>n must NOT raise (clamped to n) — defensive guard.
    try:
        og = w(10, 5)
        check("wilson k>n clamped, no math-domain error", og == w(5, 5))
    except Exception as e:
        check(f"wilson k>n clamped, no math-domain error (raised {type(e).__name__})", False)


# ------------------------------------------------- WO#2 §3.6 output classifier
def test_classify_wrong_output():
    cl = WO["wo_classify_wrong_output"]
    B, C = 23, 47                        # B*C=1081, B+C=70, all distinct
    check("pred==B*C -> correct", cl(B * C, B, C) == "correct")
    check("pred==B -> equals_B", cl(B, B, C) == "equals_B")
    check("pred==C -> equals_C", cl(C, B, C) == "equals_C")
    check("pred==B+C -> equals_B_plus_C", cl(B + C, B, C) == "equals_B_plus_C")
    check("pred==None -> parse_fail", cl(None, B, C) == "parse_fail")
    check("pred unrelated -> other", cl(99999, B, C) == "other")
    # tie-breaking by the listed priority order (correct > equals_B > ...).
    check("tie B==C, pred==B -> equals_B (not equals_C)", cl(2, 2, 2) == "equals_B")
    check("tie B*C==B+C, pred==4 -> correct (not equals_B_plus_C)", cl(4, 2, 2) == "correct")


# ------------------------------------------------- WO#2 §3.5 few-shot prompt builder
def test_fewshot_render():
    import re
    fr = WO["wo_fewshot_render"]
    rend = {c[0]: c[2] for c in WO["WO_CONDITIONS"]}["C1"]
    gt = {c[0]: c[3] for c in WO["WO_CONDITIONS"]}["C1"]
    pool = WO["wo_build_pairs"]()
    test_pair = pool[0]
    # 0-shot is exactly the bare test prompt (ends at '=', no trailing space).
    p0 = fr(rend, gt, 0, test_pair, pool, seed=0)
    check("0-shot == bare test prompt", p0 == rend(*test_pair))
    check("prompt has no trailing space", not p0.endswith(" "))
    # k-shot: k worked examples + the test prompt.
    p4 = fr(rend, gt, 4, test_pair, pool, seed=0)
    lines = p4.splitlines()
    check("4-shot has 5 lines (4 shots + test)", len(lines) == 5)
    check("4-shot last line == bare test prompt", lines[-1] == rend(*test_pair))
    # every shot line is well-formed AND its answer == b*c (correct worked example).
    shot_pairs = []
    ok_fmt = True
    for ln in lines[:-1]:
        m = re.match(r"^\( 0 \+ (\d+) \) \* (\d+) = (-?\d+)$", ln)
        if m is None:
            ok_fmt = False
            break
        b, c, ans = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ok_fmt = ok_fmt and (ans == b * c)
        shot_pairs.append((b, c))
    check("shot lines well-formed with answer==b*c", ok_fmt)
    check("shots exclude the test pair", all(sp != tuple(test_pair) for sp in shot_pairs))
    check("shots are distinct from each other", len(set(shot_pairs)) == len(shot_pairs))
    # deterministic given seed.
    check("few-shot deterministic", fr(rend, gt, 4, test_pair, pool, 7) == fr(rend, gt, 4, test_pair, pool, 7))


# ------------------------------------------------- WO#2 §3.7 shuffled-target control
def test_shuffle_control_decorrelates():
    import numpy as np
    sc = WO["wo_shuffle_control"]
    cv = WO["wo_cv_r2"]
    rng = np.random.default_rng(3)
    n, d = 128, 4096
    Bvals = rng.integers(20, 50, n).astype(float)
    X = rng.standard_normal((n, d))
    for j in range(3):                   # prominently encode B in a few dims
        X[:, j] = (Bvals - Bvals.mean()) * 2.0 + 0.5 * rng.standard_normal(n)
    r2_true = cv(X, Bvals)
    r2_shuf = cv(X, sc(Bvals, seed=0))
    check("shuffled-B target decorrelates (CV-R^2 < 0.3)", r2_shuf is not None and r2_shuf < 0.3)
    check("true B decodes far better than shuffled", r2_true - r2_shuf > 0.3)
    check("shuffle actually permutes (seeded)", not np.array_equal(sc(Bvals, 0), Bvals))
    check("shuffle deterministic given seed", np.array_equal(sc(Bvals, 5), sc(Bvals, 5)))


# ------------------------------------------------- WO#3 few-shot probe pure logic
def test_fewshot_probe_site_finder():
    # Hazard #1: under a few-shot prefix the FIRST ')' belongs to a SHOT; the probe
    # site is the TEST expression's ')' = the LAST one. Verify on a REAL rendered
    # 2-shot prompt via a tokenizer-free whitespace analog.
    lri = WO["wo_last_rparen_index"]
    rend = {c[0]: c[2] for c in WO["WO_CONDITIONS"]}["C1"]
    gt = lambda b, c: b * c
    pool = WO["wo_build_pairs"]()
    test_pair = pool[0]
    B = test_pair[0]
    prompt = WO["wo_fewshot_render"](rend, gt, 2, test_pair, pool, seed=0)
    toks = [t for t in prompt.replace("\n", " \n ").split(" ") if t != ""]
    idx = lri([t.strip() for t in toks])
    n_rparen = sum(1 for t in toks if t.strip() == ")")
    check("2-shot prompt has 3 brackets (2 shots + test)", n_rparen == 3)
    check("finder returns a ')' index", idx is not None and toks[idx].strip() == ")")
    last_rparen = max(k for k, t in enumerate(toks) if t.strip() == ")")
    check("finder picks the LAST ')' (the test expr), not a shot", idx == last_rparen)
    check("last ')' closes the TEST bracket (test B precedes it)", toks[idx - 1].strip() == str(B))
    # 0-shot prompt has exactly one ')', and it is the test's.
    p0 = WO["wo_fewshot_render"](rend, gt, 0, test_pair, pool, seed=0)
    t0 = [t for t in p0.split(" ") if t != ""]
    i0 = lri([t.strip() for t in t0])
    check("0-shot: single ')' is the test's", i0 is not None and t0[i0 - 1].strip() == str(B))


def test_fsprobe_trend():
    tr = WO["wo_fsprobe_trend"]
    check("stable -> DECODABLE_IN_BOTH", tr(0.90, 0.90, 0.90)[0] == "DECODABLE_IN_BOTH")
    check("small wobble still DECODABLE_IN_BOTH", tr(0.90, 0.92, 0.88)[0] == "DECODABLE_IN_BOTH")
    check("rise -> REPRESENTATION_IMPROVES", tr(0.70, 0.85, 0.90)[0] == "REPRESENTATION_IMPROVES")
    check("drop below 0-shot -> PROBE_SITE_SUSPECT", tr(0.90, 0.70, 0.70)[0] == "PROBE_SITE_SUSPECT")
    check("low absolute -> PROBE_SITE_SUSPECT", tr(0.40, 0.25, 0.25)[0] == "PROBE_SITE_SUSPECT")
    check("missing level -> INCONCLUSIVE", tr(0.9, None, 0.9)[0] == "INCONCLUSIVE")
    check("intermediate -> MIXED", tr(0.90, 0.83, 0.90)[0] == "MIXED")


# ================================================================= WORK ORDER #4 ==
# ------------------------------------------------- §3.1 cross-model replication verdict
def test_replication_verdict():
    rv = WO["wo_replication_verdict"]
    # single-checkpoint instruct fixture (A100_run_2026-06-24): REPLICATES_FULL.
    full = rv({"C1": 0.265, "C4": 0.9075, "C6": 1.0, "A1": 0.995, "D1": 0.0375}, 0.915)
    check("instruct fixture -> replicates_full", full["replicates_full"])
    check("instruct fixture label REPLICATES_FULL", full["label"] == "REPLICATES_FULL")
    check("all sub-flags true on full fixture",
          all(full[k] for k in ("parts_work", "compose_collapses", "operation_specific",
                                "depth_sensitive", "fewshot_recovers", "replicates_core")))
    check("full fixture not out_of_scope", not full["out_of_scope"])
    # base fixture: parts work, collapses, op-specific, fewshot recovers -> full too.
    base = rv({"C1": 0.507, "C4": 0.890, "C6": 1.0, "A1": 0.90, "D1": 0.04}, 0.882)
    check("base fixture replicates_core", base["replicates_core"])
    # capability gap: can't do C4/C6 -> OUT_OF_SCOPE (not a failure-to-replicate).
    oos = rv({"C1": 0.10, "C4": 0.40, "C6": 0.50}, 0.20)
    check("low-capability model -> out_of_scope", oos["out_of_scope"])
    check("out_of_scope label", oos["label"].startswith("OUT_OF_SCOPE"))
    check("out_of_scope not counted as replicates_core", not oos["replicates_core"])
    # an HONEST non-replicator: parts work but C1 doesn't collapse.
    nr = rv({"C1": 0.88, "C4": 0.90, "C6": 0.95, "A1": 0.92, "D1": 0.10}, 0.90)
    check("non-replicator parts_work true", nr["parts_work"])
    check("non-replicator compose_collapses false", not nr["compose_collapses"])
    check("non-replicator label DOES_NOT_REPLICATE", nr["label"] == "DOES_NOT_REPLICATE")
    # core but not full: collapses, but addition does NOT compose (op-specificity absent).
    core = rv({"C1": 0.30, "C4": 0.90, "C6": 0.95, "A1": 0.35, "D1": 0.05}, 0.40)
    check("core-only: replicates_core true", core["replicates_core"])
    check("core-only: operation_specific false (A1~C1)", not core["operation_specific"])
    check("core-only: not replicates_full", not core["replicates_full"])
    check("core-only label REPLICATES_CORE", core["label"] == "REPLICATES_CORE")
    # fewshot below the C4 ceiling -> fewshot_recovers false even if it lifts C1.
    nofs = rv({"C1": 0.30, "C4": 0.90, "C6": 0.95, "A1": 0.95, "D1": 0.05}, 0.55)
    check("fewshot below ceiling -> fewshot_recovers false", not nofs["fewshot_recovers"])
    # missing a CORE input -> INCOMPLETE (not OUT_OF_SCOPE).
    inc = rv({"C4": 0.9, "C6": 0.9}, 0.9)
    check("missing C1 -> INCOMPLETE label", inc["label"].startswith("INCOMPLETE"))
    check("missing C1 -> not out_of_scope", not inc["out_of_scope"])
    check("missing list reports C1", inc["missing"] == ["C1"])
    # missing A1 only: core can still hold; full cannot.
    noA1 = rv({"C1": 0.265, "C4": 0.9075, "C6": 1.0, "D1": 0.0375}, 0.915)
    check("missing A1: replicates_core still true", noA1["replicates_core"])
    check("missing A1: operation_specific false", not noA1["operation_specific"])
    check("missing A1: replicates_full false", not noA1["replicates_full"])
    # thresholds are overridable params.
    strict = rv({"C1": 0.265, "C4": 0.9075, "C6": 1.0, "A1": 0.40, "D1": 0.0375}, 0.915,
                thr={"opspecific_gap": 0.50})
    check("threshold override tightens operation_specific", not strict["operation_specific"])


# ------------------------------------------------- §3.2 multi-position site finders
def test_position_finders_basic():
    last = WO["wo_last_index"]
    first_after = WO["wo_first_index_after"]
    check("last_index finds final occurrence", last(["=", "x", "=", "y"], "=") == 2)
    check("last_index strips tokens", last([" = ", "x"], "=") == 0)
    check("last_index None when absent", last(["a", "b"], "=") is None)
    check("first_index_after strictly after", first_after(["*", "a", "*"], "*", 0) == 2)
    check("first_index_after None when none after", first_after(["*", "a"], "*", 0) is None)
    check("first_index_after None index -> None", first_after(["*"], "*", None) is None)


def test_locate_c1_sites():
    loc = WO["wo_locate_c1_sites"]
    # canonical single-token operands: '( 0 + 23 ) * 47 ='.
    toks = ["(", "0", "+", "23", ")", "*", "47", "="]
    r = loc(toks, 23, 47)
    check("C1 sites ok", r["ok"])
    check("C1 rparen index", r["rparen"] == 4)
    check("C1 star index (first * after ))", r["star"] == 5)
    check("C1 c_operand last token", r["c_operand"] == 6)
    check("C1 equals index", r["equals"] == 7)
    check("C1 roles verify B at )", r["roles"]["B_at_rparen"])
    check("C1 roles verify C after *", r["roles"]["C_after_star"])
    # multi-token C operand ('4','7') shifts raw indices; content-walk still locates it.
    split = ["(", "0", "+", "23", ")", "*", "4", "7", "="]
    r2 = loc(split, 23, 47)
    check("multi-token C: ok", r2["ok"])
    check("multi-token C: span both tokens", r2["c_operand_span"] == [6, 7])
    check("multi-token C: c_operand is last token", r2["c_operand"] == 7)
    check("multi-token C: equals shifted", r2["equals"] == 8)
    # multi-token B ('2','3' before the )) — B role still verifies via walk-back.
    bsplit = ["(", "0", "+", "2", "3", ")", "*", "47", "="]
    r3 = loc(bsplit, 23, 47)
    check("multi-token B: ok", r3["ok"])
    check("multi-token B: rparen after both B tokens", r3["rparen"] == 5)
    # wrong B (role mismatch) -> not ok (so a caller skips/marks tokenizer_incompatible).
    rbad = loc(toks, 99, 47)
    check("wrong B -> roles fail -> not ok", not rbad["ok"] and not rbad["roles"]["B_at_rparen"])
    # no ')' at all -> not ok, rparen None.
    rnone = loc(["2", "*", "3", "="], 2, 3)
    check("no rparen -> rparen None, not ok", rnone["rparen"] is None and not rnone["ok"])
    # last ')' is the TEST's even when a few-shot prefix has an earlier ')'.
    fs = ["(", "0", "+", "22", ")", "*", "33", "=", "660",
          "(", "0", "+", "23", ")", "*", "47", "="]
    rfs = loc(fs, 23, 47)
    check("few-shot prefix: picks LAST ) (test expr)", rfs["rparen"] == 13)
    check("few-shot prefix: star/operand/equals after test )",
          rfs["star"] == 14 and rfs["c_operand"] == 15 and rfs["equals"] == 16)
    check("few-shot prefix: ok", rfs["ok"])


def test_locate_c1_sites_on_real_render():
    # build the actual C1 surface and a whitespace-token analog (the GPU cell decodes
    # real tokens; whitespace tokens are the CPU-testable proxy, as elsewhere here).
    loc = WO["wo_locate_c1_sites"]
    rend = {c[0]: c[2] for c in WO["WO_CONDITIONS"]}["C1"]
    for (B, C) in [(20, 49), (34, 41), (23, 47)]:
        toks = [t for t in rend(B, C).split(" ") if t != ""]
        r = loc(toks, B, C)
        check(f"render C1 ({B},{C}) located ok", r["ok"])
        check(f"render C1 ({B},{C}) B precedes )", toks[r["rparen"] - 1] == str(B))
        check(f"render C1 ({B},{C}) equals is last token", r["equals"] == len(toks) - 1)


# ------------------------------------------------- §3.3 boundary surfaces + trigger
def test_boundary_conditions_surfaces():
    rend = {c[0]: c[2] for c in WO["WO_BOUNDARY_CONDITIONS"]}
    gt = {c[0]: c[3] for c in WO["WO_BOUNDARY_CONDITIONS"]}
    aux = WO["wo_aux_operands"]
    B, C = 23, 47
    A, D = aux(B, C)
    check("aux operands deterministic", aux(B, C) == aux(B, C))
    lo, hi = WO["WO_BAND"]
    check("aux operands in band", lo <= A <= hi and lo <= D <= hi)
    check("BD1 surface (A+B)*C", rend["BD1"](B, C) == f"( {A} + {B} ) * {C} =")
    check("BD1 gt = (A+B)*C", gt["BD1"](B, C) == (A + B) * C)
    check("BD2 surface A*(B+C)", rend["BD2"](B, C) == f"{A} * ( {B} + {C} ) =")
    check("BD2 gt = A*(B+C)", gt["BD2"](B, C) == A * (B + C))
    check("BD3 surface (A*B)+C", rend["BD3"](B, C) == f"( {A} * {B} ) + {C} =")
    check("BD3 gt = (A*B)+C", gt["BD3"](B, C) == (A * B) + C)
    check("BD4 surface depth-2 ((0+B)*C)*D", rend["BD4"](B, C) == f"( ( 0 + {B} ) * {C} ) * {D} =")
    check("BD4 gt = B*C*D", gt["BD4"](B, C) == B * C * D)
    check("BD5 surface (A+B)*(C+D)", rend["BD5"](B, C) == f"( {A} + {B} ) * ( {C} + {D} ) =")
    check("BD5 gt = (A+B)*(C+D)", gt["BD5"](B, C) == (A + B) * (C + D))
    # surfaces render cleanly on the full shared pairs (no crash, all in band).
    pairs = WO["wo_build_pairs"]()
    ok_all = True
    for (b, c) in pairs[:50]:
        for k in rend:
            s = rend[k](b, c)
            ok_all = ok_all and isinstance(s, str) and s.endswith("=")
    check("all boundary surfaces render on shared pairs", ok_all)


def test_boundary_summary():
    bs = WO["wo_boundary_summary"]
    # clean trigger: every outer-* surface fails, the additive-outer control works.
    clean = bs({"BD1": 0.08, "BD2": 0.12, "BD3": 0.85, "BD4": 0.05, "BD5": 0.10})
    check("clean trigger consistent", clean["consistent"])
    check("clean trigger mult fails", clean["mult_outer_fails"])
    check("clean trigger add works", clean["add_outer_works"])
    check("clean characterization mentions multiplicative outer",
          "multiplicative outer" in clean["characterization"])
    check("summary has a row per surface", len(clean["rows"]) == 5)
    # asymmetry does NOT hold: bracketed product+outer fails too -> not the clean trigger.
    messy = bs({"BD1": 0.08, "BD2": 0.12, "BD3": 0.10, "BD4": 0.05, "BD5": 0.10})
    check("additive-outer failing -> add_outer_works false", not messy["add_outer_works"])
    check("messy -> not the clean characterization",
          "iff a bracketed sub-expression" not in messy["characterization"])
    # n/a inputs don't crash and are reported as 'n/a'.
    partial = bs({"BD1": 0.08})
    check("partial input -> observed n/a where missing",
          any(r["observed"] == "n/a" for r in partial["rows"]))


# ------------------------------------------------- §3.4 demo builders + format-cue
def test_demo_builders():
    dr = WO["wo_demo_render"]
    fr = WO["wo_fewshot_render"]
    rend = {c[0]: c[2] for c in WO["WO_CONDITIONS"]}["C1"]
    gt = {c[0]: c[3] for c in WO["WO_CONDITIONS"]}["C1"]
    pool = WO["wo_build_pairs"]()
    tp = pool[0]
    # invariant: a 'correct' demo prompt IS the existing few-shot prompt (same pairs).
    check("demo_render correct == fewshot_render",
          dr("correct", rend, gt, 4, tp, pool, 3) == fr(rend, gt, 4, tp, pool, 3))
    # every type: 5 lines (4 demos + test), last line is the canonical bare test prompt.
    for t in WO["WO_DEMO_TYPES"]:
        p = dr(t, rend, gt, 4, tp, pool, 9)
        lines = p.splitlines()
        check(f"{t}: 4 demos + test = 5 lines", len(lines) == 5)
        check(f"{t}: last line is bare canonical C1", lines[-1] == rend(*tp))
        check(f"{t}: prompt ends at '=' no trailing space",
              p.endswith("=") and not p.endswith(" "))
        check(f"{t}: deterministic given seed", p == dr(t, rend, gt, 4, tp, pool, 9))
    # wrong_answer: same LHS surface as correct, but the answer is WRONG, same #digits.
    wlines = dr("wrong_answer", rend, gt, 3, tp, pool, 11).splitlines()[:-1]
    clines = dr("correct", rend, gt, 3, tp, pool, 11).splitlines()[:-1]
    ok_w = True
    for wl, cl in zip(wlines, clines):
        wlhs, wans = wl.rsplit(" ", 1)
        clhs, cans = cl.rsplit(" ", 1)
        ok_w = ok_w and (wlhs == clhs) and (wans != cans) and (len(wans) == len(cans))
    check("wrong_answer: same LHS as correct, wrong same-#digits answer", ok_w)
    # scrambled_format: same token multiset on the LHS, ends with '= <correct value>'.
    import re as _re
    sline = dr("scrambled_format", rend, gt, 1, tp, pool, 13).splitlines()[0]
    cline = dr("correct", rend, gt, 1, tp, pool, 13).splitlines()[0]
    check("scrambled same token multiset as correct demo",
          sorted(sline.split()) == sorted(cline.split()))
    check("scrambled ends with '= <correct value>'",
          _re.search(r"= (\d+)$", sline).group(1) == cline.split()[-1])
    # random_text: length-matched (same whitespace-token count), NO digits, no '='.
    rline = dr("random_text", rend, gt, 1, tp, pool, 17).splitlines()[0]
    check("random_text length-matched (token count == correct demo)",
          len(rline.split()) == len(cline.split()))
    check("random_text has no digits", not any(ch.isdigit() for ch in rline))
    check("random_text has no '='", "=" not in rline)
    check("gt_wrong is wrong + same #digits",
          WO["wo_gt_wrong"](23, 47) != 23 * 47
          and len(str(WO["wo_gt_wrong"](23, 47))) == len(str(23 * 47)))


def test_format_cue_verdict():
    v = WO["wo_format_cue_verdict"]
    # wrong ~= correct AND random flat -> FORMAT_PRIMED (the surprising WO#3 hint).
    fp = v({"correct": 0.92, "wrong_answer": 0.88, "scrambled_format": 0.80,
            "random_text": 0.30}, zeroshot_acc=0.27)
    check("format-primed label", fp["label"] == "FORMAT_PRIMED")
    check("format-primed: wrong recovers", fp["recovers"]["wrong_answer"])
    check("format-primed: random does not recover", not fp["recovers"]["random_text"])
    # only correct recovers -> CONTENT_DRIVEN.
    cd = v({"correct": 0.92, "wrong_answer": 0.30, "scrambled_format": 0.30,
            "random_text": 0.28}, zeroshot_acc=0.27)
    check("content-driven label", cd["label"] == "CONTENT_DRIVEN")
    # neither recovers / ambiguous -> MIXED.
    mx = v({"correct": 0.35, "wrong_answer": 0.33, "random_text": 0.30}, zeroshot_acc=0.27)
    check("ambiguous -> MIXED", mx["label"] == "MIXED")
    # missing zeroshot -> MIXED (can't judge recovery), no crash.
    check("missing zeroshot -> MIXED",
          v({"correct": 0.9, "wrong_answer": 0.9}, zeroshot_acc=None)["label"] == "MIXED")


# ------------------------------------------------- §3.5 refined error classifier
def test_classify_error_detail():
    ed = WO["wo_classify_error_detail"]
    B, C = 23, 47                          # prod=1081, B+C=70
    check("correct (==B*C)", ed(1081, B, C) == "correct")
    check("parse_fail (None)", ed(None, B, C) == "parse_fail")
    check("equals_B", ed(23, B, C) == "equals_B")
    check("equals_C", ed(47, B, C) == "equals_C")
    check("equals_B_plus_C", ed(70, B, C) == "equals_B_plus_C")
    check("near_product (within 10%)", ed(1040, B, C) == "near_product")  # |Δ|/1081=0.038
    check("near_product upper edge (just inside 10%)", ed(1180, B, C) == "near_product")  # 0.0916
    check("right_magnitude (4 digits, not near)", ed(1500, B, C) == "right_magnitude")
    check("unrelated (1 digit, far)", ed(7, B, C) == "unrelated")
    check("unrelated (5 digits)", ed(50000, B, C) == "unrelated")
    # priority: exact product beats near_product/right_magnitude.
    check("exact product not mislabeled near", ed(1081, B, C) == "correct")
    # exact-value buckets beat the fuzzy ones (B sits inside the magnitude bucket band?).
    check("equals_B beats fuzzy buckets", ed(B, B, C) == "equals_B")
    # categories list is exhaustive for these.
    cats = set(WO["WO_ERROR_DETAIL_CATS"])
    for pred in [None, 1081, 23, 47, 70, 1040, 1500, 7]:
        check(f"category for {pred} is registered", ed(pred, B, C) in cats)


def main():
    for fn in sorted(g for g in globals() if g.startswith("test_")):
        print(f"\n{fn}:")
        globals()[fn]()
    print("\n" + ("ALL PASS" if not _fails else f"FAILURES: {_fails}"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
