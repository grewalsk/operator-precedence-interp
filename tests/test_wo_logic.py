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


def main():
    for fn in sorted(g for g in globals() if g.startswith("test_")):
        print(f"\n{fn}:")
        globals()[fn]()
    print("\n" + ("ALL PASS" if not _fails else f"FAILURES: {_fails}"))
    return 0 if not _fails else 1


if __name__ == "__main__":
    sys.exit(main())
