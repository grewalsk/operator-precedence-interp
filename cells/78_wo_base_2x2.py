# ============================================================================
# Phase 6 / WO — STEP 0 + STEP 1 : instrument sanity, then COMPLETE the base 2x2.
# ----------------------------------------------------------------------------
# STEP 0 (work order §6): instrument sanity = G4 reproduced the known single-
#   addition localization. On a full Run-All, Phase 5 (cell 75) already asserted
#   G4 PASS, so here we only CONFIRM the gate ledger is green and STOP loudly if
#   not (a broken instrument makes everything downstream untrustworthy).
#
# STEP 1 (§6, MUST run before any Instruct conclusion): complete the
#   {spaces,no-space} x {inside-bracket, outer-compose} 2x2 by running the ONE
#   missing cell — no-space (B*C)= [C8] — on BASE, from the shared `pairs`. We
#   also re-run the full C0..C8 battery on base from the same pairs so the
#   base column is paired and self-contained for the base-vs-Instruct table.
#
# VERDICT (§6 interpretation rule):
#   - C8 ALSO collapses (~C7's 0.02) -> C7 is PURE TOKENIZATION -> surface
#     fragility is an independent CO-HEADLINE.
#   - C8 SURVIVES (~0.7+)            -> collapse is COMPOSE-SPECIFIC -> fold C7
#     into the composition story (one headline).
# ============================================================================
import numpy as np

# ---- STEP 0: confirm the instrument (G4) is green before trusting anything ----
_gates_now = get_gates()
_g4 = _gates_now.get("G4", {})
_g4_pass = bool(_g4.get("passed", _g4.get("pass")))
if not _g4_pass:
    raise RuntimeError(
        "WO STEP 0 ABORT: G4 (patching instrument) is not PASS in the gate ledger. "
        "The single-addition localization must reproduce before any C1/C2 patch is "
        "trustworthy (work order §6 Step 0). Run Phase 5 and resolve G4 first.")
log(f"WO STEP 0: instrument sanity OK — G4 PASS on disk ({_g4.get('detail','')[:80]}).")

# ---- shared deterministic pairs (one list for ALL conditions, both models) ----
WO_PAIRS = wo_build_pairs(n=WO_N, band=WO_BAND, seed=WO_SEED)
WO_PAIRS_HASH = wo_stim_hash(WO_PAIRS)
log(f"WO: {len(WO_PAIRS)} shared (B,C) pairs, band {WO_BAND}, seed {WO_SEED}, "
    f"hash {WO_PAIRS_HASH[:12]}.")

# ---- ensure BASE is live (Phase 0 loaded it; reload only if something swapped) ----
wo_load_model("base")

# ---- STEP 1: run the full C0..C8 battery on base from the shared pairs ----
WO_BASE_RES = wo_run_battery("base", WO_CONDITIONS, WO_PAIRS)

# ---- assemble the 2x2 (work order §6 table) ----
def _acc(res, k):
    return res[k]["exact_acc"]

acc_c4 = _acc(WO_BASE_RES, "C4")   # spaces, inside-bracket
acc_c1 = _acc(WO_BASE_RES, "C1")   # spaces, outer-compose
acc_c8 = _acc(WO_BASE_RES, "C8")   # NO-space, inside-bracket  (the NEW cell)
acc_c7 = _acc(WO_BASE_RES, "C7")   # NO-space, outer-compose

WO_BASE_2X2_VERDICT = wo_2x2_verdict(acc_c4=acc_c4, acc_c7=acc_c7, acc_c8=acc_c8)
WO_BASE_2X2_VERDICT["acc"]["C1_via_caller"] = acc_c1   # fill the spaced-outer cell

# ---- write results/base_2x2.csv (the 2x2 + per-cell numbers + verdict) ----
_rows = [
    {"axis_surface": "spaces",  "axis_compose": "inside_bracket", "cond": "C4",
     "surface": "( B * C ) =",  "acc": acc_c4, "corr": WO_BASE_RES["C4"]["corr"],
     "parse_fail": WO_BASE_RES["C4"]["parse_fail_rate"]},
    {"axis_surface": "spaces",  "axis_compose": "outer_compose",  "cond": "C1",
     "surface": "( 0 + B ) * C =", "acc": acc_c1, "corr": WO_BASE_RES["C1"]["corr"],
     "parse_fail": WO_BASE_RES["C1"]["parse_fail_rate"]},
    {"axis_surface": "nospace", "axis_compose": "inside_bracket", "cond": "C8",
     "surface": "(B*C)=",       "acc": acc_c8, "corr": WO_BASE_RES["C8"]["corr"],
     "parse_fail": WO_BASE_RES["C8"]["parse_fail_rate"]},
    {"axis_surface": "nospace", "axis_compose": "outer_compose",  "cond": "C7",
     "surface": "(0+B)*C=",     "acc": acc_c7, "corr": WO_BASE_RES["C7"]["corr"],
     "parse_fail": WO_BASE_RES["C7"]["parse_fail_rate"]},
]
_csv = wo_battery_csv(
    _rows, ["axis_surface", "axis_compose", "cond", "surface", "acc", "corr", "parse_fail"])
_csv += f"\n# verdict,{WO_BASE_2X2_VERDICT['verdict']}\n"
_csv += f"# headline,\"{WO_BASE_2X2_VERDICT['headline']}\"\n"
_csv += (f"# new_cell_C8_acc,{acc_c8:.4f}  (C7={acc_c7:.4f}, C4={acc_c4:.4f}, "
         f"survive>={WO_C8_SURVIVE_ACC}, collapse<=C7+{WO_C8_COLLAPSE_MARGIN})\n")
wo_save_result("base_2x2.csv", _csv)

# persist for the decision record + a full base battery snapshot for repro.
save_json("wo_base_battery", {k: {kk: vv for kk, vv in v.items()
                                  if kk not in ("correct_mask", "preds", "golds", "prompts")}
                              for k, v in WO_BASE_RES.items()})
save_json("wo_base_2x2_verdict", WO_BASE_2X2_VERDICT)

print("\n================= WO STEP 1 — BASE 2x2 (surface x compose) =================")
print(f"{'':>10} {'inside-bracket':>16} {'outer-compose':>16}")
print(f"{'spaces':>10} {('C4 ' + format(acc_c4, '.3f')):>16} {('C1 ' + format(acc_c1, '.3f')):>16}")
print(f"{'no-space':>10} {('C8 ' + format(acc_c8, '.3f')):>16} {('C7 ' + format(acc_c7, '.3f')):>16}")
print("---------------------------------------------------------------------------")
print(f"VERDICT: {WO_BASE_2X2_VERDICT['verdict']}")
print(f"  {WO_BASE_2X2_VERDICT['headline']}")
print(f"  (new cell C8 (B*C)= acc={acc_c8:.3f}; survives>={WO_C8_SURVIVE_ACC}? "
      f"{WO_BASE_2X2_VERDICT['survives']}; collapses~C7? {WO_BASE_2X2_VERDICT['collapses']})")
print("===========================================================================")
