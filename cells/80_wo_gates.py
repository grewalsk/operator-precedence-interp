# ============================================================================
# Phase 6 / WO — STEP 4 : evaluate the SIX validity gates (work order §7).
# ----------------------------------------------------------------------------
# Pure logic (wo_evaluate_gates) over the Instruct battery numbers from Step 3.
# Emits results/gate_evaluation.json: each gate's value + pass/fail, the
# localization VALID/INVALID verdict, and the G_surface scope flag. Also mirrors
# the six WO gates into the gate ledger (WO_* keys) so the dashboard can show them
# without colliding with G0..G4.
# ============================================================================
import json

assert "WO_INSTRUCT_ACC" in globals(), "Run the Instruct battery cell (79) first."

WO_GATE_EVAL = wo_evaluate_gates(
    acc=WO_INSTRUCT_ACC, corr=WO_INSTRUCT_CORR, jaccard_c1c2=WO_JACCARD_C1C2)

# Assemble the deliverable JSON (§12). Keep the raw inputs alongside the verdicts
# so every gate number is reproducible from this one file.
_out = {
    "model": WO_MODEL_REGISTRY["instruct"],
    "format": WO_INSTRUCT_FORMAT,
    "band": list(WO_BAND), "n": WO_N, "seed": WO_SEED,
    "inputs": {
        "acc": WO_INSTRUCT_ACC,
        "corr": WO_INSTRUCT_CORR,
        "parse_fail": WO_INSTRUCT_PARSEFAIL,
        "acc_delta_C1_C2": WO_ACC_DELTA_C1C2,
        "jaccard_C1_C2": WO_JACCARD_C1C2,
    },
    "gates": {g: {"definition": WO_GATE_SPEC[g],
                  "value": WO_GATE_EVAL["gates"][g]["value"],
                  "pass": WO_GATE_EVAL["gates"][g]["pass"]}
              for g in ["G_floor", "G_neutral", "G_symmetry",
                        "G_quantity", "G_surface", "G_support"]},
    "hard_gate_AND": WO_GATE_EVAL["localization_valid"],
    "failed_hard_gates": WO_GATE_EVAL["failed_hard_gates"],
    "g_surface_pass": WO_GATE_EVAL["g_surface_pass"],
    "localization_verdict": WO_GATE_EVAL["verdict"],
    "scope": WO_GATE_EVAL["scope"],
}
wo_save_result("gate_evaluation.json", json.dumps(_out, indent=2, default=str))
save_json("wo_gate_evaluation", _out)

# Mirror into the gate ledger (WO_-prefixed; informational, does not touch G0..G4).
for g in ["G_floor", "G_neutral", "G_symmetry", "G_quantity", "G_surface", "G_support"]:
    gg = WO_GATE_EVAL["gates"][g]
    set_gate(f"WO_{g}", gg["pass"], f"{WO_GATE_SPEC[g]} -> value={gg['value']}")

print("\n================= WO STEP 4 — VALIDITY GATES (§7, Instruct) =================")
for g in ["G_floor", "G_neutral", "G_symmetry", "G_quantity", "G_surface", "G_support"]:
    gg = WO_GATE_EVAL["gates"][g]
    flag = "  (SCOPE FLAG)" if g == "G_surface" else ""
    print(f"  [{'PASS' if gg['pass'] else 'FAIL'}] {g:<11} {WO_GATE_SPEC[g]:<52} "
          f"value={gg['value']}{flag}")
print("---------------------------------------------------------------------------")
print(f"  hard-gate AND (floor∧neutral∧symmetry∧quantity∧support): "
      f"{WO_GATE_EVAL['localization_valid']}")
print(f"  LOCALIZATION: {WO_GATE_EVAL['verdict']}  —  {WO_GATE_EVAL['scope']}")
print("===========================================================================")
