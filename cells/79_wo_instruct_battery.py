# ============================================================================
# Phase 6 / WO — STEP 2 + STEP 3 : Instruct G2 parity, then the full battery.
# ----------------------------------------------------------------------------
# STEP 2 (§5.4, §11): swap to -Instruct (bare-continuation = PRIMARY format) and
#   assert C1/C2 token parity on the live tokenizer. base & -Instruct SHARE the
#   tokenizer/vocab, so bare parity is inherited — but we ASSERT, never assume.
# STEP 3 (§3): run C0..C8 on Instruct, N=400, from the SAME `pairs`, greedy,
#   bare-continuation. Per-condition acc / corr / parse-fail; re-derive
#   |acc(C1)-acc(C2)| and Jaccard(C1,C2).
#
# DEGENERACY GUARD (§11): if bare-continuation is degenerate on Instruct (it
#   chats/refuses instead of emitting a number -> C0 parse-fail spikes), we WARN
#   and, iff WO_RUN_CHAT_SECONDARY, fall back to a minimal chat wrapper AND
#   re-run G2 parity on the wrapped tokenization before trusting it (wrapping
#   shifts token indices). The format that produced the gated numbers is recorded.
# ============================================================================
import numpy as np

WO_RUN_CHAT_SECONDARY = bool(CFG.get("wo_run_chat_secondary", False))  # opt-in (§11)
WO_BARE_DEGENERATE_PARSEFAIL = 0.50   # C0 parse-fail above this => bare is degenerate.

# ---- STEP 2: load Instruct + assert C1/C2 parity (bare-continuation) ----
wo_load_model("instruct")
_renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
_renderC2 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C2"]
_parity_ok, _parity_bad = wo_assert_parity(WO_PAIRS, _renderC1, _renderC2)
if not _parity_ok:
    raise RuntimeError(
        f"WO STEP 2 ABORT: C1/C2 token parity BROKEN on the Instruct tokenizer "
        f"({_parity_bad[:5]}). The patch construction requires equal length + B at "
        f"the same index; do not proceed (work order §5.4).")
log("WO STEP 2: Instruct C1/C2 parity holds (bare-continuation).")

# ---- STEP 3: full battery on Instruct (bare-continuation, the gated format) ----
WO_INSTRUCT_FORMAT = "bare-continuation"
WO_INSTRUCT_RES = wo_run_battery("instruct", WO_CONDITIONS, WO_PAIRS)

# ---- degeneracy guard: did bare-continuation collapse to chatter? ----
_c0_pf = WO_INSTRUCT_RES["C0"]["parse_fail_rate"]
if _c0_pf > WO_BARE_DEGENERATE_PARSEFAIL:
    log(f"WO WARNING: bare-continuation Instruct C0 parse-fail={_c0_pf:.2f} > "
        f"{WO_BARE_DEGENERATE_PARSEFAIL} — Instruct may be chatting instead of emitting "
        f"a number. {'Falling back to chat wrapper.' if WO_RUN_CHAT_SECONDARY else 'Set CFG[\"wo_run_chat_secondary\"]=True to run the chat fallback.'}")
    if WO_RUN_CHAT_SECONDARY:
        # minimal chat wrapper: a single user turn asking to complete the expression.
        # apply_chat_template ALREADY emits the BOS; the scored pipeline (_encode) re-adds
        # add_special_tokens=True -> a DOUBLE BOS that shifts every position and corrupts
        # the numbers. Strip the template's leading BOS so exactly one remains after _encode.
        _bos = getattr(tokenizer, "bos_token", None)
        def _chat_render(render):
            def _r(B, C):
                msg = [{"role": "user",
                        "content": "Compute and reply with ONLY the integer:\n" + render(B, C)}]
                try:
                    s = tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True)
                except Exception:
                    return render(B, C)
                if _bos and s.startswith(_bos):
                    s = s[len(_bos):]
                return s
            return _r
        # verify exactly one BOS after the scored-pipeline tokenization before trusting it.
        _probe_ids = tokenizer(_chat_render(_renderC1)(23, 47),
                               add_special_tokens=WO_ADD_SPECIAL_TOKENS)["input_ids"]
        _bos_id = tokenizer.bos_token_id
        _bos_count = sum(1 for t in _probe_ids if t == _bos_id) if _bos_id is not None else 0
        if _bos_count != 1:
            log(f"WO WARNING: chat-wrapped prompt has {_bos_count} BOS tokens (expected 1); "
                f"chat numbers may be malformed — not promoting to the gated set.")
        _chat_conditions = [(k, n + "_chat", _chat_render(r), g) for (k, n, r, g) in WO_CONDITIONS]
        # re-run parity on the WRAPPED C1/C2 (wrapping shifts indices) before trusting it.
        _cp_ok, _cp_bad = wo_assert_parity(
            WO_PAIRS, _chat_render(_renderC1), _chat_render(_renderC2))
        if not _cp_ok:
            log(f"WO: chat-wrapped C1/C2 parity BROKEN ({_cp_bad[:3]}); the chat numbers are "
                f"NOT patch-valid — reporting them as a sanity check only, not the gated set.")
        # live model stays 'instruct'; namespace the chat caches separately via cache_tag.
        WO_INSTRUCT_RES_CHAT = wo_run_battery(
            "instruct", _chat_conditions, WO_PAIRS, cache_tag="instruct_chat")
        save_json("wo_instruct_chat_battery",
                  {k: {kk: vv for kk, vv in v.items()
                       if kk not in ("correct_mask", "preds", "golds", "prompts")}
                   for k, v in WO_INSTRUCT_RES_CHAT.items()})
        if _cp_ok and _bos_count == 1:
            # chat parity + single-BOS hold -> chat becomes the gated format.
            WO_INSTRUCT_RES = WO_INSTRUCT_RES_CHAT
            WO_INSTRUCT_FORMAT = "chat-wrapped (bare was degenerate; parity + single-BOS re-asserted)"
            log("WO: switched gated battery to chat-wrapped Instruct (re-parity OK).")

# ---- derived numbers the gates need ----
WO_INSTRUCT_ACC = {k: v["exact_acc"] for k, v in WO_INSTRUCT_RES.items()}
WO_INSTRUCT_CORR = {k: v["corr"] for k, v in WO_INSTRUCT_RES.items()}
WO_INSTRUCT_PARSEFAIL = {k: v["parse_fail_rate"] for k, v in WO_INSTRUCT_RES.items()}
WO_ACC_DELTA_C1C2 = abs(WO_INSTRUCT_ACC["C1"] - WO_INSTRUCT_ACC["C2"])
WO_JACCARD_C1C2 = wo_jaccard(WO_INSTRUCT_RES["C1"]["correct_mask"],
                             WO_INSTRUCT_RES["C2"]["correct_mask"])

# ---- write results/instruct_battery.csv ----
_name_by_key = {c[0]: c[1] for c in WO_CONDITIONS}
_rows = []
for k in [c[0] for c in WO_CONDITIONS]:
    r = WO_INSTRUCT_RES[k]
    _rows.append({
        "cond": k, "name": _name_by_key[k],
        "acc": r["exact_acc"], "corr": r["corr"],
        "parse_fail": r["parse_fail_rate"], "n": r["n"], "n_parsed": r["n_parsed"],
        "mean_abs_output": r["mean_abs_output"], "format": WO_INSTRUCT_FORMAT,
    })
_csv = wo_battery_csv(
    _rows, ["cond", "name", "acc", "corr", "parse_fail", "n", "n_parsed",
            "mean_abs_output", "format"])
_csv += (f"\n# derived,|acc(C1)-acc(C2)|={WO_ACC_DELTA_C1C2:.4f},"
         f"Jaccard(C1,C2)={WO_JACCARD_C1C2:.4f}\n")
wo_save_result("instruct_battery.csv", _csv)

save_json("wo_instruct_battery", {k: {kk: vv for kk, vv in v.items()
                                      if kk not in ("correct_mask", "preds", "golds", "prompts")}
                                  for k, v in WO_INSTRUCT_RES.items()})

print("\n================= WO STEP 3 — INSTRUCT BATTERY (format: "
      f"{WO_INSTRUCT_FORMAT}) =================")
print(f"{'cond':<5}{'name':<20}{'acc':>7}{'corr':>9}{'parse_fail':>12}")
for k in [c[0] for c in WO_CONDITIONS]:
    r = WO_INSTRUCT_RES[k]
    _c = "  n/a" if r["corr"] is None else f"{r['corr']:.3f}"
    print(f"{k:<5}{_name_by_key[k]:<20}{r['exact_acc']:>7.3f}{_c:>9}{r['parse_fail_rate']:>12.3f}")
print("-------------------------------------------------------------------------")
print(f"|acc(C1)-acc(C2)| = {WO_ACC_DELTA_C1C2:.3f}   Jaccard(C1,C2) = {WO_JACCARD_C1C2:.3f}")
print("=========================================================================")
