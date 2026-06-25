# ============================================================================
# Phase 6 / WORK ORDER #4 — §3.1 CROSS-MODEL GENERALITY (GPU; the make-or-break).
# ----------------------------------------------------------------------------
# Turns a single-checkpoint behavioral finding into a cross-FAMILY, cross-SCALE
# one by re-running the EXISTING model-tag-keyed harness on instruction-following
# models <= 9B (Qwen2.5-7B, Gemma-2-9B, Mistral-7B) + a Llama-3.2 scale pair
# (1B, 3B), on the SAME shared WO_PAIRS / band / seed / K so every comparison is
# paired. For each model, on the shared pairs:
#   1. wo_load_model(tag)            — frees the previous model first (VRAM safe).
#      A model we can't access (gated/no HF_TOKEN) is SKIPPED + reported
#      (status=access_denied), NEVER crashes the run (§6 hazard).
#   2. re-validate the tokenizer     — wo_assert_parity on C1/C2 (each model
#      tokenizes the band differently); a BROKEN parity => tokenizer_incompatible
#      for patch/probe parts, but its BATTERY is still reported (a finding).
#   3. degeneracy guard (mirror 79)  — -it models often CHAT under bare-continuation
#      (C0 parse-fail spikes); fall back to a minimal chat wrapper (strip the
#      template's extra BOS — double-BOS pitfall — and re-assert parity on the
#      wrapped form). The format used is recorded PER MODEL.
#   4. run the key conditions        — C0,C1,C4,C6,C7,C8 + A1,A2,D1 + few-shot C1
#      at shots {0,2,4} (reuse the cell-82a few-shot pattern, per-item seeds).
#   5. wo_replication_verdict (cell 76, unit-tested) -> a per-model row.
#
# HONESTY (the project's hard-won lesson, §1/§6): a non-replication is a RESULT —
# report it; an out-of-scope model (can't do C4/C6) is reported SEPARATELY, not
# counted as a failure-to-replicate. Every model lands in the table.
#
# Resumable: per-model summary cached (wo_crossmodel_<tag>); each forward pass is
# cached per (cache_tag, condition) by _eval_prompts, so a disconnect resumes by
# (model, condition). Re-running a finished model is a no-op (summary on disk).
# ============================================================================
import json
import numpy as np
import torch

assert "WO_PAIRS" in globals() and "wo_replication_verdict" in globals() \
    and "wo_run_battery" in globals() and "wo_fewshot_render" in globals(), (
    "WO#4 cross-model needs WO_PAIRS (cell 78) + cell-76 logic + cell-77 setup.")

# Which models to battery. Default = the NEW models (base/instruct already have
# results and are added as reference rows below). Override via CFG to subset.
CFG.setdefault("wo_crossmodel_tags",
               ["qwen25_7b_it", "gemma2_9b_it", "mistral_7b_it",
                "llama32_1b_it", "llama32_3b_it"])
CFG.setdefault("wo_fewshot_seed", 202)         # SAME per-item seeding as cell 82a.
WO_XM_KEYCONDS = ["C0", "C1", "C4", "C6", "C7", "C8"]
WO_XM_FEWSHOT_SHOTS = [0, 2, 4]
WO_XM_BARE_DEGEN_PF = 0.50                      # C0 parse-fail above this => bare degenerate.

_xm_renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
_xm_renderC2 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C2"]
_xm_gtC1 = dict((c[0], c[3]) for c in WO_CONDITIONS)["C1"]
_xm_keyconds = [c for c in WO_CONDITIONS if c[0] in WO_XM_KEYCONDS]

# in-memory store of full battery results (with preds) for the §3.5 error-detail cell.
WO_CROSSMODEL_RES = globals().get("WO_CROSSMODEL_RES", {})


def _xm_chat_wrap_factory():
    """Minimal chat wrapper for a degenerate -it model (mirror cell 79). Wraps an
    arbitrary continuation `content` in a single user turn and strips the template's
    leading BOS so the scored pipeline's add_special_tokens=True leaves exactly one
    BOS (the double-BOS pitfall shifts every position)."""
    _bos = getattr(tokenizer, "bos_token", None)

    def wrap(content):
        msg = [{"role": "user",
                "content": "Compute and reply with ONLY the integer:\n" + content}]
        try:
            s = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        except Exception:
            return content
        if _bos and s.startswith(_bos):
            s = s[len(_bos):]
        return s
    return wrap


def _xm_wrap_conditions(conds, wrap):
    """Wrap each (key,name,render,gt) condition's render in `wrap` (default-arg
    capture so the loop closure binds the right render)."""
    out = []
    for (k, n, r, g) in conds:
        out.append((k, n, (lambda B, C, _r=r: wrap(_r(B, C))), g))
    return out


def _xm_fewshot_c1_acc(cache_tag, shots, wrap):
    """C1 accuracy under `shots` correct few-shot demos (per-item seeded, SAME recipe
    as cell 82a). 0-shot is handled by the caller (it equals battery C1)."""
    base = int(CFG["wo_fewshot_seed"]) + 1000 * int(shots)
    prompts = [wrap(wo_fewshot_render(_xm_renderC1, _xm_gtC1, shots, (B, C), WO_PAIRS,
                                      seed=base + i))
               for i, (B, C) in enumerate(WO_PAIRS)]
    golds = [B * C for (B, C) in WO_PAIRS]
    preds = [parse_int(c) for c in wo_eval(prompts, f"fewshot_C1_{shots}shot", cache_tag)]
    return float(np.mean([p is not None and p == g for p, g in zip(preds, golds)]))


def _xm_run_model(tag):
    """Battery + replication verdict for one cross-model. Resumable (summary cached);
    a load/access failure is reported, never raised."""
    ck = f"wo_crossmodel_{tag}"
    if has_artifact(ck, "json"):
        out = load_json(ck)
        log(f"WO#4 cross-model [{tag}]: cached — reused (label={out.get('label')}).")
        return out

    name = WO_MODEL_REGISTRY.get(tag, tag)
    # ---- 1) load (gated/unavailable/load-error -> report, don't crash) ----
    try:
        wo_load_model(tag)
    except Exception as e:
        _emsg = f"{type(e).__name__}: {str(e)[:300]}"
        _low = _emsg.lower()
        # distinguish TRUE gating (accept-the-license) from any OTHER load failure
        # (version/ABI clash, download error, OOM) — the old blanket 'access_denied'
        # lied about ungated models. Auth markers: 401/403/gated/restricted/awaiting.
        _is_auth = any(s in _low for s in ("401", "403", "gatedrepo", "gated repo",
                                           "restricted", "awaiting", "must accept",
                                           "unauthorized", "permission"))
        out = {"tag": tag, "model": name,
               "status": "access_denied" if _is_auth else "load_failed",
               "error": _emsg, "format": "n/a", "parity_ok": None,
               "label": ("ACCESS_DENIED (accept the license at huggingface.co/" + name + ")"
                         if _is_auth else f"LOAD_FAILED ({type(e).__name__}) — see error field")}
        save_json(ck, out)
        log(f"WO#4 cross-model [{tag}]: "
            f"{'ACCESS DENIED (gating)' if _is_auth else 'LOAD FAILED'} — {_emsg}")
        return out

    try:
        # ---- 2) tokenizer re-validation (parity is recorded, not a hard abort) ----
        parity_ok, parity_bad = wo_assert_parity(WO_PAIRS, _xm_renderC1, _xm_renderC2)

        # ---- 3) degeneracy guard: bare parse-fail -> chat fallback ----
        # Check BOTH C0 (easy) AND C1 (hard): a model can answer C0 fine yet CHAT on
        # the harder C1 (Gemma did: C0 ok, C1 48% parse-fail), which silently degrades
        # the headline number. Trigger chat-wrap if EITHER is degenerate.
        _degen_bare = wo_run_battery(tag, [c for c in _xm_keyconds if c[0] in ("C0", "C1")],
                                     WO_PAIRS, cache_tag=tag)
        c0_pf = _degen_bare["C0"]["parse_fail_rate"]
        c1_pf = _degen_bare["C1"]["parse_fail_rate"]
        degen_pf = max(c0_pf, c1_pf)
        fmt, cache_tag, wrap = "bare-continuation", tag, (lambda s: s)
        chat_ok = hasattr(tokenizer, "apply_chat_template")
        if degen_pf > WO_XM_BARE_DEGEN_PF and chat_ok:
            wrap = _xm_chat_wrap_factory()
            # single-BOS sanity on the wrapped, scored tokenization (cell 79 pattern).
            _pids = tokenizer(wrap(_xm_renderC1(23, 47)),
                              add_special_tokens=WO_ADD_SPECIAL_TOKENS)["input_ids"]
            _bid = tokenizer.bos_token_id
            _bos_n = sum(1 for t in _pids if t == _bid) if _bid is not None else 0
            # re-assert parity on the WRAPPED C1/C2 (wrapping shifts indices).
            cp_ok, _ = wo_assert_parity(WO_PAIRS,
                                        (lambda B, C: wrap(_xm_renderC1(B, C))),
                                        (lambda B, C: wrap(_xm_renderC2(B, C))))
            fmt = (f"chat-wrapped (bare parse-fail C0={c0_pf:.2f}/C1={c1_pf:.2f}; "
                   f"BOS={_bos_n}; wrapped-parity={'OK' if cp_ok else 'BROKEN'})")
            cache_tag = f"{tag}_chat"
            parity_ok = bool(cp_ok)
            log(f"WO#4 [{tag}]: bare degenerate (C0 pf={c0_pf:.2f}, C1 pf={c1_pf:.2f}) "
                f"-> chat-wrapped (BOS={_bos_n}, wrapped-parity={cp_ok}).")
        elif degen_pf > WO_XM_BARE_DEGEN_PF:
            fmt = (f"bare-continuation (DEGENERATE C0 pf={c0_pf:.2f}/C1 pf={c1_pf:.2f}; "
                   f"no chat template)")
            log(f"WO#4 [{tag}]: bare degenerate and no chat template — reporting bare anyway.")

        # ---- 4) key battery + branch-B controls in the chosen format ----
        key_res = wo_run_battery(tag, _xm_wrap_conditions(_xm_keyconds, wrap),
                                 WO_PAIRS, cache_tag=cache_tag)
        bb_res = wo_run_battery(tag, _xm_wrap_conditions(WO_BRANCHB_CONDITIONS, wrap),
                                WO_PAIRS, cache_tag=cache_tag)
        battery = dict(key_res)
        battery.update(bb_res)
        WO_CROSSMODEL_RES[tag] = battery          # in-memory (with preds) for §3.5.

        acc = {k: battery[k]["exact_acc"] for k in battery}

        # ---- few-shot C1 at {0,2,4} ----
        fewshot = {0: acc.get("C1")}
        for s in (2, 4):
            fewshot[s] = _xm_fewshot_c1_acc(cache_tag, s, wrap)

        # ---- 5) replication verdict ----
        verdict = wo_replication_verdict(acc, fewshot.get(4))

        out = {
            "tag": tag, "model": name, "status": "ok", "format": fmt,
            "parity_ok": bool(parity_ok),
            "acc": {k: acc.get(k) for k in
                    ["C0", "C1", "C4", "C6", "C7", "C8", "A1", "A2", "D1"]},
            "fewshot_c1": {str(s): fewshot.get(s) for s in WO_XM_FEWSHOT_SHOTS},
            "c1_parse_fail": battery["C1"]["parse_fail_rate"],
            "c0_bare_parse_fail": float(c0_pf),
            "c1_preds": battery["C1"]["preds"],     # for §3.5 error-detail on resume.
            "verdict": verdict, "label": verdict["label"],
            "n": len(WO_PAIRS),
        }
        save_json(ck, out)
        log(f"WO#4 cross-model [{tag}] ({fmt.split()[0]}): C1={acc.get('C1')} "
            f"C4={acc.get('C4')} C6={acc.get('C6')} fs@4={fewshot.get(4)} -> {verdict['label']}")
        return out
    except Exception as e:
        out = {"tag": tag, "model": name, "status": "error",
               "error": f"{type(e).__name__}: {str(e)[:200]}", "format": "n/a",
               "parity_ok": None, "label": f"ERROR ({type(e).__name__})"}
        save_json(ck, out)
        log(f"WO#4 cross-model [{tag}]: ERROR mid-battery ({out['error']}). Reported, run continues.")
        return out


# --- reference rows for base + instruct (prior WO run; no recompute) ----------
def _xm_reference_row(tag):
    """Build a table row for base/instruct from already-in-memory results so the
    cross-model table is COMPLETE (every model in one table). Few-shot from the
    cell-82a artifact; A1/A2/D1 from the cell-82 branch-B globals."""
    bat = globals().get("WO_INSTRUCT_RES" if tag == "instruct" else "WO_BASE_RES")
    if bat is None or "C1" not in bat:
        return None
    acc = {k: bat[k]["exact_acc"] for k in bat}
    bb = globals().get(f"WO_BRANCHB_RES_{tag}")
    for k in ("A1", "A2", "D1"):
        if bb and k in bb:
            acc[k] = bb[k]["exact_acc"]
    # few-shot C1@{0,2,4} from the saved few-shot control (cell 82a).
    fs = {0: acc.get("C1"), 2: None, 4: None}
    try:
        if has_artifact("wo_fewshot_control", "json"):
            for r in load_json("wo_fewshot_control").get("rows", []):
                if r.get("tag") == tag and int(r.get("shots", -1)) in (0, 2, 4):
                    fs[int(r["shots"])] = float(r["c1_acc"])
    except Exception:
        pass
    verdict = wo_replication_verdict(acc, fs.get(4))
    return {
        "tag": tag, "model": WO_MODEL_REGISTRY.get(tag, tag),
        "status": "reference (prior WO run)",
        "format": globals().get("WO_INSTRUCT_FORMAT", "bare-continuation") if tag == "instruct"
        else "bare-continuation",
        "parity_ok": True,
        "acc": {k: acc.get(k) for k in ["C0", "C1", "C4", "C6", "C7", "C8", "A1", "A2", "D1"]},
        "fewshot_c1": {str(s): fs.get(s) for s in WO_XM_FEWSHOT_SHOTS},
        "verdict": verdict, "label": verdict["label"], "n": len(WO_PAIRS),
    }


# --- run the cross-model set + assemble the table ----------------------------
WO_XM_RESULTS = {}
for _tag in CFG["wo_crossmodel_tags"]:
    WO_XM_RESULTS[_tag] = _xm_run_model(_tag)

# reference rows first (base, instruct), then the new models in CFG order.
_xm_rows_struct = []
for _ref in ("base", "instruct"):
    _rr = _xm_reference_row(_ref)
    if _rr is not None:
        _xm_rows_struct.append(_rr)
for _tag in CFG["wo_crossmodel_tags"]:
    _xm_rows_struct.append(WO_XM_RESULTS[_tag])


def _g(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


_XM_HEADER = ["tag", "model", "status", "format", "parity_ok",
              "C0", "C1", "C4", "C6", "C7", "C8", "A1", "A2", "D1",
              "fs_C1_0", "fs_C1_2", "fs_C1_4",
              "parts_work", "compose_collapses", "operation_specific", "depth_sensitive",
              "fewshot_recovers", "replicates_core", "replicates_full", "out_of_scope",
              "label"]
_xm_csv_rows = []
for r in _xm_rows_struct:
    v = r.get("verdict", {})
    _xm_csv_rows.append({
        "tag": r.get("tag"), "model": r.get("model"), "status": r.get("status"),
        "format": (r.get("format") or "").split(" (")[0], "parity_ok": r.get("parity_ok"),
        "C0": _g(r, "acc", "C0"), "C1": _g(r, "acc", "C1"), "C4": _g(r, "acc", "C4"),
        "C6": _g(r, "acc", "C6"), "C7": _g(r, "acc", "C7"), "C8": _g(r, "acc", "C8"),
        "A1": _g(r, "acc", "A1"), "A2": _g(r, "acc", "A2"), "D1": _g(r, "acc", "D1"),
        "fs_C1_0": _g(r, "fewshot_c1", "0"), "fs_C1_2": _g(r, "fewshot_c1", "2"),
        "fs_C1_4": _g(r, "fewshot_c1", "4"),
        "parts_work": v.get("parts_work"), "compose_collapses": v.get("compose_collapses"),
        "operation_specific": v.get("operation_specific"), "depth_sensitive": v.get("depth_sensitive"),
        "fewshot_recovers": v.get("fewshot_recovers"), "replicates_core": v.get("replicates_core"),
        "replicates_full": v.get("replicates_full"), "out_of_scope": v.get("out_of_scope"),
        "label": r.get("label"),
    })

wo_save_result("cross_model_battery.csv", wo_battery_csv(_xm_csv_rows, _XM_HEADER))
save_json("wo_crossmodel_table", {"rows": _xm_csv_rows, "header": _XM_HEADER,
                                  "thresholds": WO_REPL_THR, "shared_pairs_sha": WO_PAIRS_HASH})

# --- printed table + honest summary ------------------------------------------
print("\n================= WO#4 §3.1 — CROSS-MODEL REPLICATION (shared WO_PAIRS) =================")
print(f"{'tag':<14}{'fmt':<6}{'par':>4}{'C1':>7}{'C4':>7}{'C6':>7}{'A1':>7}{'D1':>7}"
      f"{'fs@4':>7}  label")
for r in _xm_csv_rows:
    def s(x):
        return " n/a" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    _fmt = (r["format"] or "")[:5]
    _par = "?" if r["parity_ok"] is None else ("ok" if r["parity_ok"] else "BAD")
    print(f"{str(r['tag']):<14}{_fmt:<6}{_par:>4}{s(r['C1']):>7}{s(r['C4']):>7}{s(r['C6']):>7}"
          f"{s(r['A1']):>7}{s(r['D1']):>7}{s(r['fs_C1_4']):>7}  {r['label']}")
print("-----------------------------------------------------------------------------------------")
_repl_full = [r["tag"] for r in _xm_csv_rows if r["replicates_full"]]
_repl_core = [r["tag"] for r in _xm_csv_rows if r["replicates_core"] and not r["replicates_full"]]
_nonrepl = [r["tag"] for r in _xm_csv_rows
            if r["status"] in ("ok", "reference (prior WO run)")
            and not r["replicates_core"] and not r["out_of_scope"]]
_oos = [r["tag"] for r in _xm_csv_rows if r["out_of_scope"]]
_denied = [r["tag"] for r in _xm_csv_rows if r["status"] == "access_denied"]
_failed = [r["tag"] for r in _xm_csv_rows if r["status"] in ("load_failed", "error")]
print(f"  replicates_full : {_repl_full}")
print(f"  replicates_core : {_repl_core}")
print(f"  NON-replicators : {_nonrepl}   (reported, NOT cherry-picked)")
print(f"  out_of_scope    : {_oos}   (can't do C4/C6 — capability, reported separately)")
print(f"  ACCESS_DENIED   : {_denied}   (true gating — accept the license)")
print(f"  LOAD_FAILED     : {_failed}   (NOT gating — real load error, see below)")
# surface the actual error for any non-gating failure so it's never buried again.
for _tag in CFG.get("wo_crossmodel_tags", []):
    _r = WO_XM_RESULTS.get(_tag, {})
    if _r.get("status") in ("load_failed", "error"):
        print(f"     [{_tag}] {_r.get('error', '(no error captured)')}")
_n_added = sum(1 for r in _xm_csv_rows if r["tag"] in CFG["wo_crossmodel_tags"]
               and r["status"] == "ok")
print(f"  >>> {_n_added} additional model(s) batteried on the shared pairs "
      f"(acceptance: >= 3).")
print("=========================================================================================")
