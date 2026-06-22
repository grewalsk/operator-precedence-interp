# Phase 3 — Gate G3 (first real kill-switch): behavioral validation, forward-pass only.
# Proves the model COMPUTES (not looks up) the operator-precedence stimulus and engages
# the operand IN-BAND, before any expensive probing. Three checks:
#   (1) ACCURACY        — greedy-decode (0+B)*C, robustly parse the int, vs ground truth.
#   (2) NO-OP CHECK      — predictions TRACK B*C as B,C vary; "0 +" doesn't kill the mult.
#   (3) MUST-COMPUTE     — accuracy vs operand size: graceful DEGRADATION (compute), not
#                          pinned-at-100% (lookup), not flat, not collapsed. LOCK that band.
#
# RESILIENCE: every forward pass is batched and guarded by has_artifact(...). A GPU
# disconnect mid-run resumes from cached eval results; re-running top-to-bottom recomputes
# nothing already on disk. Relies ONLY on model/tokenizer/CFG/ART/helpers from earlier cells.
#
# ADVERSARIAL-REVIEW FIXES vs prior draft:
#   * Padding-side bug: last real token is NOT mask.sum()-1 under left padding. We now
#     standardize tokenizer.padding_side='left' and read the TRUE last column, so the token
#     scored is exactly the token after "equals " regardless of pad side.
#   * Must-compute: 'graceful degradation' now requires a real accuracy DROP across the band,
#     so a flat (e.g. 50%) curve can no longer masquerade as computation.

import re, time
import numpy as np
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------
# 0) Knobs (recorded to CFG so the floor/band are auditable artifacts, not magic numbers).
# ----------------------------------------------------------------------------------------
CFG.setdefault("g3_accuracy_floor", 0.80)      # PASS band must clear this overall accuracy.
CFG.setdefault("g3_eval_batch_size", 64)        # forward-pass batch (cheap GPU step).
CFG.setdefault("g3_max_answer_tokens", 6)       # greedy-decode budget for the answer int.
CFG.setdefault("g3_noop_grid", 24)              # B,C values per axis for the no-op sweep.
CFG.setdefault("g3_per_bin_n", 96)              # stimuli sampled per operand-size bin.
# Operand-size bins (max operand magnitude). Phase 4-9 consume the LOCKED subset of these.
CFG.setdefault("g3_operand_bins", [(2, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 499)])
# "graceful degradation" band acceptance window (per-bin accuracy must sit inside this):
CFG.setdefault("g3_band_lo", 0.30)              # below this -> collapsed to chance (useless).
CFG.setdefault("g3_band_hi", 0.985)             # at/above this -> looks like memorized lookup.
CFG.setdefault("g3_min_band_drop", 0.15)        # locked band must DROP by >= this end-to-end
                                                #   (flat curve != computation -> reject).
ACC_FLOOR = float(CFG["g3_accuracy_floor"])
MIN_DROP  = float(CFG["g3_min_band_drop"])
seed = int(CFG.get("seed", 0))

# ----------------------------------------------------------------------------------------
# 1) set_gate fallback. Earlier cells normally define set_gate/GATES; tolerate their absence
#    so this cell is self-contained on a fresh kernel that only restored model/tokenizer.
# ----------------------------------------------------------------------------------------
if "set_gate" not in globals():
    def set_gate(name, passed, detail=""):
        gates = load_json("gates") if has_artifact("gates") else {}
        gates[name] = {"pass": bool(passed), "detail": str(detail), "ts": time.time()}
        save_json("gates", gates)
        log(f"[gate] {name} = {'PASS' if passed else 'FAIL'} :: {detail}")
        return gates[name]

# ----------------------------------------------------------------------------------------
# 2) PINNED CONTRACT: Phase 2 writes 'phase2_stimuli'; Phase 3 reads 'phase2_stimuli'.
#    Fail LOUDLY if absent -- Phase 3 deliberately does NOT regenerate a fallback bank,
#    because a *different* surface form would make a green G3 certify a stimulus the
#    experiment never actually runs on. Every record is the canonical PARENTHESIZED
#    SYMBOLIC surface ("( 0 + B ) * C =") -- the exact surface every downstream patch
#    indexes against.
# ----------------------------------------------------------------------------------------
if not has_artifact("phase2_stimuli", "json"):
    raise RuntimeError(
        "Phase 3 requires the Phase 2 artifact 'phase2_stimuli'. It is absent -- run the "
        "Phase 2 / G2 cell first. Phase 3 does NOT fabricate a fallback surface: a different "
        "surface form would make a green G3 certify a stimulus the experiment never runs on.")
_p2 = load_json("phase2_stimuli")
assert isinstance(_p2, list) and _p2, "phase2_stimuli must be a non-empty list of records."
STIM = []
for r in _p2:
    _missing = [k for k in ("prompt", "B", "C", "answer", "condition") if k not in r]
    assert not _missing, f"phase2_stimuli record missing keys {_missing}: {sorted(r)[:8]}"
    STIM.append({"prompt": str(r["prompt"]), "B": int(r["B"]), "C": int(r["C"]),
                 "answer": int(r["answer"]), "condition": r["condition"]})
src_name = "phase2_stimuli"
log(f"G3 operating on {len(STIM)} canonical Phase 2 stimuli (source='{src_name}'); "
    f"accuracy floor={ACC_FLOOR}")

# ----------------------------------------------------------------------------------------
# 2b) Canonical surface renderer -- byte-identical to Phase 2's phase2_stimuli surface.
#     CHECK 1 reads stored prompts directly; CHECK 2 (no-op grid) and CHECK 3 (must-compute
#     bins) need FRESH (B,C) NOT present in the artifact, so they render here. Using THIS
#     renderer (not an English paraphrase like "B times C equals") is what guarantees all
#     three G3 checks exercise the experiment's exact '(' / '*' / '=' tokenization regime --
#     which IS the operator-precedence signal under study. We verify it against a real
#     phase2_stimuli record so it can never silently drift from Phase 2's _segments.
# ----------------------------------------------------------------------------------------
def _render_canonical(B, C, condition="depth_left"):
    B, C = int(B), int(C)
    if condition == "depth_left":   return f"( 0 + {B} ) * {C} ="   # (0+B)*C = B*C  (additive identity)
    if condition == "depth_right":  return f"0 + ( {B} * {C} ) ="   # 0+(B*C) = B*C
    if condition == "bare":         return f"{B} * {C} ="           # bare-mult control (no identity)
    raise ValueError(f"unknown condition {condition!r}")

for _cond in ("depth_left", "depth_right"):
    _ex = next((r for r in STIM if r["condition"] == _cond), None)
    if _ex is not None:
        _r = _render_canonical(_ex["B"], _ex["C"], _cond)
        assert _r == _ex["prompt"], (
            f"Phase 3 surface renderer drifted from Phase 2 for {_cond}: {_r!r} != stored "
            f"{_ex['prompt']!r}. Re-sync _render_canonical with Phase 2's _segments/_assemble.")
log("Phase 3: canonical surface renderer verified against phase2_stimuli (no drift).")

# ----------------------------------------------------------------------------------------
# 3) Robust greedy decode + integer parser. Works for transformer_lens HookedTransformer
#    (model(tokens) -> logits [B,T,V]) AND an HF-style fallback wrapper (-> .logits).
#    Batched, deterministic, no sampling. Parses multi-token numbers / leading-space tokens.
#
#    PADDING SAFETY (key fix): we force LEFT padding. With left padding the real tokens of
#    every row end at the final column, so the token to score (the one after "equals ") is
#    ALWAYS at index T-1 -- no mask-sum arithmetic, no left/right ambiguity. Newly generated
#    tokens are appended on the right, so after k steps the just-produced token is again the
#    last column. We still gather per-row by the true last attended index to be airtight even
#    if a model/tokenizer ignores our padding_side request.
# ----------------------------------------------------------------------------------------
_DEVICE = CFG.get("device", "cuda")

# Force a deterministic, generation-correct padding side once.
try:
    tokenizer.padding_side = "left"
except Exception as _e:
    log(f"(could not set tokenizer.padding_side='left': {_e})")

def _to_logits(out):
    return out.logits if hasattr(out, "logits") else out

@torch.no_grad()
def _encode(prompts):
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(_DEVICE)
    mask = enc.get("attention_mask")
    mask = None if mask is None else mask.to(_DEVICE)
    return ids, mask

@torch.no_grad()
def _safe_forward(ids, mask):
    """Forward that tolerates models whose __call__ doesn't accept attention_mask."""
    try:
        return model(ids, attention_mask=mask)
    except TypeError:
        return model(ids)

def _last_real_index(mask):
    """True index of the last attended (real) token per row, correct for BOTH pad sides.
    We take the position of the last '1' in the attention mask: argmax over reversed mask."""
    T = mask.shape[1]
    # index of last nonzero per row = T-1 - (#trailing zeros). Use flip+argmax of the
    # boolean mask to find the first real token from the right.
    flipped = torch.flip(mask, dims=[1])
    # argmax returns the FIRST max (first real token scanning from the right end)
    first_from_right = torch.argmax((flipped > 0).int(), dim=1)
    return (T - 1) - first_from_right

@torch.no_grad()
def _greedy_continuations(prompts, max_new=None):
    """Greedy-decode `max_new` tokens per prompt. Returns list[str] of generated text only
    (prompt stripped). Padding-side agnostic: score each row at its TRUE last real index."""
    max_new = max_new or CFG["g3_max_answer_tokens"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        try: tokenizer.pad_token = tokenizer.eos_token
        except Exception: pass
    ids, mask = _encode(prompts)
    if mask is None:
        mask = torch.ones_like(ids)
    n = ids.shape[0]
    gen = [[] for _ in range(n)]
    eos_id = tokenizer.eos_token_id
    done = torch.zeros(n, dtype=torch.bool, device=ids.device)
    row = torch.arange(n, device=ids.device)
    for _ in range(max_new):
        out = _safe_forward(ids, mask)
        logits = _to_logits(out)
        last_idx = _last_real_index(mask)                 # TRUE last real position per row
        nxt = logits[row, last_idx, :].argmax(dim=-1)     # token after "equals " (then after each gen)
        for i in range(n):
            if not done[i]:
                tid = int(nxt[i].item())
                if eos_id is not None and tid == eos_id:
                    done[i] = True
                else:
                    gen[i].append(tid)
        if done.all():
            break
        # Append on the right; with left padding this keeps generated tokens contiguous at
        # the end, and _last_real_index still resolves the correct column on the next step.
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        mask = torch.cat([mask, (~done).long().unsqueeze(1)], dim=1)
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]

_NUM_RE = re.compile(r"-?\d[\d,]*")
def parse_int(text):
    """Pull the FIRST integer out of a greedy continuation; handle leading spaces, commas
    (1,234), and multi-token splits (already merged by decode()). None if no digits."""
    if text is None:
        return None
    m = _NUM_RE.search(text.strip())
    if not m:
        return None
    s = m.group(0).replace(",", "").rstrip("-")  # guard against stray trailing punctuation
    if s in ("", "-"):
        return None
    try:
        return int(s)
    except ValueError:
        return None

def _parse_to_nan(text):
    v = parse_int(text)
    return float(v) if v is not None else np.nan

# ----------------------------------------------------------------------------------------
# 4) Generic guarded batched evaluator. Caches predictions per artifact key so a disconnect
#    resumes exactly where it stopped (batch-level checkpointing inside the artifact).
#    Prompts for each key are regenerated deterministically (fixed seeds) so a cached
#    prefix stays aligned with the current prompt list across reruns.
# ----------------------------------------------------------------------------------------
def _eval_prompts(prompts, cache_key):
    """Greedy-decode every prompt, return list[str] continuations. Resumable: persists a
    growing list and only decodes the not-yet-done tail."""
    if has_artifact(cache_key):
        cont = load_json(cache_key)
        if len(cont) >= len(prompts):
            log(f"[{cache_key}] cached ({len(cont)} preds) — skipping forward passes.")
            return cont[:len(prompts)]
        # cached prefix shorter than needed -> resume from where it stopped.
    else:
        cont = []
    bs = int(CFG["g3_eval_batch_size"])
    start = len(cont)
    log(f"[{cache_key}] decoding {len(prompts)-start} / {len(prompts)} prompts (resume @ {start})")
    for i in range(start, len(prompts), bs):
        chunk = prompts[i:i + bs]
        cont.extend(_greedy_continuations(chunk))
        save_json(cache_key, cont)   # checkpoint after every batch -> disconnect-safe.
    return cont[:len(prompts)]

# ----------------------------------------------------------------------------------------
# CHECK 1 — ACCURACY on the experimental (0+B)*C stimuli.
# ----------------------------------------------------------------------------------------
if has_artifact("g3_accuracy_result"):
    acc_res = load_json("g3_accuracy_result")
else:
    prompts = [s["prompt"] for s in STIM]
    conts = _eval_prompts(prompts, "g3_pred_experimental")
    preds = [parse_int(c) for c in conts]
    correct = [int(p is not None and p == s["answer"]) for p, s in zip(preds, STIM)]
    overall = float(np.mean(correct)) if correct else 0.0
    parsed_rate = float(np.mean([p is not None for p in preds])) if preds else 0.0
    acc_res = {"overall_accuracy": overall, "parsed_rate": parsed_rate,
               "n": len(STIM), "floor": ACC_FLOOR,
               "examples": [{"prompt": s["prompt"], "pred": p, "gold": s["answer"]}
                            for s, p in list(zip(STIM, preds))[:8]]}
    save_json("g3_accuracy_result", acc_res)
log(f"CHECK1 ACCURACY: overall={acc_res['overall_accuracy']:.3f} "
    f"(parsed_rate={acc_res['parsed_rate']:.3f}, n={acc_res['n']}, floor={ACC_FLOOR})")

# ----------------------------------------------------------------------------------------
# CHECK 2 — NO-OP CHECK (guards the additive-identity correction).
#   (a) Hold structure fixed, sweep B,C on a grid; confirm prediction TRACKS B*C.
#   (b) Confirm "0 +" prefix does NOT make the model ignore the multiplication: compare
#       the canonical "( 0 + B ) * C =" surface against a bare "B * C =" surface (SAME
#       symbolic regime) on the same (B,C) grid; both must track B*C and agree, i.e. the
#       additive identity is a true no-op.
# ----------------------------------------------------------------------------------------
if has_artifact("g3_noop_result"):
    noop_res = load_json("g3_noop_result")
else:
    rng = np.random.default_rng(seed + 7)
    g = int(CFG["g3_noop_grid"])
    lo, hi = 2, 99                      # mid-range operands so signal isn't size-limited.
    Bs = sorted(set(int(x) for x in rng.integers(lo, hi + 1, size=g)))
    Cs = sorted(set(int(x) for x in rng.integers(lo, hi + 1, size=g)))
    pairs = [(B, C) for B in Bs for C in Cs]
    bc = np.array([B * C for (B, C) in pairs], dtype=float)

    # (a) experimental surface: canonical "( 0 + B ) * C ="  (the additive-identity surface
    #     the experiment ACTUALLY runs on -- same '(' / '*' / '=' tokenization regime).
    p_exp = [_render_canonical(B, C, "depth_left") for (B, C) in pairs]
    c_exp = _eval_prompts(p_exp, "g3_pred_noop_exp")
    y_exp = np.array([_parse_to_nan(t) for t in c_exp])

    # (b) bare-multiplication control: canonical "B * C ="  (no additive identity, SAME
    #     symbolic regime). If "( 0 + B )" is a true no-op, (a) and (b) must agree per pair.
    p_bare = [_render_canonical(B, C, "bare") for (B, C) in pairs]
    c_bare = _eval_prompts(p_bare, "g3_pred_noop_bare")
    y_bare = np.array([_parse_to_nan(t) for t in c_bare])

    def _track_stats(y):
        ok = np.isfinite(y)
        n_ok = int(ok.sum())
        if n_ok < 3:
            return {"corr_with_BC": None, "exact_match_rate": 0.0, "n_parsed": n_ok}
        corr = float(np.corrcoef(y[ok], bc[ok])[0, 1])
        exact = float(np.mean(y[ok] == bc[ok]))
        return {"corr_with_BC": corr, "exact_match_rate": exact, "n_parsed": n_ok}

    s_exp, s_bare = _track_stats(y_exp), _track_stats(y_bare)
    # additive-identity equivalence: do (0+B)*C and B*C give the same answer per pair?
    both = np.isfinite(y_exp) & np.isfinite(y_bare)
    agree_rate = float(np.mean(y_exp[both] == y_bare[both])) if both.sum() else 0.0
    noop_res = {"n_pairs": len(pairs), "exp": s_exp, "bare": s_bare,
                "additive_identity_agree_rate": agree_rate,
                "Bs": Bs, "Cs": Cs}
    save_json("g3_noop_result", noop_res)
log(f"CHECK2 NO-OP: exp corr(B*C)={noop_res['exp']['corr_with_BC']}, "
    f"bare corr(B*C)={noop_res['bare']['corr_with_BC']}, "
    f"additive-identity agree={noop_res['additive_identity_agree_rate']:.3f}")

# ----------------------------------------------------------------------------------------
# CHECK 3 — MUST-COMPUTE (lookup vs computation): accuracy as a function of operand size.
#   Sample per-bin stimuli (reusing the experimental surface), compute per-bin accuracy,
#   and LOCK the contiguous band that shows graceful DEGRADATION:
#     - every bin in band inside [band_lo, band_hi)   (not chance, not lookup),
#     - accuracy generally non-increasing across the run, AND
#     - a real end-to-end DROP (first - last >= g3_min_band_drop) so a FLAT curve cannot
#       masquerade as 'computation'.
# ----------------------------------------------------------------------------------------
if has_artifact("g3_operand_curve"):
    curve = load_json("g3_operand_curve")
else:
    rng = np.random.default_rng(seed + 13)
    bins = [tuple(b) for b in CFG["g3_operand_bins"]]
    per = int(CFG["g3_per_bin_n"])
    rows = []
    for bi, (blo, bhi) in enumerate(bins):
        # sample fresh controlled stimuli for this bin (cached via per-bin pred artifact)
        Bs = rng.integers(blo, bhi + 1, size=per).tolist()
        Cs = rng.integers(blo, bhi + 1, size=per).tolist()
        prompts = [_render_canonical(int(B), int(C), "depth_left") for B, C in zip(Bs, Cs)]
        golds = [int(B) * int(C) for B, C in zip(Bs, Cs)]
        conts = _eval_prompts(prompts, f"g3_pred_bin_{bi}")
        preds = [parse_int(c) for c in conts]
        corr = [int(p is not None and p == g) for p, g in zip(preds, golds)]
        acc = float(np.mean(corr)) if corr else 0.0
        rows.append({"bin": bi, "lo": blo, "hi": bhi,
                     "max_operand": bhi, "accuracy": acc, "n": per,
                     "parsed_rate": float(np.mean([p is not None for p in preds]))})
        log(f"  bin {bi} operands[{blo},{bhi}] acc={acc:.3f}")
    curve = {"bins": rows}
    save_json("g3_operand_curve", curve)

# --- LOCK the band: longest contiguous run of bins with band_lo <= acc < band_hi, then
#     require a genuine downward trend AND a real drop across that run. ---
b_lo, b_hi = float(CFG["g3_band_lo"]), float(CFG["g3_band_hi"])
accs = [r["accuracy"] for r in curve["bins"]]
in_band = [(b_lo <= a < b_hi) for a in accs]
# find longest contiguous True run
best = (0, -1, -1)  # (length, start, end)
i = 0
while i < len(in_band):
    if in_band[i]:
        j = i
        while j + 1 < len(in_band) and in_band[j + 1]:
            j += 1
        if (j - i + 1) > best[0]:
            best = (j - i + 1, i, j)
        i = j + 1
    else:
        i += 1
_, lstart, lend = best
locked = None
all_lookup = all(a >= b_hi for a in accs) if accs else False   # flat-100% memorization signal
all_chance = all(a < b_lo for a in accs) if accs else False
if best[0] >= 1:
    lo_operand = curve["bins"][lstart]["lo"]
    hi_operand = curve["bins"][lend]["hi"]
    band_accs = accs[lstart:lend + 1]
    # non-increasing (allow tiny noise) ...
    non_increasing = all(band_accs[k] >= band_accs[k + 1] - 0.05
                         for k in range(len(band_accs) - 1))
    # ... AND a real end-to-end drop so a FLAT band is NOT accepted as computation.
    end_to_end_drop = (band_accs[0] - band_accs[-1]) if len(band_accs) >= 2 else 0.0
    degrading = bool(non_increasing and end_to_end_drop >= MIN_DROP)
    locked = {"operand_lo": int(lo_operand), "operand_hi": int(hi_operand),
              "bin_start": int(lstart), "bin_end": int(lend),
              "bin_accuracies": [float(a) for a in band_accs],
              "non_increasing": bool(non_increasing),
              "end_to_end_drop": float(end_to_end_drop),
              "min_required_drop": MIN_DROP,
              "graceful_degradation": degrading,
              "band_lo": b_lo, "band_hi": b_hi,
              "accuracy_floor": ACC_FLOOR}

# ----------------------------------------------------------------------------------------
# 5) PLOT accuracy vs operand size (inline) + save the figure.
# ----------------------------------------------------------------------------------------
xs = [r["max_operand"] for r in curve["bins"]]
ys = [r["accuracy"] for r in curve["bins"]]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(xs, ys, "o-", color="#1f77b4", label="accuracy")
ax.axhline(b_lo, ls=":", c="grey", lw=1); ax.axhline(b_hi, ls=":", c="grey", lw=1)
ax.axhline(ACC_FLOOR, ls="--", c="green", lw=1, label=f"floor={ACC_FLOOR}")
if locked is not None:
    ax.axvspan(curve["bins"][locked["bin_start"]]["lo"],
               curve["bins"][locked["bin_end"]]["hi"],
               color="orange", alpha=0.15, label="locked band")
ax.set_xscale("log"); ax.set_xlabel("max operand magnitude (log)")
ax.set_ylabel("greedy-decode accuracy"); ax.set_ylim(-0.02, 1.02)
ax.set_title("Phase 3 / G3 — accuracy vs operand size (must-compute)")
ax.legend(loc="best", fontsize=8); fig.tight_layout()
try:
    fig.savefig(str(ART / "g3_operand_curve.png"), dpi=120)
except Exception as e:
    log(f"(plot save skipped: {e})")
plt.show()

# ----------------------------------------------------------------------------------------
# 6) GATE G3 verdict. PASS requires all three checks:
#    (1) overall accuracy >= floor (compute happens at all),
#    (2) no-op: predictions track B*C AND additive identity is a genuine no-op,
#    (3) must-compute: a graceful-DEGRADATION band exists (real drop) -> LOCK + save spec.
# ----------------------------------------------------------------------------------------
c1 = acc_res["overall_accuracy"] >= ACC_FLOOR

exp_corr = noop_res["exp"]["corr_with_BC"]
c2_track = (exp_corr is not None and exp_corr >= 0.90 and
            noop_res["exp"]["exact_match_rate"] >= 0.50)
c2_noop = noop_res["additive_identity_agree_rate"] >= 0.90    # "0 +" is a true no-op.
c2 = bool(c2_track and c2_noop)

c3 = bool(locked is not None and locked.get("graceful_degradation", False))

g3_pass = bool(c1 and c2 and c3)

if g3_pass and locked is not None:
    # Save the SINGLE locked-band spec that Phases 4-9 consume.
    band_spec = {"operand_lo": locked["operand_lo"], "operand_hi": locked["operand_hi"],
                 "accuracy_floor": ACC_FLOOR, "band_lo": b_lo, "band_hi": b_hi,
                 "bin_accuracies": locked["bin_accuracies"],
                 "end_to_end_drop": locked["end_to_end_drop"],
                 "source_stimuli": src_name, "seed": seed,
                 "overall_accuracy": acc_res["overall_accuracy"]}
    save_json("locked_band_spec", band_spec)
    CFG["locked_operand_lo"] = locked["operand_lo"]
    CFG["locked_operand_hi"] = locked["operand_hi"]
    log(f"LOCKED BAND saved: operands [{locked['operand_lo']}, {locked['operand_hi']}] "
        f"(drop={locked['end_to_end_drop']:.2f}) -> artifact 'locked_band_spec' (Phases 4-9).")

detail = (f"acc={acc_res['overall_accuracy']:.3f}/floor={ACC_FLOOR} (C1={'P' if c1 else 'F'}); "
          f"no-op exp_corr={exp_corr}, agree={noop_res['additive_identity_agree_rate']:.2f} "
          f"(C2={'P' if c2 else 'F'}); "
          f"locked={None if locked is None else (locked['operand_lo'], locked['operand_hi'])}, "
          f"drop={None if locked is None else round(locked['end_to_end_drop'],3)}, "
          f"graceful={None if locked is None else locked['graceful_degradation']} "
          f"(C3={'P' if c3 else 'F'})")
set_gate("G3", g3_pass, detail)

print("\n================= PHASE 3 / GATE G3 =================")
print(f"CHECK 1 ACCURACY     : {'PASS' if c1 else 'FAIL'}  "
      f"(overall={acc_res['overall_accuracy']:.3f} >= floor {ACC_FLOOR}?)")
print(f"CHECK 2 NO-OP        : {'PASS' if c2 else 'FAIL'}  "
      f"(track B*C corr={exp_corr}, exact={noop_res['exp']['exact_match_rate']:.2f}; "
      f"additive-identity no-op agree={noop_res['additive_identity_agree_rate']:.2f})")
print(f"CHECK 3 MUST-COMPUTE : {'PASS' if c3 else 'FAIL'}  "
      f"(graceful-degradation band={'none' if locked is None else (locked['operand_lo'], locked['operand_hi'])}"
      f"{'' if locked is None else f", drop={locked['end_to_end_drop']:.2f}>={MIN_DROP}"})")
print("-----------------------------------------------------")
print("per-bin accuracy vs operand size:")
for r in curve["bins"]:
    mark = ""
    if locked is not None and locked["bin_start"] <= r["bin"] <= locked["bin_end"]:
        mark = "  <-- LOCKED"
    print(f"   operands[{r['lo']:>3},{r['hi']:>3}]  acc={r['accuracy']:.3f}  n={r['n']}{mark}")
print("-----------------------------------------------------")
print(f"GATE G3: {'PASS' if g3_pass else 'FAIL'}")
if not g3_pass:
    print("\nFAIL GUIDANCE:")
    if not c1:
        print(" - Accuracy below floor. The base model may not do multi-digit arithmetic")
        print("   greedily. Try: (a) the -Instruct model, (b) few-shot prompting, then RE-RUN")
        print("   the Phase 2 token-/answer-control gate (G2) on the new surface form.")
    if not c2:
        if not c2_track:
            print(" - Predictions don't track B*C on the canonical '( 0 + B ) * C =' surface.")
            print("   The base model may not greedily evaluate the symbolic '*'/precedence form")
            print("   (this is itself a finding). Try -Instruct / few-shot, then RE-RUN G2 on the")
            print("   new surface so Phase 2 and Phase 3 stay on the SAME stimulus.")
        if not c2_noop:
            print(" - '0 +' is NOT a no-op (additive-identity surface disagrees with bare B*C).")
            print("   The additive-identity correction is unjustified on this model — reconsider.")
    if not c3:
        if all_lookup:
            print(" - Accuracy pinned at/above band_hi in EVERY bin -> looks like memorized")
            print("   lookup. GROW the range (add larger operand bins) until it degrades.")
        elif all_chance:
            print(" - Accuracy below band_lo everywhere -> collapsed to chance (no computation).")
            print("   SHRINK the range (lower the upper operand bins) until signal appears.")
        elif locked is not None and not locked["graceful_degradation"]:
            print(f" - A band sits in [{b_lo},{b_hi}) but is FLAT (end-to-end drop "
                  f"{locked['end_to_end_drop']:.2f} < required {MIN_DROP}); flat != computation.")
            print("   Widen the operand-size span so real degradation shows across bins.")
        else:
            print(" - No contiguous graceful-degradation band. Adjust g3_operand_bins span.")
print("=====================================================")