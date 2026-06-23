# ============================================================================
# Phase 6 / WO — STEP 5b : produce the SELECTED branch's first downstream artifact.
# ----------------------------------------------------------------------------
# Dispatch on WO_BRANCH:
#   CLEAN_REPAIR / PARTIAL_REPAIR -> §9 localization of the C1/C2 contrast.
#   NO_REPAIR                     -> §9.B Branch-B controls + §10.B C6->C1 salvage.
#
# LOCALIZATION DESIGN (the metric subtlety made explicit, work order §10):
#   C1 and C2 target the SAME product B*C, so a clean-vs-corrupted *answer* logit-
#   diff between the two parses is degenerate, AND on a REPAIRED model both parses
#   already succeed (no failure to recover). The metric-valid instrument is the G4
#   idiom applied WITHIN each parse via OPERAND corruption:
#     clean  = "( 0 + B ) * C ="   (-> product P  = B*C)
#     corrupt= "( 0 + B ) * C' ="  (-> product P' = B*C', C' same #digits as C)
#   Patch CLEAN resid_post into the CORRUPTED run at each (layer,pos); score
#   recovery of the FIRST answer token (logit-diff P-first vs P'-first at the final
#   position; §10). This yields a per-parse answer-aggregation map. The C1-vs-C2
#   comparison is differenced ONLY over TOKEN-ALIGNED positions (the shared
#   '( 0 + B' prefix and the trailing '='); the divergent middle ( ')' '*' C  vs
#   '*' C ')' ) holds different tokens in the two parses and is EXCLUDED from the
#   difference peak (subtracting non-corresponding columns would be meaningless).
#   The instrument is VERIFIED on a C0 control first (§10): it must reproduce the
#   Stolfo mid-late final-token localization before C1/C2 is trusted.
# ============================================================================
import numpy as np
import torch

assert "WO_BRANCH" in globals(), "Run the branch cell (81) first."
CFG.setdefault("wo_localize_sample", 8)     # #examples averaged per parse (GPU cost knob).
CFG.setdefault("wo_localize_seed", 101)
CFG.setdefault("wo_salvage_n", 256)         # WO#2 §3.3: enlarged subset (was 128); decodability
#                                             needs n >> reduced-dim and the exclusion fix frees it.
CFG.setdefault("wo_salvage_min_n", 80)              # WO#2 §5.4: warn if usable contrast n < this.
CFG.setdefault("wo_salvage_recovery_thresh", 0.5)   # recovery >= this => a ')'-site patch demonstrably
#                                                     moves the output (used for pos-control + "used").


# ----------------------------------------------------------------------------
# Shared patching primitives (mirror the validated G4 instrument, cell 75).
# ----------------------------------------------------------------------------
def _wo_first_tok_logit(logits, pos, tok_id):
    return float(logits[0, pos, tok_id].item())


@torch.no_grad()
def _wo_empirical_first_tok(tokens):
    """Top-1 next-token id at the final position (the model's first emitted answer
    token). We score only the FIRST answer token (§10 default; products are
    multi-token)."""
    logits = model(tokens)
    return int(logits[0, -1].argmax().item())


def _wo_corrupt_C(C, rng, lo=20, hi=49):
    """A different operand C' with the SAME digit count as C (keeps token length
    equal so positions align for patching)."""
    nd = len(str(C))
    for _ in range(64):
        Cp = int(rng.integers(lo, hi + 1))
        if Cp != C and len(str(Cp)) == nd:
            return Cp
    return None


@torch.no_grad()
def _wo_localize_parse(render, B, C, Cp):
    """G4-style operand-corruption patch for ONE (B,C,C') on the current model.
    Returns (recovery[n_layers, seq_len], seq_len, pos_labels) or None if unusable."""
    clean_tokens = model.to_tokens(render(B, C))
    corrupt_tokens = model.to_tokens(render(B, Cp))
    if clean_tokens.shape != corrupt_tokens.shape:
        return None
    n_layers = model.cfg.n_layers
    seq_len = clean_tokens.shape[1]
    final_pos = seq_len - 1

    clean_first = _wo_empirical_first_tok(clean_tokens)
    corrupt_first = _wo_empirical_first_tok(corrupt_tokens)
    if clean_first == corrupt_first:
        return None   # products' first tokens coincide -> metric degenerate, skip.

    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens, names_filter=lambda n: n.endswith("hook_resid_post"))
    corrupt_logits = model(corrupt_tokens)
    def _ld(logits):
        return (_wo_first_tok_logit(logits, final_pos, clean_first)
                - _wo_first_tok_logit(logits, final_pos, corrupt_first))
    clean_ld, corrupt_ld = _ld(clean_logits), _ld(corrupt_logits)
    if not (clean_ld > corrupt_ld):
        return None   # metric sign wrong for this example -> skip.
    denom = (clean_ld - corrupt_ld) + 1e-8
    clean_resid = [clean_cache[f"blocks.{L}.hook_resid_post"][0] for L in range(n_layers)]

    def _mk_hook(layer_resid, pos):
        def hook(resid_post, hook):
            resid_post[:, pos, :] = layer_resid[pos, :].to(resid_post.dtype)
            return resid_post
        return hook

    rec = np.zeros((n_layers, seq_len), dtype=np.float64)
    for L in range(n_layers):
        for pos in range(seq_len):
            patched = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(f"blocks.{L}.hook_resid_post", _mk_hook(clean_resid[L], pos))])
            rec[L, pos] = (_ld(patched) - corrupt_ld) / denom
    pos_labels = [tokenizer.decode([t]).replace("\n", "\\n") for t in clean_tokens[0].tolist()]
    return rec, seq_len, pos_labels


def _wo_localize_mean(render, label, tag, sample_pairs):
    """Average the operand-corruption recovery map over sample_pairs whose surfaces
    share the modal token length (so positions align). PER-EXAMPLE checkpointed
    (partial artifact keyed ck+'_partial') so a GPU disconnect resumes by skipping
    already-computed examples — mirroring cell 75's per-layer checkpoint idiom."""
    ck = f"wo_loc_{tag}_{label}"
    if has_artifact(ck, "json"):
        return load_json(ck)
    rng = np.random.default_rng(int(CFG["wo_localize_seed"]))
    cand = []
    for (B, C) in sample_pairs:
        Cp = _wo_corrupt_C(C, rng, *WO_BAND)
        if Cp is None:
            continue
        L = model.to_tokens(render(B, C)).shape[1]
        cand.append((B, C, Cp, L))
    if not cand:
        return None
    lengths = [c[3] for c in cand]
    modal = max(set(lengths), key=lengths.count)
    kept = [(B, C, Cp) for (B, C, Cp, L) in cand if L == modal]

    # resume from a partial checkpoint (per-example).
    pck = ck + "_partial"
    done, labels = {}, None
    if has_artifact(pck, "json"):
        p = load_json(pck)
        done = dict(p.get("maps", {}))
        labels = p.get("pos_labels")
        modal = int(p.get("seq_len", modal))
        log(f"  [{tag}/{label}] resuming: {len(done)} examples already done.")
    for (B, C, Cp) in kept:
        ekey = f"{B}_{C}_{Cp}"
        if ekey in done:
            continue
        out = _wo_localize_parse(render, B, C, Cp)
        if out is None:
            continue
        rec, seq_len, pos_labels = out
        if seq_len != modal:
            continue
        done[ekey] = rec.tolist()
        labels = pos_labels
        save_json(pck, {"maps": done, "pos_labels": labels, "seq_len": modal})  # checkpoint
        log(f"  [{tag}/{label}] localized {ekey} (seq_len={seq_len}); {len(done)}/{len(kept)}")
    if not done:
        return None
    mean_map = np.mean(np.stack([np.array(v) for v in done.values()], axis=0), axis=0)
    res = {"label": label, "tag": tag, "n_used": len(done), "used_pairs": list(done.keys()),
           "seq_len": modal, "pos_labels": labels, "recovery": mean_map.tolist()}
    save_json(ck, res)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(7, modal * 0.7), max(5, model.cfg.n_layers * 0.22)))
        im = ax.imshow(mean_map, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(modal)); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_xlabel("token position (clean->corrupt resid_post patch)"); ax.set_ylabel("layer")
        ax.set_title(f"WO localization — {tag}/{label} (operand-corruption recovery, n={len(done)})")
        fig.colorbar(im, ax=ax, label="recovery (0=corrupt,1=clean)")
        fig.tight_layout(); fig.savefig(str(WO_RESULTS / f"localization_{tag}_{label}.png"), dpi=130)
        plt.show()
    except Exception as e:
        log(f"(heatmap skipped: {e})")
    return res


def _wo_peak(res):
    m = np.array(res["recovery"]); L, P = np.unravel_index(int(np.argmax(m)), m.shape)
    return {"layer": int(L), "pos": int(P), "pos_token": res["pos_labels"][P],
            "recovery": float(m[L, P])}


def _wo_run_localization(tags):
    """§9: localize C1 and C2 per tag, verify on a C0 control, write
    localization_sites.csv + the TOKEN-ALIGNED difference (precedence) peak."""
    renderC0 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C0"]
    renderC1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]
    renderC2 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C2"]
    n = int(CFG["wo_localize_sample"])
    rng = np.random.default_rng(int(CFG["wo_localize_seed"]) + 1)
    sample = [WO_PAIRS[i] for i in rng.choice(len(WO_PAIRS), size=min(n, len(WO_PAIRS)),
                                              replace=False)]
    site_rows = []
    for tag in tags:
        wo_load_model(tag)
        # (§10) VERIFY the instrument on a C0 control before trusting C1/C2.
        ctrl = _wo_localize_mean(renderC0, "C0_control", tag, sample)
        ctrl_ok = False
        if ctrl is not None:
            pk = _wo_peak(ctrl)
            mid = int(np.floor(model.cfg.n_layers * CFG.get("g4_midlate_band_frac", 0.40)))
            ctrl_ok = (pk["recovery"] >= 0.50 and pk["layer"] >= mid
                       and pk["pos"] >= ctrl["seq_len"] - 2)
            log(f"WO localization control [{tag}/C0]: peak rec={pk['recovery']:.2f} @ "
                f"layer {pk['layer']} pos {pk['pos']} -> instrument {'OK' if ctrl_ok else 'SUSPECT'}")
            site_rows.append({"tag": tag, "parse": "C0_control", **pk, "instrument_ok": ctrl_ok})
        if not ctrl_ok:
            log(f"WO localization [{tag}]: C0 control did NOT reproduce Stolfo localization "
                f"-> C1/C2 sites are reported but flagged SUSPECT (do not over-trust).")
        for key, render in (("C1", renderC1), ("C2", renderC2)):
            res = _wo_localize_mean(render, key, tag, sample)
            if res is None:
                log(f"WO localization [{tag}/{key}]: no usable examples.")
                continue
            pk = _wo_peak(res)
            site_rows.append({"tag": tag, "parse": key, **pk, "instrument_ok": ctrl_ok})
        # difference map (precedence-resolution signal) — TOKEN-ALIGNED positions ONLY.
        c1 = load_json(f"wo_loc_{tag}_C1") if has_artifact(f"wo_loc_{tag}_C1", "json") else None
        c2 = load_json(f"wo_loc_{tag}_C2") if has_artifact(f"wo_loc_{tag}_C2", "json") else None
        if c1 and c2 and c1["seq_len"] == c2["seq_len"]:
            l1, l2 = c1["pos_labels"], c2["pos_labels"]
            aligned = [i for i in range(c1["seq_len"]) if l1[i] == l2[i]]
            nonaligned = [i for i in range(c1["seq_len"]) if i not in aligned]
            diff = np.array(c1["recovery"]) - np.array(c2["recovery"])
            if aligned:
                # argmax |diff| restricted to columns where BOTH parses hold the same token.
                masked = np.full(diff.shape, -np.inf)
                for i in aligned:
                    masked[:, i] = np.abs(diff[:, i])
                L, P = np.unravel_index(int(np.argmax(masked)), masked.shape)
                site_rows.append({"tag": tag, "parse": "C1_minus_C2_diff(aligned)", "layer": int(L),
                                  "pos": int(P), "pos_token": l1[P],  # identical in both at aligned P
                                  "recovery": float(diff[L, P]), "instrument_ok": ctrl_ok})
            log(f"WO localization diff [{tag}]: aligned token positions {aligned}; "
                f"non-aligned (parses differ — excluded from the diff peak) {nonaligned}.")
    csv = wo_battery_csv(
        site_rows, ["tag", "parse", "layer", "pos", "pos_token", "recovery", "instrument_ok"])
    wo_save_result("localization_sites.csv", csv)
    save_json("wo_localization_sites", site_rows)
    print("\n================= WO STEP 5b — LOCALIZATION SITES =================")
    for r in site_rows:
        print(f"  [{r['tag']:>8}/{r['parse']:<22}] peak |rec|={r['recovery']:+.2f} @ "
              f"layer {r['layer']:>2} pos {r['pos']:>2} ({r['pos_token']!r}) "
              f"instrument_ok={r['instrument_ok']}")
    print("==================================================================")


# ----------------------------------------------------------------------------
# NO_REPAIR path: §9.B controls + §10.B salvage.
# ----------------------------------------------------------------------------
def _wo_run_branchb(tag):
    """§9.B selectivity controls: addition-precedence analogue (A1/A2), depth
    control (D1), and operand-magnitude stratification of C1 vs C4."""
    wo_load_model(tag)
    res = wo_run_battery(tag, WO_BRANCHB_CONDITIONS, WO_PAIRS)
    # Expose the full result (incl. per-item 'correct_mask') in-memory so the §3.4
    # confidence-interval cell can build the A1-vs-C1 paired CI (saved JSON strips
    # correct_mask). Battery is cached, so this is index-aligned to WO_PAIRS.
    globals()[f"WO_BRANCHB_RES_{tag}"] = res
    rows = [{"cond": k, "name": res[k]["name"], "acc": res[k]["exact_acc"],
             "corr": res[k]["corr"], "parse_fail": res[k]["parse_fail_rate"]}
            for k in [c[0] for c in WO_BRANCHB_CONDITIONS]]
    add_compose_ok = (res["A1"]["exact_acc"] >= 0.80)
    rows.append({"cond": "SELECTIVITY", "name": "add_compose_works(A1>=.80)",
                 "acc": add_compose_ok, "corr": "", "parse_fail": ""})
    # operand-magnitude stratification: acc(C1) vs acc(C4) at matched |B*C| bins.
    src = WO_INSTRUCT_RES if tag == "instruct" else (WO_BASE_RES if "WO_BASE_RES" in globals() else None)
    if src is not None:
        # masks are index-aligned to WO_PAIRS (battery built from WO_PAIRS) — assert it.
        assert len(src["C1"]["correct_mask"]) == len(WO_PAIRS) and \
            len(src["C4"]["correct_mask"]) == len(WO_PAIRS), \
            "magnitude stratification: C1/C4 correct_mask not aligned to WO_PAIRS"
        bins = wo_operand_magnitude_bins(WO_PAIRS, n_bins=5)
        for b in bins:
            idx = b["idx"]
            if not idx:
                continue
            a1 = float(np.mean([src["C1"]["correct_mask"][j] for j in idx]))
            a4 = float(np.mean([src["C4"]["correct_mask"][j] for j in idx]))
            rows.append({"cond": f"MAGBIN[{b['lo']:.0f},{b['hi']:.0f})",
                         "name": f"C1 vs C4 @matched|B*C| (n={b['n']})",
                         "acc": f"C1={a1:.3f};C4={a4:.3f}", "corr": "", "parse_fail": ""})
    csv = wo_battery_csv(rows, ["cond", "name", "acc", "corr", "parse_fail"])
    wo_save_result("branchb_controls.csv", csv)
    save_json("wo_branchb_controls", {"battery": {k: {kk: vv for kk, vv in v.items()
              if kk not in ("correct_mask", "preds", "golds", "prompts")} for k, v in res.items()},
              "add_compose_works": add_compose_ok})
    print("\n================= WO STEP 5b — BRANCH-B CONTROLS =================")
    for r in rows:
        print(f"  {r['cond']:<18} {r['name']:<34} {r['acc']}")
    print("=================================================================")


def _wo_battery_res_for_salvage(tag):
    """In-memory battery result for `tag` (WO_INSTRUCT_RES / WO_BASE_RES), which
    carries per-item 'correct_mask' + 'preds' index-aligned to WO_PAIRS. Falls back
    to a cached C1-only re-run if it isn't in globals (e.g. a partial session). The
    fallback requires the `tag` model to be live (caller loads it first)."""
    g = globals().get("WO_INSTRUCT_RES" if tag == "instruct" else "WO_BASE_RES")
    if g is not None and "C1" in g and "correct_mask" in g["C1"]:
        return g
    log(f"WO salvage[{tag}]: battery not in memory — re-running C1 (cached, instant).")
    return wo_run_battery(tag, [c for c in WO_CONDITIONS if c[0] == "C1"], WO_PAIRS)


def _wo_mk_patch_hook(vec_dev, pos):
    """Factory: a resid_post forward hook that OVERWRITES position `pos` with the
    (already-on-device) donor vector `vec_dev`. Captures pos/vec by ARGUMENT (no
    loop late-binding) and is consumed synchronously by run_with_hooks (§6 gotcha)."""
    def hook(resid_post, hook):
        resid_post[:, pos, :] = vec_dev.to(resid_post.dtype)
        return resid_post
    return hook


def _wo_corrupt_B(B, rng, lo=20, hi=49):
    """A different operand B' with the SAME digit count as B (keeps token length
    equal so the ')' index aligns). Corrupting the OPERAND is what makes the ')'
    residual genuinely DIFFER between the clean and corrupt runs — the fix for the
    no-op transplant confound (copying an identical-prefix residual was an identity
    operation, so flip-rate ~ 0 was guaranteed regardless of composition)."""
    nd = len(str(B))
    for _ in range(64):
        Bp = int(rng.integers(lo, hi + 1))
        if Bp != B and len(str(Bp)) == nd:
            return Bp
    return None


def _wo_print_salvage(out):
    print("\n========== WO STEP 5b — DECODABLE-BUT-UNUSED salvage (site-matched) ==========")
    _warn = "" if out.get("n_used_ok", True) else f"  ⚠ n_used < {out.get('min_n')}"
    print(f"  [{out['tag']}] n_used={out['n_used']}/{out.get('n_sample_requested')} "
          f"(C1-wrong cand={out.get('n_wrong_candidates')}, no-B-contrast={out.get('n_no_contrast')}, "
          f"other skips={out['n_skipped']}){_warn}")
    print("  DECODABILITY @ post-bracket ')' site (clean C1; CV-R^2 best layer):")
    for tname, d in out.get("decodability_by_target", {}).items():
        print(f"     {tname:<12}: R^2={d.get('cv_r2')} @ layer {d.get('best_layer')}")
    print("  POSITIVE CONTROL — same operand-corruption patch at ')' in C6 '( 0 + B ) =' (')' value")
    print(f"     IS the answer): recovery max={out.get('pos_ctrl_recovery_max')} "
          f"(flip max={out.get('pos_ctrl_flip_max')})  [site moves output: "
          f"{'OK' if out.get('pos_ctrl_ok') else 'FAIL -> STOP'}]")
    print("  EXPERIMENT — same patch at ')' in C1 '( 0 + B ) * C =' (does ')' feed the outer * C?):")
    print(f"     recovery by layer: {out.get('recovery_by_layer')}")
    print(f"     mid-late {out.get('midlate_layers')} recovery MAX={out.get('recovery_midlate_max')} "
          f"flip MAX={out.get('flip_rate_midlate_max')}")
    if out.get("stop"):
        print("  ⛔ STOP: " + out["reading"])
    else:
        print(f"  reading: {out['reading']}")
    print("=============================================================================")


@torch.no_grad()
def _wo_salvage(tag):
    """Decodable-but-causally-unused test — REDESIGNED to fix the no-op confound.

    WHY THE REDESIGN. The prior test copied C6's ')' residual into C1's ')'. Because
    the alignment guard forced C1 '( 0 + B ) * C =' and C6 '( 0 + B ) =' to be
    token-identical through ')', and causal attention makes resid_post[')'] a function
    of ONLY those identical tokens, the donor == the target: the patch was an IDENTITY
    operation and flip-rate ~ 0 was guaranteed regardless of composition. We instead
    use OPERAND-CORRUPTION denoising at the ')' site (the validated G4 /
    _wo_localize_parse idiom), so donor != target:

      EXPERIMENT (does the operand at ')' feed the outer '* C'?):
        clean   = '( 0 + B ) * C ='     -> first token F_clean
        corrupt = '( 0 + B' ) * C ='    (B' = same #digits) -> F_corr (!= F_clean)
        Patch the CLEAN ')' residual into the CORRUPTED run at ')'. Recovery of the
        logit-diff(F_clean - F_corr) at the final position, swept over layers.
        recovery ~ 0 across the mid-late consumption zone => restoring B at ')' does
        NOT move C1's output => the operand is decoded-but-causally-unused.

      POSITIVE CONTROL — SITE-MATCHED: the SAME operand-corruption patch at the SAME
        ')' position, but in C6 '( 0 + B ) =' where the bracketed value IS the answer.
        Recovery should be HIGH (>= thresh): proof a ')'-site patch CAN move the
        output. High C6 recovery beside ~0 C1 recovery is the airtight contrast; if
        C6 also fails, the null is uninterpretable (STOP). (Replaces the old
        final-position C4 control, which was a tautological mover at a different site.)

      DECODABILITY (§3.7): CV-R^2 of B / B*C / C / shuffled-B from the clean C1 ')'
        residual. B is high (present); B*C and C are low PARTLY because C is causally
        future at ')' — so the load-bearing evidence is the experiment-vs-control
        contrast, not the decodability gap (reported honestly).

    NET BY CONSTRUCTION: clean_first != corrupt_first is required and the corrupt run's
    UNPATCHED argmax IS corrupt_first, so an unpatched example can never count as a
    flip — recovery/flip are inherently baseline-netted. Subset = C1-wrong examples
    (the failing regime). Runs on BOTH tags; checkpointed per layer; resumable."""
    done_key = f"wo_salvage_{tag}"
    deliverable = f"salvage_c6_to_c1_{tag}.json"
    if has_artifact(done_key, "json"):
        out = load_json(done_key)
        wo_save_result(deliverable, __import__("json").dumps(out, indent=2, default=str))
        log(f"WO salvage[{tag}]: complete artifact found — reused (no GPU).")
        _wo_print_salvage(out)
        return out

    wo_load_model(tag)
    _rmap = dict((c[0], c[2]) for c in WO_CONDITIONS)
    renderC1, renderC6 = _rmap["C1"], _rmap["C6"]
    n_layers = model.cfg.n_layers
    eps = 1e-8

    # --- subset: C1-wrong examples (the failing regime where the claim lives) ---
    res = _wo_battery_res_for_salvage(tag)
    c1_mask = res["C1"]["correct_mask"]
    assert len(c1_mask) == len(WO_PAIRS), (
        f"salvage[{tag}]: C1 correct_mask (len {len(c1_mask)}) not aligned to WO_PAIRS")
    wrong_idx = [i for i in range(len(WO_PAIRS)) if not c1_mask[i]]
    n_req = min(len(wrong_idx), int(CFG["wo_salvage_n"]))
    rng = np.random.default_rng(int(CFG["wo_localize_seed"]) + 2)
    sel = rng.choice(len(wrong_idx), size=n_req, replace=False) if n_req > 0 else []
    sample = [WO_PAIRS[int(wrong_idx[int(i)])] for i in sel]
    log(f"WO salvage[{tag}]: {len(wrong_idx)} C1-wrong candidates; sampling {n_req}.")

    def _rparen_pos(tokens):
        toks = [tokenizer.decode([t]).strip() for t in tokens[0].tolist()]
        for i, t in enumerate(toks):
            if t == ")":          # first ')' closes '( 0 + B )' in C1 and C6.
                return i
        return None

    def _ld(logits, a, b):
        return _wo_first_tok_logit(logits, -1, a) - _wo_first_tok_logit(logits, -1, b)

    def _collect(render, B, Bp, C):
        """Operand-corruption setup at the ')' site for one (render, B, B', C).
        Returns the corrupt tokens, the ')' index, clean/corrupt first tokens +
        logit-diffs, and the CLEAN ')' residual per layer (fp16 CPU). The string
        'no_contrast' if F_clean == F_corr; None if otherwise unusable."""
        clean_tok = model.to_tokens(render(B, C))
        corrupt_tok = model.to_tokens(render(Bp, C))
        if clean_tok.shape != corrupt_tok.shape:
            return None
        rp, rpc = _rparen_pos(clean_tok), _rparen_pos(corrupt_tok)
        if rp is None or rp != rpc:
            return None
        clean_logits, clean_cache = model.run_with_cache(
            clean_tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        corrupt_logits = model(corrupt_tok)
        clean_first = int(clean_logits[0, -1].argmax().item())
        corrupt_first = int(corrupt_logits[0, -1].argmax().item())
        if clean_first == corrupt_first:
            return "no_contrast"
        ld_clean = _ld(clean_logits, clean_first, corrupt_first)
        ld_corrupt = _ld(corrupt_logits, clean_first, corrupt_first)
        if not (ld_clean > ld_corrupt):
            return None      # metric sign wrong for this example -> skip.
        clean_resid = [clean_cache[f"blocks.{L}.hook_resid_post"][0, rp, :].half().cpu()
                       for L in range(n_layers)]
        return {"corrupt_tok": corrupt_tok, "rp": int(rp),
                "clean_first": clean_first, "corrupt_first": corrupt_first,
                "ld_clean": float(ld_clean), "ld_corrupt": float(ld_corrupt),
                "clean_resid": clean_resid}

    # --- Phase A: collect PAIRED C1 (experiment) + C6 (site-matched control) ----
    feats = {L: [] for L in range(n_layers)}
    Bvals, Cvals = [], []
    exp_ex, ctrl_ex = [], []
    n_skip = 0
    n_no_contrast = 0
    for (B, C) in sample:
        Bp = _wo_corrupt_B(B, rng, *WO_BAND)
        if Bp is None:
            n_skip += 1; continue
        c1 = _collect(renderC1, B, Bp, C)
        if c1 == "no_contrast":
            n_no_contrast += 1; continue
        if c1 is None:
            n_skip += 1; continue
        c6 = _collect(renderC6, B, Bp, C)
        if c6 == "no_contrast" or c6 is None:
            n_skip += 1; continue        # require BOTH so experiment & control share examples.
        for L in range(n_layers):
            feats[L].append(c1["clean_resid"][L].float().numpy())
        Bvals.append(int(B)); Cvals.append(int(C))
        exp_ex.append(c1); ctrl_ex.append(c6)
    n_used = len(Bvals)

    # --- decodability for FOUR targets (§3.7) from the clean C1 ')' residual -----
    Bv = np.array(Bvals, dtype=float); Cv = np.array(Cvals, dtype=float)
    prods = Bv * Cv
    shuf = wo_shuffle_control(Bv, seed=int(CFG["wo_localize_seed"]) + 3)

    def _decod_curve(y):
        return {L: wo_cv_r2(np.array(feats[L]), y)
                for L in range(n_layers) if len(feats[L]) >= 7}

    def _best(curve):
        cand = [(L, v) for L, v in curve.items() if v is not None]
        if not cand:
            return {"best_layer": None, "cv_r2": None}
        L, v = max(cand, key=lambda t: t[1])
        return {"best_layer": int(L), "cv_r2": float(v)}

    decod_B = _decod_curve(Bv)
    decodability_by_target = {
        "B": _best(decod_B),
        "B_times_C": _best(_decod_curve(prods)),
        "C": _best(_decod_curve(Cv)),
        "shuffled_B": _best(_decod_curve(shuf)),
    }
    best_layer = decodability_by_target["B"]["best_layer"]
    r2_best = decodability_by_target["B"]["cv_r2"]

    # --- Phase B: layer-swept recovery for experiment (C1) + control (C6) -------
    L_sweep = sorted(set(range(0, n_layers, 2)) | ({best_layer} if best_layer is not None else set()))
    sweep_ck = f"wo_salvage_sweep_{tag}"
    exp_rec, exp_flip, pos_rec, pos_flip = {}, {}, {}, {}
    if has_artifact(sweep_ck, "json"):
        prev = load_json(sweep_ck)
        if prev.get("best_layer") == best_layer and prev.get("n") == n_used:
            exp_rec = {int(k): v for k, v in prev.get("exp_rec", {}).items()}
            exp_flip = {int(k): v for k, v in prev.get("exp_flip", {}).items()}
            pos_rec = {int(k): v for k, v in prev.get("pos_rec", {}).items()}
            pos_flip = {int(k): v for k, v in prev.get("pos_flip", {}).items()}
            log(f"WO salvage[{tag}]: resuming sweep ({len(exp_rec)}/{len(L_sweep)} layers done).")
        else:
            log(f"WO salvage[{tag}]: stale sweep checkpoint (best_layer/n changed) — recomputing.")

    def _save_sweep():
        save_json(sweep_ck, {"exp_rec": {str(k): v for k, v in exp_rec.items()},
                             "exp_flip": {str(k): v for k, v in exp_flip.items()},
                             "pos_rec": {str(k): v for k, v in pos_rec.items()},
                             "pos_flip": {str(k): v for k, v in pos_flip.items()},
                             "best_layer": best_layer, "n": n_used})

    def _patch_recovery(ex, L):
        vec = ex["clean_resid"][L].to(model.cfg.device)
        patched = model.run_with_hooks(
            ex["corrupt_tok"],
            fwd_hooks=[(f"blocks.{L}.hook_resid_post", _wo_mk_patch_hook(vec, ex["rp"]))])
        ld_p = _ld(patched, ex["clean_first"], ex["corrupt_first"])
        rec = (ld_p - ex["ld_corrupt"]) / (ex["ld_clean"] - ex["ld_corrupt"] + eps)
        flip = 1.0 if int(patched[0, -1].argmax().item()) == ex["clean_first"] else 0.0
        return rec, flip

    def _avg_layer(examples, L):
        rs = [_patch_recovery(ex, L) for ex in examples]
        return float(np.mean([r for r, _ in rs])), float(np.mean([f for _, f in rs]))

    if best_layer is not None and n_used > 0:
        for L in L_sweep:
            if L in exp_rec and L in pos_rec:
                continue
            if L not in exp_rec:
                exp_rec[L], exp_flip[L] = _avg_layer(exp_ex, L)
            if L not in pos_rec:
                pos_rec[L], pos_flip[L] = _avg_layer(ctrl_ex, L)
            _save_sweep()                                   # checkpoint per layer (disconnect-safe)
            log(f"WO salvage[{tag}]: layer {L}/{n_layers - 1} "
                f"exp_rec={exp_rec[L]:.3f} pos_rec={pos_rec[L]:.3f}")

    # --- derived headline numbers --------------------------------------------
    recovery_by_layer = {str(L): exp_rec[L] for L in sorted(exp_rec)}
    flip_by_layer = {str(L): exp_flip[L] for L in sorted(exp_flip)}
    pos_recovery_by_layer = {str(L): pos_rec[L] for L in sorted(pos_rec)}
    pos_flip_by_layer = {str(L): pos_flip[L] for L in sorted(pos_flip)}
    midlate_lo = int(np.floor(0.6 * n_layers))
    midlate_layers = [L for L in sorted(exp_rec) if L >= midlate_lo]
    recovery_midlate_max = max((exp_rec[L] for L in midlate_layers), default=None)
    flip_rate_midlate_max = max((exp_flip[L] for L in midlate_layers), default=None)
    pos_ctrl_recovery_max = max(pos_rec.values(), default=None)
    pos_ctrl_flip_max = max(pos_flip.values(), default=None)

    thresh = float(CFG["wo_salvage_recovery_thresh"])
    n_used_ok = n_used >= int(CFG["wo_salvage_min_n"])
    decodable = (r2_best is not None and r2_best >= 0.5)
    pos_ctrl_ok = (pos_ctrl_recovery_max is not None and pos_ctrl_recovery_max >= thresh)
    causally_used = (recovery_midlate_max is not None and recovery_midlate_max >= thresh)
    stop = (best_layer is not None and n_used > 0 and not pos_ctrl_ok)

    if r2_best is None or n_used == 0 or best_layer is None:
        reading = "INCONCLUSIVE: too few usable contrast examples to estimate decodability/causal use."
    elif stop:
        reading = (f"STOP — the SITE-MATCHED positive control FAILED: the same operand-corruption patch "
                   f"at ')' in C6 (where that value IS the answer) only recovers "
                   f"{pos_ctrl_recovery_max:.2f} (< {thresh:.2f}). A ')'-site patch cannot be shown to "
                   "move the output, so the C1 null is uninterpretable — fix the instrument first.")
    elif decodable and not causally_used:
        reading = ("DECODABLE-BUT-CAUSALLY-UNUSED (site-matched): B is linearly decodable from C1's "
                   f"post-bracket ')' residual (CV-R^2={r2_best:.2f} @ layer {best_layer}); the SAME "
                   f"operand-corruption patch at ')' DOES move the output where that value is used "
                   f"(C6 control recovery={pos_ctrl_recovery_max:.2f} >= {thresh:.2f}); yet restoring "
                   f"the clean operand at ')' in C1 does NOT recover the composed output across the "
                   f"mid-late consumption zone (recovery max={recovery_midlate_max:.2f}, flip max="
                   f"{flip_rate_midlate_max:.2f}). The operand is decoded at ')' and not consumed by the "
                   "outer '* C'. (No-op transplant confound removed: donor != target via operand "
                   "corruption; control is site-matched.)")
    elif decodable and causally_used:
        reading = (f"USED: B decodable (CV-R^2={r2_best:.2f}) AND restoring the clean operand at ')' "
                   f"recovers C1's output in the consumption zone (recovery max={recovery_midlate_max:.2f} "
                   f">= {thresh:.2f}) — the operand at ')' is causally consumed.")
    else:
        reading = (f"NOT CLEANLY DECODABLE (CV-R^2={r2_best:.2f} < 0.5) at the ')' site; the "
                   "decodable-but-unused test does not apply here.")
    if not n_used_ok:
        reading = f"[WARN n_used={n_used} < {int(CFG['wo_salvage_min_n'])}] " + reading

    out = {
        "tag": tag,
        "design": "operand-corruption denoising at ')' with a site-matched C6 positive control",
        "n_used": n_used, "n_used_ok": bool(n_used_ok), "min_n": int(CFG["wo_salvage_min_n"]),
        "n_skipped": n_skip, "n_no_contrast": n_no_contrast,
        "n_wrong_candidates": len(wrong_idx), "n_sample_requested": n_req,
        # decodability (clean C1 ')' residual)
        "B_decodable_cv_r2_best": r2_best, "B_decodable_best_layer": best_layer,
        "B_decodable_cv_r2_by_layer": {str(L): v for L, v in decod_B.items()},
        "decodability_by_target": decodability_by_target,
        # experiment (C1): does restoring B at ')' move the output?
        "recovery_by_layer": recovery_by_layer,
        "flip_rate_by_layer": flip_by_layer,
        "recovery_midlate_max": recovery_midlate_max,
        "flip_rate_midlate_max": flip_rate_midlate_max,
        "midlate_layers": midlate_layers,
        # positive control (C6, SAME patch where ')' value is the answer)
        "pos_ctrl_design": "operand-corruption patch at ')' in C6 '( 0 + B ) ='",
        "pos_ctrl_recovery_by_layer": pos_recovery_by_layer,
        "pos_ctrl_flip_by_layer": pos_flip_by_layer,
        "pos_ctrl_recovery_max": pos_ctrl_recovery_max,
        "pos_ctrl_flip_max": pos_ctrl_flip_max,
        "pos_ctrl_flip_rate": pos_ctrl_flip_max,   # §4 deliverable key (headline)
        "pos_ctrl_ok": bool(pos_ctrl_ok),
        "recovery_thresh": thresh,
        # verdict
        "decodable": bool(decodable), "causally_used": bool(causally_used),
        "stop": bool(stop), "reading": reading,
    }
    wo_save_result(deliverable, __import__("json").dumps(out, indent=2, default=str))
    save_json(done_key, out)
    _wo_print_salvage(out)
    return out


# ----------------------------------------------------------------------------
# DISPATCH on the selected branch (§8). Override with CFG['wo_force_branch'] to
# exercise a specific path (testing / what-if).
# ----------------------------------------------------------------------------
_branch = CFG.get("wo_force_branch", WO_BRANCH["branch"])
log(f"WO STEP 5b: executing downstream artifact for branch = {_branch}.")
if _branch in ("CLEAN_REPAIR", "PARTIAL_REPAIR"):
    _wo_run_localization(WO_BRANCH["run_on"] if _branch == "CLEAN_REPAIR" else ["instruct"])
else:  # NO_REPAIR
    _wo_run_branchb("instruct")
    # §3.3: salvage on BOTH tags (base is the cleanest demo — C6=1.000, C1=0.51).
    # Each call loads its own model; instruct first (already characterised), then base.
    _wo_salvage("instruct")
    _wo_salvage("base")
log("WO STEP 5b complete.")
