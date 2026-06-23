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


def _wo_print_salvage(out):
    print("\n================= WO STEP 5b — C6->C1 SALVAGE (§10.B) =================")
    print(f"  [{out['tag']}] n_used={out['n_used']} (wrong candidates={out.get('n_wrong_candidates')}, "
          f"skipped={out['n_skipped']})")
    print(f"  DECODABILITY @ post-bracket ')' site (CV-R^2, best layer):")
    for tname, d in out.get("decodability_by_target", {}).items():
        print(f"     {tname:<12}: R^2={d.get('cv_r2')} @ layer {d.get('best_layer')}")
    print(f"  POSITIVE CONTROL (C4 donor -> C1 final @ layer {out.get('pos_ctrl_layer')}): "
          f"flip-rate={out.get('pos_ctrl_flip_rate')}  (>=0.50 required: "
          f"{'OK' if out.get('pos_ctrl_ok') else 'FAIL/STOP'})")
    print(f"  EXPERIMENT (C6 -> C1 ')') flip-to-correct by layer: {out.get('flip_rate_by_layer')}")
    print(f"     best-decode-layer ({out.get('patch_layer')}) flip={out.get('patch_argmax_flip_rate_to_correct')}; "
          f"mid-late ({out.get('midlate_layers')}) MAX flip={out.get('flip_rate_midlate_max')}; "
          f"unpatched-already-correct baseline={out.get('unpatched_argmax_correct_rate')}")
    if out.get("stop"):
        print("  ⛔ STOP: " + out["reading"])
    else:
        print(f"  reading: {out['reading']}")
    print("======================================================================")


@torch.no_grad()
def _wo_salvage(tag):
    """§10.B C6->C1 salvage — the causal centerpiece, hardened per Work Order #2.
    At the post-bracket ')' position of C1 (closing '( 0 + B )', where the bracketed
    value should feed the outer '* C'):

      (1) DECODABILITY (§3.7) — held-out CV-R^2 of a linear probe at the ')' site,
          for FOUR targets: B (have), the product B*C (what the model should be
          computing toward), C, and a SHUFFLED-B null control. Expect B high, the
          rest low — B is present where the product is not.
      (2) CAUSAL USE (§3.2) — patch C6's resid ('( 0 + B ) = B', correctly
          evaluated) at the ')' site into C1, SWEPT over a layer set (every other
          layer ∪ the best-decode layer), reading whether C1's argmax flips to the
          correct product's first token. Flip-rate ≈ 0 across the mid-late
          consumption zone => the operand is computed, carried, and discarded.
      (3) POSITIVE CONTROL (§3.1) — the load-bearing hygiene: patch the C4 donor
          ('( B * C ) =', evaluated correctly) final-position resid into C1's final
          position at a late layer. This SHOULD flip C1 to correct (>=0.5); if it
          does not, the hook cannot move the output and the whole null is suspect
          (STOP). Reported side-by-side with the experimental flip-rate.

    SUBSET (§3.3): keep (B,C) iff C1's full PARSED answer != B*C ("C1 is actually
    wrong"), read from the in-memory battery correct_mask (index-aligned to
    WO_PAIRS) — not the old over-strict first-token-mismatch rule that dropped n to
    33. Runs on BOTH tags (base is cleanest: C6=1.000). The readout TARGET is the
    correct product's first token (empirical, from C4 — avoids the Llama leading-
    space pitfall). rp1==rp6 + prefix identity is asserted before transplant."""
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
    renderC1, renderC6, renderC4 = _rmap["C1"], _rmap["C6"], _rmap["C4"]
    n_layers = model.cfg.n_layers
    L_pc = int(np.floor(0.75 * n_layers))                 # §3.1 positive-control layer.

    # --- §3.3 subset: examples where C1's full parsed answer is WRONG. ----------
    res = _wo_battery_res_for_salvage(tag)
    c1_mask = res["C1"]["correct_mask"]
    assert len(c1_mask) == len(WO_PAIRS), (
        f"salvage[{tag}]: C1 correct_mask (len {len(c1_mask)}) not aligned to "
        f"WO_PAIRS (len {len(WO_PAIRS)})")
    wrong_idx = [i for i in range(len(WO_PAIRS)) if not c1_mask[i]]
    n_req = min(len(wrong_idx), int(CFG["wo_salvage_n"]))
    rng = np.random.default_rng(int(CFG["wo_localize_seed"]) + 2)
    sel = rng.choice(len(wrong_idx), size=n_req, replace=False) if n_req > 0 else []
    sample_idx = sorted(int(wrong_idx[int(i)]) for i in sel)
    sample = [WO_PAIRS[i] for i in sample_idx]
    log(f"WO salvage[{tag}]: {len(wrong_idx)} C1-wrong candidates; sampling {n_req} "
        f"(seed {int(CFG['wo_localize_seed']) + 2}).")

    def _rparen_pos(tokens):
        toks = [tokenizer.decode([t]).strip() for t in tokens[0].tolist()]
        for i, t in enumerate(toks):
            if t == ")":          # first ')' closes '( 0 + B )' in both C1 and C6.
                return i
        return None

    # --- collection loop: gather per-example activations + scalars -------------
    feats = {L: [] for L in range(n_layers)}
    Bvals, Cvals = [], []
    examples = []   # dicts: c1_tok, rp1, c1_final, correct_first, base_correct, c6_vecs, c4_pc_vec
    n_skip = 0
    n_unpatched_correct = 0
    for (B, C) in sample:
        c1_tok = model.to_tokens(renderC1(B, C))
        c6_tok = model.to_tokens(renderC6(B, C))
        rp1, rp6 = _rparen_pos(c1_tok), _rparen_pos(c6_tok)
        if rp1 is None or rp6 is None:
            n_skip += 1; continue
        # alignment guard: same ')' index AND token-identical '( 0 + B )' prefix.
        if rp1 != rp6 or c1_tok[0, :rp1 + 1].tolist() != c6_tok[0, :rp6 + 1].tolist():
            n_skip += 1; continue
        # correct first answer-token = what the model emits for the correctly-
        # composed product (C4 '( B * C ) ='), derived EMPIRICALLY; we also grab
        # C4's final-position resid @ L_pc as the positive-control donor.
        c4_tok = model.to_tokens(renderC4(B, C))
        c4_logits, c4_cache = model.run_with_cache(
            c4_tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        correct_first = int(c4_logits[0, -1].argmax().item())
        c4_pc_vec = c4_cache[f"blocks.{L_pc}.hook_resid_post"][0, -1, :].half().cpu()

        c1_logits, c1_cache = model.run_with_cache(
            c1_tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        _, c6_cache = model.run_with_cache(
            c6_tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))

        for L in range(n_layers):
            feats[L].append(c1_cache[f"blocks.{L}.hook_resid_post"][0, rp1, :].float().cpu().numpy())
        Bvals.append(int(B)); Cvals.append(int(C))
        base_correct = _wo_first_tok_logit(c1_logits, -1, correct_first)
        if int(c1_logits[0, -1].argmax().item()) == correct_first:
            n_unpatched_correct += 1   # C1's first token already right (full answer still wrong).
        c6_vecs = [c6_cache[f"blocks.{L}.hook_resid_post"][0, rp6, :].half().cpu()
                   for L in range(n_layers)]   # keep on CPU (fp16) to bound GPU memory.
        examples.append({"c1_tok": c1_tok, "rp1": int(rp1), "c1_final": int(c1_tok.shape[1] - 1),
                         "correct_first": int(correct_first), "base_correct": float(base_correct),
                         "c6_vecs": c6_vecs, "c4_pc_vec": c4_pc_vec})

    n_used = len(Bvals)

    # --- (1) decodability for FOUR targets (§3.7); pure wo_cv_r2 -------------
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

    # --- (2) experimental C6->C1 patch, SWEPT over layers (§3.2); checkpointed --
    flip_by_L, delta_by_L = {}, {}
    pos_ctrl_flip_rate = None
    sweep_ck = f"wo_salvage_sweep_{tag}"
    if has_artifact(sweep_ck, "json"):
        prev = load_json(sweep_ck)
        if prev.get("best_layer") == best_layer and prev.get("n") == n_used:
            flip_by_L = {int(k): v for k, v in prev.get("flip", {}).items()}
            delta_by_L = {int(k): v for k, v in prev.get("delta", {}).items()}
            pos_ctrl_flip_rate = prev.get("pos_ctrl")
            log(f"WO salvage[{tag}]: resuming sweep ({len(flip_by_L)} layers done, "
                f"pos_ctrl={'done' if pos_ctrl_flip_rate is not None else 'pending'}).")
        else:
            log(f"WO salvage[{tag}]: stale sweep checkpoint (best_layer/n changed) — recomputing.")

    def _save_sweep():
        save_json(sweep_ck, {"flip": {str(k): v for k, v in flip_by_L.items()},
                             "delta": {str(k): v for k, v in delta_by_L.items()},
                             "pos_ctrl": pos_ctrl_flip_rate,
                             "best_layer": best_layer, "n": n_used})

    if best_layer is not None and n_used > 0:
        L_sweep = sorted(set(range(0, n_layers, 2)) | {best_layer})
        for L in L_sweep:
            if L in flip_by_L:
                continue
            flips, deltas = [], []
            for ex in examples:
                c6_dev = ex["c6_vecs"][L].to(model.cfg.device)
                patched = model.run_with_hooks(
                    ex["c1_tok"],
                    fwd_hooks=[(f"blocks.{L}.hook_resid_post", _wo_mk_patch_hook(c6_dev, ex["rp1"]))])
                flips.append(1.0 if int(patched[0, -1].argmax().item()) == ex["correct_first"] else 0.0)
                deltas.append(_wo_first_tok_logit(patched, -1, ex["correct_first"]) - ex["base_correct"])
            flip_by_L[L] = float(np.mean(flips)); delta_by_L[L] = float(np.mean(deltas))
            _save_sweep()                                   # checkpoint per layer (disconnect-safe)
            log(f"WO salvage[{tag}]: swept layer {L}/{n_layers - 1} flip={flip_by_L[L]:.3f}")

        # --- (3) positive control (§3.1): C4 final-pos donor -> C1 final @ L_pc --
        if pos_ctrl_flip_rate is None:
            pc_flips = []
            for ex in examples:
                c4_dev = ex["c4_pc_vec"].to(model.cfg.device)
                patched = model.run_with_hooks(
                    ex["c1_tok"],
                    fwd_hooks=[(f"blocks.{L_pc}.hook_resid_post", _wo_mk_patch_hook(c4_dev, ex["c1_final"]))])
                pc_flips.append(1.0 if int(patched[0, -1].argmax().item()) == ex["correct_first"] else 0.0)
            pos_ctrl_flip_rate = float(np.mean(pc_flips)) if pc_flips else None
            _save_sweep()
            log(f"WO salvage[{tag}]: positive control flip-rate={pos_ctrl_flip_rate}")

    # --- derived headline numbers --------------------------------------------
    flip_rate_by_layer = {str(L): flip_by_L[L] for L in sorted(flip_by_L)}
    delta_by_layer = {str(L): delta_by_L[L] for L in sorted(delta_by_L)}
    midlate_lo = int(np.floor(0.6 * n_layers))
    midlate_layers = [L for L in sorted(flip_by_L) if L >= midlate_lo]
    flip_rate_midlate_max = max((flip_by_L[L] for L in midlate_layers), default=None)
    exp_flip = flip_by_L.get(best_layer)
    exp_delta = delta_by_L.get(best_layer)
    unpatched_rate = (n_unpatched_correct / n_used) if n_used else None

    decodable = (r2_best is not None and r2_best >= 0.5)
    pos_ctrl_ok = (pos_ctrl_flip_rate is not None and pos_ctrl_flip_rate >= 0.5)
    causally_used = (flip_rate_midlate_max is not None and flip_rate_midlate_max >= 0.20)
    stop = (best_layer is not None and n_used > 0 and not pos_ctrl_ok)

    if r2_best is None or n_used == 0:
        reading = "INCONCLUSIVE: too few usable examples to estimate decodability/causal use."
    elif stop:
        reading = (f"STOP — positive control FAILED (flip-rate={pos_ctrl_flip_rate:.2f} < 0.50). The "
                   "patching hook cannot move C1's output even with the correctly-evaluated C4 donor, "
                   "so the experimental null is uninterpretable. Fix the instrument before reporting "
                   "the salvage (work order §3.1).")
    elif decodable and not causally_used:
        reading = ("DECODABLE-BUT-UNUSED: B is linearly decodable from C1's post-bracket ')' site "
                   f"(CV-R^2={r2_best:.2f} @ layer {best_layer}) and the POSITIVE CONTROL moves the "
                   f"output (C4 donor flip-rate={pos_ctrl_flip_rate:.2f} >= 0.50), yet patching the "
                   f"correctly-evaluated C6 subexpr at that site flips C1 to the correct product almost "
                   f"never across the consumption zone (mid-late max flip={flip_rate_midlate_max:.2f}, "
                   f"best-decode-layer flip={exp_flip:.2f}; unpatched-already-correct baseline="
                   f"{unpatched_rate:.2f}). The operand is computed, carried, and discarded — "
                   "decodable-but-not-causally-used, now layer-swept with a passing positive control.")
    elif decodable and causally_used:
        reading = (f"USED: B decodable (CV-R^2={r2_best:.2f}) AND the C6 patch recovers the correct "
                   f"product in the consumption zone (mid-late max flip={flip_rate_midlate_max:.2f} "
                   f">= 0.20) — the operand is causally consumed at some layer.")
    else:
        reading = (f"NOT CLEANLY DECODABLE (CV-R^2={r2_best:.2f} < 0.5) at the post-bracket site; the "
                   "decodable-but-unused test does not apply at this site/layer.")

    out = {
        "tag": tag, "n_used": n_used, "n_skipped": n_skip,
        "n_wrong_candidates": len(wrong_idx),
        "n_sample_requested": n_req,
        "n_unpatched_argmax_already_correct": n_unpatched_correct,
        "unpatched_argmax_correct_rate": unpatched_rate,
        # decodability
        "B_decodable_cv_r2_best": r2_best, "B_decodable_best_layer": best_layer,
        "B_decodable_cv_r2_by_layer": {str(L): v for L, v in decod_B.items()},
        "decodability_by_target": decodability_by_target,
        # experimental causal patch (swept)
        "patch_layer": best_layer,
        "patch_argmax_flip_rate_to_correct": exp_flip,
        "patch_mean_correct_logit_delta": exp_delta,
        "flip_rate_by_layer": flip_rate_by_layer,
        "delta_by_layer": delta_by_layer,
        "midlate_layers": midlate_layers,
        "flip_rate_midlate_max": flip_rate_midlate_max,
        # positive control
        "pos_ctrl_layer": L_pc,
        "pos_ctrl_flip_rate": pos_ctrl_flip_rate,
        "pos_ctrl_ok": bool(pos_ctrl_ok),
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
