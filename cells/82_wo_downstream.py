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
CFG.setdefault("wo_salvage_n", 128)         # salvage decodability needs n >> reduced-dim.


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


@torch.no_grad()
def _wo_salvage(tag):
    """§10.B C6->C1 salvage (decodable-but-unused causal test).
    At the post-bracket ')' position of C1 (closing '( 0 + B )', where the
    bracketed value should feed the outer '* C'):
      (1) DECODABILITY — is B linearly decodable from C1's resid at that site?
          Measured by held-out CV-R^2 (PCA+ridge) per layer; best layer reported.
      (2) CAUSAL USE — patch C6's resid (the correctly-evaluated '( 0 + B ) = B')
          at that site at the BEST-DECODE layer into C1, and read whether C1 now
          emits the CORRECT product (argmax flip to the true-product first token)
          and how much the correct-product logit rises.
    READING: B decodable (CV-R^2>=0.5) AND patch rarely makes C1 correct (flip-rate
    low) => decodable-but-UNUSED — the operand is computed and discarded.
    The readout TARGET is the correct product's first token (not C1's own wrong
    token), and 'used' is the scale-free argmax-flip RATE (not an arbitrary
    raw-logit cutoff). rp1==rp6 + prefix identity is asserted before transplant."""
    wo_load_model(tag)
    _rmap = dict((c[0], c[2]) for c in WO_CONDITIONS)
    renderC1, renderC6, renderC4 = _rmap["C1"], _rmap["C6"], _rmap["C4"]
    rng = np.random.default_rng(int(CFG["wo_localize_seed"]) + 2)
    n = min(len(WO_PAIRS), int(CFG["wo_salvage_n"]))
    sample = [WO_PAIRS[i] for i in rng.choice(len(WO_PAIRS), size=n, replace=False)]
    n_layers = model.cfg.n_layers

    def _rparen_pos(tokens):
        toks = [tokenizer.decode([t]).strip() for t in tokens[0].tolist()]
        for i, t in enumerate(toks):
            if t == ")":          # first ')' closes '( 0 + B )' in both C1 and C6.
                return i
        return None

    feats = {L: [] for L in range(n_layers)}
    Bvals = []
    examples = []   # (c1_tok, rp1, correct_first, base_correct_logit, c6_vecs_cpu[L])
    n_skip = 0
    n_c1_already_correct = 0
    for (B, C) in sample:
        c1_tok = model.to_tokens(renderC1(B, C))
        c6_tok = model.to_tokens(renderC6(B, C))
        rp1, rp6 = _rparen_pos(c1_tok), _rparen_pos(c6_tok)
        if rp1 is None or rp6 is None:
            n_skip += 1; continue
        # alignment guard: same ')' index AND token-identical '( 0 + B )' prefix.
        if rp1 != rp6 or c1_tok[0, :rp1 + 1].tolist() != c6_tok[0, :rp6 + 1].tolist():
            n_skip += 1; continue
        # correct first answer-token = the token the model ITSELF emits for the
        # correctly-composed product (C4 '( B * C ) =', the high-acc inside-bracket
        # surface), derived EMPIRICALLY — mirrors _wo_localize_parse and avoids the
        # Llama leading-space-token pitfall of tokenizing str(B*C) directly.
        correct_first = _wo_empirical_first_tok(model.to_tokens(renderC4(B, C)))
        c1_first = _wo_empirical_first_tok(c1_tok)
        if c1_first == correct_first:
            n_c1_already_correct += 1; continue   # C1 already correct -> nothing to salvage.
        c1_logits, c1_cache = model.run_with_cache(
            c1_tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        _, c6_cache = model.run_with_cache(
            c6_tok, names_filter=lambda nm: nm.endswith("hook_resid_post"))
        for L in range(n_layers):
            feats[L].append(c1_cache[f"blocks.{L}.hook_resid_post"][0, rp1, :].float().cpu().numpy())
        Bvals.append(B)
        base_correct = _wo_first_tok_logit(c1_logits, -1, correct_first)
        c6_vecs = [c6_cache[f"blocks.{L}.hook_resid_post"][0, rp6, :].half().cpu()
                   for L in range(n_layers)]   # keep on CPU (fp16) to bound GPU memory.
        examples.append((c1_tok, rp1, int(correct_first), base_correct, c6_vecs))

    # (1) decodability: held-out CV-R^2 per layer (wo_cv_r2 from the pure-logic cell).
    decod = {L: wo_cv_r2(np.array(feats[L]), np.array(Bvals))
             for L in range(n_layers) if len(feats[L]) >= 7}
    best_layer = max((L for L in decod if decod[L] is not None),
                     key=lambda L: decod[L], default=None)
    r2_best = decod.get(best_layer) if best_layer is not None else None

    # (2) causal use: patch C6 resid at best layer into C1; argmax-flip + logit delta.
    flips, deltas = [], []
    if best_layer is not None:
        for (c1_tok, rp1, correct_first, base_correct, c6_vecs) in examples:
            c6_resid = c6_vecs[best_layer].to(model.cfg.device)
            def hook(resid_post, hook):
                resid_post[:, rp1, :] = c6_resid.to(resid_post.dtype)
                return resid_post
            patched = model.run_with_hooks(
                c1_tok, fwd_hooks=[(f"blocks.{best_layer}.hook_resid_post", hook)])
            flips.append(1.0 if int(patched[0, -1].argmax().item()) == correct_first else 0.0)
            deltas.append(_wo_first_tok_logit(patched, -1, correct_first) - base_correct)
    flip_rate = float(np.mean(flips)) if flips else None
    mean_delta = float(np.mean(deltas)) if deltas else None

    decodable = (r2_best is not None and r2_best >= 0.5)
    used = (flip_rate is not None and flip_rate >= 0.20)
    if r2_best is None:
        reading = "INCONCLUSIVE: too few usable examples to estimate decodability."
    elif decodable and not used:
        reading = ("DECODABLE-BUT-UNUSED: B is linearly decodable from C1's post-bracket site "
                   f"(CV-R^2={r2_best:.2f}) yet patching the correctly-evaluated C6 subexpr in "
                   f"rarely makes C1 emit the correct product (flip-rate={flip_rate:.2f}). "
                   "The operand is computed and discarded — decodable-but-not-causally-used.")
    elif decodable and used:
        reading = (f"USED: B decodable (CV-R^2={r2_best:.2f}) AND the C6 patch recovers the correct "
                   f"product in C1 (flip-rate={flip_rate:.2f}) — the operand is causally consumed.")
    else:
        reading = (f"NOT CLEANLY DECODABLE (CV-R^2={r2_best:.2f}<0.5) at the post-bracket site; "
                   "the decodable-but-unused test does not apply at this site/layer.")

    out = {
        "tag": tag, "n_used": len(Bvals), "n_skipped": n_skip,
        "n_c1_already_correct_excluded": n_c1_already_correct,
        "B_decodable_cv_r2_best": r2_best, "B_decodable_best_layer": best_layer,
        "B_decodable_cv_r2_by_layer": {str(L): v for L, v in decod.items()},
        "patch_layer": best_layer,
        "patch_argmax_flip_rate_to_correct": flip_rate,
        "patch_mean_correct_logit_delta": mean_delta,
        "decodable": bool(decodable), "causally_used": bool(used),
        "reading": reading,
    }
    wo_save_result("salvage_c6_to_c1.json", __import__("json").dumps(out, indent=2, default=str))
    save_json("wo_salvage", out)
    print("\n================= WO STEP 5b — C6->C1 SALVAGE (§10.B) =================")
    print(f"  B decodable from C1 ')' site: CV-R^2={r2_best} @ layer {best_layer} "
          f"(n={len(Bvals)}, skipped={n_skip})")
    print(f"  C6->C1 patch @ layer {best_layer}: argmax-flip-to-correct rate={flip_rate}, "
          f"mean correct-logit delta={mean_delta}")
    print(f"  reading: {out['reading']}")
    print("======================================================================")


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
    _wo_salvage("instruct")
log("WO STEP 5b complete.")
