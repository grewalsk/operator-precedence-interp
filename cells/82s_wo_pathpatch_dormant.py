# ============================================================================
# Phase 6 / WORK ORDER #6 — A4 PATH PATCHING  +  B1/B2 DORMANT CERTIFICATION (GPU).
# ----------------------------------------------------------------------------
# Two finishers, reusing cell 82r's STR items + metric/hook machinery.
#
# A4 — PATH PATCHING (the canonical operand->answer edge localization). For the
#   top operand-attributed heads (from 82r), separate the DIRECT effect on the
#   logits from the effect MEDIATED through later layers (Wang et al. IOI / Goldowsky-
#   Dill et al.). Method: exact-patch head H's output (clean->corrupt, denoising) and
#   read its TOTAL recovery; then re-measure with every LATER layer's attention-head
#   outputs (hook_z) AND MLP-out FROZEN to their corrupt-run values, so H's effect can
#   reach the logits ONLY through the residual skip (the direct path) — that is H's
#   DIRECT recovery; MEDIATED = TOTAL - DIRECT. A head with mostly-direct effect writes
#   the answer-relevant signal straight to the unembedding; a mostly-mediated head's
#   effect is re-processed by downstream MLPs/heads (Stolfo's late-MLP story).
#   This is PATH patching, not bare head patching.
#
# B1 — MAKELOV DECOMPOSE-AND-COMPARE (2311.17030), the rigorous negative half. At the
#   '=' answer site the product B*C is linearly decodable (R^2~0.96). Fit the decode
#   direction w-hat, then decompose it into its LOGIT-AFFECTING component v_row (its
#   projection onto the span of the answer-token unembedding columns — what the
#   unembedding/late path can read) and its LOGIT-INERT component v_null (the
#   orthogonal remainder). Report (i) the inert SHARE of w-hat and (ii) the decode-R^2
#   recoverable from the logit-affecting subspace vs the logit-inert complement. If the
#   decodable product lives in the logit-inert subspace -> DORMANT_CERTIFIED:
#   decodable but causally disconnected from the weights (the interpretability illusion).
#   DAS is deliberately NOT used (it finds illusory subspaces and inherits the illusion).
#
# B2 — reference the full-component nulls already in hand (WO#5.1d full-residual swap;
#   WO#5 Exp B operand-explained selectivity) to complete the three-legged dissociation:
#   decodable . operand-explained . causally inert.
#
# All decision/linear-algebra math is the cell-76 pure helpers (wo_fit_ridge_probe,
# wo_cv_r2, wo_readout_decompose, wo_readout_decode_split, wo_dormant_verdict),
# unit-tested on CPU. This cell is thin orchestration; resumable + base/instruct.
# ============================================================================
import json
import numpy as np
import torch

assert "wo_readout_decompose" in globals() and "wo_readout_decode_split" in globals() \
    and "wo_dormant_verdict" in globals() and "wo_fit_ridge_probe" in globals() \
    and "wo_cv_r2" in globals() and "_loc_build" in globals() and "_loc_metric" in globals() \
    and "_loc_mk_z_hook" in globals() and "_loc_mk_seq_hook" in globals() \
    and "_loc_names_filter" in globals() and "wo_load_model" in globals(), (
    "WO#6 82s needs cell 82r (STR items + _loc_* helpers) and the cell-76 WO#6 / "
    "Makelov helpers. Run cells 76, 82d, 82r first.")

CFG.setdefault("wo_pp_tags", list(CFG.get("wo_loc_tags", ["base", "instruct"])))
CFG.setdefault("wo_pp_n", 48)                  # path patching is the costlier finisher
CFG.setdefault("wo_pp_head_topk", 8)           # # operand heads to direct-vs-mediated split
CFG.setdefault("wo_pp_direct_thr", 0.5)        # >= this share of recovery is "DIRECT"
CFG.setdefault("wo_dorm_layers", None)         # '=' layers to Makelov-decompose (None -> final + ~0.75 depth)
CFG.setdefault("wo_dorm_n", min(200, len(WO_FSPROBE_PAIRS)))  # FORWARD-only decode set; n >> r for a trustworthy R2


# ----------------------------------------------------------------------------
# A4 — PATH PATCHING.
# ----------------------------------------------------------------------------
def _pp_mk_zfull_hook(z_dev):
    """Freeze a whole layer's hook_z (all heads, all positions) to `z_dev`
    ([seq, n_heads, d_head])."""
    def hook(z, hook):
        z[:, :z_dev.shape[0], :, :] = z_dev.to(z.dtype)
        return z
    return hook


@torch.no_grad()
def _pp_corrupt_cache(prompt_ids, answer_ids):
    """Corrupt-run hook_z + hook_mlp_out for the FULL teacher-forced sequence
    (prompt+answer), per layer, on device — the freeze values for the direct path."""
    full = list(prompt_ids) + list(answer_ids)
    tok = torch.tensor([full], device=model.cfg.device, dtype=torch.long)
    _, cache = model.run_with_cache(tok, names_filter=_loc_names_filter)
    nL = model.cfg.n_layers
    z = {L: cache[f"blocks.{L}.attn.hook_z"][0].float()
         for L in range(nL) if f"blocks.{L}.attn.hook_z" in cache}
    mlp = {L: cache[f"blocks.{L}.hook_mlp_out"][0].float()
           for L in range(nL) if f"blocks.{L}.hook_mlp_out" in cache}
    return z, mlp


@torch.no_grad()
def _pp_clean_z(prompt_ids):
    """Clean-run hook_z at the prompt positions, per layer (H's clean value)."""
    tok = torch.tensor([prompt_ids], device=model.cfg.device, dtype=torch.long)
    _, cache = model.run_with_cache(tok, names_filter=lambda n: n.endswith("attn.hook_z"))
    pl = len(prompt_ids)
    return {L: cache[f"blocks.{L}.attn.hook_z"][0, :pl].float()
            for L in range(model.cfg.n_layers) if f"blocks.{L}.attn.hook_z" in cache}


def _pp_freeze_hooks(L, ans, zc_corr, mlpc_corr):
    """Freeze every layer AFTER L (attn.hook_z + hook_mlp_out) to corrupt values."""
    hooks = []
    seqlen = None
    for L2 in range(L + 1, model.cfg.n_layers):
        if L2 in zc_corr[ans]:
            hooks.append((f"blocks.{L2}.attn.hook_z", _pp_mk_zfull_hook(zc_corr[ans][L2])))
        if L2 in mlpc_corr[ans]:
            mdev = mlpc_corr[ans][L2]
            hooks.append((f"blocks.{L2}.hook_mlp_out", _loc_mk_seq_hook(mdev, mdev.shape[0])))
    return hooks


def _pp_run(tag, bundle, top_heads):
    ck = f"wo_loc_pp_{tag}"
    items = bundle["items"][: int(CFG["wo_pp_n"])]
    K = min(int(CFG["wo_pp_head_topk"]), len(top_heads))
    heads = [tuple(h) for h in top_heads[:K]]
    fp = {"pair_sha": bundle["pair_sha"], "n": len(items), "targets": bundle["targets"],
          "heads": [list(h) for h in heads]}
    state = {"fp": fp, "done": [], "total": [], "direct": []}
    if has_artifact(ck, "json"):
        prev = load_json(ck)
        if prev.get("fp") == fp:
            state = prev
            _k = len(state["done"])                       # keep per-item arrays aligned to 'done'
            state["total"] = state["total"][:_k]; state["direct"] = state["direct"][:_k]
            log(f"WO#6 pp[{tag}]: resuming path patch ({_k}/{len(items)} items).")
        else:
            log(f"WO#6 pp[{tag}]: stale path-patch ckpt — recompute.")
    wo_load_model(tag)
    for i, item in enumerate(items):
        if i in state["done"]:
            continue
        ca, kk = item["clean_ans_ids"], item["corrupt_ans_ids"]
        pl = item["prompt_len"]
        D_clean = _loc_metric(item["tok_clean"], ca, kk)
        D_corr = _loc_metric(item["tok_corr"], ca, kk)
        zc_clean = _pp_clean_z(item["tok_clean"])
        _z_ca, _m_ca = _pp_corrupt_cache(item["tok_corr"], ca)   # corrupt run, clean-answer continuation
        _z_kk, _m_kk = _pp_corrupt_cache(item["tok_corr"], kk)   # corrupt run, corrupt-answer continuation
        zc_corr = {"clean": _z_ca, "corr": _z_kk}
        mlpc_corr = {"clean": _m_ca, "corr": _m_kk}
        tot, direct = [], []
        for (L, h) in heads:
            if L not in zc_clean:
                tot.append(np.nan); direct.append(np.nan)
                continue
            zhead = zc_clean[L][:, h, :]
            patch = (f"blocks.{L}.attn.hook_z", _loc_mk_z_hook(zhead, h, pl))
            # TOTAL: plain head patch (all downstream free to respond).
            D_tot = _loc_metric(item["tok_corr"], ca, kk, fwd_hooks=[patch])
            tot.append(wo_denoise_recovery(D_tot, D_corr, D_clean))
            # DIRECT: freeze all later attn/MLP to corrupt -> only the residual skip carries H.
            # Each teacher-forced continuation is frozen to ITS OWN corrupt-run cache
            # (clean-answer vs corrupt-answer sequences differ in length, so a single
            # cache cannot serve both); this is the consistent corrupt baseline for that
            # continuation, not a confound — D_dir is the metric D under the direct-only
            # intervention, with downstream behaving as in the (per-answer) corrupt run.
            fz_clean = _pp_freeze_hooks(L, "clean", zc_corr, mlpc_corr)
            fz_corr = _pp_freeze_hooks(L, "corr", zc_corr, mlpc_corr)
            lp_c = _loc_logprob(item["tok_corr"], ca, fwd_hooks=[patch] + fz_clean)
            lp_k = _loc_logprob(item["tok_corr"], kk, fwd_hooks=[patch] + fz_corr)
            D_dir = wo_patch_metric(lp_c, lp_k)
            direct.append(wo_denoise_recovery(D_dir, D_corr, D_clean))
        state["total"].append(tot); state["direct"].append(direct)
        state["done"].append(i)
        save_json(ck, state)
        if (i + 1) % 4 == 0 or i + 1 == len(items):
            log(f"WO#6 pp[{tag}]: path patch {i + 1}/{len(items)}.")
    # aggregate.
    tot = np.array(state["total"], dtype=float)
    dr = np.array(state["direct"], dtype=float)
    out_heads = []
    for j, (L, h) in enumerate(heads):
        col_t = tot[:, j] if tot.size else np.array([])
        col_d = dr[:, j] if dr.size else np.array([])
        tmean = float(np.nanmean(col_t)) if (col_t.size and np.isfinite(col_t).any()) else None
        dmean = float(np.nanmean(col_d)) if (col_d.size and np.isfinite(col_d).any()) else None
        med = (None if (tmean is None or dmean is None) else float(tmean - dmean))
        # direct_share + classification are only meaningful for a head with a real,
        # above-noise POSITIVE total recovery; a negative/near-zero total has no
        # direct/mediated split (a negative tmean would flip the share sign).
        meaningful = (tmean is not None and dmean is not None and tmean > 0.05)
        share = float(dmean / tmean) if meaningful else None
        cls = "INCONCLUSIVE"
        if meaningful:
            cls = "DIRECT" if share >= float(CFG["wo_pp_direct_thr"]) else "MEDIATED"
        out_heads.append({"layer": int(L), "head": int(h), "total_recovery": tmean,
                          "direct_recovery": dmean, "mediated_recovery": med,
                          "direct_share": share, "classification": cls})
    return {"tag": tag, "n_used": len(items), "heads": out_heads,
            "direct_thr": float(CFG["wo_pp_direct_thr"])}


# ----------------------------------------------------------------------------
# B1 — MAKELOV DECOMPOSE-AND-COMPARE at the '=' site.
# ----------------------------------------------------------------------------
@torch.no_grad()
def _dorm_capture_eq(tag, bundle):
    """Per-(B,C) '=' residual (all layers) on the CLEAN C1 surface + B*C target + the
    clean-answer first-token id (for the answer-token readout basis). The decode +
    decompose are FORWARD-ONLY and cheap, so this uses its OWN larger pair set
    (wo_dorm_n, default ~200) — INDEPENDENT of the patching cap wo_loc_n. A large n is
    essential: with r answer-token readout dimensions and n samples, R2_row/R2_null are
    only trustworthy when n >> r (else the readout subspace overfits B*C and the dormant
    contrast is spurious — the low-n failure mode the verdict's under-power guard exists
    to catch). Cached per (dorm pair set, tag)."""
    ck = f"wo_loc_eqresid_{tag}"
    dorm_pairs = list(WO_FSPROBE_PAIRS)[: int(CFG["wo_dorm_n"])]
    dorm_sha = wo_stim_hash(dorm_pairs)
    if has_artifact(ck, "pickle"):
        b = load_pickle(ck)
        if b.get("dorm_sha") == dorm_sha:
            log(f"WO#6 dorm[{tag}]: '=' residual capture cached — reused.")
            return b
    wo_load_model(tag)
    nL = model.cfg.n_layers
    # Apply the final LayerNorm so the decode + decompose live in the EXACT space the
    # unembedding reads (logits = ln_final(resid) @ W_U). This removes the LN-as-isometry
    # hand-wave: "logit-affecting subspace = span(W_U cols)" is then exact. With fold_ln,
    # ln_final is a parameter-free LayerNormPre (center + RMS-normalize). Mocks / models
    # without ln_final fall back to the raw residual.
    _has_ln = hasattr(model, "ln_final") and model.ln_final is not None
    readout_space = "post_ln" if _has_ln else "raw"

    def _eq_resid(cache, eq):
        rstack = torch.stack([cache[f"blocks.{L}.hook_resid_post"][0, eq] for L in range(nL)]).float()
        if _has_ln:
            rstack = model.ln_final(rstack)
        return rstack.detach().cpu().numpy().astype(np.float16)

    seen, rows, prod, ft = set(), [], [], []
    for (B, C) in dorm_pairs:
        if (B, C) in seen:
            continue
        seen.add((B, C))
        p = _loc_render(B, C)                          # "( 0 + B ) * C =" (reused from 82r)
        tok = model.to_tokens(p)[0].tolist()
        strs = [tokenizer.decode([t]).strip() for t in tok]
        loc = wo_locate_operand_spans(strs, B, C)
        if not loc["ok"] or loc["equals"] is None:
            continue
        tokt = torch.tensor([tok], device=model.cfg.device, dtype=torch.long)
        _, cache = model.run_with_cache(tokt, names_filter=lambda n: n.endswith("hook_resid_post"))
        rows.append(_eq_resid(cache, loc["equals"]))
        prod.append(int(B * C))
        ft.append(int(tokenizer(" " + str(int(B * C)), add_special_tokens=False)["input_ids"][0]))
    # answer-token unembedding columns (the logit-affecting readout basis).
    W_U = model.W_U.detach().float().cpu().numpy()          # [d_model, d_vocab]
    uniq = sorted(set(ft))
    cols = np.stack([W_U[:, t] for t in uniq], axis=1)      # [d_model, r]
    cols = cols - cols.mean(axis=1, keepdims=True)          # center: only logit DIFFERENCES matter
    bundle_eq = {"tag": tag, "n_layers": nL, "dorm_sha": dorm_sha,
                 "resid_eq": np.stack(rows) if rows else np.zeros((0, nL, model.cfg.d_model), np.float16),
                 "product": prod, "first_tok": ft, "readout_basis": cols.astype(np.float32),
                 "readout_space": readout_space, "n_unique_answer_tokens": len(uniq)}
    save_pickle(ck, bundle_eq)
    log(f"WO#6 dorm[{tag}]: captured '=' residual ({readout_space}) for {len(rows)} unique (B,C); "
        f"readout basis r={cols.shape[1]} answer tokens.")
    return bundle_eq


def _dorm_run(tag, bundle):
    eqb = _dorm_capture_eq(tag, bundle)
    R = np.asarray(eqb["resid_eq"], dtype=np.float32)         # [n, nL, d]
    y = np.asarray(eqb["product"], dtype=float)
    basis = np.asarray(eqb["readout_basis"], dtype=np.float32)
    nL = eqb["n_layers"]
    if R.shape[0] < 10:
        log(f"WO#6 dorm[{tag}]: only {R.shape[0]} unique pairs — under-powered; reporting anyway.")
    layers = CFG["wo_dorm_layers"]
    if layers is None:
        layers = sorted(set([nL - 1, int(round(0.75 * (nL - 1)))]))
    n_pairs = int(R.shape[0])
    per_layer = []
    for L in layers:
        X = R[:, L, :].astype(float)
        fit = wo_fit_ridge_probe(X, y)
        decomp = None if fit is None else wo_readout_decompose(fit["w"], basis)
        split = wo_readout_decode_split(X, y, basis)
        inert = None if decomp is None else decomp["inert_share"]
        # n threaded in: the verdict downgrades to INCONCLUSIVE when under-powered
        # (n<min_n) or when overall decodability is too weak to support a dormant claim.
        verdict = wo_dormant_verdict(inert, (split or {}).get("R2_row"),
                                     (split or {}).get("R2_null"), (split or {}).get("R2_full"),
                                     n=n_pairs)
        per_layer.append({
            "layer": int(L),
            "inert_share": (None if decomp is None else float(decomp["inert_share"])),
            "row_share": (None if decomp is None else float(decomp["row_share"])),
            "r_dim": (None if split is None else split["r_dim"]),
            "R2_full": (None if split is None else split["R2_full"]),
            "R2_row": (None if split is None else split["R2_row"]),
            "R2_null": (None if split is None else split["R2_null"]),
            "verdict": verdict["label"], "reason": verdict["reason"],
        })

    # B2 — reference the existing nulls (embed the actual numbers if present).
    refs = {
        "WO5.1d_full_swap": ("full-residual swap at the '=' site flips the answer to the donor's "
                             "product (the site carries the answer), yet the probe-direction inject "
                             "is DEAD_DIRECTION — the DECODABLE direction is not a causal handle."),
        "WO5_expB_selectivity": ("OPERANDS_ONLY: product-R^2 at '=' (~0.97) does not exceed the "
                                 "linear-(B,C) baseline (~0.968) — the decode is operand-reconstructible."),
        "decode_R2": "B*C linearly decodable at the '=' site at R^2 ~ 0.96-0.98 (WO#3 / WO#5).",
    }
    for nm, art in [("WO5_expB_selectivity_json", f"probe_selectivity_{tag}"),
                    ("WO5.1d_lateswap_json", f"causal_steering_lateswap_{tag}")]:
        if has_artifact(art, "json"):
            try:
                refs[nm] = load_json(art)
            except Exception:
                pass

    headline = per_layer[-1] if per_layer else {}
    return {
        "tag": tag, "n_pairs": n_pairs, "layers": layers,
        "readout_space": eqb.get("readout_space"),
        "n_unique_answer_tokens": eqb.get("n_unique_answer_tokens"),
        "per_layer": per_layer,
        "headline_layer": (headline.get("layer") if headline else None),
        "headline_verdict": (headline.get("verdict") if headline else None),
        "references_WO5": refs,
        "note": ("decode direction fit by wo_fit_ridge_probe at the clean C1 '=' residual; "
                 "readout basis = centered unembedding columns of the realized clean-answer first "
                 "tokens (the logit-affecting subspace the '=' residual feeds). When the model exposes "
                 "ln_final the residual is taken in POST-LayerNorm space (readout_space='post_ln'), so "
                 "logit-affecting = span(W_U cols) is EXACT (no isometry assumption); else raw. The "
                 "dormant verdict downgrades to INCONCLUSIVE when under-powered (n<10) or R2_full is "
                 "too weak. DAS is NOT used."),
    }


# ----------------------------------------------------------------------------
# Driver — base + instruct.
# ----------------------------------------------------------------------------
WO_PP = {}
WO_DORM = {}
for _tag in CFG["wo_pp_tags"]:
    _bundle = _loc_build(_tag)
    if _bundle["n_used"] == 0:
        log(f"WO#6 82s[{_tag}]: 0 usable STR items — skipping.")
        continue
    # top operand heads from 82r (attribution-ranked); fall back to the saved json.
    _top = None
    if "WO_LOC" in globals() and _tag in WO_LOC:
        _top = [(d["layer"], d["head"]) for d in WO_LOC[_tag]["attribution"]["head_denoise_top"]]
    elif has_artifact(f"operand_position_patch_{_tag}", "json"):
        _top = [(d["layer"], d["head"])
                for d in load_json(f"operand_position_patch_{_tag}")["attribution"]["head_denoise_top"]]
    if not _top:
        log(f"WO#6 82s[{_tag}]: no 82r head ranking available — run 82r first; skipping path patch.")
    else:
        _pp = _pp_run(_tag, _bundle, _top)
        wo_save_result(f"head_path_patch_{_tag}.json", json.dumps(wo_jsonsafe(_pp), indent=2))
        WO_PP[_tag] = _pp
        _nd = sum(1 for h in _pp["heads"] if h["classification"] == "DIRECT")
        _nm = sum(1 for h in _pp["heads"] if h["classification"] == "MEDIATED")
        log(f"WO#6 pp[{_tag}]: {_nd} DIRECT / {_nm} MEDIATED of {len(_pp['heads'])} top heads.")

    _dorm = _dorm_run(_tag, _bundle)
    wo_save_result(f"dormant_certification_{_tag}.json", json.dumps(wo_jsonsafe(_dorm), indent=2))
    WO_DORM[_tag] = _dorm
    log(f"WO#6 dorm[{_tag}]: headline '=' verdict={_dorm['headline_verdict']} "
        f"(layer {_dorm['headline_layer']}); "
        + "; ".join(f"L{p['layer']}: inert={p['inert_share']}, R2_full={p['R2_full']}, "
                    f"R2_row={p['R2_row']}, R2_null={p['R2_null']}" for p in _dorm["per_layer"]))
