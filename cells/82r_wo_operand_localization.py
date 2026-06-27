# ============================================================================
# Phase 6 / WORK ORDER #6 — EXPERIMENT A: OPERAND-ROUTE LOCALIZATION (GPU).
# ----------------------------------------------------------------------------
# Localize WHERE and HOW Llama-3.1-8B computes the multiplication answer on the
# *failing* zero-shot C1 surface "( 0 + B ) * C =" — the positive half of the
# paper. Methodology is the literature's best practice, non-negotiable:
#
#   * Symmetric Token Replacement (STR) counterfactuals (Zhang & Nanda
#     2309.16042): flip ONE operand to a digit-count-matched alternative
#     (C->C' or B->B'), so the clean and corrupt surfaces are token-aligned.
#     NOT Gaussian noising.
#   * A teacher-forced, multi-token logprob-DIFFERENCE metric
#     D = logprob(clean_answer) - logprob(corrupt_answer) (Heimersheim & Nanda
#     2404.15255). A difference against a contrast — NOT a probability, NOT a
#     bare logprob, NOT a first-token logit (robust to a shared first token).
#   * Cheap gradient attribution patching (Syed et al.): one fwd + one bwd of D
#     per answer gives the WHOLE (layer x position) / (layer x head) / (layer)
#     map. We run BOTH directions, stated explicitly:
#       - DENOISING (patch CLEAN into CORRUPT) = sufficiency: does restoring this
#         site recover the clean answer?  (grads taken on the CORRUPT run)
#       - NOISING  (patch CORRUPT into CLEAN) = necessity: does corrupting this
#         site destroy the clean answer?    (grads taken on the CLEAN run)
#       (backup/self-repair can hide necessity, hence we report both.)
#   * EXACT activation patching verifies the attribution (a first-order estimate)
#     at the role x layer residual grid and the top-K attributed heads/MLPs, and
#     we report attribution-vs-exact agreement (Syed et al.'s verification step).
#
# Expected (Stolfo et al. 2023): the operand token positions + a few mid-layer
# heads moving operand->last-token + late MLPs carry the recovery; the '=' answer
# site does NOT (consistent with WO#5.1d). Honest fork (work order §ACCEPTANCE):
# if the operands+heads ARE the locus -> LOCALIZED_OPERAND_ROUTE (Stolfo); if no
# sparse head set recovers the answer -> DISTRIBUTED_NO_LOCUS, the bag-of-
# heuristics result (Nikankin). Either completes the paper.
#
# All pure decision/metric math lives in cell 76 (wo_build_str_counterfactual,
# wo_locate_operand_spans, wo_teacher_forced_logprob, wo_patch_metric,
# wo_denoise_recovery, wo_attribution_score, wo_attribution_exact_agreement,
# wo_topk_sites, wo_localization_verdict) and is unit-tested on CPU first. This
# cell is thin orchestration: STR capture, the fwd/bwd attribution passes, the
# exact-patch verification loop, paired bootstrap CIs, and the heatmaps. Every
# phase is has_artifact-guarded, resumable per item, and runs on base+instruct.
# ============================================================================
import json
import numpy as np
import torch

assert "WO_FSPROBE_PAIRS" in globals() and "wo_build_str_counterfactual" in globals() \
    and "wo_locate_operand_spans" in globals() and "wo_teacher_forced_logprob" in globals() \
    and "wo_patch_metric" in globals() and "wo_denoise_recovery" in globals() \
    and "wo_attribution_score" in globals() and "wo_attribution_exact_agreement" in globals() \
    and "wo_topk_sites" in globals() and "wo_localization_verdict" in globals() \
    and "wo_bootstrap_ci" in globals() and "wo_paired_delta_ci" in globals() \
    and "_wo_mk_patch_hook" in globals() and "wo_parse_int" in globals() \
    and "wo_load_model" in globals() and "WO_CONDITIONS" in globals(), (
    "WO#6 82r needs WO_FSPROBE_PAIRS (cell 82d), the cell-76 WO#6 helpers, and the "
    "cell-82 _wo_mk_patch_hook. Run the earlier cells first.")

# ---- knobs (all CFG so a re-run can retune without editing the cell) ----------
CFG.setdefault("wo_loc_tags", list(CFG.get("wo_steer_tags", ["base", "instruct"])))
CFG.setdefault("wo_loc_n", 64)                 # cap captured pairs (attribution is cheap; exact is the cost)
CFG.setdefault("wo_loc_targets", ["C", "B"])   # corrupt C and corrupt B (each surfaces different heads)
CFG.setdefault("wo_loc_seed", 606)
CFG.setdefault("wo_loc_head_topk", 16)         # # attribution-ranked heads to exact-verify + greedily compose
CFG.setdefault("wo_loc_nboot", 2000)           # bootstrap resamples for CIs
CFG.setdefault("wo_loc_gen_sample", 24)        # # items for the (costlier) greedy-decode gold flip-rate
CFG.setdefault("wo_loc_gen_k", int(globals().get("WO_MAX_NEW_TOKENS", 8)))
CFG.setdefault("wo_loc_recover_thr", 0.4)      # localization verdict threshold (fraction of D restored)
CFG.setdefault("wo_loc_roles", ["plus", "b_last", "rparen", "star", "c_last", "equals"])

_LOC_RENDER_C1 = dict((c[0], c[2]) for c in WO_CONDITIONS)["C1"]   # "( 0 + B ) * C ="
_LOC_PAIRS = list(WO_FSPROBE_PAIRS)[: int(CFG["wo_loc_n"])]
_LOC_PAIR_SHA = wo_stim_hash(_LOC_PAIRS)


def _loc_render(B, C):
    p = _LOC_RENDER_C1(B, C)
    assert p.endswith("=") and not p.endswith(" ")        # Llama tokenizer hazard (no trailing space)
    return p


# ----------------------------------------------------------------------------
# Phase 0 — BUILD: STR counterfactual items per tag. Tokenize the clean + corrupt
#   C1 surfaces, verify token-length alignment (STR), locate operand/structure
#   roles, and record the clean/corrupt answer token sequences. Cached per tag.
# ----------------------------------------------------------------------------
@torch.no_grad()
def _loc_build(tag):
    ck = f"wo_loc_items_{tag}"
    if has_artifact(ck, "pickle"):
        b = load_pickle(ck)
        if b.get("pair_sha") == _LOC_PAIR_SHA and b.get("targets") == list(CFG["wo_loc_targets"]):
            log(f"WO#6 loc[{tag}]: STR item build cached — reused.")
            return b
        log(f"WO#6 loc[{tag}]: cached build stale (pairs/targets changed) — rebuilding.")
    wo_load_model(tag)
    cfg = model.cfg
    rng = np.random.default_rng(int(CFG["wo_loc_seed"]))
    items, n_skip = [], 0
    for (B, C) in _LOC_PAIRS:
        for which in CFG["wo_loc_targets"]:
            cf = wo_build_str_counterfactual(B, C, which, rng, lo=WO_BAND[0], hi=WO_BAND[1])
            if cf is None or not cf["digit_aligned"]:
                n_skip += 1
                continue
            Bp, Cp = cf["Bp"], cf["Cp"]
            p_clean = _loc_render(B, C)
            p_corr = _loc_render(Bp, Cp)
            tok_clean = model.to_tokens(p_clean)[0].tolist()
            tok_corr = model.to_tokens(p_corr)[0].tolist()
            if len(tok_clean) != len(tok_corr):
                n_skip += 1                                # STR alignment broke (tokenizer split)
                continue
            strs = [tokenizer.decode([t]).strip() for t in tok_clean]
            loc = wo_locate_operand_spans(strs, B, C)
            if not loc["ok"]:
                n_skip += 1
                continue
            roles = {
                "plus": loc["plus"], "rparen": loc["rparen"], "star": loc["star"],
                "equals": loc["equals"], "b_last": loc["b_span"][-1], "c_last": loc["c_span"][-1],
            }
            roles = {r: roles[r] for r in CFG["wo_loc_roles"] if roles.get(r) is not None}
            prompt_len = len(tok_clean)
            items.append({
                "B": int(B), "C": int(C), "which": which, "Bp": int(Bp), "Cp": int(Cp),
                "clean_answer": int(cf["clean_answer"]), "corrupt_answer": int(cf["corrupt_answer"]),
                "tok_clean": [int(t) for t in tok_clean], "tok_corr": [int(t) for t in tok_corr],
                "clean_ans_ids": [int(t) for t in tokenizer(" " + str(cf["clean_answer"]), add_special_tokens=False)["input_ids"]],
                "corrupt_ans_ids": [int(t) for t in tokenizer(" " + str(cf["corrupt_answer"]), add_special_tokens=False)["input_ids"]],
                "roles": roles, "prompt_len": int(prompt_len),
                "flip_role": ("c_last" if which == "C" else "b_last"),
            })
    bundle = {
        "tag": tag, "n_layers": int(cfg.n_layers), "n_heads": int(cfg.n_heads),
        "d_head": int(cfg.d_head), "d_model": int(cfg.d_model),
        "n_used": len(items), "n_skipped": n_skip,
        "pair_sha": _LOC_PAIR_SHA, "targets": list(CFG["wo_loc_targets"]), "items": items,
    }
    save_pickle(ck, bundle)
    log(f"WO#6 loc[{tag}]: built {len(items)} STR items (skip={n_skip}) "
        f"over {len(_LOC_PAIRS)} pairs x {len(CFG['wo_loc_targets'])} targets.")
    return bundle


# ----------------------------------------------------------------------------
# Metric + caching primitives (the teacher-forced multi-token logprob-difference).
# ----------------------------------------------------------------------------
def _loc_names_filter(name):
    return (name.endswith("hook_resid_post") or name.endswith("attn.hook_z")
            or name.endswith("hook_mlp_out"))


@torch.no_grad()
def _loc_logprob(prompt_ids, answer_ids, fwd_hooks=None):
    """Teacher-forced sum log-prob of `answer_ids` appended to prompt_ids, under an
    optional list of fwd_hooks (reuses cell-76 wo_teacher_forced_logprob for the
    indexing). NO grad (used for baselines + exact patching)."""
    full = list(prompt_ids) + list(answer_ids)
    tok = torch.tensor([full], device=model.cfg.device, dtype=torch.long)
    lg = model(tok) if not fwd_hooks else model.run_with_hooks(tok, fwd_hooks=fwd_hooks)
    logp = torch.log_softmax(lg[0].float(), dim=-1).cpu().numpy()
    return wo_teacher_forced_logprob(logp, answer_ids, len(prompt_ids))


def _loc_metric(prompt_ids, clean_ans, corrupt_ans, fwd_hooks=None):
    """D = logprob(clean_answer) - logprob(corrupt_answer) (wo_patch_metric)."""
    return wo_patch_metric(_loc_logprob(prompt_ids, clean_ans, fwd_hooks),
                           _loc_logprob(prompt_ids, corrupt_ans, fwd_hooks))


def _loc_fwd_bwd(prompt_ids, answer_ids):
    """ONE fwd + bwd of the teacher-forced sum-logprob of `answer_ids`. Returns
    (value, acts, grads) with acts/grads dicts keyed by hook name (resid_post,
    attn.hook_z, hook_mlp_out), each [1, seq, ...]. Gradients are dL/d(activation)
    for the metric L = sum_t logprob(answer token t) — the Syed-et-al. attribution
    backward. (We compose D's gradient from two such passes: grad_D = grad(clean
    answer) - grad(corrupt answer).)"""
    model.reset_hooks()
    acts, grads = {}, {}

    def _fwd(act, hook):
        acts[hook.name] = act.detach()

    def _bwd(grad, hook):
        grads[hook.name] = grad.detach()

    model.add_hook(_loc_names_filter, _fwd, "fwd")
    model.add_hook(_loc_names_filter, _bwd, "bwd")
    full = list(prompt_ids) + list(answer_ids)
    tok = torch.tensor([full], device=model.cfg.device, dtype=torch.long)
    lg = model(tok)
    logp = torch.log_softmax(lg[0].float(), dim=-1)
    start = len(prompt_ids)
    metric = logp.new_zeros(())
    for t, a in enumerate(answer_ids):
        metric = metric + logp[start + t - 1, int(a)]
    metric.backward()
    model.reset_hooks()
    return float(metric.item()), acts, grads


def _loc_slice(cache, name, prompt_len):
    """[0, :prompt_len] of a cached activation as float32 numpy (drops batch +
    the teacher-forced answer positions; the prompt region is what we patch)."""
    return cache[name][0, :prompt_len].float().cpu().numpy()


# ----------------------------------------------------------------------------
# Phase 1 — ATTRIBUTION: the cheap whole-graph map (both directions), per item.
# ----------------------------------------------------------------------------
def _loc_grad_D(prompt_ids, clean_ans, corrupt_ans):
    """Two fwd+bwd passes on a run: the teacher-forced sum-logprob of the clean
    answer and of the corrupt answer. Returns (acts, grad_clean, grad_corrupt) —
    the activation values (from the clean-answer pass; the prompt region is shared
    by causality) and the per-answer gradient caches. grad of D = lp(clean) -
    lp(corrupt) is grad_clean - grad_corrupt, taken AFTER slicing to the shared
    prompt positions (the two answers give different sequence lengths)."""
    _, acts_c, grads_c = _loc_fwd_bwd(prompt_ids, clean_ans)
    _, _, grads_k = _loc_fwd_bwd(prompt_ids, corrupt_ans)
    return acts_c, grads_c, grads_k


def _loc_attr_maps(item, nL, nH):
    """Attribution maps for ONE item, BOTH directions. Denoising grads on the
    CORRUPT run, noising grads on the CLEAN run; clean/corrupt activation values
    from each run. Returns dict of numpy arrays:
        resid_deno [roles, nL], resid_nois [roles, nL]
        head_deno  [nL, nH],    mlp_deno   [nL]
    (heads/mlp reported for the denoising/sufficiency direction; resid both)."""
    pl = item["prompt_len"]
    roles = list(item["roles"].keys())
    ca, ck = item["clean_ans_ids"], item["corrupt_ans_ids"]

    # Activations + per-answer grads on each run (grad_D computed AFTER slicing to
    # the shared prompt region, since the two answers differ in length).
    acts_clean, gc_clean, gk_clean = _loc_grad_D(item["tok_clean"], ca, ck)   # clean run
    acts_corr, gc_corr, gk_corr = _loc_grad_D(item["tok_corr"], ca, ck)       # corrupt run

    def A(name, cache):
        return _loc_slice(cache, name, pl)

    def gD(name, gc, gk):
        return _loc_slice(gc, name, pl) - _loc_slice(gk, name, pl)            # grad of D, prompt region

    resid_deno = np.zeros((len(roles), nL), dtype=np.float64)
    resid_nois = np.zeros((len(roles), nL), dtype=np.float64)
    head_deno = np.zeros((nL, nH), dtype=np.float64)
    mlp_deno = np.zeros((nL,), dtype=np.float64)
    for L in range(nL):
        rp = f"blocks.{L}.hook_resid_post"
        zk = f"blocks.{L}.attn.hook_z"
        mp = f"blocks.{L}.hook_mlp_out"
        a_clean_r = A(rp, acts_clean); a_corr_r = A(rp, acts_corr)
        g_corr_r = gD(rp, gc_corr, gk_corr)    # grad_D on corrupt run (denoising)
        g_clean_r = gD(rp, gc_clean, gk_clean)  # grad_D on clean run (noising)
        for ri, r in enumerate(roles):
            p = item["roles"][r]
            # DENOISING: replace corrupt with clean, grads on corrupt run.
            resid_deno[ri, L] = wo_attribution_score(a_clean_r[p], a_corr_r[p], g_corr_r[p])
            # NOISING: replace clean with corrupt, grads on clean run.
            resid_nois[ri, L] = wo_attribution_score(a_corr_r[p], a_clean_r[p], g_clean_r[p])
        if zk in acts_clean and zk in acts_corr and zk in gc_corr:
            a_clean_z = A(zk, acts_clean); a_corr_z = A(zk, acts_corr); g_corr_z = gD(zk, gc_corr, gk_corr)
            for h in range(nH):
                head_deno[L, h] = wo_attribution_score(a_clean_z[:, h, :], a_corr_z[:, h, :], g_corr_z[:, h, :])
        if mp in acts_clean and mp in acts_corr and mp in gc_corr:
            mlp_deno[L] = wo_attribution_score(A(mp, acts_clean), A(mp, acts_corr), gD(mp, gc_corr, gk_corr))
    return {"roles": roles, "resid_deno": resid_deno, "resid_nois": resid_nois,
            "head_deno": head_deno, "mlp_deno": mlp_deno}


def _loc_attribution(tag, bundle):
    ck = f"wo_loc_attr_{tag}"
    nL, nH = bundle["n_layers"], bundle["n_heads"]
    items = bundle["items"]
    fp = {"pair_sha": _LOC_PAIR_SHA, "n": len(items), "targets": bundle["targets"],
          "roles": list(CFG["wo_loc_roles"]), "seed": int(CFG["wo_loc_seed"])}
    state = {"fp": fp, "done": [], "resid_deno": [], "resid_nois": [],
             "head_deno": [], "mlp_deno": []}
    if has_artifact(ck, "pickle"):
        prev = load_pickle(ck)
        if prev.get("fp") == fp:
            state = prev
            _k = len(state["done"])                       # keep per-item arrays aligned to 'done'
            for _key in ("resid_deno", "resid_nois", "head_deno", "mlp_deno"):
                state[_key] = state[_key][:_k]
            log(f"WO#6 loc[{tag}]: resuming attribution ({_k}/{len(items)} items).")
        else:
            log(f"WO#6 loc[{tag}]: stale attribution ckpt — recompute.")
    wo_load_model(tag)
    for i, item in enumerate(items):
        if i in state["done"]:
            continue
        m = _loc_attr_maps(item, nL, nH)
        state["resid_deno"].append(m["resid_deno"]); state["resid_nois"].append(m["resid_nois"])
        state["head_deno"].append(m["head_deno"]); state["mlp_deno"].append(m["mlp_deno"])
        state["done"].append(i)
        save_pickle(ck, state)                            # commit AFTER both data + done -> consistent ckpt
        if (i + 1) % 8 == 0 or i + 1 == len(items):
            log(f"WO#6 loc[{tag}]: attribution {i + 1}/{len(items)}.")
    state["role_order"] = list(items[0]["roles"].keys()) if items else []
    assert len(state["resid_deno"]) == len(state["done"]), "attribution array/done misalignment"
    if state["head_deno"] and not np.any(np.stack(state["head_deno"])):
        log(f"WO#6 loc[{tag}]: WARNING — all head attributions are zero; the model may not "
            "expose attn.hook_z (unexpected architecture). Head localization will be empty.")
    return state


# ----------------------------------------------------------------------------
# Phase 2 — EXACT verification: activation patching (denoising) at the role x
#   layer residual grid + the top-K attributed heads (greedy compose) + MLPs.
# ----------------------------------------------------------------------------
def _loc_mk_z_hook(z_clean_dev, head, upto):
    def hook(z, hook):
        z[:, :upto, head, :] = z_clean_dev[:upto, :].to(z.dtype)
        return z
    return hook


def _loc_mk_multi_z_hook(z_clean_layer_dev, heads, upto):
    def hook(z, hook):
        for h in heads:
            z[:, :upto, h, :] = z_clean_layer_dev[:upto, h, :].to(z.dtype)
        return z
    return hook


def _loc_mk_seq_hook(seq_clean_dev, upto):
    def hook(act, hook):
        act[:, :upto, :] = seq_clean_dev[:upto, :].to(act.dtype)
        return act
    return hook


@torch.no_grad()
def _loc_clean_caches(item):
    """One clean forward; return per-layer clean residual / hook_z / mlp_out at the
    prompt positions (numpy float32), reused for every exact patch of this item."""
    pl = item["prompt_len"]
    tok = torch.tensor([item["tok_clean"]], device=model.cfg.device, dtype=torch.long)
    _, cache = model.run_with_cache(tok, names_filter=_loc_names_filter)
    nL = model.cfg.n_layers
    resid = {L: cache[f"blocks.{L}.hook_resid_post"][0, :pl].float().cpu().numpy() for L in range(nL)}
    zc = {L: cache[f"blocks.{L}.attn.hook_z"][0, :pl].float().cpu().numpy()
          for L in range(nL) if f"blocks.{L}.attn.hook_z" in cache}
    mlp = {L: cache[f"blocks.{L}.hook_mlp_out"][0, :pl].float().cpu().numpy()
           for L in range(nL) if f"blocks.{L}.hook_mlp_out" in cache}
    return resid, zc, mlp


@torch.no_grad()
def _loc_gold_flip(item, L, pos, resid_vec):
    """Greedy-decode the CORRUPT surface with the CLEAN residual patched at (L,pos);
    True iff the decoded integer == the clean answer B*C (the gold flip-rate)."""
    dv = torch.tensor(resid_vec.astype(np.float32), device=model.cfg.device)
    hook = (f"blocks.{L}.hook_resid_post", _wo_mk_patch_hook(dv, pos))
    ids = list(item["tok_corr"])
    for _ in range(int(CFG["wo_loc_gen_k"])):
        tok = torch.tensor([ids], device=model.cfg.device, dtype=torch.long)
        lg = model.run_with_hooks(tok, fwd_hooks=[hook])
        ids.append(int(lg[0, -1].argmax().item()))
    return wo_parse_int(tokenizer.decode(ids[len(item["tok_corr"]):])) == item["clean_answer"]


def _loc_exact(tag, bundle, attr):
    ck = f"wo_loc_exact_{tag}"
    nL, nH = bundle["n_layers"], bundle["n_heads"]
    items = bundle["items"]
    roles = list(CFG["wo_loc_roles"])
    roles = [r for r in roles if items and r in items[0]["roles"]]
    # Attribution-rank the heads (mean over items) to pick the top-K to verify.
    head_mean = np.mean(np.stack(attr["head_deno"]), axis=0) if attr["head_deno"] else np.zeros((nL, nH))
    flat = head_mean.ravel()
    topk_idx = wo_topk_sites(flat, int(CFG["wo_loc_head_topk"]))
    top_heads = [(int(i // nH), int(i % nH)) for i in topk_idx]
    assert len(set(top_heads)) == len(top_heads), "top_heads has duplicate (layer,head) entries"

    fp = {"pair_sha": _LOC_PAIR_SHA, "n": len(items), "roles": roles, "targets": bundle["targets"],
          "top_heads": [list(h) for h in top_heads], "seed": int(CFG["wo_loc_seed"])}
    state = {"fp": fp, "done": [],
             "D_clean": [], "D_corrupt": [],
             "resid_rec": [],          # per item: [roles, nL] denoising recovery
             "head_rec": [],           # per item: [K] single-head recovery (top heads order)
             "head_greedy": [],        # per item: [K] cumulative recovery (greedy add top heads)
             "mlp_rec": [],            # per item: [nL]
             "gold_flip": [],          # per item (sampled): 1/0
             "flip_layer": None}
    _per_item_keys = ("D_clean", "D_corrupt", "resid_rec", "head_rec", "head_greedy", "mlp_rec")
    if has_artifact(ck, "json"):
        prev = load_json(ck)
        if prev.get("fp") == fp:
            state = prev
            _k = len(state["done"])                       # keep per-item arrays aligned to 'done'
            for _key in _per_item_keys:
                state[_key] = state[_key][:_k]
            state["gold_flip"] = state["gold_flip"][:min(_k, int(CFG["wo_loc_gen_sample"]))]
            log(f"WO#6 loc[{tag}]: resuming exact patch ({_k}/{len(items)} items).")
        else:
            log(f"WO#6 loc[{tag}]: stale exact ckpt — recompute.")
    wo_load_model(tag)

    # Choose the layer for the gold-flip greedy probe: the flipped-operand role's
    # best DENOISING-attribution layer (pre-registered from attribution, not the
    # exact recovery, to avoid peeking).
    rd_mean = np.mean(np.stack(attr["resid_deno"]), axis=0) if attr["resid_deno"] else np.zeros((len(roles), nL))
    flip_role = items[0]["flip_role"] if items else "c_last"
    fr_i = roles.index(flip_role) if flip_role in roles else 0
    flip_layer = int(np.argmax(np.nan_to_num(np.abs(rd_mean[fr_i]), nan=-1.0))) if rd_mean.size else 0
    state["flip_layer"] = flip_layer

    for i, item in enumerate(items):
        if i in state["done"]:
            continue
        ca, ck_ans = item["clean_ans_ids"], item["corrupt_ans_ids"]
        pl = item["prompt_len"]
        assert len(item["tok_clean"]) == len(item["tok_corr"]) == pl, (
            f"STR alignment broke for item {i} (clean/corrupt prompt lengths differ); "
            "the clean residual patched at a role position would land on a mismatched token.")
        D_clean = _loc_metric(item["tok_clean"], ca, ck_ans)
        D_corr = _loc_metric(item["tok_corr"], ca, ck_ans)
        rc, zc, mlpc = _loc_clean_caches(item)

        # --- residual role x layer exact denoising recovery ---
        rr = np.full((len(roles), nL), np.nan)
        for ri, r in enumerate(roles):
            pos = item["roles"][r]
            for L in range(nL):
                dv = torch.tensor(rc[L][pos].astype(np.float32), device=model.cfg.device)
                D_p = _loc_metric(item["tok_corr"], ca, ck_ans,
                                  fwd_hooks=[(f"blocks.{L}.hook_resid_post", _wo_mk_patch_hook(dv, pos))])
                rr[ri, L] = wo_denoise_recovery(D_p, D_corr, D_clean)

        # --- top-K heads: single-head recovery + greedy cumulative recovery ---
        hr = np.full((len(top_heads),), np.nan)
        for j, (L, h) in enumerate(top_heads):
            if L in zc:
                zdev = torch.tensor(zc[L].astype(np.float32), device=model.cfg.device)
                D_p = _loc_metric(item["tok_corr"], ca, ck_ans,
                                  fwd_hooks=[(f"blocks.{L}.attn.hook_z", _loc_mk_z_hook(zdev[:, h, :], h, pl))])
                hr[j] = wo_denoise_recovery(D_p, D_corr, D_clean)
        # greedy: each step j patches the CUMULATIVE set of the top-(j+1) heads
        # jointly (heads from the same layer collected into one hook_z patch), so
        # hg[j] is the recovery from restoring the top j+1 heads together — a true
        # joint measurement (no additivity assumption), yielding n_heads_for_half.
        hg = np.full((len(top_heads),), np.nan)
        by_layer = {}
        for j, (L, h) in enumerate(top_heads):
            by_layer.setdefault(L, []).append(h)
            hooks = []
            for L2, hs in by_layer.items():
                if L2 in zc:
                    zdev = torch.tensor(zc[L2].astype(np.float32), device=model.cfg.device)
                    hooks.append((f"blocks.{L2}.attn.hook_z", _loc_mk_multi_z_hook(zdev, list(hs), pl)))
            D_p = _loc_metric(item["tok_corr"], ca, ck_ans, fwd_hooks=hooks)
            hg[j] = wo_denoise_recovery(D_p, D_corr, D_clean)

        # --- MLP-out per layer exact denoising recovery ---
        mr = np.full((nL,), np.nan)
        for L in range(nL):
            if L in mlpc:
                mdev = torch.tensor(mlpc[L].astype(np.float32), device=model.cfg.device)
                D_p = _loc_metric(item["tok_corr"], ca, ck_ans,
                                  fwd_hooks=[(f"blocks.{L}.hook_mlp_out", _loc_mk_seq_hook(mdev, pl))])
                mr[L] = wo_denoise_recovery(D_p, D_corr, D_clean)

        state["D_clean"].append(D_clean); state["D_corrupt"].append(D_corr)
        state["resid_rec"].append(rr.tolist()); state["head_rec"].append(hr.tolist())
        state["head_greedy"].append(hg.tolist()); state["mlp_rec"].append(mr.tolist())
        if i < int(CFG["wo_loc_gen_sample"]):
            pos = item["roles"].get(flip_role, item["roles"][roles[0]])
            state["gold_flip"].append(1.0 if _loc_gold_flip(item, flip_layer, pos, rc[flip_layer][pos]) else 0.0)
        state["done"].append(i)
        save_json(ck, state)
        if (i + 1) % 4 == 0 or i + 1 == len(items):
            log(f"WO#6 loc[{tag}]: exact patch {i + 1}/{len(items)}.")
    # Metric-sign sanity: STR + a correct contrast should give D_clean>0, D_corrupt<0
    # on (nearly) every item. Report violations honestly rather than asserting (one
    # off-sign item must not crash a multi-hour run; the aggregate is robust).
    _dc = np.array(state["D_clean"], float); _dk = np.array(state["D_corrupt"], float)
    if _dc.size:
        _bad = int(np.sum(_dc <= 0) + np.sum(_dk >= 0))
        log(f"WO#6 loc[{tag}]: D_clean mean={_dc.mean():+.2f} (>0 on {int(np.mean(_dc>0)*100)}%), "
            f"D_corrupt mean={_dk.mean():+.2f} (<0 on {int(np.mean(_dk<0)*100)}%); "
            f"{_bad} off-sign metric reads (expected ~0 if STR + metric are sound).")
    state["roles"] = roles
    state["top_heads"] = top_heads
    return state


# ----------------------------------------------------------------------------
# Phase 3 — AGGREGATE + verdict + deliverables.
# ----------------------------------------------------------------------------
def _loc_ci(per_item_vals):
    lo, hi = wo_bootstrap_ci(np.asarray(per_item_vals, dtype=float),
                             n_boot=int(CFG["wo_loc_nboot"]), seed=int(CFG["wo_loc_seed"]))
    return [lo, hi]


def _loc_aggregate(tag, bundle, attr, exact):
    nL, nH = bundle["n_layers"], bundle["n_heads"]
    roles = exact["roles"]
    items = bundle["items"]
    # The attribution role×layer rows are built per item in dict-insertion order;
    # the exact grid uses the explicit CFG order. Assert they match before we compare
    # the two grids (attribution-vs-exact agreement) so a row never silently shifts.
    assert attr.get("role_order", roles) == roles, "attribution/exact role order mismatch"
    flip_role = items[0]["flip_role"] if items else "c_last"
    fr_i = roles.index(flip_role) if flip_role in roles else 0

    rd = np.mean(np.stack(attr["resid_deno"]), axis=0).tolist()      # attribution [roles, nL]
    rn = np.mean(np.stack(attr["resid_nois"]), axis=0).tolist()
    hd = np.mean(np.stack(attr["head_deno"]), axis=0)                # [nL, nH]
    md = np.mean(np.stack(attr["mlp_deno"]), axis=0).tolist()

    resid_rec = np.array(exact["resid_rec"], dtype=float)            # [items, roles, nL]
    head_rec = np.array(exact["head_rec"], dtype=float)             # [items, K]
    head_greedy = np.array(exact["head_greedy"], dtype=float)       # [items, K]
    mlp_rec = np.array(exact["mlp_rec"], dtype=float)               # [items, nL]

    rec_mean = np.nanmean(resid_rec, axis=0)                        # [roles, nL]
    # flipped-operand role: best layer recovery + CI, paired vs the '=' role.
    fr_curve = rec_mean[fr_i]
    fr_best_L = int(np.nanargmax(fr_curve))
    operand_pos_recovery = float(fr_curve[fr_best_L])
    operand_ci = _loc_ci(resid_rec[:, fr_i, fr_best_L])
    eq_i = roles.index("equals") if "equals" in roles else None
    paired_vs_equals = None
    if eq_i is not None:
        eq_best_L = int(np.nanargmax(rec_mean[eq_i]))
        paired_vs_equals = list(wo_paired_delta_ci(resid_rec[:, fr_i, fr_best_L],
                                                   resid_rec[:, eq_i, eq_best_L],
                                                   n_boot=int(CFG["wo_loc_nboot"]),
                                                   seed=int(CFG["wo_loc_seed"])))

    head_mean = np.nanmean(head_rec, axis=0) if head_rec.size else np.array([])
    greedy_mean = np.nanmean(head_greedy, axis=0) if head_greedy.size else np.array([])
    best_head_recovery = float(np.nanmax(head_mean)) if head_mean.size else 0.0
    best_head_idx = int(np.nanargmax(head_mean)) if head_mean.size else -1
    n_heads_for_half = None
    for k, v in enumerate(greedy_mean.tolist(), start=1):
        if v is not None and not np.isnan(v) and v >= 0.5:
            n_heads_for_half = k
            break
    mlp_mean = np.nanmean(mlp_rec, axis=0).tolist() if mlp_rec.size else []

    # attribution-vs-exact agreement on the residual role x layer grid.
    attr_grid = np.array(rd, dtype=float).ravel()
    exact_grid = rec_mean.ravel()
    mask = ~np.isnan(exact_grid)
    agreement = wo_attribution_exact_agreement(attr_grid[mask], exact_grid[mask],
                                               top_k=min(6, int(mask.sum()) or 1))

    verdict = wo_localization_verdict(operand_pos_recovery, best_head_recovery,
                                      n_heads_for_half, recover_thr=float(CFG["wo_loc_recover_thr"]))

    top_heads = exact["top_heads"]
    out = {
        "tag": tag, "n_used": len(items), "targets": bundle["targets"],
        "roles": roles, "n_layers": nL, "n_heads": nH, "flip_role": flip_role,
        "D_clean_mean": float(np.mean(exact["D_clean"])) if exact["D_clean"] else None,
        "D_corrupt_mean": float(np.mean(exact["D_corrupt"])) if exact["D_corrupt"] else None,
        "attribution": {"resid_denoise": rd, "resid_noise": rn, "mlp_denoise": md,
                        "head_denoise_top": [{"layer": L, "head": h, "attr": float(hd[L, h])}
                                             for (L, h) in top_heads]},
        "exact": {
            "resid_recovery": rec_mean.tolist(),
            "operand_pos_recovery": operand_pos_recovery,
            "operand_pos_best_layer": fr_best_L, "operand_pos_ci": operand_ci,
            "paired_operand_vs_equals_ci": paired_vs_equals,
            "head_recovery_top": [{"layer": top_heads[j][0], "head": top_heads[j][1],
                                   "recovery": (None if np.isnan(head_mean[j]) else float(head_mean[j]))}
                                  for j in range(len(top_heads))],
            "best_head_recovery": best_head_recovery,
            "best_head": (list(top_heads[best_head_idx]) if best_head_idx >= 0 else None),
            "head_greedy_cumulative": greedy_mean.tolist(),
            "n_heads_for_half": n_heads_for_half,
            "mlp_recovery": mlp_mean,
            "gold_flip_rate": (float(np.mean(exact["gold_flip"])) if exact["gold_flip"] else None),
            "gold_flip_layer": exact["flip_layer"],
        },
        "attribution_vs_exact_agreement": agreement,
        "verdict": verdict,
    }
    return out


def _loc_plot(tag, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        roles = out["roles"]; nL = out["n_layers"]
        attr = np.array(out["attribution"]["resid_denoise"], dtype=float)
        exact = np.array(out["exact"]["resid_recovery"], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), squeeze=False)
        for ax, mat, ttl, cmap in [
            (axes[0][0], attr, "attribution (denoise)  ΔD estimate", "RdBu_r"),
            (axes[0][1], exact, "exact recovery (denoise)  frac of D", "viridis")]:
            vmax = np.nanmax(np.abs(mat)) if np.isfinite(np.nanmax(np.abs(mat))) else 1.0
            kw = dict(aspect="auto", cmap=cmap)
            if cmap == "RdBu_r":
                kw.update(vmin=-vmax, vmax=vmax)
            else:
                kw.update(vmin=0.0, vmax=1.0)
            im = ax.imshow(mat, **kw)
            ax.set_yticks(range(len(roles))); ax.set_yticklabels(roles, fontsize=8)
            ax.set_xlabel("layer"); ax.set_title(f"{tag}: {ttl}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"WO#6 operand-route localization — {tag}  "
                     f"(verdict: {out['verdict']['label']})", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(WO_RESULTS / f"operand_position_patch_{tag}.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        log(f"(WO#6 loc heatmap [{tag}] skipped: {e})")


# ----------------------------------------------------------------------------
# Driver — base + instruct.
# ----------------------------------------------------------------------------
WO_LOC = {}
_loc_csv_rows = []
for _tag in CFG["wo_loc_tags"]:
    _bundle = _loc_build(_tag)
    if _bundle["n_used"] == 0:
        log(f"WO#6 loc[{_tag}]: 0 usable STR items — skipping (check tokenizer alignment).")
        continue
    _attr = _loc_attribution(_tag, _bundle)
    _exact = _loc_exact(_tag, _bundle, _attr)
    _out = _loc_aggregate(_tag, _bundle, _attr, _exact)
    wo_save_result(f"operand_position_patch_{_tag}.json", json.dumps(wo_jsonsafe(_out), indent=2))
    _loc_plot(_tag, _out)
    WO_LOC[_tag] = _out
    log(f"WO#6 loc[{_tag}]: verdict={_out['verdict']['label']}  "
        f"operand_recovery={_out['exact']['operand_pos_recovery']:.3f}  "
        f"best_head_recovery={_out['exact']['best_head_recovery']:.3f}  "
        f"gold_flip={_out['exact']['gold_flip_rate']}  "
        f"D_clean={_out['D_clean_mean']}  D_corrupt={_out['D_corrupt_mean']}")
    # one summary row per (tag, role) — exact best-layer recovery + attribution.
    for ri, r in enumerate(_out["roles"]):
        rec = np.array(_out["exact"]["resid_recovery"][ri], dtype=float)
        att = np.array(_out["attribution"]["resid_denoise"][ri], dtype=float)
        bestL = int(np.nanargmax(rec)) if np.isfinite(rec).any() else -1
        _loc_csv_rows.append({
            "tag": _tag, "role": r, "best_layer": bestL,
            "exact_recovery_best": (float(rec[bestL]) if bestL >= 0 else None),
            "attr_denoise_best": (float(att[int(np.argmax(np.abs(att)))]) if att.size else None),
            "verdict": _out["verdict"]["label"],
        })

if _loc_csv_rows:
    wo_save_result("operand_localization_summary.csv",
                   wo_battery_csv(_loc_csv_rows,
                                  ["tag", "role", "best_layer", "exact_recovery_best",
                                   "attr_denoise_best", "verdict"]))
    log("WO#6 loc: wrote operand_localization_summary.csv + per-tag json/png.")
