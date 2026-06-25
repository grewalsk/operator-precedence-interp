# ============================================================================
# Phase 6 / WO — GPU SETUP: model registry, memory-safe reload, tag-namespaced
#                evaluation. Thin orchestration over Phase 3's validated,
#                resumable _eval_prompts / parse_int and Phase 0's TL load path.
# ----------------------------------------------------------------------------
# Two models run in ONE session (base 2x2 first, then Instruct). To keep their
# results from colliding, EVERY forward-pass cache key is namespaced by a model
# TAG ('base' | 'instruct'). The reload helper tears down the previous model and
# frees GPU memory before loading the next, then re-asserts left padding (Phase 3
# scores the true last column under left padding).
#
# BOS: we do NOT change tokenization behaviour. The battery reuses Phase 3's
# _eval_prompts -> _encode (tokenizer(..., add_special_tokens=True)) and G4 uses
# model.to_tokens (prepend_bos=True). prepend_bos is therefore inherited identical
# to the validated pipeline; we only RECORD the effective value for repro.txt.
# ============================================================================
import gc
import torch

# Phase 3 must have defined the resumable evaluator + parser (run Phase 3 first).
assert "_eval_prompts" in globals() and "parse_int" in globals(), (
    "Phase 6 needs Phase 3 helpers (_eval_prompts, parse_int). Run Phases 0-5 first "
    "(top-to-bottom), then this work-order section.")
assert "wo_evaluate_gates" in globals() and "wo_build_pairs" in globals(), (
    "Phase 6 needs the WO logic cell (76_wo_logic) — run it before this cell.")

# Effective BOS / prepend setting actually used by the validated pipeline.
def _wo_detect_prepend_bos():
    try:
        with_sp = tokenizer("0", add_special_tokens=True)["input_ids"]
        no_sp = tokenizer("0", add_special_tokens=False)["input_ids"]
        return bool(len(with_sp) > len(no_sp))
    except Exception:
        return None

WO_PREPEND_BOS = _wo_detect_prepend_bos()
CFG["wo_prepend_bos"] = WO_PREPEND_BOS
log(f"WO: effective prepend_bos = {WO_PREPEND_BOS} (inherited from the validated pipeline).")

# Greedy budget: the work order (§5) specifies K=8 new tokens (max product 2401 <= 4
# digits). Phase 3 defaulted to 6; set the EFFECTIVE budget the WO battery uses to 8
# so the decode budget recorded in repro.txt matches what actually ran. Only affects
# WO evals (fresh wo_* cache keys); Phase 0-5 already cached at their own budget.
CFG["g3_max_answer_tokens"] = WO_MAX_NEW_TOKENS
log(f"WO: greedy max_new_tokens set to {WO_MAX_NEW_TOKENS} for the WO battery (§5).")

# Single shared add_special_tokens flag so the parity check and the scored pipeline
# (Phase 3 _encode) cannot drift. The scored pipeline tokenizes with the default
# add_special_tokens=True; mirror that exactly here.
WO_ADD_SPECIAL_TOKENS = True

# Which model is live right now. Phase 0 loaded CFG['model_name'] (base by default).
def _tag_for_name(name):
    for tag, n in WO_MODEL_REGISTRY.items():
        if n == name:
            return tag
    return name
WO_ACTIVE_TAG = _tag_for_name(CFG.get("model_name", WO_MODEL_REGISTRY["base"]))
log(f"WO: active model tag at Phase 6 start = '{WO_ACTIVE_TAG}' ({CFG.get('model_name')}).")

# Record per-tag resolved revisions for repro.txt as we encounter them.
WO_MODEL_REVISIONS = globals().get("WO_MODEL_REVISIONS", {})


def _wo_resolve_revision(repo_id):
    try:
        import os
        from huggingface_hub import model_info
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        return getattr(model_info(repo_id, token=tok), "sha", None)
    except Exception:
        return None


def wo_load_model(tag):
    """Load WO_MODEL_REGISTRY[tag] as the global `model`/`tokenizer`, freeing the
    previous model first. Reuses Phase 0's transformer_lens load path (fold_ln,
    uncentered resid — IDENTICAL to the validated instrument). Idempotent: a no-op
    if the requested tag is already live."""
    global model, tokenizer, WO_ACTIVE_TAG, USING_FALLBACK
    import gc                       # local rebind: defensive against a global `gc` that some
    #                                other cell's loop variable may have shadowed to a dict
    #                                (then gc.collect() in the teardown would AttributeError).
    if tag not in WO_MODEL_REGISTRY:
        raise ValueError(f"unknown model tag {tag!r}; expected {list(WO_MODEL_REGISTRY)}")
    name = WO_MODEL_REGISTRY[tag]
    if WO_ACTIVE_TAG == tag and "model" in globals() and model is not None:
        log(f"WO: model '{tag}' ({name}) already live — reuse.")
        WO_MODEL_REVISIONS.setdefault(tag, _wo_resolve_revision(name))
        return model

    # ---- tear down the previous model + free GPU memory ----
    if "model" in globals() and model is not None:
        try:
            del model
        except Exception:
            pass
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("WO: previous model freed (gc + cuda empty_cache).")

    _device = CFG.get("device", "cuda")
    import os
    _tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    log(f"WO: loading '{tag}' = {name} on {_device} ...")
    try:
        import transformer_lens
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _hf_tok = AutoTokenizer.from_pretrained(name, token=_tok)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, token=_tok)
        model = HookedTransformer.from_pretrained(
            name, hf_model=_hf_model, tokenizer=_hf_tok, dtype=torch.bfloat16,
            device=_device, fold_ln=True, center_writing_weights=False,
            center_unembed=False)
        tokenizer = model.tokenizer
        del _hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        USING_FALLBACK = False
    except Exception as e:
        log(f"WO: transformer_lens load failed ({type(e).__name__}: {e}); HF fallback.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _hf_tok = AutoTokenizer.from_pretrained(name, token=_tok)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, token=_tok,
            attn_implementation="eager").to(_device)
        _hf_model.eval()
        model = HFHookedWrapper(_hf_model, _hf_tok, _device)
        tokenizer = model.tokenizer
        USING_FALLBACK = True

    # Phase 3 scores the true last column under LEFT padding — enforce it.
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    if tokenizer.pad_token_id is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass
    WO_ACTIVE_TAG = tag
    CFG["model_name"] = name      # keep CFG in sync with the live model
    WO_MODEL_REVISIONS[tag] = _wo_resolve_revision(name)
    log(f"WO: '{tag}' live (n_layers={model.cfg.n_layers}, fallback={USING_FALLBACK}, "
        f"revision={WO_MODEL_REVISIONS.get(tag)}).")
    return model


def wo_eval(prompts, key, tag):
    """Resumable greedy decode, namespaced by model tag so base/instruct caches
    never collide. Delegates to Phase 3's fingerprinted _eval_prompts."""
    return _eval_prompts(list(prompts), f"wo_{tag}_{key}")


def wo_run_battery(tag, conditions, pairs, cache_tag=None):
    """Run a list of (key,name,render,gt) conditions on the live `tag` model from
    the SHARED `pairs`. `tag` identifies the LIVE MODEL (asserted); `cache_tag`
    (default=tag) namespaces the forward-pass caches. They differ only for the
    chat-wrapped secondary battery, which reuses the live 'instruct' model but
    must cache under a distinct namespace ('instruct_chat'). Returns
    {key: summary-dict-with-correct_mask, plus 'preds'/'golds'/'prompts'}."""
    cache_tag = cache_tag or tag
    assert WO_ACTIVE_TAG == tag, (
        f"wo_run_battery(tag={tag!r}) but live model is {WO_ACTIVE_TAG!r}; "
        f"call wo_load_model({tag!r}) first.")
    out = {}
    for key, name, render, gt in conditions:
        prompts = [render(B, C) for (B, C) in pairs]
        golds = [gt(B, C) for (B, C) in pairs]
        conts = wo_eval(prompts, key, cache_tag)
        preds = [parse_int(c) for c in conts]
        summ = wo_summarize(preds, golds)
        summ["name"] = name
        summ["preds"] = preds
        summ["golds"] = golds
        summ["prompts"] = prompts
        out[key] = summ
        log(f"  [{tag}/{key}] {name}: acc={summ['exact_acc']:.3f} "
            f"corr={summ['corr']} parse_fail={summ['parse_fail_rate']:.3f}")
    return out


def wo_assert_parity(pairs, render_left, render_right, max_check=50):
    """G2 parity on the LIVE tokenizer for the C1/C2 localization pair (§5.4).
    C1=( 0 + B ) * C and C2=( 0 + B * C ) share the '( 0 + B' prefix, so parity is:
      (a) equal total token length, AND
      (b) identical shared-prefix tokens up to and including B  -> B sits at the
          SAME index in both (the spec's 'B at identical token index' constraint).
    Returns (ok, bad). Cheap: parity is a tokenizer property; sampling max_check
    pairs is sufficient (and base/Instruct share the tokenizer, so this is a
    formality — but we still assert, never assume)."""
    def _ids(s):
        # tokenize EXACTLY as the scored pipeline (Phase 3 _encode uses the default
        # add_special_tokens=True); WO_ADD_SPECIAL_TOKENS pins them together so a
        # passing parity check certifies the sequence that is actually patched/scored.
        return tokenizer(s, add_special_tokens=WO_ADD_SPECIAL_TOKENS)["input_ids"]
    def _shared_prefix_len(a, b):
        n = 0
        while n < len(a) and n < len(b) and a[n] == b[n]:
            n += 1
        return n
    bad = []
    for (B, C) in pairs[:max_check]:
        l_ids, r_ids = _ids(render_left(B, C)), _ids(render_right(B, C))
        if len(l_ids) != len(r_ids):
            bad.append((B, C, f"len {len(l_ids)}!={len(r_ids)}")); continue
        # B is the last token of the shared '( 0 + B' prefix; the prefix must be
        # token-identical through B, i.e. the divergence (')' vs '*') comes AFTER B.
        spl = _shared_prefix_len(l_ids, r_ids)
        bstr = str(B)
        # locate B's last token by decode-and-walk on the LEFT surface.
        b_last = None
        acc = ""
        for i, tid in enumerate(l_ids):
            acc_piece = tokenizer.decode([tid])
            if bstr in acc_piece or (acc + acc_piece).replace(" ", "").endswith(bstr):
                b_last = i
            acc += acc_piece
        if b_last is None or spl <= b_last:
            bad.append((B, C, f"divergence at {spl} not strictly after B@{b_last}"))
    ok = (len(bad) == 0)
    log(f"WO: C1/C2 parity on live tokenizer ({WO_ACTIVE_TAG}): "
        f"{'OK' if ok else 'BROKEN'} (checked {min(len(pairs), max_check)}; "
        f"{len(bad)} bad{(' e.g. ' + str(bad[:3])) if bad else ''})")
    return ok, bad


# ---- deliverables sink (§12). Persist to ART/results (survives disconnect);
#      best-effort mirror to a repo-local ./results when the notebook runs from
#      the checked-out repo. Matches the existing "drop ART artifacts into the
#      repo" convention (RESULTS.md does this for the PNG figures).
from pathlib import Path as _Path
WO_RESULTS = ART / "results"
WO_RESULTS.mkdir(parents=True, exist_ok=True)


def wo_save_result(filename, text):
    """Write a deliverable to ART/results/<filename> (+ mirror to ./results if present)."""
    p = WO_RESULTS / filename
    p.write_text(text)
    mirrored = None
    try:
        repo_results = _Path("results")
        if repo_results.is_dir():
            (repo_results / filename).write_text(text)
            mirrored = str(repo_results / filename)
    except Exception:
        pass
    log(f"WO: wrote {p}" + (f" (+ mirror {mirrored})" if mirrored else ""))
    return str(p)


log("Phase 6 / WO setup ready: wo_load_model / wo_eval / wo_run_battery / "
    "wo_assert_parity / wo_save_result.")
