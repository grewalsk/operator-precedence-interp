# =====================================================================
# PHASE 0 — Gate G0 : load + hook Llama-3.1-8B, run 5-check smoke test
# Notebook contract: CFG, ART, has_artifact/save_*/load_*, log() already exist.
# 'model' / 'tokenizer' become globals after this cell.
#
# ADVERSARIAL-REVIEW FIXES vs original:
#   * set_gate() is NOT a contract-guaranteed helper -> call is now guarded
#     (NameError on the last line would otherwise fail G0 even on a clean pass).
#   * Smoke test now obeys the RESILIENCE RULE: if 'g0_smoke' already PASSED on
#     disk, we skip recomputation on reconnect instead of re-running the model.
#   * Fallback run_with_cache forward is hardened against output_attentions
#     being rejected by newer transformers (eager already fills attn_weights).
# transformer_lens API verified: HookedTransformer.from_pretrained(model_name,
#   hf_model=, tokenizer=, dtype=<torch.dtype>, device=) is correct; hook names
#   blocks.{L}.hook_resid_post / blocks.{L}.attn.hook_pattern /
#   blocks.{L}.hook_mlp_out are correct; Llama-3.1-8B -> n_layers=32, n_heads=32,
#   d_model=4096.
# =====================================================================
# ============================ PART A ================================
# HF AUTH + MODEL LOAD (guarded). Primary: transformer_lens HookedTransformer.
# Fallback: HF AutoModelForCausalLM wrapped to expose run_with_cache/run_with_hooks.
# ====================================================================
import os
import torch

# ---- HF auth: read token from env, never hardcode -------------------
# Llama-3.1-8B is a GATED repo. You must (a) have requested access on the HF
# model page with the same account, and (b) expose a token via env:
#     export HUGGINGFACE_TOKEN=hf_xxx   (or HF_TOKEN=hf_xxx)
_hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        log("HF: logged in from env token.")
    except Exception as e:
        log(f"HF: login() raised {type(e).__name__}: {e} (continuing; token may still be picked up from env).")
else:
    log("HF: no HUGGINGFACE_TOKEN / HF_TOKEN in env. "
        "If the gated Llama repo is inaccessible you will see a 401/403 at load; "
        "set the env var and re-run this cell.")

# ---- helper: confirm gated repo is reachable before the heavy load --
def _check_gated_repo(repo_id):
    """Return (ok: bool, msg: str). Cheap metadata probe; surfaces 401/403 clearly."""
    try:
        from huggingface_hub import model_info
        info = model_info(repo_id, token=_hf_token)
        return True, f"reachable (sha={getattr(info, 'sha', '?')})"
    except Exception as e:
        return False, (f"{type(e).__name__}: {e}\\n"
                       f"  -> Visit https://huggingface.co/{repo_id} and click 'Agree and access', "
                       f"then export HUGGINGFACE_TOKEN and re-run.")

_repo = CFG["model_name"]
_ok, _msg = _check_gated_repo(_repo)
log(f"HF: gated-repo check for {_repo}: {_msg}")

# ---- resolve the model revision/commit hash (for the version lock) --
def _resolve_revision(repo_id):
    try:
        from huggingface_hub import model_info
        return getattr(model_info(repo_id, token=_hf_token), "sha", None)
    except Exception:
        return None
_resolved_revision = _resolve_revision(_repo)

# ---- thin HF fallback wrapper ---------------------------------------
# Exposes the minimal surface later cells use:
#   .cfg.{n_layers,n_heads,d_model,device}
#   .to_tokens(str) -> LongTensor[1, seq]
#   .__call__(tokens) / .forward(tokens) -> logits[1, seq, vocab]
#   .run_with_cache(tokens) -> (logits, cache) where cache is dict keyed by
#       TL-style hook names: 'blocks.{L}.hook_resid_post',
#       'blocks.{L}.attn.hook_pattern', 'blocks.{L}.hook_mlp_out'
#   .run_with_hooks(tokens, fwd_hooks=[(name, fn), ...]) -> logits
# Attention pattern requires attn_implementation='eager' (no flash/sdpa).
# NOTE: in current transformers LlamaDecoderLayer.forward returns a *plain
# tensor* (resid_post) and LlamaMLP.forward returns a plain tensor, while
# LlamaAttention.forward returns (attn_output, attn_weights); the unwrap below
# ('out[0] if tuple else out') is robust across versions.
class _SimpleCache(dict):
    """dict that also accepts TL-ish indexing cache['blocks',L,'hook_resid_post']."""
    def __getitem__(self, key):
        if isinstance(key, tuple):
            # support cache['blocks', L, 'hook_resid_post'] -> 'blocks.{L}.hook_resid_post'
            parts = [str(k) for k in key]
            return dict.__getitem__(self, ".".join(parts))
        return dict.__getitem__(self, key)

class HFHookedWrapper:
    """Minimal transformer_lens-compatible shim over a HF CausalLM."""
    class _Cfg:
        pass

    def __init__(self, hf_model, hf_tokenizer, device):
        self._m = hf_model
        self.tokenizer = hf_tokenizer
        self.cfg = HFHookedWrapper._Cfg()
        cfg = hf_model.config
        self.cfg.n_layers = cfg.num_hidden_layers
        self.cfg.n_heads = cfg.num_attention_heads
        self.cfg.d_model = cfg.hidden_size
        self.cfg.d_vocab = cfg.vocab_size
        self.cfg.device = device
        self.device = device
        self._is_fallback = True

    # nn.Module-style methods later phases rely on (Phase 5 calls model.eval()).
    # Delegate explicitly to the wrapped HF model -- NOT via __getattr__, which
    # would infinitely recurse if self._m is ever unbound (unpickle/pre-__init__).
    def eval(self):
        self._m.eval()
        return self

    def train(self, mode=True):
        self._m.train(mode)
        return self

    def to_tokens(self, text, prepend_bos=True):
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=prepend_bos)
        return enc["input_ids"].to(self.device)

    def to_str_tokens(self, text, prepend_bos=True):
        ids = self.to_tokens(text, prepend_bos=prepend_bos)[0]
        return [self.tokenizer.decode([i]) for i in ids.tolist()]

    @torch.no_grad()
    def forward(self, tokens, return_type="logits"):
        if isinstance(tokens, str):
            tokens = self.to_tokens(tokens)
        out = self._m(input_ids=tokens)
        return out.logits

    __call__ = forward

    @torch.no_grad()
    def run_with_cache(self, tokens, names_filter=None):
        """Returns (logits, cache). Caches resid_post / attn pattern / mlp_out
        per layer under TL-style names via forward hooks."""
        if isinstance(tokens, str):
            tokens = self.to_tokens(tokens)
        cache = _SimpleCache()
        handles = []

        def _keep(name):
            return (names_filter is None) or names_filter(name)

        layers = self._m.model.layers
        for L, block in enumerate(layers):
            rp_name = f"blocks.{L}.hook_resid_post"
            mlp_name = f"blocks.{L}.hook_mlp_out"
            attn_name = f"blocks.{L}.attn.hook_pattern"

            if _keep(rp_name):
                def _rp_hook(mod, inp, out, _n=rp_name):
                    h = out[0] if isinstance(out, tuple) else out
                    cache[_n] = h.detach()
                handles.append(block.register_forward_hook(_rp_hook))

            if _keep(mlp_name):
                def _mlp_hook(mod, inp, out, _n=mlp_name):
                    h = out[0] if isinstance(out, tuple) else out
                    cache[_n] = h.detach()
                handles.append(block.mlp.register_forward_hook(_mlp_hook))

            if _keep(attn_name):
                # eager attention returns attn weights as the 2nd output element
                def _attn_hook(mod, inp, out, _n=attn_name):
                    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                        cache[_n] = out[1].detach()  # [b, n_heads, q, k]
                handles.append(block.self_attn.register_forward_hook(_attn_hook))

        try:
            # eager attn already fills attn_weights; output_attentions=True is
            # redundant and rejected on some newer transformers paths, so retry
            # without it if needed. The per-layer self_attn hooks fill the cache
            # either way.
            try:
                out = self._m(input_ids=tokens, output_attentions=True, use_cache=False)
            except (TypeError, ValueError) as _e_oa:
                log(f"HF fallback: output_attentions path rejected ({type(_e_oa).__name__}); "
                    f"retrying without it (eager hooks still capture patterns).")
                out = self._m(input_ids=tokens, use_cache=False)
            logits = out.logits
        finally:
            for h in handles:
                h.remove()
        return logits, cache

    @torch.no_grad()
    def run_with_hooks(self, tokens, fwd_hooks=None, return_type="logits"):
        """fwd_hooks: list of (tl_hook_name, fn(tensor, hook=None)->tensor|None).
        Supports resid_post / mlp_out (output rewrite). Attention-pattern editing
        is not supported in the fallback (rarely needed for G0)."""
        if isinstance(tokens, str):
            tokens = self.to_tokens(tokens)
        fwd_hooks = fwd_hooks or []
        handles = []
        layers = self._m.model.layers
        for name, fn in fwd_hooks:
            parts = name.split(".")
            L = int(parts[1])
            block = layers[L]
            if name.endswith("hook_resid_post"):
                target = block
            elif name.endswith("hook_mlp_out"):
                target = block.mlp
            else:
                raise NotImplementedError(f"fallback run_with_hooks does not support {name}")
            def _wrap(mod, inp, out, _fn=fn):
                h = out[0] if isinstance(out, tuple) else out
                new = _fn(h, hook=None)
                if new is None:
                    new = h
                if isinstance(out, tuple):
                    return (new,) + tuple(out[1:])
                return new
            handles.append(target.register_forward_hook(_wrap))
        try:
            logits = self._m(input_ids=tokens, use_cache=False).logits
        finally:
            for h in handles:
                h.remove()
        return logits

# ---- the guarded load -----------------------------------------------
USING_FALLBACK = False
if "model" not in globals():
    _device = CFG["device"]
    try:
        # ---- PRIMARY: transformer_lens HookedTransformer ----
        import transformer_lens
        from transformer_lens import HookedTransformer
        log("Loading via transformer_lens HookedTransformer (primary path)...")
        # Pass HF model+tokenizer explicitly so the gated download / dtype /
        # device placement is unambiguous and TL just wraps it. When hf_model is
        # supplied, TL skips its own dtype download path and uses these weights.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _hf_tok = AutoTokenizer.from_pretrained(CFG["model_name"], token=_hf_token)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            CFG["model_name"],
            torch_dtype=torch.bfloat16,   # still accepted (deprecated alias of dtype)
            token=_hf_token,
        )
        model = HookedTransformer.from_pretrained(
            CFG["model_name"],
            hf_model=_hf_model,
            tokenizer=_hf_tok,
            dtype=torch.bfloat16,          # torch.dtype object is accepted
            device=_device,
            fold_ln=True,
            center_writing_weights=False,  # intentional: leave resid stream uncentered
            center_unembed=False,          # intentional for Llama analyses
        )
        tokenizer = model.tokenizer
        del _hf_model  # TL has copied the weights; free the HF copy
        log(f"Loaded HookedTransformer on {model.cfg.device} "
            f"(n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, d_model={model.cfg.d_model}).")
    except Exception as e_primary:
        # ---- FALLBACK: raw HF + thin wrapper ----
        log(f"*** transformer_lens load FAILED: {type(e_primary).__name__}: {e_primary}")
        log("*** FALLING BACK to HF AutoModelForCausalLM + HFHookedWrapper. "
            "BUDGET NOTE: decide TL-vs-HF by end of Day 0; the fallback covers G0 "
            "but lacks turnkey activation/attribution patching for later phases.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _hf_tok = AutoTokenizer.from_pretrained(CFG["model_name"], token=_hf_token)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            CFG["model_name"],
            torch_dtype=torch.bfloat16,
            token=_hf_token,
            attn_implementation="eager",  # REQUIRED so self_attn returns attn weights
        ).to(_device)
        _hf_model.eval()
        model = HFHookedWrapper(_hf_model, _hf_tok, _device)
        tokenizer = model.tokenizer
        USING_FALLBACK = True
        log(f"Loaded HF fallback on {_device} "
            f"(n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, d_model={model.cfg.d_model}).")
else:
    USING_FALLBACK = bool(getattr(model, "_is_fallback", False))
    log(f"'model' already in globals (fallback={USING_FALLBACK}); skipping reload.")

# ---- version lock (pin/log versions + resolved revision) ------------
def _ver(modname):
    try:
        return __import__(modname).__version__
    except Exception:
        return "n/a"
_versions = {
    "transformer_lens": _ver("transformer_lens"),
    "torch": torch.__version__,
    "transformers": _ver("transformers"),
    "huggingface_hub": _ver("huggingface_hub"),
    "model_name": CFG["model_name"],
    "resolved_revision": _resolved_revision,
    "using_fallback": USING_FALLBACK,
    "device": str(getattr(model.cfg, "device", CFG["device"])),
    "dtype": "bfloat16",
}
import json as _json
save_text("versions_lock", _json.dumps(_versions, indent=2))
log("versions_lock: " + _json.dumps(_versions))


# ============================ PART B ================================
# G0 SMOKE TEST — 5 checks. Each prints PASS/FAIL; register the gate at end.
# RESILIENCE: if a prior PASS is already on disk, skip recomputation.
# Uses EXACT transformer_lens hook-point names:
#   resid_post : f"blocks.{L}.hook_resid_post"   -> [batch, seq, d_model=4096]
#   attn patt. : f"blocks.{L}.attn.hook_pattern" -> [batch, n_heads, seq, seq]
# run_with_cache returns (logits, cache); index cache by the string hook name.
# ====================================================================
import torch

def _g0_smoke():
    results = {}
    PROMPT = "12 + 7 ="
    device_str = str(getattr(model.cfg, "device", CFG["device"]))
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    L = min(5, n_layers - 1)  # arbitrary probe layer

    # --- Check 1: model loaded on target device ---
    try:
        on_target = (CFG["device"].split(":")[0] in device_str)
        results["1_device"] = bool(on_target)
        print(f"[G0.1] model on target device: device={device_str}, "
              f"target={CFG['device']} -> {'PASS' if on_target else 'FAIL'}")
    except Exception as e:
        results["1_device"] = False
        print(f"[G0.1] FAIL ({type(e).__name__}: {e})")

    # tokenize once, assert < 30 tokens
    tokens = model.to_tokens(PROMPT)
    seq = tokens.shape[1]
    assert seq < 30, f"smoke prompt unexpectedly long: {seq} tokens"

    # --- Check 2: forward pass -> finite logits ---
    try:
        with torch.no_grad():
            logits = model(tokens)
        finite = bool(torch.isfinite(logits).all().item())
        shape_ok = (logits.shape[0] == 1 and logits.shape[1] == seq)
        ok2 = finite and shape_ok
        results["2_forward_finite"] = ok2
        print(f"[G0.2] forward on {PROMPT!r}: logits {tuple(logits.shape)}, "
              f"all_finite={finite} -> {'PASS' if ok2 else 'FAIL'}")
    except Exception as e:
        results["2_forward_finite"] = False
        logits = None
        print(f"[G0.2] FAIL ({type(e).__name__}: {e})")

    # --- run_with_cache once for checks 3 & 4 ---
    rp_name = f"blocks.{L}.hook_resid_post"
    pat_name = f"blocks.{L}.attn.hook_pattern"
    try:
        with torch.no_grad():
            _logits_c, cache = model.run_with_cache(tokens)
    except Exception as e:
        cache = {}
        print(f"[G0.cache] run_with_cache FAILED ({type(e).__name__}: {e})")

    # --- Check 3: resid_post hook shape [batch, seq, 4096] + finite ---
    try:
        rp = cache[rp_name]
        shape_ok = (tuple(rp.shape) == (1, seq, d_model)) and (d_model == 4096)
        finite = bool(torch.isfinite(rp).all().item())
        ok3 = shape_ok and finite
        results["3_resid_post"] = ok3
        print(f"[G0.3] {rp_name}: shape={tuple(rp.shape)} "
              f"(expect (1,{seq},{d_model})), finite={finite} -> {'PASS' if ok3 else 'FAIL'}")
    except Exception as e:
        results["3_resid_post"] = False
        print(f"[G0.3] FAIL ({type(e).__name__}: {e})")

    # --- Check 4: attn pattern shape [batch, n_heads, seq, seq] + rows sum ~1 ---
    try:
        pat = cache[pat_name]
        shape_ok = (tuple(pat.shape) == (1, n_heads, seq, seq))
        row_sums = pat.float().sum(dim=-1)          # [1, n_heads, seq]
        close = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)
        max_dev = float((row_sums - 1.0).abs().max().item())
        ok4 = shape_ok and close
        results["4_attn_pattern"] = ok4
        assert close, f"attn rows do not sum to 1 (max dev {max_dev:.3e})"
        print(f"[G0.4] {pat_name}: shape={tuple(pat.shape)} "
              f"(expect (1,{n_heads},{seq},{seq})), max|rowsum-1|={max_dev:.2e} "
              f"-> {'PASS' if ok4 else 'FAIL'}")
    except Exception as e:
        results["4_attn_pattern"] = False
        print(f"[G0.4] FAIL ({type(e).__name__}: {e})")

    # --- Check 5: greedy CONTINUATION contains a plausible digit ---
    # Llama emits a LEADING SPACE token before the answer digit, so the single next token
    # after "12 + 7 =" is often whitespace, not a digit (this exact gotcha tripped Phase 5
    # too). Decode a few tokens greedily and check the short continuation has a digit --
    # robust to the space-then-digit split.
    try:
        _cur = tokens
        _gen = ""
        with torch.no_grad():
            for _ in range(3):
                _lg = model(_cur)
                _nid = int(_lg[0, -1].argmax().item())
                _gen += tokenizer.decode([_nid])
                _cur = torch.cat(
                    [_cur, torch.tensor([[_nid]], device=_cur.device, dtype=_cur.dtype)], dim=1)
                if any(ch.isdigit() for ch in _gen):
                    break
        ok5 = any(ch.isdigit() for ch in _gen)
        results["5_greedy_digit"] = ok5
        print(f"[G0.5] greedy continuation after {PROMPT!r}: {_gen!r} -> "
              f"{'PASS' if ok5 else 'FAIL'} (digit appears; leading space is expected on Llama)")
    except Exception as e:
        results["5_greedy_digit"] = False
        print(f"[G0.5] FAIL ({type(e).__name__}: {e})")

    return results

# ---- RESILIENCE: reuse a prior PASS instead of recomputing on reconnect ----
if has_artifact("g0_smoke"):
    _g0_record = load_json("g0_smoke")
    if _g0_record.get("pass", False):
        _g0_results = _g0_record.get("checks", {})
        _g0_pass = True
        log("G0 smoke already PASSED on disk; skipping recompute (resilience).")
    else:
        _g0_results = _g0_smoke()
        _g0_pass = all(_g0_results.values()) if _g0_results else False
else:
    _g0_results = _g0_smoke()
    _g0_pass = all(_g0_results.values()) if _g0_results else False

print("\\n" + "=" * 56)
print(f"G0 SMOKE SUMMARY: {sum(bool(v) for v in _g0_results.values())}/5 checks passed "
      f"(fallback={USING_FALLBACK})")
for k, v in _g0_results.items():
    print(f"   {k:20s}: {'PASS' if v else 'FAIL'}")
print(f"GATE G0 -> {'PASS' if _g0_pass else 'FAIL'}")
print("=" * 56)

# persist the smoke result (light artifact) — this is the contract-guaranteed
# source of truth for the gate.
_g0_record = {
    "pass": bool(_g0_pass),
    "checks": {k: bool(v) for k, v in _g0_results.items()},
    "using_fallback": USING_FALLBACK,
    "versions": _versions,
}
save_json("g0_smoke", _g0_record)

# register the gate IFF the harness provides set_gate; it is NOT promised by the
# notebook contract, so guard it to avoid a NameError that would fail a clean G0.
if "set_gate" in globals() and callable(globals()["set_gate"]):
    set_gate("G0", bool(_g0_pass), _g0_record)
    log("G0 registered via set_gate().")
else:
    log("set_gate not available in this kernel; G0 result persisted to 'g0_smoke' artifact only.")