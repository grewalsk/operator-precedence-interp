# ============================================================================
# PROBE — load each model through every stage; print the REAL traceback on failure.
# ============================================================================
import os
import gc
import traceback
import torch

_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# tag -> (repo, gated?, status in the 82g run)
MODELS = {
    "qwen25_7b_it":  ("Qwen/Qwen2.5-7B-Instruct",           False, "FAILED (ungated!)"),
    "mistral_7b_it": ("mistralai/Mistral-7B-Instruct-v0.3", False, "FAILED (ungated!)"),
    "llama32_3b_it": ("meta-llama/Llama-3.2-3B-Instruct",   True,  "FAILED (gated)"),
    "gemma2_9b_it":  ("google/gemma-2-9b-it",               True,  "worked — control"),
    "llama32_1b_it": ("meta-llama/Llama-3.2-1B-Instruct",   True,  "worked — control"),
}


def _free():
    for v in ("_hf", "_tl", "_tok_obj"):
        if v in globals():
            try:
                del globals()[v]
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ram():
    try:
        import psutil
        m = psutil.virtual_memory()
        return f"RAM avail {m.available / 1e9:.1f}/{m.total / 1e9:.0f} GB"
    except Exception:
        return ""


def _gpu():
    return (f"GPU alloc {torch.cuda.memory_allocated() / 1e9:.1f} GB"
            if torch.cuda.is_available() else "")


def _auth(repo):
    try:
        from huggingface_hub import auth_check
        auth_check(repo, token=_tok)
        return "GRANTED"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:160]}"


RESULTS = {}
for tag, (name, gated, prev) in MODELS.items():
    print("\n" + "=" * 88)
    print(f"### {tag}   {name}   [{('gated' if gated else 'UNGATED')}; 82g: {prev}]")
    print("=" * 88)
    r = {"name": name, "gated": gated, "auth": None, "hf_tokenizer": None,
         "hf_model_cpu": None, "to_cuda": None, "transformer_lens": None}
    RESULTS[tag] = r                       # ref: mutations below persist into RESULTS
    _free()

    # 0) auth_check — true gating vs granted
    r["auth"] = _auth(name)
    print(" [0] auth_check :", r["auth"])

    # 1) HF tokenizer
    try:
        from transformers import AutoTokenizer
        _tok_obj = AutoTokenizer.from_pretrained(name, token=_tok)
        r["hf_tokenizer"] = "OK"
        print(" [1] HF tokenizer : OK")
    except Exception as e:
        r["hf_tokenizer"] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
        print(" [1] HF tokenizer : FAIL")
        traceback.print_exc()
        continue

    # 2) HF model -> CPU (SAME call the main notebook's TL-primary path makes)
    try:
        from transformers import AutoModelForCausalLM
        _hf = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, token=_tok)
        _nb = sum(p.numel() for p in _hf.parameters()) / 1e9
        r["hf_model_cpu"] = f"OK ~{_nb:.1f}B"
        print(f" [2] HF model (CPU): OK ~{_nb:.1f}B   {_ram()}")
    except Exception as e:
        r["hf_model_cpu"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        print(f" [2] HF model (CPU): FAIL   {_ram()}")
        traceback.print_exc()
        _free()
        continue

    # 3) move to CUDA (catches CPU->GPU OOM)
    try:
        _hf = _hf.to("cuda")
        torch.cuda.synchronize()
        r["to_cuda"] = "OK"
        print(f" [3] .to(cuda)    : OK   {_gpu()}")
    except Exception as e:
        r["to_cuda"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        print(" [3] .to(cuda)    : FAIL")
        traceback.print_exc()
        _free()
        continue

    # 4) transformer_lens wrap (main notebook PRIMARY path)
    try:
        from transformer_lens import HookedTransformer
        _tl = HookedTransformer.from_pretrained(
            name, hf_model=_hf, tokenizer=_tok_obj, dtype=torch.bfloat16,
            device="cuda", fold_ln=True, center_writing_weights=False,
            center_unembed=False)
        r["transformer_lens"] = f"OK n_layers={_tl.cfg.n_layers}"
        print(f" [4] transformer_lens: OK (n_layers={_tl.cfg.n_layers})")
    except Exception as e:
        r["transformer_lens"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
        print(" [4] transformer_lens: FAIL  <-- main notebook would FALL BACK to the HF wrapper.")
        print("     If [2]+[3] were OK above, this model SHOULD still work via fallback;")
        print("     the 82g 'access_denied' was spurious. The traceback:")
        traceback.print_exc()
    _free()

# ----------------------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------------------
def _mark(x):
    if x is None:
        return "—"
    s = str(x)
    if s == "GRANTED" or s.startswith("OK"):
        return "OK"
    return "FAIL"


print("\n\n" + "#" * 88)
print("# SUMMARY")
print("#" * 88)
print(f"{'tag':<16}{'auth':<6}{'tok':<6}{'hf_cpu':<8}{'cuda':<7}{'TL':<6}  diagnosis")
for tag, r in RESULTS.items():
    a, t, c, g, tl = (_mark(r["auth"]), _mark(r["hf_tokenizer"]),
                      _mark(r["hf_model_cpu"]), _mark(r["to_cuda"]), _mark(r["transformer_lens"]))
    if a == "FAIL" and r["gated"]:
        dx = "TRUE GATING — accept the license on the model page"
    elif c == "FAIL":
        dx = "CPU model load failed (version/disk) — fallback also fails -> real blocker"
    elif g == "FAIL":
        dx = "CPU->GPU OOM — model loaded but won't fit; check GPU/runtime"
    elif tl == "FAIL":
        dx = "TL-only failure -> main notebook fallback WOULD work; 82g label was spurious"
    elif tl == "OK":
        dx = "loads cleanly end-to-end (TL primary)"
    else:
        dx = "see traceback above"
    print(f"{tag:<16}{a:<6}{t:<6}{c:<8}{g:<7}{tl:<6}  {dx}")

print("\nNext: paste this SUMMARY (and any FAIL traceback) back, and I'll patch 82g")
print("accordingly (true gating vs fallback-works vs OOM) and we proceed.")
