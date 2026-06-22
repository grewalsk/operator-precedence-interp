# ============================================================================
# SETUP — load ONLY the Llama-3.1-8B tokenizer (no model / GPU / transformer_lens),
# and provide in-memory stubs so the REAL Phase 2 cell runs standalone.
# ============================================================================
import os, json, pickle

# --- HF auth for the gated tokenizer (Colab Secret 'HF_TOKEN' or env) ---
_tok = None
try:
    from google.colab import userdata          # 🔑 Secrets in the left sidebar
    _tok = userdata.get("HF_TOKEN")
except Exception:
    pass
_tok = _tok or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if _tok:
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"] = _tok
    from huggingface_hub import login
    login(token=_tok)
    print("HF token set.")
else:
    print("WARNING: no HF_TOKEN found (add it to 🔑 Secrets as HF_TOKEN, enable Notebook access).")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
print("tokenizer:", type(tokenizer).__name__, "| is_fast:", tokenizer.is_fast)

# --- in-memory stubs so the unmodified Phase 2 cell runs with no Drive / always regenerates ---
_MEM = {}
def has_artifact(name, kind=None): return False     # force a FRESH generation every run
def save_json(name, obj): _MEM[name] = obj
def load_json(name): return _MEM[name]
def save_pickle(name, obj): _MEM[name] = obj
def load_pickle(name): return _MEM[name]
def save_text(name, s): _MEM[name] = s
def load_text(name): return _MEM[name]
def log(m): print("[log]", m)
def set_gate(g, p, d=""): print(f"[gate {g}] {'PASS' if p else 'FAIL'} — {d}")

# small + fast: enough to see parity and that pairs actually generate.
CFG = {"seed": 0, "pad_lengths": [0, 2, 4],
       "g2_target_per_factor": 300, "g2_min_valid_per_factor": 50, "g2_sample_budget": 20000}
print("setup OK — run the next cell (the real Phase 2 generator).")
