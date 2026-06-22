---
# Setup — install dependencies & authenticate (run this FIRST)

Colab ships `torch`/`transformers` but **not** `transformer_lens`, and Llama‑3.1‑8B is a **gated** model — so a fresh runtime needs one setup cell before the checkpoint/Phase 0 cells. It is safe to re‑run (installs are skipped if already present).

**Before running, two one‑time things:**
1. **Request access** at [huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) (usually granted within minutes).
2. **Provide a *read* token** ([create one here](https://huggingface.co/settings/tokens)). In Colab: click the **🔑 Secrets** icon in the left sidebar, add a secret named **`HF_TOKEN`**, and toggle **Notebook access** on. (Outside Colab: `export HF_TOKEN=hf_...` or `os.environ["HF_TOKEN"]="hf_..."`.)

> Gating too slow? `Qwen/Qwen2.5-3B` is **ungated** (no approval) and a one‑line `CFG["model_name"]` swap — see the model‑choice notes.
