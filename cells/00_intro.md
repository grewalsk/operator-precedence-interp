# Operator‑Precedence Interpretability — Feasibility Notebook (Phases 0–5)

**Goal of this notebook.** Decide whether the project is *plausible* **before** any expensive probing, by passing five gates **in order**:

| Gate | Phase | What it proves | Needs GPU? |
|------|-------|----------------|------------|
| **G0** | 0 | `transformer_lens` loads + hooks Llama‑3.1‑8B (or HF fallback) | yes (cheap) |
| **G1** | 1 | the controlled contribution is novel | no |
| **G2** | 2 | the stimulus is *genuinely* controlled (token‑identical, answer‑equal) | **no — CPU + tokenizer only** |
| **G3** | 3 | the model *computes* (not looks up), engages the operand, in‑band | yes (cheap) |
| **G4** | 5 | the activation‑patching instrument reproduces a *known* result | yes (cheap) |

The project is **PLAUSIBLE iff G0–G4 all pass**. **G2** and **G3** are the ones that, if they fail, mean the *science* is broken (not the tooling). Phase 4 is not a gate but underpins every patch.

---

## ⚠️ How to run this on a flaky GPU (read first)

This notebook is built to **survive GPU disconnects**. Every expensive step writes its result to a **persistent artifact directory** (`ART`) and is guarded by an `if has_artifact(...)` check.

**To resume after a disconnect:** just **re‑run the notebook top‑to‑bottom** (`Restart & Run All` works). Completed phases load their cached artifacts from disk in seconds and are skipped; only unfinished work re‑runs. The model itself is reloaded each fresh session (GPU memory can't be checkpointed) but everything derived from it is cached.

**One‑time setup before the first run:**
1. A GPU with **≥ 24 GB** is the honest floor: ~16 GB bf16 weights + ~2 GB activation cache (sequences are sub‑30 tokens) + overhead, so an **A10 / L4 / RTX 3090** works. **40 GB (A100/H100) recommended for comfort.** A 16 GB T4 will **not** fit.
2. Llama‑3.1‑8B is **gated** on Hugging Face — request access on the model page and set your token: `export HF_TOKEN=hf_...` (or `HUGGINGFACE_TOKEN`). The Phase 0 cell logs in with it.
3. (Colab) The checkpoint cell tries to mount Google Drive so `ART` persists across runtime resets. On a dedicated box it uses a local persistent dir under `$HOME`.

**Run order is strict** — a later gate is only meaningful if the earlier ones are green. The final dashboard cell prints the consolidated G0–G4 verdict and reconstructs it from disk even after a restart.
