# Operator‑Precedence Interpretability — Feasibility (Phases 0–5)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grewalsk/operator-precedence-interp/blob/main/operator_precedence_phases_0_5.ipynb)

A single **GPU‑resumable** Jupyter notebook that decides whether the operator‑precedence interpretability project is *plausible* — **before** any expensive probing — by passing five gates in order on **Llama‑3.1‑8B**.

| Gate | Phase | Proves | GPU |
|------|-------|--------|-----|
| **G0** | 0 | `transformer_lens` loads + hooks the model (HF fallback) | yes (cheap) |
| **G1** | 1 | the controlled contribution is novel | no |
| **G2** | 2 | the stimulus is *genuinely* controlled (token‑identical, answer‑equal) | **no — CPU + tokenizer** |
| **G3** | 3 | the model *computes* (not looks up), engages the operand, in‑band | yes (cheap) |
| **G4** | 5 | the activation‑patching instrument reproduces a *known* result | yes (cheap) |

**PLAUSIBLE iff G0–G4 all pass.** G2 + G3 decide whether the *science* is sound; G4 whether you can trust your own measurements.

## The controlled contrast (Factor A)

Token‑identical, additive‑identity (never `×1`), parentheses in both — only the boundary moves:

```
depth_left  :  ( 0 + B ) * C =      # (0+B)*C = B*C ,  '*' at paren-depth 0
depth_right :  0 + ( B * C ) =      # 0+(B*C) = B*C ,  '*' at paren-depth 1
```

Within a pair, token‑length parity and B's token index are **hard assertions**; C's structural shift is recorded. Factor B stretches length with suffix `+ 0` padding (answer + depth preserved); Factor C is a depth‑2 nesting `(0+(0+B)*C)*D = B·C·D`.

## Run it

1. **GPU ≥ 24 GB** is the honest floor — ~16 GB bf16 weights + ~2 GB activation cache (sequences are sub‑30 tokens) + overhead, so an **A10 / L4 / RTX 3090** works. **40 GB (A100/H100) recommended for comfort.** A 16 GB T4 will *not* fit.
2. Llama‑3.1‑8B is gated — request access and `export HF_TOKEN=hf_...`.
3. Open the notebook (Colab badge above) and **Run All**. It survives disconnects: every expensive step is checkpointed to a persistent `ART` dir behind an `if has_artifact(...)` guard, so re‑running top‑to‑bottom skips finished work. **G2 needs no GPU** — run the checkpoint + Phase 2 cells on CPU first to confirm the controlled stimulus actually builds against the real tokenizer.

## Editing

The notebook is assembled from raw cell files — edit those and rebuild, never hand‑edit the `.ipynb` JSON:

```bash
python3 build_notebook.py cells operator_precedence_phases_0_5.ipynb
```

`cells/NN_*.py|*.md` are concatenated in sorted order (`.py` → code cell, `.md` → markdown).
