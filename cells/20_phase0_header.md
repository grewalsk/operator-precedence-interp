---
# Phase 0 — Environment & compute · **Gate G0**

Load and hook Llama‑3.1‑8B in bf16 on a single device, then run the five‑check smoke test:

1. model loads on the target device,
2. a forward pass on a sub‑30‑token arithmetic string returns finite logits,
3. `blocks.{L}.hook_resid_post` caches a `[batch, seq, 4096]` finite tensor,
4. `blocks.{L}.attn.hook_pattern` caches `[batch, n_heads, seq, seq]` with rows summing to ≈ 1,
5. greedy next‑token on `"12 + 7 ="` is a plausible digit.

**PASS** iff all five succeed → `G0` recorded. **FAIL → fallback:** HF `AutoModelForCausalLM` with manual `register_forward_hook` on the analogous modules. Decide by end of Day 0; don't iterate on library internals past that.
