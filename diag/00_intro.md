# WO#4 — model-load diagnostic (standalone)

The main notebook's `wo_load_model` tries transformer_lens, falls back to a raw-HF
wrapper, and if **both** fail it raises — which cell 82g catches and (mis)labels
`access_denied` for *every* failure, gating or not. Qwen2.5 and Mistral are ungated,
so their `access_denied` is a lie hiding a real load error.

This notebook isolates it. For each model it runs five stages and prints the **full
traceback** of whichever one fails:

1. `auth_check` — true gating (401/403) vs. granted.
2. HF tokenizer load.
3. HF model load to CPU (same call the main notebook makes).
4. `.to("cuda")` — catches CPU→GPU OOM.
5. transformer_lens wrap — the main notebook's PRIMARY path; if only this fails but
   3+4 were OK, the model would still work in the main notebook via the HF fallback,
   meaning the `access_denied` was spurious.

The two models that already worked (Gemma-2-9b, Llama-3.2-1B) are included as controls.

**Run all.** It reuses your Drive HF cache, so already-downloaded weights don't re-download.
