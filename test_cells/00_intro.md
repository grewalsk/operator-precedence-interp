# G2 tokenizer test — does the controlled stimulus build on the **real** Llama-3.1-8B tokenizer?

This is a tiny, fast, self-contained test of the **single most validity-critical** piece (Phase 2 / Gate G2). It loads **only the tokenizer** — no model, no GPU, no `transformer_lens` (so none of the torchaudio install drama) — and runs the **real** Phase 2 generator + assertion harness against it.

It answers one question: does the parenthesized contrast `( 0 + B ) * C =` vs `0 + ( B * C ) =` tokenize with **equal token length** on real Llama (so the controlled pairs survive), or does the tokenizer fight the design?

**Before running:** put your `HF_TOKEN` in 🔑 **Secrets** (or `os.environ`). Then **Run all**. Runs fine on a free CPU runtime — you don't even need a GPU for this.

Paste the output of the **last cell** back and we wire the result into the main notebook.
