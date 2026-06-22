---
# Phase 2 — Stimulus construction & assertion harness · **Gate G2**

**The buildable‑today, validity‑critical artifact.** If G2 fails, no GPU result is trustworthy.

- **Factor A (Depth):** `( 0 + B ) * C` vs `( 0 + B * C )` — token‑identical on real Llama (both share the `( 0 + B` prefix), both evaluate to `B*C`, parens in both; only the `)`/`*` positions move (so `*` sits at paren‑depth 0 vs 1). Additive‑identity (`0 +`), **never** multiply‑by‑one (which the model may compile to a no‑op).
- **Factor B (Distance, the lead result):** same expression padded with `+ 0` chains that grow token count but preserve answer **and** tree depth.
- **Factor C (Depth‑2, the upside):** a second answer‑preserving nesting for the head‑reuse test later.

Every emitted pair must pass the machine‑checked assertions — **token‑length equality under the real tokenizer** (the hazard: multi‑digit numbers tokenize inconsistently), operand‑position equality, final‑answer equality, parens‑present, and the depth‑tree (differs for A, equal for B). Violations are **dropped with a logged reason**.
