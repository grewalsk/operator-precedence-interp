---
# Phase 4 — Token‑boundary mapping (enables the patches)

Not a gate, but getting it wrong silently corrupts every patch. Because the stimuli are **token‑aligned by construction** (guaranteed by the Phase 2 assertions), the index map is **shared across a whole template**, not computed per example.

`token_map(template, condition, pad_len)` returns the canonical token indices: where the intermediate value (`B*C`, or the held `B` after `0+B`) first becomes decodable, the critical `*` operator, the index where the structural role flips between the depth conditions, and the deterministic operand‑index shift as a function of pad length. Unit‑tested against hand‑verified examples; later phases import it and never recompute boundaries ad hoc.
