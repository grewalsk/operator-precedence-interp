---
# Phase 3 — Behavioral validation · **Gate G3** (first real kill‑switch)

Forward‑pass only, before any patching. Three checks, all cached to disk:

1. **Accuracy** — greedy decode on `(0+B)*C`‑class expressions; PASS if accuracy in the chosen band clears a recorded floor (default ≥ 80%).
2. **No‑op check** — vary `B`,`C`; the prediction must track `B*C` (the `0 +` must not let the model skip the multiplication). Guards the additive‑identity correction.
3. **Must‑compute check** — accuracy vs operand size must show *graceful degradation* (computation), **not** pinned‑100% (memorized lookup) and **not** chance.

Locks the operand band that Phases 4–9 consume. **FAIL → stop and redesign the stimulus** — this is the cheapest place to catch a fatal artifact.
