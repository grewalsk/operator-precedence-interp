---
# Plausibility verdict

The project is **PLAUSIBLE** when **G0–G4 are all green**. The two gates that decide whether the *science* is sound (not just the tooling) are **G2** (controlled stimulus — buildable with nothing but a tokenizer) and **G3** (the model genuinely computes the task). **G4** decides whether you can trust your own measurements.

Compute is never the plausibility constraint: the whole pre‑probe phase is a handful of cheap forward‑pass evaluations plus one known‑result reproduction. If every gate below is green, **Phase 6 (the depth‑1 padding‑invariance result) is the publishable spine** and the project is confirmed plausible.

The cell below reconstructs the consolidated gate status from disk (`gate_status.json`), so it reports correctly even after a GPU disconnect + restart.
