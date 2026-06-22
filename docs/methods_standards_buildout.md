# Operator-Precedence Interpretability: Methods, Standards, and Build-Out

*A working document translating the methods literature into (a) what this paper must demonstrate to be considered rigorous, (b) how each method maps onto field standards, and (c) the concrete experiment cells still to build. Companion to the Phases 0-5 feasibility repo (`operator-precedence-interp`), which implements the gates; this document specifies the experiment that runs once the gates pass.*

---

## 0. The claim, restated in falsifiable form

The paper asks whether Llama-3.1-8B resolves nested arithmetic precedence (`(0 + B) * C` vs `0 + (B * C)`) by representing hierarchical structure, or by depth-bound heuristics that reach the right answer without a depth-general procedure. This must be stated as a contest between a hypothesis and a **strong, well-supported null**, because the arithmetic-interpretability literature currently favors the null.

- **H1 (hierarchical / syntactic):** an intermediate value is carried by a *syntactically anchored* representation. Prediction: it stays linearly decodable and stable as we pad the sequence (distance-invariance), and the position that carries the structural role is causally identifiable across depth conditions.
- **H0 (depth-bound heuristic / positional):** the model uses positional carriers and per-depth handlers. Prediction: decodability degrades with padding distance; structural-role carriers do not generalize across depth.

The lead result is the **distance-padding probe invariance** test (H1 vs H0 on the *how-carried* axis). Activation patching corroborates on the *where-carried* axis. The honest most-likely outcome is partial structure (hierarchical at depth-1, degrading under padding or at depth-2), which is a precise characterization tied to bounded-depth expressivity, not a null.

**Why the null is strong.** Nikankin et al. (2024, "Arithmetic Without Algorithms," ICLR 2025) found LM arithmetic is "a bag of heuristics," neither robust algorithms nor memorization; ablating the heuristic neurons drops accuracy only ~29 points from ~95%, i.e. graded not catastrophic. Kantamneni & Tegmark (2025) reverse-engineered addition as a trigonometric "Clock" algorithm but found the helical fits **weaker specifically for Llama-3.1-8B**, "potentially indicating the use of non-Clock algorithms." So the exact target model is already flagged as heuristic-flavored. A clean hierarchy claim would be surprising; this is why the controls below are not optional.

---

## 1. What the field requires (the bar this paper is judged against)

The methods review distilled the rigor standards at BlackboxNLP / the ICML MI workshop into a small number of non-negotiables. State them up front so the design can be checked against each.

**For probing:**
1. A probe's success proves information is *linearly decodable*, not *computed-and-used*. Decodability ≠ use (Belinkov 2022; Elazar et al. 2021).
2. Report **selectivity** against a control task (Hewitt & Liang 2019): real-task accuracy high, control-task accuracy low. A non-selective probe is uninterpretable.
3. Compare against an **input-only / embedding-layer baseline**, to rule out "readable off the inputs" rather than "computed here."
4. Convert correlation to causation: **intervene along the probe direction** and show the model's output moves (amnesic probing / concept erasure; LEACE preferred over INLP, Belrose et al. 2023).

**For patching:**
5. Clean/corrupted prompts must be **token-length matched and minimally different**; what differs determines what you trace (Heimersheim & Nanda 2024).
6. Corrupt by **token substitution, not Gaussian noise**; GN puts the model off-distribution and gives unpredictable localization (Zhang & Nanda 2023).
7. Use **normalized logit difference** as the primary metric (1 = clean, 0 = corrupted); report a second continuous metric (Heimersheim & Nanda 2024; Wang et al. 2022).
8. Choose **noising (necessity) vs denoising (sufficiency)** deliberately and interpret via AND/OR-gate logic; they are not symmetric.
9. **Reproduce a known result first**; confirm the model solves the clean task and the metric sign is right.
10. Guard against **self-repair / the Hydra effect** (McGrath et al. 2023; Wang et al. 2022): a component looking unimportant under single-component ablation may simply be compensated for. Corroborate with denoising and multi-component tests, never ablation alone.

**For circuit-quality claims (if made):** faithfulness, completeness, minimality (Wang et al. 2022, IOI). For a short paper we make the weaker localization claim and explicitly disclaim minimality.

---

## 2. How our protocol maps onto each standard

The feasibility repo already satisfies several of these; the gap is the experiment cells. Mapping each standard to where it lives:

| # | Standard | Where it is handled | Status |
|---|----------|--------------------|--------|
| 1 | decodability ≠ use | Phase 6 adds a causal probe test (std. 4) | **to build** |
| 2 | control-task selectivity | Phase 6 probe must train a control-label probe in parallel | **to build** |
| 3 | input-only baseline | Phase 6 probes the embedding layer + a pre-computation prompt | partially specified in plan; **to build** |
| 4 | causal probe direction | Phase 6 erase/steer along probe dir, measure answer shift | **to build** |
| 5 | token-length-matched prompts | **Gate G2** (`assert_controls.py`): hard assertion on token-id length parity | **built + verified** |
| 6 | token-substitution corruption | **Gate G4 / Phase 5** uses operand substitution (`3 and 4` -> `3 and 5`), no GN | **built** |
| 7 | normalized logit-diff metric | **Phase 5** `recovery()` normalizes between clean/corrupted baselines | **built** |
| 8 | noising vs denoising | Phase 5 does denoising (sufficiency); Phase 7 must add noising (necessity) | denoising **built**; noising **to build** |
| 9 | reproduce known result first | **Gate G4** reproduces Stolfo-style addition localization before novel runs | **built** |
| 10 | self-repair guard | Phase 7 must corroborate ablation with denoising + multi-position | **to build** |

The takeaway: the *instrument* (gates 0-5) is built to standard. The *experiment* (6-8) is where the rigor must still be implemented, and the review tells us exactly what each cell must contain.

---

## 3. Stimulus design — already controlled, and why it is the paper's backbone

This is the part reviewers will scrutinize hardest, because a single uncontrolled pair silently confounds the result, and it is already implemented and machine-checked in Gate G2.

**The depth contrast** uses the additive identity so multiplication stays a genuine operation:
- `depth_left`: `( 0 + B ) * C` -> `(0+B)*C = B*C`, `*` at paren-depth 0
- `depth_right`: `0 + ( B * C )` -> `0+(B*C) = B*C`, `*` at paren-depth 1

Same token multiset, both evaluate to `B*C`, parentheses in both, only the boundary moves. **Multiply-by-one is forbidden** as the contrast operand because `X*1` may be a learned no-op the model skips, which would null the very mechanism under test; `0+X` leaves `B` and `C` untouched so the multiplication stays engaged. (This decision was verified empirically against a context-dependent BPE tokenizer: the token-length-parity control holds across operand magnitudes and induces no operand-magnitude confound, because `B` occupies the same structural index in both templates regardless of how it tokenizes.)

**The distance factor** (the lead result) appends `+ 0` suffix padding before `=`, growing token count while preserving both the answer and the `*`-nesting depth. The probed operand's distance to the final operator grows by a fixed amount per pad unit; the design records the shift rather than forbidding it.

**The depth-2 extension** (`( 0 + ( 0 + B ) * C ) * D = B*C*D`) is generated and lightly validated now, exercised only in Phase 8.

**The controls are machine-checked, not eyeballed** (G2 hard assertions): per pair, token-id length equality, operand-position equality, final-answer equality, parens-present in both, tree-depth differs (depth contrast) or is unchanged (padding). Pairs failing any assertion are dropped with a logged reason, so a tokenizer fighting the design is visible in the drop counts.

**One caveat to carry into the writeup, from Zhu et al. (2025):** early layers encode number *length / token-sequence length* before value emerges. So a naive "invariance under padding" could be a length artifact. The padding result must explicitly subtract a length/control baseline before claiming syntactic anchoring (see Phase 6 below). This is the single most likely reviewer objection to the lead result, and the design must pre-empt it.

---

## 4. Behavioral gate — and a sharper framing from Stolfo

Gate G3 already requires three behavioral checks before any probing: accuracy in a must-compute band, a no-op check confirming the additive identity is genuinely a no-op (predictions track `B*C`), and a must-compute check requiring graceful accuracy degradation with operand size (not pinned at 100% = lookup, not collapsed = no computation).

The review surfaces a framing improvement worth adding to the paper's motivation. Stolfo, Belinkov & Sachan (2023) found that proficient arithmetic models show a **division of labor**: attention propagates operand information to the final token, then late-token MLPs aggregate it into result information, and this division is **absent in less proficient models**. For us this means the behavioral gate is not just "can the model do the task" but "is the model in the regime where a structural mechanism could even exist." If Llama-3.1-8B is proficient in the must-compute band (G3 passes), the Stolfo result predicts there is an attention-then-MLP pipeline to localize, which is exactly what the precedence question probes. Cite this as the reason the behavioral gate licenses the mechanistic phase.

---

## 5. Phase 6 — the lead result: distance-padding probes (build spec)

This is the result the paper leads with regardless of how depth resolves, so it must satisfy probing standards 1-4 in full.

**What to decode.** A linear probe for the intermediate value at the residual stream, at the token position right before the final operation, swept across layers and across padding lengths.

**Probe class and training.** Start linear (Zhu et al. 2025 found MLP probes give no clear advantage for number value, supporting near-linear encoding). Held-out train/test split, L2 regularization, report probe capacity. Probe each layer; report the layer that is most *selective*, not merely most accurate (Hewitt & Liang 2019).

**The three mandatory controls (this is where most of the work is):**

1. **Control-task selectivity (std. 2).** Train a parallel probe to predict a *random but fixed* label assigned per stimulus. Report selectivity = real-value-probe accuracy (or 1 − normalized MSE) minus control-probe accuracy. A high-decoding probe with high control accuracy is uninterpretable and must be reported as such.

2. **Pre-computation / input-only baseline (std. 3).** Decode the same value from (a) the embedding layer and (b) a prompt that contains the operands but where the computation has not yet occurred. "Decodable because computed here" requires that the probe at the target position/layer substantially beats the input-only baseline. This is also the **length-artifact guard** demanded by Zhu et al. (2025): include a probe for sequence length and confirm value-decodability is separable from length-decodability.

3. **Causal probe-direction test (std. 4).** Convert correlation to causation: erase or steer along the probe direction (LEACE for clean erasure, Belrose et al. 2023; or add the probe direction to the residual stream as in Zhu et al. 2025 and Matsumoto et al. 2022) and measure the change in the model's arithmetic output. Only after the intervention moves the answer can the paper claim the decoded value is *used*, not merely present.

**The padding sweep and the H1/H0 decision.** Train the value probe at each padding length. The sharp prediction:
- **H1 (syntactic):** decoding accuracy stays high and stable as padding grows (the carrier is anchored to syntactic role, not absolute position), *after* the length/input-only baseline is subtracted.
- **H0 (positional):** decoding degrades monotonically with padding distance.

**Decision point (gates spending on depth):** a clean result either direction is a publishable core. A flat-high-from-the-start probe means the input-only confound bit, and the baseline work must be redone before Phase 7. Resolve before spending on depth.

**Reporting.** Report decoding curves with the control-task and input-only baselines overlaid on the same axes; report the causal-intervention effect size; report selectivity per layer. The lead figure is decoding-vs-padding-length with baselines, not a bare accuracy number.

---

## 6. Phase 7 — depth contrast via activation patching (build spec)

Corroborates the lead result on the *where* axis. The pipeline is already validated (Gate G4); Phase 7 points the validated instrument at the novel contrast.

**What to patch.** Residual stream (`resid_post`) at `[layer, position]`, swept as a layer × position grid, using `token_map` / `token_map_for_record` from Phase 4 so indices are never recomputed ad hoc. Then attention-head outputs and MLP outputs at the critical positions. Patching the residual stream (not freezing attention) is the right choice for a *binding* question, because it lets attention recompute, unlike attribution-graph / transcoder methods that freeze attention and therefore cannot see the precedence mechanism.

**Corruption (std. 5, 6).** Patch between `depth_left` and `depth_right` (token-matched by construction, G2-guaranteed). This is symmetric token substitution, the recommended in-distribution corruption, not Gaussian noise. Design the corrupted prompt so it changes *structural/role* information while operands are held constant, mirroring Stolfo's result-controlled variant, so a positive patch isolates structural carriers rather than operand information.

**Direction (std. 8) — both, deliberately.** Phase 5 already implements **denoising** (clean→corrupted; sufficiency). Phase 7 must add **noising** (corrupted→clean; necessity). Interpret the pair via AND/OR-gate logic: noising finds all components of a serial circuit, denoising finds all components of a redundant one. The clean story is denoising-sufficiency and noising-necessity agreeing on the same critical position.

**Metric (std. 7).** Normalized logit difference, as in Phase 5's `recovery()` (0 = corrupted baseline, 1 = clean). Report a second continuous metric (probability or KL) to catch negative/suppression components.

**Self-repair guard (std. 10).** Because of the Hydra effect, do not conclude a position is unimportant from a single ablation. Corroborate with denoising and with multi-position patches. Report both single- and multi-position results; a discrepancy is evidence of backup behavior, not noise, and should be reported as such.

**Framing.** Treat the patch as *corroborating* the padding result, not decisive: the `*`-token's local neighborhood differs between conditions, so a difference here could come from adjacent-token features rather than hierarchy. The publishable spine is padding-invariance + depth-patch *agreeing*.

---

## 7. Phase 8 — depth-2 composition (build spec, highest risk)

Tests whether the heads routing the intermediate at depth-1 reuse at the second bracket in depth-2 (recursion) or whether different heads activate (depth-bound handler). Theory prior is against clean reuse; bounded-depth transformers are pushed toward flattened per-depth handlers, so expect partial overlap.

**The non-negotiable methodological addition:** do **not** score this binary same-heads/different-heads. Define a quantitative reuse metric **and a chance baseline** (how much head-overlap would two arbitrary similar expressions share by coincidence?). Without the baseline, 60% overlap is uninterpretable. This is the depth-2 analogue of probe selectivity: the observed number is meaningless without the control.

**Kill-switch.** If depth-2 returns ambiguous partial overlap the baseline cannot adjudicate, demote it to "preliminary / future work." Depth-1 (padding + patch) is the publishable spine; depth-2 is upside. Budget readiness to cut it.

---

## 8. Phase 9 — analysis and honest framing

Write toward the most-probable partial outcome, which neither extreme bin covers: "hierarchical for depth-1, degrading at depth-2 / under padding." Frame the negative/partial result as a precise characterization tied to the bounded-depth expressivity prediction; that theoretical connection is what elevates it above an empirical null and is the difference between a rejected and an accepted short paper.

**Positioning against the bag-of-heuristics line.** The Nikankin / Kantamneni-Tegmark / Anthropic results are the motivating prior, not a competitor: they showed *flat* arithmetic is heuristic-driven; this paper asks whether that heuristic character *persists under nesting* when precedence forces structure. State explicitly why causal patching + probing + attention analysis (tools that see attention, are cheap, give distributional answers) are methodologically appropriate where attribution-graph / transcoder methods are not (they freeze attention, the crux of a binding question; cost months of CLT training; give single-prompt not distributional claims).

**Length.** Target a 4-page short paper for the depth-1 result alone; 8 pages only if depth-2 resolves cleanly. Lead with padding-robustness, support with the depth patch, present depth-2 at whatever confidence it earned.

---

## 9. The rigor checklist (check the protocol against this before submission)

Probing:
- [ ] Linear probe, held-out split, L2, capacity reported
- [ ] Control-task probe trained in parallel; **selectivity** reported per layer
- [ ] Embedding-layer + pre-computation **input-only baseline** reported on the same axes
- [ ] **Length-decodability** separated from value-decodability (Zhu et al. guard)
- [ ] **Causal** probe-direction intervention (LEACE/steering) moves the model output
- [ ] Lead figure = decoding-vs-padding with baselines overlaid, not a bare number

Patching:
- [ ] Clean/corrupted **token-length matched, minimally different** (G2 enforces)
- [ ] **Token substitution**, not Gaussian noise (Zhang & Nanda)
- [ ] **Normalized logit difference** primary + one more continuous metric
- [ ] **Both** noising (necessity) and denoising (sufficiency); AND/OR interpretation
- [ ] **Known result reproduced first** (G4); model solves clean task; metric sign confirmed
- [ ] **Self-repair guard**: single-component results corroborated with denoising + multi-component
- [ ] Indices from Phase 4 `token_map`, never recomputed ad hoc

Claims:
- [ ] H1 tested against an explicit H0 (depth-bound heuristic), not in isolation
- [ ] Partial outcome has a prepared, theory-linked characterization
- [ ] Minimality explicitly disclaimed (short-paper localization claim, not full circuit)
- [ ] Distribution/prompt-specificity of results stated as a limitation

---

## 10. Build order and current state

**Built and verified (the instrument):** Gates G0 (env + hooks), G2 (controlled stimulus + machine-checked assertions), G3 (behavioral kill-switch), G4 (patching pipeline validated on a known addition result), plus Phase 4 token-boundary map. The denoising direction and normalized logit-diff metric for patching are already implemented to standard.

**To build (the experiment), in order:**
1. **Phase 6 probe cell** — value probe + control-task probe + input-only/length baselines + causal intervention, swept over layers and padding lengths. This is the lead result and the largest remaining build. (Standards 1-4.)
2. **Phase 7 noising + self-repair cell** — add the necessity direction to the existing denoising sweep; add multi-position corroboration. (Standards 8, 10.)
3. **Phase 8 reuse-metric cell** — quantitative head-reuse with a chance baseline. (Selectivity analogue.)
4. **Phase 9 analysis** — figures, baselines overlaid, theory-linked framing.

**Immediate next artifact:** the Phase 6 probe cell, because it is the publishable spine, it is where the probing standards are satisfied or not, and (the value-probe + baseline portion) much of it runs without the depth machinery. It should reuse the repo's checkpoint/artifact helpers and the locked operand band from G3 so it is disconnect-resumable like the rest.

---

## Key references (by method, for the related-work section)

**Probing standards:** Alain & Bengio 2016 (linear probes); Hewitt & Liang 2019 (control tasks, selectivity); Pimentel et al. 2020 (information-theoretic probing, the contested counter-view); Belinkov 2022 (survey); Elazar et al. 2021 (amnesic probing); Ravfogel et al. 2020 (INLP); Belrose et al. 2023 (LEACE).

**Patching standards:** Vig et al. 2020 (causal mediation); Meng et al. 2022 (ROME / causal tracing); Geiger et al. 2021 (interchange interventions / causal abstraction); Wang et al. 2022 (IOI; faithfulness/completeness/minimality; mean-ablation); Conmy et al. 2023 (ACDC); Zhang & Nanda 2023 (corruption best practices); Heimersheim & Nanda 2024 (how to use/interpret patching); Nanda 2023 / Kramár et al. 2024 (attribution patching / AtP*); McGrath et al. 2023 (Hydra effect / self-repair).

**Arithmetic interpretability (direct prior art):** Stolfo, Belinkov & Sachan 2023 (causal mediation; attention-then-MLP division of labor; result-controlled variant); Nikankin et al. 2024 (bag of heuristics); Kantamneni & Tegmark 2025 (Clock algorithm; weaker fits for Llama-3.1-8B); Zhu et al. 2025 (numbers encoded linearly; length-before-value); Matsumoto et al. 2022 (MathNLP; trace-then-intervene template); Anthropic 2025 (circuit tracing / attribution graphs; addition heuristics replication).
