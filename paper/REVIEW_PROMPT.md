# Review prompt: theory-vs-data audit + venue-fit review of a MathNLP short paper

You are a senior mechanistic-interpretability researcher and an experienced MathNLP reviewer /
area chair. You have seen many interpretability papers succeed and fail at this venue. Your job
is NOT to rewrite or re-run anything. Your job is to (1) confirm the paper's theory and claims
are exactly supported by the committed data, (2) judge whether the paper is written and framed
optimally for MathNLP, and (3) hand back a concrete, prioritized plan for what to change now and
what to do next to maximize its odds of acceptance.

## Hard constraints (read-only reviewer)

- Do NOT write, edit, or run any code. Do NOT run experiments, notebooks, `build.sh`, figure
  scripts, or GPU cells. Do NOT modify the repo.
- Verify by READING only: read the paper, and read the committed data files to check numbers.
  `git show origin/main:<file>` and opening files under `paper/` are the only actions you take.
- If a claim cannot be checked from committed files, say so and treat it as unsupported; do not
  reconstruct or recompute it yourself.

## Access

Public repo: `https://github.com/grewalsk/operator-precedence-interp`, branch
`wo6-operand-localization`. `git fetch origin` first.
- Paper: `paper/paper.pdf` (compiled) and `paper/main.tex` + `paper/sections/*.tex`.
- **Ground-truth data is at the ROOT of `origin/main`** (CSV/JSON): read with
  `git show origin/main:<file>` (e.g. `instruct_battery.csv`, `band_robustness_base.json`,
  `causal_share_base.json`, `dose_response_base.json`, `probe_selectivity_base.json`,
  `operand_localization_summary.csv`, `cross_model_battery.csv`, `confidence_intervals.json`).
- `paper/DATA.md` is the authors' number-to-file mapping. `paper/OPEN_ITEMS.md` and
  `paper/FINAL_STATUS.md` state the authors' own known caveats; read them, then judge
  independently whether they are complete and whether the residuals are disqualifying.

## Venue

The target is MathNLP (the ACL Workshop on Mathematical Natural Language Processing), a short
paper (4-page body). Calibrate everything to that venue: its scope (understanding and reasoning
about mathematics in LMs), its audience (NLP + a growing interpretability contingent), its prior
work (esp. Williamson et al. 2025, Matsumoto et al. 2022, Reusch and Lehner 2022), and short-paper
norms (one crisp contribution, honest scope, no filler).

## Task 1 — Is the theory right, and exactly supported by the data?

Go claim by claim through the paper (abstract, contributions, and Sections 4-6) and, for each:
- Find the number in a committed file and confirm the paper states it correctly (value, rounding,
  which model/band/site). Flag any number that is absent, disagrees, or is presented with the
  wrong scope.
- Confirm the *interpretation* is licensed by the evidence, not just the number. Tag each result
  behavioral / decodability (correlational) / causal, and flag any place a decodability R^2 or a
  correlation is used to support a causal statement.
- Specifically stress-test the load-bearing theory:
  - **"operand-dominated"**: does gap ~ 0 at band (20,49) plus a small gap (CI excludes 0) at
    (2,99) actually justify "weakly represented"? Is band (2,99) behaviorally in scope (check
    `acc_C4_band2`, `in_scope_strict` in `band_robustness_*.json`), and is that disclosed
    honestly? Is the pre-registered layer (`prereg_gap_*.json`) genuinely independent of the
    argmax it is meant to guard against?
  - **"causally inert"**: is the null interpretable given that its positive control
    (operand-position recovery 0.50) uses a different metric and operation than the answer-site
    swap? Is `emit-P' = 0` doing real work or is it saturating (does `dose_response_*.json` show
    emit-P' = 0 even when the logprob moves with a CI clear of 0)? Is the 14%/85% causal-share
    decomposition (`causal_share_*.json`) an honest description of where a sub-flip effect lives,
    or is it over-read as a causal result? Is the instruct arm's decode-direction component
    actually significant?
  - **"decodability != computation" / dormant subspace**: is the Makelov framing correctly
    applied here, and is the operand-route localization (`operand_localization_summary.csv`,
    `head_path_patch_*.json`) strong enough to say "computed at the operands" rather than only
    "answer-site is not the locus"?
- Report anything the authors' own OPEN_ITEMS/FINAL_STATUS missed or under-stated.

## Task 2 — Is it written optimally for MathNLP?

Judge as a reviewer deciding accept/reject, focusing on presentation and fit, not just content:
- **Framing and title.** Does the title and the first half-page land the single contribution
  cleanly? Is the behavioral hook clearly subordinate to the mechanistic claim? Is the "illusion"
  framing compelling or overwrought for this audience?
- **Contribution clarity.** Are the contributions crisp, non-overlapping, and each independently
  defensible? Is the strongest result (the dissociation) foregrounded?
- **Positioning.** Is the delta over Williamson et al. 2025 (same venue) sharp and fair, or does
  the behavioral/cross-model material read as a re-run? Are Matsumoto and Reusch used well?
- **Evidence presentation.** Do Figure 1, Figure 2 (a/b/c), and the tables let a reviewer
  reconstruct the argument without the prose? Are CIs everywhere they should be? Is anything in
  the abstract stronger than the body delivers?
- **Scope and honesty.** For a workshop that rewards clean, honest, well-scoped work: is the
  single-instance, single-wrapper, one-model scope a feature or a liability here, and is it
  handled well? Is the cross-model section correctly a matched-format replication plus a format
  confound (never a family dissociation)?
- **Short-paper economy.** Is any body space wasted that should go to strengthening the core, and
  is anything crammed that should move to the appendix?

## Task 3 — What to change and do next to improve the odds

This is the primary deliverable. Give a prioritized, actionable plan, split into:
- **Fix now (writing / framing, no new experiments):** the specific edits that most raise the
  paper's ceiling at MathNLP, in priority order, each tied to a location.
- **Run before submission (analyses the authors should do):** the one or two additional
  experiments or controls most likely to convert a weak-accept into an accept (e.g. a same-metric
  answer-site positive control, a second wrapper family, a same-model chat-format check),
  with why each matters and roughly how much it would move the needle. Do not run them; recommend
  them.
- **Optional stretch:** anything that would broaden impact if time allows.

## Output

1. **Theory-vs-data ledger:** a table of the load-bearing claims, each marked supported /
   imprecise / unsupported, with the committed file and value you checked against.
2. **Venue-fit assessment:** green/yellow/red on framing, contribution clarity, positioning,
   evidence presentation, scope/honesty, short-paper economy, each with a one-line reason.
3. **Prioritized improvement plan** (Task 3), fix-now and run-next separated.
4. **Verdict and odds:** accept / weak-accept / borderline / weak-reject / reject for MathNLP,
   an honest probability-of-acceptance estimate, and the single highest-leverage change.

Be rigorous and specific; prefer findings a reviewer can confirm by opening the data directory
over stylistic nits, and make the improvement plan concrete enough to act on without you.
