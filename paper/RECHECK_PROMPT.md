# Re-check prompt: verify the round-2 revisions landed and nothing regressed

You are an expert mechanistic-interpretability researcher and MathNLP reviewer, doing a focused
re-check of a short paper you (or a peer) reviewed once already. The authors say they applied your
findings. Your job is to confirm each fix is REAL and correct against the committed data, catch any
regression or new problem, and give a final go / no-go for submission. Read-only: do not run code,
experiments, or the build; verify by reading the paper and the committed data files.

## Access
Repo `https://github.com/grewalsk/operator-precedence-interp`, branch `wo6-operand-localization`
(`git fetch origin` first). Paper: `paper/paper.pdf` and `paper/main.tex` + `paper/sections/*.tex`.
Ground-truth data at the ROOT of `origin/main` via `git show origin/main:<file>`. The authors'
claimed fixes are in `paper/FINAL_STATUS.md` (section "Round 2") and the change is committed at
`git log`. Treat these as claims to verify, not as truth.

## Verify each claimed fix actually landed (and is correct)
1. **4-page body.** Confirm the entire body, through the Conclusion, ends on page 4 and page 5
   begins with the References. This was a desk-reject risk; check it yourself, column-aware.
2. **The §5 causal-null sentence.** Confirm the first-token null (vs random/shuffled,
   non-discriminative) and the full-product dose-response are now separate and correctly
   attributed, and that the promoted dose-response numbers match `dose_response_base.json` (C1
   flat at every dose k in {1,2,4,8}, CIs bracket 0; C4 +0.08 -> +0.46, CIs exclude 0, emit-P'=0).
3. **Swap surface + magnitude.** Confirm the swap/decomposition is now named as the C4 "=" site
   (in text and Fig 2b) and that "+0.59 at L19, peak +0.66 at L23" matches
   `causal_steering_lateswap_base.json`.
4. **Terminology + honesty edits.** Confirm: "pre-registered" is gone (replaced by "C1-independent
   criterion"); the contribution bullet says "85% of the swap's answer-effect" (not "variance");
   the band-2 product reading is hedged; B-at-"=" is scoped to 0.96-0.97; the nesting control is
   within-instruct (0.25 from 0.99); the wrapper position-dependence caption is scoped to base;
   `n_boot=300` for gap CIs is stated.
5. **Repo hygiene.** Confirm `DATA.md` flags the stale `inject_C1_L4=+1.99` artifact and that
   `sections/07_conclusion.tex` is deleted.

## Independently re-audit the changed passages
For every number in the rewritten §5 and in the abstract/intro/§4 edits, re-trace it to a committed
file. Flag any number that is now wrong, any new overclaim introduced by the rewrite, any place the
tightening broke a sentence's logic, and anything the authors' Round-2 note claims but did not
actually do. Also re-scan for the standing guardrails: no claim beyond its evidence level, a
positive control beside every null, cross-model = replication + confound, no em dashes.

## Output
1. A fix-by-fix checklist: for each item above, LANDED / PARTIAL / NOT DONE, with the file+line or
   committed value you checked.
2. Any new or still-open defects (severity-ranked, located).
3. A final verdict: submission-ready as is, or the short list of must-fix items before submission,
   and an updated accept-probability estimate for MathNLP.
