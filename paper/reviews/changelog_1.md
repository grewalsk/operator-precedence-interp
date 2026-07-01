# Changelog 1 (skeleton and verification)

- Created `paper/` with ACL scaffold, `build.sh`, `DATA.md`, `CLAIM.md`, `OPEN_ITEMS.md`.
- Re-verified all §2 ground-truth numbers against committed `origin/main` files with `git show`;
  staged the exact CSV/JSON into `paper/data/` so figures are reproducible.
- Reconciled two work-order assumptions to committed data (swap does not flip -> positive
  control moved to the operand route; band2 gap CI excludes zero -> "weakly represented but
  causally dormant" branch). Recorded in OPEN_ITEMS.md; resolves the §1 Conditional.
- Generated Figure 2 and three appendix figures from committed data; wrote the TikZ hero (Fig 1).
- Assembled `refs.bib` (11 entries, venue + interpretability), all fetched or hand-written with
  URLs for Sweep-5 verification.
- Resolved: what each critic point above changed -> the swap-flip framing was removed from the
  claim before it entered any prose (prevents a fifth retraction).
