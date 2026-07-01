# Sweep 1 review — skeleton and verification

Theme: build the ACL scaffold, verify every number against committed files, generate figures
from real data, collect references. Critic pass below scores the skeleton against the exit gate
("no unverified number anywhere; OPEN_ITEMS.md lists every gated item").

## What was built
- ACL scaffold (`main.tex` + `sections/*.tex`), `acl.sty`/`acl_natbib.bst` fetched from the
  official acl-org repo, `build.sh` (figures then latexmk). Compiles to a 4-page body PDF.
- `DATA.md`: every number re-read from `origin/main` via `git show`, tagged behavioral /
  decodability / causal, with its source file and CI. This is the single source of truth.
- Figures from committed data only: `figs/make_fig2_evidence.py` (selectivity gap + causal
  share), `figs/make_figA_{wrapper,decode,steering}.py`. Figure 1 is a TikZ schematic.
- `refs.bib`: 11 entries; the 3 venue papers (Williamson 2025, Matsumoto 2022, Reusch 2022)
  and 3 interpretability papers pulled verbatim from ACL Anthology; 5 arXiv/ICLR entries.

## Critic findings (skeleton)
- [RESOLVED in build] Two work-order assumptions were contradicted by committed data and had to
  be reconciled before any prose: (a) the full-residual swap does NOT flip the answer
  (emit-P'=0; lateswap PRODUCT_NOT_AT_EQUALS, fullproduct SITE_OR_METRIC_STILL_BROKEN); (b) the
  band2 gap CI IS committed and EXCLUDES zero. Both are recorded in OPEN_ITEMS.md and folded
  into CLAIM.md. The positive control was re-based onto the operand-route patch (recovery 0.50),
  which is a real committed control.
- [NOTED] `dormant_certification_*` on main is all-null (a first-token-basis bug, fixed in cell
  82s but not re-run); the paper does not rely on it. In OPEN_ITEMS.md.
- [PASS] No number appears in the draft that is absent from DATA.md.

## Exit gate: MET
No unverified number; OPEN_ITEMS.md lists every gated/reconciled item; PDF compiles.
