# ============================================================================
# Phase 6 / WO#3 follow-up — BY-LAYER decodability curve (the load-bearing read).
# ----------------------------------------------------------------------------
# The WO#3 BEST-LAYER number (R^2~0.99) is misleading: that is a layer-1-3 value
# and B is a recent token, so an early-residual probe recovers it for free. The
# real question is whether B stays decodable INTO the mid-late consumption zone
# (layers >= 0.6*n_layers, where G4 puts composition ~L30). This cell plots the
# FULL by-layer R^2 curve per (tag, shots) from the WO#3 artifacts and reads off
# the mid-late values. Pure CPU (reads r2_by_layer; no model needed).
#
# WHAT THE 2026-06-24 RUN SHOWED (asymmetric, NOT "identical in both"):
#   - 0-shot (FAILS): B decodable ~0.91-0.99 at EVERY layer incl. L31 -> the
#     operand is present at its site through the whole net and still unused
#     (depth-robust decodable-but-unused; immune to the recent-token objection).
#   - few-shot (SUCCEEDS): operand decodability COLLAPSES mid-late (trough ~0.50),
#     monotone in shot count -> the "identical encoding" hypothesis is FALSE.
#   The few-shot drop is a confounded LEAD (consumption vs context dilution) ->
#   the 82f control disambiguates it.
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt

_FSC_SHOTS, _FSC_TAGS = [0, 2, 4], ["base", "instruct"]


def _fsc_curve(tag, shots):
    """{layer:int -> R^2} from the in-memory WO#3 probe, else the saved artifacts."""
    g = globals().get("WO_FSPROBE", {}).get(tag, {}).get(shots)
    if g is None and has_artifact(f"wo_fsprobe_summary_{tag}", "json"):
        g = load_json(f"wo_fsprobe_summary_{tag}")["by_shots"].get(str(shots))
    if g is None and has_artifact(f"wo_fsprobe_{tag}_{shots}", "json"):
        g = load_json(f"wo_fsprobe_{tag}_{shots}")
    return {int(k): (None if v is None else float(v))
            for k, v in (g or {}).get("r2_by_layer", {}).items()}


_fsc_layers = sorted({L for t in _FSC_TAGS for s in _FSC_SHOTS for L in _fsc_curve(t, s)})
_FSC_NL = (max(_fsc_layers) + 1) if _fsc_layers else 32
_FSC_ML = int(np.floor(0.6 * _FSC_NL))                       # mid-late "consumption zone" start
_FSC_G4 = 30                                                 # where G4 found the product composed
_FSC_ANCH = [L for L in (1, 8, 16, 20, 24, 28, min(31, _FSC_NL - 1)) if L < _FSC_NL]


def _fsc_vals(c):
    return np.array([c.get(L, np.nan) for L in range(_FSC_NL)], dtype=float)


def _fsc_ml(c):
    v = _fsc_vals(c)[_FSC_ML:]; v = v[np.isfinite(v)]
    return (float(np.nanmin(v)), float(np.nanmean(v))) if v.size else (np.nan, np.nan)


# ---- table + committable CSV deliverable ----
print(f"\nBy-layer B-decodability R^2 — mid-late zone = L{_FSC_ML}..{_FSC_NL - 1} "
      f"(G4 composed the product ~L{_FSC_G4})")
print(f"{'tag':<9}{'shots':>6}  " + "".join(f"L{L:<6}" for L in _FSC_ANCH) + f"{'ML.min':>8}{'ML.mean':>8}")
_fsc_rows, _fsc_ml_stats = [], {}
for tag in _FSC_TAGS:
    for s in _FSC_SHOTS:
        c = _fsc_curve(tag, s); mn, me = _fsc_ml(c); _fsc_ml_stats[(tag, s)] = (mn, me)
        print(f"{tag:<9}{s:>6}  " + "".join(f"{c.get(L, float('nan')):<7.3f}" for L in _FSC_ANCH)
              + f"{mn:>8.3f}{me:>8.3f}")
        row = {"tag": tag, "shots": s, "ml_min": mn, "ml_mean": me}
        row.update({f"L{L}": c.get(L) for L in _FSC_ANCH})
        _fsc_rows.append(row)
wo_save_result("fewshot_decodability_by_layer.csv",
               wo_battery_csv(_fsc_rows, ["tag", "shots"] + [f"L{L}" for L in _FSC_ANCH] + ["ml_min", "ml_mean"]))

# ---- asymmetric verdict (NOT the binary "decodable in both") ----
print("\nREAD (asymmetric — the failing and succeeding regimes differ mid-late):")
for tag in _FSC_TAGS:
    a0 = _fsc_ml_stats[(tag, 0)][0]                 # 0-shot mid-late MIN = decodable-but-unused anchor
    d0, dr = _fsc_ml_stats[(tag, 0)][1], _fsc_ml_stats[(tag, 4)][1]
    anchor = ("B AVAILABLE mid-late in the FAILING regime -> depth-robust decodable-but-unused holds"
              if a0 >= 0.70 else "B decays mid-late even 0-shot -> the unused anchor is weak")
    print(f"  [{tag}] 0-shot mid-late R^2 min={a0:.2f} -> {anchor}.")
    print(f"         few-shot mid-late DROP: {d0:.2f}(0sh) -> {dr:.2f}(4sh)  (Δ={d0 - dr:+.2f}); "
          f"NOT 'identical encoding'. Run cell 82f to tell consumption (R1) from dilution (R2).")

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, tag in zip(axes, _FSC_TAGS):
    for s in _FSC_SHOTS:
        ax.plot(range(_FSC_NL), _fsc_vals(_fsc_curve(tag, s)), "o-", ms=3, label=f"{s}-shot")
    ax.axvspan(_FSC_ML, _FSC_NL - 1, color="orange", alpha=0.12, label="mid-late zone")
    ax.axvline(_FSC_G4, color="k", ls="--", lw=1, label=f"G4 compose ~L{_FSC_G4}")
    ax.axhline(0.70, color="grey", ls=":", lw=1)
    ax.set_title(f"{tag}: B decodability vs layer @ ')' site"); ax.set_xlabel("layer")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
axes[0].set_ylabel("CV-R^2 (B from ')' residual)"); fig.tight_layout()
try:
    fig.savefig(str(WO_RESULTS / "fewshot_decodability_by_layer.png"), dpi=130)
except Exception as e:
    log(f"(by-layer plot save skipped: {e})")
plt.show()
