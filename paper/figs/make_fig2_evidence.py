"""Figure 2 (centerpiece). Two panels, all from committed data.
Left:  selectivity gap (R2_real - linear-(B,C) baseline) at two bands, with 95% CI.
       band (20,49) has zero headroom (gap ~ 0, operand-explained); band (2,99) has
       headroom and the gap CI EXCLUDES 0 (a small, real product component).
Right: causal-share of a full-residual swap at '='. logprob Delta(P'-true) for the whole
       swap vs. the swap with the decode direction ablated (perp) vs. the decode direction
       alone (w-only). The decodable direction reproduces a small fraction; ~85% of the
       (sub-flip) effect is orthogonal to it. emit-P' = 0 throughout (no answer flips).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (plt, BLUE, ORANGE, GREEN, GRAY, LGRAY, load_json, find_node, save)


def selectivity_panel(ax):
    # band (2,99): decision-grade paired-bootstrap gap CI (band_robustness).
    b2b = load_json("band_robustness_base.json")
    b2i = load_json("band_robustness_instruct.json")
    def band2_gap(d):
        g = (d.get("gap_ci") or {}).get("C1_equals") or {}
        gm, ci = g.get("gap_mean"), g.get("gap_ci")
        return gm, ci
    gm_b, ci_b = band2_gap(b2b)
    gm_i, ci_i = band2_gap(b2i)
    # band (20,49): probe_selectivity single-split gap (~0, zero headroom).
    psb = find_node(load_json("probe_selectivity_base.json"), site="C1_equals", target="B_times_C")
    psi = find_node(load_json("probe_selectivity_instruct.json"), site="C1_equals", target="B_times_C")
    g1_b = psb["baseline_gap"] if psb else 0.0
    g1_i = psi["baseline_gap"] if psi else 0.0

    groups = ["band (20,49)\n(zero headroom)", "band (2,99)\n(headroom ~0.14)"]
    x = [0, 1]
    w = 0.34
    # base bars
    base_vals = [g1_b, gm_b]
    inst_vals = [g1_i, gm_i]
    base_err = [[0, gm_b - ci_b[0]], [0, ci_b[1] - gm_b]]
    inst_err = [[0, gm_i - ci_i[0]], [0, ci_i[1] - gm_i]]
    ax.bar([xi - w/2 for xi in x], base_vals, w, yerr=base_err, capsize=2.5,
           color=BLUE, label="base", error_kw=dict(lw=0.8))
    ax.bar([xi + w/2 for xi in x], inst_vals, w, yerr=inst_err, capsize=2.5,
           color=ORANGE, label="instruct", error_kw=dict(lw=0.8))
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(r"selectivity gap  $R^2_{\mathrm{real}}-R^2_{\mathrm{linear}(B,C)}$")
    ax.set_title("(a) Decode is operand-dominated")
    ax.set_ylim(-0.03, 0.12)
    ax.legend(loc="upper left", frameon=False)
    ax.annotate("gap $\\approx$ 0\n(operand-explained)", xy=(0, 0.004), xytext=(-0.35, 0.055),
                fontsize=6.5, color=GRAY, ha="left",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6))
    ax.annotate("CI excludes 0\n(small real component)", xy=(1.17, gm_i), xytext=(0.55, 0.10),
                fontsize=6.5, color=GRAY, ha="left",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6))


def causal_panel(ax):
    csb = load_json("causal_share_base.json")
    csi = load_json("causal_share_instruct.json")
    labels = ["full swap", "decode dir.\nablated (perp)", "decode dir.\nonly (w-only)"]
    keys = ["full", "perp", "wonly"]
    def vals(d):
        return [d[k]["logprob_delta"] for k in keys], [d[k]["logprob_ci"] for k in keys]
    vb, cib = vals(csb); vi, cii = vals(csi)
    x = [0, 1, 2]; w = 0.34
    def err(v, ci):
        return [[v[j] - ci[j][0] for j in range(3)], [ci[j][1] - v[j] for j in range(3)]]
    ax.bar([xi - w/2 for xi in x], vb, w, yerr=err(vb, cib), capsize=2.5, color=BLUE,
           label="base", error_kw=dict(lw=0.8))
    ax.bar([xi + w/2 for xi in x], vi, w, yerr=err(vi, cii), capsize=2.5, color=ORANGE,
           label="instruct", error_kw=dict(lw=0.8))
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"full-product logprob $\Delta(P'\!-\!\mathrm{true})$ (nats)")
    ax.set_title("(b) Decodable direction is not the causal carrier")
    ax.set_ylim(0, 0.80)
    ax.legend(loc="upper right", frameon=False)
    # w_share annotation (base)
    ws = vb[2] / vb[0] if vb[0] else 0.0
    ax.annotate(f"w-only reproduces\nonly {ws:.0%} of the effect\n(emit-$P'$ = 0: no flip)",
                xy=(2 - w/2, vb[2]), xytext=(0.75, 0.60), fontsize=6.5, color=GRAY, ha="left",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6))


def site_panel(ax):
    """The dissociation the paper rests on: the same-family patching moves the answer at the
    operand positions (clean-answer recovery) but the answer-site swap flips nothing (emit-P').
    Different metrics (labeled), both a [0,1] 'fraction of the answer moved'."""
    import csv, os
    from _style import DATA
    rows = list(csv.DictReader(open(os.path.join(DATA, "operand_localization_summary.csv"))))
    def rec(tag):
        for r in rows:
            if r["tag"] == tag and r["role"] == "b_last":
                return float(r["exact_recovery_best"])
        return 0.0
    csb = load_json("causal_share_base.json"); csi = load_json("causal_share_instruct.json")
    labels = ["operand patch\n(recovery)", "answer-site swap\n(emit-$P'$)"]
    base = [rec("base"), csb["full"]["emit_Pprime_rate"]]
    inst = [rec("instruct"), csi["full"]["emit_Pprime_rate"]]
    x = [0, 1]; w = 0.34
    ax.bar([xi - w/2 for xi in x], base, w, color=BLUE, label="base")
    ax.bar([xi + w/2 for xi in x], inst, w, color=ORANGE, label="instruct")
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of answer moved")
    ax.set_title("(c) A dead site, not a dead probe")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", frameon=False)
    ax.annotate("same machinery moves\nthe answer at operands", xy=(0, 0.5), xytext=(-0.3, 0.30),
                fontsize=6.2, color=GRAY, ha="left",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6))


def main():
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(8.8, 1.38))
    selectivity_panel(axL)
    causal_panel(axM)
    site_panel(axR)
    fig.tight_layout(w_pad=1.3)
    save(fig, "fig2_evidence.pdf")


if __name__ == "__main__":
    main()
