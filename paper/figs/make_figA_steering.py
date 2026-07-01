"""Appendix: probe-direction steering at the '=' site is inert across all layers
(causal_steering_summary.csv). inject / random / shuffled mean GT-logit Delta ~ 0 with
95% CIs through zero at every layer, for base and instruct. This is the WO#5 first-token
metric (a NULL); the discriminative full-product re-score is Figure 2b in the body.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import plt, BLUE, ORANGE, GREEN, GRAY, DATA, save
import numpy as np


def series(rows, tag, cond):
    xs, ys, lo, hi = [], [], [], []
    for r in rows:
        if r["tag"] == tag and r["site"] == "equals" and r["condition"] == cond:
            xs.append(int(r["layer"])); ys.append(float(r["mean_delta"]))
            lo.append(float(r["ci_lo"])); hi.append(float(r["ci_hi"]))
    o = np.argsort(xs)
    return (np.array(xs)[o], np.array(ys)[o], np.array(lo)[o], np.array(hi)[o])


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "causal_steering_summary.csv"))))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, tag in zip(axes, ["base", "instruct"]):
        for cond, col in [("inject", BLUE), ("random", GRAY), ("shuffled", GREEN)]:
            x, y, lo, hi = series(rows, tag, cond)
            if len(x) == 0:
                continue
            ax.plot(x, y, "-o", color=col, ms=2.5, lw=1.0, label=cond)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel("layer")
        ax.set_ylim(-0.05, 0.05)
    axes[0].set_ylabel(r"mean $\Delta$ GT-token logit")
    axes[0].legend(loc="upper right", frameon=False, fontsize=6)
    save(fig, "figA_steering.pdf")


if __name__ == "__main__":
    main()
