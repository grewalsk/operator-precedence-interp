"""Appendix: wrapper blind-spot heatmap. Multiplication accuracy for each wrapper,
base and instruct, from wrapper_map_summary.csv. Shows the blind spot is (i) not
'every wrapper' (base ( B + 0 ) barely moves), (ii) not 'instruct worse everywhere'
(instruct better on plain parens), (iii) present for both ops under nesting.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import plt, DATA, save
import numpy as np

WRAPPERS = [("bare", "B"), ("paren", "( B )"), ("add0L", "( 0 + B )"),
            ("add0R", "( B + 0 )"), ("mul1L", "( 1 * B )"), ("mul1R", "( B * 1 )"),
            ("nest2", "(( 0 + B ))"), ("nest3", "((( 0 + B )))")]


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "wrapper_map_summary.csv"))))
    def acc(tag, wrapper, op):
        for r in rows:
            if r["tag"] == tag and r["wrapper"] == wrapper and r["op"] == op:
                return float(r["acc"])
        return np.nan
    tags = ["base", "instruct"]
    ops = ["mul", "add"]
    # matrix rows = wrappers, cols = base-mul, instruct-mul (main), plus add as annotation
    M = np.array([[acc(t, w[0], "mul") for t in tags] for w in WRAPPERS])
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    im = ax.imshow(M, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["base", "instruct"])
    ax.set_yticks(range(len(WRAPPERS)))
    ax.set_yticklabels([w[1] for w in WRAPPERS], fontsize=6.5, fontfamily="monospace")
    for i in range(len(WRAPPERS)):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if 0.35 < M[i, j] < 0.8 else "white")
    ax.set_title("multiplication accuracy by wrapper", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "figA_wrapper.pdf")


if __name__ == "__main__":
    main()
