"""Appendix: by-layer decodability of operand B at the ')' site (fewshot_decodability_by_layer.csv).
Key point (0-shot, solid): B is linearly decodable at EVERY layer including the last (base L31
R2=0.92, instruct 0.98) in the FAILING zero-shot regime -> decodable-but-unused is depth-robust.
The few-shot curves (dashed) drop mid-late; shown for completeness only, with NO mechanistic claim.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import plt, BLUE, ORANGE, GRAY, DATA, save

LAYERS = [1, 8, 16, 20, 24, 28, 31]
COLS = ["L1", "L8", "L16", "L20", "L24", "L28", "L31"]


def main():
    rows = list(csv.DictReader(open(os.path.join(DATA, "fewshot_decodability_by_layer.csv"))))
    def curve(tag, shots):
        for r in rows:
            if r["tag"] == tag and int(r["shots"]) == shots:
                return [float(r[c]) for c in COLS]
        return None
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for tag, col in [("base", BLUE), ("instruct", ORANGE)]:
        ax.plot(LAYERS, curve(tag, 0), "-o", color=col, ms=3, lw=1.3, label=f"{tag}, 0-shot (fails)")
        ax.plot(LAYERS, curve(tag, 4), "--s", color=col, ms=2.5, lw=0.9, alpha=0.55,
                label=f"{tag}, 4-shot (recovers)")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$R^2$: decode $B$ at the ')' site")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.9, color=GRAY, lw=0.5, ls=":")
    ax.legend(loc="lower left", frameon=False, fontsize=5.8)
    ax.set_title("Operand decodable at every layer (0-shot)", fontsize=8)
    save(fig, "figA_decode.pdf")


if __name__ == "__main__":
    main()
