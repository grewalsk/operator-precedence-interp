"""Shared figure style: colorblind-safe (Wong 2011), ACL-body font sizes, vector PDF."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = HERE

# Wong 2011 colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#787878"
LGRAY = "#BBBBBB"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 200,
    "pdf.fonttype": 42,   # editable/embedded fonts, no Type-3
    "ps.fonttype": 42,
})


def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def find_node(o, **match):
    """Recursively find the first dict in o whose items match all key=value pairs."""
    if isinstance(o, dict):
        if all(o.get(k) == v for k, v in match.items()):
            return o
        for v in o.values():
            r = find_node(v, **match)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_node(v, **match)
            if r is not None:
                return r
    return None


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}")
