# -*- coding: utf-8 -*-
"""Fig. 1 and Fig. 2 of the paper, drawn deterministically in points at their final IEEE column widths.

  fig_ambiguity : one column  (252 x 213 pt)  one retrieval, two compatible populations
  fig_reuse     : two columns (516 x 200 pt)  selection-safe evidence reuse: the pointer rule, and three ways to use
                                              a history at the same new-call budget

Design: the redesign supplied with the submission (COHORT_Figures_1_2, drawn from the current manuscript).  Geometry,
wording and typography are the supplied ones.  Two changes: the three intervals of Fig. 2 are read from
reports/icde2027_suite/reuse_intervals.json (make_reuse_intervals.py: BM25 history covering 15% of the population,
2% new calls, seed 0) instead of being typed in, and the drawing refuses to render if an interval no longer tells the
story of the caption; the text of the grey "discard" row is one step darker than its bar, for print legibility.
The sequence drawings are schematic: the number of records in a queue is not a sample size.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from fig_style import ACCENT, GRAY, INK, LIGHT, MID, RULE, TINT, WHITE, check_text_inside

_HERE = Path(__file__).resolve().parent
_RES = _HERE.parent / "reports" / "icde2027_suite"


# --------------------------------------------------------------------------- #
# drawing primitives (coordinates in points, y grows downwards)
# --------------------------------------------------------------------------- #
def canvas(w, h):
    fig = plt.figure(figsize=(w / 72, h / 72), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, label, size=8.2, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, label, fontsize=size, color=color, weight=weight, ha=ha, va=va, linespacing=1.13, **kw)


def line(ax, x1, y1, x2, y2, color=INK, lw=0.65, ls="-", **kw):
    ax.add_line(Line2D([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls, solid_capstyle="butt", **kw))


def rect(ax, x, y, w, h, fill=WHITE, edge=INK, lw=0.6, **kw):
    p = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=lw, **kw)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=0.7, scale=5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=scale, linewidth=lw,
                                 color=color, shrinkA=0, shrinkB=0))


def bracket(ax, x1, x2, y, depth=3, color=GRAY, lw=0.65):
    line(ax, x1, y + depth, x1, y, color, lw)
    line(ax, x1, y, x2, y, color, lw)
    line(ax, x2, y, x2, y + depth, color, lw)


def pointer(ax, x, y, color=INK, height=5):
    ax.add_patch(Polygon([(x, y), (x - 2.6, y + height), (x + 2.6, y + height)], closed=True,
                         facecolor=color, edgecolor="none"))


def save(fig, ax, stem, out_dir, w, h):
    """PDF next to the paper; SVG (editable text) and PNG (preview) in src/."""
    check_text_inside(ax, w, h)
    out = Path(out_dir)
    (out / "src").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", metadata={"Title": stem, "Creator": "COHORT figure source"})
    fig.savefig(out / "src" / f"{stem}.svg")
    fig.savefig(out / "src" / f"{stem}.png", dpi=300)
    plt.close(fig)
    print("saved", f"{stem}.pdf")


# --------------------------------------------------------------------------- #
# Fig. 1: one retrieved set, two compatible populations (fixed-denominator toy, N=100)
# --------------------------------------------------------------------------- #
def population_bar(ax, x, y, w, unseen_positive):
    """All three segments use exact fractions of the same N=100 scale: 5 positive, 7 negative, 88 unseen."""
    h = 11.0
    rect(ax, x, y, w * .05, h, INK, "none", 0)
    rect(ax, x + w * .05, y, w * .07, h, WHITE, "none", 0)
    rect(ax, x + w * .12, y, w * .88, h, INK if unseen_positive else WHITE, "none", 0)
    rect(ax, x, y, w, h, "none", INK, .65)
    line(ax, x + w * .05, y, x + w * .05, y + h, INK, .45)               # positive | negative
    line(ax, x + w * .12, y - 2.3, x + w * .12, y + h + 2.3, GRAY, .6)   # retrieved | unseen


def draw_ambiguity(out_dir):
    N, K, A = 100, 12, 5
    assert A / N == .05 and (A + N - K) / N == .93
    W, H = 252, 213
    fig, ax = canvas(W, H)
    text(ax, 10, 10, "One retrieval, two possible populations", 10.0, weight="bold")
    text(ax, 10, 25, r"Population $N=100$  ·  retrieved $k=12$", 8.6, GRAY)

    cell, gap = 15.0, 4.0                       # one shared observed set: five positive, seven negative records
    start = (252 - (12 * cell + 11 * gap)) / 2
    for i in range(12):
        rect(ax, start + i * (cell + gap), 38, cell, 13, INK if i < 5 else WHITE, INK, .65)
    text(ax, start + (5 * cell + 4 * gap) / 2, 61, r"$a=5$ positive", 8.2, ha="center")
    xneg = start + 5 * (cell + gap)
    text(ax, xneg + (7 * cell + 6 * gap) / 2, 61, "7 negative", 8.2, ha="center")

    text(ax, 126, 79, "88 labels remain unseen", 8.3, GRAY, ha="center")
    line(ax, 126, 87, 126, 92, MID, .65)        # a fork: compatible completions, not independent samples
    line(ax, 63, 92, 189, 92, MID, .65)
    arrow(ax, 63, 92, 63, 102, MID, .65, 5)
    arrow(ax, 189, 92, 189, 102, MID, .65, 5)

    text(ax, 63, 112, "All unseen negative", 8.4, ha="center")
    text(ax, 189, 112, "All unseen positive", 8.4, ha="center")
    text(ax, 63, 135, "5%", 20, weight="bold", ha="center")
    text(ax, 189, 135, "93%", 20, weight="bold", ha="center")
    population_bar(ax, 10, 153, 106, False)
    population_bar(ax, 136, 153, 106, True)
    text(ax, 126, 176, "Each bar: 12 retrieved + 88 unseen", 8.0, GRAY, ha="center")
    line(ax, 10, 187, 242, 187, RULE, .65)
    text(ax, 10, 199, "Deterministic bounds", 8.2, weight="bold")
    text(ax, 242, 199, r"$\dfrac{a}{N}\;\leq\;p\;\leq\;\dfrac{a+N-k}{N}$", 10.5, ha="right")
    save(fig, ax, "fig_ambiguity", out_dir, W, H)


# --------------------------------------------------------------------------- #
# Fig. 2: selection-safe evidence reuse
# --------------------------------------------------------------------------- #
def record(ax, x, y, reached=False, cached=False, w=10, h=22):
    """One record of the permutation.  Fill = already in the random prefix S; dot = its label was cached before the
    pointer got there (it says nothing about positive or negative)."""
    rect(ax, x, y, w, h, TINT if reached else WHITE, ACCENT if reached else MID, .65)
    if cached:
        ax.add_patch(Circle((x + w / 2, y + h / 2), 1.75, facecolor=INK, edgecolor="none"))


def load_intervals():
    p = _RES / "reuse_intervals.json"
    if p.exists():
        return json.load(open(p))
    # the displayed values of the paper, used only if the result file is missing
    return {"truth": 0.110, "discard": [0.01, 0.37], "pooled": [0.16, 0.27], "cohort": [0.04, 0.33]}


def draw_reuse(out_dir, iv=None):
    iv = iv or load_intervals()
    truth = iv["truth"]
    inside = lambda r: r[0] <= truth <= r[1]
    # the caption says: discarding and COHORT contain the full-run value, pooling as a sample misses it
    assert inside(iv["discard"]) and inside(iv["cohort"]) and inside(iv.get("free", iv["cohort"])) and not inside(iv["pooled"]), iv
    assert iv["cohort"][1] - iv["cohort"][0] < iv["discard"][1] - iv["discard"][0], iv

    W, H = 516, 222
    fig, ax = canvas(W, H)
    text(ax, 9, 10, "One stratum within a stage", 10.0, weight="bold")
    line(ax, 330, 8, 330, 191, RULE, .65)
    text(ax, 342, 10, "Equal new-call budget", 9.8, weight="bold")
    text(ax, 342, 25, r"$2\%$ of $N$  ·  retrieved history", 8.2, GRAY)

    # known mass E sits outside the random permutation
    text(ax, 9, 41, r"Known mass $E$", 8.5, weight="bold")
    rect(ax, 9, 53, 61, 24, LIGHT, MID, .65)
    text(ax, 39.5, 65, "history", 8.5, ha="center")
    text(ax, 9, 89, "Exact counts", 8.0)
    text(ax, 9, 101, "Frozen in stage", 7.8, GRAY)

    # the permutation of the other records, the random prefix S, and the pointer
    qx, qw, step, qy = 94, 10, 12.5, 53
    n, prefix = 18, 9
    end = qx + (n - 1) * step + qw
    edge = qx + prefix * step - (step - qw) / 2
    text(ax, qx, 25, r"Uniform random order outside $E$", 8.5, GRAY)
    bracket(ax, qx, edge - 3, 45, 3, ACCENT)
    bracket(ax, edge + 3, end, 45, 3, GRAY)
    text(ax, (qx + edge - 3) / 2, 38, r"Random prefix $S$", 8.4, ACCENT, weight="bold", ha="center")
    text(ax, (edge + 3 + end) / 2, 38, "Not yet reached", 8.3, GRAY, ha="center")
    # a dot = label cached before its own pointer visit; the next record is already cached
    cached_indices = {2, 6, 9, 14}
    for i in range(n):
        record(ax, qx + i * step, qy, i < prefix, i in cached_indices, qw, 24)
    pointer(ax, edge, 81)
    text(ax, edge, 95, "pointer", 8.0, ha="center")
    ax.add_patch(Circle((qx + 2.8, 108), 1.75, facecolor=INK, edgecolor="none"))
    text(ax, qx + 10, 108, "Cached before pointer visit (during stage)", 7.7)
    tx = qx + 14 * step + qw / 2
    line(ax, tx, 78.5, tx, 85, MID, .6)
    text(ax, tx, 94, r"Targeted exposure $T$", 7.7, GRAY, ha="center")

    # the pointer rule on a three-record excerpt
    line(ax, 9, 120, 319, 120, RULE, .65)
    text(ax, 9, 131, "A cache hit still advances the prefix", 8.8, weight="bold")
    xbefore, xafter, sy = 12, 91, 156
    text(ax, 30, 146, "Before", 7.7, GRAY, ha="center")
    text(ax, 109, 146, "After", 7.7, GRAY, ha="center")
    for i in range(3):
        record(ax, xbefore + i * 12, sy, i < 2, i == 2, 9.5, 17)
        record(ax, xafter + i * 12, sy, True, i == 2, 9.5, 17)
    pointer(ax, xbefore + 24 - 1.25, sy + 20, height=4)
    pointer(ax, xafter + 36 - 1.25, sy + 20, height=4)
    arrow(ax, 57, 164.5, 79, 164.5, GRAY, .8, 5.5)
    text(ax, 150, 155, "Prefix +1", 9.0, ACCENT, weight="bold")
    text(ax, 235, 155, "New calls 0", 9.0, ACCENT, weight="bold")
    text(ax, 150, 173, "No second exposure count", 8.2, GRAY)
    text(ax, 9, 192, r"Only $S$ feeds confidence sequences.", 8.2, GRAY)

    # three ways to use the same history, at the same new-call budget (one real session)
    px0, px1, vmax = 420, 507, .4
    X = lambda v: px0 + (px1 - px0) * v / vmax
    line(ax, X(truth), 46, X(truth), 176, GRAY, .75, (0, (3, 2.2)))
    text(ax, X(truth), 36, rf"$\theta_{{\mathrm{{run}}}}={truth:.3f}$", 8.2, ha="center")
    items = [("Discard\nhistory", iv["discard"], 56, MID, GRAY, 1.7),
             ("Free\nreads", iv["free"], 88, MID, GRAY, 1.7),
             ("Pool as\na sample", iv["pooled"], 120, INK, INK, 1.7),
             ("COHORT\nknown mass", iv["cohort"], 152, ACCENT, ACCENT, 2.4)]
    for label, (lo, hi), y, color, tcolor, width in items:
        text(ax, 411, y, label, 8.2, tcolor, "bold" if label.startswith("COHORT") else "normal", ha="right")
        line(ax, X(lo), y, X(hi), y, color, width)
        for v in (lo, hi):
            line(ax, X(v), y - 3.0, X(v), y + 3.0, color, .85)
        text(ax, (X(lo) + X(hi)) / 2, y - 10.5, f"[{lo:.2f}, {hi:.2f}]", 8.0, tcolor, ha="center",
             bbox=dict(facecolor=WHITE, edgecolor="none", pad=.65))
    text(ax, 507, 132, r"Misses $\theta_{\mathrm{run}}$", 7.4, GRAY, ha="right")
    line(ax, px0, 178, px1, 178, INK, .65)
    for v in (0, .2, .4):
        line(ax, X(v), 178, X(v), 181, INK, .65)
        text(ax, X(v), 190, f"{v:.1f}", 7.5, GRAY, ha="center")
    text(ax, (px0 + px1) / 2, 204, "Exact | Relevant", 8.0, ha="center")
    save(fig, ax, "fig_reuse", out_dir, W, H)


if __name__ == "__main__":
    out = Path(os.environ.get("FIG_OUT", _HERE / "_figout"))
    out.mkdir(parents=True, exist_ok=True)
    draw_ambiguity(out)
    draw_reuse(out)
