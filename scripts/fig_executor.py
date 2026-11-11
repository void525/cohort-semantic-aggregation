# -*- coding: utf-8 -*-
"""Fig. 3 of the paper: one stage of COHORT, drawn deterministically in points at the final two-column width.

  fig_stage : two columns (516 x 276 pt)  ledger -> constraints -> region -> projection, with the
                                          scheduler / pointer / cache / semantic-judge feedback loop

Design basis: the figure package supplied with the submission (COHORT_RAG_Figure1_Figure2_Package, drawn from the
earlier draft).  Geometry and typography are kept; notation and wording follow the current paper, and the visual
code follows Fig. 2 (scripts/fig_family.py): known mass E = grey box, random prefix S = teal, targeted exposure
T = a dot, region and certified output = teal, everything else grey.  No data are plotted.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

from fig_style import ACCENT, GRAY, INK, LIGHT, MID, RULE, TINT, WHITE

_HERE = Path(__file__).resolve().parent
MUTED, PALE = GRAY, LIGHT
S_ROW = "#EEF5F4"                      # a whisper of teal behind the random-prefix row of the ledger


class Drawing:
    def __init__(self, width, height, stem, out_dir):
        self.width, self.height, self.stem = width, height, stem
        self.out = Path(out_dir)
        self.fig = plt.figure(figsize=(width / 72, height / 72), dpi=144)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.texts = []

    def text(self, x, y, s, size=8.0, weight="normal", color=INK, ha="left", va="center", **kw):
        t = self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
                         linespacing=1.18, zorder=6, **kw)
        self.texts.append(t)
        return t

    def line(self, points, color=INK, lw=0.7, dash=None, z=3):
        xs, ys = zip(*points)
        ln, = self.ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="butt", solid_joinstyle="miter", zorder=z)
        if dash:
            ln.set_dashes(dash)
        return ln

    def arrow(self, points, color=INK, lw=0.8, dash=None, head=4.3):
        if len(points) > 2:
            self.line(points[:-1], color=color, lw=lw, dash=dash)
        ar = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=head * 1.6, linewidth=lw,
                             color=color, shrinkA=0, shrinkB=0, capstyle="butt", joinstyle="miter", zorder=4)
        if dash:
            ar.set_linestyle((0, dash))
        self.ax.add_patch(ar)
        return ar

    def rect(self, x, y, w, h, face=WHITE, edge=RULE, lw=0.65, hatch=None, z=2):
        r = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw, hatch=hatch,
                      joinstyle="miter", zorder=z)
        self.ax.add_patch(r)
        return r

    def dot(self, x, y, color=INK, r=1.25):
        self.ax.plot([x], [y], "o", markersize=2 * r, color=color, markeredgewidth=0, zorder=5)

    def save(self, formats=("pdf", "png", "svg")):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for t in self.texts:                         # safety net: nothing may leave the canvas
            bb = t.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            x0, x1 = sorted((bb.x0, bb.x1))
            y0, y1 = sorted((bb.y0, bb.y1))
            if x0 < 1 or x1 > self.width - 1 or y0 < 1 or y1 > self.height - 1:
                raise ValueError(f"text outside the canvas: {t.get_text()!r} {bb}")
        (self.out / "src").mkdir(parents=True, exist_ok=True)
        for ext in formats:
            target = self.out / f"{self.stem}.pdf" if ext == "pdf" else self.out / "src" / f"{self.stem}.{ext}"
            meta = {"Creator": "COHORT figure source"} if ext == "pdf" else None
            self.fig.savefig(target, dpi=300, transparent=False, metadata=meta)
        plt.close(self.fig)


# --------------------------------------------------------------------------- #
# Fig. 3: one stage of COHORT
# --------------------------------------------------------------------------- #
def class_swatch(d, kind, x, y, w=14, h=9):
    """The visual code of Fig. 2 for the three provenance classes."""
    if kind == "E":                                  # known mass: a plain grey box, outside the permutation
        d.rect(x, y, w, h, face=LIGHT, edge=MID, lw=0.8, z=3)
    elif kind == "S":                                # random prefix: teal
        d.rect(x, y, w, h, face=TINT, edge=ACCENT, lw=0.8, z=3)
    else:                                            # targeted exposure: a record with a cached label (dot)
        d.rect(x, y, w, h, face=WHITE, edge=MID, lw=0.8, z=3)
        d.dot(x + w / 2, y + h / 2, color=INK, r=1.25)


def draw_executor(out_dir):
    d = Drawing(516, 276, "fig_stage", out_dir)

    # ---- execution contract ------------------------------------------------------------------
    d.text(8, 10, r"Declared closure $\mathcal{Q}_e$:", size=8.4, weight="bold")
    d.text(8, 23, r"Active $\mathcal{A}_t \subseteq \mathcal{Q}_e$ (e.g., Q1: Exact | Rel)", size=7.5, color=MUTED)
    d.arrow([(160, 17), (174, 17)], lw=0.65, head=3.6)
    d.text(180, 10, "Compile basis once", size=8.4, weight="bold")
    d.text(180, 23, "2 events × 6 strata = 12 CSs", size=7.6, color=MUTED)
    d.line([(298, 4), (298, 30)], color=RULE, lw=0.55)
    d.text(308, 10, r"Frozen pool $N=31{,}482$  ·  judge $\phi$", size=8.0)
    d.text(308, 23, r"Version key $\kappa$  ·  stage risk $\alpha_e$", size=8.0, color=MUTED)
    d.line([(8, 36), (508, 36)], color=RULE, lw=0.65)

    # ---- evidence ledger: three provenance classes, one visual code ---------------------------
    d.text(8, 49, "Evidence ledger", size=9.0, weight="bold")
    d.rect(8, 62, 120, 100, face=WHITE, edge=RULE, lw=0.65)
    rows = (("E", "Known mass", "history of any origin", 62),
            ("S", "Random prefix", "uniform permutation", 90),
            ("T", "Targeted exposure", "retrieval, inspection", 118))
    for kind, title, sub, y0 in rows:
        if kind == "S":
            d.rect(8, y0, 120, 28, face=S_ROW, edge="none", lw=0)
            d.line([(8, y0), (8, y0 + 28)], color=ACCENT, lw=1.7)
        class_swatch(d, kind, 15, y0 + 6)
        d.text(35, y0 + 9, title, size=8.1, weight="bold", color=ACCENT if kind == "S" else INK)
        d.text(35, y0 + 20, sub, size=7.1, color=MUTED)
        d.text(121, y0 + 9, kind, size=8.6, color=ACCENT if kind == "S" else MUTED, ha="right", style="italic")
        d.line([(8, y0 + 28), (128, y0 + 28)], color=RULE, lw=0.5)
    d.text(68, 154, "every labelled record counted once", size=7.0, color=MUTED, ha="center")

    # ---- constraints: only S speaks statistically; every labelled record bounds the counts ----
    d.text(148, 49, "Admissible constraints", size=9.0, weight="bold")
    d.rect(148, 62, 100, 39, face=WHITE, edge=ACCENT, lw=0.9)
    d.text(198, 72, "Confidence sequences", size=8.0, weight="bold", color=ACCENT, ha="center")
    d.text(198, 84, "random prefix S only", size=7.6, ha="center")
    d.text(198, 95, "+ exact offset from E", size=7.1, color=MUTED, ha="center")
    d.arrow([(128, 104), (138, 104), (138, 82), (148, 82)], color=ACCENT, lw=0.9, head=3.8)

    d.rect(148, 117, 100, 45, face=WHITE, edge=RULE, lw=0.65)
    d.text(198, 128, "Count bounds", size=8.1, weight="bold", ha="center")
    d.text(198, 140, "every labelled record", size=7.6, ha="center")
    d.text(198, 152, "unseen mass kept", size=7.6, color=MUTED, ha="center")
    d.arrow([(128, 154), (138, 154), (138, 140), (148, 140)], lw=0.75, head=3.8)

    # ---- shared region (a schematic set, not a data plot) ------------------------------------
    d.text(322, 49, "Shared certified region", size=9.0, weight="bold", ha="center")
    poly = [(281, 93), (304, 74), (347, 82), (362, 108), (342, 137), (296, 134)]
    d.ax.add_patch(Polygon(poly, closed=True, facecolor=TINT, edgecolor=ACCENT, linewidth=1.0,
                           joinstyle="miter", zorder=2))
    d.text(321, 103, r"$\mathcal{P}_t$", size=16.5, color=ACCENT, ha="center")
    d.arrow([(248, 82), (265, 82), (281, 95)], color=ACCENT, lw=0.9, head=3.9)
    d.arrow([(248, 140), (265, 140), (291, 128)], lw=0.75, head=3.9)
    d.text(322, 153, r"$\Sigma_z\, k_{h,z} = N_h$", size=9.2, ha="center")
    d.text(322, 169, "stratum sizes add up", size=7.4, color=MUTED, ha="center")

    # ---- projection: the region, not the LLM, feeds the projector ----------------------------
    d.text(450, 49, r"Project active $\mathcal{A}_t$", size=9.0, weight="bold", ha="center")
    d.rect(392, 60, 116, 80, face=WHITE, edge=RULE, lw=0.65)
    d.text(450, 71, "Denominator guard", size=8.1, weight="bold", ha="center")
    d.text(450, 88, r"$B(\mathbf{k})>0$ on $\mathcal{P}_t$", size=8.8, ha="center")
    d.line([(402, 100), (498, 100)], color=RULE, lw=0.45)
    d.text(450, 110, "Exact projections", size=8.1, weight="bold", ha="center")
    d.text(450, 122, "partition / Charnes–Cooper", size=7.3, ha="center")
    d.text(450, 133, "outward-rounded endpoints", size=7.2, color=MUTED, ha="center")
    d.arrow([(362, 108), (392, 108)], color=ACCENT, lw=0.9, head=4)

    d.arrow([(450, 140), (450, 150)], color=ACCENT, lw=0.8, head=3.8)
    d.rect(392, 150, 116, 36, face=TINT, edge=ACCENT, lw=0.7)
    d.text(450, 160, "Certified answers", size=8.8, weight="bold", color=ACCENT, ha="center")
    d.text(450, 171, "intervals, verdicts, diffs", size=7.6, ha="center")
    d.text(450, 181, r"zero new risk on follow-ups", size=7.2, color=MUTED, ha="center")

    # ---- feedback loop: undecided claims -> scheduler -> pointer -> cache hit or paid miss ----
    d.arrow([(508, 124), (513, 124), (513, 230), (455, 230)], color=MUTED, lw=0.7, dash=(2.5, 2.4), head=3.7)
    d.text(506, 199, r"Undecided in $\mathcal{A}_t$", size=7.6, color=MUTED, ha="right")

    d.rect(347, 211, 108, 39, face=WHITE, edge=RULE, lw=0.65)
    d.text(401, 221, "Scheduler", size=8.6, weight="bold", ha="center")
    d.text(401, 233, "certificate-guided greedy", size=7.2, ha="center")
    d.text(401, 244, "which stratum next?", size=7.2, color=MUTED, ha="center")

    d.rect(235, 211, 86, 39, face=WHITE, edge=RULE, lw=0.65)
    d.text(278, 222, "Pointer", size=8.3, weight="bold", ha="center")
    d.text(278, 237, r"next record of $\pi_h$", size=7.6, color=MUTED, ha="center")
    d.arrow([(347, 230), (321, 230)], color=MUTED, lw=0.75, head=3.8)

    d.rect(129, 211, 82, 39, face=WHITE, edge=RULE, lw=0.65)
    d.text(170, 222, r"Label cache $\kappa$", size=8.3, weight="bold", ha="center")
    d.text(170, 237, "look up the record", size=7.6, color=MUTED, ha="center")
    d.arrow([(235, 230), (211, 230)], color=MUTED, lw=0.75, head=3.8)

    d.rect(8, 211, 96, 39, face=LIGHT, edge=INK, lw=1.1)
    d.text(56, 222, r"Semantic judge $\phi$", size=8.3, weight="bold", ha="center")
    d.text(56, 237, "the only LLM calls", size=7.6, color=MUTED, ha="center")
    d.arrow([(129, 230), (104, 230)], color=MUTED, lw=0.75, head=3.8)
    d.text(117, 221, "miss", size=7.3, color=MUTED, ha="center")

    d.line([(56, 211), (56, 191), (62, 191)], color=ACCENT, lw=0.8)         # paid label and free read merge
    d.line([(170, 211), (170, 191), (62, 191)], color=ACCENT, lw=0.8)
    d.dot(62, 191, color=ACCENT, r=1.1)
    d.arrow([(62, 191), (62, 164)], color=ACCENT, lw=0.8, head=3.8)
    d.text(177, 200, "hit: free read", size=7.4, color=MUTED)
    d.text(116, 179, r"extend the prefix $S$", size=7.5, color=MUTED, ha="center")

    d.line([(8, 259), (508, 259)], color=RULE, lw=0.55)
    d.text(258, 268, r"Claims in the closure share one stage risk $\alpha_e$; certificates concern the operator's "
                     r"full-run answer $\theta_{\mathrm{run}}$.", size=7.8, ha="center")
    d.save()


if __name__ == "__main__":
    out = Path(os.environ.get("FIG_OUT", _HERE / "_figout"))
    out.mkdir(parents=True, exist_ok=True)
    draw_executor(out)
    print("wrote", out)
