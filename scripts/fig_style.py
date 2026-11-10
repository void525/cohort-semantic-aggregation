# -*- coding: utf-8 -*-
"""One look for every figure of the paper: shipped fonts, a restrained palette, base rcParams.

Colour keeps one meaning in all nine figures:
  * teal (ACCENT, TINT)   what COHORT trusts and returns: the random prefix S, the shared region, the certified
                          claims, and the COHORT curves of the plots; TEAL_MID marks close variants of it
  * vermillion (BAD)      an unsafe or wrong outcome
  * greys                 everything else: known mass E, baselines, scaffolding
Fonts: Liberation Sans (SIL OFL, shipped in scripts/fonts) with a DejaVu Sans fallback; maths in STIX.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_HERE = Path(__file__).resolve().parent


def register_fonts():
    """Make the shipped Liberation Sans available to matplotlib (no system installation needed)."""
    for ttf in sorted((_HERE / "fonts").glob("LiberationSans-*.ttf")):
        fm.fontManager.addfont(str(ttf))


register_fonts()

INK, GRAY, MID, RULE, LIGHT, WHITE = "#202629", "#5F696D", "#879194", "#C5CDCE", "#F2F4F4", "#FFFFFF"
ACCENT, TINT, TEAL_MID = "#286365", "#DCEAE9", "#5FA29B"
BAD = "#D55E00"
TEAL_CMAP = LinearSegmentedColormap.from_list("cohort_teal", [WHITE, TINT, "#8DBBB6", "#3F8A84", ACCENT])

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "cohort-figures",
    "savefig.facecolor": WHITE,
    "axes.unicode_minus": False,
})


def check_text_inside(ax, width, height, margin=0.6):
    """Safety net for the hand-placed schematics: no text may touch or leave the canvas (data units = points)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        x0, x1 = sorted((bb.x0, bb.x1))
        y0, y1 = sorted((bb.y0, bb.y1))
        if x0 < margin or x1 > width - margin or y0 < margin or y1 > height - margin:
            raise ValueError(f"text outside the canvas: {t.get_text()!r} x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]"
                             f" canvas {width}x{height}")
