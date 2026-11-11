# -*- coding: utf-8 -*-
"""Publication-grade systems figures & LaTeX table generator (ICDE Best-Paper Style).

Design principles:
  1. Clean, understated noun-phrase headers; zero robotic "(a)/(b)/(c)" tags or promotional slogans.
  2. Consistent color grammar across all figures:
     - COHORT (Shared + E0): deep teal (ACCENT)
     - E-aware Dedicated Streams: navy blue (C_DEDUP)
     - Free-read Shared State (T): medium teal (TEAL_MID)
     - Claim-by-claim / Evergreen-style: muted purple (C_EG)
     - Invalid / biased pooling: crimson-rust (BAD)
  3. Zero text-text or text-line collisions; automatic canvas bounds check on custom schematics.
"""
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fig_style import ACCENT, BAD, GRAY, INK, LIGHT, MID, RULE, TEAL_CMAP, TEAL_MID, TINT, WHITE, check_text_inside

ROOT = _HERE.parent
RES = ROOT / "reports" / "icde2027_suite"
FIG = Path(os.environ.get("FIG_OUT", ROOT / "paper" / "figure"))
GEN = Path(os.environ.get("GEN_OUT", ROOT / "paper" / "gen"))
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

C_DEDUP = "#2B4C7E"
C_DEDUP_T = "#7A93B8"
C_DYN_E = "#5C4B8A"
C_EG = "#6A3D9A"
C_POOL = "#B8336A"

plt.rcParams.update({
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.3,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "lines.linewidth": 1.45,
})


def load_json(name):
    with open(RES / name) as f:
        return json.load(f)


def fmt_int(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


# =========================================================================== #
# Fig 1: Compact Motivation — Top-k retrieval leaves population ratio unidentified
# =========================================================================== #
def draw_fig1_motivation():
    W, H = 252, 134
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    ax.text(8, 10, r"Top-$k$ retrieved set ($k=12$ of $N=100$ records)",
            fontsize=8.3, weight="bold", color=INK, va="center")

    cell, gap = 14.5, 3.6
    start = (W - (12 * cell + 11 * gap)) / 2
    for i in range(12):
        ax.add_patch(Rectangle(
            (start + i * (cell + gap), 21), cell, 10.5,
            facecolor=(ACCENT if i < 5 else WHITE),
            edgecolor=(ACCENT if i < 5 else INK), lw=0.65
        ))
    ax.text(start + (5 * cell + 4 * gap) / 2, 39, r"$a=5$ positive",
            fontsize=7.4, color=ACCENT, weight="bold", ha="center", va="center")
    xneg = start + 5 * (cell + gap)
    ax.text(xneg + (7 * cell + 6 * gap) / 2, 39, "7 negative",
            fontsize=7.4, color=GRAY, ha="center", va="center")

    ax.text(126, 52, "88 unseen records outside top-k",
            fontsize=7.4, color=GRAY, ha="center", va="center")
    ax.add_line(Line2D([126, 126], [58, 63], color=MID, lw=0.65))
    ax.add_line(Line2D([64, 188], [63, 63], color=MID, lw=0.65))
    ax.add_patch(FancyArrowPatch((64, 63), (64, 71), arrowstyle="-|>", mutation_scale=4.3, lw=0.65, color=MID))
    ax.add_patch(FancyArrowPatch((188, 63), (188, 71), arrowstyle="-|>", mutation_scale=4.3, lw=0.65, color=MID))

    ax.text(64, 78, "All unseen negative", fontsize=7.5, color=INK, ha="center", va="center")
    ax.text(188, 78, "All unseen positive", fontsize=7.5, color=INK, ha="center", va="center")
    ax.text(64, 94, "5%", fontsize=14.5, weight="bold", color=INK, ha="center", va="center")
    ax.text(188, 94, "93%", fontsize=14.5, weight="bold", color=ACCENT, ha="center", va="center")

    for bx, all_pos in ((12, False), (136, True)):
        bw, bh, by = 104, 8.0, 106
        ax.add_patch(Rectangle((bx, by), bw * 0.05, bh, facecolor=ACCENT, edgecolor="none"))
        ax.add_patch(Rectangle((bx + bw * 0.05, by), bw * 0.07, bh, facecolor=WHITE, edgecolor="none"))
        ax.add_patch(Rectangle((bx + bw * 0.12, by), bw * 0.88, bh,
                               facecolor=(TINT if all_pos else LIGHT), edgecolor="none"))
        if all_pos:
            ax.add_patch(Rectangle((bx + bw * 0.12, by), bw * 0.88, bh,
                                   facecolor=ACCENT, alpha=0.72, edgecolor="none"))
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="none", edgecolor=INK, lw=0.6))
        ax.add_line(Line2D([bx + bw * 0.12, bx + bw * 0.12], [by - 1.8, by + bh + 1.8], color=GRAY, lw=0.6))

    ax.add_line(Line2D([8, 244], [120, 120], color=RULE, lw=0.6))
    ax.text(8, 128, "Unidentified range:", fontsize=7.4, color=GRAY, va="center")
    ax.text(244, 128, r"$a/N \leq \theta \leq (a+N-k)/N$", fontsize=8.0, color=INK, ha="right", va="center")

    check_text_inside(ax, W, H)
    fig.savefig(FIG / "fig_ambiguity.pdf")
    plt.close(fig)
    print("saved fig_ambiguity.pdf")


# =========================================================================== #
# Fig 2: Provenance Ledger, Pointer Rule, and WANDS Q1 Interval Comparison
# =========================================================================== #
def draw_fig2_reuse():
    iv = load_json("reuse_intervals.json")
    truth = iv["truth"]
    W, H = 516, 200
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Left section: Stratum ledger linked explicitly to Q1
    ax.text(8, 11, "Stratum ledger within a stage (answering Q1)", fontsize=8.7, weight="bold", color=INK, va="center")
    ax.text(8, 24, r"Known mass $E$ removed from frame; only uniform prefix $S$ feeds CSs",
            fontsize=7.5, color=GRAY, va="center")

    ax.text(8, 41, r"Known mass $E$", fontsize=8.0, weight="bold", color=INK, va="center")
    ax.add_patch(Rectangle((8, 50), 64, 23, facecolor=LIGHT, edgecolor=MID, lw=0.75))
    ax.text(40, 61.5, "prior history", fontsize=7.5, color=INK, ha="center", va="center")
    ax.text(8, 83, "Exact count offset", fontsize=7.3, color=GRAY, va="center")
    ax.text(8, 94, r"Outside frame $R_h$", fontsize=7.3, color=GRAY, va="center")

    qx, qw, step, qy = 90, 10, 12.2, 50
    n, prefix = 18, 9
    end = qx + (n - 1) * step + qw
    edge = qx + prefix * step - (step - qw) / 2
    ax.text(qx, 36, r"Uniform permutation $\pi_h$ of remainder $R_h = U_h \setminus E_h$",
            fontsize=7.5, color=INK, va="center")

    for x1, x2, col in ((qx, edge - 2.5, ACCENT), (edge + 2.5, end, GRAY)):
        ax.add_line(Line2D([x1, x1], [47, 44], color=col, lw=0.7))
        ax.add_line(Line2D([x1, x2], [44, 44], color=col, lw=0.7))
        ax.add_line(Line2D([x2, x2], [44, 47], color=col, lw=0.7))

    cached_indices = {2, 6, 9, 15}
    for i in range(n):
        reached = i < prefix
        ax.add_patch(Rectangle((qx + i * step, qy), qw, 23,
                               facecolor=(TINT if reached else WHITE),
                               edgecolor=(ACCENT if reached else MID), lw=0.7))
        if i in cached_indices:
            ax.add_patch(Circle((qx + i * step + qw / 2, qy + 11.5), 1.75, facecolor=INK, edgecolor="none"))

    ax.add_patch(Polygon([(edge, 76.5), (edge - 2.7, 82), (edge + 2.7, 82)],
                         closed=True, facecolor=ACCENT, edgecolor="none"))
    ax.text(edge, 89.5, r"prefix $S$ pointer", fontsize=7.4, color=ACCENT, weight="bold", ha="center", va="center")

    tx = qx + 15 * step + qw / 2
    ax.add_line(Line2D([tx, tx], [74.5, 82.5], color=MID, lw=0.6))
    ax.text(tx, 90.5, r"Targeted $T$ (cached)", fontsize=7.3, color=GRAY, ha="center", va="center")

    # Bottom-left: Cache-hit pointer step
    ax.add_line(Line2D([8, 312], [108, 108], color=RULE, lw=0.6))
    ax.text(8, 120, "Zero-cost cache-hit step when pointer reaches T", fontsize=8.1, weight="bold", color=INK, va="center")

    xb, xa, sy = 12, 92, 142
    ax.text(30, 134, "Before", fontsize=7.3, color=GRAY, ha="center", va="center")
    ax.text(110, 134, "After", fontsize=7.3, color=GRAY, ha="center", va="center")
    for i in range(3):
        ax.add_patch(Rectangle((xb + i * 12, sy), 9.5, 17,
                               facecolor=(TINT if i < 2 else WHITE),
                               edgecolor=(ACCENT if i < 2 else MID), lw=0.65))
        ax.add_patch(Rectangle((xa + i * 12, sy), 9.5, 17, facecolor=TINT, edgecolor=ACCENT, lw=0.65))
        if i == 2:
            ax.add_patch(Circle((xb + i * 12 + 4.75, sy + 8.5), 1.7, facecolor=INK))
            ax.add_patch(Circle((xa + i * 12 + 4.75, sy + 8.5), 1.7, facecolor=INK))
    ax.add_patch(FancyArrowPatch((56, 150.5), (80, 150.5), arrowstyle="-|>", mutation_scale=5.2, lw=0.8, color=ACCENT))

    ax.text(146, 144, "Prefix +1   ·   New LLM calls 0", fontsize=8.0, color=ACCENT, weight="bold", va="center")
    ax.text(146, 158, "Deterministic count unchanged (counted once)", fontsize=7.4, color=GRAY, va="center")
    ax.text(8, 184, "Never skipping cached hits preserves exact without-replacement uniformity.",
            fontsize=7.3, color=GRAY, va="center")

    # Vertical divider
    ax.add_line(Line2D([322, 322], [8, 192], color=RULE, lw=0.7))

    # Right section: Interval comparison on WANDS Q1
    ax.text(332, 11, "Resulting bounds on Q1 (Exact | Rel)", fontsize=8.7, weight="bold", color=INK, va="center")
    ax.text(332, 24, "15% BM25 history + 2% new calls (seed 0)", fontsize=7.5, color=GRAY, va="center")

    px0, px1, vmax = 398, 462, 0.40
    X = lambda v: px0 + (px1 - px0) * v / vmax
    ax.add_line(Line2D([X(truth), X(truth)], [43, 165], color=INK, lw=0.8, ls="--"))
    ax.text(X(truth), 35, rf"$\theta_{{\mathrm{{run}}}}={truth:.3f}$", fontsize=7.5, color=INK, ha="center", va="center")

    rows = [
        ("Pool as\nsample", iv["pooled"], 56, BAD, BAD, 2.0),
        ("Discard\nhistory", iv["discard"], 87, MID, GRAY, 1.6),
        ("Free reads\n(in T)", iv["free"], 118, TEAL_MID, TEAL_MID, 1.8),
        ("COHORT\n(known E)", iv["cohort"], 149, ACCENT, ACCENT, 2.4),
    ]
    for lab, (lo, hi), y, col, tcol, lw in rows:
        ax.text(391, y, lab, fontsize=7.4, color=tcol, weight=("bold" if col == ACCENT else "normal"),
                ha="right", va="center", linespacing=1.15)
        ax.add_line(Line2D([X(lo), X(hi)], [y, y], color=col, lw=lw))
        for v in (lo, hi):
            ax.add_line(Line2D([X(v), X(v)], [y - 3.0, y + 3.0], color=col, lw=0.85))
        tag = f"[{lo:.2f}, {hi:.2f}]"
        ax.text(466, y, tag, fontsize=7.3, color=tcol,
                weight=("bold" if col == ACCENT else "normal"), ha="left", va="center")
    ax.text(466, 67, "misses truth", fontsize=6.8, color=BAD, style="italic", ha="left", va="center")

    ax.add_line(Line2D([px0, px1], [171, 171], color=INK, lw=0.65))
    for v in (0.0, 0.2, 0.4):
        ax.add_line(Line2D([X(v), X(v)], [171, 174], color=INK, lw=0.65))
        ax.text(X(v), 181, f"{v:.1f}", fontsize=7.3, color=GRAY, ha="center", va="center")
    ax.text((px0 + px1) / 2, 192, "Certified interval on Q1", fontsize=7.5, color=INK, ha="center", va="center")

    check_text_inside(ax, W, H)
    fig.savefig(FIG / "fig_reuse.pdf")
    plt.close(fig)
    print("saved fig_reuse.pdf")


# =========================================================================== #
# Fig 3: System Architecture
# =========================================================================== #
def draw_fig3_architecture():
    import fig_executor
    fig_executor.draw_executor(str(FIG))
    print("saved fig_stage.pdf")


# =========================================================================== #
# Fig 4: Geometry of Partition-Aware Projection
# =========================================================================== #
def draw_fig4_geometry():
    g = load_json("exp_geometry.json")
    PA, PB = np.array(g["polygon_A"]), np.array(g["polygon_B"])
    fig, ax = plt.subplots(figsize=(3.45, 2.24))

    ax.fill(PB / 1000, PA / 1000, color=ACCENT, alpha=0.18, lw=0, zorder=2)
    ax.plot(np.r_[PB, PB[:1]] / 1000, np.r_[PA, PA[:1]] / 1000, color=ACCENT, lw=1.15, zorder=3)
    bl, bh, al, ah = g["B_lo"] / 1000, g["B_hi"] / 1000, g["A_lo"] / 1000, g["A_hi"] / 1000
    ax.add_patch(Rectangle((bl, al), bh - bl, ah - al, fill=False, ec="#777777", ls="--", lw=0.9, zorder=2))

    ax.set_xlim(13.5, 30.5)
    ax.set_ylim(9.0, 24.5)
    xs = np.linspace(13.5, 30.5, 200)
    for slope, col, ls, lab in [
        (g["L_sep"], "#777777", "--", f"Separate $A, B$ bounds: [{g['L_sep']:.2f}, {g['U_sep']:.2f}]"),
        (g["U_sep"], "#777777", "--", None),
        (g["L_joint"], ACCENT, "-", f"Partition $B=A+C$ / $\\mathcal{{P}}_t$: [{g['L_joint']:.2f}, {g['U_joint']:.2f}]"),
        (g["U_joint"], ACCENT, "-", None),
    ]:
        ys = slope * xs
        mask = (ys >= 9.0) & (ys <= 24.5)
        ax.plot(xs[mask], ys[mask], color=col, ls=ls, lw=1.0, zorder=1, label=lab)

    bx, ay = g["B_true"] / 1000, g["A_true"] / 1000
    ax.scatter([bx], [ay], marker="*", s=68, color=INK, zorder=6, label=r"Full-run count $(B^\star, A^\star)$")

    ax.set_xlabel("Relevant pairs $B$ (thousands)", labelpad=4)
    ax.set_ylabel(r"Relevant & rated $\geq 4$, $A$ (thousands)", labelpad=5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.03), ncol=1, frameon=False,
              fontsize=7.3, handlelength=2.2, labelspacing=0.36, borderaxespad=0.0)

    fig.savefig(FIG / "fig_geometry.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_geometry.pdf")


# =========================================================================== #
# Fig 5: Mechanism Attribution & Per-Target Cold-Start Trade-Off
# =========================================================================== #
def draw_fig5_ladder():
    d = load_json("exp_ladder.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.24), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.38})
    GRID = [640, 1536, 3200, 6272, 15744, 31482]
    GRID_LAB = ["2%", "5%", "10%", "20%", "50%", "100%"]

    styles = [
        ("EG-style claim-by-claim", "Claim-by-claim + cache", C_EG, "--", "s"),
        ("Shared pool, dedup (no polytope)", "Shared prefix (sep. A/B)", C_POOL, "-.", "^"),
        ("+ joint projection (RR)", "Shared state (RR)", TEAL_MID, "-", "o"),
        ("+ certificate-guided scheduler", "COHORT", ACCENT, "-", "D"),
    ]
    for key, lab, col, ls, mk in styles:
        m = []
        for b in GRID:
            r = d[f"{key}@{b}"]
            freq = {int(k): v for k, v in r["cert_qid_freq"].items()}
            m.append(sum(freq.get(q, 0) for q in (1, 2, 3, 4, 5)) / r["n_runs"])
        lw = 1.8 if lab == "COHORT" else 1.35
        axs[0].plot(GRID, m, color=col, ls=ls, marker=mk, ms=4.0, label=lab, lw=lw)

    axs[0].scatter([31482], [5], marker="*", s=68, color=BAD, zorder=5, label="Label all N")
    axs[0].set_xscale("log")
    axs[0].set_xticks(GRID)
    axs[0].set_xticklabels(GRID_LAB)
    axs[0].minorticks_off()
    axs[0].set_xlim(520, 40000)
    axs[0].set_ylim(-0.25, 5.45)
    axs[0].set_xlabel("Call-budget cap (share of N; early stopping active)")
    axs[0].set_ylabel("Certified numeric targets (of 5)")
    axs[0].grid(axis="y", lw=0.35, alpha=0.45)
    axs[0].legend(loc="lower left", bbox_to_anchor=(-0.04, 1.04), ncol=2, frameon=False,
                  fontsize=6.9, handlelength=1.7, columnspacing=0.75, labelspacing=0.30, borderaxespad=0.0)

    labels = ["Rating share", "Exact | Rel (<4)", "Exact | Rel (all)", "Exact | Rel (≥4)", "Cohort diff."]
    q_order = [4, 3, 1, 2, 5]
    sb = load_json("exp_strong_baselines.json")["cold"]
    co_rows = sb["COHORT (CERT)"]["rows"]
    rr_rows = sb["Shared-Prefix+Partition (RR)"]["rows"]
    ds_rows = sb["Dedup-Streams+Cache+Partition"]["rows"]

    med_q = lambda rows, qid: float(statistics.median([r["per_q"][str(qid)] for r in rows]))
    y = np.arange(len(q_order))
    bh = 0.25
    v_ds = [med_q(ds_rows, q) / 1000 for q in q_order]
    v_rr = [med_q(rr_rows, q) / 1000 for q in q_order]
    v_co = [med_q(co_rows, q) / 1000 for q in q_order]

    axs[1].barh(y + bh, v_ds, height=bh, color=C_DEDUP, alpha=0.90, label="Dedup-streams + cache")
    axs[1].barh(y, v_rr, height=bh, color=TEAL_MID, alpha=0.90, label="Shared state (RR)")
    axs[1].barh(y - bh, v_co, height=bh, color=ACCENT, label="COHORT")

    # Direct clean endpoint annotations on Rating share (row 0) and Cohort diff (row 4)
    axs[1].text(v_co[0] + 0.12, y[0] - bh, f"{v_co[0]:.2f}k", va="center", fontsize=6.7, color=ACCENT, weight="bold")
    axs[1].text(v_ds[0] + 0.12, y[0] + bh, f"{v_ds[0]:.2f}k", va="center", fontsize=6.7, color=C_DEDUP)

    axs[1].set_yticks(y)
    axs[1].set_yticklabels(labels, fontsize=7.5)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("Median calls to certify target (thousands, cold start)")
    axs[1].set_xlim(0, 7.3)
    axs[1].grid(axis="x", lw=0.35, alpha=0.45)
    axs[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, frameon=False,
                  fontsize=7.0, handlelength=1.8, columnspacing=0.9, labelspacing=0.30, borderaxespad=0.0)

    fig.savefig(FIG / "fig_ladder.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_ladder.pdf")


# =========================================================================== #
# Fig 6: 8-Round Session Trajectories & Crossover vs. E-Aware Dedicated Streams
# =========================================================================== #
def draw_fig6_multicohort():
    r4 = load_json("exp_review4_suite.json")
    t1 = r4["task1_8rounds"]
    t2 = r4["task2_8rounds"]

    fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.38), gridspec_kw={"wspace": 0.28})

    m_styles = [
        ("Adaptive dedup-streams + T", "Dedup-streams + free reads (T)", C_DEDUP_T, "-.", "v", 1.25, False),
        ("Shared prefix + free reads (T)", "Shared state + free reads (T)", TEAL_MID, "--", "s", 1.30, False),
        ("Adaptive dedup-streams + E0", "Dedup-streams + known mass E0", C_DEDUP, "-", "^", 1.60, True),
        ("Adaptive dedup-streams + dynamic E", "Dedup-streams + dynamic E", C_DYN_E, ":", "o", 1.40, False),
        ("COHORT (known mass E0 + shared prefix)", "COHORT (known mass E0 + shared state)", ACCENT, "-", "D", 2.05, True),
    ]

    panels = [
        (axs[0], t1, "Task 1: Relevance audit (K=2, H=6, 2→24 targets)", 4, "-55% vs. E0"),
        (axs[1], t2, "Task 2: Cascade audit (K=3, H=3, 2→24 targets)", 5, "-26% vs. E0"),
    ]

    for ax, data, panel_title, cross_r, end_tag in panels:
        rounds_x = np.arange(1, 9)
        tcum = data["targets_per_round_cum"]
        for key, lab, col, ls, mk, lw, show_iqr in m_styles:
            mdata = data["methods"][key]
            meds = np.array(mdata["round_cumulative_medians"], dtype=float) / 1000.0
            ax.plot(rounds_x, meds, color=col, ls=ls, marker=mk, ms=3.9, lw=lw, label=lab,
                    zorder=(5 if key.startswith("COHORT") else 3))
            if show_iqr:
                q25 = np.array(mdata["round_cumulative_q25"], dtype=float) / 1000.0
                q75 = np.array(mdata["round_cumulative_q75"], dtype=float) / 1000.0
                ax.fill_between(rounds_x, q25, q75, color=col, alpha=0.11, lw=0, zorder=2)

        # Mark crossover round against Adaptive dedup + E0
        co_meds = np.array(data["methods"]["COHORT (known mass E0 + shared prefix)"]["round_cumulative_medians"]) / 1000.0
        cy = co_meds[cross_r - 1]
        ax.axvline(cross_r, color="#888888", ls=":", lw=0.8, zorder=1)
        if cross_r == 4:
            ax.text(cross_r - 0.12, 19.8, "E0 crossover (R4)", fontsize=6.9, color=INK, ha="right", va="center")
            ax.text(6.1, co_meds[5] - 1.35, "0 new calls (R4–R8)", fontsize=6.9, color=ACCENT, weight="bold", ha="center")
            ax.text(8.05, co_meds[7] + 0.85, end_tag, fontsize=7.0, color=ACCENT, weight="bold", ha="right")
        else:
            ax.text(cross_r - 0.12, 19.6, "E0 crossover (R5)", fontsize=6.9, color=INK, ha="right", va="center")
            ax.text(8.05, co_meds[7] - 1.35, end_tag, fontsize=7.0, color=ACCENT, weight="bold", ha="right")

        ax.set_xticks(rounds_x)
        ax.set_xticklabels([f"R{r}\n({c})" for r, c in zip(rounds_x, tcum)], fontsize=6.9, linespacing=1.05)
        ax.set_xlabel("Session round (cumulative distinct targets)", labelpad=4)
        ax.set_ylabel("Cumulative new calls (thousands)")
        ax.set_xlim(0.75, 8.30)
        ax.grid(axis="y", lw=0.35, alpha=0.45)
        ax.set_title(panel_title, loc="left", weight="bold", fontsize=8.1, pad=6)

    axs[0].set_ylim(0, 22.0)
    axs[1].set_ylim(1.5, 21.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.22),
               fontsize=7.1, handlelength=2.1, columnspacing=1.55, labelspacing=0.36)
    fig.savefig(FIG / "fig_claims.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_claims.pdf")


# =========================================================================== #
# Fig 7: Biased History Safety, Paired Reduction, and Mid-Stage Interleaving
# =========================================================================== #
def draw_fig7_history_and_paired():
    d = load_json("exp_bias20.json")
    paired = load_json("exp_paired.json")
    inter = load_json("exp_interleave.json")

    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.24), gridspec_kw={"wspace": 0.36})

    ARM_STYLE = {
        "discard": ("Discard", "#7f7f7f", ":", "o"),
        "T-reads": ("Free reads (T)", TEAL_MID, "-.", "s"),
        "E (ours)": ("COHORT (E)", ACCENT, "-", "D"),
        "pooled-as-random": ("Pooled (CS)", BAD, "--", "^"),
        "naive": ("Pooled (Wald)", "#7a1f00", (0, (3, 1, 1, 1)), "v"),
    }
    enr = sorted({v["ratio"] for v in d.values() if v["kind"] == "enrich"})
    ax = axs[0]
    for arm, (lab, col, ls, mk) in ARM_STYLE.items():
        xs, ys = [], []
        for r in enr:
            v = d[f"enrich:{r}@0.1"]
            xs.append(v["hit_hist"] / v["hit_pop"])
            ys.append(100 * v["arms"][arm]["dec_err_rate"])
        ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=3.7, lw=1.35, label=lab)
        vb = d["bm25:0@0.1"]
        xb = vb["hit_hist"] / vb["hit_pop"]
        yb = 100 * vb["arms"][arm]["dec_err_rate"]
        ax.scatter([xb], [yb], marker="D", s=26, facecolor="white", edgecolor=col, lw=1.2, zorder=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.minorticks_off()
    ax.set_ylim(-4, 56)
    ax.set_xlabel("Exact-match enrichment in history", labelpad=4)
    ax.set_ylabel("Wrong certified claims (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.04, 1.03), ncol=2, frameon=False,
              fontsize=6.8, handlelength=1.7, columnspacing=0.75, labelspacing=0.28, borderaxespad=0.0)

    # Middle: Paired call saving of Known Mass E over Free Reads T
    ax = axs[1]
    order = [("uniform", "Unif."), ("bm25", "BM25"), ("mild", "Odds ×4*"), ("strong", "Odds ×16")]
    for i, (key, lab) in enumerate(order):
        rows = paired["full"][key]
        both = [r for r in rows if r["E"].get("calls_all") and r["T"].get("calls_all")]
        ys = [100 * (1 - r["E"]["calls_all"] / r["T"]["calls_all"]) for r in both]
        rng_j = np.linspace(-0.09, 0.09, len(ys))
        ax.scatter(i + rng_j, ys, color=ACCENT, s=22, alpha=0.82, zorder=3)
        med = statistics.median(ys)
        ax.plot([i - 0.22, i + 0.22], [med, med], color=INK, lw=1.8, zorder=4)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([x[1] for x in order], fontsize=7.2)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-4, 38)
    ax.grid(axis="y", lw=0.35, alpha=0.45)
    ax.set_xlabel("History regime (*1 empty CS)", labelpad=4)
    ax.set_ylabel("Paired saving: E vs. T (%)")
    ax.set_title("Known mass E vs. free reads T", loc="left", weight="bold", fontsize=8.1, pad=7)

    # Right: Mid-stage interleaved retrieval + prefix
    ax = axs[2]
    cats = ["Free-read\nrestart", "Known-E\nrestart", "Continuous\nCOHORT"]
    meds = [
        inter["median_restart_free_reads"] / 1000.0,
        inter["median_restart_known_mass"] / 1000.0,
        inter["median_continuous"] / 1000.0,
    ]
    cols = [TEAL_MID, C_DEDUP, ACCENT]
    ax.bar(range(3), meds, width=0.44, color=cols)
    for idx, v in enumerate(meds):
        ax.text(idx, v + 0.28, f"{v:.2f}k", ha="center", va="bottom", fontsize=7.2,
                weight=("bold" if idx == 2 else "normal"), color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cats, fontsize=7.0, linespacing=1.18)
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylim(0, 11.4)
    ax.set_xlabel("Mid-stage retrieval protocol", labelpad=4)
    ax.set_ylabel("Total calls for 5 targets (k)")
    ax.set_title("Mid-stage retrieval + prefix", loc="left", weight="bold", fontsize=8.1, pad=7)

    fig.savefig(FIG / "fig_history.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_history.pdf")


# =========================================================================== #
# Fig 8: Four-Round Session Across Compatible Stage Re-Slicing
# =========================================================================== #
def draw_fig8_session():
    d = load_json("exp_session.json")
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.88), gridspec_kw={"height_ratios": [1.30, 0.95], "hspace": 0.56})

    ax = axs[0]
    style = {
        "COHORT (E-inherit)": ("COHORT (E-inherit)", ACCENT, "-", "o"),
        "COHORT-T (restart, free reads)": ("Restart + free reads", TEAL_MID, "--", "s"),
        "EG-style": ("Claim-by-claim", C_EG, "-.", "^"),
    }
    curves = {}
    for m, (lab, col, ls, mk) in style.items():
        runs = d["runs"][m]
        xs = np.array([[e["calls"] for e in r["traj"]] for r in runs], float)
        ys = np.array([[r["traj"][0]["certified"], r["traj"][1]["certified"], r["traj"][2]["certified"],
                        r["traj"][2]["certified"] + r["traj"][3]["certified"]] for r in runs], float)
        curves[m] = (xs, ys)
        ax.errorbar(xs.mean(0) / 1000.0, ys.mean(0), yerr=ys.std(0), color=col, ls=ls, marker=mk,
                    ms=3.9, lw=1.4, capsize=1.5, elinewidth=0.7, label=lab)

    X0, _ = curves["COHORT (E-inherit)"][0].mean(0) / 1000.0, curves["COHORT (E-inherit)"][1].mean(0)
    ax.axvline(X0[2], color="#888888", ls=":", lw=0.85, zorder=0)
    ax.text(X0[2] - 0.28, 31.5, "Stage 1", fontsize=7.0, color="#555555", ha="right", va="center")
    ax.text(X0[2] + 0.28, 6.0, "Stage 2", fontsize=7.0, color="#555555", ha="left", va="center")

    ax.set_xlabel("Cumulative semantic calls (thousands)", fontsize=7.8)
    ax.set_ylabel("Certified claims (of 32)", fontsize=7.8)
    ax.set_xlim(-0.4, 11.8)
    ax.set_ylim(-1, 36.5)
    ax.set_yticks([0, 10, 20, 30])
    ax.grid(axis="y", lw=0.35, alpha=0.45)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, frameon=False,
              fontsize=7.0, handlelength=1.8, columnspacing=1.2, labelspacing=0.28, borderaxespad=0.0)

    ax2 = axs[1]
    methods_r4 = ["Claim-by-claim + cache", "Restart + free reads", "COHORT (E-inherit)"]
    calls_r4 = [3.148, 3.016, 2.794]
    cols_r4 = [C_EG, TEAL_MID, ACCENT]
    certs_r4 = ["8.4/14", "14/14", "14/14"]
    y_pos = np.arange(3)
    ax2.barh(y_pos, calls_r4, height=0.50, color=cols_r4)
    for idx, (v, ct) in enumerate(zip(calls_r4, certs_r4)):
        ax2.text(v + 0.10, idx, f"{v:.2f}k ({ct})", ha="left", va="center", fontsize=7.1,
                 weight=("bold" if idx == 2 else "normal"), color=INK)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods_r4, fontsize=7.4)
    ax2.set_ylim(-0.55, 2.55)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 5.10)
    ax2.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticklabels(["0", "1k", "2k", "3k", "4k", "5k"], fontsize=7.3)
    ax2.set_xlabel("Round 4 new calls after re-slicing (thousands)", fontsize=7.8)

    fig.savefig(FIG / "fig_session.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_session.pdf")


# =========================================================================== #
# Fig 9: Rare Denominators Heatmap
# =========================================================================== #
def draw_fig9_denom():
    d = load_json("exp_denom.json")
    pops = sorted(d["populations"], key=lambda r: r["selectivity"])
    budgets = ["0.05", "0.1", "0.2", "0.5", "1.0"]
    M = np.array([[p["budgets"][b]["cert_rate"] for b in budgets] for p in pops])
    fig, ax = plt.subplots(figsize=(3.45, 2.24))
    im = ax.imshow(M, cmap=TEAL_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(["5%", "10%", "20%", "50%", "100%"], fontsize=7.5)
    ax.set_yticks(range(len(pops)))
    ax.set_yticklabels([f"{100 * p['selectivity']:.0f}% ($\\theta$={100 * p['theta']:.0f}%)" for p in pops], fontsize=7.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}" if 0 < v < 1 else ("1" if v >= 1 else "0"), ha="center", va="center",
                    fontsize=7.1, color=("white" if v > 0.6 else INK))
    ax.set_xlabel("Call-budget cap (share of N)", labelpad=6)
    ax.set_ylabel("Denominator rate (true ratio)", labelpad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    cb.ax.tick_params(labelsize=7.2, pad=3)
    cb.set_label("Share of runs certified", fontsize=7.3, labelpad=5)
    fig.savefig(FIG / "fig_denom.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_denom.pdf")


# =========================================================================== #
# Generate LaTeX tables (Table V, Table VI, Table VII, Table VIII)
# =========================================================================== #
def write_tables():
    r4 = load_json("exp_review4_suite.json")
    hk = load_json("exp_hk_scale.json")["rows"]
    calib = load_json("exp_calib.json")

    # 1. Table V: Multi-cohort session table with E-aware dedicated baselines across R1, R4, R8
    t1 = r4["task1_8rounds"]["methods"]
    t2 = r4["task2_8rounds"]["methods"]
    rows_mc = [
        ("Shared prefix + free reads (T)", "Shared prefix + free reads ($T$)"),
        ("Adaptive dedup-streams + T", "Adaptive dedup-streams + free reads ($T$)"),
        ("Adaptive dedup-streams + E0", "Adaptive dedup-streams + known mass $E_0$"),
        ("Oracle dedup-streams + E0", "Oracle dedup-streams + known mass $E_0$"),
        ("Adaptive dedup-streams + dynamic E", "Adaptive dedup-streams + dynamic $E$"),
        ("COHORT (known mass E0 + shared prefix)", r"\textbf{\textsc{Cohort} (known mass $E_0$ + shared prefix)}"),
    ]
    lines = []
    for k, label in rows_mc:
        m1, m2 = t1[k], t2[k]
        r1_1 = fmt_int(m1["round_cumulative_medians"][0])
        r4_1 = fmt_int(m1["round_cumulative_medians"][3])
        r8_1 = fmt_int(m1["round_cumulative_medians"][7])
        s8_1 = "--" if k.startswith("COHORT") else f"{m1['cohort_paired_saving_median_by_round_pct'][7]:+.1f}\\%"

        r1_2 = fmt_int(m2["round_cumulative_medians"][0])
        r4_2 = fmt_int(m2["round_cumulative_medians"][3])
        r8_2 = fmt_int(m2["round_cumulative_medians"][7])
        s8_2 = "--" if k.startswith("COHORT") else f"{m2['cohort_paired_saving_median_by_round_pct'][7]:+.1f}\\%"

        if k.startswith("COHORT"):
            r8_1 = f"\\textbf{{{r8_1}}}"
            r8_2 = f"\\textbf{{{r8_2}}}"
            r4_1 = f"\\textbf{{{r4_1}}}"
        elif k == "Adaptive dedup-streams + dynamic E":
            r1_1_s = r1_1
            r4_1 = f"\\textbf{{{r4_1}}}"
            r4_2 = f"\\textbf{{{r4_2}}}"
        if k == "Adaptive dedup-streams + E0":
            r1_1 = f"\\textbf{{{r1_1}}}"
            r1_2 = f"\\textbf{{{r1_2}}}"

        lines.append(f"{label} & {r1_1} & {r4_1} & {r8_1} & {s8_1} & {r1_2} & {r4_2} & {r8_2} & {s8_2} \\\\")

    tab_mc = r"""\begin{table*}[t]
  \centering
  \caption{Cumulative new calls at completion of Rounds 1, 4, and 8 (2, 12, and 24 distinct targets; 15\% biased history, $\alpha=5\%$, 10 seeds). Every method shares a global cache, deduplicates ratio expressions, and uses exact partition bounds~\eqref{eq:partition}. ``\textsc{Cohort} saves'' reports $\operatorname{median}_s(1 - C_{\textsc{Cohort},s}/C_{\mathrm{base},s})$ at Round~8 over jointly completed seeds (9/10 where one run halted fail-closed on an empty CS; 10/10 otherwise). Dedicated streams are cheaper in early rounds; \textsc{Cohort} crosses $E_0$-aware dedicated streams at Round~4 (Task~1) and Round~5 (Task~2).}
  \label{tab:multicohort}
  \footnotesize
  \renewcommand{\arraystretch}{1.14}
  \setlength{\tabcolsep}{4.6pt}
  \begin{tabular}{@{}l rrr r rrr r@{}}
    \toprule
    & \multicolumn{4}{c}{\textbf{Task 1: Relevance Audit} ($K{=}2, H{=}6$, 24 targets)} & \multicolumn{4}{c}{\textbf{Task 2: Cascade Audit} ($K{=}3, H{=}3$, 24 targets)} \\
    \cmidrule(lr){2-5}\cmidrule(lr){6-9}
    Execution strategy & R1 (2) & R4 (12) & R8 (24) & \textsc{Cohort} saves & R1 (2) & R4 (12) & R8 (24) & \textsc{Cohort} saves \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
"""
    (GEN / "tab_multicohort.tex").write_text(tab_mc)

    # 2. Table VII: Calibration table with exact session-level unit in row 4
    pw, e2e = calib["pathwise"], calib["e2e"]
    tab_cal = rf"""\begin{{table}}[t]
  \centering
  \caption{{Empirical calibration of Theorem~\ref{{thm:stage}}. Rows 1--3 check any-step region coverage ($k^\star \in \mathcal{{P}}_t$ across all prefixes of 3{{,}}000 full-pool permutations). Row~4 counts end-to-end sessions emitting any false certificate across {e2e['certified_claims']:,} certified claims.}}
  \label{{tab:calib}}
  \footnotesize
  \renewcommand{{\arraystretch}}{{1.13}}
  \setlength{{\tabcolsep}}{{3.6pt}}
  \begin{{tabular}}{{@{{}}> {{\raggedright\arraybackslash}}p{{4.15cm}}ccc@{{}}}}
    \toprule
    Calibration check & Failures & Rate & 95\% UB \\
    \midrule
    Any-step $k^\star\notin\mathcal{{P}}_t$, $\alpha{{=}}1\%$ & {pw['0.01']['failures']} / 3{{,}}000 & {100*pw['0.01']['rate']:.2f}\% & {100*pw['0.01']['cp95_upper']:.2f}\% \\\addlinespace[1.5pt]
    Any-step $k^\star\notin\mathcal{{P}}_t$, $\alpha{{=}}5\%$ & {pw['0.05']['failures']} / 3{{,}}000 & {100*pw['0.05']['rate']:.2f}\% & {100*pw['0.05']['cp95_upper']:.2f}\% \\\addlinespace[1.5pt]
    Any-step $k^\star\notin\mathcal{{P}}_t$, $\alpha{{=}}10\%$ & {pw['0.1']['failures']} / 3{{,}}000 & {100*pw['0.1']['rate']:.2f}\% & {100*pw['0.1']['cp95_upper']:.2f}\% \\\addlinespace[2pt]
    Sessions with any false certificate (2\% cap; {e2e['certified_claims']:,} claims) & {e2e['sessions_with_error']} / {e2e['sessions']} & 0.00\% & {100*e2e['cp95_upper_session']:.2f}\% \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".replace("1,740", "1{,}740")
    (GEN / "tab_calib.tex").write_text(tab_cal)

    # 3. Table VI: Paired E vs T table with explicit aggregation formula
    tab_pair = r"""\begin{table}[t]
  \centering
  \caption{New calls to certify all five numeric targets under matched round-robin scheduling (15\% history, 8 seeds). ``Saving'' is $\operatorname{median}_s(1 - C_{E,s}/C_{T,s})$ across jointly completed seeds; one known-mass run at odds $\times 4$ halted fail-closed on an empty CS.}
  \label{tab:paired}
  \footnotesize
  \renewcommand{\arraystretch}{1.13}
  \setlength{\tabcolsep}{3.8pt}
  \begin{tabular}{@{}lccrrr@{}}
    \toprule
    & \multicolumn{2}{c}{Completed} & \multicolumn{2}{c}{Median calls} & \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}
    History regime & Free & Known & Free ($T$) & Known ($E$) & Saving \\
    \midrule
    Uniform sample & 8/8 & 8/8 & 5{,}504 & 4{,}864 & 10\% \\
    BM25 top-$k$ & 8/8 & 8/8 & 5{,}489 & 4{,}608 & 14\% \\
    Exact odds $\times 4$ & 8/8 & 7/8 & 5{,}488 & 4{,}224 & 26\% \\
    Exact odds $\times 16$ & 8/8 & 8/8 & 4{,}062 & 3{,}072 & 28\% \\
    \bottomrule
  \end{tabular}
\end{table}
"""
    (GEN / "tab_paired.tex").write_text(tab_pair)

    # 4. Table VIII: Structural scaling table
    hk_lines = []
    for r in hk:
        hk_lines.append(
            f"{r['H']} & {r['K']} & {r['lp_vars']} & {r['n_basis']} & {r['n_cs']} & "
            f"{r['q1_width_mean']:.3f} & {r['cpu_s_mean']:.1f} & {r['peak_mb_mean']:.1f} \\\\"
        )
    tab_hk = r"""\begin{table}[t]
  \centering
  \caption{Structural scaling across strata $H$ and semantic predicates $K$ at 3{,}200 calls ($N=31{,}482$, mean over 3 seeds). Splitting $\alpha=5\%$ across more sequences widens intervals. Projection CPU is 2.3\,s at the default $H{=}6,K{=}2$ (vs.\ 50.2\,s GPU-equivalent model time at 63.7 records/s), rising to 107.5\,s at the $H{=}24,K{=}4$ corner; heap stays at 4--6\,MB.}
  \label{tab:hkscale}
  \footnotesize
  \renewcommand{\arraystretch}{1.13}
  \setlength{\tabcolsep}{4.3pt}
  \begin{tabular}{@{}ccccccrr@{}}
    \toprule
    $H$ & $K$ & $H{\cdot}J$ & $|\mathcal{F}|$ & CSs & Q1 width & CPU\,(s) & Heap\,(MB) \\
    \midrule
""" + "\n".join(hk_lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    (GEN / "tab_hkscale.tex").write_text(tab_hk)
    print("wrote tab_multicohort.tex, tab_calib.tex, tab_paired.tex, and tab_hkscale.tex")


def main():
    draw_fig1_motivation()
    draw_fig2_reuse()
    draw_fig3_architecture()
    draw_fig4_geometry()
    draw_fig5_ladder()
    draw_fig6_multicohort()
    draw_fig7_history_and_paired()
    draw_fig8_session()
    draw_fig9_denom()
    write_tables()


if __name__ == "__main__":
    main()
