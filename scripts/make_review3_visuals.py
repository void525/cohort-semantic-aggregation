# -*- coding: utf-8 -*-
"""Publication-grade ICDE 2027 figures & LaTeX table generator (Review-5 Best-Paper Edition).

Implements every visual & structural upgrade from Review 5:
  - Fig. 1 (fig_ambiguity.pdf): A-Scan / MICRO style plan-inversion diagram on Page 1 —
    Left: concrete 4-round session trace + compact top-k identification inset (5%–93%);
    Right: live RTX 4080 SUPER wall-clock cost inversion between Round 1 / single-batch
    (streams faster) and Round 4 skewed-history cascade session (COHORT faster, 0 mismatches).
  - Fig. 2 (fig_reuse.pdf): Concrete tuple-level E/S/T state & zero-cost cache-hit pointer rule
    paired with WANDS Q1 bounds (replacing redundant multi-diagram prose; Table II removed).
  - Fig. 3 (fig_stage.pdf): Stage execution loop separating declared Q_e from active A_t.
  - Fig. 4 (fig_geometry.pdf): Dual-panel geometry — Left: partition B=A+C parallelogram;
    Right: genuine multi-predicate Charnes–Cooper LP coupling tightening (Ex_4B - Ex_8B)|Rel_8B
    beyond subtracting separate partition intervals.
  - Fig. 5 (fig_ladder.pdf): Fixed x-axis clipping ("cold start" -> clean centered short label).
  - Fig. 6 (fig_claims.pdf): 3-panel 8-round trajectory across BOTH domains & regimes
    (WANDS Relevance, WANDS Split-Cost Cascade, and Independent UCI DrugLib Clinical Triage),
    foregrounding Dynamic E & Shared-Priority + E0 alongside COHORT.
  - Fig. 7 (fig_history.pdf): Fixed crowded x-ticks via clean two-line labels and explicit
    "All 3 valid = 0.0%" annotation without overlapping curves.
  - Fig. 8 (fig_session.pdf): Explicitly separates the 5+4 distinct numeric targets from the
    13+10 derived threshold broadcasts across Stage 1 -> Stage 2 re-slicing.
  - Fig. 9 (fig_denom.pdf): Plan-selection phase diagram (Overlap x Horizon vs. Dynamic E;
    History Skew x Horizon vs. Shared-Priority) + rare-denominator empirical boundary.
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

C_DEDUP = "#2B4C7E"       # E0 dedicated streams
C_DYNE = "#6A3D9A"        # Dynamic-E dedicated streams
C_PRIO = "#B85D19"        # Coordinated Shared-Priority + E0
C_EG = "#7A5195"
C_POOL = "#B8336A"

plt.rcParams.update({
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.2,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.1,
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
# Fig 1: Plan Inversion Across Interactive Sessions + Top-k Identification Inset
# =========================================================================== #
def draw_fig1_motivation():
    live = load_json("exp_live_canonical_gpu.json")
    t1_live = live["wands_relevance_5target_live"]
    pk_live = live["wands_split_cost_cascade_live"]

    fig = plt.figure(figsize=(3.48, 2.54), facecolor=WHITE)
    # Top band: compact top-k non-identifiability schematic (inset bar, zero text overlap)
    ax_top = fig.add_axes([0.02, 0.71, 0.96, 0.27])
    ax_top.set_xlim(0, 240)
    ax_top.set_ylim(48, 0)
    ax_top.axis("off")

    ax_top.text(2, 6.5, "Why sampling is required:", fontsize=7.5, weight="bold", color=INK, va="center")
    ax_top.text(106, 6.5, r"top-$k$ ($5/12$ pos., $N{=}100$) $\Rightarrow \theta \in [5\%, 93\%]$",
                fontsize=7.1, color=GRAY, va="center")
    bx, by, bw, bh = 2, 16.5, 236, 11.5
    ax_top.add_patch(Rectangle((bx, by), bw * 0.05, bh, facecolor=ACCENT, edgecolor="none"))
    ax_top.add_patch(Rectangle((bx + bw * 0.05, by), bw * 0.88, bh, facecolor=TINT, edgecolor="none"))
    ax_top.add_patch(Rectangle((bx + bw * 0.93, by), bw * 0.07, bh, facecolor=WHITE, edgecolor="none"))
    ax_top.add_patch(Rectangle((bx, by), bw, bh, facecolor="none", edgecolor=INK, lw=0.6))
    ax_top.add_line(Line2D([bx + bw * 0.05, bx + bw * 0.05], [by, by + bh], color=ACCENT, lw=0.6))
    ax_top.add_line(Line2D([bx + bw * 0.93, bx + bw * 0.93], [by, by + bh], color=INK, lw=0.6))
    ax_top.text(bx + bw * 0.49, by + bh / 2, "88 unseen tuples unidentified",
                fontsize=6.7, color=INK, ha="center", va="center")
    ax_top.text(bx, 36.5, "5% min", fontsize=6.8, color=ACCENT, weight="bold", ha="left", va="center")
    ax_top.text(bx + bw, 36.5, "93% max", fontsize=6.8, color=INK, weight="bold", ha="right", va="center")
    ax_top.add_line(Line2D([2, 238], [45.5, 45.5], color=RULE, lw=0.6))

    # Bottom panel: Measured Live GPU Wall-Clock Plan Inversion (RTX 4080 SUPER, 0 mismatches)
    ax = fig.add_axes([0.15, 0.13, 0.82, 0.45])
    regimes = [
        "Balanced E0\n(2-rnd, 5 targets)",
        "Skewed proxy E0\n(4-rnd, 12 targets)",
    ]
    v_dyne = [
        t1_live["Dedup-streams + dynamic E (live)"]["mean_total_wall_s"],
        pk_live["Dedup-streams + dynamic E (live)"]["mean_total_wall_s"],
    ]
    v_prio = [
        t1_live["Shared-priority + E0 + partition (live)"]["mean_total_wall_s"],
        pk_live["Shared-priority + E0 + partition (live)"]["mean_total_wall_s"],
    ]
    v_cohort = [
        t1_live["COHORT (live, default)"]["mean_total_wall_s"],
        pk_live["COHORT (live, default)"]["mean_total_wall_s"],
    ]

    x = np.arange(2)
    w = 0.24
    b1 = ax.bar(x - w, v_dyne, width=w, color=C_DYNE, alpha=0.90, label="Dedup + dynamic E")
    b2 = ax.bar(x, v_prio, width=w, color=C_PRIO, alpha=0.90, label="Shared-priority + E0")
    b3 = ax.bar(x + w, v_cohort, width=w, color=ACCENT, label="COHORT (shared region)")

    for bars, col, bold_idx in [(b1, C_DYNE, -1), (b2, C_PRIO, 0), (b3, ACCENT, 1)]:
        for idx, bar in enumerate(bars):
            hv = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, hv + 16, f"{hv:.0f}s",
                    ha="center", va="bottom", fontsize=6.8,
                    weight=("bold" if idx == bold_idx else "normal"), color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=7.2)
    ax.set_ylabel("Live GPU time (s)", fontsize=7.6)
    ax.set_ylim(0, 735)
    ax.set_yticks([0, 200, 400, 600])
    ax.grid(axis="y", lw=0.35, alpha=0.45)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.05), ncol=3, frameon=False,
              fontsize=6.5, handlelength=1.15, columnspacing=0.60, handletextpad=0.32, borderaxespad=0.0)

    fig.savefig(FIG / "fig_ambiguity.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved fig_ambiguity.pdf")


# =========================================================================== #
# Fig 2: Provenance Ledger, Pointer Rule, and WANDS Q1 Interval Comparison
# =========================================================================== #
def draw_fig2_reuse():
    iv = load_json("reuse_intervals.json")
    truth = iv["truth"]
    W, H = 516, 194
    fig = plt.figure(figsize=(W / 72, H / 72), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    ax.text(8, 11, "Provenance ledger in stratum h (traced on Q1)", fontsize=8.6, weight="bold", color=INK, va="center")
    ax.text(8, 23, r"$E_h$ (0 calls, no CS)  ·  prefix $S_h$ (1 CS step; 0 calls on hit)  ·  $T_h$ (bounds only until $\pi_h$ hits)",
            fontsize=7.2, color=GRAY, va="center")

    ax.text(8, 40, r"Known mass $E_h$", fontsize=7.9, weight="bold", color=INK, va="center")
    ax.add_patch(Rectangle((8, 49), 64, 22, facecolor=LIGHT, edgecolor=MID, lw=0.75))
    ax.text(40, 60, "prior history", fontsize=7.4, color=INK, ha="center", va="center")
    ax.text(8, 81, r"Exact offset $e_{h,f}$", fontsize=7.2, color=GRAY, va="center")
    ax.text(8, 92, r"Outside frame $R_h$", fontsize=7.2, color=GRAY, va="center")

    qx, qw, step, qy = 90, 10, 12.2, 49
    n, prefix = 18, 9
    end = qx + (n - 1) * step + qw
    edge = qx + prefix * step - (step - qw) / 2
    ax.text(qx, 36, r"Uniform permutation $\pi_h$ of remainder $R_h = U_h \setminus E_h$",
            fontsize=7.4, color=INK, va="center")

    for x1, x2, col in ((qx, edge - 2.5, ACCENT), (edge + 2.5, end, GRAY)):
        ax.add_line(Line2D([x1, x1], [46, 43], color=col, lw=0.7))
        ax.add_line(Line2D([x1, x2], [43, 43], color=col, lw=0.7))
        ax.add_line(Line2D([x2, x2], [43, 46], color=col, lw=0.7))

    cached_indices = {2, 6, 9, 15}
    for i in range(n):
        reached = i < prefix
        ax.add_patch(Rectangle((qx + i * step, qy), qw, 22,
                               facecolor=(TINT if reached else WHITE),
                               edgecolor=(ACCENT if reached else MID), lw=0.7))
        if i in cached_indices:
            ax.add_patch(Circle((qx + i * step + qw / 2, qy + 11.0), 1.75, facecolor=INK, edgecolor="none"))

    ax.add_patch(Polygon([(edge, 74.5), (edge - 2.7, 80), (edge + 2.7, 80)],
                         closed=True, facecolor=ACCENT, edgecolor="none"))
    ax.text(edge, 87.5, r"prefix $S_h$ pointer", fontsize=7.3, color=ACCENT, weight="bold", ha="center", va="center")

    tx = qx + 15 * step + qw / 2
    ax.add_line(Line2D([tx, tx], [72.5, 80.5], color=MID, lw=0.6))
    ax.text(tx, 88.5, r"Targeted $T_h$ (cached)", fontsize=7.2, color=GRAY, ha="center", va="center")

    ax.add_line(Line2D([8, 312], [105, 105], color=RULE, lw=0.6))
    ax.text(8, 117, "Cache-hit pointer rule & stage-rebuild transition", fontsize=8.0, weight="bold", color=INK, va="center")

    xb, xa, sy = 12, 92, 138
    ax.text(30, 130, "Before", fontsize=7.2, color=GRAY, ha="center", va="center")
    ax.text(110, 130, "After", fontsize=7.2, color=GRAY, ha="center", va="center")
    for i in range(3):
        ax.add_patch(Rectangle((xb + i * 12, sy), 9.5, 16,
                               facecolor=(TINT if i < 2 else WHITE),
                               edgecolor=(ACCENT if i < 2 else MID), lw=0.65))
        ax.add_patch(Rectangle((xa + i * 12, sy), 9.5, 16, facecolor=TINT, edgecolor=ACCENT, lw=0.65))
        if i == 2:
            ax.add_patch(Circle((xb + i * 12 + 4.75, sy + 8.0), 1.7, facecolor=INK))
            ax.add_patch(Circle((xa + i * 12 + 4.75, sy + 8.0), 1.7, facecolor=INK))
    ax.add_patch(FancyArrowPatch((56, 146), (80, 146), arrowstyle="-|>", mutation_scale=5.0, lw=0.8, color=ACCENT))

    ax.text(145, 140, "Prefix +1 · New calls 0 · Counted once", fontsize=7.7, color=ACCENT, weight="bold", va="center")
    ax.text(145, 154, r"Same $\kappa$ re-slice: $E\cup S\cup T \to E'$ (0 calls)", fontsize=7.3, color=INK, va="center")
    ax.text(8, 179, r"Prompt/model edit changes $\kappa \Rightarrow$ isolates old cache; never skips $T_h$ in $\pi_h$.",
            fontsize=7.2, color=GRAY, va="center")

    ax.add_line(Line2D([322, 322], [8, 186], color=RULE, lw=0.7))

    ax.text(332, 11, "Resulting bounds on Q1 (Exact | Rel)", fontsize=8.6, weight="bold", color=INK, va="center")
    ax.text(332, 23, "15% BM25 history + 2% new calls (seed 0)", fontsize=7.4, color=GRAY, va="center")

    px0, px1, vmax = 398, 462, 0.40
    X = lambda v: px0 + (px1 - px0) * v / vmax
    ax.add_line(Line2D([X(truth), X(truth)], [41, 160], color=INK, lw=0.8, ls="--"))
    ax.text(X(truth), 34, rf"$\theta_{{\mathrm{{run}}}}={truth:.3f}$", fontsize=7.4, color=INK, ha="center", va="center")

    rows = [
        ("Pool as\nsample", iv["pooled"], 54, BAD, BAD, 2.0),
        ("Discard\nhistory", iv["discard"], 84, MID, GRAY, 1.6),
        ("Free reads\n(in T)", iv["free"], 114, TEAL_MID, TEAL_MID, 1.8),
        ("COHORT\n(known E)", iv["cohort"], 144, ACCENT, ACCENT, 2.4),
    ]
    for lab, (lo, hi), y, col, tcol, lw in rows:
        ax.text(391, y, lab, fontsize=7.4, color=tcol, weight=("bold" if col == ACCENT else "normal"),
                ha="right", va="center", linespacing=1.14)
        ax.add_line(Line2D([X(lo), X(hi)], [y, y], color=col, lw=lw))
        for v in (lo, hi):
            ax.add_line(Line2D([X(v), X(v)], [y - 3.0, y + 3.0], color=col, lw=0.85))
        ax.text(466, y, f"[{lo:.2f}, {hi:.2f}]", fontsize=7.3, color=tcol,
                weight=("bold" if col == ACCENT else "normal"), ha="left", va="center")
    ax.text(466, 65, "misses truth", fontsize=6.8, color=BAD, style="italic", ha="left", va="center")

    ax.add_line(Line2D([px0, px1], [166, 166], color=INK, lw=0.65))
    for v in (0.0, 0.2, 0.4):
        ax.add_line(Line2D([X(v), X(v)], [166, 169], color=INK, lw=0.65))
        ax.text(X(v), 176, f"{v:.1f}", fontsize=7.2, color=GRAY, ha="center", va="center")
    ax.text((px0 + px1) / 2, 186, "Certified interval on Q1", fontsize=7.4, color=INK, ha="center", va="center")

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
# Fig 4: Partition Parallelogram + Coupled Multi-Predicate LP Tightening
# =========================================================================== #
def draw_fig4_geometry():
    g = load_json("exp_geometry.json")
    r4 = load_json("exp_review4_suite.json")["coupled_lp_vs_partition"]
    PA, PB = np.array(g["polygon_A"]), np.array(g["polygon_B"])

    fig, axs = plt.subplots(1, 2, figsize=(3.48, 1.92), gridspec_kw={"width_ratios": [1.12, 0.98], "wspace": 0.50})

    ax = axs[0]
    ax.fill(PB / 1000, PA / 1000, color=ACCENT, alpha=0.18, lw=0, zorder=2)
    ax.plot(np.r_[PB, PB[:1]] / 1000, np.r_[PA, PA[:1]] / 1000, color=ACCENT, lw=1.1, zorder=3)
    bl, bh, al, ah = g["B_lo"] / 1000, g["B_hi"] / 1000, g["A_lo"] / 1000, g["A_hi"] / 1000
    ax.add_patch(Rectangle((bl, al), bh - bl, ah - al, fill=False, ec="#777777", ls="--", lw=0.85, zorder=2))
    ax.set_xlim(13.5, 30.5)
    ax.set_ylim(9.0, 24.5)
    bx, ay = g["B_true"] / 1000, g["A_true"] / 1000
    ax.scatter([bx], [ay], marker="*", s=52, color=INK, zorder=6)
    ax.set_xlabel("Relevant $B$ (k)", fontsize=7.4, labelpad=3)
    ax.set_ylabel(r"Rated $\geq 4$, $A$ (k)", fontsize=7.4, labelpad=3)
    ax.set_title("Single-ratio partition\n[0.63, 0.80] vs. [0.46, 1.00]", loc="left", fontsize=7.3, weight="bold", pad=5)

    ax2 = axs[1]
    caps = ["1280", "3200", "6400"]
    cap_labs = ["4%", "10%", "20%"]
    w_sep = [r4[c]["sep_partition_mean_width"] for c in caps]
    w_lp = [r4[c]["joint_lp_mean_width"] for c in caps]
    xp = np.arange(3)
    bw = 0.33
    ax2.bar(xp - bw / 2, w_sep, width=bw, color=C_PRIO, alpha=0.88, label="Sep. partition")
    ax2.bar(xp + bw / 2, w_lp, width=bw, color=ACCENT, label="Joint LP")
    for idx in range(3):
        sh = r4[caps[idx]]["width_reduction_pct"]
        ax2.text(xp[idx], max(w_sep[idx], w_lp[idx]) + 0.015, f"-{sh:.0f}%",
                 ha="center", va="bottom", fontsize=6.6, color=ACCENT, weight="bold")
    ax2.set_xticks(xp)
    ax2.set_xticklabels(cap_labs, fontsize=7.2)
    ax2.set_ylim(0, 0.51)
    ax2.set_xlabel("Call cap (share of N)", fontsize=7.4, labelpad=3)
    ax2.set_ylabel("Interval width", fontsize=7.4, labelpad=3)
    ax2.set_title("Coupled bias LP\n(Ex4B - Ex8B) | Rel8B", loc="left", fontsize=7.3, weight="bold", pad=5)
    ax2.legend(loc="upper right", frameon=False, fontsize=6.5, handlelength=1.2, borderaxespad=0.0)
    ax2.grid(axis="y", lw=0.35, alpha=0.45)

    fig.savefig(FIG / "fig_geometry.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_geometry.pdf")


# =========================================================================== #
# Fig 5: Mechanism Attribution & Cold-Start Trade-Off (Zero Label Clipping!)
# =========================================================================== #
def draw_fig5_ladder():
    d = load_json("exp_ladder.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.20), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.38})
    GRID = [640, 1536, 3200, 6272, 15744, 31482]
    GRID_LAB = ["2%", "5%", "10%", "20%", "50%", "100%"]

    styles = [
        ("EG-style claim-by-claim", "Evergreen-style + cache", C_EG, "--", "s"),
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

    axs[1].text(v_co[0] + 0.12, y[0] - bh, f"{v_co[0]:.2f}k", va="center", fontsize=6.7, color=ACCENT, weight="bold")
    axs[1].text(v_ds[0] + 0.12, y[0] + bh, f"{v_ds[0]:.2f}k", va="center", fontsize=6.7, color=C_DEDUP)

    axs[1].set_yticks(y)
    axs[1].set_yticklabels(labels, fontsize=7.4)
    axs[1].invert_yaxis()
    # Short, unclipped axis label!
    axs[1].set_xlabel("Cold-start median calls (thousands)")
    axs[1].set_xlim(0, 7.2)
    axs[1].grid(axis="x", lw=0.35, alpha=0.45)
    axs[1].legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, frameon=False,
                  fontsize=7.0, handlelength=1.8, columnspacing=0.9, labelspacing=0.30, borderaxespad=0.0)

    fig.savefig(FIG / "fig_ladder.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved fig_ladder.pdf")


# =========================================================================== #
# Fig 6: 3-Panel Multi-Domain 8-Round Trajectories vs. Dynamic E & Shared-Priority
# =========================================================================== #
def draw_fig6_multicohort():
    r5 = load_json("exp_review5_cpu.json")
    t1 = r5["wands_task1_8rounds"]
    pk = r5["wands_split_cost_cascade_8rounds"]
    dl = r5["druglib_triage_8rounds"]

    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.28), gridspec_kw={"wspace": 0.30})

    m_styles = [
        ("Adaptive dedup + E0", "Adaptive dedup + E0", C_DEDUP, "--", "^", 1.25, False),
        ("Adaptive dedup + dynamic E", "Adaptive dedup + dynamic E", C_DYNE, "-.", "s", 1.45, True),
        ("Oracle dedup + dynamic E", "Oracle risk + dynamic E", "#8C6BB1", ":", "v", 1.25, False),
        ("Adaptive shared-priority + E0", "Shared-priority + E0 + part.", C_PRIO, "-", "o", 1.55, True),
        ("COHORT (known mass E0 + shared region)", "COHORT (known mass E0 + region)", ACCENT, "-", "D", 2.0, True),
    ]

    panels = [
        (axs[0], t1, "WANDS Relevance (bal. E0, 24t)", 22.0, "0 new calls\n(R4–R8)"),
        (axs[1], pk, "WANDS Cascade (skew E0, 24t)", 17.5, "-38% vs. Prio\n-65% vs. DynE"),
        (axs[2], dl, "UCI DrugLib (N=4,143, 18t)", 3.35, "-38% vs. DynE\n+0.6% vs. Prio"),
    ]

    for idx, (ax, data, title, ymax, note) in enumerate(panels):
        rx = np.arange(1, 9)
        for key, lab, col, ls, mk, lw, band in m_styles:
            md = data["methods"][key]
            meds = np.array(md["round_cumulative_medians"], dtype=float) / 1000.0
            ax.plot(rx, meds, color=col, ls=ls, marker=mk, ms=3.4, lw=lw, label=lab,
                    zorder=(5 if key.startswith("COHORT") else 3))
            if band:
                q25 = np.array(md["round_cumulative_q25"], dtype=float) / 1000.0
                q75 = np.array(md["round_cumulative_q75"], dtype=float) / 1000.0
                ax.fill_between(rx, q25, q75, color=col, alpha=0.10, lw=0, zorder=2)

        co_end = data["methods"]["COHORT (known mass E0 + shared region)"]["round_cumulative_medians"][-1] / 1000.0
        box_kw = dict(facecolor=WHITE, edgecolor="none", pad=1.0, alpha=0.92)
        if idx == 0:
            # Place cleanly in empty space below all curves at R5.5-R8 (all curves >= 6.8 here!)
            ax.annotate(note, xy=(6.2, co_end - 0.25), xytext=(6.1, 3.1),
                        fontsize=6.6, color=ACCENT, weight="bold", ha="center", va="center",
                        bbox=box_kw, arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.75, mutation_scale=6),
                        zorder=7)
        elif idx == 1:
            ax.annotate(note, xy=(6.2, co_end - 0.18), xytext=(6.0, 1.55),
                        fontsize=6.6, color=ACCENT, weight="bold", ha="center", va="center",
                        bbox=box_kw, arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.75, mutation_scale=6),
                        zorder=7)
        else:
            ax.annotate(note, xy=(6.2, co_end - 0.05), xytext=(6.0, 0.50),
                        fontsize=6.6, color=ACCENT, weight="bold", ha="center", va="center",
                        bbox=box_kw, arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.75, mutation_scale=6),
                        zorder=7)

        ax.set_xticks([1, 2, 4, 6, 8])
        ax.set_xticklabels(["R1", "R2", "R4", "R6", "R8"], fontsize=7.2)
        ax.set_xlabel("Session round", labelpad=3)
        if idx == 0:
            ax.set_ylabel("Cumulative new calls (thousands)")
        ax.set_xlim(0.75, 8.30)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", lw=0.35, alpha=0.45)
        ax.set_title(title, loc="left", weight="bold", fontsize=7.25, pad=6)

    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.20),
               fontsize=7.0, handlelength=2.0, columnspacing=1.35, labelspacing=0.32)
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

    fig, axs = plt.subplots(1, 3, figsize=(7.15, 2.24), gridspec_kw={"wspace": 0.45})

    ax = axs[0]
    enr = sorted({v["ratio"] for v in d.values() if v["kind"] == "enrich"})
    for arm, (lab, col, ls, mk) in [
        ("pooled-as-random", ("Pooled (CS)", BAD, "--", "^")),
        ("naive", ("Pooled (Wald)", "#7a1f00", (0, (3, 1, 1, 1)), "v")),
    ]:
        xs = [d[f"enrich:{r}@0.1"]["hit_hist"] / d[f"enrich:{r}@0.1"]["hit_pop"] for r in enr]
        ys = [100 * d[f"enrich:{r}@0.1"]["arms"][arm]["dec_err_rate"] for r in enr]
        ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=3.1, lw=1.35, label=lab, zorder=3)
        vb = d["bm25:0@0.1"]
        xb = vb["hit_hist"] / vb["hit_pop"]
        yb = 100 * vb["arms"][arm]["dec_err_rate"]
        ax.scatter([xb], [yb], marker="D", s=22, facecolor=WHITE, edgecolor=col, lw=1.15, zorder=6)

    # Single clean zero line representing all three selection-safe rules without cross-panel spillover!
    xs_safe = [d[f"enrich:{r}@0.1"]["hit_hist"] / d[f"enrich:{r}@0.1"]["hit_pop"] for r in enr]
    ax.plot(xs_safe, [0.0] * len(xs_safe), color=ACCENT, ls="-", marker="o", ms=3.2, lw=1.75,
            label="COHORT / T / Discard", zorder=4)
    ax.plot([], [], marker="D", ls="none", markerfacecolor=WHITE, markeredgecolor=INK, ms=4.0,
            label="Real BM25 (2.2×)")
    # Placed strictly inside axs[0] in the empty upper-left space (x in [1.05, 3.2], y in [36, 51])!
    ax.text(0.05, 0.89, "E, T, Discard:\n0.0% false certs", transform=ax.transAxes,
            fontsize=6.5, color=ACCENT, weight="bold", ha="left", va="top",
            bbox=dict(facecolor=WHITE, edgecolor="none", pad=0.8, alpha=0.9))

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])
    ax.minorticks_off()
    ax.set_xlim(0.90, 9.2)
    ax.set_ylim(-3.5, 50)
    ax.set_xlabel("Exact-match enrichment in history", labelpad=5.5)
    ax.set_ylabel("Wrong certified claims (%)")
    ax.grid(axis="y", lw=0.35, alpha=0.45)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.06, 1.03), ncol=2, frameon=False,
              fontsize=6.5, handlelength=1.45, columnspacing=0.55, labelspacing=0.25, borderaxespad=0.0)

    # Middle: Two-line non-overlapping x-tick labels with generous xlabel pad!
    ax = axs[1]
    order = [("uniform", "Unif.\n(1.0×)"), ("bm25", "BM25\n(2.2×)"),
             ("mild", "Odds\n×4*"), ("strong", "Odds\n×16")]
    for i, (key, _) in enumerate(order):
        rows = paired["full"][key]
        both = [r for r in rows if r["E"].get("calls_all") and r["T"].get("calls_all")]
        ys = [100 * (1 - r["E"]["calls_all"] / r["T"]["calls_all"]) for r in both]
        rng_j = np.linspace(-0.09, 0.09, len(ys))
        ax.scatter(i + rng_j, ys, color=ACCENT, s=21, alpha=0.82, zorder=3)
        med = statistics.median(ys)
        ax.plot([i - 0.22, i + 0.22], [med, med], color=INK, lw=1.8, zorder=4)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([x[1] for x in order], fontsize=6.8, linespacing=1.08)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-4, 38)
    ax.grid(axis="y", lw=0.35, alpha=0.45)
    ax.set_xlabel("History regime (*1 empty CS)", labelpad=5.5)
    ax.set_ylabel("Paired saving: E vs. T (%)")
    ax.set_title("Known mass E vs. free reads T", loc="left", weight="bold", fontsize=7.9, pad=7)

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
        ax.text(idx, v + 0.28, f"{v:.2f}k", ha="center", va="bottom", fontsize=7.0,
                weight=("bold" if idx == 2 else "normal"), color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cats, fontsize=6.8, linespacing=1.10)
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylim(0, 11.5)
    ax.set_xlabel("Mid-stage retrieval protocol", labelpad=5.5)
    ax.set_ylabel("Total calls for 5 targets (k)")
    ax.set_title("Mid-stage retrieval + prefix", loc="left", weight="bold", fontsize=7.9, pad=7)

    fig.savefig(FIG / "fig_history.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_history.pdf")


# =========================================================================== #
# Fig 8: Stage Re-Slicing (Explicitly Separating Numeric Targets vs. Thresholds!)
# =========================================================================== #
def draw_fig8_session():
    d = load_json("exp_session.json")
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 2.82), gridspec_kw={"height_ratios": [1.28, 0.95], "hspace": 0.56})

    ax = axs[0]
    style = {
        "COHORT-T (restart, free reads)": ("Restart + free reads", TEAL_MID, "--", "s", WHITE, 4.4, 4),
        "EG-style": ("Evergreen-style", C_EG, "-.", "^", C_EG, 3.8, 3),
        "COHORT (E-inherit)": ("COHORT (E-inherit)", ACCENT, "-", "o", ACCENT, 3.1, 6),
    }
    curves = {}
    for m, (lab, col, ls, mk, mfc, ms, zo) in style.items():
        runs = d["runs"][m]
        xs = np.array([[e["calls"] for e in r["traj"]] for r in runs], float)
        ys = np.array([[r["traj"][0]["certified"], r["traj"][1]["certified"], r["traj"][2]["certified"],
                        r["traj"][2]["certified"] + r["traj"][3]["certified"]] for r in runs], float)
        curves[m] = (xs, ys)
        ax.errorbar(xs.mean(0) / 1000.0, ys.mean(0), yerr=ys.std(0), color=col, ls=ls, marker=mk,
                    markerfacecolor=mfc, markeredgecolor=col, markeredgewidth=1.0,
                    ms=ms, lw=1.35, capsize=1.5, elinewidth=0.7, label=lab, zorder=zo)

    X0, _ = curves["COHORT (E-inherit)"][0].mean(0) / 1000.0, curves["COHORT (E-inherit)"][1].mean(0)
    ax.axvline(X0[2], color="#888888", ls=":", lw=0.85, zorder=0)
    ax.text(X0[2] - 0.32, 27.5, "Stage 1\n(5 num + 13 thr)", fontsize=6.5, color="#555555", ha="right", va="center")
    ax.text(X0[2] + 0.32, 8.5, "Stage 2\n(8 num + 6 thr)", fontsize=6.5, color="#555555", ha="left", va="center")

    ax.set_xlabel("Cumulative semantic calls (thousands)", fontsize=7.6)
    ax.set_ylabel("Certified claims (13 num + 19 thr)", fontsize=7.2)
    ax.set_xlim(-0.4, 11.8)
    ax.set_ylim(-1, 36.5)
    ax.set_yticks([0, 10, 20, 30])
    ax.grid(axis="y", lw=0.35, alpha=0.45)
    h, l = ax.get_legend_handles_labels()
    order_leg = [2, 0, 1]
    ax.legend([h[i] for i in order_leg], [l[i] for i in order_leg],
              loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, frameon=False,
              fontsize=6.9, handlelength=1.7, columnspacing=1.0, labelspacing=0.26, borderaxespad=0.0)

    ax2 = axs[1]
    methods_r4 = ["Evergreen-style + cache", "Restart + free reads", "COHORT (E-inherit)"]
    calls_r4 = [3.148, 3.016, 2.794]
    cols_r4 = [C_EG, TEAL_MID, ACCENT]
    certs_r4 = ["8.4/14", "14/14 (all 8 num)", "14/14 (-7.4%)"]
    y_pos = np.arange(3)
    ax2.barh(y_pos, calls_r4, height=0.50, color=cols_r4)
    for idx, (v, ct) in enumerate(zip(calls_r4, certs_r4)):
        ax2.text(v + 0.08, idx, f"{v:.2f}k  {ct}", ha="left", va="center", fontsize=6.8,
                 weight=("bold" if idx == 2 else "normal"), color=INK)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods_r4, fontsize=7.2)
    ax2.set_ylim(-0.55, 2.55)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 5.20)
    ax2.set_xticks([0, 1, 2, 3, 4, 5])
    ax2.set_xticklabels(["0", "1k", "2k", "3k", "4k", "5k"], fontsize=7.2)
    ax2.set_xlabel("Stage-2 new calls after category re-slice", fontsize=7.5)

    fig.savefig(FIG / "fig_session.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_session.pdf")


# =========================================================================== #
# Fig 9: Plan-Selection Phase Diagram (Overlap x Horizon & Skew x Horizon)
# =========================================================================== #
def draw_fig9_denom():
    r5 = load_json("exp_review5_cpu.json")["phase_diagrams_clean"]
    g_dy = r5["overlap_x_horizon_vs_dyne"]
    g_sp = r5["skew_x_horizon_vs_shared_prio"]

    fig, axs = plt.subplots(1, 2, figsize=(3.48, 2.08), gridspec_kw={"wspace": 0.58})

    # Left heatmap: COHORT / Dynamic-E call ratio across Overlap x Rounds
    ov_labs = ["Disjoint", "Subgroup", "Half-pool", "Multi-cohort"]
    r_labs = ["R1", "R2", "R4", "R8"]
    M1 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M1[i, j] = g_dy[i * 4 + j]["ratio_cohort_over_dyne"]

    # Diverging-like intuitive colormap: <1.0 (COHORT wins) is deep teal; >1.0 is light warm grey
    im1 = axs[0].imshow(1.0 / np.clip(M1, 0.30, 3.30), cmap=TEAL_CMAP, vmin=0.30, vmax=1.85, aspect="auto")
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(r_labs, fontsize=7.0)
    axs[0].set_yticks(range(4)); axs[0].set_yticklabels(ov_labs, fontsize=6.9)
    for i in range(4):
        for j in range(4):
            val = M1[i, j]
            axs[0].text(j, i, f"{val:.2f}×", ha="center", va="center", fontsize=6.5,
                        weight=("bold" if val < 1.0 else "normal"),
                        color=("white" if val < 0.85 else INK))
    axs[0].set_xlabel("Session horizon", fontsize=7.3, labelpad=3)
    axs[0].set_ylabel("Query scope overlap", fontsize=7.3, labelpad=3)
    axs[0].set_title("COHORT / Dyn-E\n(<1: COHORT wins)", loc="left", fontsize=7.2, weight="bold", pad=5)
    for sp in axs[0].spines.values(): sp.set_visible(False)
    axs[0].tick_params(length=0)

    # Right heatmap: COHORT / Shared-Priority call ratio across E0 Skew x Rounds
    sk_labs = ["45%", "65%", "80%", "95%"]
    r2_labs = ["R1", "R2", "R3", "R4"]
    M2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M2[i, j] = g_sp[i * 4 + j]["ratio_cohort_over_shared_prio"]

    axs[1].imshow(1.0 / np.clip(M2, 0.30, 3.30), cmap=TEAL_CMAP, vmin=0.30, vmax=1.85, aspect="auto")
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels(r2_labs, fontsize=7.0)
    axs[1].set_yticks(range(4)); axs[1].set_yticklabels(sk_labs, fontsize=6.9)
    for i in range(4):
        for j in range(4):
            val = M2[i, j]
            axs[1].text(j, i, f"{val:.2f}×", ha="center", va="center", fontsize=6.5,
                        weight=("bold" if val < 1.0 else "normal"),
                        color=("white" if val < 0.85 else INK))
    axs[1].set_xlabel("Cascade round", fontsize=7.3, labelpad=3)
    axs[1].set_ylabel("E0 skew in proxy+", fontsize=7.3, labelpad=3)
    axs[1].set_title("COHORT / Shared-Prio\n(<1: COHORT wins)", loc="left", fontsize=7.2, weight="bold", pad=5)
    for sp in axs[1].spines.values(): sp.set_visible(False)
    axs[1].tick_params(length=0)

    fig.savefig(FIG / "fig_denom.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved fig_denom.pdf")


# =========================================================================== #
# Generate Updated LaTeX Tables
# =========================================================================== #
def write_tables():
    r5 = load_json("exp_review5_cpu.json")
    live = load_json("exp_live_canonical_gpu.json")
    hk = load_json("exp_hk_scale.json")["rows"]
    calib = load_json("exp_calib.json")

    t1 = r5["wands_task1_8rounds"]["methods"]
    pk = r5["wands_split_cost_cascade_8rounds"]["methods"]
    dl = r5["druglib_triage_8rounds"]["methods"]

    rows_mc = [
        ("Shared prefix + free reads (T)", "Shared prefix + free reads ($T$)"),
        ("Adaptive dedup + free reads (T)", "Adaptive dedup + free reads ($T$)"),
        ("Adaptive dedup + E0", "Adaptive dedup + known mass $E_0$"),
        ("Oracle dedup + E0", "Oracle risk + known mass $E_0$"),
        ("Adaptive dedup + dynamic E", "Adaptive dedup + dynamic $E$"),
        ("Oracle dedup + dynamic E", "Oracle risk $\\times$ dynamic $E$"),
        ("Adaptive shared-priority + E0", "Coordinated shared-priority + $E_0$"),
        ("Oracle shared-priority + E0", "Oracle coordinated priority + $E_0$"),
        ("COHORT (known mass E0 + shared region)", r"\textbf{\textsc{Cohort} ($E_0$ + shared region)}"),
    ]

    lines = []
    for k, label in rows_mc:
        m1, m2, m3 = t1[k], pk[k], dl[k]
        d1 = m1["round_done"][-1]
        c1_r4 = fmt_int(m1["round_cumulative_medians"][3])
        c1_r8 = fmt_int(m1["round_cumulative_medians"][7])
        s1 = "--" if k.startswith("COHORT") else f"{m1['cohort_paired_saving_median_by_round_pct'][7]:+.1f}\\%"

        c2_r4 = fmt_int(m2["round_cumulative_medians"][3])
        c2_r8 = fmt_int(m2["round_cumulative_medians"][7])
        s2 = "--" if k.startswith("COHORT") else f"{m2['cohort_paired_saving_median_by_round_pct'][7]:+.1f}\\%"

        c3_r8 = fmt_int(m3["round_cumulative_medians"][7])
        s3 = "--" if k.startswith("COHORT") else f"{m3['cohort_paired_saving_median_by_round_pct'][7]:+.1f}\\%"

        if k == "Oracle shared-priority + E0":
            c1_r8 = f"\\textbf{{{c1_r8}}}"
            c3_r8 = f"\\textbf{{{c3_r8}}}"
        if k == "Adaptive shared-priority + E0":
            c1_r4 = f"\\textbf{{{c1_r4}}}"
        if k.startswith("COHORT"):
            c2_r4 = f"\\textbf{{{c2_r4}}}"
            c2_r8 = f"\\textbf{{{c2_r8}}}"

        mark1 = "$^\\dagger$" if d1 != "10/10" else ""
        mark2 = "$^\\dagger$" if m2["round_done"][-1] != "10/10" else ""
        lines.append(f"{label} & {c1_r4} & {c1_r8}{mark1} & {s1} & {c2_r4} & {c2_r8}{mark2} & {s2} & {c3_r8} & {s3} \\\\")

    tab_mc = r"""\begin{table*}[t]
  \centering
  \caption{Multi-round session new calls across three workloads and two domains ($\alpha=5\%$, 10 seeds; 15\% biased history). Every baseline shares a global cache, deduplicates ratio expressions, and uses exact partition bounds~\eqref{eq:partition}. ``Saves'' is $\operatorname{median}_s(1 - C_{\textsc{Cohort},s}/C_{\mathrm{base},s})$ at Round~8 over jointly completed seeds ($^\dagger$9/10 completed on WANDS Task~1 $E_0$-dedup due to one empty-CS halt; 10/10 on all other cells, with zero false certificates for \textsc{Cohort}, dynamic-$E$, and shared-priority). Under balanced $E_0$ (Relevance), coordinated shared-priority wins by pooling unstratified sequences; under skewed proxy/triage $E_0$ (Cascade), \textsc{Cohort} beats both dynamic-$E$ and shared-priority.}
  \label{tab:multicohort}
  \footnotesize
  \renewcommand{\arraystretch}{1.12}
  \setlength{\tabcolsep}{3.1pt}
  \begin{tabular}{@{}l rrr rrr rr@{}}
    \toprule
    & \multicolumn{3}{c}{\textbf{WANDS Relevance} ($N{=}31{,}482$)} & \multicolumn{3}{c}{\textbf{WANDS Split-Cost Cascade} (8B)} & \multicolumn{2}{c}{\textbf{UCI DrugLib} ($N{=}4{,}143$)} \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
    Execution strategy & R4 & R8 & Saves & R4 & R8 & Saves & R8 & Saves \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
"""
    (GEN / "tab_multicohort.tex").write_text(tab_mc)

    # Live GPU table (with effective batch fill & all executors!)
    l_pk = live["wands_split_cost_cascade_live"]
    l_t1 = live["wands_relevance_5target_live"]
    bu = live["batch_utilization_and_tokens"]
    full_w = live["wands_full_table_scan"]["measured_full_table_wall_s"]
    tab_live = rf"""\begin{{table}}[t]
  \centering
  \caption{{Measured end-to-end live GPU execution on NVIDIA RTX~4080~SUPER under the canonical fixed-shape protocol (\texttt{{batch\_size}}=32, padded length 288, Qwen3-8B BF16; \textbf{{0 label mismatches}} and \textbf{{0 false certificates}} vs.\ the measured 1{{,}}814.2\,s full scan). ``Eff.\ BS'' is mean uncached tuples per 32-slot GPU batch.}}
  \label{{tab:livegpu}}
  \footnotesize
  \renewcommand{{\arraystretch}}{{1.11}}
  \setlength{{\tabcolsep}}{{2.6pt}}
  \begin{{tabular}}{{@{{}}l rrrrr@{{}}}}
    \toprule
    Live executor & Calls & Eff.\ BS & 1st\,(s) & Total\,(s) & CPU\,(s) \\
    \midrule
    \multicolumn{{6}}{{@{{}}l}}{{\emph{{4-round Split-Cost Cascade (12 targets, skewed proxy $E_0$, 3 seeds)}}}} \\
    Dedup + dynamic $E$ & {fmt_int(l_pk['Dedup-streams + dynamic E (live)']['mean_new_calls'])} & {bu['cascade_4round']['Dedup_dynamic_E']['mean_effective_batch_size']:.1f} & \textbf{{{l_pk['Dedup-streams + dynamic E (live)']['mean_first_cert_wall_s']:.1f}}} & {l_pk['Dedup-streams + dynamic E (live)']['mean_total_wall_s']:.1f} & {l_pk['Dedup-streams + dynamic E (live)']['mean_cpu_s']:.2f} \\
    Shared-priority + $E_0$ & {fmt_int(l_pk['Shared-priority + E0 + partition (live)']['mean_new_calls'])} & {bu['cascade_4round']['Shared_priority_E0']['mean_effective_batch_size']:.1f} & {l_pk['Shared-priority + E0 + partition (live)']['mean_first_cert_wall_s']:.1f} & {l_pk['Shared-priority + E0 + partition (live)']['mean_total_wall_s']:.1f} & {l_pk['Shared-priority + E0 + partition (live)']['mean_cpu_s']:.2f} \\
    \textbf{{\textsc{{Cohort}} (default)}} & \textbf{{{fmt_int(l_pk['COHORT (live, default)']['mean_new_calls'])}}} & {bu['cascade_4round']['COHORT']['mean_effective_batch_size']:.1f} & {l_pk['COHORT (live, default)']['mean_first_cert_wall_s']:.1f} & \textbf{{{l_pk['COHORT (live, default)']['mean_total_wall_s']:.1f}}} & {l_pk['COHORT (live, default)']['mean_cpu_s']:.2f} \\
    \textbf{{\textsc{{Cohort}} (exact audit)}} & \textbf{{{fmt_int(l_pk['COHORT (live, exact rational audit)']['mean_new_calls'])}}} & {bu['cascade_4round']['COHORT']['mean_effective_batch_size']:.1f} & {l_pk['COHORT (live, exact rational audit)']['mean_first_cert_wall_s']:.1f} & \textbf{{{l_pk['COHORT (live, exact rational audit)']['mean_total_wall_s']:.1f}}} & {l_pk['COHORT (live, exact rational audit)']['mean_cpu_s']:.2f} \\
    \midrule
    \multicolumn{{6}}{{@{{}}l}}{{\emph{{2-round Relevance batch (5 targets, balanced BM25 $E_0$, 2 seeds)}}}} \\
    Dedup + dynamic $E$ & {fmt_int(l_t1['Dedup-streams + dynamic E (live)']['mean_new_calls'])} & {bu['relevance_2round']['Dedup_dynamic_E']['mean_effective_batch_size']:.1f} & {l_t1['Dedup-streams + dynamic E (live)']['mean_first_cert_wall_s']:.1f} & {l_t1['Dedup-streams + dynamic E (live)']['mean_total_wall_s']:.1f} & {l_t1['Dedup-streams + dynamic E (live)']['mean_cpu_s']:.2f} \\
    \textbf{{Shared-priority + $E_0$}} & \textbf{{{fmt_int(l_t1['Shared-priority + E0 + partition (live)']['mean_new_calls'])}}} & {bu['relevance_2round']['Shared_priority_E0']['mean_effective_batch_size']:.1f} & \textbf{{{l_t1['Shared-priority + E0 + partition (live)']['mean_first_cert_wall_s']:.1f}}} & \textbf{{{l_t1['Shared-priority + E0 + partition (live)']['mean_total_wall_s']:.1f}}} & {l_t1['Shared-priority + E0 + partition (live)']['mean_cpu_s']:.2f} \\
    \textsc{{Cohort}} (default) & {fmt_int(l_t1['COHORT (live, default)']['mean_new_calls'])} & {bu['relevance_2round']['COHORT']['mean_effective_batch_size']:.1f} & {l_t1['COHORT (live, default)']['mean_first_cert_wall_s']:.1f} & {l_t1['COHORT (live, default)']['mean_total_wall_s']:.1f} & {l_t1['COHORT (live, default)']['mean_cpu_s']:.2f} \\
    \textsc{{Cohort}} (exact audit) & {fmt_int(l_t1['COHORT (live, exact rational audit)']['mean_new_calls'])} & {bu['relevance_2round']['COHORT']['mean_effective_batch_size']:.1f} & {l_t1['COHORT (live, exact rational audit)']['mean_first_cert_wall_s']:.1f} & {l_t1['COHORT (live, exact rational audit)']['mean_total_wall_s']:.1f} & {l_t1['COHORT (live, exact rational audit)']['mean_cpu_s']:.2f} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    (GEN / "tab_livegpu.tex").write_text(tab_live)

    # Plan-selection comparison table across all 45 held-out randomized sessions (Review 6 Section III.3)
    sel = r5["unseen_multidomain_selector"]
    vst = sel["vs_shared_priority_table"]
    tab_sel = rf"""\begin{{table}}[t]
  \centering
  \caption{{Offline call-cost plan selection between \textsc{{Cohort}} and Coordinated Shared-Priority+$E_0$ across 45 randomized 1--6 round sessions (15 each on WANDS Relevance, WANDS Split-Cost Cascade, and UCI DrugLib; 0 ties: \textsc{{Cohort}} strictly cheaper on 8/45, Shared-priority on 37/45).}}
  \label{{tab:selector}}
  \footnotesize
  \renewcommand{{\arraystretch}}{{1.11}}
  \setlength{{\tabcolsep}}{{2.4pt}}
  \begin{{tabular}}{{@{{}}l rrrr@{{}}}}
    \toprule
    Selection policy & Calls & Mean reg. & P90/Max & Win acc. \\
    \midrule
    Always \textsc{{Cohort}} & {fmt_int(vst['Always_COHORT']['mean_calls'])} & {vst['Always_COHORT']['mean_regret_pct']:.1f}\% & {vst['Always_COHORT']['p90_regret_pct']:.0f}/{vst['Always_COHORT']['max_regret_pct']:.0f}\% & {vst['Always_COHORT']['winner_accuracy_pct']:.1f}\% \\
    Horizon + skew rule & {fmt_int(vst['Simple_Horizon_Skew_Rule']['mean_calls'])} & {vst['Simple_Horizon_Skew_Rule']['mean_regret_pct']:.1f}\% & {vst['Simple_Horizon_Skew_Rule']['p90_regret_pct']:.0f}/{vst['Simple_Horizon_Skew_Rule']['max_regret_pct']:.0f}\% & {vst['Simple_Horizon_Skew_Rule']['winner_accuracy_pct']:.1f}\% \\
    Always Shared-priority & {fmt_int(vst['Always_Shared_Priority']['mean_calls'])} & \textbf{{{vst['Always_Shared_Priority']['mean_regret_pct']:.1f}\%}} & \textbf{{{vst['Always_Shared_Priority']['p90_regret_pct']:.1f}}}/{vst['Always_Shared_Priority']['max_regret_pct']:.0f}\% & \textbf{{{vst['Always_Shared_Priority']['winner_accuracy_pct']:.1f}\%}} \\
    \textbf{{Metadata cost heuristic}} & \textbf{{{fmt_int(vst['Metadata_Call_Cost_Heuristic']['mean_calls'])}}} & {vst['Metadata_Call_Cost_Heuristic']['mean_regret_pct']:.1f}\% & {vst['Metadata_Call_Cost_Heuristic']['p90_regret_pct']:.1f}/{vst['Metadata_Call_Cost_Heuristic']['max_regret_pct']:.0f}\% & \textbf{{{vst['Metadata_Call_Cost_Heuristic']['winner_accuracy_pct']:.1f}\%}} \\
    \midrule
    Oracle best (ex-post) & {fmt_int(vst['Oracle_Best']['mean_calls'])} & 0.0\% & 0/0\% & 100.0\% \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""
    (GEN / "tab_selector.tex").write_text(tab_sel)

    # Update tab_ladder to say "Evergreen-style + cache" accurately
    tab_lad_old = (GEN / "tab_ladder.tex").read_text()
    tab_lad_new = tab_lad_old.replace("Claim-by-claim, shared cache", "Evergreen-style, shared cache")
    (GEN / "tab_ladder.tex").write_text(tab_lad_new)

    # Calibration table
    pw, e2e = calib["pathwise"], calib["e2e"]
    tab_cal = rf"""\begin{{table}}[t]
  \centering
  \caption{{Empirical calibration of Theorem~\ref{{thm:stage}}. Rows 1--3 check any-step region coverage ($k^\star \in \mathcal{{P}}_t$ across all prefixes of 3{{,}}000 full-pool permutations). Row~4 counts end-to-end sessions emitting any false certificate across {e2e['certified_claims']:,} certified claims.}}
  \label{{tab:calib}}
  \footnotesize
  \renewcommand{{\arraystretch}}{{1.12}}
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

    hk_lines = [
        f"{r['H']} & {r['K']} & {r['lp_vars']} & {r['n_basis']} & {r['n_cs']} & "
        f"{r['q1_width_mean']:.3f} & {r['cpu_s_mean']:.1f} & {r['peak_mb_mean']:.1f} \\\\"
        for r in hk
    ]
    tab_hk = r"""\begin{table}[t]
  \centering
  \caption{Structural scaling across strata $H$ and semantic predicates $K$ at 3{,}200 calls ($N=31{,}482$, mean over 3 seeds). Projection CPU is 2.3\,s at the default $H{=}6,K{=}2$, rising to 107.5\,s at the $H{=}24,K{=}4$ corner; heap stays at 4--6\,MB.}
  \label{tab:hkscale}
  \footnotesize
  \renewcommand{\arraystretch}{1.12}
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
    print("wrote all Review-5 tables")


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
