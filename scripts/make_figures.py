# -*- coding: utf-8 -*-
"""论文图表（全部由 reports/icde2027_suite/*.json 的实测结果生成；示意图中的区间数值也来自真实运行）。

视觉规则（全文统一，见 fig_style.py）
  * 青绿 = COHORT 信任并返回的东西（随机前缀 S、共享区域、认证结果、各图中的 COHORT 曲线），浅青绿 = 其近亲变体；
    朱红 = 不安全做法 / 错误；其余（已知质量 E、基线、辅助线）一律灰色。
  * 图中字号在最终 IEEE 双栏尺寸下 >= 6.5pt；图注先写结论再写设置。

图号 -> 命令 -> 输出文件
  Fig. 1  ambig    fig_ambiguity.pdf   fig_family.py   （N=100 的玩具例子，无实验数据）
  Fig. 2  reuse    fig_reuse.pdf       fig_family.py   （区间读自 reuse_intervals.json）
  Fig. 3  stage    fig_stage.pdf       fig_executor.py （系统示意，无数据）
  Fig. 4  geom     fig_geometry.pdf    exp_geometry.json
  Fig. 5  ladder   fig_ladder.pdf      exp_ladder.json
  Fig. 6  claims   fig_claims.pdf      exp_ladder.json
  Fig. 7  history  fig_history.pdf     exp_bias20.json
  Fig. 8  session  fig_session.pdf     exp_session.json
  Fig. 9  denom    fig_denom.pdf       exp_denom.json
不带参数运行 = 重画全部九张。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from fig_style import ACCENT as C_COHORT, TEAL_MID as C_VARIANT, BAD as C_BAD, TEAL_CMAP  # also registers the fonts
import fig_executor                                # Fig. 3
import fig_family                                  # Figs. 1 and 2

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "..", "reports", "icde2027_suite")
FIG = os.environ.get("FIG_OUT", os.path.join(_HERE, "..", "paper", "figure"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 7.8, "axes.labelsize": 7.8, "axes.titlesize": 8.2,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4,
})


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------- #
# Fig. 5  阶梯：随时可用的认证数与数值区间宽度
# --------------------------------------------------------------------------- #
GRID = [640, 1536, 3200, 6272, 15744, 31482]
GRID_LAB = ["2%", "5%", "10%", "20%", "50%", "100%"]
LADDER_STYLE = [
    ("B2 per-query (no sharing)", "per-claim, no sharing", "#9a9a9a", ":", "x"),
    ("EG-style claim-by-claim", "claim-by-claim + shared cache", "#6a3d9a", "--", "s"),
    ("Shared pool, dedup (no polytope)", "shared samples, no region", "#c2185b", "-.", "^"),
    ("+ joint projection (RR)", "+ joint region", C_VARIANT, "-", "o"),
    ("+ certificate-guided scheduler", "+ certificate-guided scheduler", C_COHORT, "-", "D"),
]


def plot_ladder():
    d = load("exp_ladder.json")
    if d is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.15, 2.35), gridspec_kw={"wspace": 0.28})
    for key, lab, col, ls, mk in LADDER_STYLE:
        if f"{key}@{GRID[0]}" not in d:
            continue
        # five numeric targets (qids 1-5); the other 13 statements are thresholds on three of them
        m, sd = [], []
        for b in GRID:
            r = d[f"{key}@{b}"]
            freq = {int(k): v for k, v in r["cert_qid_freq"].items()}
            per = sum(freq.get(q, 0) for q in (1, 2, 3, 4, 5)) / r["n_runs"]
            m.append(per)
            # seed-level counts are not stored; the band is omitted for this derived count
        m = np.array(m)
        sd = np.zeros_like(m)
        w = np.array([d[f"{key}@{b}"]["num_width_mean"] for b in GRID])
        axs[0].plot(GRID, m, color=col, ls=ls, marker=mk, ms=3.6, label=lab, lw=1.35)
        axs[0].fill_between(GRID, m - sd, m + sd, color=col, alpha=0.12, lw=0)
        axs[1].plot(GRID, w, color=col, ls=ls, marker=mk, ms=3.6, lw=1.35)
    star = axs[0].scatter([31482], [5], marker="*", s=70, color="#D55E00", zorder=5, clip_on=False,
                          label="label all $N$ (exact)")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xticks(GRID)
        ax.set_xticklabels(GRID_LAB)
        ax.minorticks_off()
        ax.set_xlabel("available budget cap (share of $N$)")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axs[0].set_ylabel("numeric targets certified (of 5)")
    axs[0].set_ylim(-0.15, 5.4)
    axs[0].set_title("(a)  numeric targets", loc="left", fontweight="bold")
    axs[1].set_ylabel("mean width, numeric claims")
    axs[1].set_title("(b)  interval width", loc="left", fontweight="bold")
    axs[1].set_ylim(0.08, 1.3)
    axs[1].set_yscale("log")
    axs[1].set_yticks([0.1, 0.2, 0.5, 1.0])
    axs[1].set_yticklabels(["0.1", "0.2", "0.5", "1.0"])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13), handlelength=2.6,
               columnspacing=1.4, labelspacing=0.3)
    save(fig, "fig_ladder.pdf")


# --------------------------------------------------------------------------- #
# Fig. 7  历史选择偏差：连续偏置扫描 + 真实 BM25 检索历史
# --------------------------------------------------------------------------- #
ARM_STYLE = {
    "discard": ("discard history", "#7f7f7f", ":", "o"),
    "T-reads": ("exposed, free reads", C_VARIANT, "-.", "s"),
    "E (ours)": ("known mass (COHORT)", C_COHORT, "-", "o"),
    "pooled-as-random": ("pooled as a sample", C_BAD, "--", "^"),
    "naive": ("pooled, Wald interval", "#7a1f00", (0, (3, 1, 1, 1, 1, 1)), "v"),
}


def _boot_ci(num, den, reps=2000, seed=1):
    """95% percentile bootstrap over sessions for sum(num) / sum(den)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    n = len(num)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        dd = den[idx].sum()
        vals.append(num[idx].sum() / dd if dd > 0 else 0.0)
    return np.percentile(vals, [2.5, 97.5])


def plot_history():
    d = load("exp_bias20.json") or load("exp_bias.json")
    if d is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(516 / 72, 2.58), gridspec_kw={"wspace": 0.46})
    enr = sorted({v["ratio"] for v in d.values() if v["kind"] == "enrich"})
    n_sess = next(iter(d.values()))["arms"]["E (ours)"]["n_runs"]

    def point(a, metric):
        """value and 95% interval of one arm in one setting"""
        if metric == "dec_err_rate":
            y = 100 * a["dec_err_rate"]
            if "n_dec_err_per_seed" in a:
                lo, hi = 100 * _boot_ci(a["n_dec_err_per_seed"], a["n_cert_per_seed"])
            else:
                lo = hi = y
        else:
            y = a["n_cert_mean"]
            se = 1.96 * a["n_cert_std"] / np.sqrt(max(a["n_runs"], 1))
            lo, hi = y - se, y + se
        return y, lo, hi

    for ai, (bf, metric, ylabel, title) in enumerate([
            (0.02, "dec_err_rate", "wrong certified claims (%)", "(a)  correctness, 2% of calls"),
            (0.10, "dec_err_rate", "wrong certified claims (%)", "(b)  correctness, 10% of calls"),
            (0.10, "n_cert_mean", "statements certified (of 18)", "(c)  certified, 10% of calls")]):
        ax = axs[ai]
        for arm, (lab, col, ls, mk) in ARM_STYLE.items():
            xs, ys, los, his = [], [], [], []
            for r in enr:
                key = f"enrich:{r}@{bf}"
                if key not in d or arm not in d[key]["arms"]:
                    continue
                v = d[key]
                y, lo, hi = point(v["arms"][arm], metric)
                xs.append(v["hit_hist"] / v["hit_pop"])
                ys.append(y); los.append(lo); his.append(hi)
            if xs:
                ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=3.4, lw=1.3, label=lab)
                if arm in ("pooled-as-random", "naive", "E (ours)"):
                    ax.fill_between(xs, los, his, color=col, alpha=0.13, lw=0)
            kb = f"bm25:0@{bf}"
            if kb in d and arm in d[kb]["arms"]:
                v = d[kb]
                y, lo, hi = point(v["arms"][arm], metric)
                x = v["hit_hist"] / v["hit_pop"]
                ax.errorbar([x], [y], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt="none", ecolor=col,
                            elinewidth=0.8, capsize=1.6, zorder=5)
                ax.scatter([x], [y], marker="D", s=26, facecolor="white", edgecolor=col, lw=1.2, zorder=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_xlabel("exact-match enrichment in history ($\\times$)", fontsize=8.0)
        ax.tick_params(labelsize=8.0)
        ax.set_ylabel(ylabel, fontsize=8.0)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        if metric == "dec_err_rate":
            ax.set_ylim(-2, 58)
        else:
            ax.set_ylim(6, 19)
    from matplotlib.lines import Line2D
    h, l = axs[0].get_legend_handles_labels()
    h.append(Line2D([0], [0], marker="D", ls="", markerfacecolor="white", markeredgecolor="#333333", ms=5.2))
    l.append("real lexical top-$k$ history")
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995),
               handlelength=2.8, columnspacing=1.5, labelspacing=0.45, fontsize=8.0)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.74, bottom=0.18)
    fig.savefig(os.path.join(FIG, "fig_history.pdf"))
    plt.close(fig)
    print("saved fig_history.pdf")


# --------------------------------------------------------------------------- #
# Fig. 8  四轮会话：累计调用 vs 累计认证的目标数
# --------------------------------------------------------------------------- #
def session_curves(d):
    totals = [2, 5, 18, 32]
    out = {}
    for m, runs in d["runs"].items():
        xs, ys = [], []
        for r in runs:
            t = r["traj"]
            if len(t) < 4:
                continue
            cum = [t[0]["certified"], t[1]["certified"], t[2]["certified"], t[2]["certified"] + t[3]["certified"]]
            xs.append([e["calls"] for e in t])
            ys.append(cum)
        if xs:
            out[m] = (np.array(xs, float), np.array(ys, float), len(xs))
    return out, totals


def plot_session():
    d = load("exp_session.json")
    if d is None:
        return
    curves, totals = session_curves(d)
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    style = {"COHORT (E-inherit)": ("COHORT: history becomes known mass", C_COHORT, "-", "o"),
             "COHORT-T (restart, free reads)": ("restart; history only read for free", C_VARIANT, "--", "s"),
             "EG-style": ("claim-by-claim + shared cache", "#6a3d9a", "-.", "^")}
    for m, (lab, col, ls, mk) in style.items():
        if m not in curves:
            continue
        X, Y, n = curves[m]
        ax.errorbar(X.mean(0), Y.mean(0), yerr=Y.std(0), color=col, ls=ls, marker=mk, ms=4, lw=1.3,
                    capsize=1.6, elinewidth=0.7, label=lab)
    X0, Y0 = curves["COHORT (E-inherit)"][0].mean(0), curves["COHORT (E-inherit)"][1].mean(0)
    ax.axvline(X0[2], color="#888888", ls=":", lw=0.7, zorder=0)
    for i, (name, dx, dy) in enumerate([("R1", -2, 3.2), ("R2", -2, 3.2), ("R3", -13, 1.2), ("R4", -23, -1.0)]):
        ax.text(X0[i] + dx * 55, Y0[i] + dy, name, fontsize=6.6, color="#444444", ha="center")
    ax.set_xlabel("cumulative new semantic calls")
    ax.set_ylabel("claims certified (cumulative)")
    ax.set_xlim(-150, 12000)
    ax.set_ylim(-0.8, 36)
    ax.legend(loc="upper left", frameon=False, fontsize=6.6, handlelength=2.3, labelspacing=0.3, borderaxespad=0.1)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.text(0.5, -0.02, "Round 4 new calls, both finish all 14:  known mass 2,794    free reads 3,016",
             ha="center", va="top", fontsize=6.6, color="#333333")
    save(fig, "fig_session.pdf")


# --------------------------------------------------------------------------- #
# Fig. 9  稀有分母：认证比例随分母占比与预算的变化
# --------------------------------------------------------------------------- #
def plot_denom():
    d = load("exp_denom.json")
    if d is None:
        return
    pops = sorted(d["populations"], key=lambda r: r["selectivity"])
    budgets = ["0.05", "0.1", "0.2", "0.5", "1.0"]
    M = np.array([[p["budgets"][b]["cert_rate"] for b in budgets] for p in pops])
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    im = ax.imshow(M, cmap=TEAL_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(["5%", "10%", "20%", "50%", "100%"])
    ax.set_yticks(range(len(pops)))
    ax.set_yticklabels([f"{100 * p['selectivity']:.0f}%   $\\theta$={100 * p['theta']:.0f}%" for p in pops], fontsize=6.3)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}" if 0 < v < 1 else ("1" if v >= 1 else "0"), ha="center", va="center",
                    fontsize=6.4, color=("white" if v > 0.6 else "#333333"))
    ax.set_xlabel("new calls (share of the population)")
    ax.set_ylabel("relevant share, and the exact ratio $\\theta$")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label("share of runs certified", fontsize=6.7)
    save(fig, "fig_denom.pdf")


# --------------------------------------------------------------------------- #
# Fig. 4  区域几何：为什么联合区域给出更紧的比例范围
# --------------------------------------------------------------------------- #
def plot_geometry():
    g = load("exp_geometry.json")
    if g is None:
        return
    PA, PB = np.array(g["polygon_A"]), np.array(g["polygon_B"])
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    ax.fill(PB / 1000, PA / 1000, color=C_COHORT, alpha=0.20, lw=0, zorder=2)
    ax.plot(np.r_[PB, PB[:1]] / 1000, np.r_[PA, PA[:1]] / 1000, color=C_COHORT, lw=1.0, zorder=3)
    bl, bh, al, ah = g["B_lo"] / 1000, g["B_hi"] / 1000, g["A_lo"] / 1000, g["A_hi"] / 1000
    ax.add_patch(Rectangle((bl, al), bh - bl, ah - al, fill=False, ec="#7f7f7f", ls="--", lw=0.9, zorder=2))
    xs = np.array([0, 34])
    for slope, col, ls, lab in ((g["L_sep"], "#7f7f7f", "--", None), (g["U_sep"], "#7f7f7f", "--", None),
                                (g["L_joint"], C_COHORT, "-", None), (g["U_joint"], C_COHORT, "-", None)):
        ax.plot(xs, slope * xs, color=col, ls=ls, lw=0.9, zorder=1)
    ax.scatter([g["B_true"] / 1000], [g["A_true"] / 1000], marker="*", s=60, color="#222222", zorder=6)
    ax.annotate("truth", (g["B_true"] / 1000, g["A_true"] / 1000), xytext=(4, -9), textcoords="offset points",
                fontsize=6.4)
    ax.text(0.03, 0.99, f"separate $A,B$ intervals:\n[{g['L_sep']:.2f}, {g['U_sep']:.2f}]", transform=ax.transAxes,
            fontsize=6.5, color="#555555", va="top")
    ax.text(0.03, 0.84, f"partition formula $=$ region:\n[{g['L_joint']:.2f}, {g['U_joint']:.2f}]", transform=ax.transAxes,
            fontsize=6.5, color=C_COHORT, va="top", fontweight="bold")
    ax.set_xlim(12, 31)
    ax.set_ylim(8.5, 24.5)
    ax.set_xlabel("relevant products $B$ (thousands)")
    ax.set_ylabel("relevant and rated $\\geq 4$, $A$ (thousands)")
    save(fig, "fig_geometry.pdf")


# --------------------------------------------------------------------------- #
# Fig. 6  逐个声明何时被认证（COHORT vs 逐声明基线）
# --------------------------------------------------------------------------- #
CLAIM_LABELS = ["ratio", "ratio, rated \u2265 4", "ratio, rated < 4", "share rated \u2265 4", "difference \u2265 4 \u2212 < 4"] + \
    [f"all > {t}%" for t in (10, 20, 30, 40, 50, 60, 70)] + \
    [f"{c} > {t}%" for t in (30, 50, 70) for c in ("rated \u2265 4", "rated < 4")]


def plot_paired():
    import make_paired
    make_paired.plot(make_paired.load("exp_paired.json"))


def plot_claims():
    d = load("exp_ladder.json")
    if d is None:
        return
    labels = ["exact, all", "exact, $\\geq 4$", "exact, $< 4$", "rating share", "difference"]
    meths = [("+ certificate-guided scheduler", "COHORT"),
             ("Shared pool, dedup (no polytope)", "shared samples")]
    fig = plt.figure(figsize=(252 / 72, 2.08))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.14,
                          left=0.32, right=0.88, bottom=0.34, top=0.84)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axs[1].sharey(axs[0])
    im = None
    for ax, (key, title) in zip(axs, meths):
        M = np.zeros((5, len(GRID)))
        for j, b in enumerate(GRID):
            r = d[f"{key}@{b}"]
            for qid, c in r["cert_qid_freq"].items():
                qi = int(qid) - 1
                if qi < 5:
                    M[qi, j] = c / r["n_runs"]
        im = ax.imshow(M, cmap=TEAL_CMAP, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(GRID)))
        ax.set_xticklabels(GRID_LAB, fontsize=8.0, rotation=90, ha="center", va="top")
        ax.set_title(title, fontsize=8.0, fontweight="bold", pad=6)
        ax.tick_params(axis="x", length=0, pad=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_yticks(range(5))
    axs[0].set_yticklabels(labels, fontsize=8.0)
    axs[0].tick_params(axis="y", length=0, pad=2)
    axs[1].tick_params(axis="y", labelleft=False, length=0)
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=8.0)
    cb.set_label("share of 10 runs", fontsize=8.0, labelpad=2)
    fig.text(0.55, 0.035, "available cap", ha="center", va="bottom", fontsize=8.0)
    fig.savefig(os.path.join(FIG, "fig_claims.pdf"))
    plt.close(fig)
    print("saved fig_claims.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["ambig", "reuse", "stage", "geom", "ladder", "claims", "history", "session", "denom"]
    jobs = {
        "ambig": lambda: fig_family.draw_ambiguity(FIG),     # Fig. 1  retrieval alone does not identify a fraction
        "reuse": lambda: fig_family.draw_reuse(FIG),         # Fig. 2  pointer rule + equal-budget comparison
        "stage": lambda: fig_executor.draw_executor(FIG),    # Fig. 3  one stage of COHORT
        "geom": plot_geometry,                                # Fig. 4
        "ladder": plot_ladder,                                # Fig. 5
        "claims": plot_claims,                                # Fig. 6
        "history": plot_history,                              # Fig. 7
        "session": plot_session,                              # Fig. 8
        "denom": plot_denom,                                  # Fig. 9
        "paired": plot_paired,
    }
    for name in which:
        if name not in jobs:
            raise SystemExit(f"unknown figure {name!r}; choose from {', '.join(jobs)}")
        jobs[name]()
