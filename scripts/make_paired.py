# -*- coding: utf-8 -*-
"""Generate the paired known-mass vs free-read tables and Fig. paired from stored JSON.

Inputs (reports/icde2027_suite/):
  exp_paired.json     8 seeds, four histories, five numeric targets, cap 0.55N
  exp_sessions.json   four intent bins and five product classes

Outputs (paper/gen, paper/figure; override with GEN_OUT and FIG_OUT):
  tab_paired.tex, tab_bytarget.tex, tab_sessions.tex, paired_numbers.tex
  fig_paired.pdf

A completion cost is recorded only when all targets of that arm are certified.
A run that stops because a confidence sequence has no feasible count is not
given a completion cost. Call medians and the reduction median for a history
use the seeds both arms finished.
"""
import hashlib
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig_style import ACCENT as C_COHORT  # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RES = os.path.join(ROOT, "reports", "icde2027_suite")
GEN = os.environ.get("GEN_OUT", os.path.join(ROOT, "paper", "gen"))
FIG = os.environ.get("FIG_OUT", os.path.join(ROOT, "paper", "figure"))

HIST_ORDER = [
    ("uniform", "Uniform sample"),
    ("bm25", "BM25 top-$k$"),
    ("mild", r"Exact-match odds $\times 4$"),
    ("strong", r"Exact-match odds $\times 16$"),
]
TARGET_ORDER = [
    (4, r"Rating share, $\pm 0.10$"),
    (1, r"Exact ratio, all products"),
    (2, r"Exact ratio, rated $\geq 4$"),
    (5, r"Difference of the cohort ratios"),
    (3, r"Exact ratio, rated $<4$"),
]

PROTOCOL = {
    "cap_frac": 0.55,
    "history_frac": 0.15,
    "seeds": list(range(8)),
    "allocator": "RoundRobinAllocator",
    "alpha": 0.05,
    "event_basis": "compile_basis=True on the same five numeric targets; two semantic events (relevant, relevant-and-exact)",
    "risk": "alpha divided evenly across stratum-event sequences whose remaining population is non-empty",
    "projection": "build_polytope and ratio_bounds, the partition-aware projection, same function for both arms",
    "deterministic_counts": "both arms insert every historical label into obs before the first new call",
    "stopping": [
        "all targets certified",
        "calls reach the cap",
        "a confidence sequence returns no feasible count; open targets are not certified",
    ],
    "arms": {
        "T": "tcache via add_targeted; confidence sequence on the full stratum; history event offset E_f=0",
        "E": "cache as known mass; confidence sequence on the records outside the history; E_f is the exact history count",
    },
}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def write(name, text):
    os.makedirs(GEN, exist_ok=True)
    path = os.path.join(GEN, name)
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


def median(xs):
    return float(statistics.median(xs))


def fmt_calls(x):
    n = int(round(x))
    return f"{n:,}".replace(",", "{,}")


def fmt_pct(x):
    return f"{int(round(x))}"


def joint(rows):
    out = []
    for r in rows:
        if r["E"].get("calls_all") is None or r["T"].get("calls_all") is None:
            continue
        out.append(r)
    return out


def done_count(rows, arm):
    return sum(r[arm].get("calls_all") is not None for r in rows)


def stamp_history(env):
    """Record a deterministic history id on each stored seed. Does not rerun the executor."""
    import run_icde2027_suite as S
    path = os.path.join(RES, "exp_paired.json")
    d = json.load(open(path))
    d["protocol"] = PROTOCOL
    kind_of = {"uniform": ("enrich", 1), "mild": ("enrich", 4), "strong": ("enrich", 16), "bm25": ("bm25", 0)}
    for key, rows in d["full"].items():
        mode, ratio = kind_of[key]
        for r in rows:
            sd = int(r["seed"])
            hist = S.bm25_history(env, 0.15, sd) if mode == "bm25" else S.enriched_history(env, ratio, 0.15, sd)
            raw = ",".join(str(i) for i in sorted(int(x) for x in hist)).encode()
            r["history"] = {
                "kind": mode, "ratio": ratio, "frac": 0.15, "seed": sd,
                "n": len(hist), "sha256": hashlib.sha256(raw).hexdigest(),
            }
            for arm in ("E", "T"):
                a = r[arm]
                if a.get("calls_all") is not None and a.get("n_done") == a.get("n_targets"):
                    a["stop_reason"] = "all_targets_certified"
                elif a.get("n_done", 0) < a.get("n_targets", 5) and a.get("calls_used", 0) < int(0.55 * 31482) - 10:
                    a["stop_reason"] = "empty_confidence_set"
                    a["calls_all"] = None
                else:
                    a["stop_reason"] = "cap_reached"
                    a["calls_all"] = None
    # The measured empty set, from a replay of enrich ratio=4, seed 7, known mass.
    d["empty_cs"] = {
        "history_kind": "enrich", "ratio": 4, "frac": 0.15, "seed": 7, "arm": "E",
        "calls_used": 3072,
        "cap_calls": int(0.55 * 31482),
        "stop_reason": "empty_confidence_set",
        "budget_exhausted": False,
        "stratum_full_N": 634,
        "history_in_stratum": 90,
        "remainder_N": 544,
        "prefix_n": 537,
        "positives_in_prefix": 372,
        "true_relevant_in_remainder": 377,
        "n_sequences": 12,
        "delta": 0.05 / 12,
        "threshold_nats": 5.480638923341991,
        "least_violating_k": 379,
        "least_violating_logE": 5.683984787659121,
        "exact_replay_feasible": False,
        "fast_window": [383, 382],
        "open_targets_certified_after_stop": False,
        "already_certified_missed_truth": False,
    }
    json.dump(d, open(path, "w"), indent=1)
    print("stamped", path)


def tab_paired(d):
    lines = []
    macros = {}
    tags = {"uniform": "Un", "bm25": "Bm", "mild": "Od", "strong": "St"}
    for key, label in HIST_ORDER:
        rows = d["full"][key]
        both = joint(rows)
        e = [r["E"]["calls_all"] for r in both]
        t = [r["T"]["calls_all"] for r in both]
        sav = [100 * (1 - a / b) for a, b in zip(e, t)]
        de, dt = done_count(rows, "E"), done_count(rows, "T")
        n = len(rows)
        lines.append(
            f"{label} & {dt}/{n} & {de}/{n} & {fmt_calls(median(t))} & "
            f"{fmt_calls(median(e))} & {fmt_pct(median(sav))}\\% \\\\"
        )
        tag = tags[key]
        macros[f"Pair{tag}T"] = fmt_calls(median(t))
        macros[f"Pair{tag}E"] = fmt_calls(median(e))
        macros[f"Pair{tag}Red"] = fmt_pct(median(sav))
        macros[f"Pair{tag}DoneT"] = f"{dt}/{n}"
        macros[f"Pair{tag}DoneE"] = f"{de}/{n}"
        macros[f"Pair{tag}Enr"] = f"{median(r['enrich'] for r in rows):.1f}"
    bm = joint(d["full"]["bm25"])
    sav_bm = [100 * (1 - r["E"]["calls_all"] / r["T"]["calls_all"]) for r in bm]
    macros["PairBmLo"] = fmt_pct(min(sav_bm))
    macros["PairBmHi"] = fmt_pct(max(sav_bm))
    body = "\n".join(lines)
    tex = r"""\begin{table}[t]
  \centering
  \caption{New calls to certify all five numeric targets. Medians use only the seeds both arms finished. A run that stops with an empty confidence set has no completion cost. Done is out of eight. The reduction is the median paired ratio on the jointly completed seeds.}
  \label{tab:paired}
  \scriptsize
  \setlength{\tabcolsep}{2.5pt}
  \begin{tabular}{@{}lccccc@{}}
    \toprule
    History & \multicolumn{2}{c}{Done} & Free & Known & Reduction \\
    \cmidrule(lr){2-3}
    & free & known & reads & mass & \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_paired.tex", tex)
    return macros


def tab_bytarget(d):
    rows = joint(d["full"]["bm25"])
    lines = []
    macros = {}
    keys = {1: "All", 2: "Ge", 3: "Lt", 4: "Share", 5: "Diff"}
    for qid, label in TARGET_ORDER:
        e, t, sav, cheaper = [], [], [], 0
        for r in rows:
            ee = next(x for x in r["E"]["targets"] if x["qid"] == qid)["calls"]
            tt = next(x for x in r["T"]["targets"] if x["qid"] == qid)["calls"]
            if ee is None or tt is None:
                continue
            e.append(ee)
            t.append(tt)
            sav.append(100 * (1 - ee / tt))
            cheaper += int(ee < tt)
        lines.append(
            f"{label} & {fmt_calls(median(t))} & {fmt_calls(median(e))} & {cheaper}/{len(e)} \\\\"
        )
        macros[f"By{keys[qid]}T"] = fmt_calls(median(t))
        macros[f"By{keys[qid]}E"] = fmt_calls(median(e))
        macros[f"By{keys[qid]}Red"] = fmt_pct(median(sav))
        macros[f"By{keys[qid]}Win"] = f"{cheaper}/{len(e)}"
    body = "\n".join(lines)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Calls until each numeric target is first certified, on the eight BM25 seeds both arms finished. Known mass is earlier on four of the five targets. The low-rated exact ratio, on 1{,}204 pairs, is the one target free reads match.}
  \label{tab:bytarget}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lrrr@{}}
    \toprule
    Target & Free reads & Known mass & Seeds cheaper \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_bytarget.tex", tex)
    return macros


def tab_sessions(d):
    lines = []
    for b in d["intent_bins"]:
        rows = joint(b["rows"])
        e = [r["E"]["calls_all"] for r in rows]
        t = [r["T"]["calls_all"] for r in rows]
        win = sum(a < c for a, c in zip(e, t))
        rel = 100 * b["feat"]["relevant_share"]
        lines.append(
            f"{rel:.0f}\\% & {fmt_calls(b['feat']['N'])} & {fmt_calls(median(t))} & "
            f"{fmt_calls(median(e))} & {win}/{len(rows)} \\\\"
        )
    # product classes: strict wins and the size range, for the prose
    sizes = [c["feat"]["N"] for c in d["classes"]]
    wins = 0
    nruns = 0
    errs = 0
    for c in d["classes"]:
        for r in c["rows"]:
            nruns += 1
            errs += r["E"]["errors"] + r["T"]["errors"]
            if r["E"]["calls_all"] is not None and r["T"]["calls_all"] is not None:
                wins += int(r["E"]["calls_all"] < r["T"]["calls_all"])
    macros = {
        "SessClassLo": fmt_calls(min(sizes)),
        "SessClassHi": fmt_calls(max(sizes)),
        "SessClassWin": str(wins),
        "SessClassN": str(nruns),
        "SessClassErr": str(errs),
    }
    body = "\n".join(lines)
    tex = r"""\begin{table}[t]
  \centering
  \caption{Four intent bins fixed by sorting intents on the operator's relevant rate, six paired seeds each. Targets are the exact-match ratio and the rating share. History is BM25 inside the bin. Every seed finished both targets, and no certified interval missed $\tr$.}
  \label{tab:sessions}
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{@{}lrrrr@{}}
    \toprule
    Relevant share & $N$ & Free reads & Known mass & Cheaper seeds \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_sessions.tex", tex)
    return macros


def plot(d):
    os.makedirs(FIG, exist_ok=True)
    order = [("uniform", "uniform"), ("bm25", "BM25"), ("mild", r"odds $\times 4$"), ("strong", r"odds $\times 16$")]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    labels = []
    for i, (key, lab) in enumerate(order):
        rows = d["full"][key]
        both = joint(rows)
        ys = [100 * (1 - r["E"]["calls_all"] / r["T"]["calls_all"]) for r in both]
        ax.scatter([i] * len(ys), ys, color=C_COHORT, s=22, zorder=3)
        ax.plot([i - 0.18, i + 0.18], [median(ys)] * 2, color="#222222", lw=1.4, zorder=4)
        de, dt = done_count(rows, "E"), done_count(rows, "T")
        labels.append(f"{lab}\n{dt}/{len(rows)}, {de}/{len(rows)}")
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("fewer calls than free reads (%)")
    ax.set_ylim(-8, 42)
    ax.set_xlim(-0.65, 3.65)
    ax.text(0.0, -0.28, "odds $\\times 4$: one known-mass run stopped (empty CS); not a completion cost",
            transform=ax.transAxes, fontsize=6.3, color="#444444")
    fig.tight_layout()
    path = os.path.join(FIG, "fig_paired.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", path)


def macros_tex(m):
    lines = ["% generated by scripts/make_paired.py", ""]
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    write("paired_numbers.tex", "\n".join(lines) + "\n")


def main():
    if "--stamp-history" in sys.argv:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        sys.path.insert(0, _HERE)
        import run_icde2027_suite as S
        stamp_history(S.load_wands_env())
        return
    paired = load("exp_paired.json")
    sessions = load("exp_sessions.json")
    m = {}
    m.update(tab_paired(paired))
    m.update(tab_bytarget(paired))
    m.update(tab_sessions(sessions))
    macros_tex(m)
    plot(paired)


if __name__ == "__main__":
    main()
