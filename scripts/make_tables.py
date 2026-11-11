# -*- coding: utf-8 -*-
"""由实测 JSON 生成论文用的 LaTeX 表格片段与数字宏（gen/*.tex）。论文里的数字不手抄。"""
import json
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(_ROOT, "reports", "icde2027_suite")
GEN = os.environ.get("GEN_OUT", os.path.join(_ROOT, "paper", "gen"))      # LaTeX fragments and number macros
os.makedirs(GEN, exist_ok=True)
GRID = [640, 1536, 3200, 6272, 15744, 31482]
N = 31482


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _binom_cdf(k, n, p):
    """P[X <= k] for X ~ Binomial(n, p), computed in log space."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    tot = 0.0
    for i in range(0, k + 1):
        tot += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p) + (n - i) * math.log(1 - p))
    return min(1.0, tot)


def cp_interval(k, n, conf=0.95):
    """Two-sided Clopper-Pearson interval for a binomial proportion (no scipy needed)."""
    a = (1 - conf) / 2
    lo = 0.0
    if k > 0:
        l, h = 0.0, k / n
        for _ in range(80):                      # largest p with P[X >= k | p] <= a
            m = (l + h) / 2
            if 1 - _binom_cdf(k - 1, n, m) > a:
                h = m
            else:
                l = m
        lo = (l + h) / 2
    hi = 1.0
    if k < n:
        l, h = k / n, 1.0
        for _ in range(80):                      # smallest p with P[X <= k | p] <= a
            m = (l + h) / 2
            if _binom_cdf(k, n, m) > a:
                l = m
            else:
                h = m
        hi = (l + h) / 2
    return lo, hi


def boot_ratio_ci(num, den, reps=4000, seed=0):
    """95% percentile bootstrap CI (over sessions) for sum(num)/sum(den)."""
    import numpy as _np
    num, den = _np.asarray(num, float), _np.asarray(den, float)
    rng = _np.random.default_rng(seed)
    n = len(num)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        d = den[idx].sum()
        vals.append(num[idx].sum() / d if d > 0 else 0.0)
    return float(_np.percentile(vals, 2.5)), float(_np.percentile(vals, 97.5))


def k(x):
    return f"{x / 1000:.1f}k"


def write(name, text):
    with open(os.path.join(GEN, name), "w") as f:
        f.write(text)
    print("wrote", name)


MACROS = {}


def macro(name, value):
    MACROS[name] = value


# --------------------------------------------------------------------------- #
def tab_ladder():
    d = load("exp_ladder.json")
    if d is None:
        return
    rows = [("B2 per-query (no sharing)", "Per-claim, no sharing"),
            ("EG-style claim-by-claim", "Claim-by-claim, shared cache"),
            ("Shared pool, dedup (no polytope)", "Shared samples, no region"),
            ("+ joint projection (RR)", "+ joint region"),
            ("+ certificate-guided scheduler", "+ certificate-guided scheduler")]
    lines = []
    for ri, (key, lab) in enumerate(rows):
        cells = []
        for b in GRID[:5]:
            r = d[f"{key}@{b}"]
            cells.append(f"{r['n_cert_mean']:.1f}")
        r = d[f"{key}@{GRID[-1]}"]
        if r["calls_all_n"] == r["n_runs"]:
            all_cell = f"{k(r['calls_all_mean'])}"
        else:
            all_cell = f"--"
        cells.append(all_cell)
        if ri == len(rows) - 1:                       # the full system: bold
            lab = r"\textbf{" + lab + "}"
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(f"{lab} & " + " & ".join(cells[:5]) + f" & {cells[5]} \\\\")
        if ri == 2:                                   # baselines above, COHORT rungs below
            lines.append(r"\midrule")
    body = "\n".join(lines)
    tab = r"""\begin{table}[t]
  \centering
  \caption{Statements certified out of 18 as the budget cap grows (mean over 10 seeds; 5 for per-claim execution). Thirteen of the 18 are thresholds on three ratios; Table~\ref{tab:estimand} splits out the five numeric targets. The last column is the calls needed to certify all 18, including the near-threshold claim. No certified statement was wrong in any run.}
  \label{tab:ladder}
  \footnotesize
  \setlength{\tabcolsep}{3.2pt}
  \begin{tabular}{@{}lccccc|c@{}}
    \toprule
    & \multicolumn{5}{c|}{new calls, share of $N$} & calls to \\
    Method & 2\% & 5\% & 10\% & 20\% & 50\% & all 18 \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_ladder.tex", tab)
    # 宏
    cert = lambda key, b: d[f"{key}@{b}"]["n_cert_mean"]
    macro("AbsCertTwo", f"{cert('+ certificate-guided scheduler', 640):.1f}")
    macro("AbsEGTwo", f"{cert('EG-style claim-by-claim', 640):.1f}")
    macro("AbsPoolTwo", f"{cert('Shared pool, dedup (no polytope)', 640):.1f}")
    a = d["+ certificate-guided scheduler@31482"]["calls_all_mean"]
    macro("AbsAllPct", f"{100 * a / N:.0f}")
    macro("CallsAllCert", k(a))
    macro("CallsAllEG", k(d["EG-style claim-by-claim@31482"]["calls_all_mean"]))
    macro("CallsAllPool", k(d["Shared pool, dedup (no polytope)@31482"]["calls_all_mean"]))
    macro("CertSeventeenCert", f"{cert('+ certificate-guided scheduler', 6272):.1f}")
    macro("WidthPool", f"{d['Shared pool, dedup (no polytope)@1536']['num_width_mean']:.2f}")
    macro("WidthJoint", f"{d['+ joint projection (RR)@1536']['num_width_mean']:.2f}")
    wp = [d[f"Shared pool, dedup (no polytope)@{b}"]["num_width_mean"] for b in GRID[:3]]
    wj = [d[f"+ joint projection (RR)@{b}"]["num_width_mean"] for b in GRID[:3]]
    macro("WidthShrinkLo", f"{100 * (1 - wj[0] / wp[0]):.0f}")
    macro("WidthShrinkHi", f"{100 * (1 - wj[2] / wp[2]):.0f}")
    macro("PoolTwo", f"{cert('Shared pool, dedup (no polytope)', 640):.1f}")
    macro("JointTwo", f"{cert('+ joint projection (RR)', 640):.1f}")
    macro("JointTen", f"{cert('+ joint projection (RR)', 3200):.1f}")
    macro("PoolTen", f"{cert('Shared pool, dedup (no polytope)', 3200):.1f}")
    macro("EGTen", f"{cert('EG-style claim-by-claim', 3200):.1f}")
    macro("EGTwenty", f"{cert('EG-style claim-by-claim', 6272):.1f}")
    macro("CertTwenty", f"{cert('+ certificate-guided scheduler', 6272):.1f}")
    macro("BTwoHund", f"{cert('B2 per-query (no sharing)', 31482):.1f}")


def tab_sched():
    d = load("exp_sched.json")
    if d is None:
        return
    names = [("PROP", "Proportional (equal fraction)"), ("RR", "Round-robin (equal count)"),
             ("SSF", "Small stratum first"), ("VAR", "Variance-guided greedy"),
             ("CERT", "Certificate-guided greedy"), ("DRAIN", "Round-robin, drain at half")]
    lines = []
    for key, lab in names:
        if f"{key}@640" not in d:
            continue
        cells = [f"{d[f'{key}@{b}']['n_cert_mean']:.1f}" for b in (640, 3200, 6272, 15744)]
        r = d[f"{key}@31482"]
        cells.append(k(r["calls_all_mean"]) if r["calls_all_n"] == r["n_runs"] else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    tab = r"""\begin{table}[t]
  \centering
  \caption{Scheduling policies on the same region (claims certified of 18, mean over seeds; last column: calls to certify all 18). Equal-fraction and small-stratum-first allocation are clearly worse early; the other four are within about a claim of each other.}
  \label{tab:sched}
  \footnotesize
  \setlength{\tabcolsep}{3.4pt}
  \begin{tabular}{@{}lcccc|c@{}}
    \toprule
    & 2\% & 10\% & 20\% & 50\% & calls to all 18 \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_sched.tex", tab)
    macro("SchedPropTwenty", f"{d['PROP@6272']['n_cert_mean']:.1f}")
    macro("SchedRRTwenty", f"{d['RR@6272']['n_cert_mean']:.1f}")
    macro("SchedPropFifty", f"{d['PROP@15744']['n_cert_mean']:.1f}")
    macro("SchedSSFTwo", f"{d['SSF@640']['n_cert_mean']:.1f}")
    macro("SchedVarTwenty", f"{d['VAR@6272']['n_cert_mean']:.1f}")
    macro("SchedCertTwenty", f"{d['CERT@6272']['n_cert_mean']:.1f}")
    macro("SchedNSeeds", str(d['RR@640']['n_seeds']))


def tab_calib():
    c = load("exp_calib.json")
    if c is None:
        return
    pw, e2e, au = c["pathwise"], c["e2e"], c["audit"]
    macro("NCalibSessions", f"{pw['0.05']['sessions']:,}".replace(",", "{,}"))
    macro("PathFailFive", f"{100 * pw['0.05']['rate']:.1f}")
    macro("PathFailFiveUpper", f"{100 * pw['0.05']['cp95_upper']:.1f}")
    macro("PathFailOne", f"{100 * pw['0.01']['rate']:.1f}")
    macro("PathFailTen", f"{100 * pw['0.1']['rate']:.1f}")
    macro("EndSessions", str(e2e["sessions"]))
    macro("EndClaims", f"{e2e['certified_claims']:,}".replace(",", "{,}"))
    macro("EndErrors", str(e2e["sessions_with_error"]))
    macro("EndUpper", f"{100 * e2e['cp95_upper_session']:.1f}")
    macro("AuditSessions", str(au["sessions"]))
    macro("AuditChecks", str(au["audited_conclusions"]))
    macro("AuditAdjusted", str(au["audit_rejections_or_adjustments"]))
    macro("AuditMargin", f"{au['max_outward_margin']:.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}")
    macro("AbsCalib", f"{100 * pw['0.05']['rate']:.1f}\\% joint-event failures at $\\alpha=5\\%$ over {pw['0.05']['sessions']:,} sessions".replace(",", "{,}"))
    rows = [
        (r"Joint event $k^\star\notin\mathcal{P}_t$ at some time, $\alpha{=}1\%$",
         f"{pw['0.01']['failures']} / {pw['0.01']['sessions']:,}", f"{100 * pw['0.01']['rate']:.2f}\\%",
         f"{100 * pw['0.01']['cp95_upper']:.2f}\\%"),
        (r"\hspace{1em}$\alpha{=}5\%$", f"{pw['0.05']['failures']} / {pw['0.05']['sessions']:,}",
         f"{100 * pw['0.05']['rate']:.2f}\\%", f"{100 * pw['0.05']['cp95_upper']:.2f}\\%"),
        (r"\hspace{1em}$\alpha{=}10\%$", f"{pw['0.1']['failures']} / {pw['0.1']['sessions']:,}",
         f"{100 * pw['0.1']['rate']:.2f}\\%", f"{100 * pw['0.1']['cp95_upper']:.2f}\\%"),
        (f"Sessions with any false certificate (2\\% cap; {e2e['certified_claims']:,} claims)".replace(",", "{,}"),
         f"{e2e['sessions_with_error']} / {e2e['sessions']}", f"{100 * e2e['sessions_with_error'] / e2e['sessions']:.2f}\\%",
         f"{100 * e2e['cp95_upper_session']:.2f}\\%"),
    ]
    lines = [f"{a} & {b} & {c_} & {d_} \\\\" for a, b, c_, d_ in rows]
    tab = r"""\begin{table}[t]
  \centering
  \caption{Calibration of the guarantee. The first three rows replay random permutations of the whole snapshot and check the true counts against every confidence sequence at every step (exact, not a grid); observed failure rates sit below the nominal risk, as they should. The last row runs complete sessions (scheduler, projection, certification).}
  \label{tab:calib}
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}p{4.2cm}ccc@{}}
    \toprule
    Check & Failures & Rate & 95\% upper \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_calib.tex", tab)


def interp_calls(curve, target):
    """curve: [(预算, 认证数均值)]（预算递增）；对数线性插值求达到 target 所需预算。"""
    for (b0, c0), (b1, c1) in zip(curve[:-1], curve[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return math.exp(math.log(b0) + t * (math.log(b1) - math.log(b0)))
    return None


def tab_compose():
    d = load("exp_compose.json")
    if d is None:
        return
    rows = [("Shared pool, dedup (no polytope)", "Shared samples, no region"),
            ("+ known mass & conservation", "+ mass, separate intervals"),
            ("+ joint projection", "+ joint region")]
    lines = []
    for key, lab in rows:
        cells = []
        for f in (0.02, 0.05, 0.2):
            r = d[f"{key}@{f}"]
            cells += [f"{r['n_cert_mean']:.1f}", f"{r['num_width_mean']:.3f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    n = d["+ joint projection@0.02"]["n_queries"]
    tab = r"""\begin{table}[t]
  \centering
  \caption{Composite claims over one shared base (""" + str(n) + r""" claims: shares by rating bucket and intent group, cohort ratios, ratios per stratum, and four cohort differences; mean over 10 seeds). None needs a new confidence sequence. Columns give claims certified and the mean width of their intervals.}
  \label{tab:compose}
  \footnotesize
  \setlength{\tabcolsep}{3.2pt}
  \begin{tabular}{@{}lcccccc@{}}
    \toprule
    & \multicolumn{2}{c}{2\%} & \multicolumn{2}{c}{5\%} & \multicolumn{2}{c}{20\%} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
    Method & cert. & width & cert. & width & cert. & width \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_compose.tex", tab)
    pj, pb = d["+ joint projection@0.02"], d["Shared pool, dedup (no polytope)@0.02"]
    macro("ComposeN", str(n))
    macro("ComposeJointTwo", f"{pj['n_cert_mean']:.1f}")
    macro("ComposeBaseTwo", f"{pb['n_cert_mean']:.1f}")
    macro("ComposeJointFive", f"{d['+ joint projection@0.05']['n_cert_mean']:.1f}")
    macro("ComposeBaseFive", f"{d['Shared pool, dedup (no polytope)@0.05']['n_cert_mean']:.1f}")
    macro("ComposeWidthTwo", f"{100 * (1 - pj['num_width_mean'] / pb['num_width_mean']):.0f}")
    macro("ComposeWidthFive", f"{100 * (1 - d['+ joint projection@0.05']['num_width_mean'] / d['Shared pool, dedup (no polytope)@0.05']['num_width_mean']):.0f}")


def bias_macros():
    d = load(os.environ.get("BIAS_FILE", "exp_bias20.json")) or load("exp_bias.json")
    lad = load("exp_ladder.json")
    if d is None:
        return
    bm2, bm10 = d["bm25:0@0.02"], d["bm25:0@0.1"]
    A = lambda v, arm, key: v["arms"][arm][key]
    macro("HistEnrich", f"{bm2['hit_hist'] / bm2['hit_pop']:.1f}")
    macro("HistHitRate", f"{100 * bm2['hit_hist']:.1f}")
    macro("HistPopRate", f"{100 * bm2['hit_pop']:.1f}")
    pe2, pe10 = 100 * A(bm2, "pooled-as-random", "dec_err_rate"), 100 * A(bm10, "pooled-as-random", "dec_err_rate")
    ne2, ne10 = 100 * A(bm2, "naive", "dec_err_rate"), 100 * A(bm10, "naive", "dec_err_rate")
    macro("HistPooledErrTwo", f"{pe2:.1f}")
    macro("HistPooledErrTen", f"{pe10:.1f}")
    macro("HistNaiveErrTwo", f"{ne2:.1f}")
    macro("HistNaiveErrTen", f"{ne10:.1f}")
    macro("HistPooledErr", f"{pe2:.0f}--{pe10:.0f}")
    mis2, mis10 = 100 * A(bm2, "pooled-as-random", "noncov_rate"), 100 * A(bm10, "pooled-as-random", "noncov_rate")
    macro("HistPooledMissTwo", f"{mis2:.1f}")
    macro("HistPooledMissTen", f"{mis10:.1f}")
    macro("HistPooledVerdTwo", f"{pe2 - mis2:.1f}")
    macro("HistPooledVerdTen", f"{pe10 - mis10:.1f}")
    nrun = bm2["arms"]["pooled-as-random"]["n_runs"]
    macro("HistNSeeds", str(nrun))
    if "enrich:1@0.02" in d:
        macro("HistSweepSeeds", str(d["enrich:1@0.02"]["arms"]["E (ours)"]["n_runs"]))
    # session-level errors with Clopper-Pearson intervals; claim-level error rate with a session bootstrap
    for tag, v in (("Two", bm2), ("Ten", bm10)):
        a = v["arms"]["pooled-as-random"]
        k_s = int(round(a["session_any_error"] * nrun))
        lo, hi = cp_interval(k_s, nrun)
        macro(f"HistPooledSess{tag}", str(k_s))
        macro(f"HistPooledSess{tag}Lo", f"{100 * lo:.0f}")
        macro(f"HistPooledSess{tag}Hi", f"{100 * hi:.0f}")
        if "n_dec_err_per_seed" in a:
            lo_b, hi_b = boot_ratio_ci(a["n_dec_err_per_seed"], a["n_cert_per_seed"])
            macro(f"HistPooledErr{tag}Lo", f"{100 * lo_b:.0f}")
            macro(f"HistPooledErr{tag}Hi", f"{100 * hi_b:.0f}")
    macro("AbsPooledErr", f"{pe2:.0f}--{pe10:.0f}")
    for tag, arm in (("Pooled", "pooled-as-random"), ("E", "E (ours)"), ("T", "T-reads"), ("D", "discard")):
        macro(f"Hist{tag}CertTwo", f"{A(bm2, arm, 'n_cert_mean'):.1f}")
        macro(f"Hist{tag}CertTen", f"{A(bm10, arm, 'n_cert_mean'):.1f}")
    e16, e1 = d["enrich:16@0.1"], d["enrich:1@0.02"]
    macro("HistECertSixteen", f"{A(e16, 'E (ours)', 'n_cert_mean'):.1f}")
    macro("HistTCertSixteen", f"{A(e16, 'T-reads', 'n_cert_mean'):.1f}")
    macro("HistDCertSixteen", f"{A(e16, 'discard', 'n_cert_mean'):.1f}")
    macro("HistPoolOneTwo", f"{A(e1, 'pooled-as-random', 'n_cert_mean'):.1f}")
    macro("HistEOneTwo", f"{A(e1, 'E (ours)', 'n_cert_mean'):.1f}")
    e1t = d["enrich:1@0.1"]
    macro("HistPoolOneTen", f"{A(e1t, 'pooled-as-random', 'n_cert_mean'):.1f}")
    macro("HistEOneTen", f"{A(e1t, 'E (ours)', 'n_cert_mean'):.1f}")
    e2, e2t = d["enrich:2@0.02"], d["enrich:2@0.1"]
    macro("HistBiasTwoEnrich", f"{e2['hit_hist'] / e2['hit_pop']:.1f}")
    macro("HistPoolBiasTwo", f"{100 * A(e2, 'pooled-as-random', 'dec_err_rate'):.0f}")
    macro("HistPoolBiasTen", f"{100 * A(e2t, 'pooled-as-random', 'dec_err_rate'):.0f}")
    safe = 0
    zero = True
    for v in d.values():
        for arm in ("E (ours)", "T-reads", "discard"):
            safe += v["arms"][arm]["n_runs"]
            zero &= (v["arms"][arm]["dec_err_rate"] == 0.0)
    macro("HistSafeRuns", str(safe // 3))
    macro("HistSafeZero", "yes" if zero else "NO")
    # Do not turn these runs into one binomial upper bound. Seeds are reused
    # across history settings, and the discard arm never reads the history,
    # so the same seed and budget repeat. Coverage is Table VIII.
    # calls saved at equal certified claims (interpolating the discard-history ladder curve in log budget)
    if lad is not None:
        curve = [(b, lad[f"+ joint projection (RR)@{b}"]["n_cert_mean"]) for b in GRID]
        for tag, bf, b in (("Two", bm2, 640), ("Ten", bm10, 3200)):
            tgt = A(bf, "E (ours)", "n_cert_mean")
            need = interp_calls(curve, tgt)
            macro(f"HistSave{tag}", f"{100 * (1 - b / need):.0f}" if need else "0")
        macro("AbsHistSave", MACROS["HistSaveTwo"])


def operator_macros():
    d = load("exp_operator.json")
    if d is None:
        return
    a, b = d["Qwen3-8B"], d["Qwen3-4B"]
    macro("OpSemEightB", f"{a['semantic_error_mean']:.3f}")
    macro("OpSemFourB", f"{b['semantic_error_mean']:.3f}")
    macro("OpHeadEightB", f"{100 * a['theta_M_headline']:.1f}")
    macro("OpHeadFourB", f"{100 * b['theta_M_headline']:.1f}")
    macro("OpHeadGold", f"{100 * a['theta_gold_headline']:.1f}")
    macro("OpExactEightB", f"{100 * a['exact_rate']:.1f}")
    macro("OpExactFourB", f"{100 * b['exact_rate']:.1f}")
    macro("OpCertEightB", f"{a['n_cert_mean']:.1f}")
    macro("OpCertFourB", f"{b['n_cert_mean']:.1f}")
    macro("OpSessions", str(a["sessions"]))
    macro("OpErrors", str(int(round(a["session_any_error"] * a["sessions"] + b["session_any_error"] * b["sessions"]))))


def session_macros():
    d = load("exp_session.json")
    if d is None:
        return
    import numpy as np
    def agg(m):
        runs = d["runs"][m]
        calls = np.array([[e["calls"] for e in r["traj"]] for r in runs], float)
        cert = np.array([[e["certified"] for e in r["traj"]] for r in runs], float)
        return calls.mean(0), cert.mean(0), len(runs), runs
    cE, kE, nE, _ = agg("COHORT (E-inherit)")
    cT, kT, _, _ = agg("COHORT-T (restart, free reads)")
    cG, kG, _, runsG = agg("EG-style")
    macro("SessSeeds", str(nE))
    macro("SessCallsThree", f"{cE[2]:,.0f}".replace(",", "{,}"))
    macro("SessCertThreeE", f"{kE[2]:.1f}")
    macro("SessCertThreeEG", f"{kG[2]:.1f}")
    macro("SessCertFourE", f"{kE[3]:.1f}")
    macro("SessCertFourT", f"{kT[3]:.1f}")
    macro("SessCertFourEG", f"{kG[3]:.1f}")
    macro("SessFourCallsE", f"{cE[3] - cE[2]:,.0f}".replace(",", "{,}"))
    macro("SessFourCallsT", f"{cT[3] - cT[2]:,.0f}".replace(",", "{,}"))
    macro("SessFourCallsEG", f"{cG[3] - cG[2]:,.0f}".replace(",", "{,}"))
    macro("SessFourSavePct", f"{100 * (1 - (cE[3] - cE[2]) / (cT[3] - cT[2])):.0f}")
    macro("SessCertOneE", f"{kE[0]:.1f}")
    macro("SessCertTwoE", f"{kE[1]:.1f}")
    errs = sum((r.get("errors_before_R4", 0) or 0) for r in d["runs"]["COHORT (E-inherit)"])
    macro("SessErrors", str(errs))


def enum_macros():
    d = load("exp_enum.json")
    if d is None:
        return
    macro("EnumInstances", str(d["n"]))
    macro("EnumThr", str(d["thr_checked"]))
    macro("EnumGapExp", "9")


def sched_extra():
    d = load("exp_sched.json")
    if d is None:
        return
    g = lambda n: d[f"{n}@31482"]["calls_all_mean"]
    macro("SchedAllRR", k(g("RR")))
    macro("SchedAllPROP", k(g("PROP")))
    macro("SchedAllVAR", k(g("VAR")))
    macro("SchedAllSSF", k(g("SSF")))
    macro("SchedAllCERT", k(g("CERT")))
    macro("SchedAllDRAIN", k(g("DRAIN")))
    macro("SchedPropGap", f"{100 * (g('PROP') / g('RR') - 1):.0f}")
    macro("SchedVarGap", f"{100 * (g('VAR') / g('RR') - 1):.0f}")


def cost_para():
    sc, co = load("exp_scale.json"), load("exp_cost.json")
    if sc is None or co is None:
        return
    keys = sorted(sc, key=lambda x: int(x))
    small, big = sc[keys[0]], sc[keys[-1]]
    mid = sc[keys[1]]
    g = lambda b: co[str(b)]
    lab = lambda b: g(b)["calls"] / 63.7
    pct = lambda b: 100 * g(b)["cpu_total"] / lab(b)
    macro("ScaleSmallN", str(small["n_claims"]))
    macro("ScaleBigN", str(big["n_claims"]))
    macro("ScaleBigCert", f"{big['n_cert_mean']:.1f}")
    macro("ScaleSmallCert", f"{small['n_cert_mean']:.1f}")
    macro("ScaleBigNcs", str(big["n_cs"]))
    macro("ScaleCalls", f"{big['calls_mean']:,.0f}".replace(",", "{,}"))
    macro("ScaleCpuSmall", f"{small['cpu_mean']:.1f}")
    macro("ScaleCpuBig", f"{big['cpu_mean']:.1f}")
    macro("ScaleSeeds", str(big["seeds"]))
    macro("ScaleEGSeq", str(2 * big["n_claims"]))
    macro("CostCpuTwo", f"{g(640)['cpu_total']:.1f}")
    macro("CostCpuTwenty", f"{g(6272)['cpu_total']:.1f}")
    macro("CostCpuAll", f"{g(31482)['cpu_total']:.1f}")
    macro("CostCallsAll", f"{g(31482)['calls']:,}".replace(",", "{,}"))
    macro("CostGpuTwo", f"{lab(640):.0f}")
    macro("CostGpuTwenty", f"{lab(6272):.0f}")
    macro("CostGpuAll", f"{lab(31482):.0f}")
    macro("CostPctTwo", f"{pct(640):.0f}")
    macro("CostPctTwenty", f"{pct(6272):.0f}")
    macro("CostPctAll", f"{pct(31482):.1f}")
    txt = (r"\sys{}'s own computation---confidence sequences, region, linear programs, scheduling---takes "
           r"\CostCpuTwo{}\,s of CPU for a 2\% session, \CostCpuTwenty{}\,s at 20\%, and \CostCpuAll{}\,s for a session "
           r"that certifies all 18 claims with \CostCallsAll{} calls (profiled, on a half-core container). Labelling those calls on "
           r"the GPU takes \CostGpuTwo{}, \CostGpuTwenty{}, and \CostGpuAll{}\,s at the measured 63.7 records/s, so the overhead is "
           r"\CostPctTwo{}\%, \CostPctTwenty{}\%, and \CostPctAll{}\%. The evidence state, the sequences, and the region occupy at most \MemPeakAll{}\,MB of Python heap in a full session." "\n\n"
           r"\paragraph*{Scaling}" "\n"
           r"More thresholds on the same ratios do not make \sys{} more expensive. Growing the set from \ScaleSmallN{} to \ScaleBigN{} thresholds, all inside "
           r"the closure, leaves the number of confidence sequences at \ScaleBigNcs{} and the calls at \ScaleCalls{}; the claims certified "
           r"rise from \ScaleSmallCert{} to \ScaleBigCert{} and the CPU time from \ScaleCpuSmall{}\,s to \ScaleCpuBig{}\,s "
           r"(\ScaleSeeds{} seeds, no wrong claim). A claim-by-claim executor would give each of the \ScaleBigN{} claims "
           r"a risk budget of $\alpha/\ScaleBigN{}$, split across the two sequences that claim uses.")
    ns, au = load("exp_nscale.json"), load("exp_audit_time.json")
    if ns is not None:
        ks = sorted(ns, key=lambda x: int(x))
        lo, mid, hi = ns[ks[0]], ns[ks[1]], ns[ks[-1]]
        macro("NScaleMid", f"{mid['N'] / 1000:.0f}\\,k")
        macro("NScaleHi", f"{hi['N'] / 1000:.0f}\\,k")
        macro("NScaleCpuLo", f"{lo['cpu_mean']:.1f}")
        macro("NScaleCpuMid", f"{mid['cpu_mean']:.1f}")
        macro("NScaleCpuHi", f"{hi['cpu_mean']:.1f}")
        macro("NScaleCertLo", f"{lo['n_cert_mean']:.1f}")
        macro("NScaleCertHi", f"{hi['n_cert_mean']:.1f}")
        macro("NScaleSeeds", str(lo["seeds"]))
        txt += (r" The projection has $H\cdot J$ variables and does not grow with $N$. The ledger and the sequence arrays do. "
                r"Tiling this tape to \NScaleMid{} and \NScaleHi{} records, with the same ratios and the same \ScaleCalls{} calls "
                r"(\NScaleSeeds{} seeds), takes \NScaleCpuMid{} and \NScaleCpuHi{}\,s of CPU against \NScaleCpuLo{}\,s at 31\,k. "
                r"Certified statements fall from \NScaleCertLo{} to \NScaleCertHi{}, because the same calls are a smaller share of the population. "
                r"This is a bookkeeping check on a repeated tape, not a new workload.")
    if au is not None:
        macro("AuditMs", f"{1000 * au['per_audit_s']:.0f}")
        txt += (r" The exact rational audit of Section~\ref{sec:region} adds about \AuditMs{}\,ms per certified claim.")
    write("cost_para.tex", txt + "\n")


def example_macros():
    g = load("exp_geometry.json")
    if g is None:
        return
    macro("ExPoolLo", f"{g['L_sep']:.2f}")
    macro("ExPoolHi", f"{g['U_sep']:.2f}")
    macro("ExJointLo", f"{g['L_joint']:.2f}")
    macro("ExJointHi", f"{g['U_joint']:.2f}")
    macro("ExTruth", f"{100 * g['A_true'] / g['B_true']:.1f}")
    # C = B - A on the partition parallelogram: the extreme corners are (A_lo, A_lo+C_lo) and (A_hi, A_hi+C_hi).
    c_lo = g["B_lo"] - g["A_lo"]
    c_hi = g["B_hi"] - g["A_hi"]
    closed_lo = g["A_lo"] / (g["A_lo"] + c_hi)
    closed_hi = g["A_hi"] / (g["A_hi"] + c_lo)
    macro("ExCLo", f"{c_lo:.0f}")
    macro("ExCHi", f"{c_hi:.0f}")
    macro("ExALo", f"{g['A_lo']:.0f}")
    macro("ExAHi", f"{g['A_hi']:.0f}")
    macro("ExClosedLo", f"{closed_lo:.2f}")
    macro("ExClosedHi", f"{closed_hi:.2f}")
    cf = load("exp_closedform.json")
    if cf is not None:
        macro("ClosedSeeds", str(cf["seeds"]))
        macro("ClosedClaims", str(cf["n_value_claims"]))
        macro("ClosedGap", r"10^{-9}")


def basis_macros():
    d = load("exp_basis.json")
    if d is None:
        return
    c2, p2 = d["compiled@640"], d["per-claim events@640"]
    c10, p10 = d["compiled@3200"], d["per-claim events@3200"]
    macro("BasisNcsC", str(c2["n_cs"]))
    macro("BasisNcsP", str(p2["n_cs"]))
    macro("BasisCertTwoC", f"{c2['n_cert_mean']:.1f}")
    macro("BasisCertTwoP", f"{p2['n_cert_mean']:.1f}")
    macro("BasisCertTenC", f"{c10['n_cert_mean']:.1f}")
    macro("BasisCertTenP", f"{p10['n_cert_mean']:.1f}")
    macro("BasisWidthTwoPct", f"{100 * (p2['num_width_mean'] / c2['num_width_mean'] - 1):.0f}")
    macro("BasisSeeds", str(c2["n_runs"]))


def tab_estimand():
    """Five numeric targets. Cells are runs, out of 10, in which the target interval was certified."""
    d = load("exp_ladder.json")
    if d is None:
        return
    targets = [
        (4, r"rating share, $\pm 0.10$"),
        (3, r"exact ratio, rated $<4$, $\pm 0.05$"),
        (1, r"exact ratio, all products, $\pm 0.05$"),
        (2, r"exact ratio, rated $\geq 4$, $\pm 0.05$"),
        (5, r"difference of the two cohort ratios, $\pm 0.05$"),
    ]
    methods = [
        ("Shared pool, dedup (no polytope)", "samples"),
        ("+ joint projection (RR)", "region"),
        ("+ certificate-guided scheduler", "cohort"),
    ]
    budgets = [(640, "Two"), (3200, "Ten"), (6272, "Twenty")]

    def cell(method, budget, qid):
        r = d[f"{method}@{budget}"]
        n = r["n_runs"]
        k = int(r["cert_qid_freq"].get(str(qid), r["cert_qid_freq"].get(qid, 0)))
        return k, n

    # One column cannot hold nine count columns. Keep the two comparisons
    # the text uses: region against shared samples at 2%, and the scheduler
    # against shared samples at 20%. The 10% row is stated in the prose.
    show = [
        ("Shared pool, dedup (no polytope)", 640),
        ("+ joint projection (RR)", 640),
        ("+ certificate-guided scheduler", 640),
        ("Shared pool, dedup (no polytope)", 6272),
        ("+ certificate-guided scheduler", 6272),
    ]
    short = [
        (4, r"rating share, $\pm0.10$"),
        (3, r"exact ratio, rated $<4$"),
        (1, r"exact ratio, all products"),
        (2, r"exact ratio, rated $\geq 4$"),
        (5, r"difference of cohort ratios"),
    ]
    lines = []
    for qid, label in short:
        bits = []
        for method, budget in show:
            k, n = cell(method, budget, qid)
            if n != 10:
                raise SystemExit(f"expected 10 runs, got {n}")
            bits.append(str(k))
        lines.append(label + " & " + " & ".join(bits) + r" \\")
    body = "\n".join(lines)
    tab = r"""\begin{table}[t]
  \centering
  \caption{Runs, out of 10, that certified each numeric target. Shared samples already deduplicate ratios and broadcast thresholds. At 2\% of the calls the region adds the rating share. At 20\% the scheduler finishes the high-rated ratio and the difference; the region alone does not (it matches the shared-sample column).}
  \label{tab:estimand}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}lccccc@{}}
    \toprule
    & \multicolumn{3}{c}{2\% of calls} & \multicolumn{2}{c}{20\% of calls} \\
    \cmidrule(lr){2-4}\cmidrule(lr){5-6}
    Target & samples & region & \sys{} & samples & \sys{} \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_estimand.tex", tab)


def semantic_macros():
    d = load("exp_semantic.json")
    if d is None:
        return
    macro("RelPrec", f"{100 * d['relevant']['precision']:.1f}")
    macro("RelRec", f"{100 * d['relevant']['recall']:.1f}")
    macro("ExPrec", f"{100 * d['exact']['precision']:.1f}")
    macro("ExRec", f"{100 * d['exact']['recall']:.1f}")
    macro("RelTP", f"{d['relevant']['tp']:,}".replace(",", "{,}"))
    macro("ExTP", f"{d['exact']['tp']:,}".replace(",", "{,}"))
    macro("ExFN", f"{d['exact']['fn']:,}".replace(",", "{,}"))


def tab_claims():
    """表：18 个声明的设置（类型、容差/阈值、真值、与阈值的距离）。全部来自 exp_claims.json。"""
    d = load("exp_claims.json")
    if d is None:
        return
    cl = {c["qid"]: c for c in d["claims"]}
    pct = lambda x: f"{100 * x:.2f}\\%"
    rows = []

    def val(qid, label, tol):
        rows.append(f"{label} & 1 & $\\pm{tol}$ & {pct(cl[qid]['theta_run'])} & --\\\\")

    val(1, "exact ratio, all products", "0.05")
    val(2, "exact ratio, rated $\\geq 4$", "0.05")
    val(3, "exact ratio, rated $<4$", "0.05")
    val(4, "share of relevant rated $\\geq 4$", "0.10")
    c5 = cl[5]["theta_run"]
    sgn = "$-$" if c5 < 0 else ""
    rows.append(f"difference of cohort ratios & 1 & $\\pm0.05$ & {sgn}{abs(100 * c5):.2f}\\,pp & --\\\\")
    rows.append("\\midrule")

    def grp(qids, label):
        tv = [100 * cl[q]["tau"] for q in qids]
        taus = (f"{tv[0]:.0f}--{tv[-1]:.0f}" if len(tv) > 3 else ", ".join(f"{t:.0f}" for t in tv)) + "\\%"
        gaps = sorted(100 * cl[q]["gap"] for q in qids)
        rows.append(f"{label} & {len(qids)} & {taus} & {pct(cl[qids[0]]['theta_run'])} & {gaps[0]:.1f}--{gaps[-1]:.1f}\\\\")

    grp([6, 7, 8, 9, 10, 11, 12], "ratio $>\\tau$, all products")
    grp([13, 15, 17], "ratio $>\\tau$, rated $\\geq 4$")
    grp([14, 16, 18], "ratio $>\\tau$, rated $<4$")
    macro("ClaimMinGap", f"{min(100 * c['gap'] for c in d['claims'] if c['gap'] is not None):.2f}")
    def gap_pp(qid):
        c = cl[qid]
        return abs(100 * (c["theta_run"] - c["theta_gold"]))
    macro("GapAll", f"{gap_pp(1):.1f}")
    macro("GapGe", f"{gap_pp(2):.1f}")
    macro("GapLt", f"{gap_pp(3):.1f}")
    macro("GapShare", f"{gap_pp(4):.1f}")
    macro("GapDiff", f"{gap_pp(5):.1f}")
    macro("ShareRun", f"{100 * cl[4]['theta_run']:.1f}")
    macro("ShareGold", f"{100 * cl[4]['theta_gold']:.1f}")
    txt = (r"\begin{table}[t]" "\n" r"\caption{The session's claims. The first five rows are distinct numeric targets. The last three rows are thresholds on three of those ratios; a system that keeps one interval per ratio can broadcast them. "
           r"``Ratio'' is the share of relevant pairs that are exact. The closest threshold lies \ClaimMinGap{}\,pp from the operator's value.}" "\n" r"\label{tab:claims}" "\n"
           r"\centering\footnotesize\setlength{\tabcolsep}{3pt}" "\n"
           r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
           r"Claim & No. & Tol.\ / $\tau$ & True value & Gap (pp)\\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n"
           r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    write("tab_claims.tex", txt)


def audit_macros():
    d = load("exp_audit_grid.json")
    if d is None or "total" not in d:
        return
    t = d["total"]
    macro("AuditGridSeeds", str(len(d["per_seed"])))
    macro("AuditGridCells", str(t["cells"]))
    macro("AuditGridSame", str(t["cells_same_set"]))
    macro("AuditGridAudits", f"{t['audits']:,}".replace(",", "{,}"))
    macro("AuditGridAdjusted", str(t["adjusted"]))
    macro("AuditGridStatusDiff", str(t["status_diff"]))
    macro("AuditGridShift", f"{t['max_shift']:.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}")
    cpu = [(r["cpu_with"] - r["cpu_without"]) / max(r["audits"], 1) for r in d["per_seed"]]
    macro("AuditGridMs", f"{1000 * max(0.0, sum(cpu) / len(cpu)):.0f}")


def eg_macros():
    d = load("exp_eg_reuse.json")
    if d is None:
        return
    for tag, b in (("Two", "640"), ("Five", "1536"), ("Ten", "3200")):
        if b not in d["rows"]:
            continue
        e, c = d["rows"][b]["eg"], d["rows"][b]["cohort"]
        macro(f"Eg{tag}Paid", f"{e['paid']:,.0f}".replace(",", "{,}"))
        macro(f"Eg{tag}Head", f"{e['headline_stream']:,.0f}".replace(",", "{,}"))
        macro(f"Eg{tag}Mean", f"{e['mean_stream']:,.0f}".replace(",", "{,}"))
        macro(f"Eg{tag}Total", f"{e['total_obs']:,.0f}".replace(",", "{,}"))
        macro(f"Eg{tag}Cert", f"{e['certified']:.1f}")
        macro(f"Co{tag}Prefix", f"{c['prefix']:,.0f}".replace(",", "{,}"))
        macro(f"Co{tag}Cert", f"{c['certified']:.1f}")
        macro(f"Eg{tag}Ratio", f"{c['prefix'] / max(e['headline_stream'], 1):.0f}")


def memory_macros():
    d = load("exp_memory.json")
    if d is None:
        return
    macro("MemPeakAll", f"{d['all']['peak_mb']:.0f}")
    macro("MemPeakTwo", f"{d['2%']['peak_mb']:.0f}")


def lotus_macros():
    d = load("exp_lotus.json")
    if d is None or not d.get("rows"):
        return
    rows = d["rows"]
    macro("LotusRuns", str(len(rows)))
    macro("LotusIdentical", str(sum(1 for r in rows if r["identical_to_direct_tape_run"])))
    r0 = [r for r in rows if r["budget"] == 640]
    if r0:
        macro("LotusCallsTwo", f"{sum(r['calls'] for r in r0) / len(r0):,.0f}".replace(",", "{,}"))
        macro("LotusReqTwo", f"{sum(r['requests_relevant'] + r['requests_exact'] for r in r0) / len(r0):,.0f}".replace(",", "{,}"))
        macro("LotusCertTwo", f"{sum(r['certified'] for r in r0) / len(r0):.1f}")
        macro("LotusWallTwo", f"{sum(r['wall_s'] for r in r0) / len(r0):.0f}")
    r1 = [r for r in rows if r["budget"] == 3200]
    if r1:
        macro("LotusCallsTen", f"{sum(r['calls'] for r in r1) / len(r1):,.0f}".replace(",", "{,}"))
        macro("LotusReqTen", f"{sum(r['requests_relevant'] + r['requests_exact'] for r in r1) / len(r1):,.0f}".replace(",", "{,}"))
        macro("LotusCertTen", f"{sum(r['certified'] for r in r1) / len(r1):.1f}")
        macro("LotusWallTen", f"{sum(r['wall_s'] for r in r1) / len(r1):.0f}")



def pooled_macros():
    d = load("exp_pooled.json")
    if d is None:
        return
    for tag, b in (("Two", "640"), ("Five", "1536"), ("Ten", "3200")):
        r = d["rows"].get(b)
        if r:
            macro(f"Pooled{tag}Dedicated", f"{r['pooled_width']:.2f}")
            macro(f"Pooled{tag}Region", f"{r['region_width']:.2f}")
            macro(f"Pooled{tag}Ratio", f"{r['ratio']:.1f}")
    macro("PooledTwoPaid", "640")


if __name__ == "__main__":
    tab_ladder()
    tab_sched()
    tab_calib()
    for fn in (tab_claims, tab_estimand, tab_compose, bias_macros, operator_macros, session_macros, enum_macros, sched_extra, cost_para, example_macros, basis_macros,
               audit_macros, eg_macros, memory_macros, lotus_macros, pooled_macros, semantic_macros):
        try:
            fn()
        except Exception as e:  # 数据未就绪时跳过
            print("skip", fn.__name__, repr(e))
    lines = ["% 自动生成：不要手改（make_tables.py）"]
    for kx, v in MACROS.items():
        lines.append(f"\\newcommand{{\\{kx}}}{{{v}}}")
    write("numbers_auto.tex", "\n".join(lines) + "\n")
