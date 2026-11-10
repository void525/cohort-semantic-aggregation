# -*- coding: utf-8 -*-
"""ICDE 2027 补充实验套件（真实 WANDS 快照 + 冻结标签带，纯 CPU 重放，不调用 GPU）。

子实验（--exp）
  ladder   强基线阶梯：每一级只增加一样东西（共享缓存/共享样本/已知质量/联合投影/调度）
  sched    调度器研究：PROP / RR / SSF / VAR / CERT / DRAIN
  bias     历史选择偏差：连续偏置扫描 + BM25 真实检索历史；arms = discard/T-reads/E(ours)/pooled/naive
  denom    未知分母稀疏度扫描：相关率从 0.04 到 1.0 的人群
  session  四轮交互会话 + 第 4 轮换分层（跨阶段继承）
  calib    独立会话级校准：路径级（数千会话）+ 端到端（200 会话）+ 精确审计一致性

约定
  * 所有方法共享同一个 CS 原语（无放回超几何混合鞅）与同一个联合风险合同 α=0.05；
  * 单线程、可限流（COHORT_DUTY 环境变量，默认 0.75），避免占满单核 CPU；
  * 每个 (方法, 预算) 单元完成后立即写盘（JSON），中断可恢复。
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import run_real_data as rrd
import run_ablations as RA
from cohortrag import (Snapshot, COHORTRAG, Config, RoundRobinAllocator,
                       WorstVarianceAllocator, CertificateGuidedAllocator,
                       SmallStratumFirstAllocator, DrainAwareRoundRobinAllocator,
                       ProportionalAllocator, FinitePopulationCS, states)
from cohortrag.pipeline import QueryResult
from cohortrag.budget import BudgetContext
from cohortrag.workload import make_wands_closure_workload, Query
import cohortrag.pipeline as P

OUT_DIR = os.path.join(_HERE, "..", "reports", "icde2027_suite")
# WANDS is not redistributed: clone it to data/wands (see README) or point WANDS_DIR at a checkout.
WANDS_DIR = os.environ.get("WANDS_DIR", os.path.join(_HERE, "..", "data", "wands"))
MANIFEST = os.path.join(_HERE, "..", "data_manifests", "wands.json")
os.makedirs(OUT_DIR, exist_ok=True)
DUTY = float(os.environ.get("COHORT_DUTY", "0.75"))
ALPHA = 0.05
SEEDS = list(range(10))
FRACS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

SCHEDULERS = {
    "PROP": ProportionalAllocator, "RR": RoundRobinAllocator,
    "SSF": SmallStratumFirstAllocator, "VAR": WorstVarianceAllocator,
    "CERT": CertificateGuidedAllocator, "DRAIN": DrainAwareRoundRobinAllocator,
}


def throttle(elapsed):
    """限流：按占空比 DUTY 在每个计算单元后睡眠，避免长时间占满单核。"""
    if DUTY < 1.0:
        time.sleep(min(3.0, elapsed * (1.0 / DUTY - 1.0)))


def dump(name, obj):
    path = os.path.join(OUT_DIR, name)
    tmp = path + ".tmp"

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
    os.replace(tmp, path)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# =========================================================================== #
# 环境
# =========================================================================== #
def load_wands_env():
    data_dir = os.path.join(WANDS_DIR, "dataset")
    manifest = MANIFEST
    model = "/root/autodl-tmp/models/Qwen3-8B"      # an identifier: it is part of the cache key kappa, never loaded on replay
    ns = argparse.Namespace(
        dataset="wands", data=data_dir, manifest=manifest, max_intents=120,
        start="2024-01-01", end="2024-12-31", max_records=50000,
        id_col="Complaint ID", date_col="Date received",
        narrative_col="Consumer complaint narrative", product_col="Product",
        dry_run=False, model=model, revision=None, device="cpu", batch_size=16)
    frame, gold, meta = rrd._real_frames(ns)
    snap = Snapshot(frame, name="wands-frozen")
    contract, kappa = rrd._load_contract(ns, snap, gold)
    fb = frame["rating_bucket"].astype(str)
    by_rating = {b: sorted(frame.loc[fb == b, "stratum"].unique().tolist()) for b in fb.unique()}
    wl = make_wands_closure_workload(by_rating, {"all": meta["all_strata"]}, K=2)
    for q in wl:
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    ids_all = frame["id"].to_numpy()
    stratum_of = dict(zip(frame["id"].astype(int), frame["stratum"].astype(int)))
    full_lab = contract.judge(list(ids_all), snap.texts(list(ids_all)))
    labels_M = {int(i): full_lab[r] for r, i in enumerate(ids_all)}
    theta_M = {q.qid: rrd._query_truth(q, labels_M, ids_all, stratum_of) for q in wl}
    theta_G = {q.qid: rrd._query_truth(q, gold["labels"], ids_all, stratum_of) for q in wl}
    return {"frame": frame, "snap": snap, "contract": contract, "kappa": kappa, "gold": gold,
            "meta": meta, "by_rating": by_rating, "wl": wl, "ids_all": ids_all,
            "stratum_of": stratum_of, "labels_M": labels_M, "theta_M": theta_M,
            "theta_G": theta_G, "N": len(frame)}


def fresh_workload(env):
    """每次运行用新的 Query 对象（pipeline 会在对象上写表）。"""
    wl = make_wands_closure_workload(env["by_rating"], {"all": env["meta"]["all_strata"]}, K=2)
    for q in wl:
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    return wl


# =========================================================================== #
# 评测
# =========================================================================== #
def evaluate(res, wl, theta_M, calls, cap):
    """返回一次运行的紧凑记录。

    错误认证判定：阈值查询给出与 θ_M 相反的结论；数值/差异查询的区间不含 θ_M。
    区间非覆盖：已认证的数值/差异区间不含 θ_M。
    """
    n_cert = n_dec_err = n_noncov = 0
    cert_qids, widths = [], []
    calls_all = None
    for q in wl:
        r = res[q.qid]
        th = theta_M[q.qid]
        if q.kind in ("value", "difference"):
            lo, hi = (r.L, r.U) if (r.L is not None and r.U is not None) else \
                ((0.0, 1.0) if q.kind == "value" else (-1.0, 1.0))
            widths.append(hi - lo)
        if r.status.startswith("certified"):
            n_cert += 1
            cert_qids.append(q.qid)
            inside = (r.L - 1e-9 <= th <= r.U + 1e-9)
            if q.kind == "threshold":
                bad = (r.status == "certified_true" and not (th > q.tau)) or \
                      (r.status == "certified_false" and not (th <= q.tau))
                n_dec_err += int(bad)
            else:
                n_dec_err += int(not inside)
                n_noncov += int(not inside)
    if n_cert == len(wl):
        ca = [res[q.qid].calls_at_cert for q in wl if res[q.qid].calls_at_cert is not None]
        calls_all = max(ca) if ca else calls
    return {"calls": int(calls), "n_cert": n_cert, "n_dec_err": n_dec_err,
            "n_noncov": n_noncov, "num_width": float(np.mean(widths)) if widths else None,
            "cert_qids": cert_qids, "calls_all": calls_all,
            "any_error": int(n_dec_err > 0 or n_noncov > 0)}


def agg(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return (float(np.mean(v)) if v else None, float(np.std(v)) if v else None, len(v))


def summarize(rows):
    n_cert_tot = sum(r["n_cert"] for r in rows)
    out = {"n_runs": len(rows)}
    for k in ("calls", "n_cert", "num_width", "calls_all"):
        m, s, n = agg(rows, k)
        out[k + "_mean"], out[k + "_std"], out[k + "_n"] = m, s, n
    out["dec_err_rate"] = (sum(r["n_dec_err"] for r in rows) / n_cert_tot) if n_cert_tot else 0.0
    out["noncov_rate"] = (sum(r["n_noncov"] for r in rows) / n_cert_tot) if n_cert_tot else 0.0
    out["session_any_error"] = float(np.mean([r["any_error"] for r in rows]))
    out["all_certified_frac"] = float(np.mean([r["calls_all"] is not None for r in rows]))
    out["n_cert_per_seed"] = [r["n_cert"] for r in rows]
    # per-session arrays: needed for session-level bootstrap / Clopper-Pearson intervals
    out["n_dec_err_per_seed"] = [r["n_dec_err"] for r in rows]
    out["n_noncov_per_seed"] = [r["n_noncov"] for r in rows]
    out["any_error_per_seed"] = [r["any_error"] for r in rows]
    out["cert_qid_freq"] = {}
    for r in rows:
        for qid in r["cert_qids"]:
            out["cert_qid_freq"][qid] = out["cert_qid_freq"].get(qid, 0) + 1
    return out



# 预算网格：落在规划检查点上（128 的整数倍；渐进规划下 6272 与 15744 同样恰好是检查点）
GRID = [640, 1536, 3200, 6272, 15744, 31482]
GRID_LABEL = {640: "2%", 1536: "5%", 3200: "10%", 6272: "20%", 15744: "50%", 31482: "100%"}


def progressive_interval(calls, N, base=128):
    frac = calls / max(1, N)
    return base if frac < 0.10 else (2 * base if frac < 0.30 else 4 * base)


def snap_results(results):
    return {qid: (r.status, r.L, r.U) for qid, r in results.items()}


def metrics_at(trace, wl, theta_M, b):
    """由一条随时可用的轨迹读出预算 b 时的指标（同一种子下，各预算档的轨迹在最后一步之前完全相同）。"""
    last, first_cert = (0, {}), {}
    for calls, snap in trace:
        for qid, (st, L, U) in snap.items():
            if qid not in first_cert and st.startswith("certified"):
                first_cert[qid] = calls
        if calls <= b:
            last = (calls, snap)
    calls, snap = last
    n_cert = n_err = n_noncov = 0
    cert_qids, widths = [], []
    for q in wl:
        st, L, U = snap.get(q.qid, ("undecided", None, None))
        th = theta_M[q.qid]
        if q.kind in ("value", "difference"):
            lo, hi = (L, U) if (L is not None and U is not None) else \
                ((0.0, 1.0) if q.kind == "value" else (-1.0, 1.0))
            widths.append(hi - lo)
        if st.startswith("certified"):
            n_cert += 1
            cert_qids.append(q.qid)
            inside = (L - 1e-9 <= th <= U + 1e-9)
            if q.kind == "threshold":
                bad = (st == "certified_true" and not (th > q.tau)) or \
                      (st == "certified_false" and not (th <= q.tau))
                n_err += int(bad)
            else:
                n_err += int(not inside)
                n_noncov += int(not inside)
    calls_all = max(first_cert[q.qid] for q in wl) if n_cert == len(wl) else None
    return {"calls": int(calls), "n_cert": n_cert, "n_dec_err": n_err, "n_noncov": n_noncov,
            "num_width": float(np.mean(widths)) if widths else None, "cert_qids": cert_qids,
            "calls_all": calls_all, "any_error": int(n_err > 0 or n_noncov > 0)}


# =========================================================================== #
# 方法
# =========================================================================== #
def run_cohort(env, allocator, budget, seed, cfg=None, cache=None, e_as_random=False,
               tcache=None, wl=None):
    cfg = cfg or Config(alpha=ALPHA)
    wl = wl or fresh_workload(env)
    pipe = COHORTRAG(env["snap"], env["contract"], cfg, seed=seed, cache_labels=cache,
                     e_as_random=e_as_random)
    if tcache:                                   # 历史作为定向暴露 T（新阶段 + 指针规则下的零成本读取）
        by_h = {}
        for rid, lab in tcache.items():
            by_h.setdefault(env["stratum_of"][int(rid)], []).append((int(rid), lab))
        for h, items in by_h.items():
            pipe.ledger.add_targeted(h, [i for i, _ in items], [l for _, l in items])
    res = pipe.run(wl, allocator, budget_total=budget)
    return res, pipe.calls, wl, pipe


def run_b2_traces(env, seed):
    """逐查询独立执行（不共享状态也不共享缓存；每个查询风险 α/|Q|）：
    每个查询单独跑到「全预算的 1/|Q|」，返回各自的轨迹；任意总预算 b 下的结果由 b/|Q| 处读出。"""
    wl = fresh_workload(env)
    cap = int(math.ceil(env["N"] / len(wl)))
    traces = {}
    for q in wl:
        pipe = COHORTRAG(env["snap"], env["contract"], Config(alpha=ALPHA / len(wl)), seed=seed)
        pipe.run([q], RoundRobinAllocator(), budget_total=cap)
        traces[q.qid] = pipe.trace
    return wl, traces


def b2_metrics(wl, traces, theta_M, b):
    """总预算 b 均分给 |Q| 个查询；总调用数 = 各查询实际花费之和。"""
    c = b / len(wl)
    merged, total = {}, 0.0
    for q in wl:
        last, spent = None, 0.0
        for calls, snap in traces[q.qid]:
            if calls <= c:
                last = snap
                spent = float(calls)
        st = last[q.qid] if last is not None else ("undecided", None, None)
        merged[q.qid] = st
        total += spent if st[0].startswith("certified") else float(c)
    rec = metrics_at([(0, merged)], wl, theta_M, 10 ** 12)
    rec["calls"] = int(total)
    rec["calls_all"] = None if rec["n_cert"] < len(wl) else int(total)
    return rec


def run_shared_pool(env, budget, seed, wl=None):
    """共享样本池 + 共享缓存 + 估计量去重 + 跨阈值复用区间；
    但不使用联合多面体（无已知质量/总量守恒/逻辑包含）：A、B 各自的 CS 区间相除。"""
    wl = wl or fresh_workload(env)
    snap, contract = env["snap"], env["contract"]
    pipe = COHORTRAG(snap, contract, Config(alpha=ALPHA), seed=seed)
    led = pipe.ledger
    events = {}                              # (h, 表) -> CS，跨查询共享
    def ev(h, tab):
        t = tuple(int(round(x)) for x in tab)
        if sum(t) == 0:
            return None
        key = (h, t)
        if key not in events:
            events[key] = FinitePopulationCS(led.N_rem(h))
        return key
    sides = {}
    for q in wl:
        sides[(q.qid, 0)] = (q.scope, {h: ev(h, q.a_hz[h]) for h in q.scope},
                             {h: ev(h, q.tables["b"]) for h in q.scope})
        if q.kind == "difference":
            sides[(q.qid, 1)] = (q.scope2, {h: ev(h, q.a_hz2[h]) for h in q.scope2},
                                 {h: ev(h, q.tables["b2"]) for h in q.scope2})
    delta = ALPHA / len(events)

    def ratio(side):
        scope, ak, bk = sides[side]
        loA = hiA = loB = hiB = 0.0
        for h in scope:
            if ak[h] is not None:
                a, b = events[ak[h]].interval(delta)
                loA += a; hiA += b
            if bk[h] is not None:
                a, b = events[bk[h]].interval(delta)
                loB += a; hiB += b
        if hiB <= 0:
            return None
        if loB <= 0:
            return (0.0, 1.0, False)
        return (max(0.0, loA / hiB), min(1.0, hiA / loB), True)

    results = {q.qid: QueryResult(qid=q.qid, name=q.name, kind=q.kind) for q in wl}

    def classify():
        for q in wl:
            r = results[q.qid]
            if r.status.startswith("certified"):
                continue
            a = ratio((q.qid, 0))
            if a is None or not a[2]:
                continue
            if q.kind == "difference":
                b = ratio((q.qid, 1))
                if b is None or not b[2]:
                    continue
                L, U = a[0] - b[1], a[1] - b[0]
                r.L, r.U = L, U
                if U - L <= 2 * q.eps:
                    r.status, r.point, r.calls_at_cert = "certified", (L + U) / 2, pipe.calls
            else:
                L, U = a[0], a[1]
                r.L, r.U = L, U
                if q.kind == "value" and U - L <= 2 * q.eps:
                    r.status, r.point, r.calls_at_cert = "certified", (L + U) / 2, pipe.calls
                elif q.kind == "threshold":
                    if L > q.tau:
                        r.status, r.calls_at_cert = "certified_true", pipe.calls
                    elif U <= q.tau:
                        r.status, r.calls_at_cert = "certified_false", pipe.calls

    alloc = RoundRobinAllocator()
    trace = []
    while True:
        classify()
        trace.append((pipe.calls, snap_results(results)))
        unresolved = [q for q in wl if not results[q.qid].status.startswith("certified")]
        if not unresolved or pipe.calls >= budget:
            break
        ctx = BudgetContext(N_rem={h: led.N_rem(h) for h in led.strata},
                            n={h: led.n_prefix(h) for h in led.strata},
                            cost_per_pos={h: 1.0 for h in led.strata}, queries=[])
        plan = alloc.allocate(ctx, min(progressive_interval(pipe.calls, env["N"]), budget - pipe.calls))
        if not plan:
            break
        for h, want in plan.items():
            before = led.pos[h]
            pipe.calls += led.consume_prefix(h, want, contract)
            consumed = led.pos[h] - before
            if consumed <= 0:
                continue
            new_gs = led.prefix_g[h][-consumed:]
            for (hh, t), cs in events.items():
                if hh == h:
                    for g in new_gs:
                        cs.push(t[states.encode_state(tuple(int(v) for v in g))])
    return results, pipe.calls, wl, trace


def run_independent_ratio(env, allocator, budget, seed, wl=None):
    """+ 已知质量与总量守恒（联合多面体的约束部分），但比例用独立的 A、B 区间相除。"""
    orig = P.ratio_bounds
    P.ratio_bounds = RA.ratio_bounds_independent
    try:
        res, calls, wl, pipe = run_cohort(env, allocator, budget, seed, wl=wl)
    finally:
        P.ratio_bounds = orig
    return res, calls, wl, pipe.trace


def run_eg_style(env, budget, seed):
    """Evergreen 风格的逐声明验证（我们给它最强的合理实现）：
    每个声明独立编译为一条验证查询（独立洗牌 + CS + 早停）；跨声明共享标签缓存（零成本命中）；
    联合风险合同下按声明数 Bonferroni 拆分；各声明并发推进（每轮每个未决声明各取一批）。
    CS 原语与 COHORT 相同（无放回超几何混合鞅），因此差异只来自证据状态的设计。"""
    wl = fresh_workload(env)
    snap, contract = env["snap"], env["contract"]
    strata_ids = {h: [int(i) for i in snap.stratum_ids(h)] for h in snap.strata()}
    id_h = env["stratum_of"]
    cache, calls = {}, [0]

    def label(rid):
        if rid in cache:
            return cache[rid]
        if calls[0] >= budget:
            return None
        cache[rid] = np.asarray(contract.judge([rid], snap.texts([rid]))[0], dtype=np.int8)
        calls[0] += 1
        return cache[rid]

    results = {q.qid: QueryResult(qid=q.qid, name=q.name, kind=q.kind) for q in wl}
    claims = {}
    for q in wl:
        specs = [(q.scope, q.a_hz, q.tables["b"])]
        if q.kind == "difference":
            specs.append((q.scope2, q.a_hz2, q.tables["b2"]))
        sd = []
        for sc, a_hz, b in specs:
            pool = [rid for h in sc for rid in strata_ids[h]]
            rng = np.random.default_rng(seed * 1009 + q.qid * 37)
            perm = [pool[i] for i in rng.permutation(len(pool))]
            sd.append({"perm": perm, "pos": 0, "csA": FinitePopulationCS(len(perm)),
                       "csB": FinitePopulationCS(len(perm)), "a": a_hz, "b": b})
        claims[q.qid] = (q, sd, ALPHA / (len(wl) * 2 * len(specs)))

    def status(q, sd, d):
        ivs = []
        for s in sd:
            A, B = s["csA"].interval(d), s["csB"].interval(d)
            if A is None or B is None or B[0] <= 0:
                return None
            ivs.append((max(0.0, A[0] / B[1]), min(1.0, A[1] / B[0])))
        if q.kind == "difference":
            return (ivs[0][0] - ivs[1][1], ivs[0][1] - ivs[1][0])
        return ivs[0]

    unresolved = list(wl)
    trace = [(0, snap_results(results))]
    while unresolved and calls[0] < budget:
        progressed = False
        for q in list(unresolved):
            _, sd, d = claims[q.qid]
            for s in sd:
                end = min(s["pos"] + 64, len(s["perm"]))
                for p in range(s["pos"], end):
                    rid = s["perm"][p]
                    lab = label(rid)
                    if lab is None:
                        break
                    z = states.encode_state(tuple(int(v) for v in lab))
                    s["csA"].push(int(round(s["a"][id_h[rid]][z])))
                    s["csB"].push(int(round(s["b"][z])))
                    s["pos"] = p + 1
                    progressed = True
            iv = status(q, sd, d)
            r = results[q.qid]
            if iv is None:
                continue
            r.L, r.U = iv
            if q.kind in ("value", "difference") and iv[1] - iv[0] <= 2 * q.eps:
                r.status, r.point, r.calls_at_cert = "certified", sum(iv) / 2, calls[0]
                unresolved.remove(q)
            elif q.kind == "threshold":
                if iv[0] > q.tau:
                    r.status, r.calls_at_cert = "certified_true", calls[0]
                    unresolved.remove(q)
                elif iv[1] <= q.tau:
                    r.status, r.calls_at_cert = "certified_false", calls[0]
                    unresolved.remove(q)
            trace.append((calls[0], snap_results(results)))
        if not progressed:
            break
    return results, calls[0], wl, trace


# =========================================================================== #
# 实验 1：强基线阶梯
# =========================================================================== #
LADDER = [
    ("B2 per-query (no sharing)", "b2"),
    ("EG-style claim-by-claim", lambda env, sd: run_eg_style(env, env["N"], sd)),
    ("Shared pool, dedup (no polytope)", lambda env, sd: run_shared_pool(env, env["N"], sd)),
    ("+ known mass & conservation", lambda env, sd: run_independent_ratio(env, RoundRobinAllocator(), env["N"], sd)),
    ("+ joint projection (RR)", lambda env, sd: _cohort_trace(env, RoundRobinAllocator(), sd)),
    ("+ certificate-guided scheduler", lambda env, sd: _cohort_trace(env, CertificateGuidedAllocator(), sd)),
]


def _cohort_trace(env, allocator, seed):
    res, calls, wl, pipe = run_cohort(env, allocator, env["N"], seed)
    return res, calls, wl, pipe.trace


def _run_curve(env, runner, seeds, budgets):
    """每个种子只跑一次（预算 = N），再在检查点网格上读出各预算档的指标。"""
    rows = {b: [] for b in budgets}
    for sd in seeds:
        t0 = time.time()
        if runner == "b2":
            wl, traces = run_b2_traces(env, sd)
            for b in budgets:
                rows[b].append(b2_metrics(wl, traces, env["theta_M"], b))
        else:
            _, _, wl, trace = runner(env, sd)
            for b in budgets:
                rows[b].append(metrics_at(trace, wl, env["theta_M"], b))
        throttle(time.time() - t0)
    return rows


def _curve_exp(env, path, items, seeds, label_key, budgets=GRID, seeds_by_name=None):
    full = os.path.join(OUT_DIR, path)
    out = json.load(open(full)) if os.path.exists(full) else {}
    for name, runner in items:
        if all(f"{name}@{b}" in out for b in budgets):
            continue
        sd = (seeds_by_name or {}).get(name, seeds)
        t0 = time.time()
        rows = _run_curve(env, runner, sd, budgets)
        for b in budgets:
            out[f"{name}@{b}"] = {label_key: name, "budget": b, "frac": b / env["N"], "n_seeds": len(sd),
                                  **summarize(rows[b])}
        dump(path, out)
        log(f"[{path[4:-5]}] {name} done in {time.time() - t0:.0f}s: " + " | ".join(
            f"{GRID_LABEL.get(b, b)}: cert={out[f'{name}@{b}']['n_cert_mean']:.1f}" for b in budgets))
    return out


def exp_ladder(env, seeds=SEEDS, only=None):
    items = [(n, r) for n, r in LADDER if not only or n in only]
    return _curve_exp(env, "exp_ladder.json", items, seeds, "method",
                      seeds_by_name={"B2 per-query (no sharing)": list(range(5))})


def exp_sched(env, seeds=SEEDS):
    items = [(name, (lambda cls: (lambda env, sd: _cohort_trace(env, cls(), sd)))(cls))
             for name, cls in SCHEDULERS.items()]
    return _curve_exp(env, "exp_sched.json", items, seeds, "sched")


# =========================================================================== #
# 实验 2：历史选择偏差
# =========================================================================== #
def enriched_history(env, ratio, frac, seed):
    """分子阳性（相关且精确匹配）记录的入选几率放大 ratio 倍；ratio=1 即均匀随机历史。"""
    ids = env["ids_all"]
    lab = env["labels_M"]
    pos = np.array([int(lab[int(i)][0]) == 1 and int(lab[int(i)][1]) == 1 for i in ids])
    w = np.where(pos, float(ratio), 1.0)
    rng = np.random.default_rng(10_000 + seed)
    m = int(frac * len(ids))
    pick = rng.choice(len(ids), size=m, replace=False, p=w / w.sum())
    return {int(ids[i]): lab[int(ids[i])] for i in pick}


_TOK = re.compile(r"[a-z0-9]+")


def bm25_history(env, frac, seed, k=64):
    """真实词法检索生成的历史：随机抽若干「已问过的意图」，取每个意图 BM25 top-k 的候选对。"""
    f = env["frame"]
    parts = f["text"].str.split("商品：", n=1, expand=True)
    qtxt = parts[0].str.replace("搜索意图：", "", regex=False).str.strip().tolist()
    ptxt = parts[1].tolist()
    ptok = [_TOK.findall(t.lower()) for t in ptxt]
    N = len(ptok)
    df = {}
    for toks in ptok:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    avgdl = np.mean([len(t) for t in ptok])
    k1, b = 1.5, 0.75
    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    ids = f["id"].astype(int).to_numpy()
    intent = f["intent_id"].astype(int).to_numpy()
    by_intent = {}
    for r, it in enumerate(intent):
        by_intent.setdefault(int(it), []).append(r)
    rng = np.random.default_rng(20_000 + seed)
    order = list(rng.permutation(sorted(by_intent)))
    lab = env["labels_M"]
    chosen, target = [], int(frac * N)
    for it in order:
        rows = by_intent[int(it)]
        qt = set(_TOK.findall(qtxt[rows[0]].lower()))
        scores = []
        for r in rows:
            tf = {}
            for t in ptok[r]:
                if t in qt:
                    tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t, c in tf.items():
                s += idf.get(t, 0.0) * c * (k1 + 1) / (c + k1 * (1 - b + b * len(ptok[r]) / avgdl))
            scores.append(s)
        top = np.argsort(-np.array(scores), kind="stable")[:k]
        chosen.extend(rows[j] for j in top)
        if len(chosen) >= target:
            break
    chosen = chosen[:target]
    return {int(ids[r]): lab[int(ids[r])] for r in chosen}


def naive_eval(env, wl, exposed, labels_M, stratum_of):
    """朴素做法：把所有暴露记录池化为简单随机样本，Wald 95% 区间；返回 (认证数, 判定错误数, 数值非覆盖数)。"""
    theta_M = env["theta_M"]

    def side(q, second=False):
        a_hz, b, scope = (q.a_hz2, q.tables["b2"], q.scope2) if second else (q.a_hz, q.tables["b"], q.scope)
        A = B = 0
        for i in exposed:
            h = stratum_of[int(i)]
            if h not in a_hz:
                continue
            z = states.encode_state(tuple(int(v) for v in labels_M[int(i)]))
            A += int(round(a_hz[h][z]))
            B += int(b[z])
        if B <= 0:
            return None
        th = A / B
        se = (th * (1 - th) / B) ** 0.5
        return max(0.0, th - 1.96 * se), min(1.0, th + 1.96 * se)

    n_cert = n_err = n_noncov = 0
    for q in wl:
        th = theta_M[q.qid]
        iv = side(q)
        if iv is None:
            continue
        if q.kind == "difference":
            iv2 = side(q, True)
            if iv2 is None:
                continue
            L, U = iv[0] - iv2[1], iv[1] - iv2[0]
        else:
            L, U = iv
        if q.kind in ("value", "difference"):
            if U - L > 2 * q.eps:
                continue
            n_cert += 1
            bad = not (L - 1e-9 <= th <= U + 1e-9)
            n_err += int(bad)
            n_noncov += int(bad)
        else:
            if L > q.tau:
                n_cert += 1
                n_err += int(not (th > q.tau))
            elif U <= q.tau:
                n_cert += 1
                n_err += int(not (th <= q.tau))
    return n_cert, n_err, n_noncov


def history_arms(env, hist, budget, seed, arms=("discard", "T-reads", "E (ours)", "pooled-as-random", "naive")):
    out = {}
    for arm in arms:
        t0 = time.time()
        wl = fresh_workload(env)
        if arm == "discard":
            res, calls, wl, _ = run_cohort(env, RoundRobinAllocator(), budget, seed, wl=wl)
            out[arm] = evaluate(res, wl, env["theta_M"], calls, budget)
        elif arm == "T-reads":
            res, calls, wl, _ = run_cohort(env, RoundRobinAllocator(), budget, seed, tcache=hist, wl=wl)
            out[arm] = evaluate(res, wl, env["theta_M"], calls, budget)
        elif arm == "E (ours)":
            res, calls, wl, pipe = run_cohort(env, RoundRobinAllocator(), budget, seed, cache=hist, wl=wl)
            out[arm] = evaluate(res, wl, env["theta_M"], calls, budget)
            exposed = sorted({int(i) for h in pipe.ledger.strata for i in pipe.ledger.exposed_ids(h)})
            if "naive" in arms:
                nc, ne, nn = naive_eval(env, wl, exposed, env["labels_M"], env["stratum_of"])
                out["naive"] = {"calls": calls, "n_cert": nc, "n_dec_err": ne, "n_noncov": nn,
                                "num_width": None, "cert_qids": [], "calls_all": None,
                                "any_error": int(ne > 0)}
        elif arm == "pooled-as-random":
            res, calls, wl, _ = run_cohort(env, RoundRobinAllocator(), budget, seed, cache=hist,
                                           e_as_random=True, wl=wl)
            out[arm] = evaluate(res, wl, env["theta_M"], calls, budget)
        throttle(time.time() - t0)
    return out


def hist_stats(env, hist):
    lab = env["labels_M"]
    ids = env["ids_all"]
    hit = lambda it: float(np.mean([int(lab[int(i)][0]) == 1 and int(lab[int(i)][1]) == 1 for i in it]))
    return {"size": len(hist), "hit_hist": hit(list(hist)), "hit_pop": hit(ids)}


def exp_bias(env, ratios=(1, 2, 4, 8, 16), hist_frac=0.15, budgets=(0.02, 0.10), seeds=tuple(range(4)),
             path="exp_bias.json"):
    out = json.load(open(os.path.join(OUT_DIR, path))) if os.path.exists(os.path.join(OUT_DIR, path)) else {}
    settings = [("enrich", r) for r in ratios] + [("bm25", 0)]
    for kind, r in settings:
        for bf in budgets:
            key = f"{kind}:{r}@{bf}"
            if key in out:
                continue
            budget = int(env["N"] * bf)
            per_arm, stats = {}, []
            for sd in seeds:
                hist = enriched_history(env, r, hist_frac, sd) if kind == "enrich" \
                    else bm25_history(env, hist_frac, sd)
                stats.append(hist_stats(env, hist))
                for arm, rec in history_arms(env, hist, budget, sd).items():
                    per_arm.setdefault(arm, []).append(rec)
            out[key] = {"kind": kind, "ratio": r, "budget": bf, "hist_frac": hist_frac,
                        "hit_hist": float(np.mean([s["hit_hist"] for s in stats])),
                        "hit_pop": stats[0]["hit_pop"],
                        "arms": {a: summarize(v) for a, v in per_arm.items()}}
            dump(path, out)
            log(f"[bias] {kind}:{r} budget={bf:.2f} hit_hist={out[key]['hit_hist']:.3f} "
                + " | ".join(f"{a}: cert={s['n_cert_mean']:.1f} err={s['dec_err_rate']:.2f}"
                             for a, s in out[key]["arms"].items()))
    return out


# =========================================================================== #
# 实验 3：分母稀疏度
# =========================================================================== #
def exp_denom(env, n_pop=2000, budgets=(0.05, 0.10, 0.20, 0.50, 1.00), seeds=tuple(range(10)), n_levels=8):
    path = "exp_denom.json"
    f = env["frame"]
    lab = env["labels_M"]
    f = f.assign(p1=[int(lab[int(i)][0]) for i in f["id"]],
                 ex=[int(lab[int(i)][0]) * int(lab[int(i)][1]) for i in f["id"]])
    per_intent = f.groupby("intent_id").agg(n=("id", "size"), rel=("p1", "mean")).sort_values("rel")
    intents = per_intent.index.tolist()
    # 把意图按相关率排序后，依次装箱成 ~n_pop 条记录的人群，取 n_levels 个等间隔的人群
    bins, cur, cnt = [], [], 0
    for it in intents:
        cur.append(it)
        cnt += int(per_intent.loc[it, "n"])
        if cnt >= n_pop:
            bins.append(cur)
            cur, cnt = [], 0
    pick = np.unique(np.linspace(0, len(bins) - 1, n_levels).round().astype(int))
    # 额外两个更稀有的小人群：只取相关率最低的若干意图（~600 / ~1200 条记录）
    rare = []
    for target in (600, 1200):
        cur, cnt = [], 0
        for it in intents:
            cur.append(it)
            cnt += int(per_intent.loc[it, "n"])
            if cnt >= target:
                break
        rare.append(cur)
    pops = [rare[0], rare[1]] + [bins[bi] for bi in pick]
    out = {"populations": []}
    contract = env["contract"]
    for lvl, members in enumerate(pops):
        sub = f[f["intent_id"].isin(members)].copy()
        sub["stratum"] = 0
        snap = Snapshot(sub[["id", "stratum", "text"]].reset_index(drop=True), name=f"denom-{lvl}")
        N = len(sub)
        B = int(sub["p1"].sum())
        A = int(sub["ex"].sum())
        qv = Query(qid=1, name="theta(exact|relevant)", scope=(0,), d_idx=(0,), r_idx=(1,), kind="value", eps=0.10)
        truth = A / B if B else None
        rec = {"level": lvl, "N": N, "B": B, "selectivity": B / N, "theta": truth, "budgets": {}}
        for bf in budgets:
            budget = int(N * bf)
            rows = []
            for sd in seeds:
                t0 = time.time()
                q = Query(qid=1, name=qv.name, scope=(0,), d_idx=(0,), r_idx=(1,), kind="value", eps=0.10)
                q2 = Query(qid=2, name="thr>0.2", scope=(0,), d_idx=(0,), r_idx=(1,), kind="threshold", tau=0.2)
                pipe = COHORTRAG(snap, contract, Config(alpha=ALPHA), seed=sd)
                res = pipe.run([q, q2], RoundRobinAllocator(), budget_total=budget)
                rv = res[1]
                rows.append({"status": rv.status, "width": (None if rv.L is None else rv.U - rv.L),
                             "minB": rv.minB, "maxB": rv.maxB, "calls": pipe.calls,
                             "thr": res[2].status,
                             "cov": (None if rv.status != "certified" or truth is None
                                     else int(rv.L - 1e-9 <= truth <= rv.U + 1e-9))})
                throttle(time.time() - t0)
            st = {}
            for r in rows:
                st[r["status"]] = st.get(r["status"], 0) + 1
            ws = [r["width"] for r in rows if r["width"] is not None]
            rec["budgets"][str(bf)] = {
                "status_counts": st, "width_mean": float(np.mean(ws)) if ws else None,
                "calls_mean": float(np.mean([r["calls"] for r in rows])),
                "cert_rate": st.get("certified", 0) / len(rows),
                "cov_bad": int(sum(1 for r in rows if r["cov"] == 0)),
                "thr_status": {k: sum(1 for r in rows if r["thr"] == k) for k in {r["thr"] for r in rows}}}
        out["populations"].append(rec)
        dump(path, out)
        log(f"[denom] level={lvl} N={N} B={B} sel={B/N:.2f} theta={truth if truth is None else round(truth,3)} "
            + " ".join(f"{k}:{v['cert_rate']:.1f}" for k, v in rec["budgets"].items()))
    return out


# =========================================================================== #
# 实验 5：校准（路径级 + 端到端 + 审计一致性）
# =========================================================================== #
def _path_stat(bits, N, k_true, T):
    """整条路径上 max_s [logM_s − logL_s(k*)]；超过 log(1/δ) 即真实计数 k* 曾越出 CS。"""
    s = np.arange(1, len(bits) + 1)
    x = np.cumsum(bits)
    logM = T[x] + T[s - x] - T[s + 1]
    k = k_true
    logL = (T[k] - T[k - x]) + (T[N - k] - T[N - k - (s - x)]) - (T[N] - T[N - s])
    return float(np.max(logM - logL))


def clopper_pearson_upper(k, n, conf=0.95):
    """单侧 Clopper–Pearson 上界（k=0 时有闭式）。"""
    if k == 0:
        return 1 - (1 - conf) ** (1.0 / n)
    from math import lgamma
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        # P(X <= k | p=mid)
        cdf = sum(math.exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * math.log(mid)
                           + (n - i) * math.log(1 - mid)) for i in range(0, k + 1))
        if cdf > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def exp_calib(env, n_path=3000, n_e2e=200, n_audit=24, e2e_frac=0.02):
    path = "exp_calib.json"
    out = json.load(open(os.path.join(OUT_DIR, path))) if os.path.exists(os.path.join(OUT_DIR, path)) else {}
    snap = env["snap"]
    wl = fresh_workload(env)
    tabs = []
    for q in wl:
        tabs += [q.tables["d"], q.tables["a"]]
        if q.kind == "difference":
            tabs += [q.tables["d2"], q.tables["a2"]]
    basis, _ = states.compile_basis(tabs, [str(i) for i in range(len(tabs))])
    strata = snap.strata()
    N_h = {h: len(snap.stratum_ids(h)) for h in strata}
    z_of = {int(i): states.encode_state(tuple(int(v) for v in env["labels_M"][int(i)])) for i in env["ids_all"]}
    T = np.array([math.lgamma(v + 1.0) for v in range(max(N_h.values()) + 5)])
    n_cs = len(strata) * len(basis)
    # ---- (a) 路径级：同一联合事件在整条路径上的失败率（任意停止规则/自适应追问都被它覆盖）----
    if "pathwise" not in out:
        maxstat = []
        t0 = time.time()
        for s in range(n_path):
            m = -np.inf
            for h in strata:
                R = sorted(int(i) for i in snap.stratum_ids(h))
                rng = np.random.default_rng(100_000 + s + h * 100003)
                zs = np.array([z_of[R[i]] for i in rng.permutation(len(R))])
                for f in basis:
                    bits = f[zs].astype(np.int64)
                    m = max(m, _path_stat(bits, N_h[h], int(bits.sum()), T))
            maxstat.append(m)
            if s % 100 == 99:
                throttle((time.time() - t0) / (s + 1) * 100 * 0.3)
        maxstat = np.array(maxstat)
        res = {}
        for alpha in (0.01, 0.05, 0.10):
            fails = int((maxstat > math.log(n_cs / alpha)).sum())
            res[str(alpha)] = {"sessions": n_path, "failures": fails, "rate": fails / n_path,
                               "cp95_upper": clopper_pearson_upper(fails, n_path)}
            log(f"[calib/path] alpha={alpha} sessions={n_path} failures={fails}")
        res["n_cs"] = n_cs
        res["maxstat_quantiles"] = {str(q): float(np.quantile(maxstat, q)) for q in (0.5, 0.9, 0.99, 1.0)}
        out["pathwise"] = res
        dump(path, out)
    # ---- (b) 端到端：调度器 + 投影 + 分类 的全流程会话（快速路径）----
    if "e2e" not in out:
        rows = []
        for s in range(n_e2e):
            t0 = time.time()
            res, calls, wl, _ = run_cohort(env, CertificateGuidedAllocator(), int(env["N"] * e2e_frac), 5000 + s)
            rows.append(evaluate(res, wl, env["theta_M"], calls, 0))
            throttle(time.time() - t0)
        n_c = sum(r["n_cert"] for r in rows)
        bad_sessions = sum(r["any_error"] for r in rows)
        out["e2e"] = {"sessions": n_e2e, "certified_claims": n_c,
                      "decision_errors": sum(r["n_dec_err"] for r in rows),
                      "noncoverage": sum(r["n_noncov"] for r in rows),
                      "sessions_with_error": bad_sessions,
                      "cp95_upper_session": clopper_pearson_upper(bad_sessions, n_e2e),
                      "n_cert_mean": float(np.mean([r["n_cert"] for r in rows]))}
        dump(path, out)
        log(f"[calib/e2e] sessions={n_e2e} certified={n_c} errors={bad_sessions}")
    # ---- (c) 精确审计一致性：同一批会话开/关审计，比较认证集合与端点 ----
    if "audit" not in out:
        same = 0
        widen, max_widen, n_audits, n_adj = [], 0.0, 0, 0
        for s in range(n_audit):
            t0 = time.time()
            r0, c0, w0, _ = run_cohort(env, CertificateGuidedAllocator(), int(env["N"] * e2e_frac), 7000 + s)
            cfg = Config(alpha=ALPHA)
            cfg.audit_exact = True
            r1, c1, w1, pipe = run_cohort(env, CertificateGuidedAllocator(), int(env["N"] * e2e_frac), 7000 + s, cfg=cfg)
            e0 = evaluate(r0, w0, env["theta_M"], c0, 0)
            e1 = evaluate(r1, w1, env["theta_M"], c1, 0)
            same += int(e0["cert_qids"] == e1["cert_qids"])
            for entry in pipe.audit_log:
                if entry[0] == "ratio":
                    n_audits += 1
                    n_adj += int(entry[1] > 0)
                    max_widen = max(max_widen, entry[2] - entry[3], entry[5] - entry[4])
                else:
                    n_audits += 1
                    n_adj += int(entry[1] > 0)
            throttle(time.time() - t0)
        out["audit"] = {"sessions": n_audit, "same_certified_set": same,
                        "audited_conclusions": n_audits, "audit_rejections_or_adjustments": n_adj,
                        "max_outward_margin": max_widen}
        dump(path, out)
        log(f"[calib/audit] same_set={same}/{n_audit} audits={n_audits} adjusted={n_adj} margin={max_widen:.2e}")
    return out


# =========================================================================== #
# 实验 4：多轮交互会话
# =========================================================================== #
def product_class_map():
    import pandas as pd
    from cohortrag.datasets import _read_tabular
    prod = _read_tabular(os.path.join(WANDS_DIR, "dataset", "product.csv"), "product.csv")
    return dict(zip(prod["product_id"].astype(int), prod["product_class"].fillna("missing").astype(str)))


def session_queries(env):
    """第 1–3 轮的查询：全部落在预声明闭包内（同一统计基）。"""
    by = env["by_rating"]
    all_h = tuple(sorted(env["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in by.items() if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in by.items() if "<4" in str(lab) for h in hs}))
    flag = {h: 1 for h in ge4}
    flag.update({h: 0 for h in all_h if h not in ge4})
    qid = [0]

    def Q(name, scope, d, r, kind="value", eps=0.05, tau=0.5, **kw):
        qid[0] += 1
        return Query(qid=qid[0], name=name, scope=tuple(scope), d_idx=tuple(d), r_idx=tuple(r),
                     kind=kind, eps=eps, tau=tau, **kw)
    R1 = [Q("theta(exact|relevant)", all_h, [0], [1], eps=0.05),
          Q("theta(rating>=4|relevant)", all_h, [0], [], eps=0.10, r_meta=flag)]
    R2 = [Q("theta(exact|rel, >=4)", ge4, [0], [1], eps=0.05),
          Q("theta(exact|rel, <4)", lt4, [0], [1], eps=0.05),
          Q("diff >=4 minus <4", ge4, [0], [1], kind="difference", eps=0.05,
            scope2=lt4, d_idx2=(0,), r_idx2=(1,))]
    R3 = [Q(f"thr all>{t:.1f}", all_h, [0], [1], kind="threshold", tau=t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    for t in (0.3, 0.5, 0.7):
        R3.append(Q(f"thr >=4 >{t:.1f}", ge4, [0], [1], kind="threshold", tau=t))
        R3.append(Q(f"thr <4 >{t:.1f}", lt4, [0], [1], kind="threshold", tau=t))
    return R1, R2, R3


def class_stage_queries(snap2, class_strata):
    """第 4 轮：按商品大类分层的新阶段（新的元数据划分 → 重建统计基）。"""
    qid = [100]

    def Q(name, scope, kind="value", eps=0.10, tau=0.5, **kw):
        qid[0] += 1
        return Query(qid=qid[0], name=name, scope=tuple(scope), d_idx=(0,), r_idx=(1,), kind=kind,
                     eps=eps, tau=tau, **kw)
    qs = [Q(f"class{c} exact|rel", [h]) for c, h in enumerate(class_strata)]
    qs.append(Q("diff c0-c1", [class_strata[0]], kind="difference", scope2=(class_strata[1],),
                d_idx2=(0,), r_idx2=(1,)))
    qs.append(Q("diff c2-c3", [class_strata[2]], kind="difference", scope2=(class_strata[3],),
                d_idx2=(0,), r_idx2=(1,)))
    for c, h in enumerate(class_strata):
        qs.append(Q(f"class{c} >0.2", [h], kind="threshold", tau=0.2))
    return qs


def exp_session(env, seeds=tuple(range(10)), caps=(0.05, 0.10, 0.25, 0.35)):
    """四轮会话。累计预算上限（占 N）：R1 5%，R2 10%，R3 25%，R4 35%。
    COHORT：R1–R3 在同一阶段用 feed 追加查询；R4 重建为按大类分层的新阶段，历史标签作为精确已知质量 E 继承。
    对照：EG 风格逐声明（共享缓存）；COHORT-T（R4 重启，历史作为定向暴露：指针规则下零成本读取）。"""
    path = "exp_session.json"
    N = env["N"]
    pc = product_class_map()
    f = env["frame"].copy()
    f["cls"] = f["product_id"].astype(int).map(pc).fillna("missing")
    top = f["cls"].value_counts().index[:5].tolist()
    f["cstratum"] = f["cls"].apply(lambda c: top.index(c) if c in top else 5)
    snap2 = Snapshot(f[["id", "text"]].assign(stratum=f["cstratum"].values)[["id", "stratum", "text"]].reset_index(drop=True),
                     name="wands-class")
    class_strata = list(range(6))
    theta_cls_M = {}
    out = {"rounds": ["R1", "R2", "R3", "R4"], "caps": caps, "runs": {}}
    caps_n = [int(N * c) for c in caps]
    meths = ["COHORT (E-inherit)", "COHORT-T (restart, free reads)", "EG-style"]
    for m in meths:
        out["runs"][m] = []
    for sd in seeds:
        t0 = time.time()
        R1, R2, R3 = session_queries(env)
        # θ_M 真值
        truths = {}
        for q in R1 + R2 + R3:
            truths[q.qid] = rrd._query_truth(_prep(q), env["labels_M"], env["ids_all"], env["stratum_of"])
        sid_class = dict(zip(f["id"].astype(int), f["cstratum"].astype(int)))
        R4 = class_stage_queries(snap2, class_strata)
        for q in R4:
            truths[q.qid] = rrd._query_truth(_prep(q), env["labels_M"], env["ids_all"], sid_class)

        # ---------- COHORT：R1-R3 同阶段 ----------
        traj = []
        state = {"round": 0}
        # Stage budgets follow the paper's rule alpha_e = alpha / 2^(e+1): stage 0 (rounds 1-3) gets alpha/2,
        # stage 1 (round 4, new slicing) gets alpha/4, so the whole session is covered by a joint risk <= alpha.
        pipe = COHORTRAG(env["snap"], env["contract"], Config(alpha=ALPHA / 2), seed=sd)
        rounds = [R1, R2, R3]
        pipe_holder = {}

        def feed(p, results):
            ncert = sum(1 for r in results.values() if r.status.startswith("certified"))
            traj.append({"round": state["round"] + 1, "calls": p.calls, "certified": ncert,
                         "total": len(results)})
            state["round"] += 1
            if state["round"] >= len(rounds):
                return [], None
            return list(rounds[state["round"]]), caps_n[state["round"]]
        res = pipe.run(R1, CertificateGuidedAllocator(), budget_total=caps_n[0],
                       declared=R1 + R2 + R3, feed=feed)
        labels_so_far = {int(k): v for k, v in pipe.ledger.cache.items()}
        c_calls = pipe.calls
        # 错误检查
        err = 0
        for q in R1 + R2 + R3:
            r = res[q.qid]
            if r.status.startswith("certified"):
                th = truths[q.qid]
                if q.kind == "threshold":
                    err += int((r.status == "certified_true" and not th > q.tau) or
                               (r.status == "certified_false" and not th <= q.tau))
                else:
                    err += int(not (r.L - 1e-9 <= th <= r.U + 1e-9))
        # R4：新阶段
        def stage4(mode):
            cfg = Config(alpha=ALPHA / 4)
            wl4 = class_stage_queries(snap2, class_strata)
            if mode == "E":
                p4 = COHORTRAG(snap2, env["contract"], cfg, seed=sd + 777, cache_labels=labels_so_far)
            else:
                p4 = COHORTRAG(snap2, env["contract"], cfg, seed=sd + 777)
                if mode == "T":
                    by_h = {}
                    for rid, lab in labels_so_far.items():
                        by_h.setdefault(sid_class[rid], []).append((rid, lab))
                    for h, items in by_h.items():
                        p4.ledger.add_targeted(h, [i for i, _ in items], [l for _, l in items])
            r4 = p4.run(wl4, CertificateGuidedAllocator(), budget_total=caps_n[3] - c_calls)
            n4 = sum(1 for r in r4.values() if r.status.startswith("certified"))
            e4 = 0
            for q in wl4:
                r = r4[q.qid]
                if r.status.startswith("certified"):
                    th = truths[q.qid]
                    if q.kind == "threshold":
                        e4 += int((r.status == "certified_true" and not th > q.tau) or
                                  (r.status == "certified_false" and not th <= q.tau))
                    else:
                        e4 += int(not (r.L - 1e-9 <= th <= r.U + 1e-9))
            return {"round": 4, "calls": c_calls + p4.calls, "certified": n4, "total": len(wl4), "errors": e4}
        rec = {"seed": sd, "traj": traj + [stage4("E")], "errors_before_R4": err}
        out["runs"]["COHORT (E-inherit)"].append(rec)
        out["runs"]["COHORT-T (restart, free reads)"].append(
            {"seed": sd, "traj": traj + [stage4("T")], "errors_before_R4": err})
        # ---------- EG 风格：逐声明，逐轮追加，共享缓存 ----------
        eg_traj = _eg_session(env, snap2, sid_class, [R1, R2, R3], R4, caps_n, truths, sd)
        out["runs"]["EG-style"].append({"seed": sd, "traj": eg_traj["traj"], "errors": eg_traj["errors"]})
        dump(path, out)
        log(f"[session] seed={sd} COHORT-E: " + " -> ".join(f"{t['calls']}/{t['certified']}" for t in rec["traj"])
            + " | EG: " + " -> ".join(f"{t['calls']}/{t['certified']}" for t in eg_traj["traj"]))
        throttle(time.time() - t0)
    return out


def _prep(q):
    q.tables = q.build_tables(2)
    q.a_hz = q.numerator_tables(2, q.scope)
    if q.kind == "difference":
        q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    return q


def _eg_session(env, snap2, sid_class, rounds123, R4, caps_n, truths, seed):
    """EG 风格会话：每轮新声明各自独立执行（并发，共享缓存）；R4 的新分层声明同样独立执行。
    风险合同：总声明数已知并按 Bonferroni 拆分（对基线有利，因为真实会话的追问数事先未知）。"""
    contract = env["contract"]
    cache, calls = {}, [0]
    all_claims = [q for r in rounds123 for q in r] + R4
    alpha_q = ALPHA / len(all_claims)
    traj, errors = [], 0
    stratum_old = env["stratum_of"]

    def label(rid, snap, cap):
        if rid in cache:
            return cache[rid]
        if calls[0] >= cap:
            return None
        cache[rid] = np.asarray(contract.judge([rid], snap.texts([rid]))[0], dtype=np.int8)
        calls[0] += 1
        return cache[rid]

    def run_round(qs, snap, strata_ids, id_h, cap, pending):
        nonlocal errors
        state = {}
        for q in qs:
            _prep(q)
            specs = [(q.scope, q.a_hz, q.tables["b"])]
            if q.kind == "difference":
                specs.append((q.scope2, q.a_hz2, q.tables["b2"]))
            sd_ = []
            for sc, a_hz, b in specs:
                pool = [rid for h in sc for rid in strata_ids[h]]
                rng = np.random.default_rng(seed * 1009 + q.qid * 37)
                perm = [pool[i] for i in rng.permutation(len(pool))]
                sd_.append({"perm": perm, "pos": 0, "csA": FinitePopulationCS(len(perm)),
                            "csB": FinitePopulationCS(len(perm)), "a": a_hz, "b": b})
            state[q.qid] = (q, sd_, alpha_q / (2 * len(specs)))
        open_q = list(qs)
        certified = 0
        while open_q and calls[0] < cap:
            progressed = False
            for q in list(open_q):
                _, sd_, d = state[q.qid]
                for s in sd_:
                    end = min(s["pos"] + 64, len(s["perm"]))
                    for p in range(s["pos"], end):
                        rid = s["perm"][p]
                        lab = label(rid, snap, cap)
                        if lab is None:
                            break
                        z = states.encode_state(tuple(int(v) for v in lab))
                        s["csA"].push(int(round(s["a"][id_h[rid]][z])))
                        s["csB"].push(int(round(s["b"][z])))
                        s["pos"] = p + 1
                        progressed = True
                ivs = []
                ok = True
                for s in sd_:
                    A, B = s["csA"].interval(d), s["csB"].interval(d)
                    if A is None or B is None or B[0] <= 0:
                        ok = False
                        break
                    ivs.append((max(0.0, A[0] / B[1]), min(1.0, A[1] / B[0])))
                if not ok:
                    continue
                th = truths[q.qid]
                if q.kind == "difference":
                    L, U = ivs[0][0] - ivs[1][1], ivs[0][1] - ivs[1][0]
                else:
                    L, U = ivs[0]
                done = None
                if q.kind in ("value", "difference") and U - L <= 2 * q.eps:
                    done = "c"
                    errors += int(not (L - 1e-9 <= th <= U + 1e-9))
                elif q.kind == "threshold" and L > q.tau:
                    done = "t"
                    errors += int(not th > q.tau)
                elif q.kind == "threshold" and U <= q.tau:
                    done = "f"
                    errors += int(not th <= q.tau)
                if done:
                    certified += 1
                    open_q.remove(q)
            if not progressed:
                break
        return certified

    strata_old = {h: [int(i) for i in env["snap"].stratum_ids(h)] for h in env["snap"].strata()}
    strata_new = {h: [int(i) for i in snap2.stratum_ids(h)] for h in snap2.strata()}
    cum = 0
    for ri, qs in enumerate(rounds123):
        cum += run_round(qs, env["snap"], strata_old, stratum_old, caps_n[ri], None)
        traj.append({"round": ri + 1, "calls": calls[0], "certified": cum, "total": len(qs)})
    c4 = run_round(R4, snap2, strata_new, sid_class, caps_n[3], None)
    traj.append({"round": 4, "calls": calls[0], "certified": c4, "total": len(R4)})
    return {"traj": traj, "errors": errors}



# =========================================================================== #
# 实验 6：同一基底上的组合查询（联合投影的价值所在）
# =========================================================================== #
def compose_workload(env):
    """19 个组合查询：全部由同一组语义事件（相关、相关且精确）× 元数据原子（评分桶 × 意图组）表达，
    没有任何新的置信序列。层号 0..5 = (<4,g0),(<4,g1),(>=4,g0),(>=4,g1),(missing,g0),(missing,g1)。"""
    lt4, ge4, miss = (0, 1), (2, 3), (4, 5)
    g0, g1 = (0, 2, 4), (1, 3, 5)
    allh = tuple(range(6))
    qid = [0]

    def flag(hs):
        return {h: (1 if h in hs else 0) for h in allh}

    def Q(name, scope, d, r, kind="value", eps=0.05, **kw):
        qid[0] += 1
        return Query(qid=qid[0], name=name, scope=tuple(scope), d_idx=tuple(d), r_idx=tuple(r),
                     kind=kind, eps=eps, **kw)
    qs = []
    for nm, hs in (("<4", lt4), (">=4", ge4), ("miss", miss)):
        qs.append(Q(f"share rating {nm} | relevant", allh, [0], [], eps=0.10, r_meta=flag(hs)))
    qs.append(Q("share group1 | relevant", allh, [0], [], eps=0.10, r_meta=flag(g1)))
    for nm, hs in (("<4", lt4), (">=4", ge4), ("miss", miss)):
        qs.append(Q(f"exact|rel rating {nm}", hs, [0], [1]))
    for nm, hs in (("g0", g0), ("g1", g1)):
        qs.append(Q(f"exact|rel {nm}", hs, [0], [1]))
    for h in allh:
        qs.append(Q(f"exact|rel stratum {h}", (h,), [0], [1], eps=0.10))
    for nm, a, b in (("ge4-lt4", ge4, lt4), ("ge4-miss", ge4, miss), ("g1-g0", g1, g0), ("lt4-miss", lt4, miss)):
        qs.append(Q(f"diff {nm}", a, [0], [1], kind="difference", eps=0.10,
                    scope2=tuple(b), d_idx2=(0,), r_idx2=(1,)))
    for q in qs:
        _prep(q)
    return qs


def exp_compose(env, fracs=(0.02, 0.05, 0.20), seeds=SEEDS):
    path = "exp_compose.json"
    out = json.load(open(os.path.join(OUT_DIR, path))) if os.path.exists(os.path.join(OUT_DIR, path)) else {}
    base = compose_workload(env)
    truth = {q.qid: rrd._query_truth(q, env["labels_M"], env["ids_all"], env["stratum_of"]) for q in base}
    arms = [("Shared pool, dedup (no polytope)", lambda b, s, wl: run_shared_pool(env, b, s, wl=wl)[:3]),
            ("+ known mass & conservation", lambda b, s, wl: run_independent_ratio(env, RoundRobinAllocator(), b, s, wl=wl)[:3]),
            ("+ joint projection", lambda b, s, wl: run_cohort(env, RoundRobinAllocator(), b, s, wl=wl)[:3])]
    for name, fn in arms:
        for frac in fracs:
            key = f"{name}@{frac}"
            if key in out:
                continue
            rows = []
            for sd in seeds:
                t0 = time.time()
                wl = compose_workload(env)
                for q, q0 in zip(wl, base):
                    q.qid = q0.qid
                res, calls, wl = fn(int(env["N"] * frac), sd, wl)
                rows.append(evaluate(res, wl, truth, calls, 0))
                throttle(time.time() - t0)
            out[key] = {"method": name, "frac": frac, "n_queries": len(base), **summarize(rows)}
            dump(path, out)
            s = out[key]
            log(f"[compose] {name:34s} f={frac:.2f} cert={s['n_cert_mean']:5.2f}/{len(base)} "
                f"width={s['num_width_mean']:.3f} err={s['dec_err_rate']:.3f}")
    return out


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--fracs", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    env = load_wands_env()
    log(f"env ready: N={env['N']}, strata={env['snap'].strata()}, queries={len(env['wl'])}, duty={DUTY}")
    seeds = list(range(args.seeds))
    fr = [float(x) for x in args.fracs.split(",")] if args.fracs else FRACS
    which = args.exp.split(",") if args.exp != "all" else ["calib", "ladder", "bias", "compose", "session", "denom", "sched"]
    if args.smoke:
        global OUT_DIR
        OUT_DIR = os.path.join(OUT_DIR, "smoke")
        os.makedirs(OUT_DIR, exist_ok=True)
    only = args.only.split("|") if args.only else None
    for e in which:
        t0 = time.time()
        if e == "ladder":
            exp_ladder(env, seeds, only)
        elif e == "sched":
            exp_sched(env, seeds)
        elif e == "bias":
            if args.smoke:
                exp_bias(env, ratios=(1, 8), budgets=(0.02,), seeds=tuple(range(2)))
            else:
                exp_bias(env)
        elif e == "compose":
            exp_compose(env, fracs=((0.02,) if args.smoke else (0.02, 0.05, 0.20)),
                        seeds=(tuple(range(2)) if args.smoke else tuple(range(10))))
        elif e == "denom":
            exp_denom(env, seeds=tuple(range(3 if args.smoke else 10)))
        elif e == "session":
            exp_session(env, seeds=tuple(range(2 if args.smoke else 8)))
        elif e == "calib":
            if args.smoke:
                exp_calib(env, n_path=20, n_e2e=3, n_audit=2)
            else:
                exp_calib(env)
        log(f"== {e} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
