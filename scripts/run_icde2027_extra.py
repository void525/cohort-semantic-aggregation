# -*- coding: utf-8 -*-
"""补充：第二个语义算子（Qwen3-4B 冻结标签带）下的重放，以及语义误差与统计覆盖的正交性。

不改动 run_icde2027_suite.py（后台任务正在读它），只 import 它。
  exp_operator : 两个算子（Qwen3-8B / Qwen3-4B）各自的 θ_M 与人工标签 θ_gold 的差距，
                 以及同一套流程在各自 θ_M 上的覆盖/错误认证（独立会话）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import run_icde2027_suite as S
import run_real_data as rrd
from cohortrag import Snapshot, RoundRobinAllocator, CertificateGuidedAllocator
from cohortrag.llm_operator import LocalLLMOperator, CachedSemanticOperator
from cohortrag import contracts as _contracts

SECOND_SQLITE = os.path.join(_HERE, "..", "reports", "second_model_20261103", "labels_wands_second_model.sqlite")
KAPPA_4B = "4a928f82e4c223302a7dfdbe1e1361b8f55f0f75accbec4d04a4ee2847339fe0"


def second_contract(env):
    pack = _contracts.wands_prompt_pack()
    inner = LocalLLMOperator(model_id="/root/autodl-tmp/models/Qwen3-4B", revision=None,
                             K=len(pack["primitives"]), prompt_pack=pack, batch_size=16,
                             device="cpu", name="qwen3-4b-wands")
    cached = CachedSemanticOperator(inner, env["snap"].manifest, cache_path=SECOND_SQLITE)
    assert cached.kappa == KAPPA_4B, f"kappa mismatch: {cached.kappa}"
    return cached


def operator_view(env, contract):
    ids = env["ids_all"]
    lab = contract.judge(list(ids), env["snap"].texts(list(ids)))
    labels = {int(i): lab[r] for r, i in enumerate(ids)}
    wl = S.fresh_workload(env)
    theta_M = {q.qid: rrd._query_truth(q, labels, ids, env["stratum_of"]) for q in wl}
    sem = [abs(theta_M[q.qid] - env["theta_G"][q.qid]) for q in wl]
    rate = lambda j: float(np.mean([labels[int(i)][j] for i in ids]))
    ex = float(np.mean([int(labels[int(i)][0]) * int(labels[int(i)][1]) for i in ids]))
    return labels, theta_M, {"semantic_error_mean": float(np.mean(sem)),
                             "semantic_error_max": float(np.max(sem)),
                             "relevant_rate": rate(0), "exact_rate": ex}


def exp_operator(env, n_sessions=20, frac=3200):
    path = "exp_operator.json"
    out = {}
    ops = {"Qwen3-8B": env["contract"], "Qwen3-4B": second_contract(env)}
    for name, contract in ops.items():
        labels, theta_M, info = operator_view(env, contract)
        env2 = dict(env)
        env2["contract"] = contract
        rows = []
        for sd in range(n_sessions):
            t0 = time.time()
            res, calls, wl, pipe = S.run_cohort(env2, RoundRobinAllocator(), frac, sd)
            rows.append(S.metrics_at(pipe.trace, wl, theta_M, frac))
            S.throttle(time.time() - t0)
        out[name] = {**info, "sessions": n_sessions, "budget": frac, **S.summarize(rows),
                     "theta_M_headline": theta_M[1], "theta_gold_headline": env["theta_G"][1]}
        S.dump(path, out)
        S.log(f"[operator] {name}: sem_err={info['semantic_error_mean']:.3f} exact_rate={info['exact_rate']:.3f} "
              f"cert={out[name]['n_cert_mean']:.1f} err={out[name]['dec_err_rate']:.3f} "
              f"sessions_with_error={out[name]['session_any_error']:.2f}")
    return out


def scaled_workload(env, n_extra):
    """18 个基础声明 + n_extra 个只改阈值的追问（三个范围各 n_extra/3 个）——统计基不变。"""
    from cohortrag.workload import Query
    wl = S.fresh_workload(env)
    by = env["by_rating"]
    all_h = tuple(sorted(env["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in by.items() if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in by.items() if "<4" in str(lab) for h in hs}))
    taus = np.linspace(0.02, 0.98, max(1, n_extra // 3))
    qid = 1000
    for scope in (all_h, ge4, lt4):
        for t in taus:
            qid += 1
            q = Query(qid=qid, name=f"thr {scope}>{t:.3f}", scope=scope, d_idx=(0,), r_idx=(1,),
                      kind="threshold", tau=float(t))
            wl.append(S._prep(q))
    return wl


def exp_scale(env, sizes=(0, 36, 144), frac=3200, seeds=(0, 1, 2)):
    """追问数 |Q| 从 18 增到 162：统计基（置信序列数）不变，新增调用为 0，只多花 CPU。"""
    import time as _t
    path = "exp_scale.json"
    out = {}
    for n_extra in sizes:
        rows = []
        for sd in seeds:
            wl = scaled_workload(env, n_extra)
            truth = {q.qid: rrd._query_truth(q, env["labels_M"], env["ids_all"], env["stratum_of"]) for q in wl}
            t0 = _t.process_time()
            res, calls, wl, pipe = S.run_cohort(env, RoundRobinAllocator(), frac, sd, wl=wl)
            cpu = _t.process_time() - t0
            rec = S.metrics_at(pipe.trace, wl, truth, frac)
            rec["cpu"] = cpu
            rec["n_cs"] = pipe.risk_ledger["n_cs"]
            rec["delta"] = pipe.risk_ledger["delta"]
            rec["n_claims"] = len(wl)
            rows.append(rec)
            S.throttle(cpu)
        out[str(len(wl))] = {"n_claims": len(wl), "n_cs": rows[0]["n_cs"], "delta": rows[0]["delta"],
                             "n_cert_mean": float(np.mean([r["n_cert"] for r in rows])),
                             "calls_mean": float(np.mean([r["calls"] for r in rows])),
                             "cpu_mean": float(np.mean([r["cpu"] for r in rows])),
                             "dec_err": int(sum(r["n_dec_err"] for r in rows)), "seeds": len(seeds), "budget": frac}
        S.dump(path, out)
        S.log(f"[scale] |Q|={len(wl)} n_cs={rows[0]['n_cs']} cert={out[str(len(wl))]['n_cert_mean']:.1f} "
              f"calls={out[str(len(wl))]['calls_mean']:.0f} cpu={out[str(len(wl))]['cpu_mean']:.1f}s err={out[str(len(wl))]['dec_err']}")
    return out


def exp_cost(env, fracs=(640, 6272, 31482), seed=0):
    """COHORT 自身的 CPU 开销（不含 LLM 调用）：随会话规模的 CPU 秒与分解。"""
    import cProfile
    import pstats
    import io
    import time as _t
    path = "exp_cost.json"
    out = {}
    for b in fracs:
        wl = S.fresh_workload(env)
        pr = cProfile.Profile()
        t0 = _t.process_time()
        pr.enable()
        res, calls, wl, pipe = S.run_cohort(env, CertificateGuidedAllocator(), b, seed, wl=wl)
        pr.disable()
        cpu = _t.process_time() - t0
        st = pstats.Stats(pr)
        def tot(substr):
            return float(sum(v[3] for k, v in st.stats.items() if substr in k[2] and "cohortrag" in k[0]))
        out[str(b)] = {"calls": int(calls), "cpu_total": cpu, "cpu_cs": tot("push") + tot("interval"),
                       "cpu_lp": tot("lp_max"), "cpu_sched": tot("allocate"),
                       "gpu_equiv_s": calls / 63.7}
        S.dump(path, out)
        S.log(f"[cost] budget={b} calls={calls} cpu={cpu:.1f}s cs={out[str(b)]['cpu_cs']:.1f} lp={out[str(b)]['cpu_lp']:.1f} "
              f"sched={out[str(b)]['cpu_sched']:.1f} (labelling at 63.7 rec/s: {calls / 63.7:.0f}s)")
    return out


def exp_example(env, seed=0, budget=640):
    """论文 4.C 的示例：评级占比 θ(rating>=4 | relevant) 在 2% 预算后的区间——共享样本 vs 联合投影。"""
    path = "exp_example.json"
    wl = S.fresh_workload(env)
    qid = next(q.qid for q in wl if q.name == "theta(rating>=4 | relevant)")
    rp, cp, _, _ = S.run_shared_pool(env, budget, seed)
    rj, cj, _, _ = S.run_cohort(env, RoundRobinAllocator(), budget, seed)
    out = {"qid": qid, "seed": seed, "budget": budget, "truth": env["theta_M"][qid],
           "shared_pool": [rp[qid].L, rp[qid].U, rp[qid].status],
           "joint": [rj[qid].L, rj[qid].U, rj[qid].status]}
    S.dump(path, out)
    S.log(f"[example] truth={out['truth']:.3f} shared={out['shared_pool']} joint={out['joint']}")
    return out


def exp_basis(env, budgets=(640, 3200), seeds=(0, 1, 2, 3)):
    """统计基编译消融：不去重/不合并/不用互补，每个查询的每个分子/分母事件各建一条置信序列。"""
    path = "exp_basis.json"
    out = {}
    for compile_basis in (True, False):
        for b in budgets:
            rows, ncs = [], []
            for sd in seeds:
                cfg = S.Config(alpha=S.ALPHA)
                cfg.compile_basis = compile_basis
                res, calls, wl, pipe = S.run_cohort(env, RoundRobinAllocator(), b, sd, cfg=cfg)
                rows.append(S.metrics_at(pipe.trace, wl, env["theta_M"], b))
                ncs.append(pipe.risk_ledger["n_cs"])
                S.throttle(0.0)
            out[f"{'compiled' if compile_basis else 'per-claim events'}@{b}"] = {
                "n_cs": int(np.mean(ncs)), **S.summarize(rows)}
            S.dump(path, out)
            r = out[f"{'compiled' if compile_basis else 'per-claim events'}@{b}"]
            S.log(f"[basis] compile={compile_basis} budget={b} n_cs={r['n_cs']} cert={r['n_cert_mean']:.1f} "
                  f"width={r['num_width_mean']:.3f} err={r['dec_err_rate']}")
    return out


def exp_claims(env):
    """18 个声明的设置表：类型、容差/阈值、真值 θ_run、人工标签 θ_gold、与阈值的距离。"""
    wl = S.fresh_workload(env)
    rows = []
    for q in wl:
        th, tg = env["theta_M"][q.qid], env["theta_G"][q.qid]
        rows.append({"qid": q.qid, "name": q.name, "kind": q.kind, "eps": q.eps, "tau": q.tau if q.kind == "threshold" else None,
                     "theta_run": th, "theta_gold": tg,
                     "gap": (abs(th - q.tau) if q.kind == "threshold" else None)})
    S.dump("exp_claims.json", {"claims": rows})
    S.log(f"[claims] {len(rows)} claims; hardest gap = {min(r['gap'] for r in rows if r['gap'] is not None):.4f}")


def exp_nscale(env, ks=(1, 3, 10), budget=3200, seeds=(0, 1)):
    """总体规模 N：把真实标签带平铺 k 倍（比例不变），固定 3,200 次调用，测引擎 CPU 与认证数。"""
    import time as _t
    import pandas as pd
    from cohortrag.llm_operator import LabelTableOperator
    f = env["frame"][["id", "stratum", "text"]].copy()
    out = {}
    for k in ks:
        parts, labels = [], {}
        for j in range(k):
            g = f.copy()
            g["id"] = g["id"].astype("int64") + j * 10 ** 10
            parts.append(g)
            for i, lab in env["labels_M"].items():
                labels[int(i) + j * 10 ** 10] = lab
        frame = pd.concat(parts, ignore_index=True)
        snap = S.Snapshot(frame, name=f"tiled-{k}")
        contract = LabelTableOperator(labels, name=f"tape-x{k}", K=2)
        env2 = dict(env)
        env2["snap"], env2["contract"] = snap, contract
        env2["N"] = len(frame)
        rows = []
        for sd in seeds:
            wl = S.fresh_workload(env)
            t0 = _t.process_time()
            res, calls, wl, pipe = S.run_cohort(env2, CertificateGuidedAllocator(), budget, sd, wl=wl)
            cpu = _t.process_time() - t0
            rec = S.metrics_at(pipe.trace, wl, env["theta_M"], budget)
            rec["cpu"] = cpu
            rows.append(rec)
            S.throttle(cpu)
        out[str(len(frame))] = {"N": len(frame), "k": k, "budget": budget,
                                "cpu_mean": float(np.mean([r["cpu"] for r in rows])),
                                "n_cert_mean": float(np.mean([r["n_cert"] for r in rows])),
                                "dec_err": int(sum(r["n_dec_err"] for r in rows)), "seeds": len(seeds)}
        S.dump("exp_nscale.json", out)
        r = out[str(len(frame))]
        S.log(f"[nscale] N={r['N']} cpu={r['cpu_mean']:.1f}s cert={r['n_cert_mean']:.1f} err={r['dec_err']}")
    return out


def exp_audit_time(env, budget=640, seeds=(0, 1, 2)):
    """精确审计的 CPU 开销（每个被审计的结论）。"""
    import time as _t
    rows = []
    for sd in seeds:
        for audit in (False, True):
            cfg = S.Config(alpha=S.ALPHA)
            cfg.audit_exact = audit
            t0 = _t.process_time()
            res, calls, wl, pipe = S.run_cohort(env, CertificateGuidedAllocator(), budget, sd, cfg=cfg)
            cpu = _t.process_time() - t0
            rows.append({"seed": sd, "audit": audit, "cpu": cpu, "n_audits": len(pipe.audit_log),
                         "n_cert": sum(1 for r in res.values() if r.status.startswith("certified"))})
            S.throttle(cpu)
    a = np.mean([r["cpu"] for r in rows if r["audit"]])
    b = np.mean([r["cpu"] for r in rows if not r["audit"]])
    n = np.mean([r["n_audits"] for r in rows if r["audit"]])
    out = {"cpu_with_audit": float(a), "cpu_without": float(b), "audits_per_session": float(n),
           "per_audit_s": float((a - b) / max(n, 1))}
    S.dump("exp_audit_time.json", out)
    S.log(f"[audit-time] with={a:.1f}s without={b:.1f}s audits={n:.1f} per_audit={out['per_audit_s']:.2f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="operator")
    ap.add_argument("--sessions", type=int, default=20)
    args = ap.parse_args()
    env = S.load_wands_env()
    S.log(f"env ready duty={S.DUTY}")
    exps = set(x.strip() for x in args.exp.split(","))
    if "operator" in exps:
        exp_operator(env, n_sessions=args.sessions)
    if "scale" in exps:
        exp_scale(env)
    if "cost" in exps:
        exp_cost(env)
    if "example" in exps:
        exp_example(env)
    if "basis" in exps:
        exp_basis(env)
    if "claims" in exps:
        exp_claims(env)
    if "nscale" in exps:
        exp_nscale(env)
    if "auditime" in exps:
        exp_audit_time(env)
