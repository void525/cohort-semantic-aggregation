# -*- coding: utf-8 -*-
"""闭包外追问实验（RAG.md §16「闭包外追问」）。

要回答的机制问题：**闭包外追问是否需要新阶段，且不会破坏闭包内的统计有效性？**

背景（§5.3）：闭包 = 固定原语集合 + 预声明元数据原子。闭包外的新原语 / 新文本视图 /
新时间切分 / **新字段分组** 需要新阶段或被明确拒绝。

本实验在 WANDS 上构造一个闭包外查询：
    rating = missing 的分组（闭包只声明过 rating>=4 与 rating<4 两个 cohort）
它使用与闭包内相同的原语 (p1=相关, p2=完全匹配)，但落在从未声明过的元数据分组上。

三种情形对照：
  A. closure_only            18 个闭包内查询，单独一个 stage
  B. closure_plus_out        18 个闭包内查询 + 1 个闭包外查询，**同一个 stage**
                             （basis 一起编译 ⇒ 风险预算被稀释：n_cs 变大、δ 变小）
  C. out_own_stage           闭包外查询**独立 stage**（重建状态与统计基）

关键观察：
  (1) B 相对 A：闭包内 18 个查询的**认证数会下降**（预算稀释的代价），
      但 **coverage 必须仍是 1.000** —— 新分组不得破坏原有闭包的有效性；
  (2) C 相对 B：闭包外查询在独立 stage 中应当能被正常认证，
      证明系统确实能处理新闭包，而不是一味拒绝；
  (3) 闭包外查询在**任何**情形下都不得返回一个**未被认证**的数字。

用法：
    python scripts/run_out_of_closure.py \
        --data <WANDS/dataset> --manifest data_manifests/wands.json \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --budget-fracs 0.2 --seeds 0,1,2
"""
import argparse
import json
import os
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import numpy as np
import run_real_data as rrd
from cohortrag import Snapshot, COHORTRAG, Config, RoundRobinAllocator
from cohortrag.workload import make_wands_closure_workload, Query


def build_context(args):
    ns = argparse.Namespace(dataset="wands", data=args.data, manifest=args.manifest,
                            max_intents=120, start="2024-01-01", end="2024-12-31",
                            max_records=50000, id_col="Complaint ID",
                            date_col="Date received",
                            narrative_col="Consumer complaint narrative",
                            product_col="Product", dry_run=False, model=args.model,
                            revision=None, device="cpu", batch_size=16)
    frame, gold, meta = rrd._real_frames(ns)
    snap = Snapshot(frame, name="wands-frozen")
    contract, kappa = rrd._load_contract(ns, snap, gold)
    fb = frame["rating_bucket"].astype(str)
    by_rating = {b: sorted(frame.loc[fb == b, "stratum"].unique().tolist())
                 for b in fb.unique()}
    return frame, snap, contract, by_rating, kappa, gold, meta


def make_out_query(by_rating):
    """闭包外查询：rating=missing 这个从未声明过的 cohort。"""
    missing_h = tuple(sorted({h for lab, hs in by_rating.items()
                              if "missing" in str(lab).lower() for h in hs}))
    if not missing_h:
        return None, ()
    q = Query(qid=9001, name="OUT theta(exact|relevant, rating=missing)",
              scope=missing_h, d_idx=(0,), r_idx=(1,), kind="value",
              eps=0.05, tau=0.5, scope2=None, d_idx2=None, r_idx2=None, r_meta=None)
    return q, missing_h


def evaluate(res, workload, theta_M):
    cov_n = cov_d = 0
    cert = 0
    for q in workload:
        r = res[q.qid]
        thM = theta_M.get(q.qid)
        if r.status.startswith("certified"):
            cert += 1
            if thM is not None:
                cov_d += 1
                cov_n += int(r.L - 1e-9 <= thM <= r.U + 1e-9)
    return cert, (cov_n / cov_d if cov_d else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget-fracs", default="0.2")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    frame, snap, contract, by_rating, kappa, gold, meta = build_context(args)
    wl = make_wands_closure_workload(by_rating, {"all": meta["all_strata"]}, K=2)
    for q in wl:
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2,
                                         tables={"a": q.tables["a2"]})

    ids_all = frame["id"].to_numpy()
    lab = contract.judge(list(ids_all), snap.texts(list(ids_all)))
    labels_M = {int(i): lab[r] for r, i in enumerate(ids_all)}
    strat_map = dict(zip(frame["id"].astype(int), frame["stratum"].astype(int)))
    theta_M = {q.qid: rrd._query_truth(q, labels_M, ids_all, strat_map) for q in wl}

    out_q, missing_h = make_out_query(by_rating)
    if out_q is None:
        print("[错误] 数据里没有 missing 评分层，无法构造闭包外新分组")
        return 1
    out_q.tables = out_q.build_tables(2)
    out_q.a_hz = out_q.numerator_tables(2, out_q.scope)
    theta_out = {out_q.qid: rrd._query_truth(out_q, labels_M, ids_all, strat_map)}

    print(f"闭包内 {len(wl)} 查询｜闭包外（新分组 rating=missing，层 {missing_h}）1 个")
    print(f"N={len(frame)}｜κ={kappa[:16]}")

    fracs = [float(x) for x in args.budget_fracs.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []

    for frac in fracs:
        budget = max(1, int(len(frame) * frac))
        print(f"\n{'预算':>6} {'情形':>24} {'认证':>9} {'覆盖':>8} {'n_cs':>6} {'δ':>10}")
        print("-" * 70)
        for mode in ("A_closure_only", "B_closure_plus_out", "C_out_own_stage"):
            cert_l, cov_n, cov_d, ncs_l, dlt_l = [], 0, 0, [], []
            for sd in seeds:
                if mode == "A_closure_only":
                    pipe = COHORTRAG(snap, contract, Config(), seed=sd)
                    res = pipe.run(wl, RoundRobinAllocator(), budget_total=budget)
                    c, cv = evaluate(res, wl, theta_M)
                elif mode == "B_closure_plus_out":
                    pipe = COHORTRAG(snap, contract, Config(), seed=sd)
                    res = pipe.run(list(wl) + [out_q], RoundRobinAllocator(),
                                   budget_total=budget)
                    # 只统计闭包内 18 个查询（看它们是否被新分组稀释）
                    c, cv = evaluate(res, wl, theta_M)
                else:
                    pipe = COHORTRAG(snap, contract, Config(), seed=sd)
                    res = pipe.run([out_q], RoundRobinAllocator(), budget_total=budget)
                    c, cv = evaluate(res, [out_q], theta_out)
                cert_l.append(c)
                if np.isfinite(cv):
                    cov_n += cv; cov_d += 1
                ncs_l.append(pipe.risk_ledger.get("n_cs", 0))
                dlt_l.append(pipe.risk_ledger.get("delta") or 0.0)
            cov = cov_n / cov_d if cov_d else float("nan")
            print(f"{frac:>6.2f} {mode:>24} {np.mean(cert_l):>9.1f}{cov:>8.3f}"
                  f"{np.mean(ncs_l):>6.0f}{np.mean(dlt_l):>10.5f}")
            rows.append({"budget_fraction": frac, "mode": mode,
                         "certified_mean": float(np.mean(cert_l)),
                         "coverage_vs_thetaM": cov,
                         "n_cs_mean": float(np.mean(ncs_l)),
                         "delta_mean": float(np.mean(dlt_l)),
                         "n_seeds": len(seeds)})
        print("-" * 70)

    out = os.path.join(_HERE, "..", "reports", "out_of_closure.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"workload": "closure", "n_queries_in": len(wl),
                   "out_query_name": out_q.name,
                   "out_cohort_scope": list(missing_h),
                   "kappa": kappa, "seeds": seeds, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"报告：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
