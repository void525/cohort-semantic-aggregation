# -*- coding: utf-8 -*-
"""消融实验（草案 16）：三项都不需要重新标注，全部从冻结标签带重放，CPU 即可。

  independent_ratio  分子/分母各自建区间再相除（不利用联合状态与质量关系）
                     → 回答「联合投影的区间宽度收益有多大？」
  no_shared_state    每个查询独立执行、独立分配预算（只共享标签缓存，不共享状态）
                     → 回答「状态级共享是否优于仅标签缓存？」
  baseline           共享状态 + 联合投影（与 run_real_data.py 相同的口径，用于对齐）

用法（本地或 AutoDL 无卡模式；多项可并行各开一个 nohup）：
    python scripts/run_ablations.py --ablation independent_ratio \
        --data <WANDS/dataset> --manifest data_manifests/wands.json \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --workload closure --budget-fracs 0.02,0.2 --seeds 0,1,2
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
from cohortrag.workload import make_wands_workload, make_wands_closure_workload
import cohortrag.pipeline as P
from cohortrag.projection import min_max_B, _stratum_extreme, _as_a_hz


# --------------------------------------------------------------------------- #
# 消融 1：独立分子/分母区间（不用联合投影）
# --------------------------------------------------------------------------- #
def min_max_A(poly, a_hz, scope, J):
    """A(k) = Σ_h Σ_z a_hz[h][z] k_hz 在可行域上的最小/最大值（逐层求解相加）。"""
    a = _as_a_hz(a_hz, scope, J)
    lo = hi = 0.0
    for h in scope:
        c = np.asarray(a[h], dtype=np.float64)
        lo += _stratum_extreme(poly, h, c, J)
        hi += -_stratum_extreme(poly, h, -c, J)
    return lo, hi


def ratio_bounds_independent(poly, a, b_table, scope, J):
    """草案 16：不给分子/分母做联合投影，只各自取区间再相除。

    这是合法但更保守的区间（草案 8.4 明确：不能声称分子区间/分母区间总是最紧）。
    """
    loA, hiA = min_max_A(poly, a, scope, J)
    loB, hiB = min_max_B(poly, b_table, scope, J)
    if hiB <= 1e-12:
        return None, None                  # undefined_denominator
    if loB <= 1e-12:
        return 0.0, 1.0                    # 可行域含零分母 → 最宽，继续抽样
    # 比例必在 [0,1] 内（A <= B）：任何合理的独立区间实现都应当裁剪到该范围
    return max(0.0, loA / hiB), min(1.0, hiA / loB)


# --------------------------------------------------------------------------- #
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
    if args.workload == "closure":
        wl = make_wands_closure_workload(by_rating, {"all": meta["all_strata"]}, K=2)
    else:
        wl = make_wands_workload(by_rating, {"all": meta["all_strata"]}, K=2)
    for q in wl:
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2,
                                         tables={"a": q.tables["a2"]})
    return frame, snap, contract, wl, kappa, gold, meta


def evaluate(res, workload, theta_M, theta_G):
    """与主脚本完全一致的评测口径（覆盖率/MAE/语义误差）。"""
    cov_n = cov_d = mae_n = mae_d = 0
    sems = []
    for q in workload:
        r = res[q.qid]
        thM, thG = theta_M.get(q.qid), theta_G.get(q.qid)
        if thG is not None and thM is not None:
            sems.append(abs(thM - thG))
        if r.status.startswith("certified") and thM is not None:
            cov_d += 1
            cov_n += int(r.L - 1e-9 <= thM <= r.U + 1e-9)
        if r.point is not None and thM is not None:
            mae_n += abs(r.point - thM); mae_d += 1
    return (cov_n / cov_d if cov_d else float("nan"),
            mae_n / mae_d if mae_d else float("nan"),
            float(np.mean(sems)) if sems else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", choices=["baseline", "independent_ratio",
                                           "no_shared_state",
                                           "no_basis_compilation"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", choices=["basic", "closure"], default="closure")
    ap.add_argument("--budget-fracs", default="0.02,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    frame, snap, contract, wl, kappa, gold, meta = build_context(args)
    print(f"消融 {args.ablation}｜查询 {len(wl)}｜N={len(frame)}｜κ={kappa[:16]}")

    # θ_M / θ_gold（与主脚本同一口径）
    ids_all = frame["id"].to_numpy()
    lab = contract.judge(list(ids_all), snap.texts(list(ids_all)))
    labels_M = {int(i): lab[r] for r, i in enumerate(ids_all)}
    theta_M = {q.qid: rrd._query_truth(q, labels_M, ids_all,
                                       dict(zip(frame["id"].astype(int),
                                               frame["stratum"].astype(int))))
               for q in wl}
    theta_G = {q.qid: rrd._query_truth(q, gold["labels"], ids_all,
                                       dict(zip(frame["id"].astype(int),
                                               frame["stratum"].astype(int))))
               for q in wl} if gold.get("labels") else {}

    if args.ablation == "independent_ratio":
        P.ratio_bounds = ratio_bounds_independent

    # 无统计基编译消融：跳过去重，对每个查询的每个分子/分母事件单独建 CS。
    # 直接改 Config.compile_basis，pipeline 在编译基时据此分支（草案 8.2 / §16）。
    compile_basis_on = (args.ablation != "no_basis_compilation")

    fracs = [float(x) for x in args.budget_fracs.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    print(f"\n{'预算':>6}{'调用':>9}{'认证':>12}{'覆盖(θ_M)':>11}{'MAE(θ_M)':>10}{'基函数':>8}")
    print("-" * 60)
    for frac in fracs:
        calls_l, cert_l = [], []
        cov_n = cov_d = mae_n = mae_d = 0
        n_basis_l = []
        for sd in seeds:
            cfg = Config()
            cfg.compile_basis = compile_basis_on
            if args.ablation == "no_shared_state":
                # 每个查询独立执行：预算按查询数均分，状态不共享（只共享标签缓存）
                per = max(1, int(len(frame) * frac / max(1, len(wl))))
                res, tot = {}, 0
                for q in wl:
                    pipe = COHORTRAG(snap, contract, cfg, seed=sd)
                    r = pipe.run([q], RoundRobinAllocator(), budget_total=per)
                    res.update(r)
                    tot += pipe.calls
                calls_l.append(tot)
            else:
                pipe = COHORTRAG(snap, contract, cfg, seed=sd)
                res = pipe.run(wl, RoundRobinAllocator(),
                               budget_total=max(1, int(len(frame) * frac)))
                calls_l.append(pipe.calls)
            # 基函数个数（编译后 / 未编译）：衡量编译基的压缩比
            n_basis_l.append(pipe.risk_ledger.get("n_cs", 0))
            cert_l.append(sum(1 for r in res.values()
                              if r.status.startswith("certified")))
            c, m, _ = evaluate(res, wl, theta_M, theta_G)
            if not np.isnan(c):
                cov_n += c; cov_d += 1
            if not np.isnan(m):
                mae_n += m; mae_d += 1
        cov = cov_n / cov_d if cov_d else float("nan")
        mae = mae_n / mae_d if mae_d else float("nan")
        print(f"{frac:>6.2f}{np.mean(calls_l):>9.0f}"
              f"{np.mean(cert_l):>9.1f}±{np.std(cert_l):<4.1f}{cov:>11.3f}{mae:>10.4f}"
              f"{np.mean(n_basis_l):>8.0f}")
        rows.append({"ablation": args.ablation, "budget_fraction": frac,
                     "calls_mean": float(np.mean(calls_l)),
                     "certified_mean": float(np.mean(cert_l)),
                     "certified_std": float(np.std(cert_l)),
                     "coverage_vs_thetaM": cov, "mae_vs_thetaM": mae,
                     "n_basis_mean": float(np.mean(n_basis_l)),
                     "n_seeds": len(seeds)})
    print("-" * 52)

    out = os.path.join(_HERE, "..", "reports",
                       f"ablation_{args.ablation}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ablation": args.ablation, "workload": args.workload,
                   "n_queries": len(wl), "kappa": kappa, "seeds": seeds,
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
