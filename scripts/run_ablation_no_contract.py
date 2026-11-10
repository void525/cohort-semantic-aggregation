# -*- coding: utf-8 -*-
"""消融：去除「选择来源契约」（草案 16，🔴 必须补）。

背景（草案 7.2/7.3）
--------------------
系统把历史已见记录 E 冻结为**精确已知质量**，只对剩余总体 R 建置信序列并偏移 E_f。
本消融回答：如果不做这个区分、把历史缓存当随机样本用，会怎样？

三种方法（同一份偏置历史缓存、同一份预算、同一批暴露记录，只有推断方式不同）
--------------------------------------------------------------------------
1. correct    —— 契约 intact：E 精确已知，CS 作用于 R，偏移 E_f。
2. structured —— 去除契约但保留分层 CS 机制：把缓存记录当作本层随机排列的前缀。
                 分层结构仍在，因此期望：覆盖不掉，但**认证能力下降**（区间变宽）。
3. naive      —— 完全忽略分层与选择来源：把所有暴露记录池化为简单随机样本，
                 Wald 95% CI。这是实践中最容易犯的错误。

认证规则与 pipeline._classify 严格一致（否则不可比）：
  value      : U - L <= 2*eps
  threshold  : L > tau -> certified_true ; U <= tau -> certified_false
  difference : dU - dL <= 2*eps
覆盖率定义（与主实验一致）：在被认证的答案中，θ_M 是否落在 [L, U] 内。

用法（仓库根目录，需 cache/labels_wands.sqlite 与 WANDS 数据；不调用 LLM 之外的推理，
实际全部走冻结标签带，因此 CPU 即可跑）：
    python scripts/run_ablation_no_contract.py \
        --data /root/autodl-tmp/WANDS/dataset \
        --manifest data_manifests/wands.json \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --workload closure --cache-frac 0.3 \
        --budget-fracs 0.02,0.05,0.1,0.2,0.5,1.0 --seeds 0,1,2,3,4
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
import run_ablations as RA
import run_real_data as rrd
from cohortrag import COHORTRAG, Config, RoundRobinAllocator


# --------------------------------------------------------------------------- #
# naive 方法：把暴露记录池化为简单随机样本，用 Wald CI，认证规则与 _classify 一致
# --------------------------------------------------------------------------- #
def _wald(A, B):
    """简单随机样本下的 Wald 95% 区间（即错误假设：暴露记录是均匀随机样本）。"""
    if B <= 0:
        return None
    th = A / B
    se = (th * (1.0 - th) / B) ** 0.5
    return max(0.0, th - 1.96 * se), min(1.0, th + 1.96 * se)


def _naive_side(q, labels_M, ids, stratum_of, second=False):
    """naive 单侧比率 + Wald 区间。second=True 时取 difference 的第二侧。"""
    if not second:
        a_hz, b, scope = q.a_hz, q.tables["b"], q.scope
    else:
        a_hz, b, scope = q.a_hz2, q.tables["b2"], q.scope2
    A = B = 0
    for i in ids:
        h = stratum_of[int(i)]
        if h not in a_hz:
            continue
        from cohortrag import states
        z = states.encode_state(tuple(int(v) for v in labels_M[int(i)]))
        A += int(round(a_hz[h][z]))
        B += int(b[z])
    return A, B, _wald(A, B)


def naive_eval(wl, exposed_ids, labels_M, stratum_of, theta_M):
    """返回 (认证数, 覆盖数, 覆盖分母)。认证规则与 pipeline._classify 一致。"""
    cert = cov_n = cov_d = 0
    for q in wl:
        thM = theta_M.get(q.qid)
        if thM is None:
            continue
        if q.kind == "difference":
            _, _, iv1 = _naive_side(q, labels_M, exposed_ids, stratum_of, False)
            _, _, iv2 = _naive_side(q, labels_M, exposed_ids, stratum_of, True)
            if iv1 is None or iv2 is None:
                continue
            # Δ 的 Wald 区间：两侧独立相加（naive 分析师能做到的最好做法）
            th = iv1[0] - iv2[1], iv1[1] - iv2[0]
            mid = (th[0] + th[1]) / 2
            half = (th[1] - th[0]) / 2
            L, U = mid - half, mid + half
            if U - L > 2 * q.eps:
                continue                      # 未认证
            cert += 1
            cov_d += 1
            cov_n += int(L - 1e-9 <= thM <= U + 1e-9)
            continue
        _, _, iv = _naive_side(q, labels_M, exposed_ids, stratum_of)
        if iv is None:
            continue
        L, U = iv
        if q.kind == "value":
            if U - L > 2 * q.eps:
                continue                      # 未认证
            cert += 1
            cov_d += 1
            cov_n += int(L - 1e-9 <= thM <= U + 1e-9)
        elif q.kind == "threshold":
            if L > q.tau:
                pred = True
            elif U <= q.tau:
                pred = False
            else:
                continue                      # 未认证（区间跨阈值）
            cert += 1
            cov_d += 1
            cov_n += int(pred == (thM > q.tau))
    return cert, cov_n, cov_d


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", choices=["basic", "closure"], default="closure")
    ap.add_argument("--cache-frac", type=float, default=0.3,
                    help="历史缓存规模（占总体比例）")
    ap.add_argument("--cache-mode", choices=["rr", "targeted"], default="rr",
                    help="rr=分配器暴露（层内均匀随机，与标签无关）；"
                         "targeted=模拟定向检索（按是否命中分子条件倾斜挑选，标签依赖）")
    ap.add_argument("--budget-fracs", default="0.02,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    frame, snap, contract, wl, kappa, gold, meta = RA.build_context(args)
    print(f"选择来源契约消融｜查询 {len(wl)}｜N={len(frame)}｜κ={kappa[:16]}")

    ids_all = frame["id"].to_numpy()
    stratum_of = dict(zip(frame["id"].astype(int), frame["stratum"].astype(int)))
    lab = contract.judge(list(ids_all), snap.texts(list(ids_all)))
    labels_M = {int(i): lab[r] for r, i in enumerate(ids_all)}
    theta_M = {q.qid: rrd._query_truth(q, labels_M, ids_all, stratum_of) for q in wl}
    theta_G = ({q.qid: rrd._query_truth(q, gold["labels"], ids_all, stratum_of)
                for q in wl} if gold.get("labels") else {})

    # --- 1) 构造历史缓存 ---
    # rr：由真实分配器暴露。注意其层内是【均匀随机前缀】，与标签无关（实测 T=0），
    #     因此"当随机样本"在层内其实合法——structured 不失覆盖是预期结果。
    # targeted：模拟定向检索——按"是否命中分子条件"倾斜挑选，与标签强相关。
    #     这才是选择来源契约真正要防的场景（草案 7.3）。
    if args.cache_mode == "rr":
        pipe0 = COHORTRAG(snap, contract, Config(), seed=0)
        pipe0.run(wl, RoundRobinAllocator(),
                  budget_total=max(1, int(len(frame) * args.cache_frac)))
        hist = {}
        for h in pipe0.ledger.strata:
            for i in pipe0.ledger.exposed_ids(h):
                hist[int(i)] = labels_M[int(i)]
    else:
        rng = np.random.default_rng(7)
        target_n = max(1, int(len(frame) * args.cache_frac))
        # 关键：必须按【分子】倾斜，而不是按分母条件。若只放大分母（如 p1=1），
        # 分子与分母同比放大，条件比率 P(完全匹配|相关) 反而近似无偏——抽不出偏置。
        # 按"相关且完全匹配"（即分子）倾斜，比率才会真正被推高。
        hit = [int(i) for i in ids_all
               if int(labels_M[int(i)][0]) == 1 and int(labels_M[int(i)][1]) == 1]
        miss = [int(i) for i in ids_all
                if not (int(labels_M[int(i)][0]) == 1 and int(labels_M[int(i)][1]) == 1)]
        # 检索器：命中分子条件者以 0.8 概率入选，未命中者以 0.1 概率入选
        n_hit = min(int(round(0.8 * target_n)), len(hit))
        n_miss = min(target_n - n_hit, len(miss))
        sel = list(rng.choice(hit, size=n_hit, replace=False)) + \
              list(rng.choice(miss, size=n_miss, replace=False))
        hist = {int(i): labels_M[int(i)] for i in sel}
    if not hist:
        print("ERROR: 历史缓存为空，请调大 --cache-frac")
        sys.exit(1)

    # 缓存的"命中分子"比例 vs 总体：直接量化标签依赖程度
    def hit_rate(d):
        """分子命中率（相关且完全匹配）——即查询比率 P(完全匹配|相关) 的分子占比。"""
        n = len(d)
        if not n:
            return float("nan")
        return sum(1 for i in d
                   if int(labels_M[int(i)][0]) == 1 and int(labels_M[int(i)][1]) == 1) / n
    pop_rate = hit_rate(ids_all)
    by_h = {}
    for i in hist:
        by_h.setdefault(stratum_of[int(i)], 0)
        by_h[stratum_of[int(i)]] += 1
    bias = {int(h): round(by_h.get(h, 0) / max(1, len(snap.stratum_ids(h))), 3)
            for h in snap.strata()}
    print(f"历史缓存[{args.cache_mode}]：{len(hist)}/{len(frame)}"
          f"（{len(hist)/len(frame):.1%}）")
    print(f"各层缓存占比：{bias}")
    print(f"缓存命中分子比例 {hit_rate(list(hist)):.3f} vs 总体 {pop_rate:.3f}"
          f"（差值 {hit_rate(list(hist)) - pop_rate:+.3f}＝标签依赖强度）")

    fracs = [float(x) for x in args.budget_fracs.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = {
        "ablation": "no_source_contract",
        "kappa": kappa, "workload": args.workload, "n_queries": len(wl),
        "N": int(len(frame)),
        "cache_frac": args.cache_frac, "cache_mode": args.cache_mode,
        "hist_cache_size": len(hist),
        "hist_cache_by_stratum_frac": bias,
        "hist_cache_hit_rate": hit_rate(list(hist)),
        "population_hit_rate": pop_rate,
        "label_dependence": hit_rate(list(hist)) - pop_rate,
        "seeds": seeds,
        "methods": {
            "correct": "契约 intact：E 精确已知，CS 作用于剩余总体 R",
            "structured": "去除契约但保留分层 CS：缓存当作本层随机排列前缀",
            "naive": "忽略分层与选择来源：池化为简单随机样本 + Wald CI",
        },
        "rows": [],
    }

    print(f"\n{'方法':>11}{'预算':>7}{'调用':>8}{'认证':>12}{'覆盖(θ_M)':>11}")
    print("-" * 50)
    for mode in ("correct", "structured", "naive"):
        for frac in fracs:
            calls_l, cert_l = [], []
            cov_num = cov_den = 0
            for sd in seeds:
                pipe = COHORTRAG(snap, contract, Config(), seed=sd,
                                 cache_labels=hist,
                                 e_as_random=(mode == "structured"))
                res = pipe.run(wl, RoundRobinAllocator(),
                               budget_total=max(1, int(len(frame) * frac)))
                calls_l.append(pipe.calls)
                if mode == "naive":
                    # 同一批暴露记录、同样调用数，只是推断方式错误
                    exp = sorted({int(i) for h in pipe.ledger.strata
                                  for i in pipe.ledger.exposed_ids(h)})
                    c, cn, cd = naive_eval(wl, exp, labels_M, stratum_of, theta_M)
                    cert_l.append(c)
                    cov_num += cn
                    cov_den += cd
                else:
                    cert_l.append(sum(1 for r in res.values()
                                      if r.status.startswith("certified")))
                    cov, _, _ = RA.evaluate(res, wl, theta_M, theta_G)
                    if not np.isnan(cov):
                        # evaluate 返回的是比率，需还原分子/分母权重；这里按查询数加权
                        n_cert = sum(1 for r in res.values()
                                     if r.status.startswith("certified"))
                        cov_num += cov * n_cert
                        cov_den += n_cert
            covv = cov_num / cov_den if cov_den else float("nan")
            print(f"{mode:>11}{frac:>7.2f}{np.mean(calls_l):>8.0f}"
                  f"{np.mean(cert_l):>8.1f}±{np.std(cert_l):<3.1f}{covv:>11.3f}")
            out["rows"].append({
                "method": mode, "budget_fraction": frac,
                "calls_mean": float(np.mean(calls_l)),
                "certified_mean": float(np.mean(cert_l)),
                "certified_std": float(np.std(cert_l)),
                "coverage_vs_thetaM": covv,
                "n_certified_total": int(cov_den),
                "n_seeds": len(seeds),
            })
    print("-" * 50)

    os.makedirs(os.path.join(_HERE, "..", "reports"), exist_ok=True)
    op = os.path.join(_HERE, "..", "reports",
                      f"ablation_no_source_contract_{args.cache_mode}.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"报告：{op}")


if __name__ == "__main__":
    main()
