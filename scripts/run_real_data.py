# -*- coding: utf-8 -*-
"""真实数据端到端脚本（草案 4 / 13 / 15）——CFPB、WANDS、UCI。

先自检（不需要真实数据、不需要 GPU）：
    python scripts/run_real_data.py --dataset cfpb --dry-run
真实运行（需要 GPU + 数据文件 + 许可清单）：
    python scripts/run_real_data.py --dataset cfpb --data C:\\data\\complaints.csv ^
        --manifest data_manifests\\cfpb.json --model Qwen/Qwen3-8B ^
        --budget-fracs 0.05,0.2,0.5,1.0

流程：载入并冻结快照 → 物理层与真实工作负载（草案 13.2）→ 语义算子
      （真实 = 本地 Qwen3-8B + κ 版本化缓存；dry-run = 确定性噪声替身）→
      运行 M 与 B6 → 逐预算档评测（对 θ_M 的覆盖率、MAE、对人工标签的语义误差、token 账本）。
"""
import sys, os, json, time, argparse

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np
from cohortrag import (Snapshot, COHORTRAG, Config, RoundRobinAllocator,
                       WorstVarianceAllocator, states)
from cohortrag.workload import (make_cfpb_workload, make_wands_workload,
                                make_wands_closure_workload, Query)
from cohortrag.datasets import load_cfpb, load_wands, load_uci
from cohortrag import contracts as _contracts

# --------------------------------------------------------------------------- #
def _load_contract(args, snap, gold):
    if args.dry_run:
        from cohortrag.semantics import NoisyOracleOperator
        return NoisyOracleOperator(gold["hidden"], flip_probs=(0.08, 0.08, 0.10, 0.06),
                                   name="noisy-stand-in"), None
    from cohortrag.llm_operator import LocalLLMOperator, CachedSemanticOperator
    pack = (_contracts.cfpb_prompt_pack() if args.dataset == "cfpb"
            else _contracts.uci_prompt_pack() if args.dataset == "uci"
            else _contracts.wands_prompt_pack())
    inner = LocalLLMOperator(model_id=args.model, revision=args.revision,
                             K=len(pack["primitives"]), prompt_pack=pack,
                             batch_size=args.batch_size, device=args.device,
                             name=f"qwen3-{args.dataset}")
    cached = CachedSemanticOperator(inner, snap.manifest,
                                    cache_path=os.path.join(_HERE, "..", "cache",
                                                            f"labels_{args.dataset}.sqlite"))
    return cached, cached.kappa

def _dry_run_frames(name, N=1400):
    """合成替身：列结构与真实数据对齐，从而走同一套分层/工作负载/评测代码。"""
    from cohortrag.worldgen import make_synthetic_world
    frame, hidden = make_synthetic_world(N, 4, 4, seed=0)
    frame = frame.rename(columns={"group": "Product", "quantile": "half"})
    frame["Product"] = frame["Product"].map({0: "信用卡", 1: "支票账户"})
    frame["half"] = frame["half"].map({0: "2024H1", 1: "2024H2"})
    r = np.random.default_rng(0)
    frame["average_rating"] = r.choice([np.nan, 3.2, 4.4], size=len(frame), p=[.15, .35, .50])
    frame["rating_bucket"] = np.where(np.isnan(frame["average_rating"]), "missing",
                                      np.where(frame["average_rating"] >= 4, ">=4", "<4"))
    gold = {"hidden": hidden, "labels": {int(i): hidden[int(i)] for i in hidden}}
    meta = {"strata_by_product": {p: sorted(frame.loc[frame["Product"] == p, "stratum"]
                                            .unique().tolist())
                                  for p in frame["Product"].unique()},
            "strata_by_half": {h: sorted(frame.loc[frame["half"] == h, "stratum"]
                                         .unique().tolist())
                               for h in frame["half"].unique()},
            "strata_by_rating": {b: sorted(frame.loc[frame["rating_bucket"] == b, "stratum"]
                                           .unique().tolist())
                                 for b in frame["rating_bucket"].unique()},
            "all_strata": sorted(frame["stratum"].unique().tolist()),
            "note": "dry-run 合成替身：不得作为真实数据结果（草案 12.3）"}
    return frame, gold, meta

def _real_frames(args):
    if args.dataset == "cfpb":
        frame, info = load_cfpb(args.data, args.manifest, start=args.start, end=args.end,
                                max_records=args.max_records, id_col=args.id_col,
                                date_col=args.date_col, narrative_col=args.narrative_col,
                                product_col=args.product_col)
        frame["Product"] = frame[args.product_col].astype(str)
        frame["half"] = frame["half"].astype(str)
        meta = {**info,
                "strata_by_product": {p: sorted(frame.loc[frame["Product"] == p, "stratum"]
                                                .unique().tolist())
                                      for p in frame["Product"].unique()},
                "strata_by_half": {h: sorted(frame.loc[frame["half"] == h, "stratum"]
                                             .unique().tolist())
                                   for h in frame["half"].unique()},
                "all_strata": sorted(frame["stratum"].unique().tolist())}
        return frame, {"hidden": {}, "labels": {}}, meta
    if args.dataset == "wands":
        frame, gold_labels, info = load_wands(args.data, args.manifest,
                                              max_intents=args.max_intents, seed=0)
        fb = frame["rating_bucket"].astype(str)
        meta = {**info, "strata_by_rating": {b: sorted(frame.loc[fb == b, "stratum"]
                                                       .unique().tolist())
                                             for b in fb.unique()},
                "all_strata": sorted(frame["stratum"].unique().tolist())}
        return frame, {"hidden": {}, "labels": gold_labels}, meta
    frame, gold_labels, info = load_uci(args.data, args.manifest)
    meta = {**info, "strata_by_group": {g: sorted(frame.loc[frame["synth_group"] == g,
                                                             "stratum"].unique().tolist())
                                        for g in frame["synth_group"].unique()},
            "all_strata": sorted(frame["stratum"].unique().tolist())}
    return frame, {"hidden": {}, "labels": gold_labels}, meta

def _build_workload(args, meta, K):
    if args.dataset == "cfpb":
        return make_cfpb_workload(meta["strata_by_product"], meta["strata_by_half"], K=4)
    if args.dataset == "wands":
        if getattr(args, "workload", "basic") == "closure":
            return make_wands_closure_workload(meta["strata_by_rating"],
                                               {"all": meta["all_strata"]}, K=2)
        return make_wands_workload(meta["strata_by_rating"],
                                   {"all": meta["all_strata"]}, K=2)
    return [Query(qid=1, name="uci theta(positive|all)", scope=tuple(meta["all_strata"]),
                  d_idx=(), r_idx=(0,), kind="value", eps=0.10)]

def _ratio_side(a_hz, b_table, scope, labels_by_id, ids, stratum_of):
    """单侧条件比例 θ = A/B（分子系数 a_hz 已含元数据条件）。"""
    A = B = 0
    for i in ids:
        h = stratum_of[int(i)]
        if h not in a_hz:
            continue
        z = states.encode_state(tuple(int(v) for v in labels_by_id[int(i)]))
        A += int(round(a_hz[h][z]))
        B += int(b_table[z])
    return A / B if B > 0 else None

def _query_truth(q, labels_by_id, ids, stratum_of):
    """查询的真实值：普通查询=θ；差异查询=θ_A − θ_B（与 pipeline 的 Δ 口径一致）。"""
    ta = _ratio_side(q.a_hz, q.tables["b"], q.scope, labels_by_id, ids, stratum_of)
    if q.kind == "difference":
        tb = _ratio_side(q.a_hz2, q.tables["b2"], q.scope2, labels_by_id, ids, stratum_of)
        if ta is None or tb is None:
            return None
        return ta - tb
    return ta

# --------------------------------------------------------------------------- #

def _paired_stats(d):
    """配对差的均值、95% CI（正态近似）与双侧 p 值（不依赖 scipy）。"""
    import math
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    if not np.isfinite(se) or se == 0:
        z = 0.0 if mean == 0 else float("inf") * (1 if mean > 0 else -1)
        p = 0.0 if mean != 0 else 1.0
    else:
        z = mean / se
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"mean_diff": mean, "se": se, "n": n,
            "ci_low": mean - 1.96 * se if np.isfinite(se) else mean,
            "ci_high": mean + 1.96 * se if np.isfinite(se) else mean,
            "z": float(z), "p": float(p)}


def _holm(items, alpha=0.05):
    """Holm 逐步校正：控制族错误率，比 Bonferroni 更有功效。"""
    order = sorted(range(len(items)), key=lambda i: items[i]["p"])
    m = len(items)
    passed = True
    for rank, i in enumerate(order):
        thr = alpha / (m - rank)
        rej = items[i]["p"] <= thr and passed
        items[i]["holm_reject"] = bool(rej)
        items[i]["holm_threshold"] = thr
        if not rej:
            passed = False
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfpb", "wands", "uci"], default="cfpb")
    ap.add_argument("--data", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--budget-fracs", default="0.05,0.2,0.5,1.0")
    ap.add_argument("--seeds", default="0", help="逗号分隔的抽样种子，如 0,1,2,3,4")
    ap.add_argument("--workload", choices=["basic", "closure"], default="basic",
                    help="basic=7 个查询；closure=同闭包内扩到数十个查询（草案 8.5）")
    ap.add_argument("--max-input-tokens", type=int, default=260_000_000)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--max-records", type=int, default=50000)
    ap.add_argument("--max-intents", type=int, default=120)
    ap.add_argument("--id-col", default="Complaint ID")
    ap.add_argument("--date-col", default="Date received")
    ap.add_argument("--narrative-col", default="Consumer complaint narrative")
    ap.add_argument("--product-col", default="Product")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print(f"COHORT-RAG 真实数据端到端｜dataset={args.dataset}｜dry_run={args.dry_run}")
    print("=" * 92)

    # 1) 数据
    if args.dry_run:
        frame, gold, meta = _dry_run_frames(args.dataset)
        print("[1] dry-run 合成替身（列结构与真实数据对齐；结果不得当作真实数据结果）")
    else:
        if not args.data or not args.manifest:
            raise SystemExit("真实运行必须提供 --data 与 --manifest（草案 23.1）")
        frame, gold, meta = _real_frames(args)
        print(f"[1] 载入 {args.dataset}：{len(frame)} 条记录，{frame['stratum'].nunique()} 层")
        print(f"    许可：{meta['manifest']['license']}")
    snap = Snapshot(frame, name=f"{args.dataset}-{'dryrun' if args.dry_run else 'frozen'}")
    print(f"    快照 manifest {snap.manifest[:24]}…")

    # 2) 语义算子
    contract, kappa = _load_contract(args, snap, gold)
    print(f"[2] 语义算子 {contract.name}" + (f"（κ={kappa[:16]}…）" if kappa else ""))

    # 3) 工作负载
    K = contract.K
    workload = _build_workload(args, meta, K)
    for q in workload:
        q.tables = q.build_tables(K)
        q.a_hz = q.numerator_tables(K, q.scope)          # 与 pipeline 内部同一口径
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(K, q.scope2, meta=q.r_meta2,
                                         tables={"a": q.tables["a2"]})
    print(f"[3] 工作负载 {len(workload)} 个查询（草案 13.2 六类问题）")

    ids_all = frame["id"].to_numpy()
    stratum_of = dict(zip(frame["id"].astype(int), frame["stratum"].astype(int)))

    # 4) 全量物化参考 θ_M（只做一次；这就是 B0 的成本）
    if hasattr(contract, "count_tokens"):
        est_in = sum(contract.count_tokens(frame["text"].iloc[s:s + 512].tolist())
                     for s in range(0, len(ids_all), 512))
        print(f"[4] 全量物化输入 token ≈ {est_in/1e6:.1f}M（草案 18 上限 "
              f"{args.max_input_tokens/1e6:.0f}M）")
        if est_in > args.max_input_tokens:
            raise SystemExit("超出草案 18 的 token 硬预算：请缩小规模（不得伪造预算）")
    # 注意：必须传文本视图。ledger 内部是 contract.judge(ids, snapshot.texts(ids))；
    # 这里若传 None，CachedSemanticOperator 会下标 None 直接崩。
    full_lab = contract.judge(list(ids_all), snap.texts(list(ids_all)))
    labels_M = {int(i): full_lab[r] for r, i in enumerate(ids_all)}
    theta_M = {q.qid: _query_truth(q, labels_M, ids_all, stratum_of) for q in workload}
    theta_G = {}
    if gold["labels"]:
        theta_G = {q.qid: _query_truth(q, gold["labels"], ids_all, stratum_of)
                   for q in workload}
    print(f"    θ_M 已构建（成本 {len(ids_all)} 次语义调用）；"
          + ("θ_gold 已加载（人工标签）" if theta_G else "本数据集无现成人标"))

    # 5) 运行 + 评测（多种子：草案 15.3 要求重复抽样以评估随机不确定性）
    fracs = [float(x) for x in args.budget_fracs.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = Config()
    rows = []
    agg = {}
    print(f"\n种子 {seeds}（草案 15.3 重复抽样）")
    print(f"\n{'分配器':<6}{'预算':>6}{'调用':>7}{'认证':>9}{'未决':>6}{'覆盖(θ_M)':>11}"
          f"{'MAE(θ_M)':>10}{'语义误差':>10}")
    print("-" * 92)
    for a_name, alloc in (("M", WorstVarianceAllocator()), ("B6", RoundRobinAllocator())):
        for frac in fracs:
            cov_n = cov_d = mae_n = mae_d = 0
            sems, calls_l, cert_l, und_l = [], [], [], []
            # 草案 15.1 扩展指标：整组同时覆盖率、每次运行至少一次错误认证的频率、
            # 平均区间宽度、状态计数（undefined_denominator 等单独统计）
            run_all_covered, run_any_false, width_l = [], [], []
            status_tot = {}
            for sd in seeds:
                pipe = COHORTRAG(snap, contract, cfg, seed=sd)
                res = pipe.run(workload, alloc,
                               budget_total=max(1, int(len(frame) * frac)))
                n_cert_run = 0
                all_cov = True
                any_false = False
                for q in workload:
                    r = res[q.qid]
                    thM, thG = theta_M[q.qid], theta_G.get(q.qid)
                    r.theta_M, r.theta_gold = thM, thG
                    status_tot[r.status] = status_tot.get(r.status, 0) + 1
                    if thG is not None and thM is not None:
                        sems.append(abs(thM - thG))
                    if r.status.startswith("certified"):
                        n_cert_run += 1
                        if thM is not None:
                            cov_d += 1
                            ok = bool(r.L - 1e-9 <= thM <= r.U + 1e-9)
                            cov_n += int(ok)
                            if not ok:
                                all_cov = False
                                any_false = True
                    if r.point is not None and thM is not None:
                        mae_n += abs(r.point - thM); mae_d += 1
                    if r.L is not None and r.U is not None:
                        width_l.append(float(r.U - r.L))
                if n_cert_run > 0:
                    run_all_covered.append(1 if all_cov else 0)
                    run_any_false.append(1 if any_false else 0)
                calls_l.append(pipe.calls)
                cert_l.append(n_cert_run)
                und_l.append(sum(1 for r in res.values() if r.status == "undecided"))
            cov = cov_n / cov_d if cov_d else float("nan")
            mae = mae_n / mae_d if mae_d else float("nan")
            semv = float(np.mean(sems)) if sems else float("nan")
            print(f"{a_name:<6}{frac:>6.2f}{np.mean(calls_l):>7.0f}"
                  f"{np.mean(cert_l):>7.1f}±{np.std(cert_l):<3.1f}{np.mean(und_l):>6.1f}"
                  f"{cov:>11.3f}{mae:>10.4f}{semv:>10.4f}")
            rec = {"allocator": a_name, "budget_fraction": frac,
                   "calls_mean": float(np.mean(calls_l)),
                   "calls_std": float(np.std(calls_l)),
                   "certified_mean": float(np.mean(cert_l)),
                   "certified_std": float(np.std(cert_l)),
                   "certified_per_seed": cert_l,
                   "undecided_mean": float(np.mean(und_l)),
                   "coverage_vs_thetaM": cov, "mae_vs_thetaM": mae,
                   "semantic_error_vs_gold": semv, "n_seeds": len(seeds),
                   # 草案 15.1 扩展指标
                   "set_coverage_rate": (float(np.mean(run_all_covered))
                                         if run_all_covered else float("nan")),
                   "run_false_certification_rate": (float(np.mean(run_any_false))
                                                    if run_any_false else float("nan")),
                   "mean_interval_width": (float(np.mean(width_l)) if width_l
                                           else float("nan")),
                   "interval_width_std": (float(np.std(width_l)) if width_l
                                          else float("nan")),
                   "status_counts": status_tot}
            rows.append(rec)
            agg[f"{a_name}@{frac}"] = rec
    print("-" * 92)
    print("认证列格式：均值±标准差（跨种子）")

    # ---- 扩展指标表（草案 15.1）----
    print(f"\n扩展指标（草案 15.1）")
    print(f"{'分配器':<6}{'预算':>6}{'整组同时覆盖':>13}{'运行级误认证':>13}"
          f"{'平均区间宽':>11}{'未定义分母':>11}{'执行失败':>9}")
    print("-" * 70)
    for r in rows:
        sc = r.get("status_counts", {})
        print(f"{r['allocator']:<6}{r['budget_fraction']:>6.2f}"
              f"{r['set_coverage_rate']:>13.3f}"
              f"{r['run_false_certification_rate']:>13.3f}"
              f"{r['mean_interval_width']:>11.4f}"
              f"{sc.get('undefined_denominator', 0):>11d}"
              f"{sc.get('execution_error', 0):>9d}")
    print("-" * 70)

    # ---- 配对比较 M vs B6 + Holm 多重比较控制（草案 15.3）----
    paired = []
    for frac in fracs:
        m = next((r for r in rows if r["allocator"] == "M"
                  and r["budget_fraction"] == frac), None)
        b = next((r for r in rows if r["allocator"] == "B6"
                  and r["budget_fraction"] == frac), None)
        if not m or not b:
            continue
        a = np.asarray(m["certified_per_seed"], dtype=float)
        c = np.asarray(b["certified_per_seed"], dtype=float)
        n = min(len(a), len(c))
        d = a[:n] - c[:n]
        res_p = _paired_stats(d)
        res_p["budget_fraction"] = frac
        paired.append(res_p)
    holm = _holm(paired)
    print(f"\n配对比较 M − B6（认证答案数，逐种子配对；草案 15.3）")
    print(f"{'预算':>6}{'差值':>9}{'95% CI':>20}{'p 值':>10}{'Holm 后显著':>12}")
    print("-" * 60)
    for r in holm:
        ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"{r['budget_fraction']:>6.2f}{r['mean_diff']:>9.2f}{ci:>20}"
              f"{r['p']:>10.4f}{('是' if r['holm_reject'] else '否'):>12}")
    print("-" * 60)

    with open(os.path.join(_HERE, "..", "reports", "main_metrics_extended.json"),
              "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "workload": getattr(args, "workload", "basic"),
                   "seeds": seeds, "rows": rows, "paired_M_vs_B6": holm},
                  f, ensure_ascii=False, indent=2)
    if not theta_G:
        print("注意：CFPB 无现成人标，语义误差列不适用——"
              "草案 12.2 要求『执行近似误差』与『模型语义误差』分开报告。")

    # 6) 报告
    outp = os.path.join(_HERE, "..", "reports", f"real_{args.dataset}_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"honest_scope": "dry-run 合成替身" if args.dry_run else "真实数据",
                   "dataset": args.dataset, "dry_run": args.dry_run,
                   "snapshot_manifest": snap.manifest, "kappa": kappa,
                   "n_records": len(frame), "n_strata": int(frame["stratum"].nunique()),
                   "n_queries": len(workload),
                   "theta_M": {str(k): v for k, v in theta_M.items()},
                   "theta_gold": {str(k): v for k, v in theta_G.items()},
                   "rows": rows, "seeds": seeds, "info_note": meta.get("note", "")},
                  f, ensure_ascii=False, indent=2)
    print(f"报告：{outp}")
    print(f"总耗时 {time.time() - t0:.1f} 秒")

if __name__ == "__main__":
    main()
