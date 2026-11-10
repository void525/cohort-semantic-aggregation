# -*- coding: utf-8 -*-
"""
COHORT-RAG 端到端演示（草案 12.3：程序化压力测试，合成数据）。

⚠️ 诚实声明：以下所有结果来自【合成数据】（程序化生成的隐藏标签 +
确定性噪声算子），不得冒充真实数据结果。真实数据（CFPB/WANDS/UCI）
与真实 GPU 推理需要在草案第 17 节的硬件上、以 semantics.StubLLMOperator
接入本地模型后运行；本 pipeline 的其余部分完全不变。

演示内容：
  1) 主对比：M（最差归一化方差分配器） vs B6（round-robin 强控制），
     在 5 个预算档位 × 2 个种子下报告：调用数、认证覆盖率（对运行语义 θ_M）、
     未决数、区间宽度、MAE（对 θ_M）、语义误差（θ_M vs θ_gold）。
  2) 单一证据状态的可审计性：抽查一份草案第 11 节式声明与证书。
  3) 偏置缓存场景（草案 13.4 / 16 消融）：把『关键词匹配』偏置历史当 E 复用，
     对比『错误地把 E 当随机样本』的天真估计 vs 本方法的认证点估计。
"""
import sys, os, json, time, math

# --- 兼容 PyCharm / Windows 控制台：强制 UTF-8 输出 ---
# 中文 Windows 下 PyCharm 若以管道方式捕获输出，默认编码可能是 GBK，
# 打印中文会抛 UnicodeEncodeError；这里统一改为 UTF-8 并容错。
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

try:
    import numpy as np
    import pandas  # noqa: F401  （worldgen/snapshot 需要）
except ImportError as _e:
    raise SystemExit(
        "\n缺少依赖：{}\n"
        "请在 PyCharm 底部 Terminal 里执行：\n"
        "    pip install numpy pandas\n"
        "（当前解释器：{}）\n".format(_e.name, sys.executable))

try:
    from cohortrag import (Snapshot, COHORTRAG, Config, make_workload,
                           RoundRobinAllocator, WorstVarianceAllocator, states)
    from cohortrag.semantics import NoisyOracleOperator, OracleOperator
    from cohortrag.worldgen import make_synthetic_world
    from cohortrag.render import render_claim, template_text, certificate
except ImportError as _e:
    raise SystemExit(
        "\n无法导入 cohortrag 包：{}\n"
        "请确认在 PyCharm 里打开的是整个 cohort-rag 文件夹（scripts/ 同级必须有 src/cohortrag/），\n"
        "而不是只把这个 .py 文件单独拖进来。\n"
        "脚本所在目录：{}\n".format(_e, _HERE))

def safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")

def totals(tables, scope, labels, frame):
    A = B = 0
    strata = frame["stratum"].to_numpy()
    for i in range(len(frame)):
        if int(strata[i]) in scope:
            z = states.encode_state(tuple(int(v) for v in labels[i]))
            A += int(tables["a"][z])
            B += int(tables["b"][z])
    return A, B

def main():
    t_start = time.time()
    # 用法：python run_synthetic_demo.py          完整演示（约 3 分钟）
    #       python run_synthetic_demo.py --quick  快速自检（约 15 秒，用来确认环境能跑）
    QUICK = "--quick" in sys.argv
    N, K, H = (600, 4, 4) if QUICK else (2000, 4, 4)
    fractions = [0.5, 1.0] if QUICK else [0.1, 0.5, 0.75, 0.9, 1.0]
    seeds = [0] if QUICK else [0, 1]
    if QUICK:
        print("[快速自检模式 --quick：规模已缩小，数字与完整运行不可直接比较]")
    frame, hidden = make_synthetic_world(N, K, H, seed=0)
    snap = Snapshot(frame, name=f"synthetic-world-N{N}")
    contract = NoisyOracleOperator(hidden, flip_probs=(0.08, 0.08, 0.10, 0.06),
                                   name="noisy-semantics-v1")
    gold = OracleOperator(hidden, name="gold", K=K)
    workload = make_workload(K, seed=42)
    for q in workload:
        q.tables = q.build_tables(K)

    # 全量物化参考（B0 成本 = N 次语义调用）：θ_M（运行语义）与 θ_gold（真实标签）
    full_M = contract.judge(list(range(N)), None)
    full_g = gold.judge(list(range(N)), None)
    for q in workload:
        A1, B1 = totals(q.tables, q.scope, full_M, frame)
        A1g, B1g = totals(q.tables, q.scope, full_g, frame)
        if q.kind == "difference":
            A2, B2 = totals({"a": q.tables["a2"], "b": q.tables["b2"]}, q.scope2, full_M, frame)
            A2g, B2g = totals({"a": q.tables["a2"], "b": q.tables["b2"]}, q.scope2, full_g, frame)
            q.theta_M = A1 / B1 - A2 / B2 if (B1 and B2) else None
            q.theta_gold = A1g / B1g - A2g / B2g if (B1g and B2g) else None
        else:
            q.theta_M = A1 / B1 if B1 > 0 else None
            q.theta_gold = A1g / B1g if B1g > 0 else None

    cfg = Config()
    table_rows = []

    print("=" * 100)
    print("COHORT-RAG 合成数据端到端演示（草案 12.3 程序化压力测试；非真实数据结果）")
    print(f"总体 N={N}，原语 K={K}，层 H={H}；运行语义 = 真值 + 确定性噪声 p∈[0.06,0.10]")
    print(f"查询数 {len(workload)}；α=0.05；每轮新增调用上限 {cfg.max_new_calls}")
    print("=" * 100)
    print(f"{'分配器':<16}{'预算分位':>8}{'种子':>4}{'调用':>6}{'认证':>5}{'未决':>5}"
          f"{'覆盖(θ_M)':>11}{'MAE(θ_M)':>10}{'区间宽':>8}{'语义误差':>9}")
    print("-" * 100)

    for alloc_name, alloc in [("B6 roundrobin", RoundRobinAllocator()),
                              ("M worst-var", WorstVarianceAllocator())]:
        for frac in fractions:
            agg = {"calls": [], "cert": [], "cov": [], "mae": [], "width": [],
                   "sem": [], "undec": [], "err": []}
            for sd in seeds:
                pipe = COHORTRAG(snap, contract, cfg, seed=sd)
                budget = max(1, int(N * frac))
                results = pipe.run(workload, alloc, budget_total=budget)
                cert = [r for r in results.values()
                        if r.status in ("certified", "certified_true", "certified_false")]
                cov_n = cov_d = 0
                for r in cert:
                    thM = next(q.theta_M for q in workload if q.qid == r.qid)
                    if thM is None:
                        continue
                    cov_d += 1
                    if r.L - 1e-9 <= thM <= r.U + 1e-9:
                        cov_n += 1
                pts = [r.point for r in cert if r.point is not None]
                mae = float(np.mean([abs(r.point - next(q.theta_M for q in workload
                          if q.qid == r.qid)) for r in cert
                          if r.point is not None and
                          next(q.theta_M for q in workload if q.qid == r.qid) is not None])) \
                      if any(r.point is not None for r in cert) else float("nan")
                widths = [r.U - r.L for r in cert]
                sem = float(np.mean([abs(next(q.theta_M for q in workload if q.qid == r.qid) -
                                        next(q.theta_gold for q in workload if q.qid == r.qid))
                                     for r in results.values()]))
                undec = sum(1 for r in results.values() if r.status == "undecided")
                err = sum(1 for r in results.values() if r.status == "execution_error")
                agg["calls"].append(pipe.calls)
                agg["cert"].append(len(cert))
                agg["cov"].append(cov_n / cov_d if cov_d else float("nan"))
                agg["mae"].append(mae)
                agg["width"].append(float(np.mean(widths)) if widths else float("nan"))
                agg["sem"].append(sem)
                agg["undec"].append(undec)
                agg["err"].append(err)
                print(f"{alloc_name:<16}{frac:>8.2f}{sd:>4}{pipe.calls:>6}{len(cert):>5}"
                      f"{undec:>5}{(cov_n/cov_d if cov_d else float('nan')):>11.3f}"
                      f"{mae:>10.4f}{float(np.mean(widths)) if widths else float('nan'):>8.3f}"
                      f"{sem:>9.4f}")
            row = {
                "allocator": alloc_name, "budget_fraction": frac,
                "mean_calls": float(np.mean(agg["calls"])),
                "mean_certified": float(np.mean(agg["cert"])),
                "mean_coverage_vs_thetaM": safe_mean(agg["cov"]),
                "mean_mae_vs_thetaM": safe_mean(agg["mae"]),
                "mean_width": safe_mean(agg["width"]),
                "mean_semantic_error": float(np.mean(agg["sem"])),
                "mean_undecided": float(np.mean(agg["undec"])),
                "mean_execution_errors": float(np.mean(agg["err"])),
            }
            table_rows.append(row)

    print("-" * 100)
    print(f"全量物化参考成本 B0 = {N} 次语义调用（θ_M 的构建成本）")
    print(f"语义误差列 = |θ_M − θ_gold|（草案 12.2 的第二项：模型判断 vs 真实标签）")

    # ---- 声明抽查（草案 11）----
    print()
    print("=" * 100)
    print("声明抽查：M 分配器、预算 100%、种子 0 的一次运行（结构化声明 + 受限模板 + 证书）")
    print("=" * 100)
    pipe = COHORTRAG(snap, contract, cfg, seed=0)
    results = pipe.run(workload, WorstVarianceAllocator(), budget_total=int(1.0 * N))
    for q in workload:
        results[q.qid].theta_M = q.theta_M
        results[q.qid].theta_gold = q.theta_gold
    shown = 0
    for q in workload:
        r = results[q.qid]
        if r.status.startswith("certified") and shown < 4:
            claim = render_claim(q, r, snap, contract, cfg, pipe.risk_ledger)
            print(json.dumps(claim, ensure_ascii=False))
            print("  模板输出：", template_text(q, r))
            print(f"  [对照] θ_M={r.theta_M:.4f}  θ_gold={r.theta_gold:.4f}  "
                  f"区间 [{r.L:.4f}, {r.U:.4f}]  调用 {pipe.calls}")
            shown += 1
    cert = certificate(snap, contract, cfg, pipe.risk_ledger, results, pipe.calls)
    print("  证书摘要：", cert["digest"][:32], "（只验证推导/版本/风险分配，不证明现实事实）")

    # ---- 偏置缓存场景（草案 13.4 / 16）----
    print()
    print("=" * 100)
    print("偏置缓存场景：把『关键词匹配』历史（≈g1 正例）作为 E 复用（草案 7.2 / 16 消融）")
    print("=" * 100)
    mask = frame["text"].str.contains("digital-channel", na=False)
    E_ids = [int(i) for i in frame.loc[mask, "id"].to_numpy()[:int(0.2 * N)]]
    E_labels = contract.judge(E_ids, None)
    print(f"  E 规模 {len(E_ids)}（构建成本 {len(E_ids)} 次语义调用，另计，不计入在线预算）")
    cache = {i: lab for i, lab in zip(E_ids, E_labels)}
    pipe = COHORTRAG(snap, contract, cfg, seed=0, cache_labels=cache)
    results = pipe.run(workload, WorstVarianceAllocator(), budget_total=int(0.1 * N))
    q3 = workload[2]   # theta(g2|g1) all
    r3 = results[q3.qid]
    zE = [states.encode_state(tuple(int(v) for v in cache[i])) for i in E_ids]
    AE = sum(int(q3.tables["a"][z]) for z in zE)
    BE = sum(int(q3.tables["b"][z]) for z in zE)
    AS = BS = 0
    for h in q3.scope:
        for g in pipe.ledger.prefix_g[h]:
            z = states.encode_state(tuple(int(v) for v in g))
            AS += int(q3.tables["a"][z])
            BS += int(q3.tables["b"][z])
    naive = (AE + AS) / (BE + BS) if BE + BS > 0 else float("nan")
    # 本方法的正确处理：E 的已知质量精确计入，剩余部分按随机前缀展开估计
    Ah, Bh = AE, BE
    for h in q3.scope:
        s = [0.0, 0.0, 0]
        for g in pipe.ledger.prefix_g[h]:
            z = states.encode_state(tuple(int(v) for v in g))
            s[0] += int(q3.tables["a"][z])
            s[1] += int(q3.tables["b"][z])
            s[2] += 1
        if s[2] > 0:
            Ah += pipe.ledger.N_rem(h) * s[0] / s[2]
            Bh += pipe.ledger.N_rem(h) * s[1] / s[2]
    correct = Ah / Bh if Bh > 0 else float("nan")
    print(f"  查询 {q3.name}：θ_M = {q3.theta_M:.4f}")
    print(f"    天真估计（把 E∪S 当简单随机样本）：θ̂ = {naive:.4f}，误差 {abs(naive - q3.theta_M):.4f}")
    print(f"    本方法（E 精确计入 + 剩余展开）：θ̂ = {correct:.4f}，误差 {abs(correct - q3.theta_M):.4f}")
    if r3.status == "certified":
        print(f"    本方法认证点估计：{r3.point:.4f}（区间 [{r3.L:.4f}, {r3.U:.4f}]）")
    print(f"    （覆盖保持：θ_M 在区间内 = {r3.L - 1e-9 <= q3.theta_M <= r3.U + 1e-9}）")

    # ---- 落盘报告 ----
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
    os.makedirs(outdir, exist_ok=True)
    report = {
        "honest_scope": "synthetic programmatic stress test (draft 12.3); NOT real data results",
        "world": {"N": N, "K": K, "H": H, "seed": 0},
        "contract": contract.name,
        "alpha": cfg.alpha,
        "rows": table_rows,
    }
    path = os.path.join(outdir, "synthetic_demo_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print()
    print(f"报告已保存：{path}")
    print(f"总耗时 {time.time() - t_start:.1f} 秒")

if __name__ == "__main__":
    main()
