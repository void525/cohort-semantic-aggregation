# -*- coding: utf-8 -*-
"""B0 摊销交叉点分析（草案 15.2）。

B0（全量物化）的账目：
  一次性成本 = 全量语义标注 31,482 次调用（θ_M 标签带，草案 12.1 计入 GPU 账本）
  之后每个查询的边际成本 ≈ 0（纯 CPU 计数）

采样方法（M/B6）的账目：
  每个 18 查询工作负载花 X(f) 次在线采样调用；同一闭包内的后续查询可复用状态
  （边际成本递减——本分析按"每批都付 X(f)"计算，对采样方法是保守高估）。

交叉点 m*：问多少个查询后，全量物化的总成本开始低于采样方法。
"""
import json
import os
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(_HERE, "..", "reports", "real_wands_report.json")
N_TOTAL = 31482          # B0 一次性全量物化成本（语义调用数）
Q_PER_BATCH = 18         # 每个工作负载的查询数（closure）


def main():
    with open(REPORT, encoding="utf-8") as f:
        rep = json.load(f)
    rows = rep["rows"]

    print("=" * 78)
    print("B0 摊销分析（草案 15.2）｜全量物化一次性成本 =", N_TOTAL, "次调用")
    print(f"每个工作负载 {Q_PER_BATCH} 个查询（closure）")
    print("=" * 78)

    # 1) 每个认证答案的即时成本
    print("\n[1] 每个『认证答案』的即时成本（调用数 / 认证数）")
    print(f"{'分配器':<6}{'预算':>6}{'调用':>8}{'认证':>8}{'每次认证':>10}")
    for r in rows:
        if r["certified_mean"] > 0:
            per = r["calls_mean"] / r["certified_mean"]
            print(f"{r['allocator']:<6}{r['budget_fraction']:>6.2f}"
                  f"{r['calls_mean']:>8.0f}{r['certified_mean']:>8.1f}{per:>10.1f}")
    b0_per = N_TOTAL / Q_PER_BATCH
    print(f"{'B0':<6}{'1.00':>6}{N_TOTAL:>8}{Q_PER_BATCH:>8}{b0_per:>10.1f}"
          f"   <- 一次性，之后边际为 0")

    # 2) 交叉点
    print("\n[2] 交叉点 m*：累计提问数超过 m* 后，B0 更划算")
    print(f"{'分配器':<6}{'预算':>6}{'单批调用':>9}{'m*（个查询）':>14}")
    cross = []
    for r in rows:
        X = r["calls_mean"]
        if X <= 0:
            continue
        # k 批工作负载（每批 18 题）后采样总成本 = k·X（保守：不考虑复用递减）
        # B0 总成本 = N_TOTAL；k* = N_TOTAL / X；m* = 18 · k*
        k = N_TOTAL / X
        m = Q_PER_BATCH * k
        cross.append((r["allocator"], r["budget_fraction"], X, m))
        print(f"{r['allocator']:<6}{r['budget_fraction']:>6.2f}{X:>9.0f}{m:>14.0f}")

    # 3) 固定 m 视角（草案 15.2 要求 m ∈ {1,5,20,100}）
    print("\n[3] 摊销到每个查询的成本：min(采样, B0) 与胜者")
    print(f"{'m（累计查询数）':>12}{'B0 摊销/题':>12}{'最优采样/题':>13}{'胜者':>8}")
    for m in (1, 5, 20, 100, 900):
        b0 = N_TOTAL / max(m, 1)
        # 采样：需要 ceil(m/18) 批；用每批最便宜的可行配置（能认证全部 18 题的最低预算）
        full = [r for r in rows if r["certified_mean"] >= Q_PER_BATCH - 1e-9]
        if full:
            X = min(r["calls_mean"] for r in full)
            k = max(1, -(-m // Q_PER_BATCH))          # 向上取整的批数
            s = k * X / m
            win = "采样" if s < b0 else "B0"
            print(f"{m:>12}{b0:>12.0f}{s:>13.0f}{win:>8}")
        else:
            print(f"{m:>12}{b0:>12.0f}{'—':>13}{'B0':>8}")

    print("\n说明（草案 15.2）：交叉点是研究结果，不是应被隐藏的失败。")
    print("采样侧按『每批都付全价』保守计算；实际同一闭包内后续批次可复用状态，")
    print("真实交叉点只会更靠右（对采样方法更有利）。")

    out = os.path.join(_HERE, "..", "reports", "amortization_b0.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_total_materialization": N_TOTAL, "queries_per_batch": Q_PER_BATCH,
                   "crossovers": [{"allocator": a, "budget_fraction": fr,
                                   "calls_per_batch": X, "crossover_queries": m}
                                  for a, fr, X, m in cross]},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
