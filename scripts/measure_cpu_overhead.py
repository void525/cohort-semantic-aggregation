# -*- coding: utf-8 -*-
"""系统开销 CPU 分项（草案 §15.1「索引构建、参考标签构建、暖缓存成本都要有独立账目」）。

本脚本只测 **不需要 GPU** 的两项，从而在冻结标签带上补齐账目：
  1. 索引构建时间：数据载入（csv 解析 + 稳定哈希子集挑选）与快照冻结
     （分层、不变式校验、manifest 哈希）各自的墙钟时间。
  2. CPU 求解时间：给定预算下一次完整运行的墙钟时间。由于标签带已冻结、
     所有语义调用都是缓存命中（不进 GPU），这段时间即 LP 投影 + 端点搜索
     + 预算分配的纯 CPU 成本。

GPU 侧（参考标签构建 11.3 分钟 / 46.3 条每秒、冷启动 1.17 s、p50 18.67 ms、
p95 20.93 ms、峰值显存 16.99 GiB）已在服务器上实测，见
reports/system_overhead.json，本脚本不重复测量。

用法：
    python scripts/measure_cpu_overhead.py \
        --data <WANDS/dataset> --manifest data_manifests/wands.json \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --workload closure --budget-fracs 0.02,0.05,0.1,0.2,0.5,1.0 --seeds 0,1,2
"""
import argparse
import json
import os
import sys
import time

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
from cohortrag import Snapshot, COHORTRAG, Config, RoundRobinAllocator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", choices=["basic", "closure"], default="closure")
    ap.add_argument("--budget-fracs", default="0.02,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    # ---- 1) 索引构建：数据载入 + 快照冻结，分别计时 ----
    import argparse as _ap
    ns = _ap.Namespace(dataset="wands", data=args.data, manifest=args.manifest,
                       max_intents=120, start="2024-01-01", end="2024-12-31",
                       max_records=50000, id_col="Complaint ID",
                       date_col="Date received",
                       narrative_col="Consumer complaint narrative",
                       product_col="Product", dry_run=False, model=args.model,
                       revision=None, device="cpu", batch_size=16)
    import run_real_data as rrd
    t0 = time.perf_counter()
    frame, gold, meta = rrd._real_frames(ns)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    snap = Snapshot(frame, name="wands-frozen")
    t_snap = time.perf_counter() - t0
    print(f"索引构建｜数据载入 {t_load:.2f}s｜快照冻结 {t_snap:.2f}s"
          f"｜合计 {t_load + t_snap:.2f}s｜N={len(frame)}")

    frame_, snap_, contract, wl, kappa, gold_, meta_ = RA.build_context(args)
    fracs = [float(x) for x in args.budget_fracs.split(",") if x.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # ---- 2) CPU 求解时间（标签带全命中，无 GPU 参与）----
    print(f"\n{'预算':>7}{'调用':>9}{'CPU 求解(s)':>13}{'每千次调用(s)':>14}")
    print("-" * 45)
    rows = []
    for frac in fracs:
        ts, cs = [], []
        for sd in seeds:
            pipe = COHORTRAG(snap_, contract, Config(), seed=sd)
            t0 = time.perf_counter()
            pipe.run(wl, RoundRobinAllocator(),
                     budget_total=max(1, int(len(frame_) * frac)))
            ts.append(time.perf_counter() - t0)
            cs.append(pipe.calls)
        mt, mc = float(np.mean(ts)), float(np.mean(cs))
        print(f"{frac:>7.2f}{mc:>9.0f}{mt:>13.2f}{mt / mc * 1000:>14.2f}")
        rows.append({"budget_fraction": frac, "calls_mean": mc,
                     "cpu_solve_seconds_mean": mt,
                     "cpu_seconds_per_1k_calls": mt / mc * 1000 if mc else None,
                     "n_seeds": len(seeds)})
    print("-" * 45)

    out = {"dataset": "wands", "n_records": int(len(frame_)), "kappa": kappa,
           "workload": args.workload, "n_queries": len(wl),
           "index_build": {"data_load_seconds": t_load,
                           "snapshot_freeze_seconds": t_snap,
                           "total_seconds": t_load + t_snap},
           "cpu_solve": rows}
    os.makedirs(os.path.join(_HERE, "..", "reports"), exist_ok=True)
    op = os.path.join(_HERE, "..", "reports", "cpu_overhead.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"报告：{op}")


if __name__ == "__main__":
    main()
