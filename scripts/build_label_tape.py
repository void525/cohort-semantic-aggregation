# -*- coding: utf-8 -*-
"""全量参考标签带构建（草案 12.1 的 θ_M 参考）——分块、可续跑、失败隔离。

为什么不能直接用 run_real_data.py 里的那一行 contract.judge(全部 ids)：
  - 5 万条一次调用，中途任何一条解析失败 → LabelFailure → 整个批次回滚，
    10 多小时 GPU 全部作废（缓存只在 inner.judge 成功返回后才写盘）；
  - 没有任何进度输出，无法判断还剩多久；
  - 断了只能从头再来。

本脚本：按 --chunk 分块，每块成功后立即落 sqlite（κ 版本化，草案 7.1）；
某块失败则退化为逐条判定，只记录失败 id，不中断整体；重跑自动跳过已标注记录。

用法（AutoDL，务必用 nohup 或 tmux）：
    nohup python scripts/build_label_tape.py --dataset cfpb \
        --data /root/autodl-tmp/data/raw/complaints.csv \
        --manifest data_manifests/cfpb.json \
        --model Qwen/Qwen3-8B --chunk 500 --batch-size 32 \
        > tape_cfpb.log 2>&1 &

产物：
    cache/labels_cfpb.sqlite    （与 run_real_data.py 同一路径、同一 κ，直接命中）
    reports/label_tape_cfpb.json（统计、失败清单、κ、耗时）
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
from cohortrag import Snapshot
import run_real_data as rrd


def _count_cached(cached, ids):
    n = 0
    for rid in ids:
        if cached._get(rid) is not None:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfpb", "wands", "uci"], default="cfpb")
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=500, help="每块记录数（影响失败重试粒度）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 条（调试用）")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--max-records", type=int, default=50000)
    ap.add_argument("--max-intents", type=int, default=120)
    ap.add_argument("--id-col", default="Complaint ID")
    ap.add_argument("--date-col", default="Date received")
    ap.add_argument("--narrative-col", default="Consumer complaint narrative")
    ap.add_argument("--product-col", default="Product")
    # _load_contract 会读取 args.dry_run；全量构建必须走真实路径
    ap.add_argument("--dry-run", action="store_true",
                    help="内部兼容参数（run_real_data._load_contract 需要），勿手动使用")
    args = ap.parse_args()
    if args.dry_run:
        raise SystemExit("build_label_tape 只用于真实数据构建，不支持 --dry-run")

    print("=" * 78, flush=True)
    print(f"标签带构建｜dataset={args.dataset}｜model={args.model}｜"
          f"chunk={args.chunk}｜batch={args.batch_size}", flush=True)
    print("=" * 78, flush=True)

    frame, gold, meta = rrd._real_frames(args)
    snap = Snapshot(frame, name=f"{args.dataset}-frozen")
    contract, kappa = rrd._load_contract(args, snap, gold)
    print(f"[1] 记录 {len(frame)}｜层 {frame['stratum'].nunique()}｜"
          f"算子 {contract.name}", flush=True)
    print(f"[2] κ = {kappa}  （缓存键，版本隔离，草案 7.1）", flush=True)
    print(f"    缓存文件 {getattr(contract, 'cache_path', 'n/a')}", flush=True)

    ids = frame["id"].tolist()
    if args.limit > 0:
        ids = ids[:args.limit]
    texts = snap.texts(ids)

    done0 = _count_cached(contract, ids)
    print(f"[3] 已缓存 {done0}/{len(ids)}，剩余 {len(ids)-done0}", flush=True)

    failures = []
    t_start = time.time()
    n_done = done0
    total = len(ids)

    for s in range(0, total, args.chunk):
        chunk_ids = ids[s:s + args.chunk]
        chunk_txt = texts[s:s + args.chunk]
        t0 = time.time()
        try:
            contract.judge(chunk_ids, chunk_txt)
            n_done += len(chunk_ids)
        except Exception as e:
            # 整块失败 → 逐条隔离，保住能救的记录
            print(f"    !! 块 [{s}:{s+len(chunk_ids)}] 失败（{type(e).__name__}），"
                  f"退化为逐条判定", flush=True)
            for rid, t in zip(chunk_ids, chunk_txt):
                try:
                    contract.judge([rid], [t])
                    n_done += 1
                except Exception as e2:
                    failures.append({"id": int(rid), "error": f"{type(e2).__name__}: {e2}"})
        dt = time.time() - t0
        el = time.time() - t_start
        rate = (n_done - done0) / el if el > 0 else 0
        eta = (total - n_done) / rate / 3600 if rate > 0 else float("nan")
        print(f"    [{n_done}/{total}] 块耗时 {dt:.1f}s｜平均 {rate:.2f} 条/秒｜"
              f"已用 {el/60:.1f} min｜ETA {eta:.2f} h｜失败 {len(failures)}", flush=True)

    elapsed = time.time() - t_start
    ok = total - len(failures)
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "kappa": kappa,
        "cache_path": getattr(contract, "cache_path", None),
        "snapshot_manifest": snap.manifest,
        "n_records": total,
        "n_labelled": ok,
        "n_failed": len(failures),
        "failure_rate": len(failures) / total if total else 0.0,
        "elapsed_seconds": round(elapsed, 1),
        "records_per_second": round(ok / elapsed, 3) if elapsed > 0 else None,
        "cache_hits": getattr(contract, "hits", None),
        "cache_misses": getattr(contract, "misses", None),
        "failures": failures[:200],
        "note": "θ_M 参考标签带。构建成本计入 GPU 账本，不作为主方法免费的在线预处理（草案 12.1）。",
    }
    out = os.path.join(_HERE, "..", "reports", f"label_tape_{args.dataset}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"完成：{ok}/{total} 已标注，失败 {len(failures)} "
          f"（{summary['failure_rate']*100:.2f}%）", flush=True)
    print(f"耗时 {elapsed/3600:.2f} h｜{summary['records_per_second']} 条/秒", flush=True)
    print(f"报告：{out}", flush=True)
    if failures:
        print(f"!! 有 {len(failures)} 条失败，已列入报告。"
              "不要把它们当成负例（草案 10.1）；失败率过高请检查 prompt 包与解码配置。",
              flush=True)


if __name__ == "__main__":
    main()
