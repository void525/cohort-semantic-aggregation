# -*- coding: utf-8 -*-
"""GPU 系统开销实测：峰值显存（草案 15.1 / 17.2）+ 冷/热启动延迟分位（§15.1）。

必须在 **GPU 模式** 下运行。约 2 分钟。

    python scripts/measure_vram.py --data /root/autodl-tmp/WANDS/dataset \
        --manifest data_manifests/wands.json \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --batch-size 32 --n 256

产出：reports/system_overhead.json
"""
import argparse
import json
import os
import statistics
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

import run_real_data as rrd
from cohortrag import Snapshot
from cohortrag.llm_operator import LocalLLMOperator
from cohortrag import contracts as C


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return float(xs[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n", type=int, default=256, help="用于延迟测量的记录数")
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "本脚本必须在 GPU 模式下运行"

    ns = argparse.Namespace(dataset="wands", data=args.data, manifest=args.manifest,
                            max_intents=120, start="2024-01-01", end="2024-12-31",
                            max_records=50000, id_col="Complaint ID",
                            date_col="Date received",
                            narrative_col="Consumer complaint narrative",
                            product_col="Product", dry_run=False,
                            model=args.model, revision=None, device="cuda",
                            batch_size=args.batch_size)
    frame, gold, meta = rrd._real_frames(ns)
    snap = Snapshot(frame, name="wands-frozen")

    op = LocalLLMOperator(model_id=args.model, K=2,
                          prompt_pack=C.wands_prompt_pack(),
                          batch_size=args.batch_size, device="cuda",
                          name="vram-probe")
    print(f"设备 {torch.cuda.get_device_name(0)}｜"
          f"总显存 {torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    op._ensure_loaded()
    torch.cuda.synchronize()
    t_load = time.time() - t0
    peak_load = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"模型加载 {t_load:.1f}s｜加载后已分配 {torch.cuda.memory_allocated()/2**30:.2f} GiB｜"
          f"峰值 {peak_load:.2f} GiB")

    ids = frame["id"].tolist()[:args.n]
    texts = snap.texts(ids)

    # 冷启动：第一批（含 CUDA kernel 初始化等固定开销）
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    op.judge(ids[:args.batch_size], texts[:args.batch_size])
    torch.cuda.synchronize()
    cold_ms = (time.time() - t0) * 1000

    # 热启动：逐批计时
    per_batch, peak_judge = [], 0.0
    for s in range(0, len(ids), args.batch_size):
        b_ids = ids[s:s + args.batch_size]
        b_txt = texts[s:s + args.batch_size]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        op.judge(b_ids, b_txt)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000
        per_batch.append(dt / max(1, len(b_ids)))     # ms / 条
        peak_judge = max(peak_judge, torch.cuda.max_memory_allocated() / 2 ** 30)

    warm = per_batch[1:] if len(per_batch) > 1 else per_batch
    out = {
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2),
        "model": args.model,
        "batch_size": args.batch_size,
        "n_records": len(ids),
        "load_seconds": round(t_load, 1),
        "peak_vram_after_load_gib": round(peak_load, 2),
        "peak_vram_during_inference_gib": round(peak_judge, 2),
        "cold_first_batch_ms": round(cold_ms, 1),
        "per_record_ms_p50": round(_q(warm, 0.50), 2),
        "per_record_ms_p95": round(_q(warm, 0.95), 2),
        "per_record_ms_mean": round(statistics.fmean(warm), 2) if warm else None,
        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
        "note": "峰值显存与延迟实测（草案 15.1）。不得用估算值代替。",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = os.path.join(_HERE, "..", "reports", "system_overhead.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
