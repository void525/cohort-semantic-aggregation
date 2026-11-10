# -*- coding: utf-8 -*-
"""GPU 冒烟测试：真实模型 + 真实文本，只跑几十条。

为什么必须先跑这个（不要直接上 5 万条）：
  1. Qwen3 是混合思考模型。本仓库的 prompt 包走的是「裸 system+user 拼接」，
     不走 chat template，因此 enable_thinking 参数实际上没有生效。
     若模型漏出 <think> 推理，max_new_tokens=16 会截断 → 解析失败 →
     LabelFailure → 整批标签作废。这里会把原始输出打出来让你肉眼确认。
  2. 解析成功率必须 100%，否则 build_label_tape 会不断失败重试。
  3. 拿到真实的 ms/次 与吞吐，才能投影全量预算。

用法（AutoDL，5090）：
    python scripts/smoke_llm.py --dataset cfpb \
        --data /root/autodl-tmp/data/raw/complaints.csv \
        --manifest data_manifests/cfpb.json --n 64 --batch-size 16
"""
import argparse
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
from cohortrag import contracts as _contracts
from cohortrag.llm_operator import LocalLLMOperator
import run_real_data as rrd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cfpb", "wands"], default="cfpb")
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--max-records", type=int, default=50000)
    ap.add_argument("--max-intents", type=int, default=120)
    ap.add_argument("--id-col", default="Complaint ID")
    ap.add_argument("--date-col", default="Date received")
    ap.add_argument("--narrative-col", default="Consumer complaint narrative")
    ap.add_argument("--product-col", default="Product")
    args = ap.parse_args()

    print("=" * 78)
    print(f"GPU 冒烟测试｜dataset={args.dataset}｜model={args.model}｜n={args.n}")
    print("=" * 78)

    frame, gold, meta = rrd._real_frames(args)
    snap = Snapshot(frame, name=f"{args.dataset}-frozen")
    print(f"[1] 载入 {len(frame)} 条，{frame['stratum'].nunique()} 层；"
          f"snapshot manifest {snap.manifest[:16]}…")

    pack = _contracts.cfpb_prompt_pack() if args.dataset == "cfpb" \
        else _contracts.wands_prompt_pack()
    op = LocalLLMOperator(model_id=args.model, revision=args.revision,
                          K=len(pack["primitives"]), prompt_pack=pack,
                          batch_size=args.batch_size, device=args.device,
                          name=f"smoke-{args.dataset}")
    print(f"[2] 加载模型 {args.model} ...（首次约 1-3 分钟，需下载约 16 GB）")
    t_load = time.time()
    op._ensure_loaded()
    print(f"    加载完成 {time.time()-t_load:.1f}s｜dtype={op.dtype}｜"
          f"device={op._model.device}｜max_new_tokens={op.max_new_tokens}")

    ids = frame["id"].tolist()[:args.n]
    texts = snap.texts(ids)
    print(f"[3] 判定 {len(ids)} 条 ...")
    t0 = time.time()
    try:
        lab = op.judge(ids, texts)
    except Exception as e:
        dt = time.time() - t0
        print(f"\n!! judge 失败（{type(e).__name__}）：{e}")
        print(f"   耗时 {dt:.1f}s。下面是单条重试，用于看真实报错与原始输出：")
        _dump_raw(op, ids[:3], texts[:3])
        raise SystemExit(1)
    dt = time.time() - t0

    print(f"\n[4] 结果：{dt:.1f}s｜{dt/len(ids)*1000:.0f} ms/条｜"
          f"{len(ids)/dt:.2f} 条/秒")
    print("    标签矩阵前 5 行：")
    for r in range(min(5, lab.shape[0])):
        print("      ", "".join(str(int(v)) for v in lab[r]))

    # 原始输出抽查：确认没有思考链泄漏、确认解析来源
    print("\n[5] 原始输出抽查（前 5 条，确认没有 <think> 或多余文字）")
    _dump_raw(op, ids[:5], texts[:5])

    # 全量投影
    n_all = len(frame)
    hours = n_all * (dt / len(ids)) / 3600.0
    print(f"\n[6] 全量投影：{n_all} 条 × {dt/len(ids)*1000:.0f} ms "
          f"≈ {hours:.2f} 小时（不含模型加载）")
    print(f"    按 AutoDL 5090 ¥2.78/时 ≈ ¥{hours*2.78:.0f}"
          f"（会员 95 折 ≈ ¥{hours*2.64:.0f}）")
    print("\n判定：若 [5] 里出现 <think>、解释性文字或位数不对，"
          "必须先改 prompt 包/解码配置，不要直接跑全量。")


def _dump_raw(op, ids, texts):
    """绕过严格解析，直接看模型吐了什么。"""
    import torch
    for rid, t in zip(ids, texts):
        p = op._build_prompt(t)
        enc = op._tok(p, return_tensors="pt").to(op._model.device)
        with torch.no_grad():
            g = op._model.generate(**enc, do_sample=False,
                                   max_new_tokens=op.max_new_tokens,
                                   temperature=None, top_p=None)
        new = g[0][enc["input_ids"].shape[1]:]
        raw = op._tok.decode(new, skip_special_tokens=True)
        print(f"      id={rid}  原始输出={raw!r}  解析={op._parse_one(raw)}")


if __name__ == "__main__":
    main()
