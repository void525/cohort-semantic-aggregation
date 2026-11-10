# -*- coding: utf-8 -*-
"""获取 Wayfair WANDS 数据集并生成仓库要求的许可清单 data_manifests/wands.json。

官方来源（2026-10-07 核实，仓库在线、MIT 许可）：
  https://github.com/wayfair/WANDS
  dataset/product.csv  42,994 个候选商品
  dataset/query.csv     480 个搜索意图
  dataset/label.csv 233,448 条人工相关性标注（Exact / Partial / Irrelevant）

背景：CFPB 于 2026-09（Release 24）移除了投诉叙事字段，本项目主数据改为 WANDS。

用法（AutoDL）：
    source /etc/network_turbo
    python scripts/fetch_wands.py --dir /root/autodl-tmp/WANDS --by "你的名字"

产物：
  /root/autodl-tmp/WANDS/dataset/{product,query,label}.csv
  data_manifests/wands.json
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

REPO = "https://github.com/wayfair/WANDS.git"
FILES = ["product.csv", "query.csv", "label.csv"]
SCOPE = ("480 search intents, 233,448 labelled query-product pairs, 42,994 products; "
         "subset selected by stable salted hash (seed=0), cap 120 intents x 512 pairs")


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/autodl-tmp/WANDS")
    ap.add_argument("--by", default="unknown")
    ap.add_argument("--skip-clone", action="store_true")
    args = ap.parse_args()

    ds_dir = os.path.join(args.dir, "dataset")
    if not args.skip_clone:
        if not os.path.exists(os.path.join(args.dir, ".git")):
            print(f"克隆 {REPO} -> {args.dir}", flush=True)
            rc = subprocess.call(["git", "clone", "--depth", "1", REPO, args.dir])
            if rc != 0:
                print("\n!! git clone 失败。备选方案：")
                print(f"   curl -L -o wands.zip https://codeload.github.com/wayfair/WANDS/zip/refs/heads/main")
                print(f"   unzip wands.zip && 把解压出的 WANDS-main 目录改名为 {args.dir}")
                raise SystemExit(1)
        else:
            print(f"{args.dir} 已存在，跳过克隆")

    missing = [f for f in FILES if not os.path.exists(os.path.join(ds_dir, f))]
    if missing:
        print(f"!! {ds_dir} 缺少 {missing}。请检查目录结构。")
        raise SystemExit(1)

    per_file = {}
    for f in FILES:
        p = os.path.join(ds_dir, f)
        per_file[f] = {"sha256": sha256_of(p), "bytes": os.path.getsize(p)}
        print(f"  {f}: {per_file[f]['bytes']/1024**2:.1f} MiB  sha256={per_file[f]['sha256'][:16]}…")

    combined = hashlib.sha256(
        "".join(per_file[f]["sha256"] for f in FILES).encode()).hexdigest()

    man = {
        "dataset": "Wayfair WANDS (Wayfair ANnotation Dataset)",
        "official_url": "https://github.com/wayfair/WANDS",
        "license": "MIT (see LICENSE in the official repository)",
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sha256": combined,
        "scope": SCOPE,
        "frozen_by": args.by,
        "notes": ("WANDS 是 ECIR 2022 论文配套的人工相关性标注数据集。"
                  "研究目标只涉及【已标注的有限候选池】；未标注的意图-商品对既不是负例，"
                  "也不属于该任务总体（草案 4.2）。缓存键必须包含搜索意图 ID。"),
        "files": per_file,
        "source_commit": _git_commit(args.dir),
        "change_note": ("原定主数据 CFPB 消费者投诉数据库于 2026-09（官方 Release 24）"
                        "移除了投诉叙事字段（Consumer complaint narrative），"
                        "本项目四个语义原语依赖该字段，故主数据改为 WANDS。"),
    }

    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "data_manifests", "wands.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    print("\n=== data_manifests/wands.json ===")
    print(json.dumps({k: v for k, v in man.items() if k != "files"},
                     ensure_ascii=False, indent=2))
    print(f"\n已写入 {out}")
    print(f"\n下一步：\n  python scripts/smoke_llm.py --dataset wands "
          f"--data {ds_dir} --manifest data_manifests/wands.json --n 64")


def _git_commit(repo_dir):
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
