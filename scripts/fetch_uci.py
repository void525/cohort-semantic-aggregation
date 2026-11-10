# -*- coding: utf-8 -*-
"""获取 UCI Sentiment Labelled Sentences 并生成许可清单（草案 4.3）。

官方来源：https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
直链：https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip
许可：CC BY 4.0（官方页面标注）

用法（AutoDL）：
    python scripts/fetch_uci.py --dir /root/autodl-tmp/uci --by "你的名字"

若服务器下载失败：把本地 `data/uci/amazon_cells_labelled.txt`、`yelp_labelled.txt`
两个文件直接上传到 --dir 目录，再重跑本脚本（它会自动跳过下载）。
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import zipfile

URL = ("https://archive.ics.uci.edu/static/public/331/"
       "sentiment+labelled+sentences.zip")
FILES = ["amazon_cells_labelled.txt", "yelp_labelled.txt"]
SCOPE = ("Amazon + Yelp 现成情感标注句子，合计 2000 条；"
         "群体仅为明确标注为 synthetic 的哈希分组，不是实际市场细分（草案 4.3）")


def sha256_of(path, chunk=1 << 20):
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
    ap.add_argument("--dir", default="/root/autodl-tmp/uci")
    ap.add_argument("--by", default="unknown")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    have = all(os.path.exists(os.path.join(args.dir, f)) for f in FILES)
    if not have:
        zp = os.path.join(args.dir, "uci.zip")
        if not os.path.exists(zp):
            print(f"下载 {URL}", flush=True)
            rc = subprocess.call(["curl", "-sL", "--retry", "3", "--max-time", "300",
                                  "-o", zp, URL])
            if rc != 0 or not os.path.exists(zp):
                print("!! 下载失败。请手动下载并把两个 txt 放到", args.dir)
                raise SystemExit(1)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                base = os.path.basename(n)
                if base in FILES:
                    with z.open(n) as src, open(os.path.join(args.dir, base), "wb") as dst:
                        dst.write(src.read())
                    print(f"  解压 {base}")
    per_file = {f: {"sha256": sha256_of(os.path.join(args.dir, f)),
                    "bytes": os.path.getsize(os.path.join(args.dir, f))}
                for f in FILES}
    for f, v in per_file.items():
        print(f"  {f}: {v['bytes']/1024:.0f} KB  sha256={v['sha256'][:16]}…")

    n_lines = 0
    for f in FILES:
        with open(os.path.join(args.dir, f), encoding="utf-8", errors="replace") as fh:
            n_lines += sum(1 for ln in fh if "\t" in ln)

    man = {
        "dataset": "UCI Sentiment Labelled Sentences",
        "official_url": "https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences",
        "license": "CC BY 4.0 (as stated on the official UCI page)",
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256("".join(v["sha256"] for v in per_file.values())
                                 .encode()).hexdigest(),
        "scope": SCOPE,
        "frozen_by": args.by,
        "files": per_file,
        "n_parsed_lines": n_lines,
        "notes": ("用于情感原语与【独立标签误差】的补充检查；"
                  "不得据此推断市场细分或因果（草案 4.3）。"),
    }
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "data_manifests", "uci.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("\n=== data_manifests/uci.json ===")
    print(json.dumps({k: v for k, v in man.items() if k != "files"},
                     ensure_ascii=False, indent=2))
    print(f"\n已写入 {out}")
    print(f"\n下一步（需 GPU 模式）：\n  python scripts/run_real_data.py --dataset uci "
          f"--data {args.dir} --manifest data_manifests/uci.json "
          f"--model /root/autodl-tmp/models/Qwen3-8B --budget-fracs 0.02,0.05,0.1,0.2,0.5,1.0")


if __name__ == "__main__":
    main()
