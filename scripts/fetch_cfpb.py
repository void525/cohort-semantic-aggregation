# -*- coding: utf-8 -*-
"""下载 CFPB 全量投诉 CSV 并生成仓库要求的许可清单 data_manifests/cfpb.json。

官方来源（2026-10-06 核实）：
  页面   https://www.consumerfinance.gov/data-research/consumer-complaints/
  直链   https://files.consumerfinance.gov/ccdb/complaints.csv.zip
  字段   https://cfpb.github.io/api/ccdb/fields.html

用法（AutoDL）：
    source /etc/network_turbo                 # 学术加速，可选
    python scripts/fetch_cfpb.py --out /root/autodl-tmp/data/raw/complaints.csv
    python scripts/fetch_cfpb.py --out ... --license "public (CFPB official terms)" --by "你的名字"

产物：
  <out>.zip  原始压缩包（保留，哈希以它为准）
  <out>      解压后的 CSV
  data_manifests/cfpb.json   许可清单（字段不全 datasets.load_cfpb 会拒绝运行）

下载失败时：浏览器打开上面的直链手动下载，放到 --out 指定的 .zip 位置再重跑，脚本会跳过下载。
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import zipfile

URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
SCOPE = ("Date received 2024-01-01..2024-12-31, non-empty "
         "Consumer complaint narrative, dedup by Complaint ID, cap 50000")


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
    ap.add_argument("--out", default="/root/autodl-tmp/data/raw/complaints.csv")
    ap.add_argument("--license", default="public (CFPB official terms)")
    ap.add_argument("--by", default="unknown")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    csv_path = args.out
    zip_path = csv_path + ".zip"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    # 情况 A：只有已解压的 csv（手动上传常见）→ 直接对 csv 算哈希
    if not os.path.exists(zip_path) and os.path.exists(csv_path):
        print(f"未找到 zip，但已存在 {csv_path}，直接对 csv 做哈希与清单。")
        man = {
            "dataset": "CFPB Consumer Complaint Database",
            "official_url": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
            "license": args.license,
            "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "sha256": sha256_of(csv_path),
            "scope": SCOPE,
            "frozen_by": args.by,
            "notes": ("CFPB 投诉库不是市场消费者体验的统计样本，禁止外推（草案 4.1）。"
                      "数据库通常每日更新，因此必须冻结下载时间与文件哈希。"),
            "source_file": csv_path,
            "source_bytes": os.path.getsize(csv_path),
            "hash_basis": "csv（上传时未附带原始 zip）",
        }
        out = _write_manifest(man)
        _print_next(csv_path)
        return

    if not args.skip_download and not os.path.exists(zip_path):
        print(f"下载 {URL}\n  -> {zip_path}", flush=True)
        rc = os.system(f'curl -L --retry 5 --retry-delay 3 --max-time 3600 '
                       f'-A "Mozilla/5.0 (research; cohort-rag)" -o "{zip_path}" "{URL}"')
        if rc != 0 or not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            print("\n!! 下载失败。请手动下载后放到上面路径，再重跑本脚本。")
            raise SystemExit(1)
    print(f"zip: {zip_path}  {os.path.getsize(zip_path)/1024**2:.1f} MiB")

    if not os.path.exists(csv_path):
        print("解压 ...", flush=True)
        with zipfile.ZipFile(zip_path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(name) as src, open(csv_path, "wb") as dst:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    dst.write(b)
    print(f"csv: {csv_path}  {os.path.getsize(csv_path)/1024**2:.1f} MiB")

    man = {
        "dataset": "CFPB Consumer Complaint Database",
        "official_url": "https://www.consumerfinance.gov/data-research/consumer-complaints/",
        "license": args.license,
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_of(zip_path),
        "scope": SCOPE,
        "frozen_by": args.by,
        "notes": ("CFPB 投诉库不是市场消费者体验的统计样本，禁止外推（草案 4.1）。"
                  "数据库通常每日更新，因此必须冻结下载时间与文件哈希。"),
        "source_file": zip_path,
        "source_bytes": os.path.getsize(zip_path),
        "hash_basis": "zip（官方原始分发文件）",
    }

    out = _write_manifest(man)
    _print_next(csv_path)


def _write_manifest(man):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "data_manifests", "cfpb.json")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("\n=== data_manifests/cfpb.json ===")
    print(json.dumps(man, ensure_ascii=False, indent=2))
    print(f"\n已写入 {out}")
    return out


def _print_next(csv_path):
    print(f"\n下一步：\n  python scripts/smoke_llm.py --dataset cfpb "
          f"--data {csv_path} --manifest data_manifests/cfpb.json --n 64")


if __name__ == "__main__":
    main()
