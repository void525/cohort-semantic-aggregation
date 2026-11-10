# -*- coding: utf-8 -*-
"""缓存预检（不上 GPU、不加载模型）——长跑前必做。

目的：在花时间跑实验之前，先确认
  1) 快照 manifest == 6ca573a7…（数据没换版本）
  2) κ == dfdcec8ddaaffa6b…（模型串/prompt 包/解码配置没变）
  3) 该 κ 下的标签带是否 100% 覆盖全部 N 条记录
只要第 3 条成立，后续 run_real_data.py 全程不会加载模型，
纯 CPU 无 torch/transformers 也能跑。

用法：
    python scripts/preflight_cache.py --dataset wands \
        --data /root/autodl-tmp/WANDS/dataset \
        --manifest data_manifests/wands.json \
        --model /root/autodl-tmp/models/Qwen3-8B
"""
import sys, os, json, sqlite3, argparse

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np
from cohortrag import Snapshot
from cohortrag.datasets import load_wands, load_uci
from cohortrag import contracts as _contracts
from cohortrag.llm_operator import LocalLLMOperator, CachedSemanticOperator

# 期望值（来自 01_实验结果/label_tape_*.json，写死便于一眼核对）
# 每个数据集各自的 κ 与快照清单；--dataset 决定用哪一组。
EXPECT = {
    "wands": {
        "kappa": "dfdcec8ddaaffa6b181d56978f0a48dacc11a36df3fef26019c7d1cf3217fe6f",
        "manifest": "6ca573a7766b98a3776a59beb030a7c9169b5fb103da5fc3937c0c74c8e5381d",
    },
    "uci": {
        "kappa": "c559f7765f31a6c9912878873370392922a1e6d606a599a2a2b03940952d7bac",
        "manifest": "a8e22e2ab6b64b1bbb2353113d61cb24ba33406d0fc8a2ce0e65829493d77598",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["wands", "uci"], default="wands")
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default=None, help="默认 <repo>/cache/labels_<dataset>.sqlite")
    args = ap.parse_args()

    cache_path = args.cache or os.path.join(_HERE, "..", "cache",
                                            f"labels_{args.dataset}.sqlite")

    print("=" * 78)
    print("COHORT-RAG 缓存预检（不加载模型）")
    print("=" * 78)

    # 1) 数据 + 快照
    if args.dataset == "wands":
        frame, gold_labels, info = load_wands(args.data, args.manifest,
                                              max_intents=120, seed=0)
    else:
        frame, gold_labels, info = load_uci(args.data, args.manifest)
    print(f"[1] 载入 {args.dataset}：{len(frame)} 条，{frame['stratum'].nunique()} 层")
    snap = Snapshot(frame, name=f"{args.dataset}-frozen")
    exp = EXPECT[args.dataset]
    ok_manifest = snap.manifest == exp["manifest"]
    print(f"    快照 manifest {snap.manifest}")
    print(f"    └─ 期望 {exp['manifest'][:24]}… → {'✅ 一致' if ok_manifest else '❌ 不一致'}")

    # 2) κ（只构造算子，不加载模型）
    pack = _contracts.wands_prompt_pack() if args.dataset == "wands" \
        else _contracts.uci_prompt_pack()
    inner = LocalLLMOperator(model_id=args.model, revision=None,
                             K=len(pack["primitives"]), prompt_pack=pack,
                             batch_size=16, device="cuda",
                             name=f"qwen3-{args.dataset}")
    cached = CachedSemanticOperator(inner, snap.manifest, cache_path=cache_path)
    k = cached.kappa
    ok_kappa = k == exp["kappa"]
    print(f"[2] κ = {k}")
    print(f"    └─ 期望 {exp['kappa'][:24]}… → {'✅ 一致' if ok_kappa else '❌ 不一致'}")

    # 3) 该 κ 下的标签带覆盖情况
    db = sqlite3.connect(cache_path)
    n_cached = db.execute("SELECT COUNT(*) FROM labels WHERE kappa=?", (k,)).fetchone()[0]
    n_all = len(frame)
    db.close()
    cover = n_cached / n_all if n_all else 0.0
    print(f"[3] 标签带：{cache_path}")
    print(f"    该 κ 下已标注 {n_cached} / {n_all} 条 → 覆盖率 {cover:.4%}")
    other = None
    db = sqlite3.connect(cache_path)
    rows = db.execute("SELECT kappa, COUNT(*) FROM labels GROUP BY kappa").fetchall()
    db.close()
    if rows:
        print("    库内现有 κ 分组：")
        for kk, c in rows:
            tag = " ← 当前" if kk == k else ""
            print(f"      {kk[:16]}… : {c} 条{tag}")
        other = [kk for kk, _ in rows if kk != k]

    # 4) 实跑一小批，确认 hits/misses（真正触发一次 judge，但全部命中就不碰模型）
    ids = list(frame["id"].to_numpy()[:64])
    try:
        _ = cached.judge(ids, snap.texts(ids))
        print(f"[4] 64 条试判：命中 {cached.hits}，未命中 {cached.misses}"
              f"（未命中会在 CPU 上尝试加载 8B 模型，务必为 0）")
        ok_hit = cached.misses == 0
    except Exception as e:
        print(f"[4] 试判失败：{type(e).__name__}: {e}")
        ok_hit = False

    print("=" * 78)
    if ok_manifest and ok_kappa and cover >= 1.0 and ok_hit:
        print("✅ 预检通过：可以开始纯 CPU 重放长跑。")
        return 0
    print("❌ 预检未通过，先修好再跑长任务。常见原因：")
    if not ok_manifest:
        print("   · 数据文件与冻结清单不一致（克隆时是否用了 git -c core.autocrlf=false？）")
    if not ok_kappa:
        print("   · --model 字符串与当初建标签带时不一致（必须用 /root/autodl-tmp/models/Qwen3-8B）")
        if other:
            print("   · 库里有别的 κ，说明当初建带用的是另一个模型串：")
            for kk in other:
                print(f"       {kk[:16]}…")
    if cover < 1.0:
        print("   · 标签带不完整 → 缺的那部分会去调模型，CPU 上会爆。需要回 GPU 补齐。")
    if not ok_hit:
        print("   · 试判出现未命中。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
