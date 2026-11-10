# -*- coding: utf-8 -*-
"""WANDS 逐对分类指标（草案 15.1 明确要求）：macro-F1、混淆矩阵、语义误差分解。

草案原文：「WANDS 另报告逐对分类 macro-F1 和混淆矩阵，UCI 报告情感分类误差。
若模型本身没有改变，不能把不变的分类指标宣传为新方法的提升。」

本脚本不需要 GPU：全部从冻结标签带（θ_M 参考标签带）+ WANDS 现成人工标签计算。
"""
import argparse
import json
import os
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import numpy as np
import run_real_data as rrd
from cohortrag import Snapshot

CLASSES = ["Exact", "Partial", "Irrelevant"]


def to_class(p1, p2):
    """原语 (p1=相关, p2=完全匹配) -> 三分类；返回 None 表示组合不合法。"""
    if p1 == 1 and p2 == 1:
        return "Exact"
    if p1 == 1 and p2 == 0:
        return "Partial"
    if p1 == 0 and p2 == 0:
        return "Irrelevant"
    return None                       # (0,1)：相关但完全匹配，逻辑矛盾


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    ns = argparse.Namespace(dataset="wands", data=args.data, manifest=args.manifest,
                            max_intents=120, start="2024-01-01", end="2024-12-31",
                            max_records=50000, id_col="Complaint ID",
                            date_col="Date received",
                            narrative_col="Consumer complaint narrative",
                            product_col="Product", dry_run=False, model=args.model,
                            revision=None, device="cpu", batch_size=16)
    frame, gold, meta = rrd._real_frames(ns)
    snap = Snapshot(frame, name="wands-frozen")
    contract, kappa = rrd._load_contract(ns, snap, gold)

    ids = frame["id"].to_numpy()
    lab = contract.judge(list(ids), snap.texts(list(ids)))
    model_lab = {int(i): lab[r] for r, i in enumerate(ids)}

    y_true, y_pred, invalid = [], [], 0
    for i in ids:
        g = gold["labels"].get(int(i))
        m = model_lab.get(int(i))
        if g is None or m is None:
            continue
        # gold: p1=相关(Exact/Partial), p2=Exact
        tg = "Exact" if g[1] == 1 else ("Partial" if g[0] == 1 else "Irrelevant")
        pm = to_class(int(m[0]), int(m[1]))
        if pm is None:
            invalid += 1
            continue
        y_true.append(tg)
        y_pred.append(pm)

    n = len(y_true)
    print(f"κ={kappa[:16]}  样本 {n} 条（标签带全量）  不合法组合(p1=0,p2=1) {invalid} 条")

    # 混淆矩阵（行=真值，列=预测）
    idx = {c: k for k, c in enumerate(CLASSES)}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    print("\n混淆矩阵（行=人工真值，列=模型预测）")
    print("            " + "".join(f"{c:>12}" for c in CLASSES))
    for c in CLASSES:
        print(f"{c:>11}  " + "".join(f"{cm[idx[c], k]:>12d}" for k in range(3)))

    # macro-F1 / 逐类 PRF
    f1s, rows = [], []
    for c in CLASSES:
        k = idx[c]
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        rows.append({"class": c, "support": int(cm[k, :].sum()), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)})
        print(f"  {c:<11} support={cm[k,:].sum():6d}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    macro_f1 = float(np.mean(f1s))
    print(f"\n准确率 {acc:.4f}   macro-F1 {macro_f1:.4f}")
    print("注意（草案 15.1）：模型本身未改变时，不得把这些分类指标宣传为新方法的提升。")

    out = os.path.join(_HERE, "..", "reports", "pair_classification_wands.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"kappa": kappa, "n_pairs": n, "invalid_combinations": invalid,
                   "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
                   "per_class": rows,
                   "confusion_matrix": {"labels": CLASSES, "rows_are": "gold",
                                        "cols_are": "pred", "matrix": cm.tolist()},
                   "note": "θ_M 参考标签带 vs WANDS 人工标签；用于测量语义误差，不是方法提升"},
                  f, ensure_ascii=False, indent=2)
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
