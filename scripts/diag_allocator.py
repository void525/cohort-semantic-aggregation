# -*- coding: utf-8 -*-
"""诊断：M（WorstVariance）与 B6（RoundRobin）的预算到底花在哪一层。

本地运行（全缓存命中，不需要 GPU）：
    MSYS_NO_PATHCONV=1 <python> scripts/diag_allocator.py
"""
import argparse
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
from cohortrag import (Snapshot, COHORTRAG, Config, RoundRobinAllocator,
                       WorstVarianceAllocator)
from cohortrag.workload import make_wands_workload
import run_real_data as rrd

DATA = r"C:/Users/钦/AppData/Local/Temp/wands/WANDS-main/dataset"
MANIFEST = r"D:/钦/wands.json"
MODEL = "/root/autodl-tmp/models/Qwen3-8B"


def main():
    ns = argparse.Namespace(dataset="wands", data=DATA, manifest=MANIFEST,
                            max_intents=120, start="2024-01-01", end="2024-12-31",
                            max_records=50000, id_col="Complaint ID",
                            date_col="Date received",
                            narrative_col="Consumer complaint narrative",
                            product_col="Product", dry_run=False, model=MODEL,
                            revision=None, device="cuda", batch_size=16)
    frame, gold, meta = rrd._real_frames(ns)
    snap = Snapshot(frame, name="wands-frozen")
    contract, kappa = rrd._load_contract(ns, snap, gold)
    print(f"κ={kappa[:16]}  N={len(frame)}  层={frame['stratum'].nunique()}")

    fb = frame["rating_bucket"].astype(str)
    by_rating = {b: sorted(frame.loc[fb == b, "stratum"].unique().tolist())
                 for b in fb.unique()}
    workload = make_wands_workload(by_rating, {"all": meta["all_strata"]}, K=2)
    for q in workload:
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2,
                                         tables={"a": q.tables["a2"]})
    print(f"查询数 {len(workload)}：", [q.name[:34] for q in workload])

    N = len(frame)
    for frac in (0.02, 0.20, 1.00):
        print(f"\n{'='*74}\n预算档 {frac:.2f}（上限 {int(N*frac)} 次调用）")
        for name, alloc in (("B6", RoundRobinAllocator()),
                            ("M ", WorstVarianceAllocator())):
            pipe = COHORTRAG(snap, contract, Config(), seed=0)
            res = pipe.run(workload, alloc, budget_total=int(N * frac))
            n_cert = sum(1 for r in res.values() if r.status.startswith("certified"))
            rows = []
            for h in sorted(pipe.ledger.strata):
                Nr = pipe.ledger.N_rem(h)
                n = pipe.ledger.n_prefix(h)
                rows.append((h, Nr, n, (n / Nr if Nr else 0)))
            print(f"  {name}: 调用={pipe.calls:6d}  认证={n_cert}/{len(workload)}  "
                  f"未决={sum(1 for r in res.values() if r.status=='undecided')}")
            print("     层  N_rem   n_prefix  覆盖率")
            for h, Nr, n, f in rows:
                bar = "#" * int(f * 30)
                print(f"     {h}  {Nr:6d}  {n:8d}  {f:6.3f} {bar}")
            for q in workload:
                r = res[q.qid]
                if r.L is not None:
                    print(f"     q{q.qid} {q.name[:26]:26s} [{r.L:.4f},{r.U:.4f}] "
                          f"宽={r.U - r.L:.4f} {r.status}")
                else:
                    print(f"     q{q.qid} {q.name[:26]:26s} {r.status}")


if __name__ == "__main__":
    main()
