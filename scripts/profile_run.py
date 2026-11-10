# -*- coding: utf-8 -*-
"""单次运行剖析：定位耗时瓶颈。"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from cohortrag import Snapshot, COHORTRAG, Config, make_workload, WorstVarianceAllocator
from cohortrag.semantics import NoisyOracleOperator
from cohortrag.worldgen import make_synthetic_world
import cProfile, pstats, io

N, K, H = 2000, 4, 4
frame, hidden = make_synthetic_world(N, K, H, seed=0)
snap = Snapshot(frame, name="prof")
contract = NoisyOracleOperator(hidden, flip_probs=(0.08, 0.08, 0.10, 0.06))
workload = make_workload(K, seed=42)

pr = cProfile.Profile()
pr.enable()
t0 = time.time()
pipe = COHORTRAG(snap, contract, Config(), seed=0)
results = pipe.run(workload, WorstVarianceAllocator(), budget_total=200)
t1 = time.time()
pr.disable()
print(f"单跑（预算 200 调用）：{t1 - t0:.1f} 秒，实际调用 {pipe.calls}，轮数 {pipe.rounds_used}")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(25)
print(s.getvalue())
