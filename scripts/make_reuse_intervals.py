# -*- coding: utf-8 -*-
"""Fig. 2（右）的真实数值：同一份 BM25 检索历史（占总体 15%）、同样 2% 的新调用、三种用法下，
头条比例 θ(exact | relevant) 的区间；由 fig_family.py 读取并作图。"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
import run_icde2027_suite as S
from cohortrag import RoundRobinAllocator

env = S.load_wands_env()
seed, budget = 0, 640
hist = S.bm25_history(env, 0.15, seed)
truth = env["theta_M"][1]
out = {"truth": truth, "hist_stats": S.hist_stats(env, hist), "budget": budget, "seed": seed}
for name, kw in (("discard", {}), ("free", {"tcache": hist}), ("cohort", {"cache": hist}),
               ("pooled", {"cache": hist, "e_as_random": True})):
    res, calls, wl, pipe = S.run_cohort(env, RoundRobinAllocator(), budget, seed, **kw)
    r = res[1]
    out[name] = [float(r.L), float(r.U)]
    out[name + "_status"] = r.status
    print(name, calls, r.status, r.L, r.U)
out["pooled_wrong"] = not (out["pooled"][0] <= truth <= out["pooled"][1])
json.dump(out, open(os.path.join(S.OUT_DIR, "reuse_intervals.json"), "w"), indent=1)
print(out)
