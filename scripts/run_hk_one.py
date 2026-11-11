# -*- coding: utf-8 -*-
"""One (H, K, seed) structural-scaling cell. One process, one core."""
import os
import sys
import time
import tracemalloc

os.environ.setdefault("WANDS_DIR", "/root/autodl-tmp/WANDS")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import numpy as np

import run_icde2027_suite as S
import run_review3_suite as R3
from cohortrag import COHORTRAG, Config, RoundRobinAllocator, Snapshot
from cohortrag.llm_operator import LabelTableOperator
from cohortrag.workload import Query

H = int(sys.argv[1])
K = int(sys.argv[2])
sd = int(sys.argv[3])
out = sys.argv[4]

env = S.load_wands_env()
l8 = env["labels_M"]
l4 = R3.load_4b_labels()
ids = [int(i) for i in env["ids_all"]]
pc = S.product_class_map()
f = env["frame"].copy()
f["cls"] = f["product_id"].astype(int).map(pc).fillna("missing")
top_cls = f["cls"].value_counts().index[:7].tolist()
rb = f["rating_bucket"].map({"<4": 0, ">=4": 1, "missing": 2}).astype(int)
if H == 6:
    f["stratum"] = rb * 2 + (f["intent_id"].astype(int) % 2)
elif H == 12:
    f["stratum"] = rb * 4 + (f["intent_id"].astype(int) % 4)
elif H == 24:
    cg = f["cls"].apply(lambda c: top_cls.index(c) if c in top_cls else 7)
    f["stratum"] = rb * 8 + cg
else:
    raise SystemExit(f"bad H {H}")
sub = f[["id", "stratum", "text"]].reset_index(drop=True)
snap = Snapshot(sub, name=f"hk-{H}-{K}-{sd}")
all_h = tuple(sorted(sub["stratum"].unique().tolist()))
if K == 2:
    labs = {i: np.array([int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8) for i in ids}
    wl = [
        Query(1, "ex|rel", all_h, (0,), (1,), eps=0.05),
        Query(2, "ex|rel_s0", (0,), (0,), (1,), eps=0.10),
    ]
elif K == 4:
    labs = {
        i: np.array(
            [int(l4[i][0]), int(l4[i][0] and l4[i][1]), int(l8[i][0]), int(l8[i][0] and l8[i][1])],
            dtype=np.int8,
        )
        for i in ids
    }
    wl = [
        Query(1, "prec_rel", all_h, (0,), (2,), eps=0.05),
        Query(2, "prec_ex", all_h, (0, 1), (2, 3), eps=0.05),
        Query(3, "rec_rel", all_h, (2,), (0,), eps=0.05),
        Query(4, "overclaim", all_h, (2,), (0, 1), eps=0.05),
        Query(5, "ex8|rel8", all_h, (2,), (3,), eps=0.05),
    ]
else:
    raise SystemExit(f"bad K {K}")
for q in wl:
    q.tables = q.build_tables(K)
    q.a_hz = q.numerator_tables(K, q.scope)
contract = LabelTableOperator(labs, name=f"op-K{K}", K=K)
tracemalloc.start()
t0 = time.process_time()
pipe = COHORTRAG(snap, contract, Config(alpha=S.ALPHA), seed=sd)
res = pipe.run(wl, RoundRobinAllocator(), budget_total=3200)
cpu = time.process_time() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
width = res[1].U - res[1].L if res[1].L is not None else 1.0
line = f"{H} {K} {sd} {cpu:.4f} {width:.6f} {peak/1024/1024:.2f}\n"
with open(out, "w") as f:
    f.write(line)
print("CELL", line.strip(), flush=True)
