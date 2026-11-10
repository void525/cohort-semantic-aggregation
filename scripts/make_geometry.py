# -*- coding: utf-8 -*-
"""区域几何示意图的数据：真实运行（seed 0，2% 预算）的联合区域在 (B, A) 平面上的投影。"""
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
import run_icde2027_suite as S
from cohortrag import RoundRobinAllocator
from cohortrag.projection import _stratum_lp, ratio_bounds, min_max_B, _as_a_hz

env = S.load_wands_env()
res, calls, wl, pipe = S.run_cohort(env, RoundRobinAllocator(), 640, 0)
poly = pipe.last_poly
q = next(q for q in wl if q.name == "theta(rating>=4 | relevant)")
J = 4
scope = q.scope
a_hz = _as_a_hz(q.a_hz, scope, J)
b = q.tables["b"].astype(float)

def extreme_point(u, v):
    """max u*A + v*B over the region；返回 (A, B) 极值点。"""
    A = B = 0.0
    for h in scope:
        c = -(u * a_hz[h] + v * b)
        _, x = _stratum_lp(poly, h, c, J, need_solution=True)
        A += float(a_hz[h] @ x)
        B += float(b @ x)
    return A, B

pts = []
for ang in np.linspace(0, 2 * math.pi, 241)[:-1]:
    pts.append(extreme_point(math.cos(ang), math.sin(ang)))
pts = np.array(pts)
# 去重（保持方向顺序）
keep = [0]
for i in range(1, len(pts)):
    if np.hypot(*(pts[i] - pts[keep[-1]])) > 1e-6:
        keep.append(i)
pts = pts[keep]
A_lo, A_hi = pts[:, 0].min(), pts[:, 0].max()
B_lo, B_hi = pts[:, 1].min(), pts[:, 1].max()
L, U = ratio_bounds(poly, q.a_hz, q.tables["b"], scope, J)
ids, lab, so = env["ids_all"], env["labels_M"], env["stratum_of"]
At = Bt = 0.0
for i in ids:
    h = so[int(i)]
    z = int(lab[int(i)][0]) + 2 * int(lab[int(i)][1])
    if h in a_hz:
        At += a_hz[h][z]
        Bt += b[z]
out = {"polygon_A": pts[:, 0].tolist(), "polygon_B": pts[:, 1].tolist(), "A_lo": A_lo, "A_hi": A_hi,
       "B_lo": B_lo, "B_hi": B_hi, "L_joint": L, "U_joint": U,
       "L_sep": max(0.0, A_lo / B_hi), "U_sep": min(1.0, A_hi / B_lo), "A_true": At, "B_true": Bt,
       "calls": calls, "status": res[q.qid].status}
json.dump(out, open(os.path.join(S.OUT_DIR, "exp_geometry.json"), "w"), indent=1)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if not k.startswith("polygon")})
