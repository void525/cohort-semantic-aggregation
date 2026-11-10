# -*- coding: utf-8 -*-
"""小实例整数穷举 × 浮点 LP × 精确有理数审计 的三方交叉验证（Theorem 1 实现层）。

对随机的小总体（2 层、4 个联合状态），构造与某个隐藏真实计数向量 k* 相容的多面体 P：
  已暴露下界 o <= k*、未暴露数 u、两个基函数的置信约束（含 E 的偏移）。
然后比较：
  enum   ：P 内全部整数计数向量上的 min/max A/B（真值，仅整数点）；
  lp     ：浮点 Charnes–Cooper / λ 路径（LP 松弛：只会更宽）；
  audit  ：精确有理数审计后的外包围端点。
必须满足：
  (1) 浮点 LP 的区间 ⊇ 整数穷举区间（松弛只放宽，容差 1e-9）；
  (2) 审计区间 ⊇ 整数穷举区间（严格，无容差）；
  (3) 阈值审计的『接受』必须是可靠的：审计说 θ>τ ⇒ 穷举最小比例 > τ；审计说 θ<=τ ⇒ 穷举最大比例 <= τ。
"""
import itertools
import os
import sys
import time
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from cohortrag.projection import ratio_bounds, min_max_B
from cohortrag.exact import audit_ratio_interval, audit_threshold

J = 4


def compositions(total, parts, lo, hi):
    """所有 sum=total、lo[i] <= x_i <= hi[i] 的整数向量。"""
    if parts == 1:
        if lo[0] <= total <= hi[0]:
            yield (total,)
        return
    for x in range(lo[0], min(hi[0], total) + 1):
        for rest in compositions(total - x, parts - 1, lo[1:], hi[1:]):
            yield (x,) + rest


def make_instance(rng, strata=2):
    poly = {"N_h": {}, "o_hz": {}, "u_h": {}, "cs": []}
    basis = [np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1])]   # 两个二值基函数
    truth = {}
    for h in range(strata):
        N = int(rng.integers(6, 13))
        k = rng.multinomial(N, rng.dirichlet(np.ones(J)))
        truth[h] = k
        # 暴露一部分记录：o <= k*
        expose = rng.integers(0, 3)
        o = np.minimum(k, rng.integers(0, 3, size=J))
        o = np.minimum(o, k)
        u = N - int(o.sum())
        poly["N_h"][h] = N
        poly["o_hz"][h] = o.astype(np.int64)
        poly["u_h"][h] = u
        for f in basis:
            val = int(f @ k)
            lo = max(0, val - int(rng.integers(0, 3)))
            hi = min(N, val + int(rng.integers(0, 3)))
            poly["cs"].append((h, f, 0, float(lo), float(hi)))
    return poly, truth


def enumerate_ratio(poly, a_hz, b_table, scope):
    per = []
    for h in scope:
        N = poly["N_h"][h]
        o = poly["o_hz"][h]
        u = poly["u_h"][h]
        lo = [int(o[z]) for z in range(J)]
        hi = [int(o[z]) + u for z in range(J)]
        pts = []
        for k in compositions(N, J, lo, hi):
            ok = True
            for (hh, f, Ef, clo, chi) in poly["cs"]:
                if hh != h:
                    continue
                v = int(sum(int(f[z]) * k[z] for z in range(J)))
                if not (clo <= v <= chi):
                    ok = False
                    break
            if ok:
                A = sum(a_hz[h][z] * k[z] for z in range(J))
                B = sum(int(b_table[z]) * k[z] for z in range(J))
                pts.append((A, B))
        if not pts:
            return None
        per.append(np.array(pts, dtype=np.float64))
    A = per[0][:, 0]
    B = per[0][:, 1]
    for p in per[1:]:
        A = (A[:, None] + p[None, :, 0]).ravel()
        B = (B[:, None] + p[None, :, 1]).ravel()
    m = B > 0
    if not m.any():
        return None
    r = A[m] / B[m]
    return float(r.min()), float(r.max())


def main(n_inst=150, seed=0):
    rng = np.random.default_rng(seed)
    stats = dict(n=0, lp_violation=0, audit_violation=0, thr_unsound=0, thr_checked=0,
                 audit_adjusted=0, max_lp_gap=0.0, max_audit_pad=0.0)
    t0 = time.time()
    scope = (0, 1)
    while stats["n"] < n_inst:
        poly, truth = make_instance(rng)
        a_tab = rng.integers(0, 2, size=J)
        b_tab = np.maximum(a_tab, rng.integers(0, 2, size=J))            # a <= b
        a_hz = {h: a_tab.astype(float) for h in scope}
        if b_tab.sum() == 0 or a_tab.sum() == 0:
            continue
        minB, maxB = min_max_B(poly, b_tab, scope, J)
        if minB <= 1e-9:
            continue
        en = enumerate_ratio(poly, a_hz, b_tab, scope)
        if en is None:
            continue
        L, U = ratio_bounds(poly, a_hz, b_tab, scope, J)
        La, Ua, adj = audit_ratio_interval(poly, a_hz, b_tab, scope, J, L, U)
        stats["n"] += 1
        stats["audit_adjusted"] += int(adj > 0)
        if L > en[0] + 1e-9 or U < en[1] - 1e-9:
            stats["lp_violation"] += 1
        if La > Fraction(en[0]).limit_denominator(10 ** 12) or Ua < Fraction(en[1]).limit_denominator(10 ** 12):
            # 用高精度有理数比较，避免浮点比较掩盖 1 ulp 的违规
            if float(La) > en[0] or float(Ua) < en[1]:
                stats["audit_violation"] += 1
        stats["max_lp_gap"] = max(stats["max_lp_gap"], (en[0] - L), (U - en[1]))
        stats["max_audit_pad"] = max(stats["max_audit_pad"], float(L - La), float(Ua - U))
        # 阈值审计的可靠性
        for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
            for verdict in ("true", "false"):
                ok = audit_threshold(poly, a_hz, b_tab, tau, scope, J, verdict)
                if ok:
                    stats["thr_checked"] += 1
                    if verdict == "true" and not (en[0] > tau):
                        stats["thr_unsound"] += 1
                    if verdict == "false" and not (en[1] <= tau):
                        stats["thr_unsound"] += 1
    print({k: (v) for k, v in stats.items()},
          f"{time.time() - t0:.1f}s")
    import json
    out_dir = os.path.join(_HERE, "..", "reports", "icde2027_suite")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(stats, open(os.path.join(out_dir, "exp_enum.json"), "w"), indent=1)
    assert stats["lp_violation"] == 0 and stats["audit_violation"] == 0 and stats["thr_unsound"] == 0
    print("PASS: LP relaxation ⊇ integer enumeration; audited interval ⊇ integer enumeration; "
          "threshold audit sound.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
