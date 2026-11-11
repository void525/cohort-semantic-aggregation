# -*- coding: utf-8 -*-
"""精确有理数审计路径（Theorem 1 的「实现层」）。

问题（评审 4.2）：数学上联合事件 k* ∈ P 成立时，精确投影必然包住真实值；
但浮点 LP 的求解误差可能让输出区间向内收缩，因此「LP 容差 ζ ≤ ε_q」并不能保证覆盖。

做法：认证瞬间，用精确有理数重新验证「线性对比」不等式。
  - 区间下端 L 有效 ⇔ 对 P 内所有 k：A(k) − L·B(k) >= 0；
  - 区间上端 U 有效 ⇔ 对 P 内所有 k：A(k) − U·B(k) <= 0；
  - 阈值 θ > τ 成立 ⇔ min_P (A − τB) > 0；θ <= τ 成立 ⇔ max_P (A − τB) <= 0。
P 是各层多面体之积（层间无共享约束），上述对比问题逐层可分，每层是 <= 16 变量的小 LP，
可以用 Fraction 精确求解。P 的所有数据（N_h、已暴露计数、未暴露数、CS 端点、E 的偏移）
都是整数，因此审计是对「给定整数多面体」的严格证明；
置信序列端点本身由 lgamma 浮点计算得到，我们在 build_polytope 中向外各放宽 1 个计数
（远大于浮点误差对应的计数偏移）以吸收这部分数值误差。

审计只在认证瞬间对即将输出的结论运行（每个会话 <= |Q| 次），不进入每轮规划的热路径。
"""
import math
from fractions import Fraction


# --------------------------------------------------------------------------- #
# 精确两阶段单纯形（Bland 规则）：maximize c·x  s.t.  A_ub x <= b_ub, A_eq x = b_eq, x >= 0
# --------------------------------------------------------------------------- #
def _pivot(T, basis, r, col):
    pr = T[r + 1]
    pv = pr[col]
    T[r + 1] = pr = [v / pv for v in pr]
    for i in range(len(T)):
        if i == r + 1:
            continue
        f = T[i][col]
        if f != 0:
            T[i] = [a - f * b for a, b in zip(T[i], pr)]
    basis[r] = col


def _loop(T, basis, m, nvar):
    while True:
        ent = -1
        for j in range(nvar):
            if T[0][j] < 0:
                ent = j
                break
        if ent < 0:
            return "optimal"
        lev, best = -1, None
        for r in range(m):
            a = T[r + 1][ent]
            if a > 0:
                ratio = T[r + 1][-1] / a
                if best is None or ratio < best or (ratio == best and basis[r] < basis[lev]):
                    lev, best = r, ratio
        if lev < 0:
            return "unbounded"
        _pivot(T, basis, lev, ent)


def lp_max_exact(c, A_ub, b_ub, A_eq, b_eq):
    """返回 (status, x, obj)；全部为 Fraction。"""
    n = len(c)
    c = [Fraction(v) for v in c]
    rows = []
    for a, b in zip(A_ub, b_ub):
        a = [Fraction(v) for v in a]
        b = Fraction(b)
        rows.append((a, b, 1) if b >= 0 else ([-v for v in a], -b, -1))
    for a, b in zip(A_eq, b_eq):
        a = [Fraction(v) for v in a]
        b = Fraction(b)
        rows.append((a, b, 0) if b >= 0 else ([-v for v in a], -b, 0))
    m = len(rows)
    n_slack = len(A_ub)
    n_art = sum(1 for _, _, sg in rows if sg != 1)
    nvar = n + n_slack + n_art
    zero = Fraction(0)
    T = [[zero] * (nvar + 1) for _ in range(m + 1)]
    basis = [0] * m
    art_cols = []
    sl_i = art_i = 0
    for r, (a, b, sg) in enumerate(rows):
        T[r + 1][:n] = a
        T[r + 1][-1] = b
        if sg == 1:
            col = n + sl_i
            T[r + 1][col] = Fraction(1)
            basis[r] = col
            sl_i += 1
        else:
            if sg == -1:
                T[r + 1][n + sl_i] = Fraction(-1)
                sl_i += 1
            acol = n + n_slack + art_i
            T[r + 1][acol] = Fraction(1)
            basis[r] = acol
            art_cols.append(acol)
            art_i += 1
    # 阶段 I
    T[0] = [zero] * (nvar + 1)
    for ac in art_cols:
        T[0][ac] = Fraction(1)
    for r in range(m):
        f = T[0][basis[r]]
        if f != 0:
            T[0] = [a - f * b for a, b in zip(T[0], T[r + 1])]
    st = _loop(T, basis, m, nvar)
    if st != "optimal" or T[0][-1] < 0:
        return ("infeasible", None, None)
    art_set = set(art_cols)
    drop = []
    for r in range(m):
        if basis[r] in art_set:
            found = -1
            for j in range(nvar):
                if j not in art_set and T[r + 1][j] != 0:
                    found = j
                    break
            if found >= 0:
                _pivot(T, basis, r, found)
            else:
                drop.append(r)
    if drop:
        keep = [i for i in range(m) if i not in drop]
        T = [T[0]] + [T[i + 1] for i in keep]
        basis = [basis[i] for i in keep]
        m = len(keep)
    keep_cols = [j for j in range(nvar) if j not in art_set]
    T = [[row[j] for j in keep_cols] + [row[-1]] for row in T]
    remap = {old: new for new, old in enumerate(keep_cols)}
    basis = [remap[b] for b in basis]
    nvar2 = len(keep_cols)
    # 阶段 II
    T[0] = [zero] * (nvar2 + 1)
    for j in range(n):
        T[0][j] = -c[j]
    for r in range(m):
        f = T[0][basis[r]]
        if f != 0:
            T[0] = [a - f * b for a, b in zip(T[0], T[r + 1])]
    st = _loop(T, basis, m, nvar2)
    if st == "unbounded":
        return ("unbounded", None, None)
    x = [zero] * n
    for r in range(m):
        if basis[r] < n:
            x[basis[r]] = T[r + 1][-1]
    return ("optimal", x, T[0][-1])


# --------------------------------------------------------------------------- #
# 逐层精确极值
# --------------------------------------------------------------------------- #
def _stratum_data(poly, h, J):
    N = int(poly["N_h"][h])
    o = [int(v) for v in poly["o_hz"][h]]
    u = int(poly["u_h"][h])
    A_ub, b_ub = [], []
    for z in range(J):
        r1 = [0] * J
        r1[z] = 1
        A_ub.append(r1)
        b_ub.append(o[z] + u)
        r2 = [0] * J
        r2[z] = -1
        A_ub.append(r2)
        b_ub.append(-o[z])
    for (hh, table, E_f, lo, hi) in poly["cs"]:
        if hh != h:
            continue
        row = [int(table[z]) for z in range(J)]
        A_ub.append(row)
        b_ub.append(int(round(E_f)) + int(round(hi)))
        A_ub.append([-v for v in row])
        b_ub.append(-(int(round(E_f)) + int(round(lo))))
    return A_ub, b_ub, [[1] * J], [N]


def exact_min_stratum(poly, h, cvec, J):
    """精确 min_{k ∈ P_h} cvec·k（cvec 为 Fraction 列表）。"""
    A_ub, b_ub, A_eq, b_eq = _stratum_data(poly, h, J)
    st, _, obj = lp_max_exact([-v for v in cvec], A_ub, b_ub, A_eq, b_eq)
    if st != "optimal":
        raise RuntimeError(f"exact LP status {st} on stratum {h}")
    return -obj


def _contrast_vec(a_h, b_table, lam, J):
    """a_h − lam·b（逐状态）。"""
    return [Fraction(int(round(a_h[z]))) - lam * int(b_table[z]) for z in range(J)]


def exact_min_contrast(poly, a_hz, b_table, lam, scope, J):
    """精确 min_{k∈P} Σ_h (a_h − lam·b)·k。"""
    return sum(exact_min_stratum(poly, h, _contrast_vec(a_hz[h], b_table, lam, J), J)
               for h in scope)


def exact_max_contrast(poly, a_hz, b_table, lam, scope, J):
    """精确 max_{k∈P} Σ_h (a_h − lam·b)·k。"""
    total = Fraction(0)
    for h in scope:
        c = [-v for v in _contrast_vec(a_hz[h], b_table, lam, J)]
        total += -exact_min_stratum(poly, h, c, J)
    return total


# --------------------------------------------------------------------------- #
# 审计接口
# --------------------------------------------------------------------------- #
_DEN = 2 ** 30


def _exact_domain_bounds(poly, a_hz, b_table, scope, J):
    """Derive exact rational domain bounds [lo_frac, hi_frac] from query coefficients."""
    masked = True
    ratios = []
    for h in scope:
        for z in range(J):
            bz = int(round(float(b_table[z])))
            az = int(round(float(a_hz[h][z])))
            if bz > 0:
                ratios.append(Fraction(az, bz))
            elif az != 0:
                masked = False
    if masked and ratios:
        return min(ratios), max(ratios)
    b_min = sum(
        exact_min_stratum(poly, h, [Fraction(int(round(float(b_table[z])))) for z in range(J)], J)
        for h in scope
    )
    b_floor = b_min if b_min > 0 else Fraction(1, 1)
    max_abs_a = sum(
        int(poly["N_h"][h]) * max(abs(int(round(float(a_hz[h][z])))) for z in range(J))
        for h in scope
    )
    bound = max(Fraction(1, 1), Fraction(max_abs_a, 1) / b_floor)
    return -bound, bound


def audit_ratio_interval(poly, a_hz, b_table, scope, J, L_float, U_float):
    """返回精确验证过的外包围端点 (L_safe, U_safe, n_exact_lps_adjusted)。

    端点取 2^-30 网格上略微向外的有理数；若精确检验不通过，采用指数倍增步长 + 二分快速收敛。
    """
    lo_frac, hi_frac = _exact_domain_bounds(poly, a_hz, b_table, scope, J)
    adjust = 0
    L = max(lo_frac, Fraction(math.floor(L_float * _DEN) - 1, _DEN))
    if L > lo_frac and exact_min_contrast(poly, a_hz, b_table, L, scope, J) < 0:
        hi_bad = L
        step = Fraction(1, 2 ** 20)
        lo_ok = max(lo_frac, L - step)
        adjust += 1
        while lo_ok > lo_frac and exact_min_contrast(poly, a_hz, b_table, lo_ok, scope, J) < 0:
            hi_bad = lo_ok
            step *= 2
            lo_ok = max(lo_frac, hi_bad - step)
            adjust += 1
        # Optional tight refinement over 2^-30 grid in <= 25 steps
        for _ in range(24):
            if hi_bad - lo_ok <= Fraction(1, _DEN):
                break
            mid_k = ((lo_ok + hi_bad) * _DEN) // 2
            mid = Fraction(mid_k, _DEN)
            if mid <= lo_ok or mid >= hi_bad:
                break
            adjust += 1
            if exact_min_contrast(poly, a_hz, b_table, mid, scope, J) >= 0:
                lo_ok = mid
            else:
                hi_bad = mid
        L = lo_ok

    U = min(hi_frac, Fraction(math.ceil(U_float * _DEN) + 1, _DEN))
    if U < hi_frac and exact_max_contrast(poly, a_hz, b_table, U, scope, J) > 0:
        lo_bad = U
        step = Fraction(1, 2 ** 20)
        hi_ok = min(hi_frac, U + step)
        adjust += 1
        while hi_ok < hi_frac and exact_max_contrast(poly, a_hz, b_table, hi_ok, scope, J) > 0:
            lo_bad = hi_ok
            step *= 2
            hi_ok = min(hi_frac, lo_bad + step)
            adjust += 1
        for _ in range(24):
            if hi_ok - lo_bad <= Fraction(1, _DEN):
                break
            mid_k = ((lo_bad + hi_ok) * _DEN) // 2
            mid = Fraction(mid_k, _DEN)
            if mid <= lo_bad or mid >= hi_ok:
                break
            adjust += 1
            if exact_max_contrast(poly, a_hz, b_table, mid, scope, J) <= 0:
                hi_ok = mid
            else:
                lo_bad = mid
        U = hi_ok

    if L > U:
        raise RuntimeError(f"audit_ratio_interval inverted bounds: L={L} > U={U}")
    return L, U, adjust


def audit_threshold(poly, a_hz, b_table, tau, scope, J, verdict):
    """精确验证阈值结论。verdict: 'true' (θ>τ) 或 'false' (θ<=τ)。保留有理数阈值。"""
    t = tau if isinstance(tau, Fraction) else Fraction(tau).limit_denominator(10**15)
    if verdict == "true":
        return exact_min_contrast(poly, a_hz, b_table, t, scope, J) > 0
    return exact_max_contrast(poly, a_hz, b_table, t, scope, J) <= 0
