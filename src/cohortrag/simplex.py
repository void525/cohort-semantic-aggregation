# -*- coding: utf-8 -*-
"""小型线性规划求解器（两阶段单纯形 + Bland 规则）。

只依赖 numpy，无第三方优化库依赖。
区间投影（草案 8.4 的 Charnes-Cooper 变换）需要求解小 LP（变量 <= 129），
本模块提供正确、可审计的实现。
"""
import numpy as np

TOL = 1e-10

def _pivot(T, basis, r, col):
    """以 (r, col) 为主元做转轴；T[0] 是目标行，T[1:] 是约束行。

    用秩一外积一次完成全部行更新（纯 numpy，无 Python 行循环）。"""
    T[r + 1] = T[r + 1] / T[r + 1, col]
    row = T[r + 1].copy()
    coeffs = T[:, col].copy()
    coeffs[r + 1] = 0.0
    T -= np.outer(coeffs, row)
    basis[r] = col

def _simplex_loop(T, basis, m, nvar, max_pivots=2000):
    """最大化；目标行放于 T[0]（系数 = −c 约定，负数 reduced cost 表示可改进）。
       返回 'optimal' / 'unbounded' / 'cycle'（防退化实例死循环的兜底）。"""
    pivots = 0
    while True:
        ent = -1
        for j in range(nvar):
            if T[0, j] < -TOL:
                ent = j
                break
        if ent < 0:
            return "optimal"
        lev, best = -1, np.inf
        for r in range(m):
            a = T[r + 1, ent]
            if a > TOL:
                ratio = T[r + 1, -1] / a
                if ratio < best - TOL:
                    lev, best = r, ratio
                elif abs(ratio - best) <= TOL and (lev == -1 or basis[r] < basis[lev]):
                    lev, best = r, ratio
        if lev < 0:
            return "unbounded"
        pivots += 1
        if pivots > max_pivots:
            return "cycle"
        _pivot(T, basis, lev, ent)

def lp_max(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    """maximize c^T x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  x >= 0。

    返回 (status, x, obj)；status ∈ {"optimal", "infeasible", "unbounded"}。
    实现：全部行转等式 + 人工变量两阶段法；Bland 规则防循环。
    """
    c = np.asarray(c, dtype=np.float64).ravel()
    n = len(c)
    if A_ub is None:
        A_ub = np.zeros((0, n))
        b_ub = np.zeros(0)
    else:
        A_ub = np.asarray(A_ub, dtype=np.float64)
        b_ub = np.asarray(b_ub, dtype=np.float64).ravel()
    if A_eq is None:
        A_eq = np.zeros((0, n))
        b_eq = np.zeros(0)
    else:
        A_eq = np.asarray(A_eq, dtype=np.float64)
        b_eq = np.asarray(b_eq, dtype=np.float64).ravel()

    # 无约束情形
    if A_ub.shape[0] == 0 and A_eq.shape[0] == 0:
        if np.any(c > TOL):
            return ("unbounded", None, np.inf)
        return ("optimal", np.zeros(n), 0.0)

    rows = []          # (coeffs, rhs, slack_sign)  slack_sign: +1 松弛 / -1 剩余 / 0 无（等式）
    for i in range(A_ub.shape[0]):
        a, b = A_ub[i].copy(), float(b_ub[i])
        if b >= 0.0:
            rows.append((a, b, 1))
        else:                      # b<0：整行乘 -1 变 >=，加剩余变量 + 人工变量
            rows.append((-a, -b, -1))
    for i in range(A_eq.shape[0]):
        a, b = A_eq[i].copy(), float(b_eq[i])
        if b >= 0.0:
            rows.append((a, b, 0))
        else:
            rows.append((-a, -b, 0))

    m = len(rows)
    n_slack = A_ub.shape[0]
    n_art = sum(1 for _, _, sg in rows if sg != 1)
    nvar = n + n_slack + n_art

    T = np.zeros((m + 1, nvar + 1))
    basis = np.zeros(m, dtype=int)
    art_cols = []
    sl_i, art_i = 0, 0
    for r, (a, b, sg) in enumerate(rows):
        T[r + 1, :n] = a
        T[r + 1, -1] = b
        if sg == 1:
            col = n + sl_i
            T[r + 1, col] = 1.0
            basis[r] = col
            sl_i += 1
        else:
            if sg == -1:
                col = n + sl_i
                T[r + 1, col] = -1.0
                sl_i += 1
            acol = n + n_slack + art_i
            T[r + 1, acol] = 1.0
            basis[r] = acol
            art_cols.append(acol)
            art_i += 1

    # ---- 阶段 I：min Σ人工 = max -Σ人工（目标行 = −c：人工列系数 +1）----
    T[0, :] = 0.0
    for ac in art_cols:
        T[0, ac] = 1.0
    for r in range(m):
        T[0] = T[0] - T[0, basis[r]] * T[r + 1]
    st = _simplex_loop(T, basis, m, nvar)
    if st == "unbounded":            # 阶段 I 不可能无界（有界可行域由人工变量保证），防御
        return ("infeasible", None, None)
    if st == "cycle":
        return ("cycle", None, None)
    if T[0, -1] < -1e-7:             # Σ人工 > 0 → 原问题不可行
        return ("infeasible", None, None)

    # 驱除仍在基中的人工变量（值应≈0），必要时删除冗余行
    art_set = set(art_cols)
    drop = []
    for r in range(m):
        if basis[r] in art_set:
            if T[r + 1, -1] > 1e-7:
                return ("infeasible", None, None)
            found = -1
            for j in range(nvar):
                if j not in art_set and abs(T[r + 1, j]) > 1e-12:
                    found = j
                    break
            if found >= 0:
                _pivot(T, basis, r, found)   # rhs=0，转轴不破坏可行性
            else:
                drop.append(r)               # 冗余行：删除
    if drop:
        keep = [i for i in range(m) if i not in drop]
        T = np.vstack([T[0], T[np.array(keep) + 1]])
        basis = basis[keep]
        m = len(keep)
    # 删除人工列
    keep_cols = [j for j in range(nvar) if j not in art_set]
    T = np.hstack([T[:, keep_cols], T[:, -1:]])
    nvar2 = len(keep_cols)
    basis = np.array([keep_cols.index(b) for b in basis], dtype=int)

    # ---- 阶段 II：最大化 c^T x ----
    T[0, :] = 0.0
    T[0, :n] = -c
    for r in range(m):
        T[0] = T[0] - T[0, basis[r]] * T[r + 1]
    st = _simplex_loop(T, basis, m, nvar2)
    if st == "unbounded":
        return ("unbounded", None, np.inf)
    if st == "cycle":
        return ("cycle", None, None)
    x = np.zeros(n)
    for r in range(m):
        bv = basis[r]
        if bv < n:
            x[bv] = T[r + 1, -1]
    x = np.maximum(x, 0.0)           # 清理极小负噪声
    return ("optimal", x, float(T[0, -1]))
