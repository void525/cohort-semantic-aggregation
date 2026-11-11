# -*- coding: utf-8 -*-
"""联合计数可行域与条件比例投影（草案 8.3 / 8.4）。

置信约束、确定性已知质量与总量守恒作用于同一个联合状态：
  P = { k :  o_z <= k_z <= o_z + u_unseen （已知质量）
             Σ_z k_z = N_h （总量守恒，完整层）
             E_f + ℓ <= Σ_z f(z) k_z <= E_f + u （置信序列，作用于剩余总体） }

变量 k 定义在【完整层】上（E 已折入已知质量下界 o）；CS 区间 ℓ,u 作用在
剩余总体 R = U 减 E 上，因此按 E 的已知贡献 E_f 平移（推导见 README）。

真实计数为整数；实现放松为实数约束（草案 8.3 明确允许：集合只扩大，覆盖不损失）。
条件比例用 Charnes–Cooper 变换转成小线性规划求解——这是数值优化，不是 LLM 任务。
"""
import numpy as np
from . import simplex

class FeasibilityError(Exception):
    """可行域无解：停止认证并记录错误（草案 8.3），不得静默删样本或反复重跑。"""

def build_polytope(ledger, cs_intervals, basis_tables, e_as_random=False, pad=0):
    """cs_intervals: {(h, f_idx): (lo, hi)}。返回可传入 _lp 的数据结构。

    e_as_random：去除选择来源契约时，历史缓存不再作为精确已知下界钉死，故每类上界
    松到整层 N_h（仅由总量守恒约束），由 CS 在其上作用——即“假装随机样本”。
    """
    poly = {
        "N_h": {h: ledger.N_h[h] for h in ledger.strata},
        "o_hz": {h: ledger.obs[h].astype(np.float64).copy() for h in ledger.strata},
        "u_h": {h: (ledger.N_h[h] if e_as_random else ledger.unseen(h))
                for h in ledger.strata},
        "cs": [],
    }
    for (h, f), interval in cs_intervals.items():
        if interval is None:
            raise FeasibilityError(f"置信区间无解：层 {h} 基函数 {f}（fail-closed）")
        table = basis_tables[f]
        E_f = int(sum(table[z] * ledger.E_counts[h][z] for z in range(ledger.J)))
        # 置信序列端点向外各放宽 pad 个计数：吸收 lgamma 浮点计算对端点的任何微小扰动
        poly["cs"].append((h, table, E_f, max(0.0, float(interval[0]) - pad),
                           float(interval[1]) + pad))
    return poly

def _lp_matrices(poly, scope, J):
    """为 scope 内各层构造 LP 约束矩阵（变量 k_{h,z}，索引 h_idx*J+z）。"""
    hs = [h for h in scope]
    m = len(hs) * J
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for ih, h in enumerate(hs):
        N_h = poly["N_h"][h]
        row = np.zeros(m)
        row[ih * J:(ih + 1) * J] = 1.0
        A_eq.append(row)
        b_eq.append(float(N_h))
        o = poly["o_hz"][h]
        u = poly["u_h"][h]
        for z in range(J):
            r1 = np.zeros(m)
            r1[ih * J + z] = 1.0
            A_ub.append(r1)
            b_ub.append(float(o[z] + u))
            r2 = np.zeros(m)
            r2[ih * J + z] = -1.0
            A_ub.append(r2)
            b_ub.append(-float(o[z]))
    for (h, table, E_f, lo, hi) in poly["cs"]:
        if h not in hs:
            continue
        ih = hs.index(h)
        row = np.zeros(m)
        for z in range(J):
            row[ih * J + z] = float(table[z])
        A_ub.append(row.copy())
        b_ub.append(float(E_f + hi))
        A_ub.append(-row)
        b_ub.append(-float(E_f + lo))
    return (np.array(A_ub), np.array(b_ub),
            np.array(A_eq), np.array(b_eq), hs)

def _check(status, what):
    if status == "infeasible":
        raise FeasibilityError(what + "：可行域无解")
    if status == "unbounded":
        raise FeasibilityError(what + "：意外无界")
    if status == "cycle":
        raise FeasibilityError(what + "：单纯形退化循环")

def _stratum_lp(poly, h, c, J, need_solution=False):
    """单层小 LP：min c^T k_h s.t. k_h ∈ P_h。

    P_h 约束：Σ k = N_h；o_z <= k_z <= o_z+u；置信约束（该层）。
    层间无共享约束 → 全问题可分层分解（产品结构）。
    返回 (最优值, 解向量)（need_solution=False 时解为 None）。"""
    o = poly["o_hz"][h]
    u = poly["u_h"][h]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    row = np.ones(J)
    A_eq.append(row)
    b_eq.append(float(poly["N_h"][h]))
    for z in range(J):
        r1 = np.zeros(J)
        r1[z] = 1.0
        A_ub.append(r1)
        b_ub.append(float(o[z] + u))
        r2 = np.zeros(J)
        r2[z] = -1.0
        A_ub.append(r2)
        b_ub.append(-float(o[z]))
    for (hh, table, E_f, lo, hi) in poly["cs"]:
        if hh != h:
            continue
        rowf = table.astype(np.float64)
        A_ub.append(rowf.copy())
        b_ub.append(float(E_f + hi))
        A_ub.append(-rowf)
        b_ub.append(-float(E_f + lo))
    st, x, obj = simplex.lp_max(-np.asarray(c, dtype=np.float64),
                                np.array(A_ub), np.array(b_ub),
                                np.array(A_eq), np.array(b_eq))
    _check(st, f"stratum {h} extreme")
    return -obj, (x if need_solution else None)

def _stratum_extreme(poly, h, c, J):
    val, _ = _stratum_lp(poly, h, c, J, need_solution=False)
    return val

def min_max_B(poly, b_table, scope, J):
    """B(k) 在可行域上的最小/最大值（可行域是各层之积 → 逐层求解相加）。"""
    minB = 0.0
    maxB = 0.0
    for h in scope:
        c = b_table.astype(np.float64)
        minB += _stratum_extreme(poly, h, c, J)
        maxB += -_stratum_extreme(poly, h, -c, J)
    return minB, maxB

# ---------------- Charnes–Cooper 主路径 ----------------

def _as_a_hz(a, scope, J):
    """把分子系数统一成 {层 h: J-向量}。

    支持两种输入（向后兼容）：
      - 单个 J-向量：所有层共用同一语义分子条件（分子只依赖原语）；
      - {h: J-向量}：分子还依赖元数据（草案 5.2 的 r = r(m,g)，原子内恒定）。
    """
    if isinstance(a, dict):
        zero = np.zeros(J)
        return {h: np.asarray(a.get(h, zero), dtype=np.float64) for h in scope}
    arr = np.asarray(a, dtype=np.float64)
    return {h: arr for h in scope}

def _cc_matrices(poly, a, b_table, scope, J):
    """构造 CC 变换后的 LP 矩阵（变量 w[m] + t[m]）。"""
    A_ub, b_ub, A_eq, b_eq, hs = _lp_matrices(poly, scope, J)
    m = len(hs) * J
    n = m + 1                       # 变量：w (m) + t (最后一个)
    t_idx = m

    cc_Aub, cc_bub, cc_Aeq, cc_beq = [], [], [], []
    row = np.zeros(n)               # 归一化等式：c_B^T w = 1
    for ih in range(len(hs)):
        for z in range(J):
            row[ih * J + z] = float(b_table[z])
    cc_Aeq.append(row)
    cc_beq.append(1.0)
    for ih, h in enumerate(hs):     # 每层质量：Σ w_hz - N_h t = 0
        row = np.zeros(n)
        row[ih * J:(ih + 1) * J] = 1.0
        row[t_idx] = -poly["N_h"][h]
        cc_Aeq.append(row)
        cc_beq.append(0.0)
    for ih, h in enumerate(hs):     # 已知质量：w_z <= (o_z+u)t；-w_z <= -o_z t
        o = poly["o_hz"][h]
        u = poly["u_h"][h]
        for z in range(J):
            r1 = np.zeros(n)
            r1[ih * J + z] = 1.0
            r1[t_idx] = -(o[z] + u)
            cc_Aub.append(r1)
            cc_bub.append(0.0)
            r2 = np.zeros(n)
            r2[ih * J + z] = -1.0
            r2[t_idx] = float(o[z])
            cc_Aub.append(r2)
            cc_bub.append(0.0)
    for (h, table, E_f, lo, hi) in poly["cs"]:   # 置信约束（含 E_f 平移）
        if h not in hs:
            continue
        ih = hs.index(h)
        row = np.zeros(n)
        for z in range(J):
            row[ih * J + z] = float(table[z])
        row[t_idx] = -(E_f + hi)
        cc_Aub.append(row.copy())
        cc_bub.append(0.0)
        row2 = np.zeros(n)
        for z in range(J):
            row2[ih * J + z] = -float(table[z])
        row2[t_idx] = (E_f + lo)
        cc_Aub.append(row2)
        cc_bub.append(0.0)

    a_hz = _as_a_hz(a, scope, J)
    c = np.zeros(n)
    for ih, h in enumerate(hs):
        c[ih * J:(ih + 1) * J] = a_hz[h]
    return (np.array(cc_Aub), np.array(cc_bub),
            np.array(cc_Aeq), np.array(cc_beq), c)

def _cc_min(poly, a, b_table, scope, J):
    cc_Aub, cc_bub, cc_Aeq, cc_beq, c = _cc_matrices(poly, a, b_table, scope, J)
    st, _, obj = simplex.lp_max(-c, cc_Aub, cc_bub, cc_Aeq, cc_beq)
    _check(st, "ratio min (CC)")
    return float(np.clip(-obj, 0.0, 1.0))

def _cc_max(poly, a, b_table, scope, J):
    cc_Aub, cc_bub, cc_Aeq, cc_beq, c = _cc_matrices(poly, a, b_table, scope, J)
    st, _, obj = simplex.lp_max(c, cc_Aub, cc_bub, cc_Aeq, cc_beq)
    _check(st, "ratio max (CC)")
    return float(np.clip(obj, 0.0, 1.0))

# ---------------- λ 对偶兜底（分层可分解，Dinkelbach 超线性迭代） ----------------

def _lambda_min(poly, a, b_table, scope, J, iters=30):
    """L = min A/B。Dinkelbach：k* ∈ argmin(A−λB)（可分层分解），λ ← A(k*)/B(k*)。

    序列单调不减收敛到 θ_min；以相邻迭代变化量 < 1e-9 判收敛。
    前置 minB > 0 保证每个可行点的 B > 0，故 B(k*) > 0 恒成立。"""
    a_hz = _as_a_hz(a, scope, J)
    bb = b_table.astype(np.float64)
    lam = 0.0
    lam_next = lam
    for _ in range(iters):
        A = B = 0.0
        for h in scope:
            _, k = _stratum_lp(poly, h, a_hz[h] - lam * bb, J, need_solution=True)
            A += float(a_hz[h] @ k)
            B += float(bb @ k)
        if B <= 1e-12:              # 数值退化：回退保守二分
            return _lambda_min_bisect(poly, a, b_table, scope, J)
        lam_next = A / B
        if lam_next - lam < 1e-9:
            break
        lam = lam_next
    return float(np.clip(lam_next, 0.0, 1.0))

def _lambda_max(poly, a, b_table, scope, J, iters=30):
    """U = max A/B。Dinkelbach：k* ∈ argmax(A−λB) = argmin(λB−A)，λ ← A(k*)/B(k*)。

    序列单调不增收敛到 θ_max；以相邻迭代变化量 < 1e-9 判收敛。"""
    a_hz = _as_a_hz(a, scope, J)
    bb = b_table.astype(np.float64)
    lam = 1.0
    lam_next = lam
    for _ in range(iters):
        A = B = 0.0
        for h in scope:
            _, k = _stratum_lp(poly, h, lam * bb - a_hz[h], J, need_solution=True)
            A += float(a_hz[h] @ k)
            B += float(bb @ k)
        if B <= 1e-12:
            return _lambda_max_bisect(poly, a, b_table, scope, J)
        lam_next = A / B
        if lam - lam_next < 1e-9:
            break
        lam = lam_next
    return float(np.clip(lam_next, 0.0, 1.0))

def _lambda_min_bisect(poly, a, b_table, scope, J, iters=45):
    """数值安全网：对 λ 二分。"""
    a_hz = _as_a_hz(a, scope, J)
    bb = b_table.astype(np.float64)

    def G(lam):
        return sum(_stratum_extreme(poly, h, a_hz[h] - lam * bb, J) for h in scope)
    if G(1.0) >= -1e-9:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if G(mid) >= -1e-9:
            lo = mid
        else:
            hi = mid
    return float(lo)

def _lambda_max_bisect(poly, a, b_table, scope, J, iters=45):
    """数值安全网：对 λ 二分。"""
    a_hz = _as_a_hz(a, scope, J)
    bb = b_table.astype(np.float64)

    def H(lam):
        return sum(-_stratum_extreme(poly, h, a_hz[h] - lam * bb, J) for h in scope)
    if H(0.0) <= 1e-9:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if H(mid) <= 1e-9:
            hi = mid
        else:
            lo = mid
    return float(hi)

def threshold_margins(poly, a, b_table, tau, scope, J):
    """Direct linear contrast projection for threshold queries:
    m_q = min_{k in P} (A_q(k) - tau * B_q(k))
    M_q = max_{k in P} (A_q(k) - tau * B_q(k))
    Decomposes exactly across strata without fractional transformation.
    """
    a_hz = _as_a_hz(a, scope, J)
    bb = b_table.astype(np.float64)
    m_q = 0.0
    M_q = 0.0
    for h in scope:
        c = a_hz[h] - float(tau) * bb
        m_q += _stratum_extreme(poly, h, c, J)
        M_q += -_stratum_extreme(poly, h, -c, J)
    return float(m_q), float(M_q)

def verify_outward_enclosure(poly, a, b_table, scope, J, L_cand, U_cand, tol=1e-9):
    """Verify and enforce outward-rounded safe endpoints [L_safe, U_safe] such that
    for all k in P: A(k) - L_safe * B(k) >= -tol and A(k) - U_safe * B(k) <= tol.
    """
    a_hz = _as_a_hz(a, scope, J)
    bb = b_table.astype(np.float64)
    L_safe = max(0.0, float(L_cand) - 1e-9)
    U_safe = min(1.0, float(U_cand) + 1e-9)
    if L_safe > 0.0:
        # Check min_{k in P} (A(k) - L_safe * B(k)) >= -tol
        g_val = sum(_stratum_extreme(poly, h, a_hz[h] - L_safe * bb, J) for h in scope)
        if g_val < -1e-6:
            L_safe = max(0.0, _lambda_min_bisect(poly, a, b_table, scope, J) - 1e-9)
    if U_safe < 1.0:
        # Check max_{k in P} (A(k) - U_safe * B(k)) = - min_{k in P} (U_safe * B(k) - A(k)) <= tol
        h_val = sum(-_stratum_extreme(poly, h, U_safe * bb - a_hz[h], J) for h in scope)
        if h_val > 1e-6:
            U_safe = min(1.0, _lambda_max_bisect(poly, a, b_table, scope, J) + 1e-9)
    return float(L_safe), float(U_safe)

def ratio_bounds(poly, a, b_table, scope, J, verify_enclosure=False):
    """条件比例 [L, U]（前置：minB > 0）。a 可传单个 J-向量或 {h: J-向量}。

    当变量数 len(scope)*J > 32 时，分层分解的 Dinkelbach 超线性迭代等价于全局 CC LP，
    且避免 O((|scope|*J)^3) 大单纯形转轴；小规模直接走 Charnes–Cooper。
    向外扩充 1e-9 浮点安全边距，并在认证临界点通过 verify_outward_enclosure 验证线性对比外包围不变量。"""
    if len(scope) * J > 32:
        L = _lambda_min(poly, a, b_table, scope, J)
        U = _lambda_max(poly, a, b_table, scope, J)
    else:
        try:
            L = _cc_min(poly, a, b_table, scope, J)
        except FeasibilityError:
            L = _lambda_min(poly, a, b_table, scope, J)
        try:
            U = _cc_max(poly, a, b_table, scope, J)
        except FeasibilityError:
            U = _lambda_max(poly, a, b_table, scope, J)
        if L > U + 1e-7:                # 极端退化下两侧交叉不一致 → 双侧都走 λ
            L = _lambda_min(poly, a, b_table, scope, J)
            U = _lambda_max(poly, a, b_table, scope, J)
    if verify_enclosure:
        return verify_outward_enclosure(poly, a, b_table, scope, J, L, U)
    return float(max(0.0, L - 1e-9)), float(min(1.0, U + 1e-9))

def _ratio_bounds_cc(poly, a, b_table, scope, J):
    """（测试用）纯 CC 路径组合。"""
    L = _cc_min(poly, a, b_table, scope, J)
    U = _cc_max(poly, a, b_table, scope, J)
    if L > U + 1e-7:
        raise FeasibilityError("区间投影数值异常：L > U")
    return L, U

def ratio_bounds_lambda(poly, a, b_table, scope, J):
    """（测试用）纯 λ 路径组合。"""
    L = _lambda_min(poly, a, b_table, scope, J)
    U = _lambda_max(poly, a, b_table, scope, J)
    return L, U
