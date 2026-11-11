# -*- coding: utf-8 -*-
"""核心验证：
  1) CS 类 vs 逐候选穷举（与 validate_cs_demo.py 同一数学）；
  2) 区间投影（LP 放松）vs 整数穷举（小总体）；
  3) pipeline 端到端冒烟：覆盖率、预算、状态合法性。
"""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import pandas as pd
from cohortrag import (Snapshot, COHORTRAG, Config, make_workload,
                       RoundRobinAllocator, WorstVarianceAllocator)
from cohortrag.cs import FinitePopulationCS
from cohortrag.ledger import EvidenceLedger
from cohortrag.projection import (build_polytope, min_max_B, ratio_bounds,
                                  _ratio_bounds_cc, ratio_bounds_lambda,
                                  FeasibilityError)
from cohortrag.semantics import OracleOperator
from cohortrag.worldgen import make_synthetic_world
from cohortrag import states

# ---------- 1) CS 类 vs 穷举 ----------
def _brute_interval(N, counts, delta):
    def gammaln(x):
        return math.lgamma(x)
    def falling(a, b):
        return 0.0 if b == 0 else gammaln(a + 1) - gammaln(a - b + 1)
    thresh = -math.log(delta)
    feas = []
    for k in range(N + 1):
        rmax = float("-inf")
        ok = True
        for (s, x) in counts:
            if k < x or (N - k) < (s - x):
                ok = False
                break
            logL = falling(k, x) + falling(N - k, s - x) - falling(N, s)
            logM = gammaln(x + 1) + gammaln(s - x + 1) - gammaln(s + 2)
            rmax = max(rmax, logM - logL)
        if ok and rmax <= thresh:
            feas.append(k)
    return (min(feas), max(feas)) if feas else None

def test_cs_class_vs_bruteforce():
    checks = 0
    for N in (10, 20, 40):
        for kstar in (0, N // 2, N):
            for sd in (1, 2, 3):
                perm = [1] * kstar + [0] * (N - kstar)
                random.Random(sd).shuffle(perm)
                cs = FinitePopulationCS(N)
                x = 0
                counts = [(0, 0)]
                for s in range(1, N + 1):
                    x += perm[s - 1]
                    counts.append((s, x))
                    cs.push(perm[s - 1])
                    iv = cs.interval(0.05)
                    exp = _brute_interval(N, counts, 0.05)
                    checks += 1
                    if s == N:
                        assert iv == (x, x), (N, kstar, sd, s, iv)
                    else:
                        assert iv == exp, (N, kstar, sd, s, iv, exp)
    print(f"  CS 类 vs 穷举：{checks} 项全部一致")

# ---------- 2) 投影 LP vs 整数穷举 ----------
def _state_labels():
    """K=2、N=10：状态 00/10/01/11 各 2/3/2/3 条。"""
    arr = []
    for z, cnt in [(0, 2), (1, 3), (2, 2), (3, 3)]:
        arr += [states.decode_state(z, 2)] * cnt
    return {i: np.array(arr[i], dtype=np.int8) for i in range(10)}

def test_projection_vs_integer_enumeration():
    J = 4
    b_table = states.and_table(2, (0,))          # 分母 g0
    a_table = b_table & states.and_table(2, (1,)) # 分子 g0∧g1
    for sd in range(10):
        labels = _state_labels()
        frame = pd.DataFrame({
            "id": np.arange(10),
            "stratum": np.zeros(10, dtype=np.int64),
            "text": ["t"] * 10,
        })
        snap = Snapshot(frame, "tiny")
        ledger = EvidenceLedger(snap, seed=sd, K=2)
        contract = OracleOperator(labels, name="oracle", K=2)
        ledger.consume_prefix(0, 4, contract)     # 消费 4 个随机前缀位置

        basis = [states.and_table(2, (0,)), states.and_table(2, (1,)),
                 states.and_table(2, (0, 1))]
        cs = {(0, f): FinitePopulationCS(ledger.N_rem(0)) for f in range(len(basis))}
        for f in range(len(basis)):
            for g in ledger.prefix_g[0]:
                cs[(0, f)].push(basis[f][states.encode_state(tuple(int(v) for v in g))])
        delta = 0.05 / len(cs)
        cs_int = {(0, f): cs[(0, f)].interval(delta) for f in range(len(basis))}
        poly = build_polytope(ledger, cs_int, basis)

        # 整数穷举：所有 Σ k = N 的整数向量中满足全部约束者
        N = ledger.N_h[0]
        o = ledger.obs[0]
        u = ledger.unseen(0)
        feas = []
        for k0 in range(N + 1):
            for k1 in range(N - k0 + 1):
                for k2 in range(N - k0 - k1 + 1):
                    k3 = N - k0 - k1 - k2
                    k = np.array([k0, k1, k2, k3], dtype=float)
                    if not np.all(k >= o) or not np.all(k <= o + u):
                        continue
                    good = True
                    for (h, tab, E_f, lo, hi) in poly["cs"]:
                        v = sum(tab[z] * k[z] for z in range(J))
                        if not (E_f + lo - 1e-9 <= v <= E_f + hi + 1e-9):
                            good = False
                            break
                    if good:
                        feas.append(k)
        assert feas, "可行域不应为空"

        # B 的上下界
        minB, maxB = min_max_B(poly, b_table, (0,), J)
        Bs = [sum(b_table[z] * k[z] for z in range(J)) for k in feas]
        assert minB <= min(Bs) + 1e-6, (minB, min(Bs))
        assert maxB >= max(Bs) - 1e-6, (maxB, max(Bs))

        # 比例上下界（LP 是整数问题放松，集合更大 → L 更小、U 更大）
        if min(Bs) > 0:
            L, U = ratio_bounds(poly, a_table, b_table, (0,), J)
            ratios = [sum(a_table[z] * k[z] for z in range(J)) / Bs[i]
                      for i, k in enumerate(feas)]
            assert L <= min(ratios) + 1e-6, (L, min(ratios))
            assert U >= max(ratios) - 1e-6, (U, max(ratios))
            assert abs(L - U) < 1.0 + 1e-6
            # CC 主路径与 λ 对偶二分路径交叉验证
            try:
                Lcc, Ucc = _ratio_bounds_cc(poly, a_table, b_table, (0,), J)
                Lla, Ula = ratio_bounds_lambda(poly, a_table, b_table, (0,), J)
                assert abs(Lcc - Lla) < 1e-6, (Lcc, Lla)
                assert abs(Ucc - Ula) < 1e-6, (Ucc, Ula)
            except FeasibilityError:
                pass   # CC 退化时允许回退；λ 路径单独已验证
    print("  投影 LP vs 整数穷举：10 个种子全部通过（放松只更保守；CC 与 λ 路径交叉一致）")

# ---------- 3) pipeline 冒烟 ----------
def _totals(q, labels, frame):
    A = B = 0
    for i in range(len(frame)):
        if int(frame["stratum"].iloc[i]) in q.scope:
            z = states.encode_state(tuple(int(v) for v in labels[i]))
            A += int(q.tables["a"][z])
            B += int(q.tables["b"][z])
    return A, B

def test_pipeline_smoke():
    N, K, H = 300, 4, 4
    frame, hidden = make_synthetic_world(N, K, H, seed=3)
    snap = Snapshot(frame, "smoke")
    contract = OracleOperator(hidden, name="oracle", K=K)
    workload = make_workload(K, seed=42)[:8]
    for q in workload:
        q.tables = q.build_tables(K)
    full = contract.judge(list(range(N)), None)
    for q in workload:
        A, B = _totals(q, full, frame)
        q.theta_M = A / B if B > 0 else None

    for name, alloc in [("rr", RoundRobinAllocator()), ("wv", WorstVarianceAllocator())]:
        pipe = COHORTRAG(snap, contract, Config(), seed=0)
        results = pipe.run(workload, alloc, budget_total=60)
        assert pipe.calls <= 60, pipe.calls
        valid = {"certified", "certified_true", "certified_false", "undecided",
                 "undefined_denominator", "execution_error"}
        assert all(r.status in valid for r in results.values())
        cov_n = cov_d = 0
        for q in workload:
            r = results[q.qid]
            if r.status == "certified" and q.theta_M is not None:
                cov_d += 1
                if r.L <= q.theta_M <= r.U + 1e-9:
                    cov_n += 1
        if cov_d:
            assert cov_n / cov_d >= 0.5, f"{name} 覆盖率异常 {cov_n}/{cov_d}"
        print(f"  pipeline 冒烟（{name}）：调用 {pipe.calls}、认证覆盖 {cov_n}/{cov_d}")

# ---------- 2b) 分子带元数据条件（r_meta）的投影 ----------
def test_projection_metadata_numerator():
    """分子条件依赖元数据（草案 5.2：r=r(m,g)，原子内恒定）时的投影正确性。

    构造：2 层各 5 条；查询分母 = 原语 p1，分子 = p1 且【仅第 0 层】，
    即 a_{0,z}=p1_z, a_{1,z}=0（等价于 WANDS「Exact/Partial 中评分>=4」）。"""
    J = 2
    b_table = states.and_table(1, (0,))          # 分母：p1
    labels = {}
    for i in range(5):                            # 层 0：3 个正例
        labels[i] = np.array([1 if i < 3 else 0], dtype=np.int8)
    for i in range(5, 10):                        # 层 1：1 个正例
        labels[i] = np.array([1 if i == 5 else 0], dtype=np.int8)
    frame = pd.DataFrame({
        "id": np.arange(10),
        "stratum": np.array([0] * 5 + [1] * 5, dtype=np.int64),
        "text": ["t"] * 10,
    })
    snap = Snapshot(frame, "tiny-meta")
    contract = OracleOperator(labels, name="oracle", K=1)
    a_hz = {0: np.array([0.0, 1.0]), 1: np.array([0.0, 0.0])}   # 分子只算第 0 层
    for sd in range(6):
        ledger = EvidenceLedger(snap, seed=sd, K=1)
        for h in (0, 1):
            ledger.consume_prefix(h, 2, contract)
        basis = [b_table]
        cs = {}
        for h in (0, 1):
            cs[(h, 0)] = FinitePopulationCS(ledger.N_rem(h))
            for g in ledger.prefix_g[h]:
                cs[(h, 0)].push(basis[0][states.encode_state(tuple(int(v) for v in g))])
        cs_int = {(h, 0): cs[(h, 0)].interval(0.05 / 2) for h in (0, 1)}
        poly = build_polytope(ledger, cs_int, basis)

        # 整数穷举（两层各 5 条，组合数 21×21）
        feas = []
        for k0 in range(6):
            for k1 in range(6):
                k = {0: np.array([5 - k0, k0], dtype=float),
                     1: np.array([5 - k1, k1], dtype=float)}
                ok = True
                for h in (0, 1):
                    o = ledger.obs[h]
                    u = ledger.unseen(h)
                    if not np.all(k[h] >= o - 1e-9) or not np.all(k[h] <= o + u + 1e-9):
                        ok = False
                        break
                if not ok:
                    continue
                for (h, tab, E_f, lo, hi) in poly["cs"]:
                    v = sum(tab[z] * k[h][z] for z in range(J))
                    if not (E_f + lo - 1e-9 <= v <= E_f + hi + 1e-9):
                        ok = False
                        break
                if ok:
                    feas.append(k)
        assert feas
        # 分母与分子
        Bs = [float(sum(b_table[z] * k[h][z] for h in (0, 1) for z in range(J))) for k in feas]
        As = [float(sum(a_hz[h][z] * k[h][z] for h in (0, 1) for z in range(J))) for k in feas]
        minB, maxB = min_max_B(poly, b_table, (0, 1), J)
        assert minB <= min(Bs) + 1e-6 and maxB >= max(Bs) - 1e-6
        if min(Bs) > 0:
            L, U = ratio_bounds(poly, a_hz, b_table, (0, 1), J)
            ratios = [As[i] / Bs[i] for i in range(len(feas))]
            assert L <= min(ratios) + 1e-6, (L, min(ratios))
            assert U >= max(ratios) - 1e-6, (U, max(ratios))
            assert -1e-9 <= L <= U <= 1.0 + 1e-9
    print("  分子带元数据条件（r_meta）的投影：6 个种子全部通过")

# ---------- 2c) 语义算子失败 → fail-closed ----------
def test_label_failure_fail_closed():
    """草案 10.1：标签不可恢复失败时必须 fail-closed，不得当负例、不得崩溃。"""
    class BrokenOperator(OracleOperator):
        def judge(self, ids, texts=None):
            raise RuntimeError("模拟模型输出不可解析")

    N, K = 120, 4
    frame, hidden = make_synthetic_world(N, K, 4, seed=5)
    snap = Snapshot(frame, "broken")
    pipe = COHORTRAG(snap, BrokenOperator(hidden, name="broken", K=K), Config(), seed=0)
    results = pipe.run(make_workload(K, seed=42)[:4], RoundRobinAllocator(), budget_total=40)
    statuses = {r.status for r in results.values()}
    assert statuses <= {"execution_error", "undecided"}, statuses
    assert any(r.status == "execution_error" for r in results.values())
    assert pipe.last_error is not None and "标签失败" in pipe.last_error
    assert pipe.calls == 0, "失败前不应计入任何标签调用"
    print("  语义算子失败：正确 fail-closed，未把失败当负例")

# ---------- 2d) _lambda_max_bisect 符号修复与外包围单调性回归测试 ----------
def test_lambda_bisect_and_enclosure_regression():
    """复现并守护 review5 指出的最小反例：
    N=10 完全已知，两状态计数 [3, 7]，分子取状态 0（计数 3），分母取全部（计数 10），真实比例 0.3。
    验证：(1) _lambda_min_bisect 与 _lambda_max_bisect 均精确返回 ~0.3；
          (2) 传入偏窄候选 [0.29, 0.29] 触发 verify_outward_enclosure 修复时，修复结果包含 0.3 且 L <= U；
          (3) 传入倒置候选 [0.35, 0.25] 时，修复后仍严格满足 L <= 0.3 <= U。
    """
    from cohortrag.projection import (
        _lambda_min_bisect,
        _lambda_max_bisect,
        verify_outward_enclosure,
    )
    poly = {
        "N_h": {0: 10},
        "o_hz": {0: np.array([3.0, 7.0], dtype=np.float64)},
        "u_h": {0: 0},
        "cs": [],
    }
    a = np.array([1.0, 0.0], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)

    L_rb, U_rb = ratio_bounds(poly, a, b, (0,), 2)
    assert L_rb <= 0.3 <= U_rb and L_rb <= U_rb, (L_rb, U_rb)

    l_bi = _lambda_min_bisect(poly, a, b, (0,), 2)
    u_bi = _lambda_max_bisect(poly, a, b, (0,), 2)
    assert abs(l_bi - 0.3) < 1e-6, f"_lambda_min_bisect 错误: {l_bi}"
    assert abs(u_bi - 0.3) < 1e-6, f"_lambda_max_bisect 错误: {u_bi}"

    L_fix, U_fix = verify_outward_enclosure(poly, a, b, (0,), 2, 0.29, 0.29)
    assert L_fix <= U_fix, f"修复后上下界倒置: ({L_fix}, {U_fix})"
    assert L_fix <= 0.3 <= U_fix, f"修复区间未覆盖 0.3: ({L_fix}, {U_fix})"
    assert abs(L_fix - 0.29) < 1e-6 and abs(U_fix - 0.30) < 1e-6, (L_fix, U_fix)

    L_inv, U_inv = verify_outward_enclosure(poly, a, b, (0,), 2, 0.35, 0.25)
    assert L_inv <= 0.3 <= U_inv and L_inv <= U_inv, (L_inv, U_inv)
    print("  _lambda_max_bisect 最小反例与区间单调性回归测试：全部通过")


# ---------- 2e) Signed contrast 完整数值域测试（review6 反例：负真值、跨零、±1 边界、负阈值、精确有理数审计） ----------
def test_signed_contrast_full_domain():
    """验证 review6 第 IV.2 节提出的全部 5 类带符号对比数值域测试：
      1) 真值为负且总体全已知：k=[3, 7], a=[1, -1], b=[1, 1] => theta = -0.4
         验证普通 ratio_bounds、Dinkelbach、bisect、verify_outward_enclosure、exact.audit_ratio_interval 全部精确包围 -0.4；
      2) 区间跨零（部分未见样本）：[-0.6, +0.2] 精确一致；
      3) 正负极端边界（全 -1.0 与全 +1.0）；
      4) 负阈值判断（theta > -0.6 为 true，theta <= -0.2 为 false，含 pipeline 端到端与 audit_exact=True）；
      5) 超出 [-1, 1] 的一般加权条件对比（如 a=[-2, 3], b=[1, 1]）。
    """
    from cohortrag.projection import (
        _lambda_min_bisect,
        _lambda_max_bisect,
        threshold_margins,
        verify_outward_enclosure,
    )
    from cohortrag.exact import audit_ratio_interval, audit_threshold
    from cohortrag.workload import Query

    # 1) 全已知单点负真值 k=[3, 7], theta = -0.4
    poly_neg = {
        "N_h": {0: 10},
        "o_hz": {0: np.array([3.0, 7.0], dtype=np.float64)},
        "u_h": {0: 0},
        "cs": [],
    }
    a = np.array([1.0, -1.0], dtype=np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    a_hz = {0: a}

    L1, U1 = ratio_bounds(poly_neg, a, b, (0,), 2)
    L2, U2 = ratio_bounds(poly_neg, a, b, (0,), 2, verify_enclosure=True)
    L3, U3 = ratio_bounds_lambda(poly_neg, a, b, (0,), 2)
    Lb = _lambda_min_bisect(poly_neg, a, b, (0,), 2)
    Ub = _lambda_max_bisect(poly_neg, a, b, (0,), 2)
    La, Ua, adj = audit_ratio_interval(poly_neg, a_hz, b, (0,), 2, L1, U1)

    for tag, (lo, hi) in [("ratio_bounds", (L1, U1)), ("enclosure", (L2, U2)),
                          ("dinkelbach", (L3, U3)), ("bisect", (Lb, Ub)),
                          ("exact_audit", (float(La), float(Ua)))]:
        assert lo <= -0.4 + 1e-12 and hi >= -0.4 - 1e-12, f"{tag} 未覆盖 -0.4: {(lo, hi)}"
        assert abs(lo - (-0.4)) < 1e-6 and abs(hi - (-0.4)) < 1e-6, f"{tag} 不紧致: {(lo, hi)}"

    # 故意传入错误正区间 [0.0, 0.0] 触发修复，必须拉回到包含 -0.4 的有效区间
    L_rep, U_rep = verify_outward_enclosure(poly_neg, a, b, (0,), 2, 0.0, 0.0)
    assert L_rep <= -0.4 <= U_rep <= 0.0 + 1e-8, (L_rep, U_rep)
    La_rep, Ua_rep, _ = audit_ratio_interval(poly_neg, a_hz, b, (0,), 2, 0.0, 0.0)
    assert float(La_rep) <= -0.4 <= float(Ua_rep), (La_rep, Ua_rep)

    # 2) 区间跨零：N=10, 已见 [2, 4], 未见 u=4 => k0 in [2, 6], theta = (2*k0 - 10)/10 in [-0.6, +0.2]
    poly_cross = {
        "N_h": {0: 10},
        "o_hz": {0: np.array([2.0, 4.0], dtype=np.float64)},
        "u_h": {0: 4},
        "cs": [],
    }
    Lc, Uc = ratio_bounds(poly_cross, a, b, (0,), 2, verify_enclosure=True)
    Lac, Uac, _ = audit_ratio_interval(poly_cross, a_hz, b, (0,), 2, Lc, Uc)
    assert abs(Lc - (-0.6)) < 1e-6 and abs(Uc - 0.2) < 1e-6, (Lc, Uc)
    assert float(Lac) <= -0.6 and float(Uac) >= 0.2, (Lac, Uac)

    # 3) 正负边界：全 -1.0 与全 +1.0
    for obs_vec, target in [([0.0, 10.0], -1.0), ([10.0, 0.0], 1.0)]:
        p_edge = {"N_h": {0: 10}, "o_hz": {0: np.array(obs_vec)}, "u_h": {0: 0}, "cs": []}
        Le, Ue = ratio_bounds(p_edge, a, b, (0,), 2, verify_enclosure=True)
        Lae, Uae, _ = audit_ratio_interval(p_edge, a_hz, b, (0,), 2, Le, Ue)
        assert Le <= target <= Ue and abs(Ue - Le) < 1e-6, (target, Le, Ue)
        assert float(Lae) <= target <= float(Uae), (target, Lae, Uae)

    # 4) 负阈值判断（theta = -0.4 vs tau = -0.6 [true] and tau = -0.2 [false]）与 Pipeline 端到端
    m_gt, _ = threshold_margins(poly_neg, a, b, -0.6, (0,), 2)
    _, M_le = threshold_margins(poly_neg, a, b, -0.2, (0,), 2)
    assert m_gt > 1.99 and M_le < -1.99, (m_gt, M_le)
    assert audit_threshold(poly_neg, a_hz, b, -0.6, (0,), 2, "true") is True
    assert audit_threshold(poly_neg, a_hz, b, -0.2, (0,), 2, "false") is True

    # Pipeline end-to-end with audit_exact=True on negative contrast & negative thresholds
    frame = pd.DataFrame({"id": np.arange(10), "stratum": np.zeros(10, dtype=np.int64), "text": ["x"] * 10})
    snap = Snapshot(frame, "neg-e2e")
    labs = {i: np.array([1, 0] if i < 3 else [0, 1], dtype=np.int8) for i in range(10)}
    op = OracleOperator(labs, name="neg-op", K=2)
    t0 = states.and_table(2, (0,))
    t1 = states.and_table(2, (1,))
    ca = t0.astype(np.float64) - t1.astype(np.float64)
    qs = [
        Query(1, "signed_val", (0,), (), (), kind="value", eps=0.05, custom_a=ca, extra_basis=[t0, t1]),
        Query(2, "neg_thr_true", (0,), (), (), kind="threshold", tau=-0.6, custom_a=ca, extra_basis=[t0, t1]),
        Query(3, "neg_thr_false", (0,), (), (), kind="threshold", tau=-0.2, custom_a=ca, extra_basis=[t0, t1]),
    ]
    for exact_mode in (False, True):
        pipe = COHORTRAG(snap, op, Config(alpha=0.05, audit_exact=exact_mode), seed=0, cache_labels=labs)
        res = pipe.run(qs, RoundRobinAllocator(), budget_total=10)
        assert res[1].status == "certified" and res[1].L <= -0.4 <= res[1].U, res[1]
        assert res[2].status == "certified_true", res[2]
        assert res[3].status == "certified_false", res[3]

    # 5) 一般加权系数 [-2, 3]：k=[3, 7] => (-6 + 21)/10 = +1.5 > 1.0!
    a_gen = np.array([-2.0, 3.0], dtype=np.float64)
    Lg, Ug = ratio_bounds(poly_neg, a_gen, b, (0,), 2, verify_enclosure=True)
    Lag, Uag, _ = audit_ratio_interval(poly_neg, {0: a_gen}, b, (0,), 2, Lg, Ug)
    assert Lg <= 1.5 <= Ug and abs(Ug - Lg) < 1e-6, (Lg, Ug)
    assert float(Lag) <= 1.5 <= float(Uag), (Lag, Uag)
    print("  signed contrast 完整数值域测试（负值、跨零、±1、负阈值、一般权重、精确审计）：全部通过")

def test_audit_threshold_keeps_exact_third():
    """A count of 1 out of 3 is not strictly above 1/3.

    Rounding the threshold to nine decimal digits makes the contrast positive
    and would certify a strict exceedance. The rational threshold must not.
    """
    from fractions import Fraction
    from cohortrag.exact import audit_threshold
    poly = {
        "N_h": {0: 3},
        "o_hz": {0: np.array([1.0, 2.0])},
        "u_h": {0: 0},
        "cs": [],
    }
    a = {0: np.array([1.0, 0.0])}
    b = np.array([1.0, 1.0])
    assert audit_threshold(poly, a, b, Fraction(1, 3), (0,), 2, "true") is False
    print("  阈值 1/3 保持精确，未按九位小数放宽")


def test_dinkelbach_does_not_stop_on_downward_step():
    """A signed multi-stratum ratio whose first Dinkelbach update can move downward.

    The old stopping rule treated any negative step as convergence and returned an
    inward lower endpoint. The repaired iterator must stay within the outward
    bisection bracket, and a certificate-side enclosure must still cover it.
    """
    from cohortrag.projection import (
        _lambda_min,
        _lambda_max,
        _lambda_min_bisect,
        _lambda_max_bisect,
        verify_outward_enclosure,
    )
    rng = np.random.default_rng(7)
    H, J = 5, 8
    poly = {"N_h": {}, "o_hz": {}, "u_h": {}, "cs": []}
    for h in range(H):
        counts = rng.integers(0, 4, size=J).astype(np.float64)
        counts[0] += 2
        poly["N_h"][h] = int(counts.sum() + 6)
        poly["o_hz"][h] = counts
        poly["u_h"][h] = 6
    a = rng.integers(-2, 3, size=J).astype(np.float64)
    a[0] = 2.0
    a[1] = -2.0
    b = np.ones(J, dtype=np.float64)
    scope = tuple(range(H))
    L = _lambda_min(poly, a, b, scope, J)
    U = _lambda_max(poly, a, b, scope, J)
    Lb = _lambda_min_bisect(poly, a, b, scope, J)
    Ub = _lambda_max_bisect(poly, a, b, scope, J)
    assert L <= U, (L, U)
    assert Lb - 1e-6 <= L <= Lb + 5e-4, (L, Lb)
    assert Ub - 5e-4 <= U <= Ub + 1e-6, (U, Ub)
    L2, U2 = verify_outward_enclosure(poly, a, b, scope, J, L, U)
    assert L2 <= L + 1e-8 and U2 >= U - 1e-8 and L2 <= U2, (L2, U2, L, U)
    print("  Dinkelbach 多步带符号对比：下界不再把负向更新当成收敛")


if __name__ == "__main__":
    for t in [test_cs_class_vs_bruteforce, test_projection_vs_integer_enumeration,
              test_projection_metadata_numerator, test_label_failure_fail_closed,
              test_lambda_bisect_and_enclosure_regression,
              test_signed_contrast_full_domain,
              test_dinkelbach_does_not_stop_on_downward_step,
              test_audit_threshold_keeps_exact_third,
              test_pipeline_smoke]:
        t()
    print("test_core 全部通过")
