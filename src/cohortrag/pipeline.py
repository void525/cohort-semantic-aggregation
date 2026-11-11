# -*- coding: utf-8 -*-
"""COHORT 主执行器（草案 6、8.6、10.1 的伪代码的忠实实现，2026-11 版）。

主循环：
  编译受限查询 → 冻结阶段（E/排列/统计基）→ CS 初始化 →
  循环 { 联合可行域 → 区间投影 → 分类（certified/undecided/undefined）→
         全部解决或预算耗尽则输出；否则 分配预算 → 消费前缀 → 更新账本/CS }。

交互会话：run(..., feed=callback) 允许在同一阶段内追加新查询（闭包内的追问）；
统计基只在阶段开始时按「预声明闭包」(declared) 编译一次，追问不增加风险支出。

LLM 只做有界语义判断；比例估计、抽样合法性、数值检验全部由符号计算完成。
"""
from dataclasses import dataclass
import numpy as np
from . import states
from .cs import FinitePopulationCS
from .ledger import EvidenceLedger
from .projection import (build_polytope, min_max_B, ratio_bounds,
                         threshold_margins, verify_outward_enclosure, FeasibilityError)
from .budget import BudgetContext, QueryBudgetState, RoundRobinAllocator
from .states import compile_basis


@dataclass
class Config:
    alpha: float = 0.05
    max_new_calls: int = 64          # 每轮新增语义调用上限
    plan_interval: int = 128         # 每 128 次调用重新规划一次
    B_stab: float = 10.0             # 优化器数值稳定下限，不替代真实分母
    eps_floor: float = 0.02
    var_floor: float = 0.02 ** 2
    legacy_var_floor: bool = False   # True 复现旧版（单位有误的）方差下限
    max_rounds: int = 4000
    compile_basis: bool = True       # False：无统计基编译消融
    cs_pad: int = 1                  # 置信序列端点向外各放宽的计数（吸收 lgamma 浮点误差）
    progressive_plan: bool = True    # 规划间隔随进度变粗：<10%N 用 B，<30%N 用 2B，之后 4B
    audit_exact: bool = False        # True：认证瞬间用精确有理数 LP 复核结论（Theorem 1 实现层）


@dataclass
class QueryResult:
    qid: int
    name: str
    kind: str
    status: str = "undecided"
    L: float = None
    U: float = None
    point: float = None
    minB: float = None
    maxB: float = None
    theta_hat: float = None
    theta_M: float = None            # 评测器填写：运行语义全量物化值
    theta_gold: float = None         # 评测器填写：独立真实标签值
    calls_at_cert: int = None        # 认证时刻的累计新增调用数


class COHORTRAG:
    def __init__(self, snapshot, contract, cfg=None, seed=0, cache_labels=None,
                 e_as_random=False):
        self.snapshot = snapshot
        self.contract = contract
        self.cfg = cfg if cfg is not None else Config()
        self.seed = seed
        self.K = contract.K
        self.J = 1 << contract.K
        self.ledger = EvidenceLedger(snapshot, seed, contract.K, cache_labels,
                                     e_as_random)
        self.calls = 0
        self.risk_ledger = {}
        self.rounds_used = 0
        self.last_error = None
        self.audit_log = []          # [(kind, adjusted_steps)]，审计是否改动了端点
        self.trace = []              # [(累计调用数, {qid: (status, L, U)})]：每个规划检查点一条
        self._N_total = sum(self.ledger.N_h.values())

    def plan_interval(self):
        """下一次规划前最多消耗的新增调用数。"""
        base = self.cfg.plan_interval
        if not self.cfg.progressive_plan:
            return base
        frac = self.calls / max(1, self._N_total)
        return base if frac < 0.10 else (2 * base if frac < 0.30 else 4 * base)

    # ---------- 内部工具 ----------
    def _acc_estimate(self, q, acc, AE, BE):
        """θ̂ 与 B̂（仅用于预算分配，不用于认证）。"""
        Ah, Bh = AE[q.qid], BE[q.qid]
        for h in q.scope:
            s = acc[(q.qid, h)]
            if s[2] > 0:
                N_rem = self.ledger.N_rem(h)
                Ah += N_rem * s[0] / s[2]
                Bh += N_rem * s[1] / s[2]
        if q.kind == "difference":
            Ah2, Bh2 = AE[(q.qid, "s2")], BE[(q.qid, "s2")]
            for h in q.scope2:
                s = acc[(q.qid, ("s2", h))]
                if s[2] > 0:
                    N_rem = self.ledger.N_rem(h)
                    Ah2 += N_rem * s[0] / s[2]
                    Bh2 += N_rem * s[1] / s[2]
            thA = Ah / Bh if Bh > 0 else 0.5
            thB = Ah2 / Bh2 if Bh2 > 0 else 0.5
            return {"thA": thA, "thB": thB, "BA": Bh, "BB": Bh2}
        return {"th": Ah / Bh if Bh > 0 else 0.5, "B": Bh}

    def _side_bounds(self, poly, scope, a_hz, b_table):
        memo = poly.setdefault("_memo_bounds", {})
        key = (
            tuple(scope),
            tuple((h, tuple(float(x) for x in a_hz[h])) for h in scope),
            tuple(float(x) for x in b_table),
            id(ratio_bounds),
        )
        if key in memo:
            return memo[key]
        minB, maxB = min_max_B(poly, b_table, scope, self.J)
        if maxB <= 1e-9 or minB <= 1e-9:
            res = (minB, maxB, None, None)
        else:
            L, U = ratio_bounds(poly, a_hz, b_table, scope, self.J)
            res = (minB, maxB, L, U)
        memo[key] = res
        return res

    def _audit_side(self, poly, scope, a_hz, b_table, L, U):
        from .exact import audit_ratio_interval
        L2, U2, adj = audit_ratio_interval(poly, a_hz, b_table, scope, self.J, L, U)
        self.audit_log.append(("ratio", int(adj), float(L), float(L2), float(U), float(U2)))
        return float(L2), float(U2)

    def _classify(self, q, poly):
        """停止与回答规则（草案 8.6）。audit_exact=True 时，认证结论经精确有理数复核。"""
        cfg = self.cfg
        if q.kind == "difference":
            minB1, maxB1, L1, U1 = self._side_bounds(poly, q.scope, q.a_hz, q.tables["b"])
            minB2, maxB2, L2, U2 = self._side_bounds(poly, q.scope2, q.a_hz2, q.tables["b2"])
            if maxB1 <= 1e-9 or maxB2 <= 1e-9:
                return ("undefined", None, None, min(minB1, minB2), max(maxB1, maxB2))
            if minB1 <= 1e-9 or minB2 <= 1e-9:
                return ("insufficient", None, None, min(minB1, minB2), max(maxB1, maxB2))
            if L1 > U1 or L2 > U2:
                raise FeasibilityError(f"差值查询子区间倒置：({L1}, {U1}), ({L2}, {U2})")
            dL, dU = L1 - U2, U1 - L2
            if dL > dU:
                raise FeasibilityError(f"差值区间倒置：dL={dL} > dU={dU}")
            if 0.0 <= dU - dL <= 2 * q.eps:
                if cfg.audit_exact:
                    a1, a2 = self._audit_side(poly, q.scope, q.a_hz, q.tables["b"], L1, U1)
                    b1, b2 = self._audit_side(poly, q.scope2, q.a_hz2, q.tables["b2"], L2, U2)
                else:
                    a1, a2 = verify_outward_enclosure(poly, q.a_hz, q.tables["b"], q.scope, self.J, L1, U1)
                    b1, b2 = verify_outward_enclosure(poly, q.a_hz2, q.tables["b2"], q.scope2, self.J, L2, U2)
                dL, dU = a1 - b2, a2 - b1
                if dL > dU:
                    raise FeasibilityError(f"外包围复核后差值区间倒置：dL={dL} > dU={dU}")
                if 0.0 <= dU - dL <= 2 * q.eps:
                    return ("certified", (dL + dU) / 2, (dL, dU), None, None)
            return ("undecided", None, (dL, dU), None, None)
        minB, maxB, L, U = self._side_bounds(poly, q.scope, q.a_hz, q.tables["b"])
        if maxB <= 1e-9:
            return ("undefined", None, None, minB, maxB)
        if minB <= 1e-9:
            return ("insufficient", None, None, minB, maxB)
        if L > U:
            raise FeasibilityError(f"比例区间倒置：L={L} > U={U}")
        if q.kind == "value":
            if 0.0 <= U - L <= 2 * q.eps:
                if cfg.audit_exact:
                    L, U = self._audit_side(poly, q.scope, q.a_hz, q.tables["b"], L, U)
                else:
                    L, U = verify_outward_enclosure(poly, q.a_hz, q.tables["b"], q.scope,
                                                    self.J, L, U)
                if L > U:
                    raise FeasibilityError(f"外包围复核后比例区间倒置：L={L} > U={U}")
                if 0.0 <= U - L <= 2 * q.eps:
                    return ("certified", (L + U) / 2, (L, U), minB, maxB)
            return ("undecided", None, (L, U), minB, maxB)
        if q.kind == "threshold":
            if L > q.tau:
                if cfg.audit_exact:
                    from .exact import audit_threshold
                    ok = audit_threshold(poly, q.a_hz, q.tables["b"], q.tau, q.scope, self.J, "true")
                    self.audit_log.append(("threshold_true", int(not ok)))
                else:
                    m_q, _ = threshold_margins(poly, q.a_hz, q.tables["b"], q.tau, q.scope, self.J)
                    ok = m_q > 1e-9
                if ok:
                    return ("certified_true", None, (L, U), minB, maxB)
            if U <= q.tau:
                if cfg.audit_exact:
                    from .exact import audit_threshold
                    ok = audit_threshold(poly, q.a_hz, q.tables["b"], q.tau, q.scope, self.J, "false")
                    self.audit_log.append(("threshold_false", int(not ok)))
                else:
                    _, M_q = threshold_margins(poly, q.a_hz, q.tables["b"], q.tau, q.scope, self.J)
                    ok = M_q <= -1e-9
                if ok:
                    return ("certified_false", None, (L, U), minB, maxB)
            return ("undecided", None, (L, U), minB, maxB)
        raise ValueError(q.kind)

    def _budget_context(self, queries, unresolved, acc, AE, BE):
        cfg = self.cfg
        # 各层前缀的状态频数由账本增量维护（O(J) 读取，不再重扫前缀）
        z_counts = {h: self.ledger.prefix_z[h].astype(np.float64)
                    for h in self.ledger.strata}

        qstates = []
        for q in unresolved:
            est = self._acc_estimate(q, acc, AE, BE)
            if q.kind == "difference":
                eta = q.eps
            elif q.kind == "value":
                eta = q.eps
            else:
                eta = max(abs(est["th"] - q.tau), cfg.eps_floor)
            c_h = {}
            for h in self.ledger.strata:
                N_rem = self.ledger.N_rem(h)
                if N_rem <= 0:
                    continue
                cnt = z_counts[h]
                n_tot = float(cnt.sum())
                if n_tot < 2.0:
                    var = cfg.var_floor if cfg.legacy_var_floor else 0.0
                else:
                    vals = np.zeros(self.J, dtype=np.float64)
                    for z in range(self.J):
                        if q.kind == "difference":
                            tA = ((q.a_hz[h][z] - est["thA"] * q.tables["b"][z])
                                  / max(est["BA"], cfg.B_stab)) if h in q.scope else 0.0
                            tB = ((q.a_hz2[h][z] - est["thB"] * q.tables["b2"][z])
                                  / max(est["BB"], cfg.B_stab)) if h in q.scope2 else 0.0
                            vals[z] = tA - tB
                        else:
                            if h in q.scope:
                                vals[z] = ((q.a_hz[h][z] - est["th"] * q.tables["b"][z])
                                           / max(est["B"], cfg.B_stab))
                            else:
                                vals[z] = 0.0
                    mean_v = float(np.dot(cnt, vals) / n_tot)
                    var = float(np.dot(cnt, (vals - mean_v) ** 2) / n_tot)
                if cfg.legacy_var_floor:
                    # 旧口径：下限直接作用在已除以 B̂ 的残差方差上；B̂ ~ 10^4 时残差方差 ~ 10^-10
                    # 远低于下限，等价于「按层大小等比例分配」。
                    c_h[h] = N_rem ** 2 * max(var, cfg.var_floor)
                else:
                    # 修正：下限作用在未归一化残差的尺度上，再统一除以 B̂²。
                    if q.kind == "difference":
                        bref = max(min(est["BA"], est["BB"]), cfg.B_stab)
                    else:
                        bref = max(est["B"], cfg.B_stab)
                    c_h[h] = N_rem ** 2 * max(var, cfg.var_floor / (bref ** 2))
            qstates.append(QueryBudgetState(q.qid, eta, c_h))
        ctx = BudgetContext(
            N_rem={h: self.ledger.N_rem(h) for h in self.ledger.strata},
            n={h: self.ledger.n_prefix(h) for h in self.ledger.strata},
            cost_per_pos={h: self.ledger.cost_per_pos(h) for h in self.ledger.strata},
            queries=qstates,
        )
        return ctx

    # ---------- 主流程 ----------
    def run(self, workload, allocator=None, budget_total=None, declared=None, feed=None):
        """workload：当前要回答的查询；declared：预声明闭包（决定统计基，默认等于 workload）；
        feed(pipe, results) -> (新查询列表, 新的总预算上限)：全部解决或预算用尽时被调用，
        返回空列表即结束（用于多轮交互会话）。"""
        cfg = self.cfg
        queries = list(workload)
        declared_q = list(declared) if declared is not None else queries

        def prep(q):
            q.tables = q.build_tables(self.K)
            # 分子系数（可含元数据条件）：a_{h,z} = r_meta[h] · d_z · r_z
            q.a_hz = q.numerator_tables(self.K, q.scope)
            if q.kind == "difference":
                q.a_hz2 = q.numerator_tables(self.K, q.scope2, meta=q.r_meta2,
                                             tables={"a": q.tables["a2"]})

        seen = set()
        for q in declared_q + queries:
            if id(q) not in seen:
                seen.add(id(q))
                prep(q)

        # 统计基编译：预声明闭包的所有分母/分子事件
        tables, names = [], []
        for q in declared_q:
            tables.append(q.tables["d"])
            names.append(f"{q.name}:d")
            if "extra_basis" in q.tables:
                for bi, btab in enumerate(q.tables["extra_basis"]):
                    tables.append(np.asarray(btab, dtype=np.int64))
                    names.append(f"{q.name}:eb{bi}")
            else:
                tables.append(q.tables["a"])
                names.append(f"{q.name}:a")
            if q.kind == "difference":
                tables += [q.tables["d2"], q.tables["a2"]]
                names += [f"{q.name}:d2", f"{q.name}:a2"]
        if cfg.compile_basis:
            basis_tables, basis_names = compile_basis(tables, names)
        else:
            basis_tables, basis_names = list(tables), list(names)

        # CS 初始化：每层剩余总体 × 每个基函数
        cs = {}
        for h in self.ledger.strata:
            if self.ledger.N_rem(h) == 0:
                continue
            for f in range(len(basis_tables)):
                cs[(h, f)] = FinitePopulationCS(self.ledger.N_rem(h))
        n_cs = len(cs)
        delta = cfg.alpha / n_cs if n_cs else None
        self.risk_ledger = {"alpha": cfg.alpha, "n_cs": n_cs, "delta": delta,
                            "basis": basis_names}

        acc, AE, BE = {}, {}, {}

        def init_query_state(q):
            """为（可能中途加入的）查询建立累计量：已有随机前缀 + E 的精确贡献。"""
            for h in self.ledger.strata:
                acc[(q.qid, h)] = [0.0, 0.0, 0]
                if q.kind == "difference":
                    acc[(q.qid, ("s2", h))] = [0.0, 0.0, 0]
            for h in self.ledger.strata:
                cnt = self.ledger.prefix_z[h]
                if cnt.sum() <= 0:
                    continue
                if h in q.scope:
                    s = acc[(q.qid, h)]
                    for z in range(self.J):
                        s[0] += q.a_hz[h][z] * cnt[z]
                        s[1] += q.tables["b"][z] * cnt[z]
                        s[2] += int(cnt[z])
                if q.kind == "difference" and h in q.scope2:
                    s = acc[(q.qid, ("s2", h))]
                    for z in range(self.J):
                        s[0] += q.a_hz2[h][z] * cnt[z]
                        s[1] += q.tables["b2"][z] * cnt[z]
                        s[2] += int(cnt[z])
            ae = be = 0.0
            for h in q.scope:
                for z in range(self.J):
                    ae += q.a_hz[h][z] * self.ledger.E_counts[h][z]
                    be += q.tables["b"][z] * self.ledger.E_counts[h][z]
            AE[q.qid], BE[q.qid] = ae, be
            if q.kind == "difference":
                ae2 = be2 = 0.0
                for h in q.scope2:
                    for z in range(self.J):
                        ae2 += q.a_hz2[h][z] * self.ledger.E_counts[h][z]
                        be2 += q.tables["b2"][z] * self.ledger.E_counts[h][z]
                AE[(q.qid, "s2")], BE[(q.qid, "s2")] = ae2, be2

        def update_acc(h, g):
            z = states.encode_state(tuple(int(v) for v in g))
            for q in queries:
                if h in q.scope:
                    s = acc[(q.qid, h)]
                    s[0] += q.a_hz[h][z]
                    s[1] += q.tables["b"][z]
                    s[2] += 1
                if q.kind == "difference" and h in q.scope2:
                    s = acc[(q.qid, ("s2", h))]
                    s[0] += q.a_hz2[h][z]
                    s[1] += q.tables["b2"][z]
                    s[2] += 1

        # 去除选择来源契约（消融）：历史缓存被当作随机前缀，同等地推入 CS
        if self.ledger.e_as_random:
            for (h, f), c in cs.items():
                table = basis_tables[f]
                for g in self.ledger.prefix_g[h]:
                    z = states.encode_state(tuple(int(v) for v in g))
                    c.push(table[z])
        for q in queries:
            init_query_state(q)

        results = {q.qid: QueryResult(qid=q.qid, name=q.name, kind=q.kind)
                   for q in queries}
        unresolved = list(queries)
        plan_left = {}

        while True:
            self.rounds_used += 1
            if self.rounds_used > cfg.max_rounds:
                break

            if not plan_left:
                # ---- 规划：可行域 → 投影分类 → 预算分配 ----
                cs_int = {}
                bad = False
                for (h, f), c in cs.items():
                    iv = c.interval(delta)
                    if iv is None:
                        bad = True
                        break
                    cs_int[(h, f)] = iv
                if bad:
                    for q in unresolved:
                        results[q.qid].status = "execution_error"
                    break
                poly = build_polytope(self.ledger, cs_int, basis_tables,
                                      self.ledger.e_as_random, pad=cfg.cs_pad)
                self.last_poly = poly

                still = []
                for q in unresolved:
                    r = results[q.qid]
                    try:
                        st, pt, iv, minB, maxB = self._classify(q, poly)
                    except FeasibilityError:
                        r.status = "execution_error"
                        continue
                    r.minB, r.maxB = minB, maxB
                    if iv is not None:
                        r.L, r.U = iv[0], iv[1]
                    if st in ("certified", "certified_true", "certified_false"):
                        r.status = st
                        r.calls_at_cert = self.calls
                        if st == "certified":
                            r.point = pt
                    elif st == "undefined":
                        r.status = "undefined_denominator"
                    else:
                        r.status = "undecided"
                        still.append(q)
                unresolved = still
                self.trace.append((self.calls, {q.qid: (results[q.qid].status, results[q.qid].L,
                                                        results[q.qid].U) for q in queries}))

                out_of_budget = budget_total is not None and self.calls >= budget_total
                if not unresolved or out_of_budget:
                    if feed is None:
                        break
                    new_qs, new_budget = feed(self, results)
                    if not new_qs:
                        break
                    for q in new_qs:
                        prep(q)
                        init_query_state(q)
                        queries.append(q)
                        results[q.qid] = QueryResult(qid=q.qid, name=q.name, kind=q.kind)
                    unresolved = unresolved + list(new_qs)
                    if new_budget is not None:
                        budget_total = new_budget
                    continue                       # 回到规划：先对新查询做一次分类

                ctx = self._budget_context(queries, unresolved, acc, AE, BE)
                C = self.plan_interval()
                if budget_total is not None:
                    C = min(C, budget_total - self.calls)
                if C <= 0:
                    break
                plan = allocator.allocate(ctx, C) if allocator is not None \
                    else RoundRobinAllocator().allocate(ctx, C)
                plan_left = {h: p for h, p in plan.items() if p > 0}
                if not plan_left:
                    break

            # ---- 消费：每轮最多 max_new_calls 个真实调用；位置消费按排列顺序 ----
            C = cfg.max_new_calls
            if budget_total is not None:
                C = min(C, budget_total - self.calls)
            if C <= 0:
                break
            taken = 0
            failed = None
            for h in list(plan_left):
                if taken >= C:
                    break
                want = min(plan_left[h], C - taken)
                before = self.ledger.pos[h]
                try:
                    self.calls += self.ledger.consume_prefix(h, want, self.contract)
                except Exception as exc:
                    # 标签不可恢复失败 → fail-closed：不当负例、不静默删样本、不跳过位置
                    failed = f"层 {h} 语义标签失败：{type(exc).__name__}: {exc}"
                    break
                consumed = self.ledger.pos[h] - before
                if consumed <= 0:
                    del plan_left[h]
                    continue
                taken += consumed
                plan_left[h] -= consumed
                if plan_left[h] <= 0:
                    del plan_left[h]
                new_gs = self.ledger.prefix_g[h][-consumed:]
                for g in new_gs:
                    update_acc(h, g)
                for f in range(len(basis_tables)):
                    table = basis_tables[f]
                    for g in new_gs:
                        z = states.encode_state(tuple(int(v) for v in g))
                        cs[(h, f)].push(table[z])
            if failed is not None:
                self.last_error = failed
                for q in unresolved:
                    if results[q.qid].status == "undecided":
                        results[q.qid].status = "execution_error"
                break

        return results
