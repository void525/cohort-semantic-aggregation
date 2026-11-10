# -*- coding: utf-8 -*-
"""查询预算分配器（草案 9.3，2026-11 版）。

分配器只决定「下一批语义调用花在哪一层」，统计有效性不依赖它们（Theorem 1 对任意自适应
采样顺序成立）。这里提供四个策略，用于回答「哪种调度真正提高认证吞吐」：

- RoundRobinAllocator        ：轮询所有未抽干的层（B6，强默认）。
- SmallStratumFirstAllocator ：各层先预热到 n_min，再按层大小升序逐层抽干（小层优先）。
- WorstVarianceAllocator     ：方差代理的贪心水填充（M_var）。
- CertificateGuidedAllocator ：以「未决查询的总归一化证书松弛」为目标（M_cert，无自由参数）。

关于「整层耗尽」候选的一条数学事实（评审指出，已在论文中修正表述）
--------------------------------------------------------------------
对方差代理 V(n) = D (1/n − 1/N)：
    单步增益  g_step  = D / (n (n+1))
    整层耗尽均摊增益 g_drain = D / (n N)
    g_drain / g_step = (n+1)/N <= 1，
所以在方差代理下，耗尽候选始终被同层单步候选弱支配——它并不提供额外的「耗尽小层」激励。
本文的证书松弛分配器改用置信区间半宽 ∝ sqrt((N-n)/(nN)) 作为代理 w(n)：
对这个代理，整层耗尽的平均单位增益 w(n)/(N-n) 与单步边际增益 -w'(n) 之比为 2n/N（连续近似；
已数值验证），因此「耗尽」候选恰在 n > N/2 的层上胜出——最后几个样本最值钱，因为 n = N 时宽度精确为零。
这是离散的证书效应，方差代理看不见。
（更正：w(n) 本身由凸转凹的拐点在 n = 3N/4，而不是 N/2；N/2 来自「耗尽 vs 单步」的增益比，不是凹性。）
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class QueryBudgetState:
    qid: int
    eta: float                      # 归一化精度需求
    c_h: dict = field(default_factory=dict)   # h -> N_rem² * S̃²_qh


@dataclass
class BudgetContext:
    N_rem: dict
    n: dict
    cost_per_pos: dict
    queries: list


class RoundRobinAllocator:
    """B6：把预算轮转到所有仍有剩余位置的层。"""

    def allocate(self, ctx: BudgetContext, C: float):
        hs = [h for h in ctx.N_rem if ctx.n.get(h, 0) < ctx.N_rem[h]]
        if not hs:
            return {}
        alloc = {h: 0 for h in hs}
        cost = 0.0
        i = 0
        while cost < C and hs:
            h = hs[i % len(hs)]
            if ctx.n[h] + alloc[h] >= ctx.N_rem[h]:
                hs.remove(h)
                continue
            alloc[h] += 1
            cost += ctx.cost_per_pos[h]
            i += 1
        return {h: v for h, v in alloc.items() if v > 0}


class ProportionalAllocator:
    """按层大小等比例：始终给「已抽样比例 n_h/N_h 最低」的层再加一个样本。

    这是论文早期版本里 M 的真实面目：方差下限单位有误时，方差代理退化为与数据无关的常数，
    水填充等价于让所有层保持同一抽样比例。它在含群体子查询的工作负载上输给轮询，
    因为群体查询只依赖较小的层，而等比例分配对小层的抽样比例太低。
    """

    def allocate(self, ctx: BudgetContext, C: float):
        n = {h: ctx.n.get(h, 0) for h in ctx.N_rem}
        alloc = {h: 0 for h in ctx.N_rem}
        cost = 0.0
        while cost < C:
            cand = [h for h in ctx.N_rem if n[h] < ctx.N_rem[h]]
            if not cand:
                break
            h = min(cand, key=lambda x: (n[x] / max(ctx.N_rem[x], 1), x))
            n[h] += 1
            alloc[h] += 1
            cost += ctx.cost_per_pos.get(h, 1.0)
        return {h: v for h, v in alloc.items() if v > 0}


class SmallStratumFirstAllocator:
    """小层优先：先把每层预热到 n_min，再按层大小升序逐层抽干。"""

    def __init__(self, n_min: int = 32):
        self.n_min = int(n_min)

    def allocate(self, ctx: BudgetContext, C: float):
        order = sorted((h for h in ctx.N_rem if ctx.n.get(h, 0) < ctx.N_rem[h]),
                       key=lambda h: ctx.N_rem[h])
        if not order:
            return {}
        alloc = {h: 0 for h in order}
        room = float(C)
        # 预热：每层至少 n_min（受层大小限制）
        for h in order:
            need = max(0, min(self.n_min, ctx.N_rem[h]) - ctx.n.get(h, 0))
            cph = max(ctx.cost_per_pos.get(h, 1.0), 1e-9)
            take = int(min(need, room // cph))
            alloc[h] += take
            room -= take * cph
        # 抽干：由小到大
        for h in order:
            if room <= 0:
                break
            rem = ctx.N_rem[h] - ctx.n.get(h, 0) - alloc[h]
            cph = max(ctx.cost_per_pos.get(h, 1.0), 1e-9)
            take = int(min(rem, room // cph))
            alloc[h] += take
            room -= take * cph
        return {h: v for h, v in alloc.items() if v > 0}


class DrainAwareRoundRobinAllocator:
    """轮询 + 过半即耗尽（M_drain，无自由参数）。

    对宽度代理 w(n) = sqrt((N−n)/(nN))，整层耗尽的平均单位增益与单步边际增益之比为 2n/N，
    在 n = N/2 处反转：一个层一旦被抽到一半以上，继续「整层耗尽」的单位成本增益就超过逐个加样本——
    最后几个样本最值钱，因为 n = N 时该层的未知计数与置信区间精确坍缩为零。
    （注意 w(n) 自身的拐点在 3N/4；N/2 来自增益比，不来自凹性。）据此：
      1) 还没抽到一半的层：轮询（保持随时可用的认证吞吐，小预算下与 B6 一致）；
      2) 已抽到一半以上的层：按「剩余量由小到大」先耗尽。
    阈值 1/2 由宽度函数的拐点决定，不是调出来的参数。
    """

    def allocate(self, ctx: BudgetContext, C: float):
        n = {h: ctx.n.get(h, 0) for h in ctx.N_rem}
        alloc = {h: 0 for h in ctx.N_rem}
        active = [h for h in ctx.N_rem if n[h] < ctx.N_rem[h]]
        if not active:
            return {}
        cost = 0.0
        rr = 0
        while cost < C and active:
            half = [h for h in active if 2 * n[h] >= ctx.N_rem[h]]
            if half:
                h = min(half, key=lambda x: ctx.N_rem[x] - n[x])
                cph = max(ctx.cost_per_pos.get(h, 1.0), 1e-9)
                take = int(min(ctx.N_rem[h] - n[h], max(1, (C - cost) // cph)))
                n[h] += take
                alloc[h] += take
                cost += take * cph
            else:
                h = active[rr % len(active)]
                n[h] += 1
                alloc[h] += 1
                cost += ctx.cost_per_pos.get(h, 1.0)
                rr += 1
            active = [x for x in active if n[x] < ctx.N_rem[x]]
        return {h: v for h, v in alloc.items() if v > 0}


class WorstVarianceAllocator:
    """M_var：硬预算约束下的『边际方差下降 / 单位成本』贪心水填充。

    对凸的方差代理，整层耗尽的均摊增益恒不超过同层单步增益（见模块注释），
    因此这里只评估单步动作。未采样的层（n=0）边际增益视为无穷，必先获得 1 个样本。
    """

    @staticmethod
    def _gain(q: QueryBudgetState, h, n_h, N_h):
        c = q.c_h.get(h, 0.0)
        if c <= 0 or n_h >= N_h:
            return 0.0
        if n_h <= 0:
            return np.inf
        return c / (max(q.eta, 1e-6) ** 2 * n_h * (n_h + 1))

    def allocate(self, ctx: BudgetContext, C: float):
        if not ctx.queries or C <= 0:
            return {}
        n = {h: ctx.n.get(h, 0) for h in ctx.N_rem}
        alloc = {h: 0 for h in ctx.N_rem}
        cost = 0.0
        while True:
            best = None
            for h in ctx.N_rem:
                N_h = ctx.N_rem[h]
                if n[h] >= N_h:
                    continue
                cph = max(ctx.cost_per_pos.get(h, 1.0), 1e-9)
                g1 = max((self._gain(q, h, n[h], N_h) for q in ctx.queries), default=0.0)
                if g1 > 0:
                    r = g1 / cph
                    if best is None or r > best[0]:
                        best = (r, h, 1)
            if best is None:
                break
            _, h, want = best
            cph = ctx.cost_per_pos.get(h, 1.0)
            room = C - cost
            step = int(min(want, max(1, int(room // max(cph, 1e-9)))))
            step = min(step, ctx.N_rem[h] - n[h])
            if step <= 0 or step * cph > room + 1e-12:
                break
            n[h] += step
            alloc[h] += step
            cost += step * cph
        return {h: v for h, v in alloc.items() if v > 0}


class CertificateGuidedAllocator:
    """M_cert：最小化未决查询的『总归一化证书松弛』（无自由参数）。

    证书的半宽主要来自各层置信区间的宽度，近似为
        slack_q = Σ_h  sqrt(c_{q,h}) · sqrt((N_h − n_h) / (n_h N_h))  /  η_q,
    其中 c_{q,h} = N_h² S̃²_{q,h}（与方差代理同一来源），η_q 是该查询距认证条件的差距
    （数值/差异查询取 ε_q；阈值查询取 |θ̂_q − τ_q|）。把所有未决查询的 slack 相加，
    每步选「单位成本下 slack 下降最多」的动作；候选动作 = {+1, +B, 整层耗尽}。

    与方差分配器的区别（也是它在认证数上更好的原因）：
      1) 目标是置信区间宽度（∝ sqrt(方差)）而不是方差——对不同层的相对价值排序不同；
      2) 对所有未决查询求和而不是只看最差者——共享的层对多个查询同时有价值；
      3) 对宽度代理，耗尽的平均增益在 n > N/2 的层上超过单步边际增益（比值 2n/N），
         耗尽候选会在小层上真正胜出，这正是「抽干小层使其不确定性精确归零」的离散证书效应。
    它是一个启发式；不声称最优，统计有效性与其预测是否准确无关。
    """

    def __init__(self, batch: int = 64):
        self.batch = int(batch)

    @staticmethod
    def _width(N, n):
        """FPC 修正的 √ 律宽度代理；n=N 时精确为 0。"""
        if n <= 0:
            return np.inf
        return np.sqrt(max(N - n, 0.0) / (n * N))

    def allocate(self, ctx: BudgetContext, C: float):
        if not ctx.queries or C <= 0:
            return {}
        # 每层的聚合敏感度：所有未决查询 sqrt(c)/η 之和
        w = {}
        for h in ctx.N_rem:
            w[h] = float(sum(np.sqrt(max(q.c_h.get(h, 0.0), 0.0)) / max(q.eta, 1e-2)
                             for q in ctx.queries))
        n = {h: ctx.n.get(h, 0) for h in ctx.N_rem}
        alloc = {h: 0 for h in ctx.N_rem}
        cost = 0.0
        while True:
            best = None                      # (score, h, step)
            for h in ctx.N_rem:
                N_h = ctx.N_rem[h]
                if n[h] >= N_h or w[h] <= 0:
                    continue
                cph = max(ctx.cost_per_pos.get(h, 1.0), 1e-9)
                if n[h] < 2:                 # 先看一眼：未采样的层必须先拿到起码的样本
                    best = (np.inf, h, 1)
                    break
                rem = N_h - n[h]
                w0 = self._width(N_h, n[h])
                for step in {1, min(self.batch, rem), rem}:
                    gain = w[h] * (w0 - self._width(N_h, n[h] + step)) / (step * cph)
                    if best is None or gain > best[0]:
                        best = (gain, h, step)
            if best is None:
                break
            _, h, want = best
            cph = ctx.cost_per_pos.get(h, 1.0)
            room = C - cost
            step = int(min(want, max(1, int(room // max(cph, 1e-9)))))
            step = min(step, ctx.N_rem[h] - n[h])
            if step <= 0 or step * cph > room + 1e-12:
                break
            n[h] += step
            alloc[h] += step
            cost += step * cph
        return {h: v for h, v in alloc.items() if v > 0}
