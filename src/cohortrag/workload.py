# -*- coding: utf-8 -*-
"""受限查询 AST 与工作负载生成（草案 5.2、5.3、13.2、13.3）。

查询支持四类：
  value     条件比例 θ = A/B，目标：区间宽 <= 2ε 即认证；
  threshold 阈值声明 θ > τ，只有 L > τ 或 U <= τ 才认证；
  difference 两个群体的条件比例差，安全基本区间 [L_A-U_B, U_A-L_B]；
  分母为零 → undefined_denominator；可行域含零分母 → insufficient（继续抽样）。

所有查询都限制在预声明的『原语组合 × 元数据原子并集』闭包内（草案 5.3）。
"""
from dataclasses import dataclass, field
import numpy as np
from . import states

@dataclass
class Query:
    qid: int
    name: str
    scope: tuple                    # 元数据原子（层）并集
    d_idx: tuple                    # 分母原语索引集合（AND）
    r_idx: tuple                    # 分子附加原语索引集合（AND）
    kind: str = "value"             # value | threshold | difference
    eps: float = 0.05               # 数值精度（value/difference）
    tau: float = 0.5                # 阈值（threshold）
    # 比较查询（difference）：第二个群体的 (scope, d_idx, r_idx)
    scope2: tuple = None
    d_idx2: tuple = None
    r_idx2: tuple = None
    # 分子中的元数据条件（草案 5.2：r = r(m, g)，在原子内恒定，草案 5.3）：
    #   r_meta: {层 h: 0/1}。缺省 None 表示分子只依赖语义原语。
    # 例：WANDS「Exact/Partial 商品中评分>=4 的比例」→ 评分>=4 的层为 1，其余为 0。
    r_meta: dict = None
    r_meta2: dict = None            # difference 查询第二侧的元数据分子条件

    def build_tables(self, K):
        d = states.and_table(K, self.d_idx)
        r = states.and_table(K, self.r_idx)
        a = d * r                    # 语义部分：a_z = d_z * r_z（草案 5.2）
        if self.kind == "difference":
            d2 = states.and_table(K, self.d_idx2)
            r2 = states.and_table(K, self.r_idx2)
            a2 = d2 * r2
            return {"d": d, "a": a, "b": d, "d2": d2, "a2": a2, "b2": d2}
        return {"d": d, "a": a, "b": d}

    def numerator_tables(self, K, scope, meta=None, tables=None):
        """返回 {层 h: 分子系数向量 a_h}，把元数据条件折进分子（a_{h,z}=r_meta[h]·a_z）。

        统计基仍只编译【语义】事件（d 与 d∧r），因为元数据乘子是精确已知量，
        不需要统计预算（草案 5.3：原子内元数据指示器恒定）。"""
        tb = self.tables if tables is None else tables
        base = np.asarray(tb["a"], dtype=np.float64)
        rm = self.r_meta if meta is None else meta
        out = {}
        for h in scope:
            mult = 1.0 if rm is None else float(int(rm.get(h, 0)))
            out[h] = mult * base
        return out

def make_workload(K, seed=42):
    """预注册式固定工作负载：阈值只从预声明集合取值，不针对答案挑阈值（草案 13.2）。"""
    qs = []
    def add(qid, name, scope, d, r, kind="value", eps=0.05, tau=None,
            scope2=None, d2=None, r2=None):
        qs.append(Query(qid=qid, name=name, scope=tuple(scope), d_idx=tuple(d),
                        r_idx=tuple(r), kind=kind, eps=eps, tau=tau if tau else 0.5,
                        scope2=None if scope2 is None else tuple(scope2),
                        d_idx2=None if d2 is None else tuple(d2),
                        r_idx2=None if r2 is None else tuple(r2)))

    g1, g2, g3, g4 = 0, 1, 2, 3
    # 精度取预声明集合 ε∈{0.02,0.05,0.10}（草案 8.6）。粗精度问题先认证、
    # 细精度问题需要更多预算——演示里故意混合，展示认证随预算分层的真实行为。
    add(1,  "theta(g2|g1) group0",    [0],       [g1], [g2], eps=0.10)
    add(2,  "theta(g2|g1) group1",    [1],       [g1], [g2], eps=0.10)
    add(3,  "theta(g2|g1) all",       [0, 1, 2, 3], [g1], [g2], eps=0.05)
    add(4,  "theta(g4|g2) all",       [0, 1, 2, 3], [g2], [g4], eps=0.10)
    add(5,  "theta(g3|g1) all",       [0, 1, 2, 3], [g1], [g3], eps=0.10)
    add(6,  "theta(g4|g3) rare-denom",[0, 1, 2, 3], [g3], [g4], eps=0.10)
    add(7,  "theta(g1|g2) all",       [0, 1, 2, 3], [g2], [g1], eps=0.10)
    add(8,  "theta(g2&g4|g1) all",    [0, 1, 2, 3], [g1], [g2, g4], eps=0.10)
    add(9,  "theta(g2|g1&g4) all",    [0, 1, 2, 3], [g1, g4], [g2], eps=0.10)
    add(10, "thr theta(g2|g1)>0.5",   [0, 1, 2, 3], [g1], [g2], kind="threshold", tau=0.5)
    add(11, "thr theta(g4|g2)>0.4",   [0, 1, 2, 3], [g2], [g4], kind="threshold", tau=0.4)
    add(12, "thr theta(g3|g1)>0.1",   [0, 1, 2, 3], [g1], [g3], kind="threshold", tau=0.1)
    add(13, "diff g2|g1 (g0 vs g1)",  [0], [g1], [g2], kind="difference", eps=0.10,
         scope2=[1], d2=[g1], r2=[g2])
    add(14, "diff g4|g2 (g0-2 vs g1-3)", [0, 2], [g2], [g4], kind="difference", eps=0.10,
         scope2=[1, 3], d2=[g2], r2=[g4])
    add(15, "theta(g2|g1) layer2",    [2],       [g1], [g2], eps=0.10)
    add(16, "theta(g2|g1) layer3",    [3],       [g1], [g2], eps=0.10)
    add(17, "theta(g4|g1) all",       [0, 1, 2, 3], [g1], [g4], eps=0.10)
    add(18, "theta(g3|g4) all",       [0, 1, 2, 3], [g4], [g3], eps=0.10)
    add(19, "thr theta(g1|g4)>0.7",   [0, 1, 2, 3], [g4], [g1], kind="threshold", tau=0.7)
    add(20, "theta(g2|g1) narrow",    [0, 1, 2, 3], [g1], [g2], eps=0.02)
    return qs

# ---------------------------------------------------------------------------
# 真实数据工作负载（草案 4.1 CFPB / 4.2 WANDS / 13.2 六类问题）
# ---------------------------------------------------------------------------

def make_cfpb_workload(strata_by_product, strata_by_half, K=4,
                       tau_set=(0.3, 0.5, 0.7)):
    """CFPB：原语 g1=数字渠道困难, g2=客服沟通困难, g3=重复联系, g4=未解决。

    strata_by_product: {产品名: [该产品对应的层编号]}
    strata_by_half:    {半年度标签: [该半年度对应的层编号]}
    六类问题：总体条件比例、两产品组差异、两半年度差异、阈值追问（草案 13.2）。
    只输出描述性条件比例与组间差异；禁止因果/市场总体外推（草案 4.1）。
    """
    all_h = tuple(sorted({h for hs in strata_by_product.values() for h in hs} |
                         {h for hs in strata_by_half.values() for h in hs}))
    prods = sorted(strata_by_product)
    halves = sorted(strata_by_half)
    qs = []
    qid = 0

    def add(name, scope, d, r, kind="value", eps=0.05, tau=0.5,
            scope2=None, d2=None, r2=None, r_meta=None, r_meta2=None):
        nonlocal qid
        qid += 1
        qs.append(Query(qid=qid, name=name, scope=tuple(scope), d_idx=tuple(d),
                        r_idx=tuple(r), kind=kind, eps=eps, tau=tau,
                        scope2=None if scope2 is None else tuple(scope2),
                        d_idx2=None if d2 is None else tuple(d2),
                        r_idx2=None if r2 is None else tuple(r2),
                        r_meta=r_meta, r_meta2=r_meta2))

    g1, g2, g3, g4 = 0, 1, 2, 3
    add("cfpb theta(g2|g1) 全体", all_h, [g1], [g2], eps=0.05)
    add("cfpb theta(g4|g2) 全体", all_h, [g2], [g4], eps=0.05)
    add("cfpb theta(g3|g1) 全体", all_h, [g1], [g3], eps=0.10)
    add("cfpb theta(g4|g1) 全体", all_h, [g1], [g4], eps=0.05)
    if len(prods) >= 2:
        a, b = prods[0], prods[1]
        add(f"cfpb diff theta(g2|g1) {a} vs {b}",
            strata_by_product[a], [g1], [g2], kind="difference", eps=0.10,
            scope2=strata_by_product[b], d2=[g1], r2=[g2])
    if len(halves) >= 2:
        h1, h2 = halves[0], halves[-1]
        add(f"cfpb diff theta(g4|g2) {h1} vs {h2}",
            strata_by_half[h1], [g2], [g4], kind="difference", eps=0.10,
            scope2=strata_by_half[h2], d2=[g2], r2=[g4])
    for tau in tau_set:
        add(f"cfpb thr theta(g2|g1)>{tau}", all_h, [g1], [g2],
            kind="threshold", tau=tau)
    return qs

def make_wands_workload(strata_relevant_rating, strata_other, K=2,
                        tau_set=(0.4, 0.5)):
    """WANDS：原语 p1=判定为相关(Exact 或 Partial), p2=判定为 Exact（草案 4.2）。

    strata_relevant_rating: {评分区间标签: [层编号]}，只含"评分非缺失"的层。
    strata_other:           {其余标签: [层编号]}，含评分缺失层（独立类别，不默认成 0）。
    派生任务示例：
      - Exact/(Exact+Partial)：θ(p2|p1)
      - Exact/Partial 商品中评分>=4 的比例：θ(∅ | p1) 且 r_meta 把"评分>=4"的层置 1
      - 某个相关性条件比例是否超过阈值
    """
    all_h = tuple(sorted({h for hs in strata_relevant_rating.values() for h in hs} |
                         {h for hs in strata_other.values() for h in hs}))
    qs = []
    qid = 0

    def add(name, scope, d, r, kind="value", eps=0.05, tau=0.5, r_meta=None,
            scope2=None, d2=None, r2=None, r_meta2=None):
        nonlocal qid
        qid += 1
        qs.append(Query(qid=qid, name=name, scope=tuple(scope), d_idx=tuple(d),
                        r_idx=tuple(r), kind=kind, eps=eps, tau=tau,
                        scope2=None if scope2 is None else tuple(scope2),
                        d_idx2=None if d2 is None else tuple(d2),
                        r_idx2=None if r2 is None else tuple(r2),
                        r_meta=r_meta, r_meta2=r_meta2))

    p1, p2 = 0, 1
    add("wands theta(exact|relevant)", all_h, [p1], [p2], eps=0.05)
    # 分子带元数据条件：评分>=4（非缺失层）
    rating_ge4 = {}
    for lab, hs in strata_relevant_rating.items():
        flag = 1 if ">=4" in str(lab) else 0
        for h in hs:
            rating_ge4[h] = flag
    add("wands theta(rating>=4 | relevant)", all_h, [p1], [],
        eps=0.10, r_meta=rating_ge4)
    for tau in tau_set:
        add(f"wands thr theta(exact|relevant)>{tau}", all_h, [p1], [p2],
            kind="threshold", tau=tau)
    # 草案 13.2 的六类问题补齐：两个群体的条件比例 + 群体差异
    ge4 = tuple(sorted({h for lab, hs in strata_relevant_rating.items()
                        if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in strata_relevant_rating.items()
                        if "<4" in str(lab) for h in hs}))
    if ge4 and lt4:
        add("wands theta(exact|relevant, rating>=4)", ge4, [p1], [p2], eps=0.05)
        add("wands theta(exact|relevant, rating<4)", lt4, [p1], [p2], eps=0.05)
        add("wands diff theta(exact|relevant) >=4 minus <4", ge4, [p1], [p2],
            kind="difference", eps=0.05, scope2=lt4, d2=[p1], r2=[p2])
    return qs


def make_wands_closure_workload(strata_relevant_rating, strata_other, K=2,
                                tau_grid=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70),
                                group_tau_grid=(0.30, 0.50, 0.70)):
    """WANDS 查询【闭包】压力集（草案 5.3、8.5）：

    全部查询共用同一组原语（p1/p2）、同一组元数据原子（评分区间）与同一份联合状态，
    只改变阈值 τ 或组别。按草案 8.5，这类闭包内追问【不需要】为每个查询重新分配
    置信预算——统计代价只取决于预声明的统计基约束数，而不是用户追问了多少次。
    本函数用于实证这一性质：在闭包内把查询数从若干个扩到数十个，观察覆盖率是否
    仍然维持、认证成本如何变化。

    不得用它冒充"开放域任意自然语言查询"：闭包外的新原语/新分组仍需新阶段。
    """
    all_h = tuple(sorted({h for hs in strata_relevant_rating.values() for h in hs} |
                         {h for hs in strata_other.values() for h in hs}))
    ge4 = tuple(sorted({h for lab, hs in strata_relevant_rating.items()
                        if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in strata_relevant_rating.items()
                        if "<4" in str(lab) for h in hs}))
    qs, qid = [], 0

    def add(name, scope, d, r, kind="value", eps=0.05, tau=0.5,
            r_meta=None, scope2=None, d2=None, r2=None):
        nonlocal qid
        qid += 1
        qs.append(Query(qid=qid, name=name, scope=tuple(scope), d_idx=tuple(d),
                        r_idx=tuple(r), kind=kind, eps=eps, tau=tau,
                        scope2=None if scope2 is None else tuple(scope2),
                        d_idx2=None if d2 is None else tuple(d2),
                        r_idx2=None if r2 is None else tuple(r2), r_meta=r_meta))

    p1, p2 = 0, 1
    # 数值查询：总体 + 两个评分组别
    add("theta(exact|relevant)", all_h, [p1], [p2], eps=0.05)
    if ge4:
        add("theta(exact|relevant, rating>=4)", ge4, [p1], [p2], eps=0.05)
    if lt4:
        add("theta(exact|relevant, rating<4)", lt4, [p1], [p2], eps=0.05)
    rating_ge4 = {}
    for lab, hs in strata_relevant_rating.items():
        flag = 1 if ">=4" in str(lab) else 0
        for h in hs:
            rating_ge4[h] = flag
    add("theta(rating>=4 | relevant)", all_h, [p1], [], eps=0.10, r_meta=rating_ge4)
    if ge4 and lt4:
        add("diff theta(exact|relevant) >=4 minus <4", ge4, [p1], [p2],
            kind="difference", eps=0.05, scope2=lt4, d2=[p1], r2=[p2])
    # 阈值追问：只改 τ，不新增统计基（草案 8.5）
    for tau in tau_grid:
        add(f"thr theta(exact|relevant)>{tau:.2f}", all_h, [p1], [p2],
            kind="threshold", tau=tau)
    for tau in group_tau_grid:
        if ge4:
            add(f"thr theta(exact|relevant,rating>=4)>{tau:.2f}", ge4, [p1], [p2],
                kind="threshold", tau=tau)
        if lt4:
            add(f"thr theta(exact|relevant,rating<4)>{tau:.2f}", lt4, [p1], [p2],
                kind="threshold", tau=tau)
    return qs
