# -*- coding: utf-8 -*-
"""
COHORT-RAG 草案第 8.1 节「有限总体无放回置信序列」的 CPU 原型验证。

这是【真实计算】：不需要 GPU、不需要 LLM、不需要真实数据集，
只用标准库 math.lgamma 实现草案中的数学，验证两件事：

1) 端点算法正确性：二分端点 vs 逐候选穷举，对大量随机抽样路径逐一比对。
2) 覆盖保证：用【精确动态规划】计算「任意时刻越界概率」，检验它 <= alpha。

数学（对应草案 8.1 节）：
  L_{N,k}(prefix)  = k^underline{x} * (N-k)^underline{s-x} / N^underline{s}   （超几何似然，只依赖计数 s,x）
  M_s              = B(x+1, s-x+1)                                          （均匀先验混合 = beta 函数）
  E_s(k)           = M_s / L_{N,k}(prefix)                                   （混合似然比）
  C_n(delta)       = { k : max_{0<=s<=n} E_s(k) <= 1/delta }
  保证：Pr( 所有时刻 k* 都在 C 内 ) >= 1 - delta
"""
import math
import random

def gammaln(x):
    return math.lgamma(x)

def betaln(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def log_falling(a, b):
    """log( a*(a-1)*...*(a-b+1) )，b>=0"""
    if b < 0:
        raise ValueError("b must be >= 0")
    if b == 0:
        return 0.0
    return gammaln(a + 1) - gammaln(a - b + 1)

def log_likelihood(N, k, s, x):
    """log L：给定总数 N、真实总数 k，前 s 次无放回抽样观察到 x 个正例的有序序列概率对数。"""
    if k < x or (N - k) < (s - x):
        return float("-inf")   # k 与已观察样本不相容
    return log_falling(k, x) + log_falling(N - k, s - x) - log_falling(N, s)

def log_mixture(s, x):
    """log M_s = log B(x+1, s-x+1)"""
    return betaln(x + 1, s - x + 1)

def running_max_logE(N, k, counts):
    """counts: [(s0,x0), (s1,x1), ...] 前缀每个时刻的 (已抽数, 正例数)。
       返回 max_s log E_s(k)。k 与某前缀不相容时返回 +inf。"""
    rmax = float("-inf")
    for (s, x) in counts:
        ll = log_likelihood(N, k, s, x)
        if ll == float("-inf"):
            return float("inf")
        e = log_mixture(s, x) - ll
        if e > rmax:
            rmax = e
    return rmax

def cs_exhaustive(N, counts, delta):
    """逐候选穷举：返回置信集 [lo, hi] 及全部可行 k。这是'参考真值'。"""
    thresh = -math.log(delta)
    feas = [k for k in range(N + 1) if running_max_logE(N, k, counts) <= thresh]
    if not feas:
        return None
    return (min(feas), max(feas), feas)

def cs_bisect(N, counts, delta):
    """二分端点：先找锚点（末时刻似然最大者），再左右二分边界。返回 [lo, hi]。"""
    thresh = -math.log(delta)
    def feasible(k):
        return running_max_logE(N, k, counts) <= thresh
    s_n, x_n = counts[-1]
    if s_n == 0:
        return (0, N)                       # 无任何观察 -> 全部 k 都可行
    # 锚点：末时刻最大似然估计
    anchor = max(range(N + 1), key=lambda k: log_likelihood(N, k, s_n, x_n))
    if not feasible(anchor):
        anchor = next((k for k in range(N + 1) if feasible(k)), None)
        if anchor is None:
            return None
    # 左端点
    if feasible(0):
        lo = 0
    else:
        lo_lo, lo_hi = 0, anchor             # feasible(0)=False, feasible(anchor)=True
        while lo_hi - lo_lo > 1:
            mid = (lo_lo + lo_hi) // 2
            if feasible(mid):
                lo_hi = mid
            else:
                lo_lo = mid
        lo = lo_hi
    # 右端点
    if feasible(N):
        hi = N
    else:
        hi_lo, hi_hi = anchor, N             # feasible(anchor)=True, feasible(N)=False
        while hi_hi - hi_lo > 1:
            mid = (hi_lo + hi_hi) // 2
            if feasible(mid):
                hi_lo = mid
            else:
                hi_hi = mid
        hi = hi_lo
    return (lo, hi)

def violation_probability_dp(N, kstar, delta):
    """精确动态规划：计算'任意时刻越界概率' = Pr(存在 s: k* 不在 C_s 内)。
       状态 (s, x) = 前 s 次抽到 x 个正例；E_s(k*) 只依赖 (s,x)，故可逐状态局部判越界。
       一旦越界即吸收为'已越界'，不再传播。"""
    thresh = -math.log(delta)
    def f(s, x):
        ll = log_likelihood(N, kstar, s, x)
        return log_mixture(s, x) - ll
    states = {0: 1.0}                        # x -> 质量，当前 s=0
    viol = 0.0
    for s in range(0, N):
        nxt = {}
        for x, mass in states.items():
            p_succ = (kstar - x) / (N - s)   # 下一次抽到正例的概率
            if p_succ > 0:
                nxt[x + 1] = nxt.get(x + 1, 0.0) + mass * p_succ
            if p_succ < 1.0:
                nxt[x] = nxt.get(x, 0.0) + mass * (1.0 - p_succ)
        states = {}
        for x, mass in nxt.items():
            if f(s + 1, x) > thresh:         # 该时刻越界 -> 吸收
                viol += mass
            else:
                states[x] = mass
    for x, mass in states.items():           # s=N 终点（x 必为 kstar）
        if f(N, x) > thresh:
            viol += mass
    return viol

def simulate_violation(N, kstar, delta, trials, seed=0):
    """蒙特卡洛交叉验证：随机打乱，模拟到首次越界。"""
    rng = random.Random(seed)
    thresh = -math.log(delta)
    viol = 0
    for _ in range(trials):
        perm = [1] * kstar + [0] * (N - kstar)
        rng.shuffle(perm)
        x = 0
        bad = False
        for s in range(1, N + 1):
            x += perm[s - 1]
            if log_mixture(s, x) - log_likelihood(N, kstar, s, x) > thresh:
                bad = True
                break
        if bad:
            viol += 1
    return viol / trials

def main():
    print("=" * 72)
    print("验证 1：二分端点 vs 逐候选穷举（对随机抽样路径逐一比对）")
    print("=" * 72)
    total, mismatch = 0, 0
    for N in (10, 20, 40):
        for kstar in (0, 1, N // 2, N - 1, N):
            for seed in (1, 2, 3):
                perm = [1] * kstar + [0] * (N - kstar)
                random.Random(seed).shuffle(perm)
                x = 0
                counts = [(0, 0)]
                for s in range(1, N + 1):
                    x += perm[s - 1]
                    counts.append((s, x))
                for n in range(0, N + 1):
                    prefix = counts[: n + 1]
                    for delta in (0.05, 0.1, 0.01):
                        ex = cs_exhaustive(N, prefix, delta)
                        bi = cs_bisect(N, prefix, delta)
                        total += 1
                        if ex is None:
                            if bi is not None:
                                mismatch += 1
                        else:
                            if bi is None or (ex[0], ex[1]) != bi:
                                mismatch += 1
    print(f"  端点比对总数：{total} 项")
    print(f"  不一致数：{mismatch}")
    print(f"  结论：{'全部一致，二分端点正确' if mismatch == 0 else '存在不一致，需排查'}")

    print()
    print("=" * 72)
    print("验证 2：精确动态规划计算『任意时刻越界概率』（应 <= alpha）")
    print("=" * 72)
    print(f"  {'N':>3}  {'alpha':>6}  {'最大越界概率':>14}  {'<= alpha?':>10}")
    for N in (10, 20, 40):
        for alpha in (0.05, 0.1):
            maxv = max(violation_probability_dp(N, kstar, alpha)
                       for kstar in range(N + 1))
            ok = "是" if maxv <= alpha + 1e-12 else "否"
            print(f"  {N:>3}  {alpha:>6.2f}  {maxv:>14.6f}  {ok:>10}")

    print()
    print("=" * 72)
    print("验证 3：动态规划 vs 蒙特卡洛模拟（交叉验证正确性）")
    print("=" * 72)
    for (N, kstar, alpha, trials) in [(20, 5, 0.05, 200000),
                                      (20, 10, 0.05, 200000),
                                      (40, 20, 0.05, 200000)]:
        dp = violation_probability_dp(N, kstar, alpha)
        mc = simulate_violation(N, kstar, alpha, trials, seed=12345)
        print(f"  N={N:>2} k*={kstar:>2} alpha={alpha}  DP={dp:.6f}  蒙特卡洛({trials}次)={mc:.6f}")

    print()
    print("=" * 72)
    print("说明：以上是 CPU 端真实计算（标准库 math 即可完成）。")
    print("它证明的是『这个置信序列的数学与实现是对的』——")
    print("既不需要 GPU，也不需要 LLM，更不涉及真实数据集。")
    print("草案第 24 节那个 0.029858 就是这一类小总体覆盖检查的数字。")
    print("=" * 72)

if __name__ == "__main__":
    main()
