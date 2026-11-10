# -*- coding: utf-8 -*-
"""有限总体无放回置信序列（草案 8.1 / 8.2）。

核心数学已由 validate_cs_demo.py 独立验证：
  - 二分端点 vs 逐候选穷举：3285 项全部一致；
  - N∈{10,20,40}、α∈{0.05,0.10} 下最大『任意时刻越界概率』均 <= α
    （N=40、α=0.05 → 0.029858，与草案第 24 节一致）。

实现说明（2026-11 重写，数学定义不变）
-------------------------------------
置信集 C_n(δ) = {k : max_{s<=n} M_s / L_{N,k}(x_{1:s}) <= 1/δ} 随 n 单调收缩（max 只增不减）。
因此：
  * 对每个候选真实计数 k，只需维护「运行最大对数似然比」rm[k]；
  * 一旦 k 与观测不相容（k < x_s 或 N-k < s-x_s）或 rm[k] 超过阈值，k 永久不可行，
    可从窗口里删除——窗口只缩不涨；
  * push 的代价 O(窗口)，interval() 只需在窗口里取首尾可行点，不再做二分。
窗口按「首次调用 interval 的 δ」收缩；若之后用更小的 δ（更宽的区间）查询，
自动回退到由累计观测重建的慢速精确路径（保持任意 δ 下结果正确）。
"""
import math
import numpy as np

# 全局预计算 lgamma 查表：T[v] = lgamma(v + 1)
_MAX_PRECOMP = 60000
_GLOBAL_T = np.array([math.lgamma(float(v + 1)) for v in range(_MAX_PRECOMP)],
                     dtype=np.float64)


class FinitePopulationCS:
    """对固定剩余总体（大小 N）的单个二值统计基函数维持 anytime-valid 置信区间。

    push(bit)：推进一个合法随机前缀位置（按排列顺序，不可跳过）。
    interval(delta)：返回当前置信区间 [lo, hi]（包含真实计数 k* 的概率 >= 1-delta）。
    """

    def __init__(self, N: int):
        assert isinstance(N, (int, np.integer)) and N >= 0
        self.N = int(N)
        if self.N + 3 <= _MAX_PRECOMP:
            self._T = _GLOBAL_T[:self.N + 3]
        else:
            self._T = np.array([math.lgamma(float(v + 1)) for v in range(0, self.N + 3)],
                               dtype=np.float64)
        self._n = 0
        self._x = 0
        self._bits = np.zeros(self.N + 1, dtype=np.int8)   # 完整观测序列（慢路径与审计用）
        # 窗口：候选计数 k ∈ [wlo, whi]；rm[j] 对应 k = wlo + j（运行最大对数似然比，s=0 项为 0）
        self._wlo = 0
        self._whi = self.N
        self._rm = np.zeros(self.N + 1, dtype=np.float64)
        self._trim_delta = None            # 窗口已按该 δ 收缩过
        self._iv_cache = {}

    # ------------------------------------------------------------------ #
    @property
    def n(self):
        return self._n

    @property
    def x(self):
        return self._x

    def _lg(self, a):
        """lgamma(a)（支持标量与 numpy 数组），要求 a >= 1。"""
        return self._T[np.asarray(a, dtype=np.int64) - 1]

    def push(self, bit):
        """追加一个观测，并用 O(窗口) 的向量运算更新运行最大对数似然比。"""
        N, T = self.N, self._T
        b = 1 if bit else 0
        self._bits[self._n] = b
        s = self._n + 1
        x = self._x + b
        self._n, self._x = s, x
        self._iv_cache.clear()

        # 与观测不相容的 k 永久剔除：k >= x 且 N-k >= s-x
        lo = max(self._wlo, x)
        hi = min(self._whi, N - (s - x))
        if lo > hi:                        # 理论上不会发生（真实 k* 恒相容），防御
            self._wlo, self._whi = lo, lo - 1
            self._rm = self._rm[:0]
            return
        rm = self._rm[lo - self._wlo: hi - self._wlo + 1]
        y = s - x
        # logM_s = log B(x+1, s-x+1)
        logM = T[x] + T[y] - T[s + 1]
        # logL_s(k) = [lg(k+1)-lg(k-x+1)] + [lg(N-k+1)-lg(N-k-y+1)] - [lg(N+1)-lg(N-s+1)]
        #           = T[k]-T[k-x] + T[N-k]-T[N-k-y] - (T[N]-T[N-s])
        a1 = T[lo: hi + 1]
        a2 = T[lo - x: hi - x + 1]
        c1 = T[N - hi: N - lo + 1][::-1]
        c2 = T[N - hi - y: N - lo - y + 1][::-1]
        logL = (a1 - a2) + (c1 - c2) - (T[N] - T[N - s])
        self._rm = np.maximum(rm, logM - logL)
        self._wlo, self._whi = lo, hi

    # ------------------------------------------------------------------ #
    def _max_logE(self, k):
        """慢路径：由累计观测直接计算 log(max_s E_s(k))；k 与观察不相容 → +inf。"""
        N, n, xn = self.N, self._n, self._x
        k = int(k)
        if k < xn or (N - k) < (n - xn):
            return np.inf
        T, lg = self._T, self._lg
        X = np.concatenate(([0], np.cumsum(self._bits[:n], dtype=np.int64)))
        s = np.arange(n + 1, dtype=np.int64)
        logM = T[X] + T[s - X] - T[s + 1]
        logL = (lg(k + 1) - lg(k - X + 1)) + (lg(N - k + 1) - lg(N - k - (s - X) + 1)) \
               - (lg(N + 1) - lg(N - s + 1))
        return float(np.max(logM - logL))

    def _interval_slow(self, delta):
        """慢速精确路径（任意 δ）：对 k 做二分，端点由 _max_logE 判定。"""
        N, n, xn = self.N, self._n, self._x
        if n == 0:
            return (0, N)
        if n == N:
            return (xn, xn)
        thresh = -math.log(delta)
        T = self._T
        ks = np.arange(xn, N - (n - xn) + 1, dtype=np.int64)
        logL_final = (T[ks] - T[ks - xn]) + (T[N - ks] - T[N - ks - (n - xn)]) \
                     - (T[N] - T[N - n])
        anchor = int(ks[int(np.argmax(logL_final))])

        def feasible(k):
            return self._max_logE(k) <= thresh

        if not feasible(anchor):
            anchor = next((k for k in range(N + 1) if feasible(k)), None)
            if anchor is None:
                return None
        if feasible(0):
            lo = 0
        else:
            a, b = 0, anchor
            while b - a > 1:
                m = (a + b) // 2
                if feasible(m):
                    b = m
                else:
                    a = m
            lo = b
        if feasible(N):
            hi = N
        else:
            a, b = anchor, N
            while b - a > 1:
                m = (a + b) // 2
                if feasible(m):
                    a = m
                else:
                    b = m
            hi = a
        return (lo, hi)

    def interval(self, delta):
        """返回 [lo, hi]；n=0 → (0,N)；n=N → (x,x)；无可行点 → None。"""
        N, n = self.N, self._n
        if n == 0:
            return (0, N)
        if n == N:
            return (self._x, self._x)
        key = (n, float(delta))
        if key in self._iv_cache:
            return self._iv_cache[key]
        if self._trim_delta is None:
            self._trim_delta = float(delta)         # 第一次查询确定窗口对应的 δ
        if delta < self._trim_delta:
            # 更小 δ → 可行集比窗口更宽，窗口已不够用：走精确慢路径（结果仍正确）
            res = self._interval_slow(delta)
            self._iv_cache[key] = res
            return res
        thresh = -math.log(delta)
        ok = self._rm <= thresh
        if not ok.any():
            self._iv_cache[key] = None
            return None
        idx = np.flatnonzero(ok)
        j0, j1 = int(idx[0]), int(idx[-1])
        res = (self._wlo + j0, self._wlo + j1)
        # 嵌套性：同一 δ 下当前可行集 ⊆ 之前所有可行集，窗口可安全收缩到可行首尾；
        # δ 更大（区间更窄）时只读，不收缩，以免破坏 trim_delta 对应的窗口。
        if float(delta) == self._trim_delta:
            self._rm = self._rm[j0: j1 + 1]
            self._wlo, self._whi = res
        self._iv_cache[key] = res
        return res
