# -*- coding: utf-8 -*-
"""联合语义状态空间与统计基编译（草案 5.1、7.4、8.2）。

K 个二值原语 → J = 2^K 个联合状态；分子/分母条件都是状态上的二值函数。
统计基编译：去常数、去重复真值表、互补函数去重（用 N - f 替代）。
"""
import numpy as np

def encode_state(g):
    """原语元组 g=(g1,...,gK)（0/1）→ 状态编号 z（g1 为最低位）。"""
    z = 0
    for j, v in enumerate(g):
        z |= (1 if v else 0) << j
    return z

def decode_state(z, K):
    """状态编号 → 原语元组。"""
    return tuple((z >> j) & 1 for j in range(K))

def state_space_size(K):
    return 1 << K

def truth_table(K, f):
    """f: (g1,...,gK) -> bool；返回长度 2^K 的 0/1 numpy 数组（按状态编号索引）。"""
    return np.array([1 if f(decode_state(z, K)) else 0 for z in range(1 << K)],
                    dtype=np.int8)

def and_table(K, primitives):
    """原语索引集合的 AND 条件的真值表。"""
    return truth_table(K, lambda g: all(g[j] for j in primitives))

def compile_basis(tables, names):
    """统计基编译（草案 8.2）。

    tables: 查询需要的分母/分子真值表列表；names: 对应名称。
    返回 (basis_tables, basis_names)：
      - 丢弃常数表（无需估计）；
      - 合并完全相同的表；
      - 互补表不重复分配预算（用 N - f 表达）。
    """
    uniq_t, uniq_n = [], []
    for name, t in zip(names, tables):
        if t.sum() == 0 or t.sum() == len(t):
            continue                                    # 常数函数
        if any(np.array_equal(t, u) for u in uniq_t):
            continue                                    # 完全相同
        if any(np.array_equal(t, 1 - u) for u in uniq_t):
            continue                                    # 互补：用 N - f
        uniq_t.append(t)
        uniq_n.append(name)
    return uniq_t, uniq_n
