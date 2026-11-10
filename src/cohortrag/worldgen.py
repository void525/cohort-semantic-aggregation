# -*- coding: utf-8 -*-
"""合成世界生成器（草案 12.3 的『程序化压力测试』数据源）。

生成：
  - 冻结快照（id / stratum / 模板文本）；
  - 隐藏真值标签 g（K 个二值原语，含条件依赖与稀有原语，制造稀少分母）；
  - 元数据分层（2 个元数据原子 × 2 个分位层 = 4 层，含组间差异）。

所有由此产生的结果必须标注为 synthetic，不得冒充真实数据结果（草案 12.3）。
"""
import numpy as np
import pandas as pd

TOKENS = {
    0: "digital-channel",
    1: "support-contact",
    2: "repeated-transfer",
    3: "unresolved",
}

def make_synthetic_world(N=2000, K=4, H=4, seed=0,
                         flip_probs=(0.08, 0.08, 0.10, 0.06)):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=N)          # 元数据原子 1（如产品组）
    quantile = rng.integers(0, 2, size=N)       # 元数据原子 2（如分位层）
    stratum = group * 2 + quantile              # H = 4 层

    # ---- 隐藏真值生成：条件依赖 + 稀有原语 ----
    g = np.zeros((N, K), dtype=np.int8)
    p1 = 0.45 + 0.06 * (group == 0) - 0.04 * (quantile == 0)   # g1 有组间差异
    g[:, 0] = (rng.random(N) < p1).astype(np.int8)
    p2 = np.where(g[:, 0] == 1, 0.55 + 0.10 * (group == 1), 0.08)  # g2|g1 随组别变化
    g[:, 1] = (rng.random(N) < p2).astype(np.int8)
    g[:, 2] = (rng.random(N) < 0.06).astype(np.int8)            # g3 稀有
    p4 = 0.30 + 0.15 * g[:, 1]                                  # g4 条件于 g2
    g[:, 3] = (rng.random(N) < p4).astype(np.int8)

    # ---- 模板文本（含关键词信号，供偏置缓存/检索演示使用）----
    texts = []
    for i in range(N):
        sig = [TOKENS[j] for j in range(K) if g[i, j] == 1]
        txt = ("complaint about product group " + str(int(group[i])) +
               " : " + (" ".join(sig) if sig else "no specific issue") +
               " (record " + str(i) + ")")
        texts.append(txt)

    frame = pd.DataFrame({
        "id": np.arange(N, dtype=np.int64),
        "stratum": stratum.astype(np.int64),
        "group": group.astype(np.int64),
        "quantile": quantile.astype(np.int64),
        "text": texts,
    })
    hidden = {int(i): g[i].copy() for i in range(N)}
    return frame, hidden
