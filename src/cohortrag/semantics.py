# -*- coding: utf-8 -*-
"""语义算子契约（草案 4、7.1、17.4）。

LLM 只负责有界的语义判断；契约（模型/prompt/视图版本）进入缓存键。
本模块提供三种算子：
  - OracleOperator       ：读隐藏真值（仅评测器使用，模拟全量物化参考）
  - NoisyOracleOperator  ：隐藏真值 + 固定逐记录噪声 → 模拟『运行语义』
  - StubLLMOperator      ：占位接口，接本地 Qwen3-8B 时在这里实现（草案 17.1）
"""
import numpy as np

class SemanticContract:
    """契约：版本名 + 原语数 K + judge(ids, texts) -> (n, K) 0/1 数组。"""

    def __init__(self, name: str, K: int):
        self.name = name
        self.K = K
        self.version = name

    def judge(self, ids, texts):
        raise NotImplementedError

class OracleOperator(SemanticContract):
    """读隐藏真值标签（只给评测器/全量物化参考用）。"""

    def __init__(self, hidden_labels, name="oracle", K=None):
        self.hidden = hidden_labels
        if K is None:
            K = len(next(iter(hidden_labels.values())))
        super().__init__(name, K)

    def judge(self, ids, texts=None):
        out = np.zeros((len(ids), self.K), dtype=np.int8)
        for r, i in enumerate(ids):
            out[r] = self.hidden[int(i)]
        return out

class NoisyOracleOperator(SemanticContract):
    """真值 + 逐 (id, j) 固定噪声翻转（确定性，可复现）→ 模拟冻结模型的运行语义。

    同一 id 的噪声与调用顺序无关（按 id 派生种子），因此在线判断与
    『全量物化』逐条判断完全一致——这正是置信序列所覆盖的 θ_M。
    """

    def __init__(self, hidden_labels, flip_probs, name="noisy-oracle", K=None):
        if K is None:
            K = len(next(iter(hidden_labels.values())))
        super().__init__(name, K)
        self.hidden = hidden_labels
        self.flip_probs = np.asarray(flip_probs, dtype=np.float64)

    def judge(self, ids, texts=None):
        out = np.zeros((len(ids), self.K), dtype=np.int8)
        for r, i in enumerate(ids):
            g = self.hidden[int(i)]
            for j in range(self.K):
                rng = np.random.default_rng(int(i) * 1000 + j)
                bit = int(g[j])
                if rng.random() < self.flip_probs[j]:
                    bit = 1 - bit
                out[r, j] = bit
        return out

class StubLLMOperator(SemanticContract):
    """本地模型占位：接入 Qwen3-8B（草案 17.1）时在此实现 judge()。

    要点（草案 17.4/7.1）：
      - 冻结模型 revision、prompt 全文、解码参数、文本视图 → 全部进入缓存键；
      - 批量推理、短输出（<=16 token）、确定性解码；
      - 四个原语打包成一个固定 prompt（整体版本化）。
    """

    def __init__(self, name="stub-llm", K=4):
        super().__init__(name, K)

    def judge(self, ids, texts=None):
        raise NotImplementedError(
            "在 GPU 机器上实现：批量送入本地 Qwen3-8B，解析 4 个二值原语。"
        )
