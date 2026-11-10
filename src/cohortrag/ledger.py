# -*- coding: utf-8 -*-
"""证据账本（草案 7 节）：历史已知 E / 随机前缀 S / 定向额外暴露 T。

关键规则（草案 7.2/7.3）：
  - 阶段开始时，任意历史已见记录（E）冻结为精确已知质量，绝不当作随机样本；
  - 随机样本必须是该层剩余总体均匀排列的【完整前缀】，不可按内容/标签跳过；
  - 定向检索暴露的记录（T）只贡献确定性已知质量，不进入随机计数；
  - 随机排列随后走到 T 中的记录时：读缓存、计入随机前缀（合法位置消费，不产生新调用）。
"""
import numpy as np
from . import states

class EvidenceLedger:
    def __init__(self, snapshot, seed, K, cache_labels=None, e_as_random=False):
        self.snapshot = snapshot
        self.K = K
        self.J = 1 << K
        self.seed = seed
        self.e_as_random = bool(e_as_random)
        self.strata = snapshot.strata()
        self.N_h = {h: len(snapshot.stratum_ids(h)) for h in self.strata}
        id_to_h = dict(zip(snapshot.frame["id"], snapshot.frame["stratum"]))

        self.cache = {}
        self.E_ids = {h: set() for h in self.strata}
        self.E_counts = {h: np.zeros(self.J, dtype=np.int64) for h in self.strata}
        self.obs = {h: np.zeros(self.J, dtype=np.int64) for h in self.strata}
        self._cached_ids = {h: [] for h in self.strata}
        if cache_labels:
            valid_ids = set(int(i) for i in snapshot.frame["id"])
            for i, lab in cache_labels.items():
                i = int(i)
                if i not in valid_ids:
                    continue
                self.cache[i] = np.asarray(lab, dtype=np.int8)
                h = int(id_to_h[i])
                z = states.encode_state(tuple(int(v) for v in lab))
                if self.e_as_random:
                    # 错误方法（去除选择来源契约）：历史已见记录被错误地当作随机样本，
                    # 不记入精确已知 E，也不作为观测下界钉死（obs 仅由真实随机前缀填充）；
                    # 它们只通过 CS（置信序列）与累计量以“随机观察”身份进入估计。
                    self._cached_ids[h].append(i)
                else:
                    self.E_ids[h].add(i)
                    self.E_counts[h][z] += 1
                    self.obs[h][z] += 1

        # 剩余总体 R 与随机排列
        self.R_ids = {}
        self.perm = {}
        self.pos = {h: 0 for h in self.strata}
        self.S_ids = {h: set() for h in self.strata}
        self.T_ids = {h: set() for h in self.strata}
        self.prefix_g = {h: [] for h in self.strata}
        # 每层随机前缀的联合状态计数（增量维护，规划时 O(J) 读取而非重扫前缀）
        self.prefix_z = {h: np.zeros(self.J, dtype=np.int64) for h in self.strata}
        self.hits = {h: 0 for h in self.strata}
        self.misses = {h: 0 for h in self.strata}
        for h in self.strata:
            all_ids = set(int(i) for i in snapshot.stratum_ids(h))
            if self.e_as_random:
                # 把缓存记录放到随机排列的“前 |缓存| 个位置”，假装它们是随机样本；
                # R 取整层，CS 在其上作用（偏移 E_f=0，因 E_counts 为空）。
                cached = sorted(self._cached_ids[h])
                rest = sorted(all_ids - set(cached))
                rng = np.random.default_rng(seed + h * 100003)
                # 「假装这批记录是随机样本」= 均匀随机子集 + 均匀随机顺序。
                # 顺序必须是随机的：可交换性是置信序列成立的前提。若按 id 排序
                # （WANDS 的 id = 意图×1e6+商品，等于按搜索意图聚簇），
                # 序列高度相关，混合鞅会被击穿、置信序列变空——那是排序假象，
                # 不是「去除契约」的后果，因此这里必须打乱。
                perm_cached = [cached[i] for i in rng.permutation(len(cached))]
                perm_rest = [rest[i] for i in rng.permutation(len(rest))]
                self.R_ids[h] = list(all_ids)
                self.perm[h] = perm_cached + perm_rest
                self.pos[h] = len(cached)
                self.S_ids[h] = set(cached)
                self.prefix_g[h] = [self.cache[i] for i in perm_cached]
                for i in perm_cached:
                    self.prefix_z[h][states.encode_state(
                        tuple(int(v) for v in self.cache[i]))] += 1
            else:
                R = sorted(all_ids - self.E_ids[h])
                self.R_ids[h] = R
                rng = np.random.default_rng(seed + h * 100003)
                self.perm[h] = [R[i] for i in rng.permutation(len(R))]

    # ---- 只读视图 ----
    def N_rem(self, h):
        return len(self.R_ids[h])

    def n_prefix(self, h):
        return self.pos[h]

    def unseen(self, h):
        return self.N_rem(h) - len(self.S_ids[h] | self.T_ids[h])

    def exposed_ids(self, h):
        return self.E_ids[h] | self.S_ids[h] | self.T_ids[h]

    def exposed_count(self, h):
        return int(self.obs[h].sum())

    def cost_per_pos(self, h):
        tot = self.hits[h] + self.misses[h]
        return 1.0 if tot == 0 else self.misses[h] / tot

    # ---- 写入 ----
    def add_targeted(self, h, ids, labels):
        """定向检索额外暴露（T）。调用者负责为这些标签付费。"""
        ids = [int(i) for i in ids]
        for i, lab in zip(ids, labels):
            assert i in self.R_ids[h] and i not in self.S_ids[h], \
                "定向暴露必须来自剩余总体且未进入随机前缀"
            if i not in self.cache:
                self.cache[i] = np.asarray(lab, dtype=np.int8)
                self.obs[h][states.encode_state(tuple(int(v) for v in lab))] += 1
            self.T_ids[h].add(i)

    def consume_prefix(self, h, n_positions, contract):
        """按排列顺序消费前缀位置（不可跳过）；批量调用语义算子；返回新增调用数。

        顺序保证：先按窗口内未缓存 id 批量判标签，再按排列顺序回填前缀计数，
        因此随机前缀始终是排列的完整前缀（草案 10.1 不变式）。
        """
        start = self.pos[h]
        end = min(start + n_positions, len(self.perm[h]))
        window = [self.perm[h][p] for p in range(start, end)]
        uncached = [i for i in window if i not in self.cache]
        if uncached:
            labels = contract.judge(uncached, self.snapshot.texts(uncached))
            for i, lab in zip(uncached, labels):
                self.cache[i] = np.asarray(lab, dtype=np.int8)
                self.obs[h][states.encode_state(tuple(int(v) for v in lab))] += 1
                self.misses[h] += 1
        fresh = set(uncached)
        for p in range(start, end):
            i = self.perm[h][p]
            # 指针走到已缓存（含定向暴露 T）的记录：读缓存、零调用、前缀长度 +1；
            # 暴露计数 obs 不重复累加（该记录已在首次取得标签时计入）。
            self.hits[h] += 0 if i in fresh else 1
            self.S_ids[h].add(i)
            lab = self.cache[i]
            self.prefix_g[h].append(np.asarray(lab, dtype=np.int8))
            self.prefix_z[h][states.encode_state(tuple(int(v) for v in lab))] += 1
        self.pos[h] = end
        return len(uncached)
