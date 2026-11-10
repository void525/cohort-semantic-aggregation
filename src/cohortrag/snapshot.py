# -*- coding: utf-8 -*-
"""冻结数据快照与精确关系层（草案 4、5.1、6、10.2 的 CPU 部分）。

生产中该层使用 DuckDB + Parquet（草案 17.1）；本实现用 pandas 提供同一接口，
使整条 pipeline 可以在无 GPU、无外部服务的机器上先跑通。
快照一经创建即冻结：内容与 manifest 哈希绑定。
"""
import hashlib
import pandas as pd

class Snapshot:
    """冻结快照。不变式（草案 10.2）：id 唯一、每条记录恰好一个层、可哈希校验。"""

    def __init__(self, frame: pd.DataFrame, name: str, unit: str = "record", manifest_override: str = None):
        self.frame = frame.copy()
        self.name = name
        self.unit = unit
        self._assert_invariants()
        self.manifest = manifest_override if manifest_override else self._compute_manifest()
        self._strata = sorted(int(h) for h in self.frame["stratum"].unique())
        self._stratum_ids = {
            h: self.frame.loc[self.frame["stratum"] == h, "id"].to_numpy()
            for h in self._strata
        }
        self._text_by_id = dict(zip(self.frame["id"].astype(int), self.frame["text"].astype(str)))

    def _assert_invariants(self):
        f = self.frame
        assert f["id"].is_unique, "主键 id 必须唯一（草案 10.2）"
        assert f["id"].notna().all(), "id 不得缺失"
        assert f["stratum"].notna().all(), "stratum 不得缺失"
        assert f["text"].notna().all(), "text 视图不得缺失"

    def _compute_manifest(self):
        h = hashlib.sha256()
        for col in ["id", "stratum", "text"]:
            s = self.frame[col]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                h.update(s.to_numpy().tobytes())
            else:
                # 字符串列（object / pandas 3.x 的 str / Arrow 后端都算）必须按 UTF-8
                # 内容哈希：ndarray.tobytes() 序列化的是对象指针而非内容，跨进程不稳定
                # → 快照哈希/κ 每个进程都不同，缓存永远 miss（草案 7.1、11、23.2）。
                for v in s.to_numpy():
                    h.update(str(v).encode("utf-8"))
                    h.update(b"\x00")
        h.update(self.name.encode("utf-8"))
        return h.hexdigest()

    @property
    def N(self):
        return len(self.frame)

    def strata(self):
        return list(self._strata)

    def stratum_ids(self, h):
        return self._stratum_ids[int(h)]

    def texts(self, ids):
        """给定 id 数组，返回对齐的文本视图列表（顺序与输入一致）。"""
        tmap = self._text_by_id
        return [tmap[int(i)] for i in ids]

    def check_mass_conservation(self, ledger):
        """不变式抽查：每层已暴露计数之和 = 已暴露记录数。"""
        for h in self.strata():
            ids = set(self.stratum_ids(h))
            o = ledger.exposed_ids(h) & ids
            assert ledger.exposed_count(h) == len(o), f"层 {h} 质量丢失"
        return True
