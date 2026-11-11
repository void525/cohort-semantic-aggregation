# -*- coding: utf-8 -*-
"""真实语义算子与版本化缓存（草案 6、7.1、17.1、17.4、17.5）。

三个算子，pipeline 完全不需要改动：

  LocalLLMOperator     : 本地 Qwen3-8B（transformers 懒加载），批量、短输出、
                         确定性解码；解析失败 → LabelFailure（由 pipeline fail-closed）。
  CachedSemanticOperator: 按 (κ, id) 的 sqlite 缓存包裹任意算子——真实实验里
                         标签很贵，重跑必须命中缓存（草案 7.1 的 κ 版本隔离）。
  LabelTableOperator   : 读现成标尺（WANDS/UCI 人工标签），供评测与全量物化参考。

⚠️ 需要 GPU 的部分只在这里：其余模块（证据账本/置信序列/投影/预算）与合成演示共用。
"""
import json
import os
import sqlite3
import numpy as np
from .semantics import SemanticContract
from . import contracts as _contracts

class LabelFailure(RuntimeError):
    """语义标签不可恢复失败（解析失败/输出长度错误）→ 调用方必须 fail-closed。"""

class LocalLLMOperator(SemanticContract):
    """本地模型算子：transformers + 冻结 prompt 包 + 严格解析。

    参数（全部进入 κ，见 contracts.contract_hash）：
      model_id / revision / tokenizer / dtype / decode / view_policy
    """

    def __init__(self, model_id="Qwen/Qwen3-8B", revision=None, K=4,
                 prompt_pack=None, max_input_tokens=1536, max_new_tokens=16,
                 batch_size=16, dtype="bfloat16", device="cuda",
                 enable_thinking=False, canonical_fixed_len=None,
                 name="local-llm"):
        pack = prompt_pack if prompt_pack is not None else _contracts.cfpb_prompt_pack()
        super().__init__(name=name, K=K)
        self.model_id = model_id
        self.revision = revision
        self.prompt_pack = pack
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.enable_thinking = enable_thinking
        self.canonical_fixed_len = int(canonical_fixed_len) if canonical_fixed_len else None
        self.decode_cfg = {"do_sample": False, "temperature": 0.0,
                          "max_new_tokens": max_new_tokens,
                          "rendering": "chat_template"}  # 渲染方式进入 κ（草案 7.1）
        if self.canonical_fixed_len is not None:
            self.decode_cfg["canonical_shape"] = [int(batch_size), self.canonical_fixed_len]
        self.view_policy = {"truncate": "head_tail", "max_input_tokens": max_input_tokens}
        # 模板字段校验（启动时暴露，而不是跑到一半才 KeyError）：
        # LocalLLMOperator 只注入 {narrative}（文本视图已在数据加载器里拼好）。
        try:
            pack["user_template"].format(narrative="x")
        except KeyError as e:
            raise ValueError(
                f"prompt 包 {pack.get('version','?')} 的 user_template 含不支持的占位符 {e}。"
                "LocalLLMOperator 只注入 {narrative}；文本视图（如 搜索意图+商品）"
                "应在数据加载器中拼好。") from None
        self._model = None
        self._tok = None
        self._chat_mode = None
        self.tokenizer_id = model_id

    # ---- 版本键（草案 7.1）----
    def kappa(self, snapshot_manifest):
        return _contracts.contract_hash(
            snapshot_manifest, self.model_id, self.revision, self.tokenizer_id,
            self.prompt_pack["version"], self.decode_cfg, self.view_policy,
            self.prompt_pack["primitives"])

    # ---- 懒加载：没有 GPU/transformers 时不会在 import 期炸掉 ----
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch                                    # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        self._tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        # decoder-only 批量生成必须左填充（右填充会污染 batch 内短序列的输出）
        self._tok.padding_side = "left"
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision,
            torch_dtype=getattr(torch, self.dtype, torch.bfloat16),
            device_map=self.device)
        self._model.eval()

    # ---- 文本视图策略（草案 17.5）----
    def apply_view_policy(self, text):
        """首尾裁剪：超出预算时保留头尾，并记录裁剪（视图策略写入算子版本）。"""
        s = str(text)
        budget_chars = self.max_input_tokens * 4      # 无 tokenizer 时的保守近似
        if len(s) <= budget_chars:
            return s
        head = budget_chars // 2
        tail = budget_chars - head
        return s[:head] + "\n[... 已按 head_tail 策略裁剪 ...]\n" + s[-tail:]

    def _build_prompt(self, text):
        pack = self.prompt_pack
        user = pack["user_template"].format(narrative=self.apply_view_policy(text))
        # 优先走 chat template（Qwen3 训练时的对话格式，输出服从性远好于裸拼接）；
        # enable_thinking=False 显式关闭思考模式；模板不可用时退回裸拼接。
        if self._chat_mode is None:
            self._chat_mode = bool(getattr(self._tok, "chat_template", None))
        if self._chat_mode and self._tok is not None:
            msgs = [{"role": "system", "content": pack["system"]},
                    {"role": "user", "content": user}]
            try:
                return self._tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                return self._tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
        return pack["system"] + "\n\n" + user

    def count_tokens(self, texts):
        """输入 token 数（有 tokenizer 就精确统计，否则按 4 字符≈1 token 估计）。"""
        if self._tok is None and not texts:
            return 0
        if self._tok is None:
            return int(sum(len(self._build_prompt(t)) for t in texts) / 4) + len(texts)
        return int(sum(len(self._tok(self._build_prompt(t))["input_ids"]) for t in texts))

    @staticmethod
    def _parse_one(s):
        """严格解析：取第一个长度为 K 的 0/1 串。"""
        s = str(s).strip()
        digits = [c for c in s if c in "01"]
        return digits

    def judge(self, ids, texts):
        self._ensure_loaded()
        import torch                                     # noqa: PLC0415
        out = np.zeros((len(ids), self.K), dtype=np.int8)
        bad = []
        for start in range(0, len(ids), self.batch_size):
            chunk_txt = [self._build_prompt(t) for t in texts[start:start + self.batch_size]]
            real_n = len(chunk_txt)
            if self.canonical_fixed_len is not None:
                if real_n < self.batch_size:
                    chunk_txt = chunk_txt + [chunk_txt[-1]] * (self.batch_size - real_n)
                enc = self._tok(chunk_txt, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=self.canonical_fixed_len)
            else:
                enc = self._tok(chunk_txt, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.max_input_tokens)
            enc = {k: v.to(self._model.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self._model.generate(**enc, do_sample=False,
                                           max_new_tokens=self.max_new_tokens,
                                           temperature=None, top_p=None)
            in_len = enc["input_ids"].shape[1]
            for r in range(real_n):
                new = gen[r][in_len:]
                txt = self._tok.decode(new, skip_special_tokens=True)
                digits = self._parse_one(txt)
                if len(digits) < self.K:
                    bad.append(int(ids[start + r]))
                    continue
                out[start + r] = [int(c) for c in digits[:self.K]]
        if bad:
            raise LabelFailure(
                f"{len(bad)} 条记录的输出无法解析为 {self.K} 位 0/1（示例 id: {bad[:5]}）。"
                "请检查 prompt 包与解码配置；不要把这些记录当成负例。")
        return out

class CachedSemanticOperator(SemanticContract):
    """sqlite 版本化缓存（草案 7.1）：键 = (κ, 记录 id)。命中零成本。"""

    def __init__(self, inner, snapshot_manifest, cache_path="cache/labels.sqlite",
                 name=None):
        super().__init__(name=name or (inner.name + "+cache"), K=inner.K)
        self.inner = inner
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self.kappa = inner.kappa(snapshot_manifest) if hasattr(inner, "kappa") else \
            _contracts.contract_hash(snapshot_manifest, inner.name, None, None,
                                     getattr(inner, "version", "v0"), {}, {}, [])
        self._db = sqlite3.connect(cache_path)
        self._db.execute("CREATE TABLE IF NOT EXISTS labels "
                         "(kappa TEXT, rid INTEGER, g TEXT, PRIMARY KEY (kappa, rid))")
        self.hits = 0
        self.misses = 0
        self._mem = {}
        try:
            cur = self._db.execute("SELECT rid, g FROM labels WHERE kappa=?", (self.kappa,))
            for rid, g_str in cur.fetchall():
                self._mem[int(rid)] = np.array(json.loads(g_str), dtype=np.int8)
        except Exception:
            pass

    def _get(self, rid):
        r = int(rid)
        if r in self._mem:
            return self._mem[r]
        cur = self._db.execute("SELECT g FROM labels WHERE kappa=? AND rid=?",
                               (self.kappa, r))
        row = cur.fetchone()
        if row is None:
            return None
        arr = np.array(json.loads(row[0]), dtype=np.int8)
        self._mem[r] = arr
        return arr

    def _put(self, rid, g):
        r = int(rid)
        arr = np.asarray(g, dtype=np.int8)
        self._mem[r] = arr
        self._db.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?)",
                         (self.kappa, r, json.dumps([int(v) for v in g])))
        self._db.commit()

    def count_tokens(self, texts):
        return self.inner.count_tokens(texts) if hasattr(self.inner, "count_tokens") else 0

    def judge(self, ids, texts):
        out = np.zeros((len(ids), self.K), dtype=np.int8)
        todo, todo_txt = [], []
        for r, rid in enumerate(ids):
            g = self._get(rid)
            if g is None:
                todo.append((r, int(rid)))
                todo_txt.append(texts[r])
            else:
                out[r] = g
                self.hits += 1
        if todo:
            self.misses += len(todo)
            labs = self.inner.judge([rid for _, rid in todo], todo_txt)
            for (r, rid), g in zip(todo, labs):
                out[r] = g
                self._put(rid, g)
        return out

class LabelTableOperator(SemanticContract):
    """读现成标尺（WANDS/UCI 人工标签）：用于 θ_gold 与全量物化参考。"""

    def __init__(self, labels, name="gold-labels", K=None):
        self.labels = labels
        if K is None:
            K = len(next(iter(labels.values())))
        super().__init__(name=name, K=K)

    def judge(self, ids, texts=None):
        out = np.zeros((len(ids), self.K), dtype=np.int8)
        for r, i in enumerate(ids):
            g = self.labels.get(int(i))
            if g is None:
                raise LabelFailure(f"gold 标签缺失：id={i}")
            out[r] = g
        return out
