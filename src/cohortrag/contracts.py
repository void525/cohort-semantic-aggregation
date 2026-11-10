# -*- coding: utf-8 -*-
"""语义契约：原语定义、固定 prompt 包、以及缓存键版本哈希（草案 4、7.1、17.4）。

要点（草案 7.1）：
  - 多个原语打包进一个固定 prompt 是降低成本的工程选择，但整个包必须【整体版本化】；
  - 缓存键 κ 必须包含：数据版本、记录 id、文本视图哈希、模型与 revision、
    tokenizer、prompt 包版本、解码配置、文本视图策略；
  - 增减一个问题可能改变模型对其它问题的输出，不能假设 prompt 语义等价。
"""
import hashlib
import json

# ---------------------------------------------------------------------------
# CFPB 的四个固定语义原语（草案 4.1）。原语测的是「文本明确陈述」，不是世界事实。
# ---------------------------------------------------------------------------
CFPB_PRIMITIVES = [
    {"key": "g1", "name": "数字渠道使用困难",
     "rule": "叙事明确描述在应用、网站、登录或数字账户操作上的困难",
     "forbidden": "仅有线上提交渠道本身并不等于数字使用困难"},
    {"key": "g2", "name": "客服沟通困难",
     "rule": "叙事明确描述联系支持人员、获得有效回应或沟通方面的困难",
     "forbidden": "不把所有负面投诉都判为沟通困难"},
    {"key": "g3", "name": "重复联系或多次转接",
     "rule": "叙事明确描述重复联系或多次转接",
     "forbidden": "不从情绪强度推断联系次数"},
    {"key": "g4", "name": "结束时尚未解决",
     "rule": "叙事明确表示相关问题在叙述结束时尚未解决",
     "forbidden": "不把字段缺失或没有结尾说明当作未解决"},
]

# WANDS 的相关性契约（草案 4.2）：把三分类映射成两个二值原语，便于联合状态编码。
WANDS_PRIMITIVES = [
    {"key": "p1", "name": "相关（Exact 或 Partial）",
     "rule": "该商品与搜索意图相关，即人工标注为 Exact 或 Partial",
     "forbidden": "不要仅凭商品名称字面匹配就判为相关"},
    {"key": "p2", "name": "完全匹配（Exact）",
     "rule": "该商品是搜索意图的完全匹配，即人工标注为 Exact",
     "forbidden": "不要把 Partial 判成 Exact"},
]

PROMPT_PACK_VERSION = "cfpb-primitives-v2"   # v2: 渲染改为 chat template（κ 变更）
WANDS_PROMPT_PACK_VERSION = "wands-relevance-v3"
UCI_PROMPT_PACK_VERSION = "uci-sentiment-v1"

def cfpb_prompt_pack():
    """固定 prompt 包：4 个原语一次性打包（整体版本化）。"""
    lines = ["你在对公开投诉叙事做严格的有界标注。只判断文本【明确写出】了什么。"]
    for p in CFPB_PRIMITIVES:
        lines.append(f"- {p['key']}（{p['name']}）：{p['rule']}。注意：{p['forbidden']}。")
    system = "\n".join(lines)
    user = ("叙事：\n<<<\n{narrative}\n>>>\n\n"
            "直接输出答案：一个 4 位字符串，依次对应 g1 g2 g3 g4，1=是，0=否。\n"
            "禁止举例、禁止解释、禁止输出标点或其它任何字符，"
            "你的回答必须恰好是 4 个字符。")
    return {"version": PROMPT_PACK_VERSION, "primitives": CFPB_PRIMITIVES,
            "system": system, "user_template": user,
            "output_contract": "4 位 0/1 字符串（长度必须为 4）"}

def wands_prompt_pack():
    """WANDS 相关性判定 prompt 包。

    文本视图由 load_wands 拼好（"搜索意图：…\\n商品：…"），所以这里只用 {narrative}
    占位符，与 LocalLLMOperator 的统一注入接口一致（v2 修复：v1 误用了
    {query}/{product} 两个占位符，与算子的注入约定不匹配，构建 prompt 即 KeyError）。
    """
    lines = ["你在判断商品与搜索意图的相关性。只判断给定文本能支持的相关性等级。"]
    for p in WANDS_PRIMITIVES:
        lines.append(f"- {p['key']}（{p['name']}）：{p['rule']}。注意：{p['forbidden']}。")
    system = "\n".join(lines)
    user = ("下面给出一个搜索意图和一件商品的文本，请判断二者的相关性。\n"
            "<<<\n{narrative}\n>>>\n\n"
            "直接输出答案：一个 2 位字符串，依次对应 p1 p2，1=是，0=否。\n"
            "禁止举例、禁止解释、禁止输出标点或其它任何字符，"
            "你的回答必须恰好是 2 个字符。")
    return {"version": WANDS_PROMPT_PACK_VERSION, "primitives": WANDS_PRIMITIVES,
            "system": system, "user_template": user,
            "output_contract": "2 位 0/1 字符串（长度必须为 2）"}

UCI_PRIMITIVES = [
    {"key": "s1", "name": "正面情感",
     "rule": "句子明确表达满意、推荐或正面评价",
     "forbidden": "不把中性陈述、纯产品描述或疑问判为正面"},
]


def uci_prompt_pack():
    """UCI 情感原语 prompt 包（草案 4.3）：单原语 s1=正面情感。

    UCI 的现成 0/1 标签只用于测量【独立标签误差】：模型判断 vs 人工标注。
    不得把它当成市场细分或因果结论。
    """
    lines = ["你在判断句子的情感倾向。只判断文本明确表达的情感。"]
    for p in UCI_PRIMITIVES:
        lines.append(f"- {p['key']}（{p['name']}）：{p['rule']}。注意：{p['forbidden']}。")
    system = "\n".join(lines)
    user = ("句子：\n<<<\n{narrative}\n>>>\n\n"
            "直接输出答案：一个 1 位字符串，1=正面，0=非正面。\n"
            "禁止举例、禁止解释、禁止输出标点或其它任何字符。")
    return {"version": UCI_PROMPT_PACK_VERSION, "primitives": UCI_PRIMITIVES,
            "system": system, "user_template": user,
            "output_contract": "1 位 0/1 字符串（长度必须为 1）"}


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def contract_hash(snapshot_manifest, model_id, model_revision, tokenizer_id,
                  prompt_pack_version, decode_cfg, view_policy, primitives):
    """κ：缓存键版本哈希（草案 7.1）。

    同一 κ 下的标签可安全复用；任一版本要素变化都会得到新 κ，
    从而自动隔离旧标签（避免「跨不兼容版本复用证书/缓存」）。
    """
    payload = {
        "snapshot": snapshot_manifest,
        "model": model_id,
        "model_revision": model_revision,
        "tokenizer": tokenizer_id,
        "prompt_pack": prompt_pack_version,
        "decode": decode_cfg,
        "view_policy": view_policy,
        "primitives": [p["key"] for p in primitives],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

def record_cache_key(kappa, record_id, text):
    """单条记录的缓存键：κ + 记录 id + 文本视图哈希（草案 7.1 的 κ 公式）。"""
    th = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]
    return f"{kappa[:16]}:{int(record_id)}:{th}"
