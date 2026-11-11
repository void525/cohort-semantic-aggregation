# -*- coding: utf-8 -*-
"""真实数据加载与冻结（草案 4、13.1、23.1、23.2）。

设计原则：
  - 加载器只负责「读文件 → 精确过滤 → 冻结 → 生成 Snapshot 的 frame」，
    执行核心（账本/置信序列/投影/预算）一行都不用改；
  - 每条记录必须唯一 id、必须属于恰好一个 stratum、必须有 text 视图；
  - 许可与来源信息写在 data_manifests/*.json 里，由 require_data_manifest 强制检查；
  - 本模块**不能也不应该**替代你对数据许可、字段含义与再分发条件的核查。

⚠️ 关于列名：CFPB / WANDS / UCI 的官方文件列名可能随版本变化。
   所有列名都做成参数，并在缺失时给出明确报错，请以你下载到的文件为准。
"""
import hashlib
import json
import os
import numpy as np
import pandas as pd

class DataPreparationError(RuntimeError):
    pass

# ---------------------------------------------------------------------------
# 许可/来源清单（草案 23.1 data_manifests）
# ---------------------------------------------------------------------------
REQUIRED_MANIFEST_FIELDS = ["dataset", "official_url", "license", "downloaded_at",
                            "sha256", "scope"]

def require_data_manifest(path):
    """读取并校验数据清单；字段不全直接拒绝运行（不得「不知道许可就用」）。"""
    if not os.path.exists(path):
        raise DataPreparationError(
            f"缺少数据清单：{path}\n"
            "请按草案 23.1 填写 JSON，至少包含字段：\n"
            f"  {', '.join(REQUIRED_MANIFEST_FIELDS)}\n"
            '示例：{"dataset":"CFPB Consumer Complaint Database",'
            '"official_url":"https://www.consumerfinance.gov/data-research/consumer-complaints/",'
            '"license":"public (CFPB official terms)", '
            '"downloaded_at":"2026-10-01T10:00:00+08:00", '
            '"sha256":"<文件哈希>", "scope":"2024-01-01..2024-12-31, non-empty narrative"}')
    with open(path, "r", encoding="utf-8") as f:
        man = json.load(f)
    missing = [k for k in REQUIRED_MANIFEST_FIELDS if not man.get(k)]
    if missing:
        raise DataPreparationError(f"数据清单 {path} 缺少字段：{missing}")
    return man

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# 元数据原子 → 物理层（草案 5.3：H <= 8，原子内指示器恒定）
# ---------------------------------------------------------------------------
def build_strata(frame, atom_cols, numeric_bins=None, max_categories=4, seed=0):
    """把若干元数据原子组合成物理层编号。

    atom_cols:      用于分层的列名列表（如 CFPB 的 ["Product","half"]）
    numeric_bins:   {列名: 分位层数}，数值列按分位数分箱；其余按类别取前 max_categories
    返回 (stratum Series, mapping dict)，mapping 记录每个原子取值 → 层编号。
    """
    numeric_bins = numeric_bins or {}
    keys = []
    mapping = {}
    for col in atom_cols:
        s = frame[col]
        if col in numeric_bins:
            q = int(numeric_bins[col])
            binned = pd.qcut(s, q=q, duplicates="drop").astype(str)
        else:
            vc = s.astype(str).value_counts()
            keep = list(vc.index[:max_categories])
            binned = s.astype(str).where(s.astype(str).isin(keep), "__other__")
        keys.append(binned.rename(col))
        mapping[col] = {v: i for i, v in enumerate(sorted(binned.unique()))}
    combo = pd.concat(keys, axis=1).astype(str).agg("|".join, axis=1)
    codes = pd.factorize(combo, sort=True)[0]
    mapping["__combo_to_stratum__"] = {c: int(i) for i, c in enumerate(sorted(combo.unique()))}
    return pd.Series(codes, index=frame.index, name="stratum"), mapping

def _finalize(frame, name, id_col, text_col, extra_cols=()):
    """统一成 Snapshot 需要的三列 + 元数据列，并做冻结前检查。"""
    out = pd.DataFrame({
        "id": frame[id_col].astype("int64"),
        "stratum": frame["stratum"].astype("int64"),
        "text": frame[text_col].astype(str),
    })
    for c in extra_cols:
        if c in frame.columns:
            out[c] = frame[c].values
    if not out["id"].is_unique:
        raise DataPreparationError(f"{name}: id 不唯一，请检查去重主键与拼接规则")
    if (out["text"].str.len() == 0).any():
        raise DataPreparationError(f"{name}: 存在空文本视图")
    return out

# ---------------------------------------------------------------------------
# CFPB（草案 4.1）
# ---------------------------------------------------------------------------
def load_cfpb(csv_path, manifest_path, start="2024-01-01", end="2024-12-31",
              max_records=50000, seed=0,
              id_col="Complaint ID", date_col="Date received",
              narrative_col="Consumer complaint narrative",
              product_col="Product", issue_col="Issue"):
    """载入 CFPB 投诉数据并冻结为 Snapshot 的 frame。

    - 只保留指定接收时间范围、且叙事非空的记录；
    - 按 id 去重（一条投诉 = 一个分析单位，草案 4.1）；
    - 用加盐哈希做稳定抽样到 max_records（不复制记录补齐）；
    - 分层：产品(前 4 类) × 半年度。
    """
    man = require_data_manifest(manifest_path)
    usecols = [id_col, date_col, narrative_col, product_col, issue_col]
    df = pd.read_csv(csv_path, usecols=lambda c: c in usecols, low_memory=False)
    missing = [c for c in (id_col, date_col, narrative_col, product_col) if c not in df.columns]
    if missing:
        raise DataPreparationError(
            f"CFPB 文件缺少列 {missing}；实际列：{list(df.columns)[:15]}… "
            "请用 --id-col/--date-col/--narrative-col/--product-col 指定正确列名。")
    df = df.dropna(subset=[id_col, date_col, narrative_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[(df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))]
    df[narrative_col] = df[narrative_col].astype(str).str.strip()
    df = df[df[narrative_col].str.len() > 0]
    df = df.drop_duplicates(subset=[id_col])
    n_qualified = len(df)
    if n_qualified == 0:
        raise DataPreparationError("过滤后没有合格记录：请检查时间范围与列名")
    # 稳定加盐哈希抽样（不依赖标签、可复现）
    h = df[id_col].astype("int64").apply(
        lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())
    df = df.assign(_h=h).sort_values("_h").head(int(max_records)).drop(columns=["_h"])
    df = df.sort_values(id_col).reset_index(drop=True)
    df["half"] = df[date_col].dt.year.astype(str) + "H" + \
        ((df[date_col].dt.month > 6).astype(int) + 1).astype(str)
    strata, mapping = build_strata(df, [product_col, "half"], max_categories=4)
    df["stratum"] = strata.values
    frame = _finalize(df, "CFPB", id_col, narrative_col,
                      extra_cols=[product_col, "half", issue_col])
    info = {"dataset": "CFPB", "manifest": man, "n_qualified": n_qualified,
            "n_used": len(frame), "strata_mapping": mapping,
            "note": "CFPB 投诉库不是市场消费者体验的统计样本，禁止外推（草案 4.1）"}
    return frame, info

# ---------------------------------------------------------------------------
# WANDS（草案 4.2）
# ---------------------------------------------------------------------------
def _read_tabular(path, label=""):
    """WANDS 官方三个 csv 实为制表符分隔（product.csv 含未转义内容，逗号解析会直接报错）。

    自动在制表符与逗号之间选择「能产生更多列」的分隔符；两种都失败才报错。
    这只是数据入口的格式探测，不触碰统计口径与标签隔离（草案 4.2）。
    """
    best, errors = None, {}
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep)
        except (pd.errors.ParserError, ValueError) as e:
            errors[repr(sep)] = str(e).split("\n")[0][:140]
            continue
        if best is None or len(df.columns) > len(best[1].columns):
            best = (sep, df)
    if best is None:
        raise DataPreparationError(
            f"{label} 无法解析（制表符与逗号都试过）：{errors}；"
            "请确认文件完整下载、没有截断。")
    if best[0] != ",":
        print(f"  [格式] {label}: 使用制表符分隔（{len(best[1].columns)} 列，"
              f"{len(best[1])} 行）；用逗号解析会失败，这是官方文件的已知格式。")
    return best[1]


def load_wands(data_dir, manifest_path, max_intents=120, max_pairs_per_intent=512,
               seed=0, label_col="label", query_id_col="query_id",
               product_id_col="product_id", query_text_col="query",
               product_name_col="product_name", rating_col="average_rating",
               dev_intents=24):
    """载入 Wayfair WANDS，构造「已标注有限候选池」并冻结。

    重要（草案 4.2）：研究目标只涉及【已标注的有限候选池】；
    未标注的意图—商品对既不是负例，也不属于该任务总体。
    测试时 label 列完全隐藏：这里把 gold 单独返回，绝不写进 frame。

    返回 (frame, gold, info)：
      frame: id/stratum/text + intent 列（**不含 label**）
      gold : {id: np.array([p1=相关, p2=Exact])}  —— 只允许评测器访问
    """
    man = require_data_manifest(manifest_path)
    cache_pkl = f"/tmp/wands_frozen_{max_intents}_{max_pairs_per_intent}_{seed}.pkl"
    if os.path.exists(cache_pkl):
        import pickle
        try:
            with open(cache_pkl, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    prod = _read_tabular(os.path.join(data_dir, "product.csv"), "product.csv")
    qry = _read_tabular(os.path.join(data_dir, "query.csv"), "query.csv")
    lab = _read_tabular(os.path.join(data_dir, "label.csv"), "label.csv")
    for name, d, cols in (("product.csv", prod, [product_id_col, product_name_col, rating_col]),
                          ("query.csv", qry, [query_id_col, query_text_col]),
                          ("label.csv", lab, [query_id_col, product_id_col, label_col])):
        miss = [c for c in cols if c not in d.columns]
        if miss:
            raise DataPreparationError(
                f"{name} 缺少列 {miss}；实际列：{list(d.columns)}…")
    lab = lab.dropna(subset=[label_col]).drop_duplicates(subset=[query_id_col, product_id_col])
    # 稳定哈希挑选意图与候选池（与标签值无关）
    intents = sorted(lab[query_id_col].unique())
    rng = np.random.default_rng(seed)
    if len(intents) > max_intents:
        intents = sorted(rng.choice(intents, size=max_intents, replace=False).tolist())
    lab = lab[lab[query_id_col].isin(intents)]
    kept = []
    for q, g in lab.groupby(query_id_col):
        g = g.assign(_h=g[product_id_col].apply(
            lambda v: hashlib.sha256(f"{seed}:{q}:{v}".encode()).hexdigest()))
        kept.append(g.sort_values("_h").head(max_pairs_per_intent))
    lab = pd.concat(kept, ignore_index=True)
    m = lab.merge(qry[[query_id_col, query_text_col]], on=query_id_col, how="left") \
           .merge(prod[[product_id_col, product_name_col, rating_col]],
                  on=product_id_col, how="left")
    m["id"] = (m[query_id_col].astype("int64") * 1_000_000
               + m[product_id_col].astype("int64"))
    # 文本视图：查询意图 + 商品信息
    m["text"] = ("搜索意图：" + m[query_text_col].fillna("").astype(str) +
                 "\n商品：" + m[product_name_col].fillna("").astype(str))
    # 分层：评分区间（缺失单列，不默认成 0，草案 4.2）
    r = pd.to_numeric(m[rating_col], errors="coerce")
    m["rating_bucket"] = np.where(r.isna(), "missing",
                                  np.where(r >= 4, ">=4", "<4"))
    m["intent_group"] = pd.factorize(m[query_id_col], sort=True)[0] % 2
    strata, mapping = build_strata(m, ["rating_bucket", "intent_group"], max_categories=8)
    m["stratum"] = strata.values
    gold = {}
    for _, row in m.iterrows():
        lb = str(row[label_col])
        exact = 1 if lb.lower().startswith("exact") else 0
        relevant = 1 if lb.lower().startswith(("exact", "partial")) else 0
        gold[int(row["id"])] = np.array([relevant, exact], dtype=np.int8)
    frame = _finalize(m, "WANDS", "id", "text",
                      extra_cols=[query_id_col, product_id_col, "rating_bucket"])
    frame["intent_id"] = m[query_id_col].astype("int64").values  # 缓存键必须含意图
    info = {"dataset": "WANDS", "manifest": man, "n_intents": int(m[query_id_col].nunique()),
            "n_pairs": len(frame), "gold_size": len(gold),
            "strata_mapping": mapping, "dev_intents": dev_intents,
            "note": "聚合问题是本研究从原始标注构造的派生任务（草案 4.2）"}
    try:
        import pickle
        with open(cache_pkl, "wb") as f:
            pickle.dump((frame, gold, info), f)
    except Exception:
        pass
    return frame, gold, info

# ---------------------------------------------------------------------------
# UCI Sentiment Labelled Sentences（草案 4.3）
# ---------------------------------------------------------------------------
def load_uci(data_dir, manifest_path, files=("amazon_cells_labelled.txt",
                                             "yelp_labelled.txt")):
    """载入 UCI 情感句子（现成标签），作为情感原语与独立标签误差的补充检查。"""
    man = require_data_manifest(manifest_path)
    rows = []
    for fn in files:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            raise DataPreparationError(f"缺少文件：{path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                txt, lab = line.rsplit("\t", 1)
                rows.append((fn, txt.strip(), int(lab)))
    if not rows:
        raise DataPreparationError("没有解析到任何句子，请检查文件格式（应为 句子<TAB>标签）")
    df = pd.DataFrame(rows, columns=["source", "text", "label"])
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["id"] = np.arange(len(df), dtype=np.int64)
    # 需要可控群体时只能生成明确标注为 synthetic 的哈希分组（草案 4.3）
    df["synth_group"] = (df["id"] % 2).astype(str)
    strata, mapping = build_strata(df, ["source", "synth_group"], max_categories=4)
    df["stratum"] = strata.values
    gold = {int(i): np.array([int(v)], dtype=np.int8)
            for i, v in zip(df["id"], df["label"])}
    frame = _finalize(df, "UCI", "id", "text", extra_cols=["source", "synth_group"])
    info = {"dataset": "UCI", "manifest": man, "n_sentences": len(frame),
            "strata_mapping": mapping,
            "note": "synth_group 为 synthetic 哈希分组，不是实际市场细分（草案 4.3）"}
    return frame, gold, info


# ---------------------------------------------------------------------------
# UCI DrugLib Pharmacovigilance Corpus (CC BY 4.0, N = 4,143)
# ---------------------------------------------------------------------------
def _classify_therapeutic_area(cond_str):
    s = str(cond_str).lower()
    if any(w in s for w in ("depress", "anxiet", "insomnia", "panic", "migraine",
                            "bipolar", "adhd", "attention", "seizure", "sleep", "stress")):
        return "psych_neuro"
    if any(w in s for w in ("pain", "arthritis", "fibromyalgia", "muscle", "gout",
                            "back", "spasm", "sciatica", "inflammation", "osteo")):
        return "pain_inflam"
    if any(w in s for w in ("infection", "acne", "sinus", "bronchitis", "skin",
                            "rosacea", "herpes", "fungal", "pneumonia", "rash", "eczema")):
        return "infect_derm"
    return "other_chronic"


def load_druglib(data_dir, manifest_path):
    """载入 UCI DrugLib (CC BY 4.0) 全部 4,143 条患者临床用药评价并冻结快照。

    - Stage 1 默认分层 (H=6)：rating_tier ({<=4, 5..7, >=8}) × drug_freq ({head40, tail})；
    - 保留 therapeutic_area (H=4) 元数据列用于 Stage 2 兼容重切分；
    - held-out gold 保存患者自评结构化标签 (Moderate+ sideEffects, Considerably+ effectiveness)，
      仅用于算子质量诊断，绝不进入 frame 文本视图。
    """
    man = require_data_manifest(manifest_path)
    p1 = os.path.join(data_dir, "drugLibTrain_raw.tsv")
    p2 = os.path.join(data_dir, "drugLibTest_raw.tsv")
    for p in (p1, p2):
        if not os.path.exists(p):
            raise DataPreparationError(f"缺少 DrugLib 文件：{p}")
    d1 = pd.read_csv(p1, sep="\t")
    d2 = pd.read_csv(p2, sep="\t")
    dl = pd.concat([d1, d2], ignore_index=True)
    id_col = dl.columns[0]
    dl["id"] = pd.to_numeric(dl[id_col], errors="raise").astype("int64")
    dl = dl.drop_duplicates(subset=["id"]).sort_values("id").reset_index(drop=True)

    dl["text"] = (
        "Drug: " + dl["urlDrugName"].fillna("unknown").astype(str).str.strip()
        + " | Condition: " + dl["condition"].fillna("unknown").astype(str).str.strip()
        + "\nBenefits: " + dl["benefitsReview"].fillna("none").astype(str).str.strip()
        + "\nSide effects: " + dl["sideEffectsReview"].fillna("none").astype(str).str.strip()
        + "\nComments: " + dl["commentsReview"].fillna("none").astype(str).str.strip()
    )
    r = pd.to_numeric(dl["rating"], errors="coerce").fillna(5.0)
    dl["rating_tier"] = np.where(r <= 4, "<=4", np.where(r <= 7, "5..7", ">=8"))
    top40_drugs = set(dl["urlDrugName"].fillna("unknown").astype(str).value_counts().head(40).index)
    dl["drug_freq"] = np.where(dl["urlDrugName"].fillna("unknown").astype(str).isin(top40_drugs),
                               "head", "tail")
    dl["therapeutic_area"] = dl["condition"].apply(_classify_therapeutic_area)

    strata, mapping = build_strata(dl, ["rating_tier", "drug_freq"], max_categories=8)
    dl["stratum"] = strata.values

    gold = {}
    mod_plus = {"Moderate Side Effects", "Severe Side Effects", "Extremely Severe Side Effects"}
    eff_plus = {"Highly Effective", "Considerably Effective"}
    for _, row in dl.iterrows():
        g_ae = 1 if str(row["sideEffects"]) in mod_plus else 0
        g_ef = 1 if str(row["effectiveness"]) in eff_plus else 0
        gold[int(row["id"])] = np.array([g_ae, g_ef], dtype=np.int8)

    frame = _finalize(dl, "DrugLib", "id", "text",
                      extra_cols=["urlDrugName", "condition", "rating_tier", "drug_freq", "therapeutic_area"])
    info = {"dataset": "UCI DrugLib", "manifest": man, "n_records": len(frame),
            "strata_mapping": mapping}
    return frame, gold, info
