# -*- coding: utf-8 -*-
"""生成层（草案 11）：结构化声明 + 受限模板 + 证书。

生成器只能选择模板与引用记录；数字与比较运算符由符号渲染器写入，
不允许自由添加数量、比较、因果表述或市场外推（草案 11）。
"""
import json
import hashlib

def render_claim(q, res, snapshot, contract, cfg, risk_ledger):
    """草案 11 的 JSON 输出结构。"""
    return {
        "status": res.status.replace("certified_true", "certified")
                  .replace("certified_false", "certified"),
        "population_id": f"{snapshot.name}#{snapshot.manifest[:16]}",
        "unit": snapshot.unit,
        "semantic_contract": contract.version,
        "claim_ast": {"kind": q.kind, "scope": list(q.scope),
                      "denominator": list(q.d_idx), "numerator": list(q.r_idx),
                      "eps": q.eps, "tau": q.tau},
        "interval": [res.L, res.U] if res.L is not None else None,
        "point": res.point,
        "denominator_interval": [res.minB, res.maxB],
        "risk_scope": "oracle-relative session-level simultaneous coverage",
        "risk_ledger": risk_ledger,
    }

def template_text(q, res):
    """草案 11 推荐模板：数值全部来自已验证 AST。"""
    if res.status == "undefined_denominator":
        return (f"在指定快照与筛选口径内，该条件比例的分母为零："
                f"该查询语义下无定义（undefined_denominator）。")
    if res.status == "execution_error":
        return "执行失败：证据状态不一致，已 fail-closed，不输出数值声明。"
    if res.status in ("certified", "certified_true", "certified_false"):
        if q.kind == "threshold":
            verdict = "成立" if res.status == "certified_true" else "不成立"
            return (f"按版本为 {res.name} 的判定规则，该阈值声明被认证为【{verdict}】"
                    f"（阈值 {q.tau:.2f}，区间 [{res.L:.3f}, {res.U:.3f}]）。")
        if res.status == "certified":
            return (f"在指定快照及筛选口径内，按版本为 {res.name} 的文本判定规则，"
                    f"该条件比例的区间为 [{res.L:.3f}, {res.U:.3f}]。")
    return ("当前证据不足以认证该声明（未决）：继续抽样或输出 "
            "insufficient_denominator 均可，不得输出已定义比例的精确声明。")

def certificate(snapshot, contract, cfg, risk_ledger, results, calls):
    """证书只验证推导/版本/风险分配（草案 11），不声称证明现实商业事实。"""
    payload = {
        "snapshot_manifest": snapshot.manifest,
        "contract": contract.version,
        "risk_ledger": risk_ledger,
        "calls": calls,
        "results": {qid: r.status for qid, r in results.items()},
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"digest": h, "payload": payload}
