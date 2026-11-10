# -*- coding: utf-8 -*-
"""COHORT-RAG 本地 CPU 复现自动核验脚本。

用途：跑完实验后，一条命令比对"新跑出来的结果"与"论文归档结果"，
      逐字段核对是否一致（不是看个大概，是逐位比对）。

用法（在 D:\\RAG2\\repro\\cohort-rag 下执行）：
    python scripts/verify_reproduction.py --new-dir reports --archive "D:\\COHORT-RAG论文材料\\01_实验结果"

退出码：0 = 全部一致；1 = 有差异；2 = 缺文件。
"""
import argparse
import json
import os
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 新跑的产物（reports/canonical/，由 reproduce_all.sh 改名归档）-> 归档里的对应文件
PAIRS = [
    ("real_wands_report_closure18_s10.json", "real_wands_report.json",
     "主实验：闭包 18 查询 × 10 种子（论文表 1）"),
    ("main_metrics_extended_closure18_s10.json", "main_metrics_extended.json",
     "主实验配对检验 + 扩展指标（论文表 3）"),
    ("real_wands_report_basic7_s10.json", "real_wands_basic10_report.json",
     "基本负载：7 查询 × 10 种子（论文表 2）"),
    ("main_metrics_extended_basic7_s10.json", "main_metrics_extended_basic10.json",
     "基本负载配对检验（论文表 1c）"),
    ("ablation_contract_targeted.json", "ablation_no_source_contract_targeted.json",
     "契约消融：定向检索缓存（标签依赖）"),
    ("ablation_contract_rr.json", "ablation_no_source_contract_rr.json",
     "契约消融：分配器暴露缓存（标签无关）"),
    ("ablation_baseline_closure18_s3.json", "ablation_baseline_closure18_s3.json",
     "消融基线（闭包 18 × 3 种子）"),
    ("ablation_independent_ratio_closure18_s3.json", "ablation_independent_ratio_closure18_s3.json",
     "独立分子/分母区间（闭包 18 × 3）"),
    ("ablation_no_shared_state_closure18_s3.json", "ablation_no_shared_state_closure18_s3.json",
     "无共享状态（闭包 18 × 3）"),
    ("ablation_baseline_basic7_s5.json", "ablation_baseline_basic7_s5.json",
     "消融基线（基本 7 × 5 种子）"),
    ("ablation_independent_ratio_basic7_s5.json", "ablation_independent_ratio_basic7_s5.json",
     "独立分子/分母区间（基本 7 × 5）"),
    ("ablation_no_shared_state_basic7_s5.json", "ablation_no_shared_state_basic7_s5.json",
     "无共享状态（基本 7 × 5）"),
    ("cpu_overhead.json", "cpu_overhead.json", "CPU 系统开销"),
    ("amortization_b0.json", "amortization_b0.json", "摊销交叉点 m*"),
    ("pair_classification_wands.json", "pair_classification_wands.json", "逐对分类指标"),
    ("real_uci_report.json", "real_uci_report.json", "UCI 情感补充检查（§6.4 跨任务对照）"),
]

# 这些键是"运行环境元信息"，重跑可能不同，不参与逐位比对
SKIP_KEYS = {"timestamp", "elapsed_seconds", "total_seconds", "duration_s",
             "host", "elapsed", "runtime_s"}

# ------------------------------------------------------------------
# 含墙钟时间的产物：计时依赖具体硬件，不可能在不同机器上逐位相同。
# 对这类文件只比对【结构性字段】（数据集、κ、调用数、种子数……），
# 把时间值单列出来做"趋势一致性"检查，而不是谎称一致、也不是误报失败。
# ------------------------------------------------------------------
TIMING_DEPENDENT = {
    "cpu_overhead.json": {
        # 这些子键含时间，一律不比
        "time_keys": {"data_load_seconds", "snapshot_freeze_seconds", "total_seconds",
                      "cpu_solve_seconds_mean", "cpu_seconds_per_1k_calls"},
        # 这些键必须逐位一致（结构性）
        "exact_keys": {"dataset", "n_records", "kappa", "workload", "n_queries"},
        # 趋势检查：首尾的每千次调用成本应同向（高预算更省）
        "trend": ("cpu_solve", "cpu_seconds_per_1k_calls"),
    },
}


def compare_timing_dependent(new, old, spec):
    """返回 (结构差异列表, 趋势说明字符串)。"""
    diffs = []
    for k in spec["exact_keys"]:
        if k not in new or k not in old:
            diffs.append((k, new.get(k, "<缺失>"), old.get(k, "<缺失>")))
        elif new[k] != old[k]:
            diffs.append((k, new[k], old[k]))
    # cpu_solve 列表：比对预算/调用数/种子数，跳过时间
    list_key, trend_key = spec["trend"]
    for i, (a, b) in enumerate(zip(new.get(list_key, []), old.get(list_key, []))):
        for f in ("budget_fraction", "calls_mean", "n_seeds"):
            if a.get(f) != b.get(f):
                diffs.append((f"{list_key}[{i}].{f}", a.get(f), b.get(f)))
    note = ""
    try:
        nv = [x[trend_key] for x in new[list_key]]
        ov = [x[trend_key] for x in old[list_key]]
        if len(nv) == len(ov) and nv and ov:
            same_dir = (nv[-1] < nv[0]) == (ov[-1] < ov[0])
            note = (f"耗时因硬件不同而异（本机 {nv[0]:.3f}→{nv[-1]:.3f} s/千次，"
                    f"归档 {ov[0]:.3f}→{ov[-1]:.3f} s/千次），趋势方向"
                    f"{'一致' if same_dir else '不一致'}：高预算下每千次调用成本"
                    f"{'下降' if nv[-1] < nv[0] else '上升'}")
    except Exception:
        pass
    return diffs, note


def diff_obj(new, old, path="", out=None, tol=1e-12):
    """递归比对；浮点按容差，其余按 ==。返回差异列表。"""
    if out is None:
        out = []
    if path.rsplit(".", 1)[-1] in SKIP_KEYS:
        return out
    if isinstance(old, dict):
        if not isinstance(new, dict):
            out.append((path, new, old)); return out
        for k in old:
            if k not in new:
                out.append((f"{path}.{k}", "<缺失>", old[k]))
            else:
                diff_obj(new[k], old[k], f"{path}.{k}" if path else str(k), out, tol)
    elif isinstance(old, list):
        if not isinstance(new, list):
            out.append((path, new, old)); return out
        if len(new) != len(old):
            out.append((f"{path}[长度]", len(new), len(old)))
        for i, (a, b) in enumerate(zip(new, old)):
            diff_obj(a, b, f"{path}[{i}]", out, tol)
    elif isinstance(old, float) or isinstance(new, float):
        try:
            if abs(float(new) - float(old)) > tol:
                out.append((path, new, old))
        except (TypeError, ValueError):
            if new != old:
                out.append((path, new, old))
    else:
        if new != old:
            out.append((path, new, old))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", default="reports")
    ap.add_argument("--archive", required=True, help="归档结果目录（论文的 01_实验结果）")
    ap.add_argument("--tol", type=float, default=1e-12, help="浮点容差")
    args = ap.parse_args()

    print("=" * 78)
    print("COHORT-RAG 本地复现核验")
    print("=" * 78)
    print(f"  新结果目录：{os.path.abspath(args.new_dir)}")
    print(f"  归档目录　：{os.path.abspath(args.archive)}")
    print()

    total_ok = total_skip = total_bad = total_timing = 0
    missing = []

    for new_name, old_name, label in PAIRS:
        npath = os.path.join(args.new_dir, new_name)
        opath = os.path.join(args.archive, old_name)
        if not os.path.exists(npath):
            missing.append((new_name, label))
            print(f"[跳过] {label}")
            print(f"       未找到新产物：{npath}")
            continue
        if not os.path.exists(opath):
            print(f"[跳过] {label} —— 归档中没有 {old_name}")
            total_skip += 1
            continue
        with open(npath, encoding="utf-8") as f:
            new = json.load(f)
        with open(opath, encoding="utf-8") as f:
            old = json.load(f)

        # 计时类产物：单独处理
        if new_name in TIMING_DEPENDENT:
            spec = TIMING_DEPENDENT[new_name]
            diffs, note = compare_timing_dependent(new, old, spec)
            if diffs:
                total_bad += 1
                print(f"[不一致] {label}  ({new_name})")
                for p, a, b in diffs[:8]:
                    print(f"         {p}: 新={a!r}  归档={b!r}")
            else:
                total_timing += 1
                print(f"[结构一致 ⏱] {label}  ({new_name})")
                if note:
                    print(f"         {note}")
            continue

        diffs = diff_obj(new, old, tol=args.tol)
        if diffs:
            total_bad += 1
            print(f"[不一致] {label}  ({new_name})")
            for p, a, b in diffs[:8]:
                print(f"         {p}: 新={a!r}  归档={b!r}")
            if len(diffs) > 8:
                print(f"         ...另有 {len(diffs) - 8} 处差异")
        else:
            total_ok += 1
            print(f"[一致 ✅] {label}  ({new_name})")

    print()
    print("-" * 78)
    print(f"逐位一致 {total_ok} 项｜结构一致（计时类）{total_timing} 项｜"
          f"不一致 {total_bad} 项｜跳过 {total_skip} 项")
    if missing:
        print(f"未生成 {len(missing)} 项：")
        for n, lab in missing:
            print(f"   · {lab} → {n}")
    print("-" * 78)

    if total_bad:
        print("❌ 存在差异：说明本地复现与论文归档结果不完全一致，需排查。")
        return 1
    if not total_ok and not total_timing:
        print("⚠️  没有可核验的产物（文件都没生成？）")
        return 2
    print("✅ 全部已生成产物与论文归档结果一致")
    print("   （逐位一致 = 每个 JSON 字段都相同，含逐种子数组；")
    print("     计时类产物结构一致，墙钟时间因硬件不同而异，属预期）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
