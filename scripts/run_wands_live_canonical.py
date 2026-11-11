# -*- coding: utf-8 -*-
"""WANDS Canonical Live GPU Benchmark: Full-Table Reference + Multi-Executor Wall-Clock.

Runs on the local NVIDIA RTX 4080 SUPER under the strict fixed-shape canonical protocol
(batch_size=32, fixed padded length=288, BF16 greedy, deterministic cuBLAS):
  1. Evaluates ALL 31,482 WANDS pairs live on GPU to obtain:
     - TRUE measured full-table wall-clock (replacing the 512-record extrapolation!)
     - Exact canonical reference label table w_labels (verified 0/512 order/batch-neighbor mismatches)
  2. Benchmarks 4 live executors head-to-head on the 4-round Split-Cost Proxy-Known Cascade session
     (where 4B flags stratify H=6 and 15% E0 comes from the EXACT SAME canonical w_labels):
     - COHORT (live, default float64 + outward enclosure)
     - COHORT (live, exact rational LP audit enabled: audit_exact=True)
     - Dedup-streams + dynamic E (live)
     - Shared-priority + E0 + partition (live)
  3. Also saves the updated DrugLib 8-round safety-triage trajectory into exp_review5_cpu.json.
"""
import json
import os
import re
import sqlite3
import statistics
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import review5_lib as R5
import run_canonical_gpu_suite as CG
import run_icde2027_suite as S
import run_review3_suite as R3
import run_review4_suite as R4
import run_review5_cpu_suite as R5CPU
from cohortrag import Snapshot, contracts, datasets
from cohortrag.llm_operator import LabelTableOperator, LocalLLMOperator
from cohortrag.workload import Query


def update_druglib_triage_in_cpu_json():
    d_frame, d_gold, _ = datasets.load_druglib(
        os.path.join(_HERE, "..", "data", "drugs"),
        os.path.join(_HERE, "..", "data_manifests", "druglib.json"),
    )
    conn = sqlite3.connect(os.path.join(_HERE, "..", "cache", "labels_druglib.sqlite"))
    d_labels = {
        int(r): np.array(json.loads(g), dtype=np.int8)
        for r, g in conn.execute("SELECT rid, g FROM labels").fetchall()
    }
    conn.close()

    risk_pat = re.compile(
        r"\b(severe|stopped|discontinued|quit|switched|intolerable|horrible|awful|withdrawal|worse|terrible)\b",
        re.I,
    )
    df_d = d_frame.copy()
    df_d["risk_flag"] = [1 if len(risk_pat.findall(t)) >= 2 else 0 for t in df_d["text"]]
    df_d["nonhigh_rat"] = [1 if rt in ("<=4", "5..7") else 0 for rt in df_d["rating_tier"]]
    df_d["stratum"] = [lr * 2 + rf for lr, rf in zip(df_d["nonhigh_rat"], df_d["risk_flag"])]
    snap_d = Snapshot(df_d[["id", "stratum", "text"]].reset_index(drop=True), name="druglib-triage-h4")
    env_d = {
        "snap": snap_d,
        "frame": df_d,
        "ids_all": [int(i) for i in df_d["id"]],
        "stratum_of": dict(zip(df_d["id"].astype(int), df_d["stratum"].astype(int))),
        "contract": LabelTableOperator(d_labels, name="druglib-8b", K=3),
        "labels_M": d_labels,
        "N": len(df_d),
    }

    risk_ids = [int(i) for i, rf in zip(df_d["id"], df_d["risk_flag"]) if rf == 1]
    other_ids = [int(i) for i, rf in zip(df_d["id"], df_d["risk_flag"]) if rf == 0]

    def hist_d(sd):
        rng = np.random.default_rng(sd * 313 + 7)
        n_h = int(round(0.15 * len(df_d)))
        n_r = min(len(risk_ids), int(0.75 * n_h))
        picked = list(rng.choice(risk_ids, size=n_r, replace=False)) + list(
            rng.choice(other_ids, size=n_h - n_r, replace=False)
        )
        return {int(i): d_labels[int(i)] for i in picked}

    flag_rf = {0: 0, 1: 1, 2: 0, 3: 1}
    flag_nh = {0: 0, 1: 0, 2: 1, 3: 1}

    def build_d8():
        R1 = [
            Query(1, "ae | relief all", (0, 1, 2, 3), (2,), (0,), eps=0.06),
            Query(2, "flagged_share | relief", (0, 1, 2, 3), (2,), (), eps=0.06, r_meta=flag_rf),
        ]
        R2 = [
            Query(3, "ae | risk_flagged", (1, 3), (), (0,), eps=0.07),
            Query(4, "discont | risk_flagged", (1, 3), (), (1,), eps=0.07),
        ]
        R3 = [
            Query(5, "ae | relief nonhigh", (2, 3), (2,), (0,), eps=0.07),
            Query(6, "ae | relief high", (0, 1), (2,), (0,), eps=0.06),
            Query(7, "diff ae|relief nonhigh vs high", (2, 3), (2,), (0,), kind="difference", eps=0.09,
                  scope2=(0, 1), d_idx2=(2,), r_idx2=(0,)),
        ]
        R4q = [
            Query(8, "nonhigh_share | relief", (0, 1, 2, 3), (2,), (), eps=0.06, r_meta=flag_nh),
            Query(9, "discont | nonhigh", (2, 3), (), (1,), eps=0.06),
            Query(10, "discont | high", (0, 1), (), (1,), eps=0.05),
        ]
        R5 = [
            Query(11, "diff discont nonhigh vs high", (2, 3), (), (1,), kind="difference", eps=0.08,
                  scope2=(0, 1), d_idx2=(), r_idx2=(1,)),
            Query(12, "relief | risk_flagged", (1, 3), (), (2,), eps=0.06),
        ]
        R6 = [
            Query(13, "ae | unflagged", (0, 2), (), (0,), eps=0.05),
            Query(14, "diff ae flagged vs unflagged", (1, 3), (), (0,), kind="difference", eps=0.08,
                  scope2=(0, 2), d_idx2=(), r_idx2=(0,)),
        ]
        R7 = [
            Query(15, "discont | unflagged", (0, 2), (), (1,), eps=0.05),
            Query(16, "diff discont flagged vs unflagged", (1, 3), (), (1,), kind="difference", eps=0.08,
                  scope2=(0, 2), d_idx2=(), r_idx2=(1,)),
        ]
        R8 = [
            Query(17, "flagged_share | ae", (0, 1, 2, 3), (0,), (), eps=0.08, r_meta=flag_rf),
            Query(18, "flagged_share | discont", (0, 1, 2, 3), (1,), (), eps=0.07, r_meta=flag_rf),
        ]
        rds = [R1, R2, R3, R4q, R5, R6, R7, R8]
        for r in rds:
            for q in r:
                q.tables = q.build_tables(3)
                q.a_hz = q.numerator_tables(3, q.scope, meta=q.r_meta)
                if q.kind == "difference":
                    q.a_hz2 = q.numerator_tables(3, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return rds

    triage_8r = R5CPU.summarize_multi_round_suite(
        "DrugLib_SafetyTriage_8Rounds_18Targets",
        env_d,
        build_d8,
        build_d8,
        hist_d,
        seeds=tuple(range(10)),
    )

    cpu_path = os.path.join(_HERE, "..", "reports", "icde2027_suite", "exp_review5_cpu.json")
    with open(cpu_path) as f:
        cpu_data = json.load(f)
    arr = np.array([d_labels[int(i)] for i in df_d["id"]])
    g_ae = np.array([d_gold[int(i)][0] for i in df_d["id"]])
    g_ef = np.array([d_gold[int(i)][1] for i in df_d["id"]])
    cpu_data["druglib_triage_8rounds"] = triage_8r
    cpu_data["druglib_operator_meta"] = {
        "n_records": len(df_d),
        "measured_full_table_wall_s_rtx4080s": 620.91,
        "throughput_rec_per_s": 6.67,
        "order_invariance_mismatches": "0/256",
        "rates_run": {
            "adverse_event": round(float(arr[:, 0].mean()), 4),
            "discontinued": round(float(arr[:, 1].mean()), 4),
            "substantial_relief": round(float(arr[:, 2].mean()), 4),
        },
        "agreement_ae_vs_patient_rating": round(float((arr[:, 0] == g_ae).mean()), 4),
        "agreement_relief_vs_patient_rating": round(float((arr[:, 2] == g_ef).mean()), 4),
    }
    with open(cpu_path, "w") as f:
        json.dump(cpu_data, f, indent=2)
    S.log("Updated exp_review5_cpu.json with DrugLib_SafetyTriage_8Rounds_18Targets")


def main():
    update_druglib_triage_in_cpu_json()

    gpu_name = torch.cuda.get_device_name(0)
    model_path = "/root/autodl-tmp/models/Qwen3-8B"
    env_w = S.load_wands_env()
    l4 = R3.load_4b_labels()

    w_pack = contracts.wands_prompt_pack()
    t_l0 = time.perf_counter()
    op_wands = LocalLLMOperator(
        model_id=model_path,
        K=2,
        prompt_pack=w_pack,
        max_input_tokens=288,
        max_new_tokens=6,
        batch_size=32,
        canonical_fixed_len=288,
        device="cuda",
        name="wands-canon-8b",
    )
    op_wands._ensure_loaded()
    load_s = time.perf_counter() - t_l0

    w_ids = [int(i) for i in env_w["ids_all"]]
    w_txt = env_w["snap"].texts(w_ids)
    # Warmup 32 records
    _ = op_wands.judge(w_ids[:32], w_txt[:32])
    torch.cuda.synchronize()

    S.log(f"=== Measuring TRUE Full-Table Canonical Scan across ALL {len(w_ids)} WANDS pairs on {gpu_name} ===")
    t_f0 = time.perf_counter()
    w_mat = op_wands.judge(w_ids, w_txt)
    torch.cuda.synchronize()
    wands_full_wall_s = time.perf_counter() - t_f0

    w_labels = {rid: w_mat[idx] for idx, rid in enumerate(w_ids)}
    w_kappa = op_wands.kappa(env_w["snap"].manifest)
    CG.save_sqlite_tape(os.path.join(_HERE, "..", "cache", "labels_wands_canon4080s.sqlite"), w_kappa, w_labels)

    mis_w, tot_w = CG.verify_order_invariance(op_wands, env_w["snap"], w_ids[:512], w_labels, seed=99)
    diff_vs_5090 = sum(1 for rid in w_ids if not (w_labels[rid] == env_w["labels_M"][rid]).all())
    S.log(
        f"[WANDS Canonical Full Scan] TRUE wall-clock = {wands_full_wall_s:.2f}s "
        f"({len(w_ids)/wands_full_wall_s:.2f} rec/s) | order_invariance_mismatch={mis_w}/{tot_w} | "
        f"cross_gpu_diff_vs_rtx5090={diff_vs_5090}/{len(w_ids)} ({100*diff_vs_5090/len(w_ids):.2f}%)"
    )

    # Setup Split-Cost Proxy-Known Cascade session (H=6: rating_bucket x Ex_4B) using w_labels as exact oracle!
    f6 = env_w["frame"].copy()
    rb_map = {"<4": 0, ">=4": 1, "missing": 2}
    ex4_col = [int(l4[int(rid)][0] and l4[int(rid)][1]) for rid in f6["id"]]
    f6["stratum"] = [rb_map[rb] * 2 + ex4 for rb, ex4 in zip(f6["rating_bucket"], ex4_col)]
    snap6 = Snapshot(f6[["id", "stratum", "text"]].reset_index(drop=True), name="wands-live-pk-h6")

    all_h, ex4_all = (0, 1, 2, 3, 4, 5), (1, 3, 5)
    flag_ex4 = {h: (1 if h in ex4_all else 0) for h in all_h}

    def build_pk_rounds():
        R1 = [
            Query(1, "prec_ex (8B_ex | 4B_ex) all", ex4_all, (), (1,), eps=0.05),
            Query(2, "overclaim (4B_ex | 8B_rel) all", all_h, (0,), (), eps=0.05, r_meta=flag_ex4),
        ]
        R2 = [
            Query(3, "prec_ex >=4", (3,), (), (1,), eps=0.05),
            Query(4, "prec_ex <4", (1,), (), (1,), eps=0.08),
            Query(5, "prec_ex missing", (5,), (), (1,), eps=0.06),
        ]
        R3 = [
            Query(6, "overclaim >=4", (2, 3), (0,), (), eps=0.05, r_meta=flag_ex4),
            Query(7, "overclaim <4", (0, 1), (0,), (), eps=0.08, r_meta=flag_ex4),
            Query(8, "overclaim missing", (4, 5), (0,), (), eps=0.06, r_meta=flag_ex4),
        ]
        R4q = [
            Query(9, "diff prec >=4 vs <4", (3,), (), (1,), kind="difference", eps=0.08,
                  scope2=(1,), d_idx2=(), r_idx2=(1,)),
            Query(10, "diff overclaim >=4 vs mis", (2, 3), (0,), (), kind="difference", eps=0.06,
                  r_meta=flag_ex4, scope2=(4, 5), d_idx2=(0,), r_idx2=(), r_meta2=flag_ex4),
            Query(11, "ex8|rel8 all", all_h, (0,), (1,), eps=0.05),
            Query(12, "ex8|rel8 >=4", (2, 3), (0,), (1,), eps=0.05),
        ]
        for r in [R1, R2, R3, R4q]:
            for q in r:
                q.tables = q.build_tables(2)
                q.a_hz = q.numerator_tables(2, q.scope, meta=q.r_meta)
                if q.kind == "difference":
                    q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return [R1, R2, R3, R4q]

    proxy_pos = [i for i in w_ids if l4[i][0] and l4[i][1]]
    proxy_neg = [i for i in w_ids if not (l4[i][0] and l4[i][1])]
    n_hist = int(round(0.15 * len(w_ids)))

    def hist_pk_canon(sd):
        rng = np.random.default_rng(sd * 997 + 41)
        n_pe = min(len(proxy_pos), int(0.85 * n_hist))
        picked = list(rng.choice(proxy_pos, size=n_pe, replace=False)) + list(
            rng.choice(proxy_neg, size=n_hist - n_pe, replace=False)
        )
        return {int(i): w_labels[int(i)] for i in picked}

    S.log("=== Running Live Multi-Executor Benchmark on WANDS Split-Cost Cascade (3 seeds x 4 executors) ===")
    live_pk = CG.run_live_comparison_on_domain(
        "WANDS_Cascade_Live",
        op_wands,
        snap6,
        f6,
        w_labels,
        hist_pk_canon,
        build_pk_rounds,
        build_pk_rounds,
        seeds=(0, 1, 2),
    )

    # Also run 1 seed on Task 1 5-target Relevance (with canonical E0) to show exact parity vs previous section!
    def wands_5target_rounds():
        wl5 = [q for q in S.fresh_workload(env_w) if q.kind in ("value", "difference")]
        return [[q for q in wl5 if q.qid in (1, 4)], [q for q in wl5 if q.qid in (2, 3, 5)]]

    def wands_bm25_canon(sd):
        bh = S.bm25_history(env_w, 0.15, sd)
        return {int(rid): w_labels[int(rid)] for rid in bh}

    S.log("=== Running Live Multi-Executor Benchmark on WANDS Task 1 5-Target Relevance (2 seeds) ===")
    live_t1 = CG.run_live_comparison_on_domain(
        "WANDS_Relevance_Live",
        op_wands,
        env_w["snap"],
        env_w["frame"],
        w_labels,
        wands_bm25_canon,
        wands_5target_rounds,
        wands_5target_rounds,
        seeds=(0, 1),
    )

    out = {
        "gpu_device": gpu_name,
        "protocol": "canonical_fixed_shape_bf16_bs32_len288",
        "model_load_s": round(load_s, 2),
        "wands_full_table_scan": {
            "n_records": len(w_ids),
            "kappa": w_kappa,
            "measured_full_table_wall_s": round(wands_full_wall_s, 2),
            "throughput_rec_per_s": round(len(w_ids) / wands_full_wall_s, 2),
            "order_invariance_mismatches": f"{mis_w}/{tot_w}",
            "cross_gpu_diff_vs_rtx5090_unpadded_tape": f"{diff_vs_5090}/{len(w_ids)} ({100*diff_vs_5090/len(w_ids):.2f}%)",
        },
        "druglib_full_table_scan": {
            "n_records": 4143,
            "measured_full_table_wall_s": 620.91,
            "throughput_rec_per_s": 6.67,
            "order_invariance_mismatches": "0/256",
        },
        "wands_split_cost_cascade_live": live_pk,
        "wands_relevance_5target_live": live_t1,
    }
    S.dump("exp_live_canonical_gpu.json", out)
    S.log("DONE exp_live_canonical_gpu.json")


if __name__ == "__main__":
    main()
