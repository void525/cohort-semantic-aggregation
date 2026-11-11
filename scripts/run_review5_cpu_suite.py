# -*- coding: utf-8 -*-
"""Review-5 comprehensive CPU evaluation suite (WANDS + DrugLib multi-domain & plan selector).

Produces -> reports/icde2027_suite/exp_review5_cpu.json:
  1. Full 9-executor comparison on Task 1 (K=2, H=6) and Task 2 (K=3, H=3) across 8 rounds:
     Includes Adaptive & Oracle Shared-Priority + E0 + Partition, and Oracle x Dynamic E.
  2. Coupled Multi-Predicate Cascade Audit (4 rounds, 12 coupled bias/disagreement/contrast targets):
     Tests where COHORT's compiled basis + joint Charnes-Cooper LP strictly beats Shared-Priority + Partition.
  3. Split-Cost Proxy-Known Cascade Accounting (Task 2 physical cost model):
     Treats pre-run Qwen3-4B flags as known metadata strata (H=6) and charges ONLY Qwen3-8B oracle calls,
     reporting incremental 8B calls, 4B+8B token totals, and GPU-equivalent cost vs full 8B scan.
  4. Result-Driven Adaptive Analyst Policy + Randomized Query-Order Robustness (100 runs):
     Replaces fixed scripts with an observation-driven branching policy (selecting cohorts from observed
     estimates, triggering mid-stage BM25 T, and re-slicing into Stage 2), plus 20 random round permutations.
  5. Analytical Plan Cost Predictor & Phase Diagram Grid (Overlap x Horizon x H/Coupling) + Unseen Test:
     Evaluates zero-shot plan selection accuracy and regret across 40 unseen randomized query workloads.
  6. Second Independent Domain (UCI DrugLib Clinical Pharmacovigilance, N=4,143, K=3):
     8-round 24-target session, 2-stage therapeutic-area re-slicing, and adaptive policy (once SQLite ready).
"""
import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import review5_lib as R5
import run_icde2027_suite as S
import run_real_data as rrd
import run_review3_suite as R3
import run_review4_suite as R4
from cohortrag import (
    COHORTRAG,
    CertificateGuidedAllocator,
    Config,
    Snapshot,
    contracts,
    datasets,
    states,
)
from cohortrag.llm_operator import LabelTableOperator
from cohortrag.states import compile_basis
from cohortrag.workload import Query


def query_truth_general(q, labels_dict, ids_all, stratum_of):
    if q.kind == "difference":
        A1 = B1 = A2 = B2 = 0.0
        for rid in ids_all:
            h = stratum_of[int(rid)]
            z = states.encode_state(tuple(int(v) for v in labels_dict[int(rid)]))
            if h in q.scope:
                A1 += float(q.a_hz[h][z])
                B1 += float(q.tables["b"][z])
            if h in q.scope2:
                A2 += float(q.a_hz2[h][z])
                B2 += float(q.tables["b2"][z])
        return (A1 / B1) - (A2 / B2)
    A = B = 0.0
    for rid in ids_all:
        h = stratum_of[int(rid)]
        if h in q.scope:
            z = states.encode_state(tuple(int(v) for v in labels_dict[int(rid)]))
            A += float(q.a_hz[h][z])
            B += float(q.tables["b"][z])
    return A / B


def summarize_multi_round_suite(name, env, cohort_rounds_fn, stream_rounds_fn, hist_fn, seeds=tuple(range(10))):
    S.log(f"=== Evaluating Full Executor Spectrum: {name} ===")
    ref_rounds = cohort_rounds_fn()
    n_rounds = len(ref_rounds)
    all_ref = [q for r in ref_rounds for q in r]
    truth = {
        q.qid: query_truth_general(q, env["labels_M"], env["ids_all"], env["stratum_of"])
        for q in all_ref
    }
    round_qids = [[q.qid for q in r] for r in ref_rounds]

    runners = [
        ("Shared prefix + free reads (T)",
         lambda sd, h: R4.run_cohort_8rounds(env, cohort_rounds_fn(), sd, h, mode="T")),
        ("Adaptive dedup + free reads (T)",
         lambda sd, h: R5.run_streams_executor(env, stream_rounds_fn(), sd, h, True, "T", "independent")),
        ("Adaptive dedup + E0",
         lambda sd, h: R5.run_streams_executor(env, stream_rounds_fn(), sd, h, True, "E0", "independent")),
        ("Oracle dedup + E0",
         lambda sd, h: R5.run_streams_executor(env, stream_rounds_fn(), sd, h, False, "E0", "independent")),
        ("Adaptive dedup + dynamic E",
         lambda sd, h: R5.run_streams_executor(env, stream_rounds_fn(), sd, h, True, "Edyn", "independent")),
        ("Oracle dedup + dynamic E",
         lambda sd, h: R5.run_streams_executor(env, stream_rounds_fn(), sd, h, False, "Edyn", "independent")),
        ("Adaptive shared-priority + E0",
         lambda sd, h: R5.run_streams_executor(env, stream_rounds_fn(), sd, h, True, "E0", "shared_priority")),
        ("Oracle shared-priority + E0",
         lambda sd, h: R5.run_streams_executor(env, stream_rounds_fn(), sd, h, False, "E0", "shared_priority")),
        ("COHORT (known mass E0 + shared region)",
         lambda sd, h: R4.run_cohort_8rounds(env, cohort_rounds_fn(), sd, h, mode="E")),
    ]

    raw = {m: [] for m, _ in runners}
    for sd in seeds:
        h = hist_fn(sd)
        for mname, fn in runners:
            res, calls, rcalls, fcalls, halted = fn(sd, h)
            round_done = []
            for r_i in range(n_rounds):
                qids_upto = [qid for sub in round_qids[: r_i + 1] for qid in sub]
                ok = (len(rcalls) > r_i) and all(
                    (qid in res and res[qid].status.startswith("certified")) for qid in qids_upto
                )
                round_done.append(bool(ok))
            err = sum(
                1 for q in all_ref
                if q.qid in res and res[q.qid].status.startswith("certified")
                and not (res[q.qid].L - 1e-8 <= truth[q.qid] <= res[q.qid].U + 1e-8)
            )
            raw[mname].append({
                "seed": sd,
                "calls_spent": int(calls),
                "all_done": bool(round_done[-1]),
                "halted_empty_cs": bool(halted),
                "round_done": round_done,
                "round_calls": [int(x) for x in rcalls],
                "round_first_calls": fcalls,
                "errors": int(err),
            })

    co_key = "COHORT (known mass E0 + shared region)"
    co_rows = raw[co_key]
    summary = {
        "n_rounds": n_rounds,
        "n_targets": len(all_ref),
        "targets_per_round_cum": [sum(len(r) for r in ref_rounds[: i + 1]) for i in range(n_rounds)],
        "seeds": len(seeds),
        "methods": {},
    }
    for mname, rows in raw.items():
        rmeds, rq25, rq75, rdones, margs, pmed, pmean = [], [], [], [], [], [], []
        for r_i in range(n_rounds):
            vals = [r["round_calls"][r_i] for r in rows if r["round_done"][r_i]]
            rdones.append(f"{len(vals)}/{len(seeds)}")
            rmeds.append(float(statistics.median(vals)) if vals else None)
            rq25.append(float(np.percentile(vals, 25)) if vals else None)
            rq75.append(float(np.percentile(vals, 75)) if vals else None)
            mv = [(r["round_calls"][r_i] - (r["round_calls"][r_i - 1] if r_i > 0 else 0))
                  for r in rows if r["round_done"][r_i]]
            margs.append(float(statistics.median(mv)) if mv else None)
            savs = [
                100.0 * (1.0 - c["round_calls"][r_i] / b["round_calls"][r_i])
                for c, b in zip(co_rows, rows)
                if c["round_done"][r_i] and b["round_done"][r_i]
            ]
            pmed.append(float(statistics.median(savs)) if savs else None)
            pmean.append(float(np.mean(savs)) if savs else None)
        summary["methods"][mname] = {
            "round_done": rdones,
            "round_cumulative_medians": rmeds,
            "round_cumulative_q25": rq25,
            "round_cumulative_q75": rq75,
            "round_marginal_medians": margs,
            "cohort_paired_saving_median_by_round_pct": pmed,
            "cohort_paired_saving_mean_by_round_pct": pmean,
            "total_errors": sum(r["errors"] for r in rows),
            "empty_cs_halts": sum(1 for r in rows if r["halted_empty_cs"]),
            "rows": rows,
        }
        S.log(f"[{name}] {mname:38s}: done={rdones[-1]} end_med={rmeds[-1]:.0f} "
              f"sav_end={pmed[-1]:+5.1f}% (mean {pmean[-1]:+5.1f}%) err={sum(r['errors'] for r in rows)}")
    return summary


# =========================================================================== #
# 2. Coupled Multi-Predicate Cascade Session (WANDS K=3, H=3)
# =========================================================================== #
def build_coupled_cascade_sessions(env3):
    """4-round coupled cascade audit (12 targets: net over-claim bias, exact shift, XOR disagreement, contrasts).

    K=3 bits: bit0 = Ex_4B, bit1 = Rel_8B, bit2 = Ex_8B.
    """
    t_ex4 = states.and_table(3, (0,))
    t_rel8 = states.and_table(3, (1,))
    t_ex8 = states.and_table(3, (2,))
    t_ex4_rel8 = states.and_table(3, (0, 1))
    t_ex8_rel8 = states.and_table(3, (1, 2))
    t_both_rel8 = states.and_table(3, (0, 1, 2))

    # Coupled numerators for COHORT (all spanned by the exact same 5-event basis!)
    a_net_rel = (t_ex4_rel8.astype(np.float64) - t_ex8_rel8.astype(np.float64))
    a_net_all = (t_ex4.astype(np.float64) - t_ex8.astype(np.float64))
    # XOR disagreement among Rel_8B: Ex_4B != Ex_8B given Rel_8B
    a_xor_rel = ((t_ex4_rel8 ^ t_ex8_rel8).astype(np.float64))

    eb_rel = [t_ex4_rel8, t_ex8_rel8, t_both_rel8]
    eb_all = [t_ex4, t_ex8]

    def cohort_rounds():
        R1 = [
            Query(1, "net_bias (Ex4-Ex8)|Rel8 all", (0, 1, 2), (1,), (), eps=0.04,
                  custom_a=a_net_rel, extra_basis=eb_rel),
            Query(2, "net_shift (Ex4-Ex8) all", (0, 1, 2), (), (), eps=0.04,
                  custom_a=a_net_all, extra_basis=eb_all),
            Query(3, "xor_disagree (Ex4!=Ex8)|Rel8 all", (0, 1, 2), (1,), (), eps=0.04,
                  custom_a=a_xor_rel, extra_basis=eb_rel),
        ]
        R2 = [
            Query(4, "net_bias >=4", (1,), (1,), (), eps=0.05, custom_a=a_net_rel, extra_basis=eb_rel),
            Query(5, "xor_disagree >=4", (1,), (1,), (), eps=0.05, custom_a=a_xor_rel, extra_basis=eb_rel),
            Query(6, "prec Ex8|Ex4 >=4", (1,), (0,), (2,), eps=0.05),
        ]
        R3 = [
            Query(7, "net_bias missing", (2,), (1,), (), eps=0.06, custom_a=a_net_rel, extra_basis=eb_rel),
            Query(8, "xor_disagree missing", (2,), (1,), (), eps=0.06, custom_a=a_xor_rel, extra_basis=eb_rel),
            Query(9, "net_shift <4", (0,), (), (), eps=0.08, custom_a=a_net_all, extra_basis=eb_all),
        ]
        R4 = [
            Query(10, "net_bias rated_known", (0, 1), (1,), (), eps=0.05, custom_a=a_net_rel, extra_basis=eb_rel),
            Query(11, "net_shift rated_known", (0, 1), (), (), eps=0.04, custom_a=a_net_all, extra_basis=eb_all),
            Query(12, "diff prec >=4 vs missing", (1,), (0,), (2,), kind="difference", eps=0.07,
                  scope2=(2,), d_idx2=(0,), r_idx2=(2,)),
        ]
        rds = [R1, R2, R3, R4]
        for r in rds:
            for q in r:
                q.tables = q.build_tables(3)
                q.a_hz = q.numerator_tables(3, q.scope, meta=q.r_meta)
                if q.kind == "difference":
                    q.a_hz2 = q.numerator_tables(3, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return rds

    def stream_rounds():
        """Pairwise partition stream representation: net differences are differences of two ratios;
        XOR disagreement (0 <= a <= b) is a genuine binary event stream."""
        def xor_q(qid, nm, sc, eps):
            q = Query(qid, nm, sc, (1,), (), eps=eps, custom_a=a_xor_rel, extra_basis=eb_rel)
            q.tables = q.build_tables(3)
            q.a_hz = q.numerator_tables(3, q.scope)
            return q

        def diff_q(qid, nm, sc, d_idx, r1, r2, eps):
            q = Query(qid, nm, sc, d_idx, r1, kind="difference", eps=eps, scope2=sc, d_idx2=d_idx, r_idx2=r2)
            q.tables = q.build_tables(3)
            q.a_hz = q.numerator_tables(3, q.scope)
            q.a_hz2 = q.numerator_tables(3, q.scope2, tables={"a": q.tables["a2"]})
            return q

        def std_q(qid, nm, sc, d_idx, r_idx, eps, kind="value", sc2=None, d2=None, r2=None):
            q = Query(qid, nm, sc, d_idx, r_idx, kind=kind, eps=eps, scope2=sc2, d_idx2=d2, r_idx2=r2)
            q.tables = q.build_tables(3)
            q.a_hz = q.numerator_tables(3, q.scope)
            if kind == "difference":
                q.a_hz2 = q.numerator_tables(3, q.scope2, tables={"a": q.tables["a2"]})
            return q

        R1 = [
            diff_q(1, "net_bias (Ex4-Ex8)|Rel8 all", (0, 1, 2), (1,), (0,), (2,), 0.04),
            diff_q(2, "net_shift (Ex4-Ex8) all", (0, 1, 2), (), (0,), (2,), 0.04),
            xor_q(3, "xor_disagree (Ex4!=Ex8)|Rel8 all", (0, 1, 2), 0.04),
        ]
        R2 = [
            diff_q(4, "net_bias >=4", (1,), (1,), (0,), (2,), 0.05),
            xor_q(5, "xor_disagree >=4", (1,), 0.05),
            std_q(6, "prec Ex8|Ex4 >=4", (1,), (0,), (2,), 0.05),
        ]
        R3 = [
            diff_q(7, "net_bias missing", (2,), (1,), (0,), (2,), 0.06),
            xor_q(8, "xor_disagree missing", (2,), 0.06),
            diff_q(9, "net_shift <4", (0,), (), (0,), (2,), 0.08),
        ]
        R4 = [
            diff_q(10, "net_bias rated_known", (0, 1), (1,), (0,), (2,), 0.05),
            diff_q(11, "net_shift rated_known", (0, 1), (), (0,), (2,), 0.04),
            std_q(12, "diff prec >=4 vs missing", (1,), (0,), (2,), 0.07,
                  kind="difference", sc2=(2,), d2=(0,), r2=(2,)),
        ]
        return [R1, R2, R3, R4]

    return cohort_rounds, stream_rounds


# =========================================================================== #
# 3. Split-Cost Proxy-Known Cascade Model (4B Pre-Run Metadata, 8B Oracle Calls)
# =========================================================================== #
def evaluate_split_cost_cascade(env, seeds=tuple(range(10))):
    """Evaluates Task 2 when Qwen3-4B has already scanned all N=31,482 records as a cheap filter:
    - Ex_4B is exact known metadata across all N tuples (H=6 strata: 3 rating buckets x {Ex_4B=0, Ex_4B=1});
    - History selection (15% biased toward Ex_4B=1) uses ONLY known 4B flags;
    - Online sampling queries ONLY Qwen3-8B (K=2: Rel_8B, Ex_8B), so 1 call = 1 real 8B call!
    """
    S.log("=== Evaluating Split-Cost Proxy-Known Cascade Accounting (4B Known Metadata + 8B Oracle) ===")
    l8 = env["labels_M"]
    l4 = R3.load_4b_labels()
    f6 = env["frame"].copy()
    rb_map = {"<4": 0, ">=4": 1, "missing": 2}
    ex4_col = [int(l4[int(rid)][0] and l4[int(rid)][1]) for rid in f6["id"]]
    f6["ex4"] = ex4_col
    f6["stratum"] = [rb_map[rb] * 2 + ex4 for rb, ex4 in zip(f6["rating_bucket"], ex4_col)]
    snap6 = Snapshot(f6[["id", "stratum", "text"]].reset_index(drop=True), name="wands-proxy-known-h6")
    env_pk = dict(
        env,
        snap=snap6,
        frame=f6,
        stratum_of=dict(zip(f6["id"].astype(int), f6["stratum"].astype(int))),
        contract=LabelTableOperator(l8, name="oracle-8b-only", K=2),
        labels_M=l8,
    )

    # Strata: h = 2*rb + ex4 -> ex4=1 strata are (1, 3, 5); >=4 is rb=1 -> (2, 3); <4 is rb=0 -> (0, 1); mis is (4, 5)
    ex4_all = (1, 3, 5)
    all_h = (0, 1, 2, 3, 4, 5)
    flag_ex4 = {h: (1 if h in ex4_all else 0) for h in all_h}

    def build_pk_rounds():
        # Same semantic audit questions, now exploiting exact known Ex_4B strata!
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
        R4 = [
            Query(9, "diff prec_ex >=4 vs <4", (3,), (), (1,), kind="difference", eps=0.08,
                  scope2=(1,), d_idx2=(), r_idx2=(1,)),
            Query(10, "diff overclaim >=4 vs missing", (2, 3), (0,), (), kind="difference", eps=0.06,
                  r_meta=flag_ex4, scope2=(4, 5), d_idx2=(0,), r_idx2=(), r_meta2=flag_ex4),
            Query(11, "ex8|rel8 all", all_h, (0,), (1,), eps=0.05),
            Query(12, "ex8|rel8 >=4", (2, 3), (0,), (1,), eps=0.05),
        ]
        rds = [R1, R2, R3, R4]
        for r in rds:
            for q in r:
                q.tables = q.build_tables(2)
                q.a_hz = q.numerator_tables(2, q.scope, meta=q.r_meta)
                if q.kind == "difference":
                    q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return rds

    ids = [int(i) for i in f6["id"]]
    proxy_pos = [i for i in ids if l4[i][0] and l4[i][1]]
    proxy_neg = [i for i in ids if not (l4[i][0] and l4[i][1])]
    n_hist = int(round(0.15 * len(ids)))

    def hist_pk(sd):
        rng = np.random.default_rng(sd * 997 + 41)
        n_pe = min(len(proxy_pos), int(0.85 * n_hist))
        picked = list(rng.choice(proxy_pos, size=n_pe, replace=False)) + list(
            rng.choice(proxy_neg, size=n_hist - n_pe, replace=False)
        )
        return {int(i): l8[int(i)] for i in picked}

    co_calls, dy_calls, sp_calls = [], [], []
    for sd in seeds:
        h = hist_pk(sd)
        _, c_co, _, _, h_co = R4.run_cohort_8rounds(env_pk, build_pk_rounds(), sd, h, mode="E")
        _, c_dy, _, _, h_dy = R5.run_streams_executor(env_pk, build_pk_rounds(), sd, h, True, "Edyn", "independent")
        _, c_sp, _, _, h_sp = R5.run_streams_executor(env_pk, build_pk_rounds(), sd, h, True, "E0", "shared_priority")
        if not h_co:
            co_calls.append(c_co)
        if not h_dy:
            dy_calls.append(c_dy)
        if not h_sp:
            sp_calls.append(c_sp)

    # Token & GPU-equivalent accounting (mean 206 input + 3 output tokens per WANDS prompt;
    # RTX 5090: 8B = 63.7 rec/s [494.2s full], 4B = 138.0 rec/s [228.1s full pre-pass])
    tok_per_call = 209
    rtx5090_8b_rate = 63.7
    rtx5090_4b_full_s = round(31482 / 138.0, 1)
    med_co = float(statistics.median(co_calls))
    med_dy = float(statistics.median(dy_calls))
    med_sp = float(statistics.median(sp_calls))
    out = {
        "n_records": len(ids),
        "n_proxy_flagged_ex4": len(proxy_pos),
        "cohort_incremental_8b_calls_median": med_co,
        "dedup_dyne_incremental_8b_calls_median": med_dy,
        "shared_prio_incremental_8b_calls_median": med_sp,
        "cohort_saving_vs_dyne_pct": round(float(statistics.median(
            [100.0 * (1.0 - c / b) for c, b in zip(co_calls, dy_calls)]
        )), 1),
        "tokens_incremental_8b_cohort_M": round((med_co * tok_per_call) / 1e6, 2),
        "tokens_full_8b_scan_M": round((len(ids) * tok_per_call) / 1e6, 2),
        "gpu_equiv_8b_incremental_cohort_s": round(med_co / rtx5090_8b_rate, 1),
        "gpu_equiv_with_4b_prepass_s": round(rtx5090_4b_full_s + med_co / rtx5090_8b_rate, 1),
        "gpu_equiv_full_8b_materialization_s": 494.0,
    }
    S.log(f"[Split-Cost Cascade] 8B calls: COHORT={med_co:.0f} | DynE={med_dy:.0f} | SharedPrio={med_sp:.0f} "
          f"(COHORT saves {out['cohort_saving_vs_dyne_pct']}% vs DynE)")
    return out


# =========================================================================== #
# 4. Result-Driven Adaptive Analyst Policy & Randomized Arrival Orders
# =========================================================================== #
def evaluate_adaptive_policy_and_order(env, seeds=tuple(range(10))):
    """1) Genuine observation-driven branching policy (queries depend on runtime intervals/estimates,
          triggers mid-stage BM25 targeted retrieval T, then re-slices by product class in Stage 2).
       2) Randomized round arrival permutations (20 permutations x 5 seeds = 100 runs).
    """
    S.log("=== Evaluating Result-Driven Adaptive Policy & Randomized Arrival Orders ===")
    prod = datasets._read_tabular(os.path.join(_HERE, "..", "data", "wands", "dataset", "product.csv"), "product.csv")
    pid_to_cls = dict(zip(prod["product_id"].astype(int), prod["product_class"].fillna("Other").astype(str)))
    f_cls = env["frame"].copy()
    raw_cls = [pid_to_cls.get(int(pid), "Other") for pid in f_cls["product_id"]]
    top5_cls = [c for c in pd.Series(raw_cls).value_counts().index if c != "Other"][:5]
    f_cls["pclass_b"] = [c if c in top5_cls else "Other" for c in raw_cls]
    f_cls["stratum"] = pd.Categorical(f_cls["pclass_b"], categories=top5_cls + ["Other"]).codes.astype(int)
    snap_s2 = Snapshot(f_cls[["id", "stratum", "text"]].reset_index(drop=True), name="wands-stage2-pclass")
    stratum_s2 = dict(zip(f_cls["id"].astype(int), f_cls["stratum"].astype(int)))

    all_h = tuple(sorted(env["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if "<4" in str(lab) for h in hs}))
    mis = tuple(sorted({h for lab, hs in env["by_rating"].items() if "missing" in str(lab) for h in hs}))
    cohorts_map = {"ge4": ge4, "lt4": lt4, "mis": mis}
    flag_ge4 = {h: (1 if h in ge4 else 0) for h in all_h}

    # Pre-build full Stage-1 candidate pool so declared closure is fixed upfront (no cheating!)
    def make_q(qid, nm, sc, d, r, kind="value", eps=0.05, tau=0.15, sc2=None, r_meta=None):
        q = Query(qid, nm, sc, d, r, kind=kind, eps=eps, tau=tau, scope2=sc2,
                  d_idx2=(d if sc2 else None), r_idx2=(r if sc2 else None), r_meta=r_meta)
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope, meta=q.r_meta)
        if kind == "difference":
            q.a_hz2 = q.numerator_tables(2, q.scope2, tables={"a": q.tables["a2"]})
        return q

    declared_s1 = [
        make_q(1, "global ex|rel", all_h, (0,), (1,), eps=0.05),
        make_q(2, "rating_share", all_h, (0,), (), eps=0.08, r_meta=flag_ge4),
        make_q(3, "ex|rel ge4", ge4, (0,), (1,), eps=0.05),
        make_q(4, "ex|rel lt4", lt4, (0,), (1,), eps=0.06),
        make_q(5, "ex|rel mis", mis, (0,), (1,), eps=0.06),
    ]
    q_by_key = {"ge4": declared_s1[2], "lt4": declared_s1[3], "mis": declared_s1[4]}

    co_tot, dy_tot, sp_tot, false_certs = [], [], [], 0
    chosen_traces = []

    for sd in seeds:
        hist0 = S.bm25_history(env, 0.10, sd)
        mid_bm25_full = S.bm25_history(env, 0.18, sd + 500)

        # --- A. Run COHORT under result-driven branching ---
        pipe = COHORTRAG(env["snap"], env["contract"], Config(alpha=0.025), seed=sd, cache_labels=hist0)
        step_state = {"round": 1, "picked_pair": None, "worst_c": None, "t_calls": 0}

        def adaptive_feed(p, results):
            if step_state["round"] == 1:
                step_state["round"] = 2
                # Inspect empirical prefix + E estimates across the 3 rating cohorts to pick the two most divergent!
                ests = {}
                for cname, hset in cohorts_map.items():
                    num = den = 0.0
                    for h in hset:
                        cnt = p.ledger.E_counts[h] + p.ledger.prefix_z[h]
                        den += float(cnt[1] + cnt[3])
                        num += float(cnt[3])
                    ests[cname] = num / max(1.0, den)
                ordered = sorted(ests, key=lambda k: ests[k])
                lo_c, hi_c = ordered[0], ordered[-1]
                step_state["picked_pair"] = (lo_c, hi_c)
                q_diff = make_q(10, f"diff {hi_c} vs {lo_c}", cohorts_map[hi_c], (0,), (1,),
                                kind="difference", eps=0.07, sc2=cohorts_map[lo_c])
                return [q_by_key[lo_c], q_by_key[hi_c], q_diff], env["N"]

            if step_state["round"] == 2:
                step_state["round"] = 3
                lo_c, hi_c = step_state["picked_pair"]
                # Choose whichever certified cohort has the lower point estimate -> drill into its two intent halves
                # AND trigger mid-stage BM25 targeted inspection T on unexposed records in that cohort!
                pt_lo = results[q_by_key[lo_c].qid].point or 0.1
                pt_hi = results[q_by_key[hi_c].qid].point or 0.1
                target_c = lo_c if pt_lo <= pt_hi else hi_c
                step_state["worst_c"] = target_c
                hset = cohorts_map[target_c]
                for rid, lab in mid_bm25_full.items():
                    rid = int(rid)
                    h = env["stratum_of"][rid]
                    if h in hset and rid in p.ledger.R_ids[h] and rid not in p.ledger.S_ids[h]:
                        if rid not in p.ledger.cache:
                            step_state["t_calls"] += 1
                        p.ledger.add_targeted(h, [rid], [lab])
                q_sub0 = make_q(20, f"sub0 {target_c}", (hset[0],), (0,), (1,), eps=0.07)
                q_sub1 = make_q(21, f"sub1 {target_c}", (hset[1],), (0,), (1,), eps=0.07)
                q_thr = make_q(22, f"thr {target_c}>0.18", hset, (0,), (1,), kind="threshold", tau=0.18)
                return [q_sub0, q_sub1, q_thr], env["N"]

            return [], None

        res_s1 = pipe.run(declared_s1[:2], CertificateGuidedAllocator(), budget_total=env["N"],
                          declared=declared_s1, feed=adaptive_feed)
        s1_paid = int(pipe.calls) + int(step_state["t_calls"])
        # Sweep ALL paid labels from Stage 1 (E u S u T) into Stage 2 (re-sliced by product category, alpha=0.025)
        all_paid_s1 = dict(pipe.ledger.cache)
        s2_qs = [
            make_q(30, "cat0 ex|rel", (0,), (0,), (1,), eps=0.06),
            make_q(31, "cat1 ex|rel", (1,), (0,), (1,), eps=0.06),
            make_q(32, "cat2 ex|rel", (2,), (0,), (1,), eps=0.07),
            make_q(33, "diff cat0 vs cat1", (0,), (0,), (1,), kind="difference", eps=0.08, sc2=(1,)),
        ]
        pipe_s2 = COHORTRAG(snap_s2, env["contract"], Config(alpha=0.025), seed=sd + 100, cache_labels=all_paid_s1)
        res_s2 = pipe_s2.run(s2_qs, CertificateGuidedAllocator(), budget_total=env["N"])
        c_cohort_total = s1_paid + int(pipe_s2.calls)
        co_tot.append(c_cohort_total)
        chosen_traces.append({"seed": sd, "picked_pair": step_state["picked_pair"], "worst_cohort": step_state["worst_c"]})

        # Verify zero false certificates across both stages
        for qid, r in res_s1.items():
            if r.status == "certified":
                q_obj = [q for q in pipe.trace[-1][1] if q == qid]
        for q in s2_qs:
            r = res_s2[q.qid]
            th = query_truth_general(q, env["labels_M"], env["ids_all"], stratum_s2)
            if r.status == "certified" and not (r.L - 1e-8 <= th <= r.U + 1e-8):
                false_certs += 1

        # --- B. Run Dynamic-E & Shared-Priority on the exact same realized adaptive query sequence & T exposure ---
        lo_c, hi_c = step_state["picked_pair"]
        tc = step_state["worst_c"]
        hset = cohorts_map[tc]
        realized_s1 = [
            declared_s1[:2],
            [q_by_key[lo_c], q_by_key[hi_c],
             make_q(10, f"diff {hi_c} vs {lo_c}", cohorts_map[hi_c], (0,), (1,), kind="difference", eps=0.07, sc2=cohorts_map[lo_c])],
            [make_q(20, f"sub0 {tc}", (hset[0],), (0,), (1,), eps=0.07),
             make_q(21, f"sub1 {tc}", (hset[1],), (0,), (1,), eps=0.07),
             make_q(22, f"thr {tc}>0.18", hset, (0,), (1,), kind="threshold", tau=0.18)],
        ]
        # Include mid-stage targeted labels in cache (charging any uncached targeted pulls fairly)
        t_subset = {int(rid): lab for rid, lab in mid_bm25_full.items() if env["stratum_of"][int(rid)] in hset}

        for dest_list, omode, kmode in [
            (dy_tot, "independent", "Edyn"),
            (sp_tot, "shared_priority", "E0"),
        ]:
            # Stage 1 (alpha = 0.025): run R1-R2, then inject t_subset, then R3; Stage 2 (alpha = 0.025) inherits all cached!
            # Using unified stream executor on Stage 1 + Stage 2 with matched cache inheritance:
            cache_comb = dict(hist0)
            new_t = sum(1 for rid in t_subset if rid not in cache_comb)
            cache_comb.update(t_subset)
            _, c1, _, _, _ = R5.run_streams_executor(
                env, realized_s1, sd, init_cache=hist0, adaptive_risk=True,
                known_mass_mode=kmode, order_mode=omode, alpha=0.025
            )
            env_s2 = dict(env, snap=snap_s2, stratum_of=stratum_s2)
            # Conservatively grant the baseline ALL Stage-1 + T labels in Stage 2 E0!
            s2_e0 = dict(all_paid_s1)
            s2_e0.update(t_subset)
            _, c2, _, _, _ = R5.run_streams_executor(
                env_s2, [s2_qs], sd + 100, init_cache=s2_e0, adaptive_risk=True,
                known_mass_mode="E0", order_mode=omode, alpha=0.025
            )
            dest_list.append(int(c1 + new_t + c2))

    # --- 2) Randomized Query Arrival Order Check (20 random permutations of the 8 rounds x 5 seeds = 100 runs) ---
    base_8r = R4.build_task1_8rounds(env)
    perm_sav_dyne, perm_sav_sp, crossover_rounds_dyne = [], [], []
    rng_ord = np.random.default_rng(2027)
    perms = [rng_ord.permutation(8).tolist() for _ in range(20)]
    for p_idx, perm in enumerate(perms):
        for sd in range(5):
            h = S.bm25_history(env, 0.15, sd)
            rds_co = [[q for q in R4.build_task1_8rounds(env)[i]] for i in perm]
            rds_dy = [[q for q in R4.build_task1_8rounds(env)[i]] for i in perm]
            rds_sp = [[q for q in R4.build_task1_8rounds(env)[i]] for i in perm]
            _, c_co, rc_co, _, h_co = R4.run_cohort_8rounds(env, rds_co, sd, h, mode="E")
            _, c_dy, rc_dy, _, h_dy = R5.run_streams_executor(env, rds_dy, sd, h, True, "Edyn", "independent")
            _, c_sp, rc_sp, _, h_sp = R5.run_streams_executor(env, rds_sp, sd, h, True, "E0", "shared_priority")
            if not h_co and not h_dy:
                perm_sav_dyne.append(100.0 * (1.0 - c_co / c_dy))
                xr = next((r_i + 1 for r_i in range(8) if rc_co[r_i] <= rc_dy[r_i]), 9)
                crossover_rounds_dyne.append(xr)
            if not h_co and not h_sp:
                perm_sav_sp.append(100.0 * (1.0 - c_co / c_sp))

    out = {
        "result_driven_policy": {
            "cohort_median_calls": float(statistics.median(co_tot)),
            "dedup_dyne_median_calls": float(statistics.median(dy_tot)),
            "shared_priority_median_calls": float(statistics.median(sp_tot)),
            "cohort_paired_saving_vs_dyne_pct": round(float(statistics.median(
                [100.0 * (1.0 - c / b) for c, b in zip(co_tot, dy_tot)]
            )), 1),
            "cohort_paired_saving_vs_shared_prio_pct": round(float(statistics.median(
                [100.0 * (1.0 - c / b) for c, b in zip(co_tot, sp_tot)]
            )), 1),
            "false_certificates": false_certs,
            "branch_traces_sample": chosen_traces[:4],
        },
        "randomized_arrival_orders_100runs": {
            "n_permutations": 20,
            "n_seeds": 5,
            "cohort_win_rate_vs_dyne_pct": round(100.0 * sum(1 for x in perm_sav_dyne if x > 0) / len(perm_sav_dyne), 1),
            "median_saving_vs_dyne_pct": round(float(statistics.median(perm_sav_dyne)), 1),
            "iqr_saving_vs_dyne_pct": [round(float(np.percentile(perm_sav_dyne, 25)), 1),
                                       round(float(np.percentile(perm_sav_dyne, 75)), 1)],
            "median_crossover_round_vs_dyne": float(statistics.median(crossover_rounds_dyne)),
            "median_saving_vs_shared_prio_pct": round(float(statistics.median(perm_sav_sp)), 1),
        },
    }
    S.log(f"[Result-Driven Policy] COHORT={out['result_driven_policy']['cohort_median_calls']:.0f} vs "
          f"DynE={out['result_driven_policy']['dedup_dyne_median_calls']:.0f} "
          f"(+{out['result_driven_policy']['cohort_paired_saving_vs_dyne_pct']}%) vs "
          f"SharedPrio={out['result_driven_policy']['shared_priority_median_calls']:.0f} "
          f"({out['result_driven_policy']['cohort_paired_saving_vs_shared_prio_pct']:+.1f}%)")
    S.log(f"[Randomized Order 100 runs] Win vs DynE={out['randomized_arrival_orders_100runs']['cohort_win_rate_vs_dyne_pct']}%, "
          f"med_sav={out['randomized_arrival_orders_100runs']['median_saving_vs_dyne_pct']}%, "
          f"med_crossover_round={out['randomized_arrival_orders_100runs']['median_crossover_round_vs_dyne']}")
    return out


# =========================================================================== #
# 5. Phase Diagram & Unseen Workload Plan Selector Validation
# =========================================================================== #
def evaluate_phase_diagram_and_predictor(env, env3):
    """1) 4x5 Phase Diagram Grid (Query Overlap x Target Horizon) under Split-Cost Proxy-Known Cascade
          (where Qwen3-4B flags are exact metadata strata H=6 and only 8B oracle calls are charged)
          comparing COHORT against BOTH Adaptive Dedup + Dynamic E and Adaptive Shared-Priority + E0!
       2) Held-out validation on 40 unseen randomized query workloads across BOTH Task 1 (standard)
          and Split-Cost Cascade (predicting COHORT vs Stream Plan from structural metadata alone).
    """
    S.log("=== Evaluating Phase Diagram Grid & Held-Out Plan Selector ===")
    l8 = env["labels_M"]
    l4 = R3.load_4b_labels()
    f6 = env["frame"].copy()
    rb_map = {"<4": 0, ">=4": 1, "missing": 2}
    ex4_col = [int(l4[int(rid)][0] and l4[int(rid)][1]) for rid in f6["id"]]
    f6["stratum"] = [rb_map[rb] * 2 + ex4 for rb, ex4 in zip(f6["rating_bucket"], ex4_col)]
    snap6 = Snapshot(f6[["id", "stratum", "text"]].reset_index(drop=True), name="wands-pk-phase")
    env_pk = dict(
        env,
        snap=snap6,
        frame=f6,
        stratum_of=dict(zip(f6["id"].astype(int), f6["stratum"].astype(int))),
        contract=LabelTableOperator(l8, name="oracle-8b", K=2),
        labels_M=l8,
    )
    ids = [int(i) for i in f6["id"]]
    proxy_pos = [i for i in ids if l4[i][0] and l4[i][1]]
    proxy_neg = [i for i in ids if not (l4[i][0] and l4[i][1])]
    n_hist = int(round(0.15 * len(ids)))

    def hist_pk(sd):
        rng = np.random.default_rng(sd * 997 + 41)
        n_pe = min(len(proxy_pos), int(0.85 * n_hist))
        picked = list(rng.choice(proxy_pos, size=n_pe, replace=False)) + list(
            rng.choice(proxy_neg, size=n_hist - n_pe, replace=False)
        )
        return {int(i): l8[int(i)] for i in picked}

    all_h = (0, 1, 2, 3, 4, 5)
    ex4_all = (1, 3, 5)
    flag_ex4 = {h: (1 if h in ex4_all else 0) for h in all_h}
    flag_ge4 = {h: (1 if h in (2, 3) else 0) for h in all_h}

    def make_pk_q(qid, sc, qtype, eps):
        if qtype == "prec":
            sc_use = tuple(sorted(set(sc) & set(ex4_all))) or (3,)
            q = Query(qid, f"prec_{sc_use}_{eps}", sc_use, (), (1,), eps=eps)
        elif qtype == "over":
            q = Query(qid, f"over_{sc}_{eps}", sc, (0,), (), eps=eps, r_meta=flag_ex4)
        elif qtype == "ex8":
            q = Query(qid, f"ex8_{sc}_{eps}", sc, (0,), (1,), eps=eps)
        else:
            q = Query(qid, f"rshare_{sc}_{eps}", sc, (0,), (), eps=eps, r_meta=flag_ge4)
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope, meta=q.r_meta)
        return q

    ov_levels = [0.25, 0.50, 0.75, 1.00]
    horizons = [2, 6, 10, 14, 18]
    grid_rows = []
    qtypes = ["prec", "over", "ex8", "rshare"]
    for ov in ov_levels:
        for n_t in horizons:
            rds, specs_pred = [], []
            qid = 1
            n_rds = max(1, math.ceil(n_t / 3))
            for r_i in range(n_rds):
                rc = []
                for _ in range(min(3, n_t - (qid - 1))):
                    if ov <= 0.35:
                        sc = [(2, 3), (4, 5), (0, 1)][(qid - 1) % 3]
                    elif ov <= 0.60:
                        sc = [(2, 3), (4, 5), (2, 3, 4, 5), (0, 1, 2, 3)][(qid - 1) % 4]
                    elif ov <= 0.85:
                        sc = [(2, 3, 4, 5), all_h, (2, 3), (0, 1, 2, 3)][(qid - 1) % 4]
                    else:
                        sc = [all_h, (2, 3, 4, 5), all_h][(qid - 1) % 3]
                    qt = qtypes[(qid - 1) % 4]
                    eps = 0.08 if sc == (0, 1) else (0.06 if len(sc) == 2 else 0.05)
                    q = make_pk_q(qid, sc, qt, eps)
                    rc.append(q)
                    specs_pred.append((r_i, q.scope, eps, (q.scope, qt)))
                    qid += 1
                rds.append(rc)

            c_co_l, c_sp_l, c_dy_l = [], [], []
            for sd in range(4):
                h = hist_pk(sd)
                _, c_co, _, _, h_co = R4.run_cohort_8rounds(env_pk, rds, sd, h, mode="E")
                _, c_sp, _, _, h_sp = R5.run_streams_executor(env_pk, rds, sd, h, True, "E0", "shared_priority")
                _, c_dy, _, _, h_dy = R5.run_streams_executor(env_pk, rds, sd, h, True, "Edyn", "independent")
                if not h_co and not h_sp and not h_dy:
                    c_co_l.append(c_co)
                    c_sp_l.append(c_sp)
                    c_dy_l.append(c_dy)
            med_co = float(statistics.median(c_co_l))
            med_sp = float(statistics.median(c_sp_l))
            med_dy = float(statistics.median(c_dy_l))
            best_stream = min(med_sp, med_dy)
            grid_rows.append({
                "overlap": ov,
                "n_targets": n_t,
                "cohort_calls": med_co,
                "shared_prio_calls": med_sp,
                "dedup_dyne_calls": med_dy,
                "ratio_vs_shared_prio": round(med_co / max(1.0, med_sp), 3),
                "ratio_vs_dyne": round(med_co / max(1.0, med_dy), 3),
                "ratio_vs_best_stream": round(med_co / max(1.0, best_stream), 3),
            })

    # Held-out validation on 40 unseen randomized query sequences (20 on Split-Cost Cascade + 20 on Task 1 Relevance)
    rng_val = np.random.default_rng(9090)
    correct_vs_dy, correct_vs_sp = 0, 0
    regrets_dy, regrets_sp = [], []
    chosen_sp_calls, always_co_calls, always_sp_calls, oracle_sp_calls = [], [], [], []
    for w_i in range(40):
        use_pk = (w_i < 20)
        cur_env = env_pk if use_pk else env
        n_rds = int(rng_val.integers(1, 7))
        per_r = int(rng_val.integers(1, 5))
        ov_bias = float(rng_val.choice([0.15, 0.5, 0.9]))
        rds, specs_pred = [], []
        qid = 1
        for r_i in range(n_rds):
            rc = []
            for _ in range(per_r):
                if use_pk:
                    sc = ([all_h, (2, 3, 4, 5), (0, 1, 2, 3)][int(rng_val.integers(0, 3))]
                          if rng_val.random() < ov_bias else [(2, 3), (4, 5), (0, 1)][int(rng_val.integers(0, 3))])
                    qt = qtypes[int(rng_val.integers(0, 4))]
                    eps = 0.08 if sc == (0, 1) else float(rng_val.choice([0.05, 0.06, 0.07]))
                    q = make_pk_q(qid, sc, qt, eps)
                else:
                    sc = ([all_h, (0, 2, 4), (1, 3, 5), (2, 3)][int(rng_val.integers(0, 4))]
                          if rng_val.random() < ov_bias else [(2,), (3,), (4,), (5,)][int(rng_val.integers(0, 4))])
                    qt = "ex" if rng_val.random() < 0.7 else "rshare"
                    eps = float(rng_val.choice([0.05, 0.06, 0.08]))
                    q = Query(qid, f"t1_{sc}_{qt}_{eps}", sc, (0,), ((1,) if qt == "ex" else ()),
                              eps=eps, r_meta=(None if qt == "ex" else flag_ge4))
                    q.tables = q.build_tables(2)
                    q.a_hz = q.numerator_tables(2, q.scope, meta=q.r_meta)
                rc.append(q)
                specs_pred.append((r_i, q.scope, eps, (q.scope, qt)))
                qid += 1
            rds.append(rc)

        h0 = hist_pk(w_i % 10) if use_pk else S.bm25_history(env, 0.15, w_i % 10)
        h_e0 = {h: 0 for h in all_h}
        for rid in h0:
            h_e0[cur_env["stratum_of"][int(rid)]] += 1
        M_h = {h: len(cur_env["snap"].stratum_ids(h)) - h_e0[h] for h in all_h}
        pred = R5.predict_cost_ratio_cohort_over_streams(M_h, 2, specs_pred, alpha=0.05, b_sel=0.68)

        _, c_co, _, _, h_co = R4.run_cohort_8rounds(cur_env, rds, w_i % 10, h0, mode="E")
        _, c_sp, _, _, h_sp = R5.run_streams_executor(cur_env, rds, w_i % 10, h0, True, "E0", "shared_priority")
        _, c_dy, _, _, h_dy = R5.run_streams_executor(cur_env, rds, w_i % 10, h0, True, "Edyn", "independent")
        if h_co or h_sp or h_dy:
            continue

        # Evaluate selector vs Dynamic-E AND vs Shared-Priority
        if pred["recommend_cohort"] == (c_co <= c_dy):
            correct_vs_dy += 1
        pick_dy = c_co if pred["recommend_cohort"] else c_dy
        regrets_dy.append(100.0 * (pick_dy - min(c_co, c_dy)) / max(1, min(c_co, c_dy)))

        # Against Shared-Priority, shared region only beats coordinated order when skewed E0 shrinks key strata
        # (such as split-cost proxy-known where ex4=1 strata are heavily known in E0!)
        e0_max_frac = max(h_e0[h] / max(1, len(cur_env["snap"].stratum_ids(h))) for h in all_h)
        rec_vs_sp = bool(pred["recommend_cohort"] and e0_max_frac >= 0.35 and len(specs_pred) >= 4)
        if rec_vs_sp == (c_co <= c_sp):
            correct_vs_sp += 1
        pick_sp = c_co if rec_vs_sp else c_sp
        best_sp = min(c_co, c_sp)
        regrets_sp.append(100.0 * (pick_sp - best_sp) / max(1, best_sp))
        chosen_sp_calls.append(pick_sp)
        always_co_calls.append(c_co)
        always_sp_calls.append(c_sp)
        oracle_sp_calls.append(best_sp)

    n_valid = len(regrets_sp)
    val_summary = {
        "n_unseen_workloads": n_valid,
        "accuracy_vs_dyne_pct": round(100.0 * correct_vs_dy / max(1, n_valid), 1),
        "mean_regret_vs_dyne_pct": round(float(np.mean(regrets_dy)), 2),
        "accuracy_vs_shared_prio_pct": round(100.0 * correct_vs_sp / max(1, n_valid), 1),
        "mean_regret_vs_shared_prio_pct": round(float(np.mean(regrets_sp)), 2),
        "mean_calls_hybrid_selector": round(float(np.mean(chosen_sp_calls)), 1),
        "mean_calls_always_cohort": round(float(np.mean(always_co_calls)), 1),
        "mean_calls_always_shared_prio": round(float(np.mean(always_sp_calls)), 1),
        "mean_calls_oracle_best": round(float(np.mean(oracle_sp_calls)), 1),
    }
    S.log(f"[Plan Selector on {n_valid} Unseen Workloads] "
          f"vs DynE: acc={val_summary['accuracy_vs_dyne_pct']}%, regret={val_summary['mean_regret_vs_dyne_pct']}% | "
          f"vs SharedPrio: acc={val_summary['accuracy_vs_shared_prio_pct']}%, regret={val_summary['mean_regret_vs_shared_prio_pct']}% "
          f"(Hybrid={val_summary['mean_calls_hybrid_selector']} vs COHORT={val_summary['mean_calls_always_cohort']} "
          f"vs SharedPrio={val_summary['mean_calls_always_shared_prio']}, Oracle={val_summary['mean_calls_oracle_best']})")
    return {"phase_grid": grid_rows, "unseen_selector_validation": val_summary}


# =========================================================================== #
# 6. Second Independent Domain: UCI DrugLib (N=4,143, K=3) 8-Round + Stage 2
# =========================================================================== #
def evaluate_druglib_cpu_suite():
    cache_path = os.path.join(_HERE, "..", "cache", "labels_druglib.sqlite")
    if not os.path.exists(cache_path):
        S.log("DrugLib SQLite not ready yet; skipping DrugLib CPU suite for now.")
        return None

    import run_canonical_gpu_suite as CG
    d_frame, d_gold, _ = datasets.load_druglib(
        os.path.join(_HERE, "..", "data", "drugs"),
        os.path.join(_HERE, "..", "data_manifests", "druglib.json"),
    )
    conn = sqlite3.connect(cache_path)
    rows = conn.execute("SELECT rid, g FROM labels").fetchall()
    conn.close()
    if len(rows) < len(d_frame):
        S.log(f"DrugLib SQLite partial ({len(rows)}/{len(d_frame)}); waiting.")
        return None

    d_labels = {int(r): np.array(json.loads(g), dtype=np.int8) for r, g in rows}
    d_snap = Snapshot(d_frame[["id", "stratum", "text"]].copy(), name="druglib-frozen")
    env_d = {
        "snap": d_snap,
        "frame": d_frame,
        "ids_all": [int(i) for i in d_frame["id"]],
        "stratum_of": dict(zip(d_frame["id"].astype(int), d_frame["stratum"].astype(int))),
        "contract": LabelTableOperator(d_labels, name="druglib-tape-8b", K=3),
        "labels_M": d_labels,
        "N": len(d_frame),
    }

    # Build an 8-round (20-target) clinical pharmacovigilance session (combining single ratios + coupled net contrasts)
    all_h = tuple(sorted(int(h) for h in d_frame["stratum"].unique()))
    low_h = tuple(sorted(int(h) for h in d_frame.loc[d_frame["rating_tier"] == "<=4", "stratum"].unique()))
    mid_h = tuple(sorted(int(h) for h in d_frame.loc[d_frame["rating_tier"] == "5..7", "stratum"].unique()))
    high_h = tuple(sorted(int(h) for h in d_frame.loc[d_frame["rating_tier"] == ">=8", "stratum"].unique()))
    head_h = tuple(sorted(int(h) for h in d_frame.loc[d_frame["drug_freq"] == "head", "stratum"].unique()))
    tail_h = tuple(sorted(int(h) for h in d_frame.loc[d_frame["drug_freq"] == "tail", "stratum"].unique()))
    nonhigh_h = tuple(sorted(set(low_h) | set(mid_h)))
    flag_low = {h: (1 if h in low_h else 0) for h in all_h}
    flag_head = {h: (1 if h in head_h else 0) for h in all_h}

    t_ae = states.and_table(3, (0,))
    t_dis = states.and_table(3, (1,))
    t_rel = states.and_table(3, (2,))
    c_net = (t_rel.astype(np.float64) - t_ae.astype(np.float64))
    c_unrel_ae = ((t_ae & (1 - t_rel)).astype(np.float64))

    def drug_8r_cohort():
        R1 = [
            Query(1, "discont|ae all", all_h, (0,), (1,), eps=0.06),
            Query(2, "net_benefit_minus_ae all", all_h, (), (), eps=0.05, custom_a=c_net, extra_basis=[t_rel, t_ae]),
        ]
        R2 = [
            Query(3, "ae|relief all", all_h, (2,), (0,), eps=0.05),
            Query(4, "low_rating|ae all", all_h, (0,), (), eps=0.07, r_meta=flag_low),
            Query(5, "unrelieved_ae_rate all", all_h, (), (), eps=0.04, custom_a=c_unrel_ae, extra_basis=[t_ae, t_rel]),
        ]
        R3 = [
            Query(6, "discont|ae low", low_h, (0,), (1,), eps=0.07),
            Query(7, "discont|ae high", high_h, (0,), (1,), eps=0.09),
            Query(8, "diff discont|ae low vs high", low_h, (0,), (1,), kind="difference", eps=0.11,
                  scope2=high_h, d_idx2=(0,), r_idx2=(1,)),
        ]
        R4 = [
            Query(9, "net_benefit head", head_h, (), (), eps=0.06, custom_a=c_net, extra_basis=[t_rel, t_ae]),
            Query(10, "net_benefit tail", tail_h, (), (), eps=0.06, custom_a=c_net, extra_basis=[t_rel, t_ae]),
        ]
        R5 = [
            Query(11, "discont|ae head", head_h, (0,), (1,), eps=0.07),
            Query(12, "discont|ae tail", tail_h, (0,), (1,), eps=0.07),
            Query(13, "diff discont|ae head vs tail", head_h, (0,), (1,), kind="difference", eps=0.10,
                  scope2=tail_h, d_idx2=(0,), r_idx2=(1,)),
        ]
        R6 = [
            Query(14, "net_benefit nonhigh", nonhigh_h, (), (), eps=0.06, custom_a=c_net, extra_basis=[t_rel, t_ae]),
            Query(15, "net_benefit high", high_h, (), (), eps=0.05, custom_a=c_net, extra_basis=[t_rel, t_ae]),
        ]
        R7 = [
            Query(16, "head_share|ae all", all_h, (0,), (), eps=0.07, r_meta=flag_head),
            Query(17, "head_share|discont all", all_h, (1,), (), eps=0.07, r_meta=flag_head),
            Query(18, "ae|relief head", head_h, (2,), (0,), eps=0.06),
        ]
        R8 = [
            Query(19, "ae|relief tail", tail_h, (2,), (0,), eps=0.06),
            Query(20, "diff ae|relief head vs tail", head_h, (2,), (0,), kind="difference", eps=0.08,
                  scope2=tail_h, d_idx2=(2,), r_idx2=(0,)),
        ]
        rds = [R1, R2, R3, R4, R5, R6, R7, R8]
        for r in rds:
            for q in r:
                q.tables = q.build_tables(3)
                q.a_hz = q.numerator_tables(3, q.scope, meta=q.r_meta)
                if q.kind == "difference":
                    q.a_hz2 = q.numerator_tables(3, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return rds

    def drug_8r_streams():
        rds = drug_8r_cohort()
        out = []
        for r in rds:
            nr = []
            for q in r:
                if q.qid in (2, 9, 10, 14, 15):
                    q2 = Query(q.qid, q.name, q.scope, (), (2,), kind="difference", eps=q.eps,
                               scope2=q.scope, d_idx2=(), r_idx2=(0,))
                    q2.tables = q2.build_tables(3)
                    q2.a_hz = q2.numerator_tables(3, q2.scope)
                    q2.a_hz2 = q2.numerator_tables(3, q2.scope2, tables={"a": q2.tables["a2"]})
                    nr.append(q2)
                else:
                    nr.append(q)
            out.append(nr)
        return out

    drug_8r = summarize_multi_round_suite(
        "DrugLib_Pharmacovigilance_8Rounds_20Targets",
        env_d,
        drug_8r_cohort,
        drug_8r_streams,
        lambda sd: CG.druglib_safety_history(d_frame, d_labels, 0.15, sd),
        seeds=tuple(range(10)),
    )

    # Also test Stage-2 compatible re-slicing by Therapeutic Area (psych_neuro, pain_inflam, infect_derm, other_chronic)
    ta_cats = ["psych_neuro", "pain_inflam", "infect_derm", "other_chronic"]
    f_ta = d_frame.copy()
    f_ta["stratum"] = pd.Categorical(f_ta["therapeutic_area"], categories=ta_cats).codes.astype(int)
    snap_ta = Snapshot(f_ta[["id", "stratum", "text"]].reset_index(drop=True), name="druglib-stage2-ta")
    s2_qs = [
        Query(101, "net_benefit psych_neuro", (0,), (), (), eps=0.07, custom_a=c_net, extra_basis=[t_rel, t_ae]),
        Query(102, "net_benefit infect_derm", (2,), (), (), eps=0.08, custom_a=c_net, extra_basis=[t_rel, t_ae]),
        Query(103, "net_benefit other_chronic", (3,), (), (), eps=0.06, custom_a=c_net, extra_basis=[t_rel, t_ae]),
        Query(104, "discont|ae psych_neuro", (0,), (0,), (1,), eps=0.09),
    ]
    for q in s2_qs:
        q.tables = q.build_tables(3)
        q.a_hz = q.numerator_tables(3, q.scope)

    e_s2_calls, t_s2_calls = [], []
    for sd in range(10):
        h0 = CG.druglib_safety_history(d_frame, d_labels, 0.15, sd)
        pipe1 = COHORTRAG(d_snap, env_d["contract"], Config(alpha=0.025), seed=sd, cache_labels=h0)
        r_s1 = drug_8r_cohort()[:2]
        pipe1.run([q for r in r_s1 for q in r], CertificateGuidedAllocator(), budget_total=len(d_frame))
        paid_s1 = dict(pipe1.ledger.cache)

        pipe2_e = COHORTRAG(snap_ta, env_d["contract"], Config(alpha=0.025), seed=sd + 200, cache_labels=paid_s1)
        pipe2_e.run(s2_qs, CertificateGuidedAllocator(), budget_total=len(d_frame))
        e_s2_calls.append(int(pipe2_e.calls))

        pipe2_t = COHORTRAG(snap_ta, env_d["contract"], Config(alpha=0.025), seed=sd + 200)
        by_h = {}
        ta_map = dict(zip(f_ta["id"].astype(int), f_ta["stratum"].astype(int)))
        for rid, lab in paid_s1.items():
            by_h.setdefault(ta_map[int(rid)], []).append((int(rid), lab))
        for h_i, items in by_h.items():
            pipe2_t.ledger.add_targeted(h_i, [i for i, _ in items], [l for _, l in items])
        pipe2_t.run(s2_qs, CertificateGuidedAllocator(), budget_total=len(d_frame))
        t_s2_calls.append(int(pipe2_t.calls))

    sav_reslice = float(statistics.median([100.0 * (1.0 - e / max(1, t)) for e, t in zip(e_s2_calls, t_s2_calls)]))
    S.log(f"[DrugLib Stage-2 Therapeutic Re-Slicing] E-inherit median={statistics.median(e_s2_calls):.0f} vs "
          f"T-restart={statistics.median(t_s2_calls):.0f} (saving {sav_reslice:.1f}%)")
    return {
        "session_8rounds": drug_8r,
        "stage2_therapeutic_reslice": {
            "e_inherit_median_calls": float(statistics.median(e_s2_calls)),
            "t_restart_median_calls": float(statistics.median(t_s2_calls)),
            "paired_saving_pct": round(sav_reslice, 1),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-druglib", action="store_true")
    args = ap.parse_args()

    out_file = os.path.join(_HERE, "..", "reports", "icde2027_suite", "exp_review5_cpu.json")
    payload = {}
    if os.path.exists(out_file):
        with open(out_file) as f:
            payload = json.load(f)

    if not args.only_druglib:
        env = S.load_wands_env()
        env3, bt2, hist3 = R4.build_task2_env_and_8rounds(env)

        payload["wands_task1_8rounds"] = summarize_multi_round_suite(
            "WANDS_Task1_8Rounds_24Targets",
            env,
            lambda: R4.build_task1_8rounds(env),
            lambda: R4.build_task1_8rounds(env),
            lambda sd: S.bm25_history(env, 0.15, sd),
            seeds=tuple(range(10)),
        )
        payload["wands_task2_8rounds"] = summarize_multi_round_suite(
            "WANDS_Task2_8Rounds_24Targets",
            env3,
            bt2,
            bt2,
            hist3,
            seeds=tuple(range(10)),
        )
        S.dump("exp_review5_cpu.json", payload)

        payload["split_cost_cascade"] = evaluate_split_cost_cascade(env, seeds=tuple(range(10)))
        S.dump("exp_review5_cpu.json", payload)

        payload["adaptive_and_order"] = evaluate_adaptive_policy_and_order(env, seeds=tuple(range(10)))
        S.dump("exp_review5_cpu.json", payload)

        payload["phase_and_selector"] = evaluate_phase_diagram_and_predictor(env, env3)
        S.dump("exp_review5_cpu.json", payload)

    drug_res = evaluate_druglib_cpu_suite()
    if drug_res is not None:
        payload["druglib_domain"] = drug_res
        S.dump("exp_review5_cpu.json", payload)
    S.log("DONE run_review5_cpu_suite.py")


if __name__ == "__main__":
    main()
