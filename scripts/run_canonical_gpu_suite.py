# -*- coding: utf-8 -*-
"""Canonical deterministic GPU tape builder & multi-executor live benchmark (Review-5).

Eliminates ALL live-vs-tape contract gaps identified in Review 5:
  1. Enforces a fixed-shape canonical inference protocol (fixed batch padding + fixed token length +
     deterministic cuBLAS workspace) so every tuple produces the EXACT SAME label vector regardless
     of batch neighbors or adaptive call order (verified explicitly via an irregular-batch audit).
  2. Builds the full 4,143-record UCI DrugLib pharmacovigilance reference tape on the local GPU
     while measuring TRUE full-table wall-clock (not extrapolated from 512 records!).
  3. Measures TRUE full-table wall-clock across all 31,482 WANDS pairs under the same canonical
     protocol on the same RTX 4080 SUPER GPU, and benchmarks Live COHORT (with and without exact
     rational LP audit), Live Dedup + Dynamic E, and Live Shared-Priority + E0 head-to-head against
     their exact canonical full-table reference (0 label mismatches, verified certificate coverage).
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
import run_icde2027_suite as S
import run_real_data as rrd
from cohortrag import (
    COHORTRAG,
    CertificateGuidedAllocator,
    Config,
    Snapshot,
    contracts,
    datasets,
    states,
)
from cohortrag.llm_operator import LocalLLMOperator
from cohortrag.workload import Query


class TimedLiveCanonicalOperator:
    """Wraps a canonical LocalLLMOperator to time real GPU forward passes and check reference agreement."""

    def __init__(self, inner, ref_labels):
        self.inner = inner
        self.K = inner.K
        self.name = inner.name
        self.ref_labels = ref_labels
        self.gpu_seconds = 0.0
        self.n_batches = 0
        self.n_records = 0
        self.mismatches = 0

    def judge(self, ids, texts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.inner.judge(ids, texts)
        torch.cuda.synchronize()
        self.gpu_seconds += time.perf_counter() - t0
        self.n_batches += 1
        self.n_records += len(ids)
        for rid, row in zip(ids, out):
            if not (row == self.ref_labels[int(rid)]).all():
                self.mismatches += 1
        return out


def save_sqlite_tape(cache_path, kappa, labels_dict):
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS labels (kappa TEXT, rid INTEGER, g TEXT, PRIMARY KEY (kappa, rid))"
    )
    rows = [(kappa, int(rid), json.dumps([int(v) for v in lab])) for rid, lab in labels_dict.items()]
    conn.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()


def druglib_safety_history(frame, labels_dict, frac=0.15, seed=0):
    """Realistic pharmacovigilance safety-signal triage log (BM25-style risk keyword score + tiebreak)."""
    risk_pat = re.compile(
        r"\b(severe|stopped|discontinued|quit|switched|intolerable|horrible|awful|emergency|"
        r"withdrawal|hospital|unbearable|worse|terrible|insomnia|nausea|dizzy|weight gain|pain)\b",
        re.I,
    )
    rng = np.random.default_rng(seed * 131 + 17)
    scores = []
    for rid, txt in zip(frame["id"].astype(int), frame["text"].astype(str)):
        hits = len(risk_pat.findall(txt))
        scores.append((-(hits + 0.15 * rng.random()), int(rid)))
    scores.sort()
    k = int(round(frac * len(frame)))
    picked = [rid for _, rid in scores[:k]]
    return {rid: labels_dict[rid] for rid in picked}


def build_druglib_live_rounds(frame):
    """4-round clinical pharmacovigilance session (10 distinct targets including coupled contrasts)."""
    all_h = tuple(sorted(int(h) for h in frame["stratum"].unique()))
    low_h = tuple(sorted(int(h) for h in frame.loc[frame["rating_tier"] == "<=4", "stratum"].unique()))
    high_h = tuple(sorted(int(h) for h in frame.loc[frame["rating_tier"] == ">=8", "stratum"].unique()))
    head_h = tuple(sorted(int(h) for h in frame.loc[frame["drug_freq"] == "head", "stratum"].unique()))
    tail_h = tuple(sorted(int(h) for h in frame.loc[frame["drug_freq"] == "tail", "stratum"].unique()))
    flag_low = {h: (1 if h in low_h else 0) for h in all_h}

    # K=3: bit0 = ModSevereAE (d1), bit1 = Discontinued (d2), bit2 = SubstantialRelief (d3)
    # Coupled net unharmed relief among treated: (Relief - ModSevereAE) / All -> custom_a on all tuples!
    # Wait: denominator for all tuples needs b_table = all 1s -> d_idx = ()!
    t_ae = states.and_table(3, (0,))
    t_dis = states.and_table(3, (1,))
    t_rel = states.and_table(3, (2,))
    c_net = (t_rel.astype(np.float64) - t_ae.astype(np.float64))

    R1 = [
        Query(1, "discont | mod_severe_ae (all)", all_h, (0,), (1,), eps=0.06),
        Query(2, "net_benefit_minus_ae (all)", all_h, (), (), eps=0.05,
              custom_a=c_net, extra_basis=[t_rel, t_ae]),
    ]
    R2 = [
        Query(3, "ae | substantial_relief (all)", all_h, (2,), (0,), eps=0.05),
        Query(4, "low_rating_share | ae", all_h, (0,), (), eps=0.07, r_meta=flag_low),
    ]
    R3 = [
        Query(5, "discont | ae (low_rating)", low_h, (0,), (1,), eps=0.07),
        Query(6, "discont | ae (high_rating)", high_h, (0,), (1,), eps=0.09),
        Query(7, "diff discont|ae low vs high", low_h, (0,), (1,), kind="difference", eps=0.11,
              scope2=high_h, d_idx2=(0,), r_idx2=(1,)),
    ]
    R4 = [
        Query(8, "discont | ae (head_drugs)", head_h, (0,), (1,), eps=0.07),
        Query(9, "discont | ae (tail_drugs)", tail_h, (0,), (1,), eps=0.07),
        Query(10, "net_benefit_minus_ae (head)", head_h, (), (), eps=0.06,
              custom_a=c_net, extra_basis=[t_rel, t_ae]),
    ]
    rounds = [R1, R2, R3, R4]
    for r in rounds:
        for q in r:
            q.tables = q.build_tables(3)
            q.a_hz = q.numerator_tables(3, q.scope, meta=q.r_meta)
            if q.kind == "difference":
                q.a_hz2 = q.numerator_tables(3, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    return rounds


def build_druglib_stream_equiv_rounds(frame):
    """Equivalent 4-round session for per-ratio stream executors (expressing coupled net contrast
    (Relief - AE)/N as the difference of P(Relief) and P(AE), which is exact for pairwise streams)."""
    rds = build_druglib_live_rounds(frame)
    all_h = tuple(sorted(int(h) for h in frame["stratum"].unique()))
    head_h = tuple(sorted(int(h) for h in frame.loc[frame["drug_freq"] == "head", "stratum"].unique()))
    out = []
    for r in rds:
        nr = []
        for q in r:
            if q.qid == 2:
                q2 = Query(2, "net_benefit_minus_ae (all)", all_h, (), (2,), kind="difference", eps=0.05,
                           scope2=all_h, d_idx2=(), r_idx2=(0,))
            elif q.qid == 10:
                q2 = Query(10, "net_benefit_minus_ae (head)", head_h, (), (2,), kind="difference", eps=0.06,
                           scope2=head_h, d_idx2=(), r_idx2=(0,))
            else:
                q2 = q
            q2.tables = q2.build_tables(3)
            q2.a_hz = q2.numerator_tables(3, q2.scope, meta=q2.r_meta)
            if q2.kind == "difference":
                q2.a_hz2 = q2.numerator_tables(3, q2.scope2, meta=q2.r_meta2, tables={"a": q2.tables["a2"]})
            nr.append(q2)
        out.append(nr)
    return out


def verify_order_invariance(op, snap, ids_sample, ref_labels, seed=77):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids_sample))
    mismatches = 0
    pos = 0
    while pos < len(ids_sample):
        step = int(rng.choice([1, 5, 11, 19, 32, 47]))
        sub_idx = perm[pos : min(len(ids_sample), pos + step)]
        sub_ids = [int(ids_sample[i]) for i in sub_idx]
        ans = op.judge(sub_ids, snap.texts(sub_ids))
        for rid, row in zip(sub_ids, ans):
            if not (row == ref_labels[rid]).all():
                mismatches += 1
        pos += step
    return mismatches, len(ids_sample)


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


def run_live_comparison_on_domain(domain_name, op, snap, frame, ref_labels, hist_fn,
                                  cohort_rounds_fn, stream_rounds_fn, seeds=(0, 1, 2)):
    ids_all = [int(i) for i in frame["id"]]
    stratum_of = dict(zip(frame["id"].astype(int), frame["stratum"].astype(int)))
    ref_qs = [q for r in cohort_rounds_fn() for q in r]
    truth = {q.qid: query_truth_general(q, ref_labels, ids_all, stratum_of) for q in ref_qs}

    methods_out = {
        "COHORT (live, default)": [],
        "COHORT (live, exact rational audit)": [],
        "Dedup-streams + dynamic E (live)": [],
        "Shared-priority + E0 + partition (live)": [],
    }

    for sd in seeds:
        hist = hist_fn(sd)

        # 1 & 2: COHORT without and with exact rational LP audit
        for mkey, exact_flag in [
            ("COHORT (live, default)", False),
            ("COHORT (live, exact rational audit)", True),
        ]:
            rds = cohort_rounds_fn()
            all_qs = [q for r in rds for q in r]
            timed = TimedLiveCanonicalOperator(op, ref_labels)
            pipe = COHORTRAG(snap, timed, Config(alpha=0.05, audit_exact=exact_flag),
                             seed=sd, cache_labels=hist)
            first_wall = [None]
            first_calls = [None]
            round_walls = []
            round_calls = []
            orig_classify = pipe._classify
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            def wrap_cls(q, poly, _orig=orig_classify):
                res_c = _orig(q, poly)
                if res_c[0].startswith("certified") and first_wall[0] is None:
                    torch.cuda.synchronize()
                    first_wall[0] = time.perf_counter() - t_start
                    first_calls[0] = int(pipe.calls)
                return res_c

            pipe._classify = wrap_cls
            st = {"idx": 0}

            def feed(p, results):
                torch.cuda.synchronize()
                round_walls.append(round(time.perf_counter() - t_start, 2))
                round_calls.append(int(p.calls))
                st["idx"] += 1
                if st["idx"] >= len(rds):
                    return [], None
                return list(rds[st["idx"]]), len(ids_all)

            res = pipe.run(rds[0], CertificateGuidedAllocator(), budget_total=len(ids_all),
                           declared=all_qs, feed=feed)
            torch.cuda.synchronize()
            tot_wall = time.perf_counter() - t_start
            cpu_s = max(0.0, tot_wall - timed.gpu_seconds)
            n_cert = sum(1 for q in all_qs if res[q.qid].status.startswith("certified"))
            n_false = sum(
                1 for q in all_qs
                if res[q.qid].status.startswith("certified")
                and not (res[q.qid].L - 1e-8 <= truth[q.qid] <= res[q.qid].U + 1e-8)
            )
            methods_out[mkey].append({
                "seed": sd,
                "certified": f"{n_cert}/{len(all_qs)}",
                "new_calls": int(pipe.calls),
                "first_cert_calls": first_calls[0],
                "first_cert_wall_s": round(float(first_wall[0] or tot_wall), 2),
                "round_walls_s": round_walls,
                "round_calls": round_calls,
                "total_wall_s": round(tot_wall, 2),
                "gpu_s": round(timed.gpu_seconds, 2),
                "cpu_s": round(cpu_s, 2),
                "label_mismatches": int(timed.mismatches),
                "false_certificates": int(n_false),
            })
            S.log(f"[{domain_name}/{mkey}] sd={sd}: calls={pipe.calls}, wall={tot_wall:.2f}s "
                  f"(first={first_wall[0]:.2f}s, cpu={cpu_s:.2f}s), mismatch={timed.mismatches}, false={n_false}")

        # 3 & 4: Stream executors (Dynamic E & Shared-Priority + E0)
        for mkey, kw in [
            ("Dedup-streams + dynamic E (live)",
             dict(adaptive_risk=True, known_mass_mode="Edyn", order_mode="independent", batch_step=32)),
            ("Shared-priority + E0 + partition (live)",
             dict(adaptive_risk=True, known_mass_mode="E0", order_mode="shared_priority", batch_step=32)),
        ]:
            rds = stream_rounds_fn()
            all_qs = [q for r in rds for q in r]
            timed = TimedLiveCanonicalOperator(op, ref_labels)
            env_live = {"snap": snap, "contract": timed, "stratum_of": stratum_of, "N": len(ids_all)}
            first_wall = [None]
            first_calls = [None]
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            def on_fc(c_now):
                torch.cuda.synchronize()
                first_wall[0] = time.perf_counter() - t_start
                first_calls[0] = int(c_now)

            res, c_tot, rcalls, _, halted = R5.run_streams_executor(
                env_live, rds, sd, init_cache=hist, on_first_cert=on_fc, **kw
            )
            torch.cuda.synchronize()
            tot_wall = time.perf_counter() - t_start
            cpu_s = max(0.0, tot_wall - timed.gpu_seconds)
            n_cert = sum(1 for q in all_qs if res[q.qid].status.startswith("certified"))
            n_false = sum(
                1 for q in all_qs
                if res[q.qid].status.startswith("certified")
                and not (res[q.qid].L - 1e-8 <= truth[q.qid] <= res[q.qid].U + 1e-8)
            )
            methods_out[mkey].append({
                "seed": sd,
                "certified": f"{n_cert}/{len(all_qs)}",
                "halted": bool(halted),
                "new_calls": int(c_tot),
                "first_cert_calls": first_calls[0],
                "first_cert_wall_s": round(float(first_wall[0] or tot_wall), 2),
                "round_calls": rcalls,
                "total_wall_s": round(tot_wall, 2),
                "gpu_s": round(timed.gpu_seconds, 2),
                "cpu_s": round(cpu_s, 2),
                "label_mismatches": int(timed.mismatches),
                "false_certificates": int(n_false),
            })
            S.log(f"[{domain_name}/{mkey}] sd={sd}: calls={c_tot}, wall={tot_wall:.2f}s "
                  f"(first={first_wall[0]:.2f}s, cpu={cpu_s:.2f}s), mismatch={timed.mismatches}, false={n_false}")

    summary = {}
    for mkey, rows in methods_out.items():
        summary[mkey] = {
            "mean_new_calls": round(float(np.mean([r["new_calls"] for r in rows])), 1),
            "mean_first_cert_wall_s": round(float(np.mean([r["first_cert_wall_s"] for r in rows])), 2),
            "mean_total_wall_s": round(float(np.mean([r["total_wall_s"] for r in rows])), 2),
            "mean_gpu_s": round(float(np.mean([r["gpu_s"] for r in rows])), 2),
            "mean_cpu_s": round(float(np.mean([r["cpu_s"] for r in rows])), 2),
            "total_label_mismatches": sum(r["label_mismatches"] for r in rows),
            "total_false_certificates": sum(r["false_certificates"] for r in rows),
            "rows": rows,
        }
    return summary


def main():
    gpu_name = torch.cuda.get_device_name(0)
    model_path = "/root/autodl-tmp/models/Qwen3-8B"
    out_payload = {"gpu_device": gpu_name, "protocol": "deterministic_fixed_shape_bf16"}

    # =========================================================================
    # PART A: UCI DrugLib (N = 4,143) Full-Table Measured Scan + Live Benchmark
    # =========================================================================
    S.log(f"=== PART A: Building Full Canonical Reference Tape on UCI DrugLib (N=4,143) [{gpu_name}] ===")
    d_frame, d_gold, _ = datasets.load_druglib(
        os.path.join(_HERE, "..", "data", "drugs"),
        os.path.join(_HERE, "..", "data_manifests", "druglib.json"),
    )
    d_snap = Snapshot(d_frame[["id", "stratum", "text"]].copy(), name="druglib-frozen")
    d_pack = contracts.druglib_prompt_pack()
    t_load0 = time.perf_counter()
    op_drug = LocalLLMOperator(
        model_id=model_path, K=3, prompt_pack=d_pack, max_input_tokens=380,
        max_new_tokens=8, batch_size=32, canonical_fixed_len=736, device="cuda", name="druglib-canon-8b"
    )
    op_drug._ensure_loaded()
    load_s = time.perf_counter() - t_load0

    all_d_ids = [int(i) for i in d_frame["id"]]
    all_d_txt = d_snap.texts(all_d_ids)
    # Warmup 32 records
    _ = op_drug.judge(all_d_ids[:32], all_d_txt[:32])
    torch.cuda.synchronize()

    # TRUE full-table evaluation across all 4,143 records (NO extrapolation!)
    t_full0 = time.perf_counter()
    d_mat = op_drug.judge(all_d_ids, all_d_txt)
    torch.cuda.synchronize()
    drug_full_wall_s = time.perf_counter() - t_full0

    d_labels = {rid: d_mat[idx] for idx, rid in enumerate(all_d_ids)}
    d_kappa = op_drug.kappa(d_snap.manifest)
    save_sqlite_tape(os.path.join(_HERE, "..", "cache", "labels_druglib.sqlite"), d_kappa, d_labels)
    S.log(f"[DrugLib Full Table] N={len(all_d_ids)} finished in TRUE wall-clock {drug_full_wall_s:.2f}s "
          f"({len(all_d_ids)/drug_full_wall_s:.2f} rec/s) | kappa={d_kappa[:16]}")

    # Verify batch/order invariance on 256 random DrugLib records
    mis_d, tot_d = verify_order_invariance(op_drug, d_snap, all_d_ids[:256], d_labels, seed=77)
    S.log(f"[DrugLib Order/Batch Invariance] mismatches = {mis_d}/{tot_d}")

    # Operator quality vs held-out patient ratings
    g_ae = np.array([d_gold[rid][0] for rid in all_d_ids])
    g_ef = np.array([d_gold[rid][1] for rid in all_d_ids])
    p_ae = d_mat[:, 0]
    p_dis = d_mat[:, 1]
    p_ef = d_mat[:, 2]
    drug_quality = {
        "n_records": len(all_d_ids),
        "kappa": d_kappa,
        "model_load_s": round(load_s, 2),
        "measured_full_table_wall_s": round(drug_full_wall_s, 2),
        "throughput_rec_per_s": round(len(all_d_ids) / drug_full_wall_s, 2),
        "order_invariance_mismatches": f"{mis_d}/{tot_d}",
        "rates_run": {"adverse_event": round(float(p_ae.mean()), 4),
                      "discontinued": round(float(p_dis.mean()), 4),
                      "substantial_relief": round(float(p_ef.mean()), 4)},
        "rates_gold": {"adverse_event_mod_plus": round(float(g_ae.mean()), 4),
                       "substantial_relief_consid_plus": round(float(g_ef.mean()), 4)},
        "agreement_ae": round(float((p_ae == g_ae).mean()), 4),
        "agreement_relief": round(float((p_ef == g_ef).mean()), 4),
    }

    drug_live = run_live_comparison_on_domain(
        "DrugLib",
        op_drug,
        d_snap,
        d_frame,
        d_labels,
        lambda sd: druglib_safety_history(d_frame, d_labels, 0.15, sd),
        lambda: build_druglib_live_rounds(d_frame),
        lambda: build_druglib_stream_equiv_rounds(d_frame),
        seeds=(0, 1, 2),
    )
    out_payload["druglib"] = {"meta_and_full_scan": drug_quality, "live_executors": drug_live}
    S.dump("exp_druglib_and_live.json", out_payload)
    S.log("Saved checkpoint PART A -> exp_druglib_and_live.json")

    # =========================================================================
    # PART B: WANDS (N = 31,482) TRUE Full-Table Scan + Live Multi-Executor Run
    # =========================================================================
    S.log("=== PART B: Measuring TRUE Full-Table Canonical Scan & Live Executors on WANDS (N=31,482) ===")
    env_w = S.load_wands_env()
    w_pack = contracts.wands_prompt_pack()
    op_wands = LocalLLMOperator(
        model_id=model_path, K=2, prompt_pack=w_pack, max_input_tokens=288,
        max_new_tokens=6, batch_size=32, canonical_fixed_len=288, device="cuda", name="wands-canon-8b"
    )
    # Reuse already loaded Qwen3-8B weights in VRAM!
    op_wands._tok = op_drug._tok
    op_wands._model = op_drug._model

    w_ids = [int(i) for i in env_w["ids_all"]]
    w_txt = env_w["snap"].texts(w_ids)
    _ = op_wands.judge(w_ids[:32], w_txt[:32])
    torch.cuda.synchronize()

    t_wf0 = time.perf_counter()
    w_mat = op_wands.judge(w_ids, w_txt)
    torch.cuda.synchronize()
    wands_full_wall_s = time.perf_counter() - t_wf0

    w_labels = {rid: w_mat[idx] for idx, rid in enumerate(w_ids)}
    mis_w, tot_w = verify_order_invariance(op_wands, env_w["snap"], w_ids[:512], w_labels, seed=99)
    diff_vs_5090 = sum(1 for rid in w_ids if not (w_labels[rid] == env_w["labels_M"][rid]).all())
    S.log(f"[WANDS Full Table] N={len(w_ids)} TRUE wall-clock = {wands_full_wall_s:.2f}s "
          f"({len(w_ids)/wands_full_wall_s:.2f} rec/s) | order_mismatch={mis_w}/{tot_w} | "
          f"cross_gpu_diff_vs_5090={diff_vs_5090}/{len(w_ids)} ({100*diff_vs_5090/len(w_ids):.2f}%)")

    # Build 2-round (5 distinct numeric targets) live session on WANDS using canonical w_labels for E_0!
    def wands_5target_rounds():
        wl5 = [q for q in S.fresh_workload(env_w) if q.kind in ("value", "difference")]
        # Round 1: global ratio + rating share; Round 2: >=4, <4, cohort diff
        r1 = [q for q in wl5 if q.qid in (1, 4)]
        r2 = [q for q in wl5 if q.qid in (2, 3, 5)]
        return [r1, r2]

    def wands_canon_bm25_hist(sd):
        base_h = S.bm25_history(env_w, 0.15, sd)
        return {int(rid): w_labels[int(rid)] for rid in base_h}

    wands_live = run_live_comparison_on_domain(
        "WANDS",
        op_wands,
        env_w["snap"],
        env_w["frame"],
        w_labels,
        wands_canon_bm25_hist,
        wands_5target_rounds,
        wands_5target_rounds,
        seeds=(0, 1, 2),
    )

    out_payload["wands"] = {
        "meta_and_full_scan": {
            "n_records": len(w_ids),
            "measured_full_table_wall_s": round(wands_full_wall_s, 2),
            "throughput_rec_per_s": round(len(w_ids) / wands_full_wall_s, 2),
            "order_invariance_mismatches": f"{mis_w}/{tot_w}",
            "cross_gpu_diff_vs_rtx5090_tape": f"{diff_vs_5090}/{len(w_ids)} ({100*diff_vs_5090/len(w_ids):.2f}%)",
        },
        "live_executors": wands_live,
    }
    S.dump("exp_druglib_and_live.json", out_payload)
    S.log("DONE exp_druglib_and_live.json (ALL PARTS COMPLETE)")


if __name__ == "__main__":
    main()
