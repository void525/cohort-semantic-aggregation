# -*- coding: utf-8 -*-
"""Review-3 critical experiment 6: Multi-Cohort Interactive Session vs Strong Composite Baselines.

Compares on a 4-round session of 12 distinct numeric/cohort targets (no duplicate thresholds!):
  Round 1 (2): global exact|rel, rating_share
  Round 2 (3): exact|rel on rated>=4, rated<4, and their cohort difference
  Round 3 (3): exact|rel on intent_group 0, intent_group 1, and their cohort difference
  Round 4 (4): exact|rel on missing ratings, plus 3 overlapping (rating x intent) sub-cohorts

Baselines (all sharing global label cache, identical CS primitive, and partition-aware closed-form):
  1. Adaptive Dedup-Streams + Cache + Partition: allocates geometric round risk alpha/2^r across newly
     arrived distinct streams in round r (realistic interactive arrival when follow-ups are unknown upfront).
  2. Oracle Upfront Dedup-Streams + Cache + Partition: knows all 9 distinct cohort streams in advance
     (oracle risk alpha/18 per CS), sharing cache and deterministic bounds across streams.
  3. Shared Prefix + Free Reads (CERT): COHORT's shared stratified state (12 sequences at alpha/12 across
     all rounds, zero extra risk for follow-ups), with 15% BM25 history as targeted exposure (free reads).
  4. COHORT (Known Mass + Shared Prefix + CERT): files 15% BM25 history as known mass E and shares one
     stratified state across all 12 distinct targets and 4 adaptive rounds.

Also runs the analogous 10-distinct-target multi-cohort session on Task 2 (K=4 Proxy-Cascade Audit).
"""
import json
import os
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import run_icde2027_suite as S
import run_real_data as rrd
import run_review3_suite as R3
from cohortrag import (
    COHORTRAG,
    CertificateGuidedAllocator,
    Config,
    FinitePopulationCS,
    RoundRobinAllocator,
    states,
)
from cohortrag.llm_operator import LabelTableOperator
from cohortrag.pipeline import QueryResult
from cohortrag.workload import Query


def build_task1_12_rounds(env):
    all_h = tuple(sorted(env["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if "<4" in str(lab) for h in hs}))
    mis = tuple(sorted({h for lab, hs in env["by_rating"].items() if "missing" in str(lab) for h in hs}))
    ig0 = (0, 2, 4)
    ig1 = (1, 3, 5)
    flag_ge4 = {h: (1 if h in ge4 else 0) for h in all_h}

    R1 = [
        Query(1, "ex|rel all", all_h, (0,), (1,), eps=0.05),
        Query(2, "rating>=4|rel", all_h, (0,), (), eps=0.10, r_meta=flag_ge4),
    ]
    R2 = [
        Query(3, "ex|rel >=4", ge4, (0,), (1,), eps=0.05),
        Query(4, "ex|rel <4", lt4, (0,), (1,), eps=0.05),
        Query(5, "diff >=4 vs <4", ge4, (0,), (1,), kind="difference", eps=0.05, scope2=lt4, d_idx2=(0,), r_idx2=(1,)),
    ]
    R3 = [
        Query(6, "ex|rel ig0", ig0, (0,), (1,), eps=0.05),
        Query(7, "ex|rel ig1", ig1, (0,), (1,), eps=0.05),
        Query(8, "diff ig0 vs ig1", ig0, (0,), (1,), kind="difference", eps=0.05, scope2=ig1, d_idx2=(0,), r_idx2=(1,)),
    ]
    R4 = [
        Query(9, "ex|rel missing", mis, (0,), (1,), eps=0.05),
        Query(10, "ex|rel >=4_ig0", (ge4[0],), (0,), (1,), eps=0.06),
        Query(11, "ex|rel >=4_ig1", (ge4[1],), (0,), (1,), eps=0.06),
        Query(12, "ex|rel mis_ig0", (mis[0],), (0,), (1,), eps=0.08),
    ]
    for q in R1 + R2 + R3 + R4:
        q.tables = q.build_tables(2)
        q.a_hz = q.numerator_tables(2, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    return [R1, R2, R3, R4]


def build_task2_10_rounds(env4):
    all_h = tuple(sorted(env4["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in env4["by_rating"].items() if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in env4["by_rating"].items() if "<4" in str(lab) for h in hs}))
    ig0 = (0, 2, 4)
    ig1 = (1, 3, 5)

    R1 = [
        Query(1, "prec_rel all", all_h, (0,), (2,), eps=0.05),
        Query(2, "prec_ex all", all_h, (0, 1), (2, 3), eps=0.05),
    ]
    R2 = [
        Query(3, "rec_rel all", all_h, (2,), (0,), eps=0.05),
        Query(4, "overclaim all", all_h, (2,), (0, 1), eps=0.05),
    ]
    R3 = [
        Query(5, "prec_ex >=4", ge4, (0, 1), (2, 3), eps=0.06),
        Query(6, "prec_ex <4", lt4, (0, 1), (2, 3), eps=0.10),
        Query(7, "diff prec_ex >=4 vs <4", ge4, (0, 1), (2, 3), kind="difference", eps=0.10, scope2=lt4, d_idx2=(0, 1), r_idx2=(2, 3)),
    ]
    R4 = [
        Query(8, "prec_ex ig0", ig0, (0, 1), (2, 3), eps=0.06),
        Query(9, "prec_ex ig1", ig1, (0, 1), (2, 3), eps=0.06),
        Query(10, "diff prec_ex ig0 vs ig1", ig0, (0, 1), (2, 3), kind="difference", eps=0.08, scope2=ig1, d_idx2=(0, 1), r_idx2=(2, 3)),
    ]
    for q in R1 + R2 + R3 + R4:
        q.tables = q.build_tables(4)
        q.a_hz = q.numerator_tables(4, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(4, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    return [R1, R2, R3, R4]


def run_fast_dedup_streams(env, rounds, seed, init_cache=None, adaptive_risk=False):
    """Dedup-streams + cache + partition. History is a free read and a deterministic floor.

    The permutation still covers the full cohort scope, and a cached hit is pushed into
    that stream's confidence sequence. This is not known-mass removal.
    """
    K = env["contract"].K
    snap, contract = env["snap"], env["contract"]
    strata_ids = {h: [int(i) for i in snap.stratum_ids(h)] for h in snap.strata()}
    id_h = env["stratum_of"]
    cache = {}
    h_obs = {h: np.zeros(1 << K, dtype=np.int64) for h in snap.strata()}
    h_seen = {h: 0 for h in snap.strata()}

    def record_cache(rid, lab):
        rid = int(rid)
        if rid not in cache:
            cache[rid] = lab
            h = id_h[rid]
            z = states.encode_state(tuple(int(v) for v in lab))
            h_obs[h][z] += 1
            h_seen[h] += 1

    if init_cache:
        for rid, lab in init_cache.items():
            record_cache(rid, np.asarray(lab, dtype=np.int8))

    calls = [0]
    budget = env["N"]

    def get_label(rid):
        if rid in cache:
            return cache[rid]
        if calls[0] >= budget:
            return None
        lab = np.asarray(contract.judge([rid], snap.texts([rid]))[0], dtype=np.int8)
        record_cache(rid, lab)
        calls[0] += 1
        return lab

    # Pre-count total distinct streams if oracle upfront risk
    all_qs = [q for r in rounds for q in r]
    ratio_specs = {}
    q_sides = {}

    def side_key(scope, a_hz, b_tab):
        return (
            tuple(scope),
            tuple((h, tuple(int(round(x)) for x in a_hz[h])) for h in scope),
            tuple(int(round(x)) for x in b_tab),
        )

    if not adaptive_risk:
        uniq = set()
        for q in all_qs:
            uniq.add(side_key(q.scope, q.a_hz, q.tables["b"]))
            if q.kind == "difference":
                uniq.add(side_key(q.scope2, q.a_hz2, q.tables["b2"]))
        oracle_delta = S.ALPHA / (2 * len(uniq))
    else:
        oracle_delta = None

    def ensure_side(scope, a_hz, b_tab, delta_for_new):
        key = side_key(scope, a_hz, b_tab)
        if key not in ratio_specs:
            pool = [rid for h in scope for rid in strata_ids[h]]
            rng = np.random.default_rng(seed * 1009 + len(ratio_specs) * 97 + 13)
            perm = [pool[i] for i in rng.permutation(len(pool))]
            ratio_specs[key] = {
                "scope": scope,
                "a": a_hz,
                "b": b_tab,
                "perm": perm,
                "pos": 0,
                "csA": FinitePopulationCS(len(perm)),
                "csC": FinitePopulationCS(len(perm)),
                "delta": delta_for_new,
            }
        return key

    def side_interval(key):
        s = ratio_specs[key]
        d = s["delta"]
        ivA = s["csA"].interval(d)
        ivC = s["csC"].interval(d)
        if ivA is None or ivC is None:
            return None
        detA = detC = det_seen = 0
        for h in s["scope"]:
            det_seen += h_seen[h]
            obs = h_obs[h]
            av_vec = np.round(s["a"][h]).astype(np.int64)
            bv_vec = np.round(s["b"]).astype(np.int64)
            detA += int(np.dot(obs, av_vec))
            detC += int(np.dot(obs, bv_vec - av_vec))
        unseen = len(s["perm"]) - det_seen
        loA, hiA = max(ivA[0], detA), min(ivA[1], detA + unseen)
        loC, hiC = max(ivC[0], detC), min(ivC[1], detC + unseen)
        if hiA + hiC <= 0:
            return ("undef", None, None)
        if loA + loC <= 0:
            return ("insuf", 0.0, 1.0)
        L = loA / (loA + hiC)
        U = hiA / (hiA + loC)
        return ("ok", max(0.0, L - 1e-9), min(1.0, U + 1e-9))

    results = {}
    round_calls = []

    for r_idx, r_qs in enumerate(rounds):
        # Determine new streams in this round
        new_keys = set()
        for q in r_qs:
            k1 = side_key(q.scope, q.a_hz, q.tables["b"])
            if k1 not in ratio_specs:
                new_keys.add(k1)
            if q.kind == "difference":
                k2 = side_key(q.scope2, q.a_hz2, q.tables["b2"])
                if k2 not in ratio_specs:
                    new_keys.add(k2)
        if adaptive_risk:
            # Allocate alpha / 2^(r+1) across new streams in round r (last round gets remaining alpha/2^R)
            r_share = S.ALPHA / (2 ** (r_idx + 1)) if r_idx < len(rounds) - 1 else S.ALPHA / (2 ** r_idx)
            d_new = r_share / max(1, 2 * len(new_keys))
        else:
            d_new = oracle_delta

        for q in r_qs:
            k1 = ensure_side(q.scope, q.a_hz, q.tables["b"], d_new)
            if q.kind == "difference":
                k2 = ensure_side(q.scope2, q.a_hz2, q.tables["b2"], d_new)
                q_sides[q.qid] = (k1, k2)
            else:
                q_sides[q.qid] = (k1,)
            results[q.qid] = QueryResult(qid=q.qid, name=q.name, kind=q.kind)

        unresolved = list(r_qs)

        def check_open():
            for q in list(unresolved):
                res = results[q.qid]
                ks = q_sides[q.qid]
                s1 = side_interval(ks[0])
                if s1 is None or s1[0] != "ok":
                    continue
                if q.kind == "difference":
                    s2 = side_interval(ks[1])
                    if s2 is None or s2[0] != "ok":
                        continue
                    L, U = s1[1] - s2[2], s1[2] - s2[1]
                else:
                    L, U = s1[1], s1[2]
                res.L, res.U = L, U
                if U - L <= 2 * q.eps:
                    res.status, res.point, res.calls_at_cert = "certified", 0.5 * (L + U), calls[0]
                    unresolved.remove(q)

        # Notice: existing streams might already have observations from earlier rounds, AND cache hits are free!
        check_open()
        while unresolved and calls[0] < budget:
            active_keys = set()
            for q in unresolved:
                for k in q_sides[q.qid]:
                    if ratio_specs[k]["pos"] < len(ratio_specs[k]["perm"]):
                        active_keys.add(k)
            if not active_keys:
                break
            progressed = False
            for k in sorted(active_keys, key=lambda x: str(x)):
                s = ratio_specs[k]
                end = min(s["pos"] + 32, len(s["perm"]))
                for p in range(s["pos"], end):
                    rid = s["perm"][p]
                    lab = get_label(rid)
                    if lab is None:
                        break
                    h = id_h[rid]
                    z = states.encode_state(tuple(int(v) for v in lab))
                    av = int(round(s["a"][h][z]))
                    bv = int(round(s["b"][z]))
                    s["csA"].push(av)
                    s["csC"].push(bv - av)
                    s["pos"] = p + 1
                    progressed = True
            check_open()
            if not progressed:
                break
        round_calls.append(int(calls[0]))

    return results, int(calls[0]), round_calls


def run_cohort_rounds(env, rounds, seed, hist, mode="E"):
    all_qs = [q for r in rounds for q in r]
    kw = {"cache_labels": hist} if mode == "E" else {}
    pipe = COHORTRAG(env["snap"], env["contract"], Config(alpha=S.ALPHA), seed=seed, **kw)
    if mode == "T" and hist:
        by_h = {}
        for rid, lab in hist.items():
            by_h.setdefault(env["stratum_of"][int(rid)], []).append((int(rid), lab))
        for h, items in by_h.items():
            pipe.ledger.add_targeted(h, [i for i, _ in items], [l for _, l in items])

    st = {"idx": 0, "rcalls": []}

    def feed(p, results):
        st["rcalls"].append(int(p.calls))
        st["idx"] += 1
        if st["idx"] >= len(rounds):
            return [], None
        return list(rounds[st["idx"]]), env["N"]

    res = pipe.run(rounds[0], CertificateGuidedAllocator(), budget_total=env["N"], declared=all_qs, feed=feed)
    return res, int(pipe.calls), st["rcalls"]


def evaluate_suite(name, env, rounds_fn, hist_fn, seeds=tuple(range(8))):
    S.log(f"=== Running Multi-Cohort Session Suite: {name} ===")
    ref_rounds = rounds_fn()
    all_ref = [q for r in ref_rounds for q in r]
    truth = {
        q.qid: rrd._query_truth(q, env["labels_M"], env["ids_all"], env["stratum_of"])
        for q in all_ref
    }
    methods = {
        "Adaptive Dedup-Streams+Cache+Partition": [],
        "Oracle Dedup-Streams+Cache+Partition": [],
        "Shared-Prefix + Free Reads": [],
        "COHORT (Known Mass + Shared Prefix)": [],
    }
    for sd in seeds:
        t0 = time.time()
        hist = hist_fn(sd)
        r_ad, c_ad, rc_ad = run_fast_dedup_streams(env, rounds_fn(), sd, init_cache=hist, adaptive_risk=True)
        r_or, c_or, rc_or = run_fast_dedup_streams(env, rounds_fn(), sd, init_cache=hist, adaptive_risk=False)
        r_fr, c_fr, rc_fr = run_cohort_rounds(env, rounds_fn(), sd, hist, mode="T")
        r_co, c_co, rc_co = run_cohort_rounds(env, rounds_fn(), sd, hist, mode="E")

        for mname, (res, calls, rc) in [
            ("Adaptive Dedup-Streams+Cache+Partition", (r_ad, c_ad, rc_ad)),
            ("Oracle Dedup-Streams+Cache+Partition", (r_or, c_or, rc_or)),
            ("Shared-Prefix + Free Reads", (r_fr, c_fr, rc_fr)),
            ("COHORT (Known Mass + Shared Prefix)", (r_co, c_co, rc_co)),
        ]:
            n_done = sum(1 for q in all_ref if res[q.qid].status.startswith("certified"))
            err = sum(
                1 for q in all_ref
                if res[q.qid].status.startswith("certified")
                and not (res[q.qid].L - 1e-8 <= truth[q.qid] <= res[q.qid].U + 1e-8)
            )
            methods[mname].append({
                "seed": sd, "total_calls": calls if n_done == len(all_ref) else None,
                "calls_spent": calls, "n_done": n_done, "n_total": len(all_ref),
                "round_calls": rc, "errors": err,
            })
        S.log(
            f"[{name}] seed={sd} ({time.time()-t0:.1f}s): "
            f"AdaptDS={c_ad} | OracleDS={c_or} | FreeReads={c_fr} | COHORT={c_co}"
        )

    summary = {"n_targets": len(all_ref), "n_rounds": len(ref_rounds), "seeds": len(seeds), "methods": {}}
    co_rows = methods["COHORT (Known Mass + Shared Prefix)"]
    for mname, rows in methods.items():
        done = [r["total_calls"] for r in rows if r["total_calls"] is not None]
        rc_med = [
            float(statistics.median([r["round_calls"][i] for r in rows if len(r["round_calls"]) > i]))
            for i in range(len(ref_rounds))
        ]
        # Paired saving of COHORT relative to this baseline: 1 - c_COHORT / c_base
        sav = [
            100.0 * (1.0 - c["total_calls"] / b["total_calls"])
            for c, b in zip(co_rows, rows)
            if c["total_calls"] is not None and b["total_calls"] is not None
        ]
        summary["methods"][mname] = {
            "done": f"{len(done)}/{len(seeds)}",
            "median_calls": float(statistics.median(done)) if done else None,
            "mean_calls": float(np.mean(done)) if done else None,
            "round_cumulative_medians": rc_med,
            "cohort_paired_saving_vs_this_pct": float(statistics.median(sav)) if sav else None,
            "errors": sum(r["errors"] for r in rows),
            "rows": rows,
        }
        S.log(
            f"[{name}/summary] {mname}: done={summary['methods'][mname]['done']} "
            f"med={summary['methods'][mname]['median_calls']} "
            f"rounds={rc_med} cohort_saves={summary['methods'][mname]['cohort_paired_saving_vs_this_pct']:.1f}% "
            f"err={summary['methods'][mname]['errors']}"
        )
    return summary


def main():
    env = S.load_wands_env()
    # Task 1: Relevance Audit (K=2, H=6, 12 distinct numeric/cohort targets across 4 rounds)
    s1 = evaluate_suite(
        "Task1_Relevance_12Targets",
        env,
        lambda: build_task1_12_rounds(env),
        lambda sd: S.bm25_history(env, 0.15, sd),
        seeds=tuple(range(10)),
    )

    # Task 2: Proxy-Cascade Audit (K=3, H=3 rating strata, 12 distinct numeric/cohort targets across 4 rounds)
    l8 = env["labels_M"]
    l4 = R3.load_4b_labels()
    ids = [int(i) for i in env["ids_all"]]
    lab3 = {
        i: np.array([int(l4[i][0] and l4[i][1]), int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8)
        for i in ids
    }
    f3 = env["frame"].copy()
    f3["stratum"] = f3["rating_bucket"].map({"<4": 0, ">=4": 1, "missing": 2}).astype(int)
    from cohortrag import Snapshot
    snap3 = Snapshot(f3[["id", "stratum", "text"]].reset_index(drop=True), name="wands-h3-k3")
    env3 = dict(env)
    env3["snap"] = snap3
    env3["frame"] = f3
    env3["stratum_of"] = dict(zip(f3["id"].astype(int), f3["stratum"].astype(int)))
    env3["contract"] = LabelTableOperator(lab3, name="cascade-k3", K=3)
    env3["labels_M"] = lab3
    env3["meta"] = {"all_strata": [0, 1, 2]}
    env3["by_rating"] = {"<4": [0], ">=4": [1], "missing": [2]}

    def build_task2_12_rounds():
        R1 = [
            Query(1, "prec_ex (8B_ex | 4B_ex) all", (0, 1, 2), (0,), (2,), eps=0.05),
            Query(2, "overclaim (4B_ex | 8B_rel) all", (0, 1, 2), (1,), (0,), eps=0.05),
        ]
        R2 = [
            Query(3, "prec_ex >=4", (1,), (0,), (2,), eps=0.05),
            Query(4, "prec_ex <4", (0,), (0,), (2,), eps=0.08),
            Query(5, "prec_ex missing", (2,), (0,), (2,), eps=0.06),
        ]
        R3 = [
            Query(6, "overclaim >=4", (1,), (1,), (0,), eps=0.05),
            Query(7, "overclaim <4", (0,), (1,), (0,), eps=0.08),
            Query(8, "overclaim missing", (2,), (1,), (0,), eps=0.06),
        ]
        R4 = [
            Query(9, "diff prec_ex >=4 vs <4", (1,), (0,), (2,), kind="difference", eps=0.08, scope2=(0,), d_idx2=(0,), r_idx2=(2,)),
            Query(10, "diff overclaim >=4 vs missing", (1,), (1,), (0,), kind="difference", eps=0.06, scope2=(2,), d_idx2=(1,), r_idx2=(0,)),
            Query(11, "ex8|rel8 all", (0, 1, 2), (1,), (2,), eps=0.05),
            Query(12, "ex8|rel8 >=4", (1,), (1,), (2,), eps=0.05),
        ]
        for q in R1 + R2 + R3 + R4:
            q.tables = q.build_tables(3)
            q.a_hz = q.numerator_tables(3, q.scope)
            if q.kind == "difference":
                q.a_hz2 = q.numerator_tables(3, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return [R1, R2, R3, R4]

    proxy_ex_ids = [i for i in ids if lab3[i][0] == 1]
    other_ids = [i for i in ids if lab3[i][0] == 0]
    n_hist = int(round(0.15 * env3["N"]))

    def hist3(sd):
        rng = np.random.default_rng(sd * 997 + 41)
        n_pe = min(len(proxy_ex_ids), int(0.85 * n_hist))
        picked = list(rng.choice(proxy_ex_ids, size=n_pe, replace=False)) + list(
            rng.choice(other_ids, size=n_hist - n_pe, replace=False)
        )
        return {int(i): lab3[int(i)] for i in picked}

    s2_aligned = evaluate_suite(
        "Task2_Cascade_H3K3_12Targets",
        env3,
        build_task2_12_rounds,
        hist3,
        seeds=tuple(range(10)),
    )

    # Also keep Task 2 over-fragmented K=4 H=6 (42 CSs) for diagnostic transparency!
    lab4 = {
        i: np.array([int(l4[i][0]), int(l4[i][0] and l4[i][1]), int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8)
        for i in ids
    }
    env4 = dict(env)
    env4["contract"] = LabelTableOperator(lab4, name="cascade-4b-8b", K=4)
    env4["labels_M"] = lab4

    def hist4(sd):
        rng = np.random.default_rng(sd * 997 + 41)
        n_pe = min(len(proxy_ex_ids), int(0.85 * n_hist))
        picked = list(rng.choice(proxy_ex_ids, size=n_pe, replace=False)) + list(
            rng.choice(other_ids, size=n_hist - n_pe, replace=False)
        )
        return {int(i): lab4[int(i)] for i in picked}

    s2_over = evaluate_suite(
        "Task2_Cascade_H6K4_10Targets",
        env4,
        lambda: build_task2_10_rounds(env4),
        hist4,
        seeds=tuple(range(8)),
    )

    S.dump(
        "exp_multicohort_sessions.json",
        {"task1_12targets": s1, "task2_aligned_12targets": s2_aligned, "task2_overpartitioned_10targets": s2_over},
    )
    S.log("DONE exp_multicohort_sessions.json")


if __name__ == "__main__":
    main()
