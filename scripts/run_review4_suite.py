# -*- coding: utf-8 -*-
"""Review-4 decisive experiments (100% real WANDS frozen tapes + live local GPU execution).

Produces -> reports/icde2027_suite/exp_review4_suite.json:
  1. E-aware Dedicated Streams & 1..8 Round Crossover Trajectories on Task 1 (K=2, H=6) and Task 2 (K=3, H=3):
     - Shared-Prefix + Free Reads (T)
     - Adaptive Dedup-Streams + Free Reads (T)
     - Adaptive Dedup-Streams + Known Mass (E0)   [exact stage-start history removal on each stream R_q = U_q \\ E_0]
     - Adaptive Dedup-Streams + Dynamic E         [each newly arrived round removes ALL cached labels known so far]
     - Oracle Dedup-Streams + Known Mass (E0)     [knows all future distinct streams at t=0]
     - COHORT (Known Mass E0 + Shared Prefix)
     Records per-round cumulative calls, marginal calls, first-answer calls, completion rates, errors,
     and both median and mean paired per-seed savings across session lengths R in {1, 2, 3, 4, 5, 6, 7, 8}.

  2. Query-Overlap Regime Test (Low-Overlap Disjoint vs. High-Overlap Multi-Dimensional Drill-Down):
     Quantifies how follow-up overlap governs the crossover between E-aware dedicated streams and COHORT.

  3. Live End-to-End GPU Execution Benchmark (Local Qwen3-8B on GPU, zero tape shortcuts):
     Measures true interactive wall-clock time to first numeric certificate, time to all 5 numeric targets,
     GPU forward-pass wall-clock vs. cache-lookup vs. COHORT CPU projection/scheduling time, and bit-for-bit
     agreement with the frozen reference tape.
"""
import argparse
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
import run_multicohort_sessions as MC
import run_real_data as rrd
import run_review3_suite as R3
from cohortrag import (
    COHORTRAG,
    CertificateGuidedAllocator,
    Config,
    FinitePopulationCS,
    Snapshot,
    contracts,
    states,
)
from cohortrag.llm_operator import LabelTableOperator, LocalLLMOperator
from cohortrag.pipeline import QueryResult
from cohortrag.workload import Query


# =========================================================================== #
# Workloads: 8-round extensions inside the exact same semantic/stratum closure
# =========================================================================== #
def build_task1_8rounds(env):
    """Rounds 1-4 match build_task1_12_rounds identically; Rounds 5-8 continue interactive drill-down."""
    base = MC.build_task1_12_rounds(env)
    all_h = tuple(sorted(env["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if "<4" in str(lab) for h in hs}))
    mis = tuple(sorted({h for lab, hs in env["by_rating"].items() if "missing" in str(lab) for h in hs}))
    ig0 = (0, 2, 4)
    ig1 = (1, 3, 5)
    rated_known = tuple(sorted(set(ge4) | set(lt4)))
    flag_ge4 = {h: (1 if h in ge4 else 0) for h in all_h}
    flag_ig0 = {h: (1 if h in ig0 else 0) for h in all_h}

    R5 = [
        Query(13, "ex|rel mis_ig1", (mis[1],), (0,), (1,), eps=0.08),
        Query(14, "diff >=4: ig0 vs ig1", (ge4[0],), (0,), (1,), kind="difference", eps=0.08,
              scope2=(ge4[1],), d_idx2=(0,), r_idx2=(1,)),
        Query(15, "rating>=4|rel on ig0", ig0, (0,), (), eps=0.08, r_meta=flag_ge4),
    ]
    R6 = [
        Query(16, "rating>=4|rel on ig1", ig1, (0,), (), eps=0.08, r_meta=flag_ge4),
        Query(17, "diff rating_share ig0 vs ig1", ig0, (0,), (), kind="difference", eps=0.10,
              r_meta=flag_ge4, scope2=ig1, d_idx2=(0,), r_idx2=(), r_meta2=flag_ge4),
        Query(18, "ig0 share among rel", all_h, (0,), (), eps=0.06, r_meta=flag_ig0),
    ]
    R7 = [
        Query(19, "ex|rel rated_known", rated_known, (0,), (1,), eps=0.05),
        Query(20, "diff rated_known vs missing", rated_known, (0,), (1,), kind="difference", eps=0.07,
              scope2=mis, d_idx2=(0,), r_idx2=(1,)),
        Query(21, "rating>=4|exact all", all_h, (0, 1), (), eps=0.10, r_meta=flag_ge4),
    ]
    R8 = [
        Query(22, "ig0 share among exact", all_h, (0, 1), (), eps=0.10, r_meta=flag_ig0),
        Query(23, "ex|rel rated_known_ig0", tuple(sorted(set(rated_known) & set(ig0))), (0,), (1,), eps=0.06),
        Query(24, "ex|rel rated_known_ig1", tuple(sorted(set(rated_known) & set(ig1))), (0,), (1,), eps=0.06),
    ]
    extra = [R5, R6, R7, R8]
    for r in extra:
        for q in r:
            q.tables = q.build_tables(2)
            q.a_hz = q.numerator_tables(2, q.scope, meta=q.r_meta)
            if q.kind == "difference":
                q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    return base + extra


def build_task2_env_and_8rounds(env):
    l8 = env["labels_M"]
    l4 = R3.load_4b_labels()
    ids = [int(i) for i in env["ids_all"]]
    lab3 = {
        i: np.array([int(l4[i][0] and l4[i][1]), int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8)
        for i in ids
    }
    f3 = env["frame"].copy()
    f3["stratum"] = f3["rating_bucket"].map({"<4": 0, ">=4": 1, "missing": 2}).astype(int)
    snap3 = Snapshot(f3[["id", "stratum", "text"]].reset_index(drop=True), name="wands-h3-k3")
    env3 = dict(env)
    env3["snap"] = snap3
    env3["frame"] = f3
    env3["stratum_of"] = dict(zip(f3["id"].astype(int), f3["stratum"].astype(int)))
    env3["contract"] = LabelTableOperator(lab3, name="cascade-k3", K=3)
    env3["labels_M"] = lab3
    env3["meta"] = {"all_strata": [0, 1, 2]}
    env3["by_rating"] = {"<4": [0], ">=4": [1], "missing": [2]}

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

    def build_rounds():
        flag_ge4 = {0: 0, 1: 1, 2: 0}
        flag_rated = {0: 1, 1: 1, 2: 0}
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
            Query(9, "diff prec_ex >=4 vs <4", (1,), (0,), (2,), kind="difference", eps=0.08,
                  scope2=(0,), d_idx2=(0,), r_idx2=(2,)),
            Query(10, "diff overclaim >=4 vs missing", (1,), (1,), (0,), kind="difference", eps=0.06,
                  scope2=(2,), d_idx2=(1,), r_idx2=(0,)),
            Query(11, "ex8|rel8 all", (0, 1, 2), (1,), (2,), eps=0.05),
            Query(12, "ex8|rel8 >=4", (1,), (1,), (2,), eps=0.05),
        ]
        R5 = [
            Query(13, "ex8|rel8 missing", (2,), (1,), (2,), eps=0.06),
            Query(14, "ex8|rel8 <4", (0,), (1,), (2,), eps=0.08),
            Query(15, "diff ex8|rel8 >=4 vs missing", (1,), (1,), (2,), kind="difference", eps=0.07,
                  scope2=(2,), d_idx2=(1,), r_idx2=(2,)),
        ]
        R6 = [
            Query(16, "rel_prec (8B_rel | 4B_ex) all", (0, 1, 2), (0,), (1,), eps=0.05),
            Query(17, "rel_prec >=4", (1,), (0,), (1,), eps=0.05),
            Query(18, "rel_prec missing", (2,), (0,), (1,), eps=0.06),
        ]
        R7 = [
            Query(19, "prec_ex rated_known", (0, 1), (0,), (2,), eps=0.05),
            Query(20, "overclaim rated_known", (0, 1), (1,), (0,), eps=0.05),
            Query(21, "diff prec_ex rated vs missing", (0, 1), (0,), (2,), kind="difference", eps=0.07,
                  scope2=(2,), d_idx2=(0,), r_idx2=(2,)),
        ]
        R8 = [
            Query(22, "ge4 share among 4B_ex", (0, 1, 2), (0,), (), eps=0.06, r_meta=flag_ge4),
            Query(23, "ge4 share among 8B_rel", (0, 1, 2), (1,), (), eps=0.06, r_meta=flag_ge4),
            Query(24, "rated share among 4B_ex", (0, 1, 2), (0,), (), eps=0.06, r_meta=flag_rated),
        ]
        rounds = [R1, R2, R3, R4, R5, R6, R7, R8]
        for r in rounds:
            for q in r:
                q.tables = q.build_tables(3)
                q.a_hz = q.numerator_tables(3, q.scope, meta=q.r_meta)
                if q.kind == "difference":
                    q.a_hz2 = q.numerator_tables(3, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return rounds

    return env3, build_rounds, hist3


# =========================================================================== #
# Unified Dedicated-Stream Executor (supporting T, E0, Edyn, Adaptive & Oracle)
# =========================================================================== #
def run_dedup_streams_unified(env, rounds, seed, init_cache=None, adaptive_risk=True,
                              known_mass_mode="E0", cs_pad=1):
    """Deduplicated per-ratio streams + shared global cache + partition bounds + fail-closed CS.

    known_mass_mode:
      - "T"   : history remains inside each stream's sampling frame (free reads + deterministic floor).
      - "E0"  : stage-start history E0 is removed from each stream's frame R_s = U_s \\ E0,
                contributing exact numerator/complement offsets (eA, eC).
      - "Edyn": each newly arrived stream at round r removes ALL labels present in the global cache
                at the start of round r (E_0 plus labels paid in rounds 1..r-1).
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

    e0_ids = set()
    if init_cache:
        for rid, lab in init_cache.items():
            record_cache(rid, np.asarray(lab, dtype=np.int8))
            e0_ids.add(int(rid))

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
            full_pool = [rid for h in scope for rid in strata_ids[h]]
            if known_mass_mode == "T":
                known_set = set()
            elif known_mass_mode == "E0":
                known_set = e0_ids
            elif known_mass_mode == "Edyn":
                known_set = set(cache.keys())
            else:
                raise ValueError(known_mass_mode)

            eA = eC = 0
            rem_pool = []
            bv_vec = np.round(b_tab).astype(np.int64)
            av_map = {h: np.round(a_hz[h]).astype(np.int64) for h in scope}
            for rid in full_pool:
                if rid in known_set:
                    h = id_h[rid]
                    z = states.encode_state(tuple(int(v) for v in cache[rid]))
                    av = int(av_map[h][z])
                    bv = int(bv_vec[z])
                    eA += av
                    eC += (bv - av)
                else:
                    rem_pool.append(rid)

            rng = np.random.default_rng(seed * 1009 + len(ratio_specs) * 97 + 13)
            perm = [rem_pool[i] for i in rng.permutation(len(rem_pool))] if rem_pool else []
            ratio_specs[key] = {
                "scope": scope,
                "full_len": len(full_pool),
                "rem_len": len(perm),
                "a": a_hz,
                "b": b_tab,
                "eA": eA,
                "eC": eC,
                "perm": perm,
                "pos": 0,
                "csA": FinitePopulationCS(len(perm)),
                "csC": FinitePopulationCS(len(perm)),
                "delta": delta_for_new,
                "failed": False,
            }
        return key

    def side_interval(key):
        s = ratio_specs[key]
        if s["failed"]:
            return ("empty", None, None)
        M = s["rem_len"]
        if M == 0:
            lo_csA = hi_csA = lo_csC = hi_csC = 0
        else:
            ivA = s["csA"].interval(s["delta"])
            ivC = s["csC"].interval(s["delta"])
            if ivA is None or ivC is None:
                s["failed"] = True
                return ("empty", None, None)
            lo_csA, hi_csA = max(0, ivA[0] - cs_pad), min(M, ivA[1] + cs_pad)
            lo_csC, hi_csC = max(0, ivC[0] - cs_pad), min(M, ivC[1] + cs_pad)
            if s["pos"] == M:
                lo_csA = hi_csA = ivA[0]
                lo_csC = hi_csC = ivC[0]

        detA = detC = det_seen = 0
        for h in s["scope"]:
            det_seen += h_seen[h]
            obs = h_obs[h]
            av_vec = np.round(s["a"][h]).astype(np.int64)
            bv_vec = np.round(s["b"]).astype(np.int64)
            detA += int(np.dot(obs, av_vec))
            detC += int(np.dot(obs, bv_vec - av_vec))
        unseen = s["full_len"] - det_seen
        loA = max(s["eA"] + lo_csA, detA)
        hiA = min(s["eA"] + hi_csA, detA + unseen)
        loC = max(s["eC"] + lo_csC, detC)
        hiC = min(s["eC"] + hi_csC, detC + unseen)
        if hiA + hiC <= 0:
            return ("undef", None, None)
        if loA + loC <= 0:
            return ("insuf", 0.0, 1.0)
        L = loA / (loA + hiC)
        U = hiA / (hiA + loC)
        return ("ok", max(0.0, L - 1e-9), min(1.0, U + 1e-9))

    results = {}
    round_calls = []
    round_first_calls = []
    halted = False

    for r_idx, r_qs in enumerate(rounds):
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
            # Strict infinite-horizon geometric schedule alpha / 2^(r+1) (never peeks at total R)
            r_share = S.ALPHA / (2 ** (r_idx + 1))
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
        first_cert_in_round = [None]

        def check_open():
            nonlocal halted
            for q in list(unresolved):
                res = results[q.qid]
                ks = q_sides[q.qid]
                s1 = side_interval(ks[0])
                if s1[0] == "empty":
                    res.status = "execution_error"
                    unresolved.remove(q)
                    halted = True
                    continue
                if s1[0] != "ok":
                    continue
                if q.kind == "difference":
                    s2 = side_interval(ks[1])
                    if s2[0] == "empty":
                        res.status = "execution_error"
                        unresolved.remove(q)
                        halted = True
                        continue
                    if s2[0] != "ok":
                        continue
                    L, U = s1[1] - s2[2], s1[2] - s2[1]
                else:
                    L, U = s1[1], s1[2]
                res.L, res.U = L, U
                if U - L <= 2 * q.eps:
                    res.status, res.point, res.calls_at_cert = "certified", 0.5 * (L + U), calls[0]
                    if first_cert_in_round[0] is None:
                        first_cert_in_round[0] = int(calls[0])
                    unresolved.remove(q)

        if not halted:
            check_open()
        while unresolved and not halted and calls[0] < budget:
            active_keys = {
                k for q in unresolved for k in q_sides[q.qid]
                if not ratio_specs[k]["failed"] and ratio_specs[k]["pos"] < len(ratio_specs[k]["perm"])
            }
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
        round_first_calls.append(first_cert_in_round[0])

    return results, int(calls[0]), round_calls, round_first_calls, halted


def run_cohort_8rounds(env, rounds, seed, hist, mode="E"):
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

    res = pipe.run(rounds[0], CertificateGuidedAllocator(), budget_total=env["N"],
                   declared=all_qs, feed=feed)
    # Extract first-answer call per round from calls_at_cert
    first_calls = []
    for r_qs in rounds:
        certs = [res[q.qid].calls_at_cert for q in r_qs
                 if q.qid in res and res[q.qid].calls_at_cert is not None]
        first_calls.append(min(certs) if certs else None)
    halted = any(r.status == "execution_error" for r in res.values())
    return res, int(pipe.calls), st["rcalls"], first_calls, halted


def evaluate_8round_task(name, env, rounds_fn, hist_fn, seeds=tuple(range(10))):
    S.log(f"=== Running 8-Round E-Aware Session Suite: {name} ===")
    ref_rounds = rounds_fn()
    n_rounds = len(ref_rounds)
    all_ref = [q for r in ref_rounds for q in r]
    truth = {
        q.qid: rrd._query_truth(q, env["labels_M"], env["ids_all"], env["stratum_of"])
        for q in all_ref
    }
    round_qids = [[q.qid for q in r] for r in ref_rounds]

    method_runners = [
        ("Shared prefix + free reads (T)",
         lambda sd, h: run_cohort_8rounds(env, rounds_fn(), sd, h, mode="T")),
        ("Adaptive dedup-streams + T",
         lambda sd, h: run_dedup_streams_unified(env, rounds_fn(), sd, h, adaptive_risk=True, known_mass_mode="T")),
        ("Adaptive dedup-streams + E0",
         lambda sd, h: run_dedup_streams_unified(env, rounds_fn(), sd, h, adaptive_risk=True, known_mass_mode="E0")),
        ("Adaptive dedup-streams + dynamic E",
         lambda sd, h: run_dedup_streams_unified(env, rounds_fn(), sd, h, adaptive_risk=True, known_mass_mode="Edyn")),
        ("Oracle dedup-streams + E0",
         lambda sd, h: run_dedup_streams_unified(env, rounds_fn(), sd, h, adaptive_risk=False, known_mass_mode="E0")),
        ("COHORT (known mass E0 + shared prefix)",
         lambda sd, h: run_cohort_8rounds(env, rounds_fn(), sd, h, mode="E")),
    ]

    raw = {m: [] for m, _ in method_runners}
    for sd in seeds:
        t0 = time.time()
        h = hist_fn(sd)
        msg = []
        for mname, fn in method_runners:
            res, calls, rcalls, fcalls, halted = fn(sd, h)
            # Check prefix round completion up to each round r
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
            msg.append(f"{mname.split()[0]}={calls}{'*' if not round_done[-1] else ''}")
        S.log(f"[{name}] seed={sd} ({time.time()-t0:.1f}s): " + " | ".join(msg))

    co_key = "COHORT (known mass E0 + shared prefix)"
    co_rows = raw[co_key]
    summary = {
        "n_rounds": n_rounds,
        "n_targets": len(all_ref),
        "targets_per_round_cum": [sum(len(r) for r in ref_rounds[: i + 1]) for i in range(n_rounds)],
        "seeds": len(seeds),
        "methods": {},
    }

    for mname, rows in raw.items():
        round_meds, round_q25, round_q75, round_dones = [], [], [], []
        paired_med_by_r, paired_mean_by_r, paired_n_by_r = [], [], []
        marg_meds = []
        first_meds = []
        for r_i in range(n_rounds):
            vals = [r["round_calls"][r_i] for r in rows if r["round_done"][r_i]]
            round_dones.append(f"{len(vals)}/{len(seeds)}")
            round_meds.append(float(statistics.median(vals)) if vals else None)
            round_q25.append(float(np.percentile(vals, 25)) if vals else None)
            round_q75.append(float(np.percentile(vals, 75)) if vals else None)

            margs = [
                (r["round_calls"][r_i] - (r["round_calls"][r_i - 1] if r_i > 0 else 0))
                for r in rows if r["round_done"][r_i]
            ]
            marg_meds.append(float(statistics.median(margs)) if margs else None)

            fvals = [r["round_first_calls"][r_i] for r in rows
                     if r["round_done"][r_i] and r["round_first_calls"][r_i] is not None]
            first_meds.append(float(statistics.median(fvals)) if fvals else None)

            savs = [
                100.0 * (1.0 - c["round_calls"][r_i] / b["round_calls"][r_i])
                for c, b in zip(co_rows, rows)
                if c["round_done"][r_i] and b["round_done"][r_i]
            ]
            paired_n_by_r.append(len(savs))
            paired_med_by_r.append(float(statistics.median(savs)) if savs else None)
            paired_mean_by_r.append(float(np.mean(savs)) if savs else None)

        summary["methods"][mname] = {
            "round_done": round_dones,
            "round_cumulative_medians": round_meds,
            "round_cumulative_q25": round_q25,
            "round_cumulative_q75": round_q75,
            "round_marginal_medians": marg_meds,
            "round_first_cert_medians": first_meds,
            "cohort_paired_saving_median_by_round_pct": paired_med_by_r,
            "cohort_paired_saving_mean_by_round_pct": paired_mean_by_r,
            "cohort_paired_joint_seeds_by_round": paired_n_by_r,
            "total_errors": sum(r["errors"] for r in rows),
            "empty_cs_halts": sum(1 for r in rows if r["halted_empty_cs"]),
            "rows": rows,
        }
        S.log(
            f"[{name}/summary] {mname}:\n"
            f"    done@R4={round_dones[3]} med@R4={round_meds[3]} sav@R4={paired_med_by_r[3]:.1f}% (mean {paired_mean_by_r[3]:.1f}%)\n"
            f"    done@R8={round_dones[7]} med@R8={round_meds[7]} sav@R8={paired_med_by_r[7]:.1f}% (mean {paired_mean_by_r[7]:.1f}%)\n"
            f"    rounds={[int(round(x)) for x in round_meds]} err={sum(r['errors'] for r in rows)}"
        )
    return summary


# =========================================================================== #
# 2. Query-Overlap Regime Test (Low-Overlap Disjoint vs. High-Overlap)
# =========================================================================== #
def evaluate_overlap_regimes(env, seeds=tuple(range(10))):
    """4-round comparison: low-overlap isolated single-stratum targets vs. high-overlap multi-cohort queries."""
    S.log("=== Running Query-Overlap Regime Experiment ===")
    all_h = tuple(sorted(env["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if ">=4" in str(lab) for h in hs}))
    ig0 = (0, 2, 4)
    ig1 = (1, 3, 5)
    flag_ge4 = {h: (1 if h in ge4 else 0) for h in all_h}
    flag_ig0 = {h: (1 if h in ig0 else 0) for h in all_h}

    # Low-overlap 4-round session: 4 disjoint single-stratum queries (1 per round, zero scope overlap!)
    # Large enough strata (2, 3, 4, 5) so each finishes cleanly
    def build_low_overlap():
        rds = [
            [Query(1, "ex|rel h2", (2,), (0,), (1,), eps=0.06)],
            [Query(2, "ex|rel h3", (3,), (0,), (1,), eps=0.06)],
            [Query(3, "ex|rel h4", (4,), (0,), (1,), eps=0.07)],
            [Query(4, "ex|rel h5", (5,), (0,), (1,), eps=0.07)],
        ]
        for r in rds:
            for q in r:
                q.tables = q.build_tables(2)
                q.a_hz = q.numerator_tables(2, q.scope)
        return rds

    # High-overlap 4-round session: 16 overlapping global/cohort/composition/contrast queries
    def build_high_overlap():
        R1 = [
            Query(1, "ex|rel all", all_h, (0,), (1,), eps=0.05),
            Query(2, "rating>=4|rel all", all_h, (0,), (), eps=0.08, r_meta=flag_ge4),
            Query(3, "ig0|rel all", all_h, (0,), (), eps=0.06, r_meta=flag_ig0),
            Query(4, "rating>=4|exact all", all_h, (0, 1), (), eps=0.10, r_meta=flag_ge4),
        ]
        R2 = [
            Query(5, "ex|rel >=4", ge4, (0,), (1,), eps=0.05),
            Query(6, "ex|rel ig0", ig0, (0,), (1,), eps=0.05),
            Query(7, "ex|rel ig1", ig1, (0,), (1,), eps=0.05),
            Query(8, "diff ig0 vs ig1", ig0, (0,), (1,), kind="difference", eps=0.06, scope2=ig1, d_idx2=(0,), r_idx2=(1,)),
        ]
        R3 = [
            Query(9, "rating>=4|rel ig0", ig0, (0,), (), eps=0.08, r_meta=flag_ge4),
            Query(10, "rating>=4|rel ig1", ig1, (0,), (), eps=0.08, r_meta=flag_ge4),
            Query(11, "diff rating_share ig0 vs ig1", ig0, (0,), (), kind="difference", eps=0.10,
                  r_meta=flag_ge4, scope2=ig1, d_idx2=(0,), r_idx2=(), r_meta2=flag_ge4),
            Query(12, "ig0|exact all", all_h, (0, 1), (), eps=0.10, r_meta=flag_ig0),
        ]
        R4 = [
            Query(13, "ex|rel >=4_ig0", (ge4[0],), (0,), (1,), eps=0.06),
            Query(14, "ex|rel >=4_ig1", (ge4[1],), (0,), (1,), eps=0.06),
            Query(15, "diff >=4: ig0 vs ig1", (ge4[0],), (0,), (1,), kind="difference", eps=0.08,
                  scope2=(ge4[1],), d_idx2=(0,), r_idx2=(1,)),
            Query(16, "ig0|rel >=4", ge4, (0,), (), eps=0.06, r_meta=flag_ig0),
        ]
        rds = [R1, R2, R3, R4]
        for r in rds:
            for q in r:
                q.tables = q.build_tables(2)
                q.a_hz = q.numerator_tables(2, q.scope, meta=q.r_meta)
                if q.kind == "difference":
                    q.a_hz2 = q.numerator_tables(2, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
        return rds

    out = {}
    for reg_name, fn in [("low_overlap_4targets", build_low_overlap),
                         ("high_overlap_16targets", build_high_overlap)]:
        ds_e0_calls, ds_edyn_calls, co_calls, sav_e0, sav_edyn = [], [], [], [], []
        for sd in seeds:
            hist = S.bm25_history(env, 0.15, sd)
            r_e0, c_e0, _, _, h_e0 = run_dedup_streams_unified(env, fn(), sd, hist, True, "E0")
            r_ed, c_ed, _, _, h_ed = run_dedup_streams_unified(env, fn(), sd, hist, True, "Edyn")
            r_co, c_co, _, _, h_co = run_cohort_8rounds(env, fn(), sd, hist, mode="E")
            if not h_e0 and not h_co:
                ds_e0_calls.append(c_e0)
                co_calls.append(c_co)
                sav_e0.append(100.0 * (1.0 - c_co / c_e0))
            if not h_ed and not h_co:
                ds_edyn_calls.append(c_ed)
                sav_edyn.append(100.0 * (1.0 - c_co / c_ed))
        out[reg_name] = {
            "dedup_E0_median": float(statistics.median(ds_e0_calls)),
            "dedup_Edyn_median": float(statistics.median(ds_edyn_calls)),
            "cohort_median": float(statistics.median(co_calls)),
            "cohort_paired_saving_vs_E0_median_pct": float(statistics.median(sav_e0)),
            "cohort_paired_saving_vs_Edyn_median_pct": float(statistics.median(sav_edyn)),
            "joint_seeds_E0": len(sav_e0),
        }
        S.log(f"[Overlap/{reg_name}] DedupE0={out[reg_name]['dedup_E0_median']:.0f} | "
              f"DedupEdyn={out[reg_name]['dedup_Edyn_median']:.0f} | "
              f"COHORT={out[reg_name]['cohort_median']:.0f} | "
              f"sav_vs_E0={out[reg_name]['cohort_paired_saving_vs_E0_median_pct']:+.1f}%")
    return out


def evaluate_coupled_lp_vs_partition(env3, build_t2, hist3, seeds=tuple(range(10))):
    """Compares joint Charnes-Cooper LP on P_t vs subtracting separate partition bounds on (Ex_4B - Ex_8B) | Rel_8B."""
    from cohortrag.projection import ratio_bounds
    all_qs = [q for r in build_t2() for q in r]
    q_over = [q for q in all_qs if q.qid == 2][0]
    q_ex8 = [q for q in all_qs if q.qid == 11][0]
    a_diff_hz = {h: q_over.a_hz[h] - q_ex8.a_hz[h] for h in (0, 1, 2)}
    b_rel = q_over.tables["b"]
    out = {}
    for cap in (1280, 3200, 6400):
        w_sep, w_lp = [], []
        for sd in seeds:
            pipe = COHORTRAG(env3["snap"], env3["contract"], Config(alpha=S.ALPHA), seed=sd, cache_labels=hist3(sd))
            pipe.run([q_over, q_ex8], CertificateGuidedAllocator(), budget_total=cap, declared=all_qs)
            poly = pipe.last_poly
            L1, U1 = ratio_bounds(poly, q_over.a_hz, b_rel, (0, 1, 2), 8)
            L2, U2 = ratio_bounds(poly, q_ex8.a_hz, b_rel, (0, 1, 2), 8)
            lp_L, lp_U = ratio_bounds(poly, a_diff_hz, b_rel, (0, 1, 2), 8)
            w_sep.append(float((U1 - L2) - (L1 - U2)))
            w_lp.append(float(lp_U - lp_L))
        out[str(cap)] = {
            "sep_partition_mean_width": round(float(np.mean(w_sep)), 4),
            "joint_lp_mean_width": round(float(np.mean(w_lp)), 4),
            "width_reduction_pct": round(100.0 * (1.0 - float(np.mean(w_lp)) / float(np.mean(w_sep))), 2),
        }
    return out


# =========================================================================== #
# 3. Live End-to-End GPU Execution Benchmark (Zero Tape Shortcuts)
# =========================================================================== #
class TimedLiveOperator:
    """Wraps LocalLLMOperator to record exact GPU wall-clock seconds, batches, and tape agreement."""
    def __init__(self, inner, ref_labels):
        self.inner = inner
        self.K = inner.K
        self.name = inner.name
        self.ref_labels = ref_labels
        self.gpu_seconds = 0.0
        self.n_batches = 0
        self.n_records = 0
        self.mismatches = 0
        self.first_record_ts = None

    def judge(self, ids, texts):
        t0 = time.perf_counter()
        if self.first_record_ts is None:
            self.first_record_ts = t0
        out = self.inner.judge(ids, texts)
        dt = time.perf_counter() - t0
        self.gpu_seconds += dt
        self.n_batches += 1
        self.n_records += len(ids)
        for rid, row in zip(ids, out):
            if not (row == self.ref_labels[int(rid)]).all():
                self.mismatches += 1
        return out


def evaluate_live_gpu(env, model_path="/root/autodl-tmp/models/Qwen3-8B", seeds=(0, 1)):
    import torch
    if not torch.cuda.is_available() or not os.path.exists(model_path):
        S.log("Skipping live GPU benchmark (no CUDA or model path missing)")
        return None

    gpu_name = torch.cuda.get_device_name(0)
    S.log(f"=== Running LIVE GPU End-to-End Benchmark on {gpu_name} ===")
    pack = contracts.wands_prompt_pack()
    t_load0 = time.perf_counter()
    base_op = LocalLLMOperator(
        model_id=model_path, K=len(pack["primitives"]), prompt_pack=pack,
        batch_size=64, device="cuda", name="live-qwen3-8b"
    )
    base_op._ensure_loaded()
    load_s = time.perf_counter() - t_load0

    # Warmup 32 records and measure steady-state full-batch throughput on this GPU
    w_ids = env["ids_all"][:32]
    _ = base_op.judge(w_ids, env["snap"].texts(w_ids))
    torch.cuda.synchronize()
    b_ids = env["ids_all"][32:544]  # 512 records full-batch throughput probe
    t_b0 = time.perf_counter()
    _ = base_op.judge(b_ids, env["snap"].texts(b_ids))
    torch.cuda.synchronize()
    dt_512 = time.perf_counter() - t_b0
    live_rec_per_s = 512.0 / dt_512
    est_full_pool_s = env["N"] / live_rec_per_s
    S.log(f"[Live GPU] Model loaded in {load_s:.2f}s | 512-rec throughput: {live_rec_per_s:.1f} rec/s "
          f"(full {env['N']}-pool = {est_full_pool_s:.1f}s)")

    runs = []
    for sd in seeds:
        hist = S.bm25_history(env, 0.15, sd)
        wl = [q for q in S.fresh_workload(env) if q.kind in ("value", "difference")]
        timed_op = TimedLiveOperator(base_op, env["labels_M"])
        pipe = COHORTRAG(env["snap"], timed_op, Config(alpha=S.ALPHA), seed=sd, cache_labels=hist)

        # Hook trace to capture true wall-clock timestamp of FIRST certified numeric target!
        first_cert_wall = [None]
        first_cert_calls = [None]
        orig_classify = pipe._classify
        t_start = time.perf_counter()

        def wrapped_classify(q, poly):
            out = orig_classify(q, poly)
            if out[0] == "certified" and first_cert_wall[0] is None:
                torch.cuda.synchronize()
                first_cert_wall[0] = time.perf_counter() - t_start
                first_cert_calls[0] = int(pipe.calls)
            return out

        pipe._classify = wrapped_classify
        res = pipe.run(wl, CertificateGuidedAllocator(), budget_total=env["N"])
        torch.cuda.synchronize()
        total_wall = time.perf_counter() - t_start
        cpu_engine_s = max(0.0, total_wall - timed_op.gpu_seconds)
        n_cert = sum(1 for q in wl if res[q.qid].status == "certified")
        runs.append({
            "seed": sd,
            "n_certified": f"{n_cert}/{len(wl)}",
            "incremental_calls": int(pipe.calls),
            "first_cert_calls": first_cert_calls[0],
            "first_cert_wall_s": round(float(first_cert_wall[0]), 2),
            "all_5_wall_s": round(float(total_wall), 2),
            "gpu_inference_s": round(float(timed_op.gpu_seconds), 2),
            "cohort_cpu_s": round(float(cpu_engine_s), 2),
            "cpu_share_pct": round(100.0 * cpu_engine_s / total_wall, 2),
            "tape_mismatches": int(timed_op.mismatches),
        })
        S.log(f"[Live GPU] seed={sd}: calls={pipe.calls}, first_cert={first_cert_wall[0]:.2f}s ({first_cert_calls[0]} calls), "
              f"all_5={total_wall:.2f}s (GPU={timed_op.gpu_seconds:.2f}s, CPU={cpu_engine_s:.2f}s [{100*cpu_engine_s/total_wall:.1f}%]), "
              f"mismatches={timed_op.mismatches}")

    return {
        "gpu_device": gpu_name,
        "batch_size": 64,
        "model_load_s": round(load_s, 2),
        "steady_throughput_rec_per_s": round(live_rec_per_s, 2),
        "full_pool_live_equiv_s": round(est_full_pool_s, 1),
        "runs": runs,
        "mean_first_cert_wall_s": round(float(np.mean([r["first_cert_wall_s"] for r in runs])), 2),
        "mean_all_5_wall_s": round(float(np.mean([r["all_5_wall_s"] for r in runs])), 2),
        "mean_gpu_inference_s": round(float(np.mean([r["gpu_inference_s"] for r in runs])), 2),
        "mean_cohort_cpu_s": round(float(np.mean([r["cohort_cpu_s"] for r in runs])), 2),
        "total_mismatches": sum(r["tape_mismatches"] for r in runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    env = S.load_wands_env()
    t1_8r = evaluate_8round_task(
        "Task1_Relevance_8Rounds_24Targets",
        env,
        lambda: build_task1_8rounds(env),
        lambda sd: S.bm25_history(env, 0.15, sd),
        seeds=tuple(range(10)),
    )

    env3, build_t2_8r, hist3 = build_task2_env_and_8rounds(env)
    t2_8r = evaluate_8round_task(
        "Task2_Cascade_8Rounds_24Targets",
        env3,
        build_t2_8r,
        hist3,
        seeds=tuple(range(10)),
    )

    overlap = evaluate_overlap_regimes(env, seeds=tuple(range(10)))
    coupled_lp = evaluate_coupled_lp_vs_partition(env3, build_t2_8r, hist3, seeds=tuple(range(10)))
    live_gpu = None if args.skip_gpu else evaluate_live_gpu(env, seeds=(0, 1, 2))

    payload = {
        "task1_8rounds": t1_8r,
        "task2_8rounds": t2_8r,
        "overlap_regimes": overlap,
        "coupled_lp_vs_partition": coupled_lp,
        "live_gpu_session": live_gpu,
    }
    S.dump("exp_review4_suite.json", payload)
    S.log("DONE exp_review4_suite.json")


if __name__ == "__main__":
    main()
