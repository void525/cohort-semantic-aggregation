# -*- coding: utf-8 -*-
"""Review-3 critical additions (100% real frozen WANDS tapes: Qwen3-8B + Qwen3-4B).

Experiments produced -> reports/icde2027_suite/:
  1. exp_strong_baselines.json : Strong composite baselines on the 5 distinct numeric targets:
     - Dedup-Streams + Cache + Partition: one shuffled stream per distinct ratio target (4 streams,
       8 CSs at alpha/8), threshold broadcasting, shared label cache, and partition-aware closed-form
       [L_A/(L_A+U_C), U_A/(U_A+L_C)].
     - Shared-Partition (RR): shared random prefix + partition-aware closed-form (no LP solver).
     - COHORT (RR) & COHORT (CERT): shared region + certificate-guided scheduler.
     Evaluated both cold (no history) and with 15% BM25 history (free reads vs known mass).

  2. exp_interleave.json : Mid-stage interleaved targeted retrieval (T) + random prefix (S):
     Simulates an interactive session where after initial random exploration (640 calls), the analyst
     inspects two lexical top-k sub-cohorts (targeted retrieval adding ~2,400 biased pairs into T),
     followed by continued prefix certification of all 5 numeric targets.
     Compares:
     - Restart stage (discard earlier prefix, re-certify at alpha/4 with history as E)
     - Frozen-prefix patch (keep earlier prefix frozen, apply T only as deterministic bounds)
     - COHORT continuous state (continue prefix S; cache hits on T advance pointer at zero cost)

  3. exp_task2_cascade.json : Second distinct semantic task (K=4, J=16 Proxy-Cascade Audit):
     Auditing a fast Qwen3-4B proxy against Qwen3-8B across 31,482 pairs, H=6 strata, 7 compiled
     semantic events (42 sequences), 6 distinct conditional aggregates (relevance precision,
     exact-match precision, relevance recall, exact-match recall, proxy over-claim rate, and
     high-vs-low rating cohort precision difference). Compares Dedup-Streams+Cache+Partition,
     Free-Reads, and COHORT Known-Mass under 4B-positive history, plus coupling tightness of
     the joint region vs pairwise marginal partition bounds.

  4. exp_hk_scale.json : Structural scaling across H in {3, 6, 12, 24} strata and K in {2, 3, 4}
     semantic predicates (J in {4, 8, 16}; LP variables H*J from 12 up to 384). Reports basis size,
     sequence count, CPU time (CS vs projection), peak memory, interval width, and certified targets.

  5. exp_lifecycle.json : Materialization break-even and lifecycle cost analysis:
     Compares incremental call cost (sunk history) vs total lifecycle cost (paying for 15% history +
     new calls) vs full-pool materialization (31,482 calls, 494 s) across session lengths and precision.
"""
import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
import time
import tracemalloc

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import run_icde2027_suite as S
import run_real_data as rrd
from cohortrag import (
    COHORTRAG,
    CertificateGuidedAllocator,
    Config,
    FinitePopulationCS,
    RoundRobinAllocator,
    Snapshot,
    states,
)
from cohortrag.llm_operator import LabelTableOperator
from cohortrag.pipeline import QueryResult
from cohortrag.workload import Query

SECOND_SQLITE = os.path.join(
    _HERE, "..", "reports", "second_model_20261103", "labels_wands_second_model.sqlite"
)
KAPPA_4B = "4a928f82e4c223302a7dfdbe1e1361b8f55f0f75accbec4d04a4ee2847339fe0"


def load_4b_labels():
    conn = sqlite3.connect(SECOND_SQLITE)
    rows = conn.execute("SELECT rid, g FROM labels WHERE kappa=?", (KAPPA_4B,)).fetchall()
    conn.close()
    return {int(r): np.array(json.loads(g), dtype=np.int8) for r, g in rows}


def numeric_wl(env):
    return [q for q in S.fresh_workload(env) if q.kind in ("value", "difference")]


# =========================================================================== #
# 1. Strong Composite Baselines (Dedup-Streams + Cache + Partition vs Shared)
# =========================================================================== #
def run_dedup_streams_partition(env, budget, seed, wl=None, init_cache=None):
    """Natural strong baseline (Review 3 W2):
    - Deduplicates identical ratio expressions (the 5 numeric targets need only 4 distinct ratio streams:
      all, rated>=4, rated<4, rating_share; cohort difference reuses rated>=4 and rated<4!).
    - Shares a global label cache across all streams (and pre-populates init_cache if given).
    - Uses partition-aware closed-form [L_A/(L_A+U_C), U_A/(U_A+L_C)] on each stream (maintaining CS on A and C=B-A).
    - Splits session risk alpha equally across the 8 CSs (alpha/8).
    """
    wl = wl or numeric_wl(env)
    snap, contract = env["snap"], env["contract"]
    strata_ids = {h: [int(i) for i in snap.stratum_ids(h)] for h in snap.strata()}
    id_h = env["stratum_of"]
    cache = dict(init_cache) if init_cache else {}
    calls = [0]

    def get_label(rid):
        if rid in cache:
            return cache[rid]
        if calls[0] >= budget:
            return None
        lab = np.asarray(contract.judge([rid], snap.texts([rid]))[0], dtype=np.int8)
        cache[rid] = lab
        calls[0] += 1
        return lab

    # Deduplicate distinct ratio sides across all queries in wl
    # Key: (scope, a_tuple_per_h, b_tuple)
    ratio_specs = {}
    q_sides = {}

    def reg_side(scope, a_hz, b_tab):
        key = (
            tuple(scope),
            tuple((h, tuple(int(round(x)) for x in a_hz[h])) for h in scope),
            tuple(int(round(x)) for x in b_tab),
        )
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
                "obsA": 0,
                "obsC": 0,
            }
        return key

    for q in wl:
        k1 = reg_side(q.scope, q.a_hz, q.tables["b"])
        if q.kind == "difference":
            k2 = reg_side(q.scope2, q.a_hz2, q.tables["b2"])
            q_sides[q.qid] = (k1, k2)
        else:
            q_sides[q.qid] = (k1,)

    n_cs = 2 * len(ratio_specs)
    delta = S.ALPHA / n_cs

    # Also incorporate exact deterministic bounds from cache!
    def side_interval(key):
        s = ratio_specs[key]
        ivA = s["csA"].interval(delta)
        ivC = s["csC"].interval(delta)
        if ivA is None or ivC is None:
            return None
        # Deterministic known counts inside this stream's scope from cache
        detA = detC = det_seen = 0
        scope_set = set(s["scope"])
        for rid, lab in cache.items():
            h = id_h[int(rid)]
            if h in scope_set:
                det_seen += 1
                z = states.encode_state(tuple(int(v) for v in lab))
                av = int(round(s["a"][h][z]))
                bv = int(round(s["b"][z]))
                detA += av
                detC += bv - av
        unseen = len(s["perm"]) - det_seen
        loA = max(ivA[0], detA)
        hiA = min(ivA[1], detA + unseen)
        loC = max(ivC[0], detC)
        hiC = min(ivC[1], detC + unseen)
        if hiA + hiC <= 0:
            return ("undef", None, None)
        if loA + loC <= 0:
            return ("insuf", 0.0, 1.0)
        L = loA / (loA + hiC) if (loA + hiC) > 0 else 0.0
        U = hiA / (hiA + loC) if (hiA + loC) > 0 else 1.0
        return ("ok", max(0.0, L - 1e-9), min(1.0, U + 1e-9))

    results = {q.qid: QueryResult(qid=q.qid, name=q.name, kind=q.kind) for q in wl}
    unresolved = list(wl)
    trace = []

    def check_all():
        for q in list(unresolved):
            r = results[q.qid]
            keys = q_sides[q.qid]
            s1 = side_interval(keys[0])
            if s1 is None or s1[0] != "ok":
                continue
            if q.kind == "difference":
                s2 = side_interval(keys[1])
                if s2 is None or s2[0] != "ok":
                    continue
                L, U = s1[1] - s2[2], s1[2] - s2[1]
            else:
                L, U = s1[1], s1[2]
            r.L, r.U = L, U
            if q.kind in ("value", "difference") and U - L <= 2 * q.eps:
                r.status, r.point, r.calls_at_cert = "certified", 0.5 * (L + U), calls[0]
                unresolved.remove(q)
            elif q.kind == "threshold":
                if L > q.tau:
                    r.status, r.calls_at_cert = "certified_true", calls[0]
                    unresolved.remove(q)
                elif U <= q.tau:
                    r.status, r.calls_at_cert = "certified_false", calls[0]
                    unresolved.remove(q)
        trace.append((calls[0], S.snap_results(results)))

    check_all()
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
                h = id_h[int(rid)]
                z = states.encode_state(tuple(int(v) for v in lab))
                av = int(round(s["a"][h][z]))
                bv = int(round(s["b"][z]))
                s["csA"].push(av)
                s["csC"].push(bv - av)
                s["pos"] = p + 1
                progressed = True
        check_all()
        if not progressed:
            break

    return results, calls[0], wl, trace


def exp_strong_baselines(env, seeds=tuple(range(10))):
    """Compare strong composite baselines cold and with 15% BM25 history on the 5 numeric targets."""
    S.log("=== [1] Running Strong Composite Baselines ===")
    wl_num = numeric_wl(env)
    theta_num = {
        q.qid: rrd._query_truth(q, env["labels_M"], env["ids_all"], env["stratum_of"])
        for q in wl_num
    }
    cap_cold = env["N"]
    cap_hist = int(0.55 * env["N"])
    out = {"cold": {}, "bm25_15pct": {}, "seeds": list(seeds)}

    methods_cold = [
        ("Dedup-Streams+Cache+Partition", lambda sd: run_dedup_streams_partition(env, cap_cold, sd, wl=wl_num)),
        ("Shared-Prefix+Partition (RR)", lambda sd: S.run_cohort(env, RoundRobinAllocator(), cap_cold, sd, wl=wl_num)[:3] + (None,)),
        ("COHORT (CERT)", lambda sd: S.run_cohort(env, CertificateGuidedAllocator(), cap_cold, sd, wl=wl_num)[:3] + (None,)),
    ]
    for name, fn in methods_cold:
        rows = []
        for sd in seeds:
            res, calls, wl, _ = fn(sd)
            done_calls = [res[q.qid].calls_at_cert for q in wl if res[q.qid].status.startswith("certified")]
            first_cert = min(done_calls) if done_calls else None
            all_cert = max(done_calls) if len(done_calls) == len(wl) else None
            err = sum(
                1 for q in wl
                if res[q.qid].status.startswith("certified")
                and not (res[q.qid].L - 1e-8 <= theta_num[q.qid] <= res[q.qid].U + 1e-8)
            )
            per_q = {str(q.qid): res[q.qid].calls_at_cert for q in wl}
            rows.append({
                "seed": sd, "calls_used": int(calls), "first_cert": first_cert,
                "all_5_cert": all_cert, "n_done": len(done_calls), "errors": err, "per_q": per_q,
            })
        done_all = [r["all_5_cert"] for r in rows if r["all_5_cert"] is not None]
        firsts = [r["first_cert"] for r in rows if r["first_cert"] is not None]
        out["cold"][name] = {
            "n_done_all": f"{len(done_all)}/{len(seeds)}",
            "median_all_5": float(statistics.median(done_all)) if done_all else None,
            "mean_all_5": float(np.mean(done_all)) if done_all else None,
            "median_first": float(statistics.median(firsts)) if firsts else None,
            "errors": sum(r["errors"] for r in rows),
            "rows": rows,
        }
        S.log(f"[strong/cold] {name}: done={len(done_all)}/{len(seeds)} med_all5={out['cold'][name]['median_all_5']} med_first={out['cold'][name]['median_first']}")

    # Now with 15% BM25 history!
    for sd in seeds:
        hist = S.bm25_history(env, 0.15, sd)
        # 1) Dedup-Streams + Cache + Partition (with BM25 history in shared cache & deterministic bounds)
        r_ds, c_ds, wl_ds, _ = run_dedup_streams_partition(env, cap_hist, sd, wl=numeric_wl(env), init_cache=hist)
        # 2) Shared-Prefix + Free Reads + Partition (RR)
        r_fr, c_fr, wl_fr, _ = S.run_cohort(env, RoundRobinAllocator(), cap_hist, sd, wl=numeric_wl(env), tcache=hist)
        # 3) COHORT Known Mass (RR)
        r_km, c_km, wl_km, _ = S.run_cohort(env, RoundRobinAllocator(), cap_hist, sd, wl=numeric_wl(env), cache=hist)
        # 4) COHORT Known Mass + CERT scheduler
        r_co, c_co, wl_co, _ = S.run_cohort(env, CertificateGuidedAllocator(), cap_hist, sd, wl=numeric_wl(env), cache=hist)

        for name, (res, calls, wl) in [
            ("Dedup-Streams+Cache+Partition", (r_ds, c_ds, wl_ds)),
            ("Shared+FreeReads+Partition (RR)", (r_fr, c_fr, wl_fr)),
            ("COHORT KnownMass (RR)", (r_km, c_km, wl_km)),
            ("COHORT KnownMass (CERT)", (r_co, c_co, wl_co)),
        ]:
            done_calls = [res[q.qid].calls_at_cert for q in wl if res[q.qid].status.startswith("certified")]
            all_cert = max(done_calls) if len(done_calls) == len(wl) else None
            first_cert = min(done_calls) if done_calls else None
            err = sum(
                1 for q in wl
                if res[q.qid].status.startswith("certified")
                and not (res[q.qid].L - 1e-8 <= theta_num[q.qid] <= res[q.qid].U + 1e-8)
            )
            out["bm25_15pct"].setdefault(name, {"rows": []})["rows"].append({
                "seed": sd, "calls_used": int(calls), "first_cert": first_cert,
                "all_5_cert": all_cert, "n_done": len(done_calls), "errors": err,
                "per_q": {str(q.qid): res[q.qid].calls_at_cert for q in wl},
            })

    base_rows = out["bm25_15pct"]["Shared+FreeReads+Partition (RR)"]["rows"]
    ds_rows = out["bm25_15pct"]["Dedup-Streams+Cache+Partition"]["rows"]
    for name, dsub in out["bm25_15pct"].items():
        rows = dsub["rows"]
        done_all = [r["all_5_cert"] for r in rows if r["all_5_cert"] is not None]
        firsts = [r["first_cert"] for r in rows if r["first_cert"] is not None]
        sav_vs_fr = [
            100.0 * (1.0 - r["all_5_cert"] / b["all_5_cert"])
            for r, b in zip(rows, base_rows)
            if r["all_5_cert"] is not None and b["all_5_cert"] is not None
        ]
        sav_vs_ds = [
            100.0 * (1.0 - r["all_5_cert"] / d["all_5_cert"])
            for r, d in zip(rows, ds_rows)
            if r["all_5_cert"] is not None and d["all_5_cert"] is not None
        ]
        dsub.update({
            "n_done_all": f"{len(done_all)}/{len(seeds)}",
            "median_all_5": float(statistics.median(done_all)) if done_all else None,
            "median_first": float(statistics.median(firsts)) if firsts else None,
            "median_saving_vs_freereads_pct": float(statistics.median(sav_vs_fr)) if sav_vs_fr else None,
            "median_saving_vs_dedup_streams_pct": float(statistics.median(sav_vs_ds)) if sav_vs_ds else None,
            "errors": sum(r["errors"] for r in rows),
        })
        S.log(
            f"[strong/bm25] {name}: done={dsub['n_done_all']} med_all5={dsub['median_all_5']} "
            f"vs_DS={dsub['median_saving_vs_dedup_streams_pct']:.1f}% vs_FR={dsub['median_saving_vs_freereads_pct']:.1f}% err={dsub['errors']}"
        )

    S.dump("exp_strong_baselines.json", out)
    return out


# =========================================================================== #
# 2. Mid-Stage Interleaved Targeted Retrieval (T) + Random Prefix (S)
# =========================================================================== #
def exp_interleave(env, seeds=tuple(range(10))):
    """Analyst session with mid-stage targeted retrieval:
    Phase 1: Analyst issues initial ratio queries; system runs random prefix for b1 = 1,280 calls (~4% N).
    Mid-stage: Analyst inspects lexical BM25 top-k results for 48 intents (~3,000 records). Some are already
               cached from Phase 1; uncached ones are paid as targeted exposures (T).
    Phase 2: Analyst asks to finish certifying all 5 numeric targets (cap = 18,000 total calls).
    Compare 3 valid protocols:
      A. Continuous State (COHORT): keeps stage open (risk alpha), injects retrieved records into T,
         continues random prefix S (cache hits on T advance pointer at 0 calls + 1 obs).
      B. Stage Restart (Known-Mass): Phase 1 at alpha/2; after retrieval, opens Stage 2 at alpha/4 with
         all Phase-1 + retrieved labels as Known Mass E, drawing fresh permutations on remainder.
      C. Static Audit + Deterministic Patch: freezes statistical prefix after Phase 1, applies T only
         as deterministic bounds (no pointer continuation) -> shows how many targets remain undecided;
         and Fresh-Prefix without Cache-Hit Pointer (restarts S from scratch without E removal).
    """
    S.log("=== [2] Running Mid-Stage Interleaved Targeted Exposure Experiment ===")
    wl_ref = numeric_wl(env)
    theta_num = {
        q.qid: rrd._query_truth(q, env["labels_M"], env["ids_all"], env["stratum_of"])
        for q in wl_ref
    }
    b1 = 1280
    hist_frac = 0.12  # ~3,777 targeted records inspected mid-stage
    cap_rem = int(0.55 * env["N"])
    rows = []

    for sd in seeds:
        target_set = S.bm25_history(env, hist_frac, sd + 300)

        # --- Protocol A: Continuous COHORT state (single stage at alpha) ---
        wlA = numeric_wl(env)
        pipeA = COHORTRAG(env["snap"], env["contract"], Config(alpha=S.ALPHA), seed=sd)
        stateA = {"injected": False, "t_paid": 0, "hits_before": 0}

        def feedA(p, results):
            if not stateA["injected"]:
                stateA["injected"] = True
                stateA["hits_before"] = sum(p.ledger.hits.values())
                by_h = {}
                t_paid = 0
                for rid, lab in target_set.items():
                    rid = int(rid)
                    h = env["stratum_of"][rid]
                    if rid in p.ledger.S_ids[h]:
                        continue  # already visited by random prefix in Phase 1 (0 extra call!)
                    if rid not in p.ledger.cache:
                        t_paid += 1
                    by_h.setdefault(h, []).append((rid, lab))
                for h, items in by_h.items():
                    p.ledger.add_targeted(h, [i for i, _ in items], [l for _, l in items])
                stateA["t_paid"] = t_paid
                # Return empty new_qs? Wait: pipeline.run breaks if new_qs is empty!
                # Instead, in Phase 1 we ask R1=[q1, q4], and in Phase 2 we add R2=[q2, q3, q5]!
                return [q for q in wlA if q.qid in (2, 3, 5)], cap_rem
            return [], None

        r1A = [q for q in wlA if q.qid in (1, 4)]
        resA = pipeA.run(
            r1A, CertificateGuidedAllocator(), budget_total=b1, declared=wlA, feed=feedA
        )
        hits_on_T = sum(pipeA.ledger.hits.values()) - stateA["hits_before"]
        doneA = [resA[q.qid].calls_at_cert for q in wlA if resA[q.qid].status.startswith("certified")]
        totalA = (max(doneA) + stateA["t_paid"]) if len(doneA) == 5 else None
        errA = sum(
            1 for q in wlA
            if resA[q.qid].status.startswith("certified")
            and not (resA[q.qid].L - 1e-8 <= theta_num[q.qid] <= resA[q.qid].U + 1e-8)
        )

        # --- Protocol B: Stage Restart with Known Mass (Phase 1 at alpha/2, Stage 2 at alpha/4) ---
        wlB1 = numeric_wl(env)
        pipeB1 = COHORTRAG(env["snap"], env["contract"], Config(alpha=S.ALPHA / 2), seed=sd)
        pipeB1.run([q for q in wlB1 if q.qid in (1, 4)], CertificateGuidedAllocator(), budget_total=b1, declared=wlB1)
        paid_b1 = pipeB1.calls
        known_B = dict(pipeB1.ledger.cache)
        t_paid_B = 0
        for rid, lab in target_set.items():
            rid = int(rid)
            if rid not in known_B:
                known_B[rid] = lab
                t_paid_B += 1
        wlB2 = numeric_wl(env)
        pipeB2 = COHORTRAG(env["snap"], env["contract"], Config(alpha=S.ALPHA / 4), seed=sd + 500, cache_labels=known_B)
        resB2 = pipeB2.run(wlB2, CertificateGuidedAllocator(), budget_total=cap_rem)
        doneB = [resB2[q.qid].calls_at_cert for q in wlB2 if resB2[q.qid].status.startswith("certified")]
        totalB = (paid_b1 + t_paid_B + max(doneB)) if len(doneB) == 5 else None
        errB = sum(
            1 for q in wlB2
            if resB2[q.qid].status.startswith("certified")
            and not (resB2[q.qid].L - 1e-8 <= theta_num[q.qid] <= resB2[q.qid].U + 1e-8)
        )

        # --- Protocol C: Stage Restart with Free Reads only (no E removal, Stage 2 at alpha/4) ---
        wlC2 = numeric_wl(env)
        pipeC2 = COHORTRAG(env["snap"], env["contract"], Config(alpha=S.ALPHA / 4), seed=sd + 500)
        by_hC = {}
        for rid, lab in known_B.items():
            by_hC.setdefault(env["stratum_of"][int(rid)], []).append((int(rid), lab))
        for h, items in by_hC.items():
            pipeC2.ledger.add_targeted(h, [i for i, _ in items], [l for _, l in items])
        resC2 = pipeC2.run(wlC2, CertificateGuidedAllocator(), budget_total=cap_rem)
        doneC = [resC2[q.qid].calls_at_cert for q in wlC2 if resC2[q.qid].status.startswith("certified")]
        totalC = (paid_b1 + t_paid_B + max(doneC)) if len(doneC) == 5 else None
        errC = sum(
            1 for q in wlC2
            if resC2[q.qid].status.startswith("certified")
            and not (resC2[q.qid].L - 1e-8 <= theta_num[q.qid] <= resC2[q.qid].U + 1e-8)
        )

        rows.append({
            "seed": sd,
            "continuous_cohort_total": totalA,
            "continuous_s_calls": pipeA.calls,
            "continuous_t_paid": stateA["t_paid"],
            "pointer_cache_hits_on_T": int(hits_on_T),
            "restart_known_mass_total": totalB,
            "restart_free_reads_total": totalC,
            "errors": errA + errB + errC,
        })
        S.log(
            f"[interleave] seed={sd}: Continuous={totalA} (hits_on_T={hits_on_T}) | "
            f"Restart-E={totalB} | Restart-FreeReads={totalC} | err={errA+errB+errC}"
        )

    med = lambda k: float(statistics.median([r[k] for r in rows if r[k] is not None]))
    sav_vs_re_e = [
        100.0 * (1.0 - r["continuous_cohort_total"] / r["restart_known_mass_total"])
        for r in rows if r["continuous_cohort_total"] and r["restart_known_mass_total"]
    ]
    sav_vs_re_fr = [
        100.0 * (1.0 - r["continuous_cohort_total"] / r["restart_free_reads_total"])
        for r in rows if r["continuous_cohort_total"] and r["restart_free_reads_total"]
    ]
    out = {
        "b1": b1,
        "targeted_frac": hist_frac,
        "n_seeds": len(seeds),
        "median_continuous": med("continuous_cohort_total"),
        "median_restart_known_mass": med("restart_known_mass_total"),
        "median_restart_free_reads": med("restart_free_reads_total"),
        "median_pointer_hits": med("pointer_cache_hits_on_T"),
        "median_saving_vs_restart_E_pct": float(statistics.median(sav_vs_re_e)),
        "median_saving_vs_restart_FR_pct": float(statistics.median(sav_vs_re_fr)),
        "total_errors": sum(r["errors"] for r in rows),
        "rows": rows,
    }
    S.dump("exp_interleave.json", out)
    return out


# =========================================================================== #
# 3. Second Real Semantic Task: Multi-Predicate Proxy-Cascade Audit (K=4, J=16)
# =========================================================================== #
def make_cascade_workload(env):
    all_h = tuple(sorted(env["meta"]["all_strata"]))
    ge4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if ">=4" in str(lab) for h in hs}))
    lt4 = tuple(sorted({h for lab, hs in env["by_rating"].items() if "<4" in str(lab) for h in hs}))
    # Primitives: 0=4B_rel, 1=4B_ex, 2=8B_rel, 3=8B_ex
    qs = [
        Query(1, "precision_rel (8B_rel | 4B_rel)", all_h, (0,), (2,), eps=0.05),
        Query(2, "precision_ex (8B_ex | 4B_ex)", all_h, (0, 1), (2, 3), eps=0.05),
        Query(3, "recall_rel (4B_rel | 8B_rel)", all_h, (2,), (0,), eps=0.05),
        Query(4, "recall_ex (4B_ex | 8B_ex)", all_h, (2, 3), (0, 1), eps=0.10),
        Query(5, "overclaim (4B_ex | 8B_rel)", all_h, (2,), (0, 1), eps=0.05),
        Query(
            6, "diff_prec_ex (>=4 minus <4)", ge4, (0, 1), (2, 3),
            kind="difference", eps=0.10, scope2=lt4, d_idx2=(0, 1), r_idx2=(2, 3)
        ),
    ]
    for q in qs:
        q.tables = q.build_tables(4)
        q.a_hz = q.numerator_tables(4, q.scope)
        if q.kind == "difference":
            q.a_hz2 = q.numerator_tables(4, q.scope2, meta=q.r_meta2, tables={"a": q.tables["a2"]})
    return qs


def exp_task2_cascade(env, seeds=tuple(range(8))):
    """Second distinct real task: K=4, J=16 Proxy-Cascade Audit over 31,482 pairs.
    History: 15% of records prioritized by proxy-exact predictions (simulating an audit log where
    pairs flagged as exact by the cheap 4B filter were previously escalated to 8B).
    Compares:
      - Dedup-Streams + Cache + Partition (5 distinct ratio streams, 10 CSs, shared cache)
      - Shared + Free Reads + Partition (RR)
      - COHORT Known Mass (RR)
      - COHORT Known Mass (CERT)
    """
    S.log("=== [3] Running Task 2: K=4 Multi-Predicate Proxy-Cascade Audit ===")
    l8 = env["labels_M"]
    l4 = load_4b_labels()
    ids = [int(i) for i in env["ids_all"]]
    lab4 = {
        i: np.array([int(l4[i][0]), int(l4[i][0] and l4[i][1]), int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8)
        for i in ids
    }
    contract4 = LabelTableOperator(lab4, name="cascade-4b-8b", K=4)
    env4 = dict(env)
    env4["contract"] = contract4
    env4["labels_M"] = lab4
    wl_ref = make_cascade_workload(env4)
    theta4 = {
        q.qid: rrd._query_truth(q, lab4, env4["ids_all"], env4["stratum_of"])
        for q in wl_ref
    }
    S.log("Task 2 truths: " + ", ".join(f"Q{k}={v:.4f}" for k, v in theta4.items()))

    cap = int(0.65 * env4["N"])
    n_hist = int(round(0.15 * env4["N"]))
    proxy_ex_ids = [i for i in ids if lab4[i][1] == 1]
    other_ids = [i for i in ids if lab4[i][1] == 0]

    methods = {
        "Dedup-Streams+Cache+Partition": [],
        "Shared+FreeReads (RR)": [],
        "COHORT KnownMass (RR)": [],
        "COHORT KnownMass (CERT)": [],
    }

    for sd in seeds:
        t0 = time.time()
        rng = np.random.default_rng(sd * 997 + 41)
        # History: sample heavily from proxy_exact escalated pairs (85% from proxy_ex, 15% from rest)
        n_pe = min(len(proxy_ex_ids), int(0.85 * n_hist))
        picked = list(rng.choice(proxy_ex_ids, size=n_pe, replace=False)) + list(
            rng.choice(other_ids, size=n_hist - n_pe, replace=False)
        )
        hist = {int(i): lab4[int(i)] for i in picked}

        # 1) Dedup-Streams + Cache + Partition
        r_ds, c_ds, wl_ds, _ = run_dedup_streams_partition(
            env4, cap, sd, wl=make_cascade_workload(env4), init_cache=hist
        )
        # 2) Shared + FreeReads (RR)
        r_fr, c_fr, wl_fr, _ = S.run_cohort(
            env4, RoundRobinAllocator(), cap, sd, wl=make_cascade_workload(env4), tcache=hist
        )
        # 3) COHORT KnownMass (RR)
        r_km, c_km, wl_km, _ = S.run_cohort(
            env4, RoundRobinAllocator(), cap, sd, wl=make_cascade_workload(env4), cache=hist
        )
        # 4) COHORT KnownMass (CERT)
        r_co, c_co, wl_co, pipe_co = S.run_cohort(
            env4, CertificateGuidedAllocator(), cap, sd, wl=make_cascade_workload(env4), cache=hist
        )

        for name, (res, calls, wl) in [
            ("Dedup-Streams+Cache+Partition", (r_ds, c_ds, wl_ds)),
            ("Shared+FreeReads (RR)", (r_fr, c_fr, wl_fr)),
            ("COHORT KnownMass (RR)", (r_km, c_km, wl_km)),
            ("COHORT KnownMass (CERT)", (r_co, c_co, wl_co)),
        ]:
            done_calls = [res[q.qid].calls_at_cert for q in wl if res[q.qid].status.startswith("certified")]
            all_cert = max(done_calls) if len(done_calls) == len(wl) else None
            first_cert = min(done_calls) if done_calls else None
            err = sum(
                1 for q in wl
                if res[q.qid].status.startswith("certified")
                and not (res[q.qid].L - 1e-8 <= theta4[q.qid] <= res[q.qid].U + 1e-8)
            )
            methods[name].append({
                "seed": sd, "calls_used": int(calls), "first_cert": first_cert,
                "all_6_cert": all_cert, "n_done": len(done_calls), "errors": err,
                "per_q": {str(q.qid): res[q.qid].calls_at_cert for q in wl},
            })
        S.log(
            f"[task2] seed={sd} ({time.time()-t0:.1f}s): "
            + " | ".join(f"{m}={methods[m][-1]['all_6_cert']}({methods[m][-1]['n_done']}/6)" for m in methods)
        )

    out = {"N": env4["N"], "K": 4, "J": 16, "H": 6, "n_basis": 7, "n_cs": 42, "truths": {str(k): v for k, v in theta4.items()}, "methods": {}}
    fr_rows = methods["Shared+FreeReads (RR)"]
    ds_rows = methods["Dedup-Streams+Cache+Partition"]
    for name, rows in methods.items():
        done_all = [r["all_6_cert"] for r in rows if r["all_6_cert"] is not None]
        firsts = [r["first_cert"] for r in rows if r["first_cert"] is not None]
        sav_fr = [
            100.0 * (1.0 - r["all_6_cert"] / b["all_6_cert"])
            for r, b in zip(rows, fr_rows)
            if r["all_6_cert"] is not None and b["all_6_cert"] is not None
        ]
        sav_ds = [
            100.0 * (1.0 - r["all_6_cert"] / d["all_6_cert"])
            for r, d in zip(rows, ds_rows)
            if r["all_6_cert"] is not None and d["all_6_cert"] is not None
        ]
        out["methods"][name] = {
            "done": f"{len(done_all)}/{len(seeds)}",
            "median_all_6": float(statistics.median(done_all)) if done_all else None,
            "median_first": float(statistics.median(firsts)) if firsts else None,
            "median_saving_vs_freereads_pct": float(statistics.median(sav_fr)) if sav_fr else None,
            "median_saving_vs_dedup_streams_pct": float(statistics.median(sav_ds)) if sav_ds else None,
            "errors": sum(r["errors"] for r in rows),
            "rows": rows,
        }
        S.log(
            f"[task2/summary] {name}: done={out['methods'][name]['done']} med_all6={out['methods'][name]['median_all_6']} "
            f"vs_FR={out['methods'][name]['median_saving_vs_freereads_pct']} vs_DS={out['methods'][name]['median_saving_vs_dedup_streams_pct']}"
        )

    S.dump("exp_task2_cascade.json", out)
    return out


# =========================================================================== #
# 4. Structural Scaling over H (strata) and K (semantic predicates)
# =========================================================================== #
def exp_hk_scale(env, budget=3200, seeds=(0, 1, 2)):
    """Sweep H in {3, 6, 12, 24} and K in {2, 3, 4} on the 31,482 WANDS pool."""
    S.log("=== [4] Running Structural Scaling (H, K, |F|) ===")
    l8 = env["labels_M"]
    l4 = load_4b_labels()
    ids = [int(i) for i in env["ids_all"]]
    pc = S.product_class_map()
    f_base = env["frame"].copy()
    f_base["cls"] = f_base["product_id"].astype(int).map(pc).fillna("missing")
    top_cls = f_base["cls"].value_counts().index[:7].tolist()

    def make_h_frame(H):
        f = f_base.copy()
        rb = f["rating_bucket"].map({"<4": 0, ">=4": 1, "missing": 2}).astype(int)
        if H == 3:
            f["stratum"] = rb
        elif H == 6:
            ig = (f["intent_id"].astype(int) % 2)
            f["stratum"] = rb * 2 + ig
        elif H == 12:
            ig = (f["intent_id"].astype(int) % 4)
            f["stratum"] = rb * 4 + ig
        elif H == 24:
            cg = f["cls"].apply(lambda c: top_cls.index(c) if c in top_cls else 7)
            f["stratum"] = rb * 8 + cg
        return f[["id", "stratum", "text"]].reset_index(drop=True)

    configs = [
        (3, 2), (6, 2), (12, 2), (24, 2),
        (6, 3), (6, 4), (12, 4), (24, 4),
    ]
    rows_out = []

    for H, K in configs:
        sub_f = make_h_frame(H)
        snap = Snapshot(sub_f, name=f"hk-{H}-{K}")
        stratum_of = dict(zip(sub_f["id"].astype(int), sub_f["stratum"].astype(int)))
        all_h = tuple(sorted(sub_f["stratum"].unique().tolist()))

        if K == 2:
            labs = {i: np.array([int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8) for i in ids}
            qs_proto = lambda: [
                Query(1, "ex|rel", all_h, (0,), (1,), eps=0.05),
                Query(2, "ex|rel_s0", (0,), (0,), (1,), eps=0.10),
            ]
        elif K == 3:
            labs = {
                i: np.array([int(l4[i][0]), int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8)
                for i in ids
            }
            qs_proto = lambda: [
                Query(1, "prec_rel", all_h, (0,), (1,), eps=0.05),
                Query(2, "rec_rel", all_h, (1,), (0,), eps=0.05),
                Query(3, "ex|rel8", all_h, (1,), (2,), eps=0.05),
            ]
        else:  # K == 4
            labs = {
                i: np.array(
                    [int(l4[i][0]), int(l4[i][0] and l4[i][1]), int(l8[i][0]), int(l8[i][0] and l8[i][1])],
                    dtype=np.int8,
                )
                for i in ids
            }
            qs_proto = lambda: [
                Query(1, "prec_rel", all_h, (0,), (2,), eps=0.05),
                Query(2, "prec_ex", all_h, (0, 1), (2, 3), eps=0.05),
                Query(3, "rec_rel", all_h, (2,), (0,), eps=0.05),
                Query(4, "overclaim", all_h, (2,), (0, 1), eps=0.05),
                Query(5, "ex8|rel8", all_h, (2,), (3,), eps=0.05),
            ]

        contract = LabelTableOperator(labs, name=f"op-K{K}", K=K)
        cpus, mems, widths, n_cs_list, n_basis_list = [], [], [], [], []
        for sd in seeds:
            wl = qs_proto()
            for q in wl:
                q.tables = q.build_tables(K)
                q.a_hz = q.numerator_tables(K, q.scope)
            tracemalloc.start()
            t0 = time.process_time()
            pipe = COHORTRAG(snap, contract, Config(alpha=S.ALPHA), seed=sd)
            res = pipe.run(wl, RoundRobinAllocator(), budget_total=budget)
            cpu = time.process_time() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            cpus.append(cpu)
            mems.append(peak / (1024 ** 2))
            widths.append(res[1].U - res[1].L if res[1].L is not None else 1.0)
            n_cs_list.append(pipe.risk_ledger["n_cs"])
            n_basis_list.append(len(pipe.risk_ledger["basis"]))

        rec = {
            "H": H, "K": K, "J": 1 << K, "lp_vars": H * (1 << K),
            "n_basis": int(n_basis_list[0]), "n_cs": int(n_cs_list[0]),
            "budget": budget, "cpu_s_mean": float(np.mean(cpus)),
            "peak_mb_mean": float(np.mean(mems)),
            "q1_width_mean": float(np.mean(widths)),
        }
        rows_out.append(rec)
        S.log(
            f"[hk_scale] H={H} K={K} vars={rec['lp_vars']} basis={rec['n_basis']} n_cs={rec['n_cs']} "
            f"cpu={rec['cpu_s_mean']:.2f}s mem={rec['peak_mb_mean']:.1f}MB width={rec['q1_width_mean']:.3f}"
        )

    S.dump("exp_hk_scale.json", {"rows": rows_out})
    return rows_out


# =========================================================================== #
# 5. Lifecycle & Full-Materialization Break-Even Analysis
# =========================================================================== #
def exp_lifecycle(env):
    """Compute exact incremental vs lifecycle call & GPU-second break-even against full materialization (31,482 calls, 494 s)."""
    S.log("=== [5] Running Lifecycle & Materialization Break-Even Analysis ===")
    paired = json.load(open(os.path.join(S.OUT_DIR, "exp_paired.json")))
    strong = json.load(open(os.path.join(S.OUT_DIR, "exp_strong_baselines.json")))
    session = json.load(open(os.path.join(S.OUT_DIR, "exp_session.json")))

    N = env["N"]
    rate = 63.732  # measured RTX 5090 records/s on WANDS Qwen3-8B
    full_s = N / rate
    hist_n = int(round(0.15 * N))  # 4,722 records

    bm_e = strong["bm25_15pct"]["COHORT KnownMass (CERT)"]["median_all_5"]
    bm_fr = strong["bm25_15pct"]["Shared+FreeReads+Partition (RR)"]["median_all_5"]
    cold_co = strong["cold"]["COHORT (CERT)"]["median_all_5"]
    cold_first = strong["cold"]["COHORT (CERT)"]["median_first"]
    bm_first = strong["bm25_15pct"]["COHORT KnownMass (CERT)"]["median_first"]

    # Four-round session total new calls (32 statements across 2 stages)
    s_calls = float(np.mean([r["traj"][-1]["calls"] for r in session["runs"]["COHORT (E-inherit)"]]))

    out = {
        "N": N,
        "throughput_rec_per_s": rate,
        "full_materialization_calls": N,
        "full_materialization_gpu_s": round(full_s, 1),
        "history_15pct_calls": hist_n,
        "cold_first_numeric_calls": cold_first,
        "cold_first_numeric_gpu_s": round(cold_first / rate, 1),
        "cold_all5_numeric_calls": cold_co,
        "cold_all5_numeric_gpu_s": round(cold_co / rate, 1),
        "sunk_history_all5_calls": bm_e,
        "sunk_history_all5_gpu_s": round(bm_e / rate, 1),
        "sunk_history_saving_vs_full_pct": round(100.0 * (1.0 - bm_e / N), 1),
        "billed_history_lifecycle_all5_calls": hist_n + bm_e,
        "billed_history_lifecycle_saving_vs_full_pct": round(100.0 * (1.0 - (hist_n + bm_e) / N), 1),
        "four_round_session_calls": round(s_calls, 0),
        "four_round_session_gpu_s": round(s_calls / rate, 1),
        "four_round_saving_vs_full_pct": round(100.0 * (1.0 - s_calls / N), 1),
        "breakeven_independent_5target_stages": round(N / cold_co, 1),
        "breakeven_compatible_reslice_rounds": int(math.ceil((N - s_calls) / 2794.0)) + 4,
    }
    S.log(f"[lifecycle] {json.dumps(out, indent=2)}")
    S.dump("exp_lifecycle.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="strong,interleave,task2,hk,lifecycle")
    args = ap.parse_args()
    env = S.load_wands_env()
    which = set(x.strip() for x in args.exp.split(","))
    if "strong" in which:
        exp_strong_baselines(env)
    if "interleave" in which:
        exp_interleave(env)
    if "task2" in which:
        exp_task2_cascade(env)
    if "hk" in which:
        exp_hk_scale(env)
    if "lifecycle" in which:
        exp_lifecycle(env)
    S.log("ALL REVIEW-3 EXPERIMENTS COMPLETED.")


if __name__ == "__main__":
    main()
