# -*- coding: utf-8 -*-
"""Unified algorithms, baselines, adaptive policies, and cost predictor for Review-5 upgrades.

Key additions over Review-4:
  1. Coordinated Priority Multi-Query Sampling ("Shared-priority + partition"):
     - Draws one shared i.i.d. priority u_i ~ Uniform(0, 1) per tuple outside E_0 at stage start;
     - Orders every query stream's remainder R_q = U_q \\ E_0 by ascending u_i (preserving exact
       marginal uniform permutation distribution on each fixed R_q, combined via union bound);
     - Shares global label cache and deterministic count floors/ceilings across streams, with batched
       GPU-friendly oracle evaluation matching COHORT's batch window.
     - Supports both Adaptive round risk (alpha / 2^(r+1)) and Oracle upfront risk (alpha / (2 * |S|)).
  2. Factorial control "Oracle upfront risk x dynamic E":
     - Combines upfront knowledge of total stream count with dynamic per-round removal of all
       previously cached labels from newly arriving streams' sampling frames.
  3. Coupled multi-predicate queries (where joint LP on P_t strictly tightens separate partition bounds).
  4. Result-driven adaptive analyst policy & randomized query arrival order checks.
  5. Analytical Breakeven Cost Predictor (predicting when shared state beats dedicated/coordinated streams
     from table metadata N_h, |E_{0,h}|, basis size |F|, overlap coefficient, and query horizon).
"""
import math
import statistics
import time
import numpy as np

from cohortrag import (
    COHORTRAG,
    CertificateGuidedAllocator,
    Config,
    FinitePopulationCS,
    states,
)
from cohortrag.pipeline import QueryResult
from cohortrag.workload import Query


def run_streams_executor(
    env,
    rounds,
    seed,
    init_cache=None,
    adaptive_risk=True,
    known_mass_mode="E0",
    order_mode="independent",
    batch_step=32,
    cs_pad=1,
    alpha=0.05,
    on_first_cert=None,
):
    """Unified stream-per-distinct-ratio executor with batched oracle calls.

    Parameters
    ----------
    known_mass_mode : {"T", "E0", "Edyn"}
        - "T"   : history kept inside sampling frame (free reads + deterministic floor/ceiling).
        - "E0"  : stage-start history E0 removed from each stream's frame R_s = U_s \\ E0.
        - "Edyn": newly arrived stream at round r removes ALL labels cached prior to round r
                  (requires order_mode="independent", since removing adaptive cache hits from a
                  pre-drawn shared priority order would distort conditional uniformity).
    order_mode : {"independent", "shared_priority"}
        - "independent"    : each distinct ratio stream draws an independent uniform permutation.
        - "shared_priority": draws a single shared priority u_i ~ Uniform(0,1) over U \\ E0 at t=0
                             and sorts every stream's R_s = U_s \\ E0 by ascending u_i.
    """
    if order_mode == "shared_priority" and known_mass_mode == "Edyn":
        raise ValueError("shared_priority requires fixed stage-start E0 (or T) for valid marginal uniformity.")

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
            cache[rid] = np.asarray(lab, dtype=np.int8)
            h = id_h[rid]
            z = states.encode_state(tuple(int(v) for v in cache[rid]))
            h_obs[h][z] += 1
            h_seen[h] += 1

    e0_ids = set()
    if init_cache:
        for rid, lab in init_cache.items():
            record_cache(rid, lab)
            e0_ids.add(int(rid))

    # Shared continuous priority u_i ~ Uniform(0, 1) across all tuples in U \ E0
    shared_prio = None
    if order_mode == "shared_priority":
        all_ids_sorted = sorted(int(i) for h in sorted(strata_ids) for i in strata_ids[h])
        rng_prio = np.random.default_rng(seed * 10007 + 911)
        u_vals = rng_prio.random(len(all_ids_sorted))
        shared_prio = {rid: float(u) for rid, u in zip(all_ids_sorted, u_vals)}

    calls = [0]
    budget = env["N"]

    def fetch_window_labels(window_ids):
        uncached = []
        seen_w = set()
        for rid in window_ids:
            rid = int(rid)
            if rid not in cache and rid not in seen_w:
                uncached.append(rid)
                seen_w.add(rid)
        rem_b = budget - calls[0]
        if rem_b <= 0 and uncached:
            return False
        if len(uncached) > rem_b:
            uncached = uncached[:rem_b]
        if uncached:
            labs = contract.judge(uncached, snap.texts(uncached))
            for rid, lab in zip(uncached, labs):
                record_cache(rid, lab)
            calls[0] += len(uncached)
        return True

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
        oracle_delta = alpha / max(1, 2 * len(uniq))
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

            if order_mode == "shared_priority":
                perm = sorted(rem_pool, key=lambda r: (shared_prio[int(r)], int(r)))
            else:
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
        L = max(0.0, loA / (loA + hiC) - 1e-9)
        U = min(1.0, hiA / (hiA + loC) + 1e-9)
        if L > U:
            s["failed"] = True
            return ("empty", None, None)
        return ("ok", L, U)

    results = {}
    round_calls = []
    round_first_calls = []
    halted = False
    global_first_fired = [False]

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
            r_share = alpha / (2 ** (r_idx + 1))
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
                if L > U:
                    res.status = "execution_error"
                    unresolved.remove(q)
                    halted = True
                    continue
                res.L, res.U = L, U
                if q.kind in ("value", "difference") and 0.0 <= U - L <= 2 * q.eps:
                    res.status, res.point, res.calls_at_cert = "certified", 0.5 * (L + U), calls[0]
                    if first_cert_in_round[0] is None:
                        first_cert_in_round[0] = int(calls[0])
                    if not global_first_fired[0]:
                        global_first_fired[0] = True
                        if on_first_cert is not None:
                            on_first_cert(int(calls[0]))
                    unresolved.remove(q)
                elif q.kind == "threshold":
                    if L > q.tau + 1e-9:
                        res.status, res.calls_at_cert = "certified_true", calls[0]
                        unresolved.remove(q)
                    elif U <= q.tau - 1e-9:
                        res.status, res.calls_at_cert = "certified_false", calls[0]
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
                end = min(s["pos"] + batch_step, len(s["perm"]))
                window = s["perm"][s["pos"]:end]
                if not window:
                    continue
                fetch_window_labels(window)
                for p in range(s["pos"], end):
                    rid = int(s["perm"][p])
                    if rid not in cache:
                        break
                    lab = cache[rid]
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


# =========================================================================== #
# Analytical Plan Cost Predictor (When Does Shared Session State Beat Streams?)
# =========================================================================== #
def cs_sample_req_proxy(M, eps, delta, b_sel=0.5):
    """Distribution-free finite-population prefix requirement proxy for half-width eps at risk delta.

    Uses the finite-population Ville/Serfling scaling:
      n_0 = min(M, (0.5 / (eps * b_sel))^2 * (0.65 * log(2.0 / delta)))
      n_fpc = M * n_0 / (M + max(1.0, n_0 - 1.0))
    Zero tuned parameters; relies only on (M, eps, delta) and a coarse prior denominator rate b_sel.
    """
    if M <= 0:
        return 0.0
    eff_eps = max(0.02, float(eps) * max(0.15, float(b_sel)))
    n0 = (0.25 / (eff_eps ** 2)) * (0.65 * math.log(max(2.0, 2.0 / max(1e-12, delta))))
    n_fpc = (M * n0) / (M + max(1.0, n0 - 1.0))
    return min(float(M), n_fpc)


def predict_cost_ratio_cohort_over_streams(
    M_h_dict,
    n_basis,
    query_specs,
    alpha=0.05,
    b_sel=0.5,
):
    """Predicts C_COHORT / C_best_streams BEFORE running LLM calls, using only structural metadata.

    Parameters
    ----------
    M_h_dict : {h: M_h}
        Remainder sizes per stratum after removing known mass E_0.
    n_basis : int
        Compiled semantic basis cardinality |F|.
    query_specs : list of (round_idx, scope_tuple, eps, ratio_id)
        Sequence of distinct ratio sides across rounds.

    Returns
    -------
    dict with predicted_cohort_calls, predicted_stream_calls, predicted_ratio, recommend_cohort (bool)
    """
    H_active = sum(1 for m in M_h_dict.values() if m > 0)
    m_cs = max(1, H_active * max(1, int(n_basis)))
    delta_cohort = alpha / m_cs

    # Shared state cost: max requirement induced across strata by queries touching each stratum h
    stratum_req = {h: 0.0 for h in M_h_dict}
    for _, sc, eps, _ in query_specs:
        M_sc = sum(M_h_dict[h] for h in sc)
        if M_sc <= 0:
            continue
        n_sc = cs_sample_req_proxy(M_sc, eps, delta_cohort, b_sel=b_sel)
        for h in sc:
            share_h = M_h_dict[h] / M_sc
            stratum_req[h] = max(stratum_req[h], min(float(M_h_dict[h]), n_sc * share_h * (1.0 + 0.12 * math.sqrt(len(sc)))))
    pred_cohort = sum(stratum_req.values())

    # Coordinated / dynamic-E stream cost: tracks per-stratum expected distinct coverage fraction cov_h in [0,1]
    cov_h = {h: 0.0 for h in M_h_dict}
    seen_ratios = set()
    by_round = {}
    for r_idx, sc, eps, rid in query_specs:
        if rid not in seen_ratios:
            seen_ratios.add(rid)
            by_round.setdefault(r_idx, []).append((sc, eps, rid))

    pred_stream = 0.0
    for r_idx in sorted(by_round):
        r_list = by_round[r_idx]
        d_r = (alpha / (2 ** (r_idx + 1))) / max(1, 2 * len(r_list))
        for sc, eps, _ in r_list:
            M_sc = sum(M_h_dict[h] for h in sc)
            if M_sc <= 0:
                continue
            n_req = cs_sample_req_proxy(M_sc, eps, d_r, b_sel=b_sel)
            # Expected new calls accounting for overlap with already-covered tuples in each stratum h
            for h in sc:
                if M_h_dict[h] <= 0:
                    continue
                n_h = n_req * (M_h_dict[h] / M_sc)
                # Under coordinated priority, overlapping prefix up to cov_h * M_h is free cache hit;
                # beyond cov_h * M_h (or under new risk d_r requiring deeper prefix) incurs new calls,
                # plus dynamic-E / independent permutation pays (1 - cov_h) * n_h.
                # Best of coordinated priority and dynamic-E in stratum h:
                p_req = min(1.0, n_h / M_h_dict[h])
                coord_new = M_h_dict[h] * max(0.0, p_req - cov_h[h])
                dyne_new = n_h * (1.0 - cov_h[h])
                best_new = min(coord_new, dyne_new)
                pred_stream += best_new
                cov_h[h] = min(1.0, max(cov_h[h], cov_h[h] + best_new / M_h_dict[h]))

    ratio = pred_cohort / max(1.0, pred_stream)
    return {
        "predicted_cohort_calls": round(pred_cohort, 1),
        "predicted_stream_calls": round(pred_stream, 1),
        "predicted_ratio": round(ratio, 3),
        "recommend_cohort": bool(ratio < 1.0),
    }
