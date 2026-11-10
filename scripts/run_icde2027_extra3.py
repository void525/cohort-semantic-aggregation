# -*- coding: utf-8 -*-
"""Round-3 additions to the ICDE 2027 suite (CPU replay on the frozen WANDS label tape; no GPU).

  audit    exact rational audit on/off over the whole budget grid x seeds: does switching the audit on change
           any certified claim or any endpoint?  (implementation layer of Theorem 1)
  eg       why does claim-by-claim execution lose?  observations that each paid label buys, per claim stream,
           versus the shared random prefix of COHORT
  memory   peak Python memory of the evidence state, region and sequences for a full session
  lotus    COHORT running on top of an existing semantic engine (LOTUS sem_filter) via a replay endpoint
           (run with the venv that has lotus installed)
  bias20   the history sweep with 20 independent sessions per setting (-> exp_bias20.json)

Results are written next to the other experiment outputs (reports/icde2027_suite/*.json).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import run_icde2027_suite as S
from cohortrag import RoundRobinAllocator, CertificateGuidedAllocator, FinitePopulationCS


# --------------------------------------------------------------------------- #
# 1. exact audit on/off across the budget grid
# --------------------------------------------------------------------------- #
def _compare_traces(t0, t1):
    """Entry-by-entry comparison of two anytime traces [(calls, {qid: (status, L, U)})]."""
    same_calls = [c for c, _ in t0] == [c for c, _ in t1]
    n_cmp = n_status_diff = 0
    max_shift = 0.0
    for (_, sa), (_, sb) in zip(t0, t1):
        for qid, (st0, L0, U0) in sa.items():
            st1, L1, U1 = sb[qid]
            n_cmp += 1
            n_status_diff += int(st0 != st1)
            if L0 is not None and L1 is not None:
                max_shift = max(max_shift, abs(L0 - L1), abs(U0 - U1))
    return same_calls, n_cmp, n_status_diff, max_shift


def exp_audit_grid(env, seeds=tuple(range(10))):
    path = "exp_audit_grid.json"
    out = {"seeds": list(seeds), "grid": S.GRID, "per_seed": []}
    tot = {"cells": 0, "cells_same_set": 0, "entries": 0, "status_diff": 0, "max_shift": 0.0,
           "audits": 0, "adjusted": 0, "certified_statements": 0}
    for sd in seeds:
        t0 = time.time()
        runs = {}
        for audit in (False, True):
            cfg = S.Config(alpha=S.ALPHA)
            cfg.audit_exact = audit
            c0 = time.process_time()
            res, calls, wl, pipe = S.run_cohort(env, CertificateGuidedAllocator(), env["N"], sd, cfg=cfg)
            runs[audit] = {"trace": pipe.trace, "wl": wl, "cpu": time.process_time() - c0,
                           "audit_log": list(pipe.audit_log), "res": res}
        same_calls, n_cmp, n_diff, shift = _compare_traces(runs[False]["trace"], runs[True]["trace"])
        same_sets = []
        for b in S.GRID:
            m0 = S.metrics_at(runs[False]["trace"], runs[False]["wl"], env["theta_M"], b)
            m1 = S.metrics_at(runs[True]["trace"], runs[True]["wl"], env["theta_M"], b)
            same_sets.append(m0["cert_qids"] == m1["cert_qids"])
            tot["certified_statements"] += m1["n_cert"]
        alog = runs[True]["audit_log"]
        n_adj = sum(1 for e in alog if e[1] > 0)
        rec = {"seed": sd, "same_trace_calls": same_calls, "entries": n_cmp, "status_diff": n_diff,
               "max_endpoint_shift": shift, "same_certified_set_by_budget": same_sets,
               "cpu_without": runs[False]["cpu"], "cpu_with": runs[True]["cpu"],
               "audits": len(alog), "adjusted_or_rejected": n_adj}
        out["per_seed"].append(rec)
        tot["cells"] += len(S.GRID)
        tot["cells_same_set"] += int(sum(same_sets))
        tot["entries"] += n_cmp
        tot["status_diff"] += n_diff
        tot["max_shift"] = max(tot["max_shift"], shift)
        tot["audits"] += len(alog)
        tot["adjusted"] += n_adj
        out["total"] = tot
        S.dump(path, out)
        S.log(f"[audit] seed={sd} audits={len(alog)} adjusted/rejected={n_adj} status_diff={n_diff} "
              f"max_shift={shift:.2e} same_sets={int(sum(same_sets))}/{len(S.GRID)} "
              f"cpu {rec['cpu_without']:.1f}s -> {rec['cpu_with']:.1f}s ({time.time() - t0:.0f}s)")
    return out


# --------------------------------------------------------------------------- #
# 2. why claim-by-claim execution loses: what does a paid label buy?
# --------------------------------------------------------------------------- #
class _CountingCS(FinitePopulationCS):
    registry = []

    def __init__(self, N):
        super().__init__(N)
        _CountingCS.registry.append(self)


def exp_eg_reuse(env, budgets=(640, 1536, 3200), seeds=tuple(range(5))):
    """For the claim-by-claim executor: length of every claim's own observation stream at a given number of paid calls.
    For COHORT: length of the shared random prefix that every claim reads."""
    path = "exp_eg_reuse.json"
    out = {"budgets": list(budgets), "seeds": list(seeds), "rows": {}}
    orig = S.FinitePopulationCS
    for b in budgets:
        eg_rows, co_rows = [], []
        for sd in seeds:
            _CountingCS.registry = []
            S.FinitePopulationCS = _CountingCS
            try:
                res, calls, wl, _ = S.run_eg_style(env, b, sd)
            finally:
                S.FinitePopulationCS = orig
            reg = _CountingCS.registry
            # registry order: per claim, per spec: csA then csB
            streams, i = [], 0
            for q in wl:
                n_spec = 2 if q.kind == "difference" else 1
                lens = []
                for _ in range(n_spec):
                    lens.append(reg[i].n)
                    i += 2
                streams.append(min(lens))
            pushes = int(sum(reg[j].n for j in range(0, len(reg), 2)))   # every (stream, record) observation
            eg_rows.append({"paid": int(calls), "headline_stream": int(streams[0]),
                            "mean_stream": float(np.mean(streams)), "median_stream": float(np.median(streams)),
                            "total_obs": pushes,
                            "certified": sum(1 for r in res.values() if r.status.startswith("certified"))})
            resc, callsc, wlc, pipe = S.run_cohort(env, CertificateGuidedAllocator(), b, sd)
            prefix = int(sum(pipe.ledger.n_prefix(h) for h in pipe.ledger.strata))
            co_rows.append({"paid": int(callsc), "prefix": prefix,
                            "certified": sum(1 for r in resc.values() if r.status.startswith("certified"))})
        agg = lambda rows, k: float(np.mean([r[k] for r in rows]))
        out["rows"][str(b)] = {
            "eg": {k: agg(eg_rows, k) for k in eg_rows[0]},
            "cohort": {k: agg(co_rows, k) for k in co_rows[0]},
            "n_claims": len(wl),
        }
        S.dump(path, out)
        e, c = out["rows"][str(b)]["eg"], out["rows"][str(b)]["cohort"]
        S.log(f"[eg] budget={b}: EG paid={e['paid']:.0f} headline stream={e['headline_stream']:.0f} "
              f"mean stream={e['mean_stream']:.0f} total obs={e['total_obs']:.0f} cert={e['certified']:.1f} | "
              f"COHORT paid={c['paid']:.0f} shared prefix={c['prefix']:.0f} cert={c['certified']:.1f}")
    return out


# --------------------------------------------------------------------------- #
# 2b. a dedicated pooled stream vs the stratified region, for the headline (population-wide) claim
# --------------------------------------------------------------------------- #
def exp_pooled(env, budgets=(640, 1536, 3200), seeds=tuple(range(5))):
    ids = env["ids_all"]
    lab = env["labels_M"]
    rel = np.array([int(lab[int(i)][0]) for i in ids])
    ex = np.array([int(lab[int(i)][0]) * int(lab[int(i)][1]) for i in ids])
    N = len(ids)
    out = {"budgets": list(budgets), "seeds": list(seeds), "rows": {}}
    for b in budgets:
        wp, wr = [], []
        for sd in seeds:
            perm = np.random.default_rng(sd).permutation(N)
            csA, csB = FinitePopulationCS(N), FinitePopulationCS(N)
            for j in perm[:b]:
                csA.push(int(ex[j]))
                csB.push(int(rel[j]))
            d = S.ALPHA / 12                      # same per-sequence risk as the region's 12 sequences
            A, B = csA.interval(d), csB.interval(d)
            wp.append(min(1.0, A[1] / B[0]) - max(0.0, A[0] / B[1]))
            res, calls, wl, pipe = S.run_cohort(env, CertificateGuidedAllocator(), b, sd)
            wr.append(res[1].U - res[1].L)
        out["rows"][str(b)] = {"pooled_width": float(np.mean(wp)), "region_width": float(np.mean(wr)),
                               "ratio": float(np.mean(wr) / np.mean(wp))}
        S.dump("exp_pooled.json", out)
        S.log(f"[pooled] b={b}: dedicated pooled stream width={np.mean(wp):.3f} | stratified region width={np.mean(wr):.3f} "
              f"(x{np.mean(wr) / np.mean(wp):.2f})")
    return out


# --------------------------------------------------------------------------- #
# 3. memory of the evidence state
# --------------------------------------------------------------------------- #
def exp_memory(env, seed=0):
    import tracemalloc
    out = {}
    for name, b in (("2%", 640), ("all", env["N"])):
        tracemalloc.start()
        res, calls, wl, pipe = S.run_cohort(env, CertificateGuidedAllocator(), b, seed)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[name] = {"budget": b, "calls": int(calls), "peak_mb": peak / 2 ** 20, "end_mb": cur / 2 ** 20}
        S.log(f"[memory] {name}: calls={calls} peak={peak / 2 ** 20:.1f} MB end={cur / 2 ** 20:.1f} MB")
    S.dump("exp_memory.json", out)
    return out


# --------------------------------------------------------------------------- #
# 4. COHORT on top of LOTUS sem_filter (replay endpoint)
# --------------------------------------------------------------------------- #
def exp_lotus(env, runs=((640, 0), (640, 1), (3200, 0))):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import lotus_replay as LR
    from lotus.models import LM
    rows = []
    for b, sd in runs:
        if True:
            tape = LR.Tape(env["labels_M"])
            srv = LR.make_server(tape)
            lm = LM(model=os.environ.get("LOTUS_MODEL", "openai/gpt-4"),
                    api_base=f"http://127.0.0.1:{srv.server_port}/v1", api_key="none", max_batch_size=64)
            judge = LR.LotusJudge(lm)
            env2 = dict(env)
            env2["contract"] = judge
            t0 = time.time()
            res, calls, wl, pipe = S.run_cohort(env2, CertificateGuidedAllocator(), b, sd)
            wall = time.time() - t0
            res0, calls0, wl0, pipe0 = S.run_cohort(env, CertificateGuidedAllocator(), b, sd)
            same = all((res[q.qid].status, res[q.qid].L, res[q.qid].U) ==
                       (res0[q.qid].status, res0[q.qid].L, res0[q.qid].U) for q in wl)
            rows.append({"budget": b, "seed": sd, "calls": int(calls), "calls_direct": int(calls0),
                         "records_judged": int(judge.n_records), "requests_relevant": int(tape.requests[0]),
                         "requests_exact": int(tape.requests[1]), "prompt_chars": int(tape.prompt_chars),
                         "certified": sum(1 for r in res.values() if r.status.startswith("certified")),
                         "identical_to_direct_tape_run": bool(same), "wall_s": wall})
            S.log(f"[lotus] b={b} seed={sd}: calls={calls} (direct {calls0}) LOTUS requests={sum(tape.requests)} "
                  f"identical={same} wall={wall:.1f}s")
            srv.shutdown()
            S.dump("exp_lotus.json", {"rows": rows, "lotus_version": __import__("lotus").__version__
                                      if hasattr(__import__("lotus"), "__version__") else "1.2.4"})
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="eg,memory")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    env = S.load_wands_env()
    S.log(f"env ready duty={S.DUTY}")
    which = set(x.strip() for x in args.exp.split(","))
    if "eg" in which:
        exp_eg_reuse(env)
    if "pooled" in which:
        exp_pooled(env)
    if "memory" in which:
        exp_memory(env)
    if "lotus" in which:
        exp_lotus(env)
    if "audit" in which:
        exp_audit_grid(env, seeds=tuple(range(args.seeds)))
    if "bias_real" in which:      # real lexical history: 20 independent sessions per budget
        S.exp_bias(env, ratios=(), seeds=tuple(range(20)), path="exp_bias20.json")
    if "bias_sweep" in which:     # continuous bias sweep: 20 sessions per setting
        S.exp_bias(env, ratios=(1, 2, 4, 8, 16), seeds=tuple(range(20)), path="exp_bias20.json")
    S.log("EXTRA3 DONE: " + ",".join(sorted(which)))


if __name__ == "__main__":
    main()
