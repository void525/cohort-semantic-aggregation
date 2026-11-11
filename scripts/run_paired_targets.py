# -*- coding: utf-8 -*-
"""Paired calls-to-certificate: known mass versus free reads.

Same tape, same round-robin scheduler, same five numeric targets, same risk.
History is either a BM25 top-k set or a within-stratum-agnostic weighted sample.
The run stops when every numeric target is certified or the cap is hit.
Unfinished targets stay in the record as censored at the cap.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
import run_icde2027_suite as S
from cohortrag import RoundRobinAllocator
from cohortrag.snapshot import Snapshot

OUT = os.path.join(S.OUT_DIR, "exp_paired.json")
SEEDS = tuple(range(8))
CAP_FRAC = 0.55


def numeric_workload(env):
    wl = S.fresh_workload(env)
    return [q for q in wl if q.kind in ("value", "difference")]


def truths(env, wl):
    return {q.qid: S.rrd._query_truth(S._prep(q), env["labels_M"], env["ids_all"], env["stratum_of"])
            for q in wl}


def one_arm(env, hist, seed, cap, wl, mode):
    kw = {"cache": hist} if mode == "E" else {"tcache": hist}
    res, calls, wl, pipe = S.run_cohort(env, RoundRobinAllocator(), cap, seed, wl=wl, **kw)
    theta = env["theta_num"]
    rows = []
    bad = 0
    for q in wl:
        r = res[q.qid]
        ok = r.status.startswith("certified")
        if ok and r.L is not None:
            th = theta[q.qid]
            if not (r.L - 1e-8 <= th <= r.U + 1e-8):
                bad += 1
        rows.append({"qid": q.qid, "name": q.name,
                     "calls": None if not ok else int(r.calls_at_cert),
                     "censored": not ok})
    done = [r["calls"] for r in rows if r["calls"] is not None]
    return {"calls_used": int(calls),
            "calls_all": None if any(r["censored"] for r in rows) else int(max(done)),
            "n_done": int(sum(not r["censored"] for r in rows)),
            "n_targets": len(rows),
            "errors": bad,
            "targets": rows}


def class_env(env, mask, name):
    sub = env["frame"].loc[mask].copy()
    order = {"<4": 0, ">=4": 1, "missing": 2}
    sub["stratum"] = sub["rating_bucket"].map(order).astype(int)
    snap = Snapshot(sub[["id", "stratum", "text"]].reset_index(drop=True), name=name)
    stratum_of = dict(zip(sub["id"].astype(int), sub["stratum"].astype(int)))
    labels = {int(i): env["labels_M"][int(i)] for i in sub["id"]}
    by = {}
    for b, h in order.items():
        if (sub["stratum"] == h).any():
            by[b] = [h]
    env2 = dict(env)
    env2.update(frame=sub, snap=snap, stratum_of=stratum_of, labels_M=labels,
                ids_all=sub["id"].to_numpy(), N=len(sub), by_rating=by,
                meta={"all_strata": sorted(int(x) for x in sub["stratum"].unique())})
    wl = numeric_workload(env2)
    env2["theta_num"] = truths(env2, wl)
    # relevant rate and exact-match rate among relevant, for the session description
    rel = np.array([int(labels[int(i)][0]) == 1 for i in sub["id"]])
    exact = np.array([int(labels[int(i)][0]) == 1 and int(labels[int(i)][1]) == 1 for i in sub["id"]])
    env2["feat"] = {"N": int(len(sub)),
                    "relevant_share": float(rel.mean()),
                    "exact_given_relevant": float(exact.sum() / max(rel.sum(), 1)),
                    "n_lt4": int((sub["stratum"] == 0).sum()),
                    "n_ge4": int((sub["stratum"] == 1).sum())}
    return env2, wl


def history_for(env, kind, seed, frac=0.15):
    if kind == "bm25":
        return S.bm25_history(env, frac, seed)
    ratio = {"uniform": 1, "mild": 4, "strong": 16}[kind]
    return S.enriched_history(env, ratio, frac, seed)


def run_setting(env, wl, kind, seeds):
    cap = int(env["N"] * CAP_FRAC)
    rows = []
    for sd in seeds:
        t0 = time.time()
        hist = history_for(env, kind, sd)
        stats = S.hist_stats(env, hist)
        e = one_arm(env, hist, sd, cap, wl, "E")
        t = one_arm(env, hist, sd, cap, wl, "T")
        rows.append({"seed": sd, "hist_n": stats["size"],
                     "enrich": stats["hit_hist"] / stats["hit_pop"] if stats["hit_pop"] else None,
                     "E": e, "T": t, "seconds": time.time() - t0})
        print(f"  {kind} seed {sd}: E {e['calls_all']} ({e['n_done']}/{e['n_targets']}) "
              f"T {t['calls_all']} ({t['n_done']}/{t['n_targets']}) err {e['errors']+t['errors']} "
              f"{rows[-1]['seconds']:.1f}s", flush=True)
    return rows


def main():
    t0 = time.time()
    env = S.load_wands_env()
    wl = numeric_workload(env)
    env["theta_num"] = truths(env, wl)
    out = {"cap_frac": CAP_FRAC, "targets": [{"qid": q.qid, "name": q.name, "eps": q.eps} for q in wl],
           "full": {}, "classes": []}
    for kind in ("bm25", "uniform", "mild", "strong"):
        print("FULL", kind, flush=True)
        out["full"][kind] = run_setting(env, wl, kind, SEEDS)
        json.dump(out, open(OUT, "w"))
    # product classes, largest first, predefined before looking at the paired outcome
    pc = S.product_class_map()
    f = env["frame"].copy()
    f["cls"] = f["product_id"].astype(int).map(pc).fillna("missing")
    counts = f["cls"].value_counts()
    classes = [c for c in counts.index if c != "missing" and counts[c] >= 1800][:5]
    for cls in classes:
        mask = f["cls"] == cls
        env2, wl2 = class_env(env, mask, "class-" + str(cls)[:24])
        print(f"CLASS {cls} N={env2['N']} rel={env2['feat']['relevant_share']:.2f}", flush=True)
        rows = run_setting(env2, wl2, "bm25", SEEDS)
        out["classes"].append({"name": str(cls), "feat": env2["feat"], "rows": rows})
        json.dump(out, open(OUT, "w"))
    print(f"done {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
