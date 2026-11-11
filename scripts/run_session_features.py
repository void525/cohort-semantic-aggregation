# -*- coding: utf-8 -*-
"""Predefined sessions for known mass versus free reads.

Sessions are fixed before looking at the paired outcome:
  * four intent bins, intents sorted by the operator's relevant rate and split
    into equal counts;
  * the five largest product classes.

Targets are the two ratios that do not live only on the tiny low-rated cohort:
the exact-match share among relevant pairs, and the share of relevant pairs
rated at least 4. Cap is the whole session population. Scheduler is round-robin.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
import run_paired_targets as P
import run_icde2027_suite as S

OUT = os.path.join(S.OUT_DIR, "exp_sessions.json")
SEEDS = tuple(range(6))


def two_targets(wl):
    keep = {"theta(exact|relevant)", "theta(rating>=4 | relevant)"}
    return [q for q in wl if q.name in keep]


def run_pop(env, wl, kind="bm25"):
    # whole population: these sessions are a few thousand pairs at most
    old = P.CAP_FRAC
    P.CAP_FRAC = 1.0
    try:
        return P.run_setting(env, wl, kind, SEEDS)
    finally:
        P.CAP_FRAC = old


def main():
    t0 = time.time()
    env = S.load_wands_env()
    out = {"seeds": list(SEEDS), "intent_bins": [], "classes": []}
    f = env["frame"].copy()
    lab = env["labels_M"]
    rel = np.array([int(lab[int(i)][0]) == 1 for i in f["id"]])
    f = f.assign(_rel=rel)
    per = f.groupby("intent_id")["_rel"].mean().sort_values()
    intents = per.index.to_numpy()
    # four bins of equal intent count, low relevant rate to high
    bins = np.array_split(intents, 4)
    for i, members in enumerate(bins):
        mask = f["intent_id"].isin(members)
        env2, wl2 = P.class_env(env, mask, f"intent-bin-{i}")
        wl2 = two_targets(wl2)
        env2["theta_num"] = P.truths(env2, wl2)
        print(f"BIN {i} N={env2['N']} rel={env2['feat']['relevant_share']:.3f} "
              f"exact|rel={env2['feat']['exact_given_relevant']:.3f}", flush=True)
        rows = run_pop(env2, wl2)
        out["intent_bins"].append({"bin": i, "n_intents": int(len(members)),
                                   "feat": env2["feat"], "rows": rows})
        json.dump(out, open(OUT, "w"))
    pc = S.product_class_map()
    f["cls"] = f["product_id"].astype(int).map(pc).fillna("missing")
    counts = f["cls"].value_counts()
    classes = [c for c in counts.index if c != "missing"][:5]
    for cls in classes:
        mask = f["cls"] == cls
        env2, wl2 = P.class_env(env, mask, "class")
        wl2 = two_targets(wl2)
        env2["theta_num"] = P.truths(env2, wl2)
        print(f"CLASS {cls} N={env2['N']} rel={env2['feat']['relevant_share']:.3f}", flush=True)
        rows = run_pop(env2, wl2)
        out["classes"].append({"name": str(cls), "feat": env2["feat"], "rows": rows})
        json.dump(out, open(OUT, "w"))
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
