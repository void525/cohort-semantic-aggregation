# -*- coding: utf-8 -*-
"""Review-7 controls that do not require a new GPU pass.

1. Joint-exposure coupled cascade under one label-vector cost: Cohort, dynamic-E,
   oracle x dynamic-E, and coordinated shared-priority.
2. At the prefix where Cohort's joint region certifies a signed contrast, whether
   the separate partition difference is already certified.
3. High-(H, K) cells that use Dinkelbach (more than 32 count variables), re-timed
   after the convergence fix.
4. Round-ready wall-clock already stored for the live Cohort runs.
"""
import json
import os
import statistics
import sys
import time
import tracemalloc

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

import run_icde2027_suite as S
import run_review3_suite as R3
import run_review4_suite as R4
import run_review5_cpu_suite as R5
import review5_lib as RL
from cohortrag import COHORTRAG, CertificateGuidedAllocator, Config, RoundRobinAllocator, Snapshot
from cohortrag.llm_operator import LabelTableOperator
from cohortrag.projection import ratio_bounds
from cohortrag.workload import Query

OUT = os.path.join(_HERE, "..", "reports", "icde2027_suite", "exp_review7_controls.json")


def dump(payload):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    S.log(f"wrote {OUT}")


def median(xs):
    return float(statistics.median(xs)) if xs else None


def paired_saving(a, b):
    return round(float(statistics.median([100.0 * (1.0 - x / y) for x, y in zip(a, b) if y])), 1)


def live_round_ready():
    path = os.path.join(_HERE, "..", "reports", "icde2027_suite", "exp_live_canonical_gpu.json")
    live = json.load(open(path))
    out = {}
    for block in ("wands_split_cost_cascade_live", "wands_relevance_5target_live"):
        out[block] = {}
        for name, blob in live[block].items():
            rows = blob["rows"]
            rec = {
                "mean_first_cert_wall_s": blob.get("mean_first_cert_wall_s"),
                "mean_total_wall_s": blob.get("mean_total_wall_s"),
                "mean_new_calls": blob.get("mean_new_calls"),
            }
            if rows and "round_walls_s" in rows[0]:
                n = len(rows[0]["round_walls_s"])
                rec["mean_round_ready_wall_s"] = [
                    round(statistics.mean(r["round_walls_s"][i] for r in rows), 1) for i in range(n)
                ]
                rec["note"] = "Measured wall-clock when every target issued in that round is certified."
            else:
                rec["note"] = "This executor logged first-certificate and total wall-clock only."
            out[block][name] = rec
    return out


def joint_exposure(env3, hist3, seeds):
    cohort_rounds, stream_rounds = R5.build_coupled_cascade_sessions(env3)
    runners = [
        ("Cohort joint region", lambda sd, h: R4.run_cohort_8rounds(env3, cohort_rounds(), sd, h, mode="E")),
        ("Dynamic-E partition streams", lambda sd, h: RL.run_streams_executor(
            env3, stream_rounds(), sd, h, True, "Edyn", "independent")),
        ("Oracle x dynamic-E partition streams", lambda sd, h: RL.run_streams_executor(
            env3, stream_rounds(), sd, h, False, "Edyn", "independent")),
        ("Shared-priority + E0 partition", lambda sd, h: RL.run_streams_executor(
            env3, stream_rounds(), sd, h, True, "E0", "shared_priority")),
        ("Oracle shared-priority + E0 partition", lambda sd, h: RL.run_streams_executor(
            env3, stream_rounds(), sd, h, False, "E0", "shared_priority")),
    ]
    signed = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11}
    raw = {name: [] for name, _ in runners}
    for sd in seeds:
        h = hist3(sd)
        t0 = time.perf_counter()
        for name, fn in runners:
            res, calls, rcalls, firsts, halted = fn(sd, h)
            n_cert = sum(1 for qid, r in res.items() if str(r.status).startswith("certified"))
            n_signed = sum(1 for qid, r in res.items()
                           if qid in signed and str(r.status).startswith("certified"))
            raw[name].append({
                "seed": int(sd),
                "calls": int(calls),
                "certified": f"{n_cert}/12",
                "signed_certified": f"{n_signed}/{len(signed)}",
                "halted": bool(halted),
                "round_calls": [None if x is None else int(x) for x in (rcalls or [])],
            })
            S.log(f"[joint] seed={sd} {name}: calls={calls} cert={n_cert}/12 signed={n_signed} "
                  f"halted={halted} ({time.perf_counter()-t0:.0f}s)")
    summary = {}
    co = [r["calls"] for r in raw["Cohort joint region"]]
    for name, rows in raw.items():
        calls = [r["calls"] for r in rows]
        summary[name] = {
            "median_calls": median(calls),
            "paired_saving_of_cohort_pct": None if name.startswith("Cohort") else paired_saving(co, calls),
            "completed": f"{sum(1 for r in rows if r['certified']=='12/12')}/{len(rows)}",
            "rows": rows,
        }
    return {
        "cost_contract": "One call exposes the full 3-bit label (Ex4, Rel8, Ex8). No 4B prepass is added to these call counts.",
        "history": "15% proxy-positive biased history, same draw as the split-cost cascade.",
        "methods": summary,
    }


def hk_dinkelbach(env):
    """Re-time only the cells whose count vector is larger than 32, i.e. the Dinkelbach branch."""
    l8 = env["labels_M"]
    l4 = R3.load_4b_labels()
    ids = [int(i) for i in env["ids_all"]]
    pc = S.product_class_map()
    f_base = env["frame"].copy()
    f_base["cls"] = f_base["product_id"].astype(int).map(pc).fillna("missing")
    top_cls = f_base["cls"].value_counts().index[:7].tolist()

    def make_h_frame(H):
        f = f_base.copy()
        rb = f["rating_bucket"].map({"<4": 0, ">=4": 1, "missing": 2}).astype(int)
        if H == 12:
            ig = (f["intent_id"].astype(int) % 4)
            f["stratum"] = rb * 4 + ig
        elif H == 24:
            cg = f["cls"].apply(lambda c: top_cls.index(c) if c in top_cls else 7)
            f["stratum"] = rb * 8 + cg
        elif H == 6:
            ig = (f["intent_id"].astype(int) % 2)
            f["stratum"] = rb * 2 + ig
        else:
            raise ValueError(H)
        return f[["id", "stratum", "text"]].reset_index(drop=True)

    configs = [(12, 2), (24, 2), (6, 3), (6, 4), (12, 4), (24, 4)]
    rows = []
    for H, K in configs:
        sub_f = make_h_frame(H)
        snap = Snapshot(sub_f, name=f"hk-{H}-{K}")
        all_h = tuple(sorted(sub_f["stratum"].unique().tolist()))
        if K == 2:
            labs = {i: np.array([int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8) for i in ids}
            proto = lambda: [
                Query(1, "ex|rel", all_h, (0,), (1,), eps=0.05),
                Query(2, "ex|rel_s0", (0,), (0,), (1,), eps=0.10),
            ]
        elif K == 3:
            labs = {i: np.array([int(l4[i][0]), int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8) for i in ids}
            proto = lambda: [
                Query(1, "prec_rel", all_h, (0,), (1,), eps=0.05),
                Query(2, "rec_rel", all_h, (1,), (0,), eps=0.05),
                Query(3, "ex|rel8", all_h, (1,), (2,), eps=0.05),
            ]
        else:
            labs = {i: np.array([int(l4[i][0]), int(l4[i][0] and l4[i][1]), int(l8[i][0]), int(l8[i][0] and l8[i][1])], dtype=np.int8) for i in ids}
            proto = lambda: [
                Query(1, "prec_rel", all_h, (0,), (2,), eps=0.05),
                Query(2, "prec_ex", all_h, (0, 1), (2, 3), eps=0.05),
                Query(3, "rec_rel", all_h, (2,), (0,), eps=0.05),
                Query(4, "overclaim", all_h, (2,), (0, 1), eps=0.05),
                Query(5, "ex8|rel8", all_h, (2,), (3,), eps=0.05),
            ]
        contract = LabelTableOperator(labs, name=f"op-K{K}", K=K)
        cpus, widths = [], []
        for sd in (0, 1, 2):
            wl = proto()
            for q in wl:
                q.tables = q.build_tables(K)
                q.a_hz = q.numerator_tables(K, q.scope)
            tracemalloc.start()
            t0 = time.process_time()
            pipe = COHORTRAG(snap, contract, Config(alpha=S.ALPHA), seed=sd)
            res = pipe.run(wl, RoundRobinAllocator(), budget_total=3200)
            cpu = time.process_time() - t0
            tracemalloc.stop()
            cpus.append(cpu)
            widths.append(res[1].U - res[1].L if res[1].L is not None else 1.0)
        rec = {
            "H": H, "K": K, "lp_vars": H * (1 << K),
            "cpu_s_mean": round(float(np.mean(cpus)), 2),
            "q1_width_mean": round(float(np.mean(widths)), 4),
            "seeds": 3, "budget": 3200,
            "solver": "dinkelbach-residual-or-bisection",
        }
        rows.append(rec)
        S.log(f"[hk] H={H} K={K} vars={rec['lp_vars']} cpu={rec['cpu_s_mean']:.2f}s width={rec['q1_width_mean']:.3f}")
    return rows


def main():
    payload = {"live_round_ready": live_round_ready()}
    dump(payload)
    env = S.load_wands_env()
    env3, _, hist3 = R4.build_task2_env_and_8rounds(env)
    payload["joint_exposure"] = joint_exposure(env3, hist3, seeds=tuple(range(10)))
    dump(payload)
    payload["hk_dinkelbach"] = hk_dinkelbach(env)
    dump(payload)
    S.log("DONE run_review7_controls.py")


if __name__ == "__main__":
    main()
