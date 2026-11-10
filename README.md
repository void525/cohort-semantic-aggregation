# COHORT — Supplemental Material

**Artifact for:** *COHORT: Selection-Safe Evidence Reuse for Conditional Aggregation over
Semantic Operators* (ICDE 2027 submission).

This repository contains the implementation, the frozen semantic-label tapes, the experiment
harnesses, and the table/figure generators behind every number in the paper. Every experiment
replays a frozen label tape on CPU: **no GPU, no model download, and no network access are needed to
regenerate the results**. The only GPU-dependent facts in the paper are the time to *build* the tape
(494 s on one RTX 5090, 63.7 records/s) and the latency/VRAM measurements; they are reported from the
machine that built the tape and are not reproduced here.

---

## 1. Quick start

```bash
pip install -r requirements.txt            # numpy, pandas, matplotlib

# 1) unit / mathematical validation (no data, no model)
python tests/test_simplex.py
python tests/test_core.py
python tests/test_audit_enum.py             # exact audit vs. integer enumeration on small instances
python scripts/validate_cs_demo.py          # confidence-sequence endpoints vs. brute force

# 2) fetch WANDS (not redistributed) and check that the frozen label tape matches it (misses must be 0)
git -c core.autocrlf=false clone https://github.com/wayfair/WANDS.git data/wands   # or: export WANDS_DIR=<your checkout>
python scripts/preflight_cache.py --dataset wands --data data/wands/dataset \
    --manifest data_manifests/wands.json --model /root/autodl-tmp/models/Qwen3-8B
# (--model is an identifier string that is part of the cache key; no model is downloaded or loaded)

# 3) regenerate the experiment results, then the LaTeX tables/macros and figures
cd scripts
python run_icde2027_suite.py --exp ladder,sched,compose,session,denom,calib --seeds 10
python run_icde2027_extra.py --exp operator,scale,cost,example,basis,claims,nscale,auditime
python run_icde2027_extra3.py --exp eg,pooled,memory,audit,bias_real,bias_sweep
python make_geometry.py && python make_reuse_intervals.py
python make_tables.py                        # -> paper/gen/*.tex (numbers are never typed by hand)
python make_figures.py                       # all nine figures -> paper/figure/*.pdf (and figure/src/*.svg, *.png)
```

`make_tables.py` writes to `paper/gen/` and `make_figures.py` to `paper/figure/` inside the repository
(override with the environment variables `GEN_OUT` and `FIG_OUT`). Without arguments `make_figures.py`
redraws all nine figures; `ambig reuse stage geom ladder claims history session denom` select single ones.
The `scripts/run_*.sh` files are the orchestration scripts of the original machine (absolute paths); the
commands above are the portable way to run the same experiments.

The LOTUS integration (Section VI-G) needs `lotus-ai==1.2.4` in its own virtual environment:

```bash
/path/to/venv_lotus/bin/python scripts/probe_lotus.py                       # 3-record smoke test
/path/to/venv_lotus/bin/python scripts/run_icde2027_extra3.py --exp lotus   # COHORT on top of sem_filter
```

Importing LOTUS is slow on small machines (about 70 s on a half-core container); the run itself is short.

---

## 2. Why the experiments reproduce without a GPU

The semantic operator in the paper is a local Qwen3-8B (and Qwen3-4B for the second operator). Its
outputs were frozen once into versioned label tapes (`cache/labels_wands.sqlite`, 12.4 MB;
`reports/second_model_20261103/labels_wands_second_model.sqlite`). A record found in the tape is
returned at zero cost and never touches a model. The cache key **κ** hashes the snapshot manifest, model,
tokenizer, prompt pack, decoding configuration, and view policy, so a tape built under a different
configuration is ignored rather than silently reused. When κ matches, `torch` and `transformers` are never
imported. `scripts/preflight_cache.py` prints the coverage and the number of cache misses (it must be 0).

All experiments are deterministic functions of the seeds and the tape (seeds are listed in every result
file), so a rerun reproduces the stored JSON up to wall-clock timing fields.

---

## 3. Layout

```
src/cohortrag/        the implementation
  snapshot.py         frozen snapshot, primary-key uniqueness, manifest hash
  states.py           joint state space (2^K), statistical-basis compilation
  ledger.py           evidence ledger: known mass E / random prefix S / targeted exposure T, pointer rule
  cs.py               finite-population confidence sequences (sampling without replacement)
  projection.py       joint region and Charnes-Cooper / linear-contrast projections
  exact.py            exact rational audit of emitted statements (implementation layer of Theorem 1)
  simplex.py          two-phase simplex LP solver (no third-party dependency)
  budget.py           schedulers: proportional, round-robin, small-stratum-first, variance-guided,
                      certificate-guided, drain-at-half
  pipeline.py         one stage: plan -> consume -> project -> certify
  llm_operator.py     LocalLLMOperator / CachedSemanticOperator / LabelTableOperator
  contracts.py        prompt packs and the κ version key
  datasets.py         WANDS / UCI loaders with license-manifest validation
  workload.py         query ASTs and the frozen workloads
scripts/              experiment harnesses and generators (see the mapping below)
  fonts/              Liberation Sans (SIL OFL), registered by the figure scripts so all figures share one typeface
  fig_style.py        palette (teal = what COHORT trusts, vermillion = unsafe, greys = the rest), fonts, text-overflow check
  fig_family.py       Fig. 1 and Fig. 2
  fig_executor.py     Fig. 3 (one stage of COHORT)
tests/                unit and mathematical validation
cache/                frozen label tapes (κ-partitioned sqlite)
data/uci/             the two UCI sentiment files used by the label-tape preflight (CC BY 4.0)
data_manifests/       source, license, SHA-256, scope of every dataset
reports/icde2027_suite/   result JSON of every experiment in the paper (+ run.log)
reports/second_model_20261103/   label tape and report of the second operator (Qwen3-4B)
reports/logs_round3/  logs of the final experiment runs
```

### Paper element → script → result file

| Paper element | Script | Result |
|---|---|---|
| Fig. 1 (retrieval alone does not identify a fraction) | `fig_family.py` via `make_figures.py ambig` | none (fixed N=100 toy; no data) |
| Fig. 2 (pointer rule; three ways to use a history) | `make_reuse_intervals.py`, `fig_family.py` via `make_figures.py reuse` | `reuse_intervals.json` |
| Fig. 3 (one stage of COHORT) | `fig_executor.py` via `make_figures.py stage` | none (schematic; no data) |
| Fig. 4 (geometry of the joint region) | `make_geometry.py`, `make_figures.py geom` | `exp_geometry.json` |
| Table V (the 18 claims) | `run_icde2027_extra.py --exp claims` | `exp_claims.json` |
| Fig. 5, Table VI (ladder of baselines) | `run_icde2027_suite.py --exp ladder` | `exp_ladder.json` |
| Fig. 6 (which claims settle when) | `run_icde2027_suite.py --exp ladder` | `exp_ladder.json` |
| Claim-by-claim reuse diagnostic (§VI-B) | `run_icde2027_extra3.py --exp eg` | `exp_eg_reuse.json` |
| Pooled stream vs. stratified region (§VI-B) | `run_icde2027_extra3.py --exp pooled` | `exp_pooled.json` |
| Basis-compilation ablation (§VI-B) | `run_icde2027_extra.py --exp basis` | `exp_basis.json` |
| Table VII (composite claims) | `run_icde2027_suite.py --exp compose` | `exp_compose.json` |
| Fig. 7 (history reuse, bias sweep + real BM25 history; 20 sessions per setting) | `run_icde2027_extra3.py --exp bias_real,bias_sweep` | `exp_bias20.json` (`exp_bias.json` is the earlier 4-session run, kept for provenance) |
| Fig. 8 (four-round session; stage budgets α/2, α/4) | `run_icde2027_suite.py --exp session` | `exp_session.json` |
| Table VIII, audit grid (§VI-F) | `run_icde2027_suite.py --exp calib`, `run_icde2027_extra3.py --exp audit`, `tests/test_audit_enum.py` | `exp_calib.json`, `exp_audit_grid.json`, `exp_enum.json` |
| Table IX (schedulers) | `run_icde2027_suite.py --exp sched` | `exp_sched.json` |
| Fig. 9 (rare denominators) | `run_icde2027_suite.py --exp denom` | `exp_denom.json` |
| Second operator (§VI-G) | `run_icde2027_extra.py --exp operator` | `exp_operator.json` |
| COHORT on LOTUS (§VI-G) | `run_icde2027_extra3.py --exp lotus` (venv with `lotus-ai`), `lotus_replay.py` | `exp_lotus.json` |
| Cost, memory, scaling (§VI-G) | `run_icde2027_extra.py --exp cost,scale,nscale,auditime`, `run_icde2027_extra3.py --exp memory` | `exp_cost.json`, `exp_scale.json`, `exp_nscale.json`, `exp_audit_time.json`, `exp_memory.json` |
| Every number, macro, and table in the text | `make_tables.py` | `paper/gen/*.tex` |

---

## 4. Data and licenses

| Item | Source | License |
|---|---|---|
| WANDS (Wayfair ANnotation Dataset) | https://github.com/wayfair/WANDS | MIT |
| UCI sentiment sentences (used only in the label-tape preflight and the supplementary UCI check) | https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences | CC BY 4.0 |
| WANDS subset | 120 search intents × ≤512 annotated pairs → N = 31,482 records, 6 strata | `data_manifests/wands.json` |

The WANDS files are not redistributed; fetch them from the official repository and check the SHA-256
digests in the manifest (`git -c core.autocrlf=false clone https://github.com/wayfair/WANDS.git`; the CSV
files are tab-separated). The frozen label tapes contain labels computed by the open-weight Qwen3 models and
are released under the terms of the underlying data.

**Why not CFPB complaints?** The original plan used the CFPB complaint database. The CFPB stopped publishing
the `Consumer complaint narrative` field in 2026, and the main dataset was switched to WANDS. A frozen
historical snapshot of the narrative archive could serve as a second domain; labelling it needs a GPU and was
not done for this submission.

---

## 5. Honest scope

* Certificates concern the operator's full-run answer θ_run, not human truth θ_gold; the two are reported
  separately.
* Claims are programmatic ASTs over a declared closure; natural-language translation is not implemented.
* The Evergreen-style baseline is a re-implementation of its estimation path (shuffle, confidence
  sequences, Bonferroni split, prompt cache) on our operator and tape; the original system was not run.
* The LOTUS experiment replays the frozen tape through LOTUS's own `sem_filter`; it validates the interface
  and the request/accounting, not LOTUS's semantic accuracy.
* Some scripts date from an earlier draft (`run_real_data.py`, `run_ablations.py`, ...). They are kept because
  the current harnesses import their helpers; the paper no longer relies on their headline comparisons.

---

## 6. License and citation

The implementation (`src/`, `scripts/`, `tests/`) is released under the MIT license (see `LICENSE`). The frozen
label tapes contain only record identifiers and binary labels computed by open-weight Qwen3 models; they carry
no WANDS text. This repository is the supplemental material of the ICDE 2027 submission named above; a citation
entry will be added when the paper is published.
