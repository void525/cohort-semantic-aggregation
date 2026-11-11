# COHORT — Supplemental Material

**Artifact for:** *COHORT: Session-Wide Certification for Conditional Semantic Aggregates* (ICDE 2027 submission).

This repository contains the implementation, frozen semantic-label tapes, experiment harnesses,
and figure/table generators behind every result in the paper. All replay experiments run deterministically
on CPU with **zero GPU, model download, or network access required**; optional live GPU execution
(`run_review4_suite.py` without `--skip-gpu`) reproduces the measured wall-clock latency numbers when a
local GPU and `Qwen3-8B` weights are present.

---

## 1. Quick start

```bash
pip install -r requirements.txt            # numpy, pandas, matplotlib

# 1) unit & mathematical validation (no data, no model)
python tests/test_simplex.py
python tests/test_core.py
python tests/test_audit_enum.py             # exact rational audit vs. integer enumeration
python scripts/validate_cs_demo.py          # confidence-sequence endpoints vs. brute force

# 2) fetch WANDS (not redistributed) and verify frozen label tape (misses must be 0)
git -c core.autocrlf=false clone https://github.com/wayfair/WANDS.git data/wands
python scripts/preflight_cache.py --dataset wands --data data/wands/dataset \
    --manifest data_manifests/wands.json --model /root/autodl-tmp/models/Qwen3-8B

# 3) regenerate experiment JSONs, LaTeX tables/macros, and all 9 figures
cd scripts
python run_icde2027_suite.py --exp ladder,sched,compose,session,denom,calib --seeds 10
python run_icde2027_extra.py --exp operator,scale,cost,example,basis,claims,nscale,auditime
python run_icde2027_extra3.py --exp eg,pooled,memory,audit,bias_real,bias_sweep
python run_paired_targets.py                 # -> reports/icde2027_suite/exp_paired.json
python run_session_features.py               # -> reports/icde2027_suite/exp_sessions.json
python run_review3_suite.py                  # -> exp_strong_baselines, exp_interleave, exp_task2_cascade, exp_hk_scale, exp_lifecycle
python run_multicohort_sessions.py           # -> exp_multicohort_sessions.json
python run_review4_suite.py --skip-gpu       # -> exp_review4_suite.json (E-aware dedicated streams, 8-round crossover & overlap regimes; drop --skip-gpu for live GPU wall-clock)
python make_geometry.py && python make_reuse_intervals.py
python make_tables.py && python make_paired.py
python make_review3_visuals.py               # writes all 9 PDF figures + Tables V–VIII to paper/figure/ and paper/gen/
```

The LOTUS integration (Section VI-G) uses `lotus-ai==1.2.4` in its dedicated virtual environment:

```bash
/path/to/venv_lotus/bin/python scripts/probe_lotus.py
/path/to/venv_lotus/bin/python scripts/run_icde2027_extra3.py --exp lotus
```

---

## 2. Complete Table / Figure → Command → Raw Per-Seed JSON → Aggregation → Output Mapping

| Paper element | Command | Workload / Config | Raw per-seed JSON (`reports/icde2027_suite/`) | Aggregation & generator | Final output (`paper/`) |
|---|---|---|---|---|---|
| **Fig. 1** (top-k identification gap) | `python scripts/make_review3_visuals.py` | $N=100, k=12, a=5$ | deterministic schematic | `draw_fig1_motivation()` | `figure/fig_ambiguity.pdf` |
| **Fig. 2** (ledger, pointer rule & Q1 intervals) | `python scripts/make_reuse_intervals.py && python scripts/make_review3_visuals.py` | WANDS Q1, 15% BM25 + 2% calls, seed 0 | `reuse_intervals.json` | `draw_fig2_reuse()` | `figure/fig_reuse.pdf` |
| **Fig. 3** (stage architecture & $\mathcal{Q}_e$ vs. $\mathcal{A}_t$) | `python scripts/make_review3_visuals.py` | 1-stage execution loop | schematic (`fig_executor.py`) | `draw_fig3_architecture()` | `figure/fig_stage.pdf` |
| **Fig. 4** (partition-aware geometry) | `python scripts/make_geometry.py && python scripts/make_review3_visuals.py` | WANDS rating share @ 2% calls | `exp_geometry.json` | `draw_fig4_geometry()` | `figure/fig_geometry.pdf` |
| **Table I & II** (running session & ledger rules) | `pdflatex main.tex` | 4-round session & $(E,S,T)$ rules | specification in `main.tex` | LaTeX source | `main.pdf` pp. 3–4 |
| **Table III** (18 relevance claims) | `python scripts/run_icde2027_extra.py --exp claims` | 5 numeric + 13 thresholds | `exp_claims.json` | `make_tables.py` (`tab_claims`) | `gen/tab_claims.tex` |
| **Fig. 5 & Table IV** (mechanism attribution & cold-start per-target calls) | `python scripts/run_icde2027_suite.py --exp ladder` & `python scripts/run_review3_suite.py --exp strong` | 18 claims & 5 numeric targets, cold start, 10 seeds | `exp_ladder.json`, `exp_strong_baselines.json` (`rows[*].per_q`) | means (`tab_ladder`), per-target medians (`draw_fig5_ladder`) | `figure/fig_ladder.pdf`, `gen/tab_ladder.tex` |
| **Fig. 6 & Table V** ($E$-aware dedicated streams, 8-round crossover & overlap regimes) | `python scripts/run_review4_suite.py --skip-gpu && python scripts/make_review3_visuals.py` | Task 1 ($K{=}2,H{=}6$) & Task 2 ($K{=}3,H{=}3$), 8 rounds (24 targets), 15% biased history, 10 seeds + low/high overlap | `exp_review4_suite.json` (`methods[*].rows` per seed) | per-round medians, IQR, and $\operatorname{median}_s(1 - C_{\text{COHORT},s}/C_{\text{base},s})$ | `figure/fig_claims.pdf`, `gen/tab_multicohort.tex` |
| **Fig. 7 & Table VI** (biased history safety, paired $E$ vs. $T$, mid-stage interleaving) | `python scripts/run_icde2027_extra3.py`, `run_paired_targets.py`, `run_review3_suite.py --exp interleave` | 15% history (Unif, BM25, $\times 4$, $\times 16$), 8–20 seeds; mid-stage 12% BM25 | `exp_bias20.json`, `exp_paired.json`, `exp_interleave.json` | $\operatorname{median}_s(1 - C_{E,s}/C_{T,s})$ (`draw_fig7_history_and_paired`, `write_tables`) | `figure/fig_history.pdf`, `gen/tab_paired.tex` |
| **Fig. 8** (multi-stage compatible re-slicing) | `python scripts/run_icde2027_suite.py --exp session` | 32 claims across Stage 1 ($\alpha/2$) & Stage 2 ($\alpha/4$), 8 seeds | `exp_session.json` (`runs[*][*].traj`) | mean $\pm$ SD trajectory & Round-4 calls (`draw_fig8_session`) | `figure/fig_session.pdf` |
| **Table VII** (coverage calibration & rational LP audit) | `python scripts/run_icde2027_suite.py --exp calib` & `run_icde2027_extra3.py --exp audit` | 3,000 full permutations + 200 sessions (1,740 claims) + 60 audit cells | `exp_calib.json`, `exp_audit_grid.json`, `exp_enum.json` | exact rates & 95% Clopper–Pearson UB (`write_tables`) | `gen/tab_calib.tex` |
| **Table VIII** ($H, K$ structural scaling) | `python scripts/run_review3_suite.py --exp hk` | $H\in\{3,6,12,24\}, K\in\{2,3,4\}$ @ 3,200 calls, 3 seeds | `exp_hk_scale.json` (`rows`) | mean width, CPU (s), peak heap (`write_tables`) | `gen/tab_hkscale.tex` |
| **Fig. 9** (rare denominators) | `python scripts/run_icde2027_suite.py --exp denom` | 10 WANDS intent groups by relevance rate | `exp_denom.json` | share of runs certified (`draw_fig9_denom`) | `figure/fig_denom.pdf` |
| **Live GPU wall-clock, LOTUS & operator quality** (§VI-G) | `python scripts/run_review4_suite.py` (live GPU), `run_icde2027_extra3.py --exp lotus`, `run_icde2027_extra.py --exp operator` | Live Qwen3-8B (batch 64, 3 seeds), `lotus-ai 1.2.4`, WANDS & UCI | `exp_review4_suite.json` (`live_gpu_session`), `exp_lotus.json`, `exp_operator.json`, `exp_semantic.json` | mean wall-clock (first cert, all 5, GPU vs. CPU) | `gen/eval_body.tex` (§VI-G) |

---

## 3. Data and licenses

| Item | Source | License |
|---|---|---|
| WANDS (Wayfair ANnotation Dataset) | https://github.com/wayfair/WANDS | MIT |
| UCI sentiment sentences | https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences | CC BY 4.0 |
| WANDS frozen candidate pool | 120 search intents × ≤512 annotated pairs → $N = 31{,}482$ records | `data_manifests/wands.json` |

The implementation (`src/`, `scripts/`, `tests/`) is released under the MIT license (see `LICENSE` and `NOTICE`).
