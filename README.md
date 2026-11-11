# COHORT — Supplemental Material

**Artifact for:** *COHORT: Session-Wide Certification for Conditional Semantic Aggregates* (ICDE 2027 submission).

This repository contains the implementation, frozen semantic-label tapes across two independent real-world text domains (Wayfair **WANDS** e-commerce search, $N = 31{,}482$, and **UCI DrugLib** clinical pharmacovigilance narratives, $N = 4{,}143$), experiment harnesses, canonical live GPU runners, and figure/table generators behind every result in the paper. All replay experiments run deterministically on CPU with **zero GPU, model download, or network access required**; `scripts/run_wands_live_canonical.py` reproduces the measured canonical full-table scans and live multi-executor wall-clock latency when a local GPU and `Qwen3-8B` weights are available.

---

## 1. Quick start

```bash
pip install -r requirements.txt            # numpy, pandas, matplotlib

# 1) unit, rational simplex & numerical enclosure regression tests (no data, no model)
python tests/test_core.py                   # includes _lambda_max_bisect & L <= U enclosure regression
python tests/test_simplex.py
python tests/test_audit_enum.py             # exact rational LP audit vs. integer enumeration
python scripts/validate_cs_demo.py          # confidence-sequence endpoints vs. brute force

# 2) fetch WANDS (MIT; DrugLib CC BY 4.0 TSVs are bundled in data/drugs/) and check cache
git -c core.autocrlf=false clone https://github.com/wayfair/WANDS.git data/wands
python scripts/preflight_cache.py --dataset wands --data data/wands/dataset \
    --manifest data_manifests/wands.json --model /root/autodl-tmp/models/Qwen3-8B

# 3) regenerate experiment JSONs, LaTeX tables/macros, and all 9 figures
cd scripts
python run_icde2027_suite.py --exp ladder,sched,compose,session,denom,calib --seeds 10
python run_icde2027_extra.py --exp operator,scale,cost,example,basis,claims,nscale,auditime
python run_icde2027_extra3.py --exp eg,pooled,memory,audit,bias_real,bias_sweep
python run_paired_targets.py                 # -> reports/icde2027_suite/exp_paired.json
python run_review3_suite.py                  # -> exp_strong_baselines, exp_interleave, exp_hk_scale, exp_lifecycle
python run_review4_suite.py --skip-gpu       # -> exp_review4_suite.json (coupled LP vs. partition, overlap regimes)
python run_review5_cpu_suite.py              # -> exp_review5_cpu.json (Shared-priority + E0, Oracle x dynamic E, DrugLib, adaptive policy & plan selector)
# Optional live canonical GPU full-table scan + live multi-executor benchmark:
# python run_wands_live_canonical.py         # -> exp_live_canonical_gpu.json
python make_geometry.py && python make_reuse_intervals.py
python make_tables.py && python make_paired.py
python make_review3_visuals.py               # writes all 9 PDF figures + Tables IV–VIII to paper/figure/ and paper/gen/
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
| **Fig. 1** (top-k identification gap & live GPU plan inversion) | `python scripts/make_review3_visuals.py` | $N=100$ inset + live RTX 4080 SUPER balanced vs. skewed-$E_0$ sessions | `exp_live_canonical_gpu.json` | `draw_fig1_motivation()` | `figure/fig_ambiguity.pdf` |
| **Fig. 2** (ledger, pointer rule, $\kappa$ re-slice & Q1 bounds) | `python scripts/make_reuse_intervals.py && python scripts/make_review3_visuals.py` | WANDS Q1, 15% BM25 + 2% calls, seed 0 | `reuse_intervals.json` | `draw_fig2_reuse()` | `figure/fig_reuse.pdf` |
| **Fig. 3** (stage architecture: $\mathcal{Q}_e$ vs. active $\mathcal{A}_t$) | `python scripts/make_review3_visuals.py` | 1-stage execution loop | schematic (`fig_executor.py`) | `draw_fig3_architecture()` | `figure/fig_stage.pdf` |
| **Fig. 4** (partition parallelogram & coupled multi-predicate LP) | `python scripts/make_geometry.py && python scripts/make_review3_visuals.py` | Rating share @ 2% + $(\mathrm{Ex}_{\mathrm{4B}}-\mathrm{Ex}_{\mathrm{8B}})\mid\mathrm{Rel}_{\mathrm{8B}}$ @ 4–20% | `exp_geometry.json`, `exp_review4_suite.json` (`coupled_lp_vs_partition`) | `draw_fig4_geometry()` | `figure/fig_geometry.pdf` |
| **Table I & II** (running session & 18 relevance claims) | `python scripts/run_icde2027_extra.py --exp claims` | 4-round session & 5 numeric + 13 thresholds | `exp_claims.json` | `make_tables.py` (`tab_claims`) | `gen/tab_claims.tex` |
| **Fig. 5 & Table III** (mechanism attribution & cold-start per-target calls) | `python scripts/run_icde2027_suite.py --exp ladder` & `python scripts/run_review3_suite.py --exp strong` | 18 claims & 5 numeric targets, cold start, 10 seeds | `exp_ladder.json`, `exp_strong_baselines.json` | means (`tab_ladder`), per-target medians (`draw_fig5_ladder`) | `figure/fig_ladder.pdf`, `gen/tab_ladder.tex` |
| **Fig. 6 & Table IV** (9-executor 8-round sessions across WANDS Relevance, Split-Cost Cascade & UCI DrugLib) | `python scripts/run_review5_cpu_suite.py && python scripts/make_review3_visuals.py` | Dynamic $E$, Oracle$\times$dynamic $E$, Shared-priority+$E_0$ (Adaptive & Oracle), COHORT (10 seeds) | `exp_review5_cpu.json` (`wands_task1_8rounds`, `wands_split_cost_cascade_8rounds`, `druglib_triage_8rounds`) | medians, IQR, $\operatorname{median}_s(1 - C_{\text{COHORT},s}/C_{\text{base},s})$ | `figure/fig_claims.pdf`, `gen/tab_multicohort.tex` |
| **Fig. 7** (plan-selection phase diagrams & unseen 45-session selector) | `python scripts/run_review5_cpu_suite.py && python scripts/make_review3_visuals.py` | Overlap $\times$ Horizon vs. Dyn-$E$; $E_0$ Skew $\times$ Round vs. Shared-Priority; 45 unseen sessions | `exp_review5_cpu.json` (`phase_diagrams_clean`, `unseen_multidomain_selector`) | median call ratios & zero-param cost model (`draw_fig9_denom`) | `figure/fig_denom.pdf` |
| **Fig. 8 & Table V** (biased history safety, paired $E$ vs. $T$, mid-stage interleaving) | `python scripts/run_icde2027_extra3.py`, `run_paired_targets.py`, `run_review3_suite.py --exp interleave` | 15% history (Unif, BM25, $\times 4$, $\times 16$), 8–20 seeds; mid-stage 12% BM25 | `exp_bias20.json`, `exp_paired.json`, `exp_interleave.json` | $\operatorname{median}_s(1 - C_{E,s}/C_{T,s})$ (`draw_fig7_history_and_paired`) | `figure/fig_history.pdf`, `gen/tab_paired.tex` |
| **Fig. 9** (two-stage re-slicing on WANDS & DrugLib) | `python scripts/run_icde2027_suite.py --exp session` & `run_review5_cpu_suite.py` | Stage 1 ($\alpha/2$) $\to$ Stage 2 ($\alpha/4$) category & therapeutic re-slice | `exp_session.json`, `exp_review5_cpu.json` (`druglib_domain`) | `draw_fig8_session()` | `figure/fig_session.pdf` |
| **Table VI** (coverage calibration & exact rational LP audit) | `python scripts/run_icde2027_suite.py --exp calib` & `run_icde2027_extra3.py --exp audit` | 3,000 full permutations + 200 sessions (1,740 claims) + 60 audit cells | `exp_calib.json`, `exp_audit_grid.json`, `exp_enum.json` | exact rates & 95% Clopper–Pearson UB | `gen/tab_calib.tex` |
| **Table VII** (canonical live GPU wall-clock vs. measured full-table scans) | `python scripts/run_wands_live_canonical.py` | Live Qwen3-8B (`batch_size=32, len=288`), 0 mismatches vs. 1,814s scan | `exp_live_canonical_gpu.json` | `write_tables()` (`tab_livegpu`) | `gen/tab_livegpu.tex` |
| **Table VIII** ($H, K$ structural scaling) | `python scripts/run_review3_suite.py --exp hk` | $H\in\{3,6,12,24\}, K\in\{2,3,4\}$ @ 3,200 calls, 3 seeds | `exp_hk_scale.json` | `write_tables()` (`tab_hkscale`) | `gen/tab_hkscale.tex` |

---

## 3. Data and licenses

| Item | Source | License |
|---|---|---|
| WANDS (Wayfair ANnotation Dataset, $N=31{,}482$ frozen pool) | https://github.com/wayfair/WANDS | MIT (`data_manifests/wands.json`) |
| UCI Drug Review Dataset (DrugLib.com, $N=4{,}143$ patient reviews) | https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com | CC BY 4.0 (`data_manifests/druglib.json`) |
| UCI Sentiment Labelled Sentences ($N=2{,}000$) | https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences | CC BY 4.0 (`data_manifests/uci.json`) |

The implementation (`src/`, `scripts/`, `tests/`) is released under the MIT license (see `LICENSE` and `NOTICE`).
