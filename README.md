# COHORT — Supplemental Material

**Artifact for:** *Cohort: Session-Wide Certification for Conditional Semantic Aggregates* (ICDE 2027 submission).

**Submission revision:** tag [`v1.1-icde2027`](https://github.com/void525/cohort-semantic-aggregation/tree/v1.1-icde2027). This tag is the projection-fix revision on top of `b053af7`. It changes the Dinkelbach stopping rule, preserves rational thresholds, and adds the Review-7 control log. It does not replace the earlier GPU logs.

This repository contains the implementation, frozen semantic-label tapes across two independent real-world text domains (Wayfair **WANDS** e-commerce search, $N = 31{,}482$, and **UCI DrugLib** clinical pharmacovigilance narratives, $N = 4{,}143$), experiment harnesses, canonical live GPU runners, metadata call-cost heuristic, and figure/table generators behind every result in the paper. All replay experiments run deterministically on CPU with **zero GPU, model download, or network access required**; `scripts/run_wands_live_canonical.py` and `scripts/run_canonical_gpu_suite.py` reproduce the measured canonical full-table scans and live multi-executor wall-clock benchmarks when a local GPU and `Qwen3-8B` weights are available.

---

## 1. Quick start

```bash
pip install -r requirements.txt            # numpy, pandas, matplotlib

# 1) numerical enclosure regressions for this submission (no data, no model, no GPU)
PYTHONPATH=src python3 -c "
from tests.test_core import (
    test_lambda_bisect_and_enclosure_regression,
    test_dinkelbach_does_not_stop_on_downward_step,
    test_audit_threshold_keeps_exact_third,
)
test_lambda_bisect_and_enclosure_regression()
test_dinkelbach_does_not_stop_on_downward_step()
test_audit_threshold_keeps_exact_third()
"
python tests/test_core.py                   # full local suite, including the three checks above
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
python run_review5_cpu_suite.py              # -> exp_review5_cpu.json (Shared-priority + E0, Oracle x dynamic E, DrugLib, adaptive policy)
python run_review6_audit.py                  # -> 45-case call-cost selector table & per-case log, candidate-family adaptive policy, live GPU batch utilization
# Optional live canonical GPU full-table scan + live multi-executor benchmark:
# python run_wands_live_canonical.py         # -> exp_live_canonical_gpu.json
python make_geometry.py && python make_reuse_intervals.py
python make_tables.py && python make_paired.py
python make_review3_visuals.py               # writes all 9 PDF figures + Tables IV–IX to paper/figure/ and paper/gen/
```

The LOTUS integration (Section VI-G) uses `lotus-ai==1.2.4` in its dedicated virtual environment:

```bash
/path/to/venv_lotus/bin/python scripts/probe_lotus.py
/path/to/venv_lotus/bin/python scripts/run_icde2027_extra3.py --exp lotus
```

---

## 2. Key Code & Manifest Pointers

- **Signed-contrast domain & numerical enclosure (`projection.py`, `exact.py`, `tests/test_core.py`):**
  - `_domain_bounds` and `verify_outward_enclosure` in `src/cohortrag/projection.py` derive exact feasible ranges $[\underline{\theta}_q, \overline{\theta}_q]$ from query coefficients (e.g., $[-1,1]$ for masked signed contrasts and general bounds for weighted contrasts) across Charnes–Cooper LP, Dinkelbach, bisection fallbacks (`_lambda_min_bisect`, `_lambda_max_bisect`), and outward verification.
  - `_exact_domain_bounds` and `audit_ratio_interval` in `src/cohortrag/exact.py` perform exact `Fraction` simplex auditing over $[\underline{\theta}_q, \overline{\theta}_q]$ on a $2^{-30}$ rational grid.
  - `test_lambda_bisect_and_enclosure_regression` and `test_signed_contrast_full_domain` in `tests/test_core.py` test negative point truth (`-0.4`), zero-crossing intervals, $\pm 1$ boundaries, negative thresholds, general weights, and `audit_exact=True`.
- **UCI DrugLib manifest, prompt pack, and simulated triage history:**
  - Dataset manifest, Stage-1 $H=4$ stratum sizes ($N_h = 2{,}137, 192, 1{,}448, 366$), 11-term safety-keyword regex ($\ge 2$ matches), simulated triage-biased 15% history procedure, and Stage-2 therapeutic-area counts: `data_manifests/druglib.json` and `src/cohortrag/datasets.py` (`load_druglib`).
  - Canonical 3-primitive clinical prompt pack: `src/cohortrag/contracts.py` (`druglib_prompt_pack`).
- **Metadata call-cost heuristic & 45-session benchmark:**
  - Complete closed-form heuristic implementation (`metadata_call_cost_heuristic` with $\gamma \in \{0.30, 0.36, 0.40, 0.50\}$ sensitivity): `scripts/review5_lib.py`.
  - All 45 randomized sessions, per-case predictions, tie counts, and full baseline table (`Always_COHORT`, `Simple_Horizon_Skew_Rule`, `Always_Shared_Priority`, `Metadata_Call_Cost_Heuristic`, `Oracle_Best`): `scripts/run_review6_audit.py` and `reports/icde2027_suite/exp_review5_cpu.json` (`unseen_multidomain_selector`).
- **Observation-driven adaptive policy (candidate-family risk & aligned vs. closed-loop traces):**
  - Full-candidate-family upfront risk pre-allocation (`candidate_queries` in `scripts/review5_lib.py`) and aligned realized-trace vs. independent closed-loop comparison with per-seed cohort traces: `scripts/run_review6_audit.py` and `reports/icde2027_suite/exp_review5_cpu.json` (`adaptive_and_order.result_driven_cascade_policy`).
- **Canonical live GPU protocol, per-seed logs & batch utilization:**
  - Fixed-shape canonical inference (`batch_size=32`, `canonical_fixed_len=288` on WANDS and `736` on DrugLib, deterministic cuBLAS): `src/cohortrag/llm_operator.py`, `scripts/run_canonical_gpu_suite.py`, and `scripts/run_wands_live_canonical.py`.
  - Per-seed wall-clock logs, 0-mismatch full-scan verification, and GPU batch fill / dummy-slot padding breakdown: `reports/icde2027_suite/exp_live_canonical_gpu.json` (`batch_utilization_and_tokens`).

---

## 3. Complete Table / Figure → Command → Raw Per-Seed JSON → Aggregation → Output Mapping

| Paper element | Command | Workload / Config | Raw per-seed JSON (`reports/icde2027_suite/`) | Aggregation & generator | Final output (`paper/`) |
|---|---|---|---|---|---|
| **Fig. 1** (top-k identification gap & live GPU plan inversion) | `python scripts/make_review3_visuals.py` | $N=100$ inset + live RTX 4080 SUPER balanced vs. skewed-$E_0$ sessions | `exp_live_canonical_gpu.json` | `draw_fig1_motivation()` | `figure/fig_ambiguity.pdf` |
| **Fig. 2** (ledger, pointer rule, $\kappa$ re-slice & Q1 bounds) | `python scripts/make_reuse_intervals.py && python scripts/make_review3_visuals.py` | WANDS Q1, 15% BM25 + 2% calls, seed 0 | `reuse_intervals.json` | `draw_fig2_reuse()` | `figure/fig_reuse.pdf` |
| **Fig. 3** (stage architecture: $\mathcal{Q}_e$ vs. active $\mathcal{A}_t$) | `python scripts/make_review3_visuals.py` | 1-stage execution loop | schematic (`fig_executor.py`) | `draw_fig3_architecture()` | `figure/fig_stage.pdf` |
| **Fig. 4** (partition parallelogram & coupled multi-predicate LP) | `python scripts/make_geometry.py && python scripts/make_review3_visuals.py` | Rating share @ 2% + $(\mathrm{Ex}_{\mathrm{4B}}-\mathrm{Ex}_{\mathrm{8B}})\mid\mathrm{Rel}_{\mathrm{8B}}$ @ 4–20% | `exp_geometry.json`, `exp_review4_suite.json` (`coupled_lp_vs_partition`) | `draw_fig4_geometry()` | `figure/fig_geometry.pdf` |
| **Table I & II** (running session & 18 relevance claims) | `python scripts/run_icde2027_extra.py --exp claims` | 4-round session & 5 numeric + 13 thresholds | `exp_claims.json` | `make_tables.py` (`tab_claims`) | `gen/tab_claims.tex` |
| **Fig. 5 & Table III** (mechanism attribution & cold-start per-target calls) | `python scripts/run_icde2027_suite.py --exp ladder` & `python scripts/run_review3_suite.py --exp strong` | 18 claims & 5 numeric targets, cold start, 10 seeds | `exp_ladder.json`, `exp_strong_baselines.json` | means (`tab_ladder`), per-target medians (`draw_fig5_ladder`) | `figure/fig_ladder.pdf`, `gen/tab_ladder.tex` |
| **Fig. 6 & Table IV** (9-executor 8-round sessions across WANDS Relevance, Split-Cost Cascade & UCI DrugLib) | `python scripts/run_review5_cpu_suite.py && python scripts/make_review3_visuals.py` | Dynamic $E$, Oracle$\times$dynamic $E$, Shared-priority+$E_0$ (Adaptive & Oracle), COHORT (10 seeds) | `exp_review5_cpu.json` (`wands_task1_8rounds`, `wands_split_cost_cascade_8rounds`, `druglib_triage_8rounds`) | medians, IQR, $\operatorname{median}_s(1 - C_{\text{COHORT},s}/C_{\text{base},s})$ | `figure/fig_claims.pdf`, `gen/tab_multicohort.tex` |
| **Fig. 7 & Table V** (empirical plan-crossover phase diagrams & 45-session offline call-cost selector) | `python scripts/run_review6_audit.py && python scripts/make_review3_visuals.py` | Overlap $\times$ Horizon vs. Dyn-$E$; $E_0$ Skew $\times$ Round vs. Shared-Priority; 45 sessions across 3 workloads | `exp_review5_cpu.json` (`phase_diagrams_clean`, `unseen_multidomain_selector`) | median call ratios (`draw_fig9_denom`) & selector baselines (`tab_selector`) | `figure/fig_denom.pdf`, `gen/tab_selector.tex` |
| **Fig. 8 & Table VI** (biased history safety, paired $E$ vs. $T$, mid-stage interleaving) | `python scripts/run_icde2027_extra3.py`, `run_paired_targets.py`, `run_review3_suite.py --exp interleave` | 15% history (Unif, BM25, $\times 4$, $\times 16$), 8–20 seeds; mid-stage 12% BM25 | `exp_bias20.json`, `exp_paired.json`, `exp_interleave.json` | $\operatorname{median}_s(1 - C_{E,s}/C_{T,s})$ (`draw_fig7_history_and_paired`) | `figure/fig_history.pdf`, `gen/tab_paired.tex` |
| **Fig. 9** (two-stage re-slicing on WANDS & DrugLib) | `python scripts/run_icde2027_suite.py --exp session` & `run_review5_cpu_suite.py` | Stage 1 ($\alpha/2$) $\to$ Stage 2 ($\alpha/4$) category & therapeutic re-slice | `exp_session.json`, `exp_review5_cpu.json` (`druglib_domain`) | `draw_fig8_session()` | `figure/fig_session.pdf` |
| **Table VII** (coverage calibration & exact rational LP audit) | `python scripts/run_icde2027_suite.py --exp calib` & `run_icde2027_extra3.py --exp audit` | 3,000 full permutations + 200 sessions (1,740 claims) + 60 audit cells | `exp_calib.json`, `exp_audit_grid.json`, `exp_enum.json` | exact rates & 95% Clopper–Pearson UB | `gen/tab_calib.tex` |
| **Table VIII** (canonical live GPU wall-clock, first-answer latency & batch fill vs. measured full-table scans) | `python scripts/run_wands_live_canonical.py && python scripts/run_review6_audit.py` | Live Qwen3-8B (`batch_size=32, len=288`), 0 mismatches vs. 1,814.2s scan | `exp_live_canonical_gpu.json` | `write_tables()` (`tab_livegpu`) | `gen/tab_livegpu.tex` |
| **Table IX** ($H, K$ structural scaling) | Charnes–Cooper cells: `python scripts/run_review3_suite.py --exp hk`. Dinkelbach cells ($H{\cdot}J>32$), after the residual-aware fix: `WANDS_DIR=/path/to/WANDS PYTHONPATH=src python3 scripts/run_hk_one.py H K seed /tmp/cell.txt` (one core per seed). Joint-label 4-round control: `WANDS_DIR=/path/to/WANDS PYTHONPATH=src python3 scripts/run_review7_controls.py` | 3,200 calls, 3 seeds. Repaired cells: $(H,K)\in\{(12,2),(24,2),(6,3),(6,4),(12,4),(24,4)\}$ | Pre-fix full grid: `exp_hk_scale.json`. Post-fix CPU and Q1 width: `exp_review7_controls.json` key `hk_dinkelbach`. Joint-label calls: same file, key `joint_exposure` | Means in `tab_hkscale.tex`. Default $H{=}6,K{=}2$ stays 2.3 s from `exp_hk_scale.json`. $H{=}24,K{=}4$ is 342.6 s in `hk_dinkelbach` (seeds 356.3, 340.6, 331.1 s); Q1 width remains 0.403 | `gen/tab_hkscale.tex` |

---

## 4. Data and licenses

| Item | Source | License |
|---|---|---|
| WANDS (Wayfair ANnotation Dataset, $N=31{,}482$ frozen pool) | https://github.com/wayfair/WANDS | MIT (`data_manifests/wands.json`) |
| UCI Drug Review Dataset (DrugLib.com, $N=4{,}143$ patient reviews) | https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com | CC BY 4.0 (`data_manifests/druglib.json`) |
| UCI Sentiment Labelled Sentences ($N=2{,}000$) | https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences | CC BY 4.0 (`data_manifests/uci.json`) |

The implementation (`src/`, `scripts/`, `tests/`) is released under the MIT license (see `LICENSE` and `NOTICE`).
