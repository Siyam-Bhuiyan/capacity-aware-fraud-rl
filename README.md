# Fairness-Aware, Capacity-Constrained RL for Fraud Alert Triage

Built on the **FiFAR** benchmark (BAF dataset + synthetic human-analyst experts).
Three research themes, each owned end-to-end by one team member.

| Theme | Owner | Scope |
|---|---|---|
| 1. Classical/Static Baseline, Fairness & Calibration | **Siam** | LightGBM baseline, FairGBM, calibration, conformal prediction |
| 2. Sequential/Adaptive RL Algorithms | Tanvir | DQN, contextual bandits, RL algorithm comparison |
| 3. Human-AI Teaming / Learning-to-Defer | Mahdeen | FiFAR's L2D baseline, rejection learning, RL-based deferral policy |

## Repo contents

- `verify_dataset.py` — Phase 0 dataset verification (schema, temporal split, label
  integrity, missing-value sentinels, scaler correctness). 13/13 checks pass.
- `Reinforcement_Learning.ipynb` — Phase 0 data engineering: merges FiFAR + BAF,
  encodes categoricals, temporal train/eval split, feature scaling.
- `Baseline_LightGBM_FairGBM.ipynb` — Theme 1 notebook (Siam), executed end-to-end.
- `FIFAR Paper.pdf` — reference paper for the FiFAR benchmark.

Datasets are **not** committed (see `.gitignore`) — `fifar_prepared/` and
`FiFAR_extracted/` are regenerated locally by running `Reinforcement_Learning.ipynb`
against the shared FiFAR/BAF source data.

## Theme 1 status (Siam) — reference floor for the project

All four required deliverables are complete, with results from a real executed
run (`Baseline_LightGBM_FairGBM.ipynb`):

| Metric | Value |
|---|---|
| Recall @ 5% FPR — plain LightGBM (**reference floor**) | **0.1946** |
| ROC-AUC — plain LightGBM | 0.6789 |
| FPR gap, age<50 vs age>=50 — plain LightGBM | 0.0256 |
| Recall @ 5% FPR — FairGBM (tree count matched to baseline) | 0.1960 |
| FPR gap — FairGBM | 0.0246 |
| Brier score — raw / Platt / isotonic | 0.1169 / 0.1163 / 0.1157 |
| Conformal prediction — empirical coverage (target 90%) | 0.8761 |
| Conformal prediction — avg. prediction-set size | 1.0437 |

**Checklist:**
- [x] Verify prepared dataset (Phase 0)
- [x] Plain LightGBM baseline, excluding `standard#0-49` expert columns — Recall @ 5% FPR reported
- [x] FairGBM vs. plain baseline comparison
- [x] Calibration: Platt scaling + isotonic regression
- [x] Conformal prediction for uncertainty quantification

**Known limitations** (documented in the notebook):
- No hyperparameter tuning was performed — these are untuned-default results.
- The FairGBM comparison reads group FPRs off a threshold recalibrated for 5%
  overall FPR, not FairGBM's own training-time constraint threshold; the gap
  reduction should be read as directional, not an exact bound.

## Reproducing locally

1. Obtain the FiFAR/BAF data and run `Reinforcement_Learning.ipynb` to produce
   `fifar_prepared/` (train/eval parquet files + scaler).
2. Run `python verify_dataset.py` — should report 13/13 checks passing.
3. Run `Baseline_LightGBM_FairGBM.ipynb`. The FairGBM section requires Linux
   (Google Colab or WSL) — no Windows wheel is published for `fairgbm`.
