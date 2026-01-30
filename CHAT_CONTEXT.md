# DriftSeal Chat Context (2026-01-17)

This file captures the working context, important decisions, code changes, commands run, and final metrics from this chat session so it can be resumed later.

## Workspace

- Repo root: `t:\DriftSeal`
- OS: Windows
- Key project areas (top-level folders):
  - `Agriculture/`
  - `Entertainment/`
  - `Energy/`
  - `Finance/`
  - `Healthcare/`
  - `Manufacturing/`
  - `Retail/`
  - `Tech/`
  - `Transportation/`

Most directories contain a single Python entrypoint script that trains/evaluates models and writes `*.meta.json` + predictions CSVs into the same folder.

## What Was Done

### 1) Agriculture — Yield XGBoost debugging + fix

**Goal**

- Fix a validation/test mismatch and a broken baseline feature where `lag1_yield` in the test set became constant, causing very poor test performance.

**Root cause**

- In `train_xgb_yield`, the code built per-group “last value” mappings using:
  - `grp["hg/ha_yield"].nth(-1)` / `nth(-2)` / `nth(-3)`
- `SeriesGroupBy.nth(-1)` returns a Series indexed by original row index, not by `(Area, Item)`.
- Mapping test rows with `idx_ai_test.map(last_map)` then produced all `NaN`, which later got filled by `overall_mean`, making test `lag1_yield` effectively constant.

**Fix applied**

- Replace `.nth(-1)` with `.last()` (proper MultiIndex by group).
- Compute `lag2_map` / `lag3_map` per group with safe fallbacks to `NaN` when history is too short.

**Code change**

- File: `Agriculture/agriculture_models.py`
- Location: mapping block inside `train_xgb_yield`
  - Link: [agriculture_models.py](file:///t:/DriftSeal/Agriculture/agriculture_models.py)

**Training command**

```bash
python Agriculture\agriculture_models.py xgb_yield --agriculture-dir Agriculture --seed 42 --test-frac 0.15 --max-rows 0 --n-estimators 900 --max-depth 8 --learning-rate 0.05 --subsample 0.9 --colsample-bytree 0.9
```

**Result (after fix)**

- Test metrics (XGBoost yield):
  - RMSE: **16083.65**
  - MAE: **7167.46**
  - R²: **0.9673**

**Baseline for comparison**

- Baseline using `lag1_yield` feature directly on the same test split:
  - RMSE: **17718.28**
  - MAE: **7775.55**
  - R²: **0.9603**

**Artifacts written**

- `Agriculture/yield_xgb.meta.json`
- `Agriculture/yield_xgb_predictions.csv`

### 2) Entertainment — Models built + trained + tested

**Goal**

- Use datasets under `t:\DriftSeal\Entertainment` and build:
  - Recommendation: Deep CF (collaborative filtering)
  - GANs: content generation (synthetic Spotify tabular feature vectors)
  - Transformers: NLP model
- Clean datasets during ingestion, then train/test and output final test metrics.

**Implementation**

- New entrypoint script:
  - `Entertainment/entertainment_models.py`
  - Link: [entertainment_models.py](file:///t:/DriftSeal/Entertainment/entertainment_models.py)

#### A) Deep CF recommender (MovieLens ratings)

- Dataset:
  - `Entertainment/MovieLens dataset/rating.csv`
- Cleaning:
  - Numeric coercion for `userId`, `movieId`, `rating`
  - Parse timestamp
  - Drop invalid rows, drop duplicates, sort by `(userId, timestamp)`
  - Keep only users with ≥2 ratings
- Split:
  - For each user: last rating = test; second-last = validation (if user has ≥3 ratings); remainder = train
- Outputs:
  - `Entertainment/deep_cf_predictions.csv`
  - `Entertainment/deep_cf.meta.json`

Command used:

```bash
python Entertainment\entertainment_models.py deep_cf --entertainment-dir Entertainment --seed 42 --max-ratings 200000 --epochs 5 --batch-size 4096 --embed-dim 32
```

Test results:

- Rating prediction:
  - RMSE: **0.96998**
  - MAE: **0.76375**
  - R²: **0.11959**
- Ranking (sampled 99 negatives per user):
  - HitRate@10: **0.33260**
  - NDCG@10: **0.17135**

#### B) GAN “content generation” (Spotify tabular synthesis)

- Dataset:
  - `Entertainment/Spotify dataset/data/data.csv`
- Task:
  - Train a simple GAN on normalized numeric audio/track features and generate synthetic samples.
- Cleaning:
  - Coerce selected numeric columns to float
  - Drop NaNs
  - Standardize with `StandardScaler`
- Outputs:
  - `Entertainment/spotify_gan_samples.csv`
  - `Entertainment/spotify_gan.meta.json`

Command used:

```bash
python Entertainment\entertainment_models.py gan_spotify --entertainment-dir Entertainment --seed 42 --max-rows 50000 --steps 400 --batch-size 512 --latent-dim 32 --n-samples 3000
```

Sample metrics (generated vs real distributions):

- rel_mean_abs_diff: **0.4382**
- rel_std_abs_diff: **0.3173**
- corr_fro_norm_diff: **4.2915**

Notes:

- `rel_*` are average absolute diffs normalized by real feature std; closer to 0 is better.
- `corr_fro_norm_diff` is Frobenius norm difference between correlation matrices; closer to 0 is better.

#### C) Transformer NLP (MovieLens title → genres)

- Dataset:
  - `Entertainment/MovieLens dataset/movie.csv`
- Task:
  - Multi-label classification: predict movie genres from the cleaned title text.
- Cleaning:
  - Remove “(YYYY)” suffix from titles
  - Drop `(no genres listed)`
- Model:
  - `distilbert-base-uncased`
  - Head initialized for multi-label classification and trained briefly (1 epoch).
- Outputs:
  - `Entertainment/nlp_transformer_predictions.csv`
  - `Entertainment/nlp_transformer.meta.json`

Command used (final run):

```bash
python Entertainment\entertainment_models.py nlp_transformer --entertainment-dir Entertainment --seed 42 --model-name distilbert-base-uncased --max-movies 2000 --epochs 1 --batch-size 32 --max-length 32 --threshold 0.2
```

Test results:

- micro_f1: **0.28738**
- macro_f1: **0.13662**
- subset_accuracy: **0.0**
- threshold: **0.2**

## How To Resume Quickly

### Re-run Agriculture yield model

```bash
python Agriculture\agriculture_models.py xgb_yield --agriculture-dir Agriculture --seed 42 --test-frac 0.15 --max-rows 0 --n-estimators 900 --max-depth 8 --learning-rate 0.05 --subsample 0.9 --colsample-bytree 0.9
```

### Re-run Entertainment models

```bash
python Entertainment\entertainment_models.py deep_cf --entertainment-dir Entertainment
python Entertainment\entertainment_models.py gan_spotify --entertainment-dir Entertainment
python Entertainment\entertainment_models.py nlp_transformer --entertainment-dir Entertainment
```

## Files Modified / Added

- Modified:
  - `Agriculture/agriculture_models.py` (fixed yield lag feature mapping for test)
- Added:
  - `Entertainment/entertainment_models.py` (Deep CF, Spotify GAN, Transformer NLP pipelines)

---

# DriftSeal Security-Framework Imprinting Update (2026-01-28)

## System Architecture (from design sketches)

High-level flow captured in the attached sketches:

1. Home page
2. Security framework flow:
   - Select domain (Agriculture / Energy / Entertainment / Healthcare / Transportation / etc.)
   - Select model type + dataset used (training/evaluation source)
3. Framework checks whether the model is “clean” (no anomaly / misbehavior)
   - If clean: verify and mark model ready for production use
   - If anomaly detected:
     - Identify the anomalous / misbehaving aspect
     - Clean or remove infected data causing the anomaly
     - Re-run the model to re-check sanity
4. Generate a certificate asserting the model is production-ready

## What Has Been Implemented So Far (models + imprint readiness)

### 1) Promotion / eligibility gates (performance vs baseline)

Many training entrypoints write (or compute) a `promotion` block that encodes:

- Task type (classification / regression / time-series / RL / tabular generation)
- Primary metric
- Baseline metric
- Eligible boolean, based on “beats baseline” rules

Examples of implemented gates in code:

- Multiclass classification: accuracy > majority-baseline + margin
  - `Agriculture/agriculture_models.py` (`train_crop_cnn`)
- Regression / time-series regression: rmse < baseline_rmse * factor
  - `Agriculture/agriculture_models.py` (`train_xgb_yield`)
  - `Energy/energy_models.py` (`train_lstm_load`, `train_prophet_style_load`)
  - `Transportation/transportation_models.py` (`train_cnn_traffic`)
  - `Entertainment/entertainment_models.py` (`train_deep_cf`)
- Reinforcement learning: success_rate > random baseline + margin
  - `Transportation/transportation_models.py` (`train_rl_routing`)
- Tabular generation: correlation-matrix difference < baseline * factor
  - `Entertainment/entertainment_models.py` (`train_spotify_gan`)

### 2) Deterministic fingerprinting (tamper / drift detection primitives)

Training entrypoints write fingerprint hashes into metadata to support “imprinting”:

- Model file hash (when a model artifact is saved)
- Predictions/sample outputs hash
- Canary hash over a small prefix of outputs for quick consistency checks
- RL Q-table hash (routing RL)

Fingerprints are present in many `*.meta.json` files and are written by the training scripts.

### 3) Spotify GAN “cure” (making the not-eligible model eligible)

The Spotify tabular GAN was made reliably eligible by adding a correlation-matching objective:

- A generator-side penalty that pushes generated feature correlations toward real feature correlations
- Exposed as a `corr_lambda` knob for tuning

Result in the latest run:

- `corr_fro_norm_diff = 0.3441`
- Baseline (`gaussian_independent`) `corr_fro_norm_diff = 3.7745`
- Eligible: `True`

See: `Entertainment/spotify_gan.meta.json`

## Current Eligibility Snapshot (from latest metadata in repo)

This table reflects the latest `*.meta.json` artifacts present in the working tree at the time of this update.

| Model | Eligible | Task | Primary (vs baseline) |
|---|---:|---|---|
| bert_stackoverflow | ✅ | multiclass_classification | accuracy 0.64 vs majority 0.4275 |
| crop_cnn | ✅ | multiclass_classification | accuracy 0.7548 vs majority(test-set) 0.0263 |
| deep_cf | ✅ | regression | rmse 0.9157 vs user_mean 0.9823 |
| faults_xgb | ✅ | binary_classification | pr_auc_mean 0.4216 vs pos_rate_mean 0.0117 |
| finance_fraud_xgb | ✅ | binary_classification | recall@precision0.90 = 0.6218 (policy gate) |
| healthcare_imaging_cnn | ✅ | binary_classification | auc 0.8830 vs random_auc 0.5 |
| healthcare_tabular_xgb | ✅ | binary_classification | roc_auc 0.8720 vs random_auc 0.5 |
| load_lstm | ✅ | time_series_regression | rmse 26.22 vs baseline 53.12 |
| load_prophet_style | ✅ | time_series_regression | rmse 32.77 vs baseline 53.12 |
| mnist_cnn | ✅ | multiclass_classification | accuracy 0.9870 vs majority 0.1135 |
| nlp_transformer | ✅ | multilabel_classification | micro_f1 0.4663 vs prevalence 0.2856 |
| retail_demand_xgb | ✅ | regression | rmse 232.16 vs lag1_sales 438.26 |
| routing_rl | ✅ | reinforcement_learning | success_rate 1.0 vs random_policy 0.785 |
| spotify_gan | ✅ | tabular_generation | corr_diff 0.3441 vs gaussian_independent 3.7745 |
| steel_rf | ✅ | multiclass_classification | accuracy 0.8098 vs majority 0.3470 |
| stock_lstm_model | ✅ | regression | rmse 2.4682 vs last_value 2.5198 |
| traffic_cnn | ✅ | time_series_regression | rmse 4.2440 vs last_value 5.2358 |
| turbofan_lstm | ✅ | regression | rmse 15.26 vs baseline 41.98 |
| yield_xgb | ✅ | regression | rmse 16083.65 vs lag1_yield 17718.28 |

## What Remains To Be Implemented Next (security framework)

### 1) Central “security framework” layer (currently missing as a unified module)

Implement a single framework entrypoint that:

- Registers models (domain, dataset, training entrypoint, artifact paths)
- Loads latest metadata (`*.meta.json`) and verifies:
  - Promotion gate: eligible must be true
  - Fingerprints: match expected hashes for “imprinted” models
- Runs real-time checks:
  - Output distribution drift vs imprint baseline (e.g., PSI, KL, mean/std bounds)
  - Performance regression checks when labels are available (online evaluation)
- Produces a pass/fail decision and detailed reasons

### 2) Anomaly detection + automatic remediation loop

Implement the “anomaly detected → identify → clean → re-run → verify” loop:

- Define anomaly types:
  - Data quality (missingness, out-of-range, schema changes)
  - Model behavior (sudden prediction shifts, confidence collapse, bias drift)
  - Integrity (fingerprint mismatches)
- Provide automated mitigations:
  - Data cleaning / filtering rules
  - Retraining triggers
  - Rollback to last good imprint

### 3) Certificate generation

Generate a signed (or at least deterministic) “certificate” artifact per imprint that includes:

- Model identity + versioning (hashes)
- Dataset references
- Promotion decision + key metrics
- Timestamp + reproducibility info (seed, params)

### 4) Metadata schema consistency (important for automation)

Some `*.meta.json` currently do not include a `promotion` block even though the training code writes one.
This indicates the metadata files were generated before those changes (or need regeneration).

For consistent automation, ensure every training script writes:

- `promotion` (task, primary, baseline, eligible)
- `fingerprint` (hashes)

Then re-run the affected models so the repository metadata matches the current code behavior.

### 5) UI / Home page wiring

The sketches describe a Home Page → select domain/model → run security checks flow.
A UI layer is not yet present in the repo as a unified application. Next steps:

- Decide UI approach (CLI-only first, then web UI)
- Hook UI selections into the central framework entrypoint
- Surface:
  - Eligibility
  - Fingerprint verification status
  - Anomaly detection outcomes
  - Certificate artifact download/view
