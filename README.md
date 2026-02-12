# DriftSeal

DriftSeal is a lightweight, model-agnostic framework for detecting **training data poisoning / compromised training runs** by looking for out-of-family behavior in a model’s *fingerprint* (metadata + artifact hashes + optional canary behavior), and scoring that fingerprint with a one-class ensemble detector.

This repo contains:
- A single “framework” CLI for fitting/scoring/verifying detectors: [drift_seal_framework.py](file:///t:/DriftSeal/drift_seal_framework.py)
- Multiple example model training entrypoints across domains (each writes a `*.meta.json`)
- A security test example showing a clean MNIST model vs a backdoored MNIST model

## Repo Layout
- Framework
  - [drift_seal_framework.py](file:///t:/DriftSeal/drift_seal_framework.py): fit detector, score models, verify models, materialize canaries, serve an API
  - `driftseal_detector/`: saved detector + reports (created by `fit`, `score_all`, `verify_all`, etc.)
- Example model entrypoints (each produces at least a `*.meta.json`)
  - [clean_model_fingerprint.py](file:///t:/DriftSeal/clean_model_fingerprint.py): MNIST CNN “clean” model (plus a small Gradio UI)
  - [SecurityTest/train_mnist_backdoor.py](file:///t:/DriftSeal/SecurityTest/train_mnist_backdoor.py): trains a backdoored MNIST model starting from the clean one
  - Domain scripts:
    - [Agriculture/agriculture_models.py](file:///t:/DriftSeal/Agriculture/agriculture_models.py)
    - [Energy/energy_models.py](file:///t:/DriftSeal/Energy/energy_models.py)
    - [Entertainment/entertainment_models.py](file:///t:/DriftSeal/Entertainment/entertainment_models.py)
    - [Finance/fraud_clean_model_fingerprint.py](file:///t:/DriftSeal/Finance/fraud_clean_model_fingerprint.py)
    - [Finance/stock_lstm_prediction_model.py](file:///t:/DriftSeal/Finance/stock_lstm_prediction_model.py)
    - [Healthcare/Tabular_clean_model_healthcare_fingerprint.py](file:///t:/DriftSeal/Healthcare/Tabular_clean_model_healthcare_fingerprint.py)
    - [Healthcare/Imaging_clean_model_healthcare_fingerprint.py](file:///t:/DriftSeal/Healthcare/Imaging_clean_model_healthcare_fingerprint.py)
    - [Manufacturing/manufacturing_models.py](file:///t:/DriftSeal/Manufacturing/manufacturing_models.py)
    - [Retail/retail_demand_xgb.py](file:///t:/DriftSeal/Retail/retail_demand_xgb.py)
    - [Tech/bert_stackoverflow_questions.py](file:///t:/DriftSeal/Tech/bert_stackoverflow_questions.py)
    - [Transportation/transportation_models.py](file:///t:/DriftSeal/Transportation/transportation_models.py)

## What DriftSeal Scores
DriftSeal works off model “fingerprints” extracted from each model’s `*.meta.json`. Depending on what a model provides, the fingerprint may include:
- **Metadata features**: domain, model kind, training time, parameter count, etc.
- **Artifact integrity**: SHA-256 hashes for `model_path`, `predictions_path`, and related artifacts
- **Canary replay hashes**: deterministic hashes of predictions on a fixed canary subset (when available)
- **Behavior tests (optional)**: for classification-style models with a usable `model_path`, DriftSeal can apply a synthetic trigger and look for “collapse/spike” behavior

Verification combines multiple signals:
- One-class “poison score” (ensemble of autoencoder + IsolationForest + OneClassSVM)
- Artifact integrity mismatches (hash mismatch = high concern)
- Prediction anomaly scan over numeric columns in `predictions_path` (optional)
- Canary behavior trigger tests (optional; depends on model kind + artifact availability)

## Quickstart (MNIST clean vs backdoored)
This is the most “out of the box” demo because it downloads MNIST automatically.

### 1) Train the clean MNIST CNN
```bash
python clean_model_fingerprint.py train
```
Outputs (in repo root by default):
- `mnist_cnn.keras`
- `mnist_cnn.meta.json`

Optional: run a local UI for the clean model:
```bash
python clean_model_fingerprint.py ui
```

### 2) Train the backdoored MNIST CNN
This starts from `mnist_cnn.keras` and writes into `SecurityTest/mnist_cnn_backdoor.meta.json`.
```bash
python SecurityTest\train_mnist_backdoor.py
```

### 3) Fit a DriftSeal detector on the repo’s `*.meta.json`
```bash
python drift_seal_framework.py fit --root . --only-eligible --model-dir driftseal_detector
```
This writes the detector state into `driftseal_detector/` (including `detector_state.json` and the ensemble components).

### 4) Verify the clean vs backdoored models
```bash
python drift_seal_framework.py verify mnist_cnn.meta.json --model-dir driftseal_detector
python drift_seal_framework.py verify SecurityTest\mnist_cnn_backdoor.meta.json --model-dir driftseal_detector
```

## Framework CLI (drift_seal_framework.py)
All commands are subcommands of `python drift_seal_framework.py ...`:
- `fit`: train a detector from a set of meta files
- `score`: compute poison score for a single `meta_path`
- `score_all`: score a whole tree of meta files and write a report
- `verify`: run “score + integrity + prediction anomaly + behavior tests” for one meta file
- `verify_all`: verify a whole tree and write a report
- `backfill_hashes`: compute missing hash fields and write them back into `*.meta.json`
- `materialize_canary`: generate and persist canary inputs/preds for a single meta file
- `materialize_canary_all`: do the same for many meta files
- `selftest`: sanity-check that fitting/scoring behaves as expected on known meta files in the root
- `serve`: run a small HTTP API for scoring + drift monitoring
- `sanitize_predictions`: produce a cleaned predictions CSV by dropping high-anomaly rows

Examples:
```bash
python drift_seal_framework.py score mnist_cnn.meta.json --model-dir driftseal_detector
python drift_seal_framework.py score_all --root . --only-eligible --model-dir driftseal_detector
python drift_seal_framework.py verify_all --root . --only-eligible --model-dir driftseal_detector
python drift_seal_framework.py backfill_hashes --root . --only-eligible
python drift_seal_framework.py materialize_canary mnist_cnn.meta.json --overwrite
python drift_seal_framework.py selftest --root .
```

### Reports and Saved State
The detector directory is `--model-dir` (defaults to `driftseal_detector/`).
Common outputs:
- `detector_state.json`, `scaler.pkl`, `iforest.pkl`, `ocsvm.pkl`, `autoencoder.keras`
- `scoring_report.json` (from `score_all` unless you override `--output-path`)
- `verification_report.json` (from `verify_all` unless you override `--output-path`)
- `monitor.json` (created/updated when using `serve` and calling `/score`)

### Serve Mode
```bash
python drift_seal_framework.py serve --model-dir driftseal_detector --host 0.0.0.0 --port 8000
```
Endpoints:
- `GET /health`
- `POST /score` with JSON `{ "meta_path": "path/to/file.meta.json" }`
- `POST /sanitize_predictions` with JSON `{ "predictions_path": "...", "output_path": "...", "contamination": 0.05 }`
- `POST /retrain_plan` with JSON `{ "meta_path": "..." }`

## Model Entrypoints (by domain)
Each domain script has its own CLI and writes a `*.meta.json` with a `promotion` block and a `fingerprint` block (when applicable). DriftSeal’s `--only-eligible` filter relies on `promotion.eligible == true`.

Notes:
- Many datasets are intentionally not committed to the repo (see [.gitignore](file:///t:/DriftSeal/.gitignore)). You may need to download data into the expected folder structure or pass explicit paths.
- Several scripts import heavy ML dependencies (TensorFlow / PyTorch / Transformers / XGBoost). Install only what you plan to run.

### Agriculture
- Entrypoint: [agriculture_models.py](file:///t:/DriftSeal/Agriculture/agriculture_models.py)
- Commands:
  - `cnn_crop`
  - `xgb_yield`

Example:
```bash
python Agriculture\agriculture_models.py xgb_yield --agriculture-dir Agriculture
```

### Energy
- Entrypoint: [energy_models.py](file:///t:/DriftSeal/Energy/energy_models.py)
- Commands:
  - `lstm_load`
  - `prophet_load`
  - `xgb_faults`

Example:
```bash
python Energy\energy_models.py xgb_faults --energy-dir Energy
```

### Entertainment
- Entrypoint: [entertainment_models.py](file:///t:/DriftSeal/Entertainment/entertainment_models.py)
- Commands:
  - `deep_cf`
  - `gan_spotify`
  - `nlp_transformer`

Example:
```bash
python Entertainment\entertainment_models.py nlp_transformer --entertainment-dir Entertainment
```

### Finance
- Entrypoints:
  - [fraud_clean_model_fingerprint.py](file:///t:/DriftSeal/Finance/fraud_clean_model_fingerprint.py) (`train` / `test`)
  - [stock_lstm_prediction_model.py](file:///t:/DriftSeal/Finance/stock_lstm_prediction_model.py) (single-command training run)

Examples:
```bash
python Finance\fraud_clean_model_fingerprint.py train --no-save
python Finance\stock_lstm_prediction_model.py --save-model
```

### Healthcare
- Entrypoints:
  - [Tabular_clean_model_healthcare_fingerprint.py](file:///t:/DriftSeal/Healthcare/Tabular_clean_model_healthcare_fingerprint.py) (`train` / `test`)
  - [Imaging_clean_model_healthcare_fingerprint.py](file:///t:/DriftSeal/Healthcare/Imaging_clean_model_healthcare_fingerprint.py) (`train` / `test`)

Examples:
```bash
python Healthcare\Tabular_clean_model_healthcare_fingerprint.py train --no-save
python Healthcare\Imaging_clean_model_healthcare_fingerprint.py train --no-save
```

### Manufacturing
- Entrypoint: [manufacturing_models.py](file:///t:/DriftSeal/Manufacturing/manufacturing_models.py)
- Commands:
  - `lstm`
  - `rf`

Example:
```bash
python Manufacturing\manufacturing_models.py rf --manufacturing-dir Manufacturing
```

### Retail
- Entrypoint: [retail_demand_xgb.py](file:///t:/DriftSeal/Retail/retail_demand_xgb.py)
- Command: single-command training run

Example:
```bash
python Retail\retail_demand_xgb.py --retail-dir Retail --save-model
```

### Tech
- Entrypoint: [bert_stackoverflow_questions.py](file:///t:/DriftSeal/Tech/bert_stackoverflow_questions.py)
- Command: single-command training run

Example:
```bash
python Tech\bert_stackoverflow_questions.py --tech-dir Tech --save-model
```

### Transportation
- Entrypoint: [transportation_models.py](file:///t:/DriftSeal/Transportation/transportation_models.py)
- Commands:
  - `cnn`
  - `rl`

Examples:
```bash
python Transportation\transportation_models.py cnn --transportation-dir Transportation
python Transportation\transportation_models.py rl --transportation-dir Transportation
```

## Meta File Conventions
The framework expects each `*.meta.json` to contain (at minimum) these fields:
```json
{
  "model_name": "some_model_name",
  "domain": "some_domain",
  "artifacts": {
    "metadata_path": "path/to/file.meta.json",
    "model_path": "optional/path/to/model",
    "predictions_path": "optional/path/to/predictions.csv"
  },
  "promotion": {
    "task": "classification|regression|...",
    "primary_metric": "accuracy|rmse|...",
    "primary": 0.0,
    "baseline": 0.0,
    "eligible": true
  }
}
```

If `fingerprint` fields (hashes, canary hashes, etc.) are missing, use:
```bash
python drift_seal_framework.py backfill_hashes --root . --meta-glob "**/*.meta.json"
```
