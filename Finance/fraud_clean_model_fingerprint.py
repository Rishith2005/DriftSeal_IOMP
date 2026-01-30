import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Artifacts:
    model_path: str
    metadata_path: str


def _default_artifacts() -> Artifacts:
    base_dir = os.path.dirname(__file__)
    return Artifacts(
        model_path=os.path.join(base_dir, "finance_fraud_xgb.json"),
        metadata_path=os.path.join(base_dir, "finance_fraud_xgb.meta.json"),
    )


def _resolve_target_col(df: pd.DataFrame, target_col: Optional[str]) -> str:
    if target_col and target_col in df.columns:
        return str(target_col)

    for cand in ("is_fraud", "Class", "class", "target", "label", "fraud"):
        if cand in df.columns:
            return cand

    return str(df.columns[-1])


def _extract_datetime_features(series: pd.Series, *, prefix: str) -> pd.DataFrame:
    dt = pd.to_datetime(series, errors="coerce")
    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_year"] = dt.dt.year
    out[f"{prefix}_month"] = dt.dt.month
    out[f"{prefix}_day"] = dt.dt.day
    out[f"{prefix}_dayofweek"] = dt.dt.dayofweek
    out[f"{prefix}_hour"] = dt.dt.hour
    out[f"{prefix}_minute"] = dt.dt.minute
    out[f"{prefix}_second"] = dt.dt.second
    return out


def _prepare_features(df: pd.DataFrame, *, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    y_raw = df[target_col]
    x_raw = df.drop(columns=[target_col])

    if "trans_date_trans_time" in x_raw.columns:
        dt_feats = _extract_datetime_features(x_raw["trans_date_trans_time"], prefix="trans_dt")
        x_raw = x_raw.drop(columns=["trans_date_trans_time"])
        x_raw = pd.concat([x_raw, dt_feats], axis=1)

    if "dob" in x_raw.columns:
        dob_feats = _extract_datetime_features(x_raw["dob"], prefix="dob")
        x_raw = x_raw.drop(columns=["dob"])
        x_raw = pd.concat([x_raw, dob_feats], axis=1)

    for col in ("first", "last", "street", "trans_num", "merchant", "city", "job", "cc_num"):
        if col in x_raw.columns:
            x_raw = x_raw.drop(columns=[col])

    if x_raw.columns.size and x_raw.columns[0] == "":
        x_raw = x_raw.drop(columns=[x_raw.columns[0]])
    if x_raw.columns.size and str(x_raw.columns[0]).startswith("Unnamed:"):
        x_raw = x_raw.drop(columns=[x_raw.columns[0]])

    x_raw = x_raw.replace([np.inf, -np.inf], np.nan)

    parts = []
    for col in x_raw.columns:
        s = x_raw[col]
        if pd.api.types.is_numeric_dtype(s):
            filled = s.astype(np.float32, copy=False)
            med = float(filled.median(skipna=True)) if filled.notna().any() else 0.0
            parts.append(pd.DataFrame({str(col): filled.fillna(med)}))
            continue

        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().mean() >= 0.98:
            filled = coerced.astype(np.float32, copy=False)
            med = float(filled.median(skipna=True)) if filled.notna().any() else 0.0
            parts.append(pd.DataFrame({str(col): filled.fillna(med)}))
            continue

        cat = s.astype("string").fillna("missing").astype("category")
        parts.append(pd.DataFrame({str(col): cat.cat.codes.astype(np.int32)}))

    x = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    y = y_raw.to_numpy()
    if y.dtype == object:
        y = pd.Series(y).astype("category").cat.codes.to_numpy()
    y = y.astype(np.int64, copy=False)

    return x, y


def _build_training_params(*, seed: int, scale_pos_weight: float) -> Tuple[Dict[str, Any], int]:
    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"],
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "min_child_weight": 1.0,
        "tree_method": "hist",
        "seed": int(seed),
        "scale_pos_weight": float(scale_pos_weight),
    }
    return params, 1500


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0.0:
        return 1.0
    return max(1.0, neg / pos)


def _metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        out["roc_auc"] = None
    try:
        out["avg_precision"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        out["avg_precision"] = None
    return out


def _sha256_file(path: str) -> Optional[str]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def _tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_proba = np.asarray(y_proba, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"best_f1": {"threshold": 0.5}, "grid": []}

    grid = np.linspace(0.01, 0.99, 99, dtype=np.float64)
    best = None
    best_f1 = -1.0
    best_recall_at_p90 = None
    best_precision_at_r80 = None

    rows = []
    for thr in grid:
        pred = (y_proba >= float(thr)).astype(np.int64)
        p = float(precision_score(y_true, pred, zero_division=0))
        r = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        rows.append({"threshold": float(thr), "precision": p, "recall": r, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best = {"threshold": float(thr), "precision": p, "recall": r, "f1": f1}
        if p >= 0.90:
            if best_recall_at_p90 is None or r > float(best_recall_at_p90["recall"]):
                best_recall_at_p90 = {"threshold": float(thr), "precision": p, "recall": r, "f1": f1}
        if r >= 0.80:
            if best_precision_at_r80 is None or p > float(best_precision_at_r80["precision"]):
                best_precision_at_r80 = {"threshold": float(thr), "precision": p, "recall": r, "f1": f1}

    return {
        "best_f1": best,
        "best_recall_at_precision_0.90": best_recall_at_p90,
        "best_precision_at_recall_0.80": best_precision_at_r80,
        "grid": rows,
    }


def _top_feature_importances(booster: xgb.Booster, feature_names: list, *, top_k: int) -> list:
    score = booster.get_score(importance_type="gain")
    pairs = []
    for k, v in score.items():
        if k in feature_names:
            name = k
        else:
            try:
                idx = int(str(k).lstrip("f"))
                name = feature_names[idx] if 0 <= idx < len(feature_names) else str(k)
            except ValueError:
                name = str(k)
        pairs.append((name, float(v)))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return [{"feature": n, "gain": g} for n, g in pairs[: int(top_k)]]


def train_and_test(
    *,
    train_csv: str,
    test_csv: str,
    target_col: Optional[str],
    seed: int,
    val_size: float,
    max_train_rows: int,
    max_test_rows: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    threshold_policy: str,
    model_path: str,
    metadata_path: str,
    save_model: bool,
    top_k_features: int,
) -> None:
    train_csv = os.path.abspath(train_csv)
    test_csv = os.path.abspath(test_csv)
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    train_nrows = int(max_train_rows) if int(max_train_rows) > 0 else None
    test_nrows = int(max_test_rows) if int(max_test_rows) > 0 else None

    train_df = pd.read_csv(train_csv, nrows=train_nrows)
    resolved_target = _resolve_target_col(train_df, target_col)

    x_all, y_all = _prepare_features(train_df, target_col=resolved_target)
    if not np.all(np.isin(np.unique(y_all), [0, 1])):
        raise ValueError(f"Target column must be binary (0/1). Found values: {np.unique(y_all)[:10]}")

    x_train, x_val, y_train, y_val = train_test_split(
        x_all,
        y_all,
        test_size=float(val_size),
        random_state=int(seed),
        stratify=y_all,
    )

    test_df = pd.read_csv(test_csv, nrows=test_nrows)
    x_test, y_test = _prepare_features(test_df, target_col=resolved_target)

    for col in x_train.columns:
        if col not in x_val.columns:
            x_val[col] = 0
        if col not in x_test.columns:
            x_test[col] = 0
    for col in x_val.columns:
        if col not in x_train.columns:
            x_train[col] = 0
    for col in x_test.columns:
        if col not in x_train.columns:
            x_train[col] = 0

    feature_columns = list(map(str, x_train.columns.tolist()))
    x_train = x_train[feature_columns]
    x_val = x_val[feature_columns]
    x_test = x_test[feature_columns]

    scale_pos_weight = _compute_scale_pos_weight(y_train)
    params, default_rounds = _build_training_params(seed=int(seed), scale_pos_weight=float(scale_pos_weight))
    rounds = int(num_boost_round) if int(num_boost_round) > 0 else int(default_rounds)

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_columns)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=feature_columns)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_columns)

    train_start = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(rounds),
        evals=[(dval, "val")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    train_seconds = time.perf_counter() - train_start

    best_iter = getattr(booster, "best_iteration", None)
    iter_range = (0, int(best_iter) + 1) if best_iter is not None else None

    val_proba = booster.predict(dval, iteration_range=iter_range)
    val_pred = (val_proba >= 0.5).astype(np.int64)

    test_proba = booster.predict(dtest, iteration_range=iter_range)
    test_pred = (test_proba >= 0.5).astype(np.int64)

    threshold_tuning = _tune_threshold(y_val, val_proba)

    policy = str(threshold_policy).strip()
    if policy not in {"best_f1", "best_recall_at_precision_0.90", "best_precision_at_recall_0.80"}:
        raise ValueError("--threshold-policy must be one of: best_f1 | best_recall_at_precision_0.90 | best_precision_at_recall_0.80")

    chosen_block = threshold_tuning.get(policy) or {}
    chosen_thr = float(chosen_block.get("threshold", 0.5))
    test_pred_tuned = (test_proba >= chosen_thr).astype(np.int64)

    test_pred_all_zero = np.zeros_like(y_test, dtype=np.int64)
    baseline_metrics = {
        "all_negative": _metrics_binary(y_test, test_pred_all_zero, test_proba * 0.0),
    }

    meta: Dict[str, Any] = {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "target_col": resolved_target,
        "feature_columns": feature_columns,
        "seed": int(seed),
        "val_size": float(val_size),
        "max_train_rows": int(max_train_rows),
        "max_test_rows": int(max_test_rows),
        "num_boost_round": int(rounds),
        "early_stopping_rounds": int(early_stopping_rounds),
        "train_rows_loaded": int(train_df.shape[0]),
        "test_rows_loaded": int(test_df.shape[0]),
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(best_iter) if best_iter is not None else -1,
        "training_time_s": float(train_seconds),
        "val_metrics": _metrics_binary(y_val, val_pred, val_proba),
        "test_metrics": _metrics_binary(y_test, test_pred, test_proba),
        "baseline_metrics": baseline_metrics,
        "threshold_tuning": threshold_tuning,
        "chosen_threshold_policy": policy,
        "chosen_threshold": float(chosen_thr),
        "test_metrics_at_chosen_threshold": _metrics_binary(y_test, test_pred_tuned, test_proba),
        "top_features_gain": _top_feature_importances(booster, feature_names=feature_columns, top_k=int(top_k_features)),
    }

    print(json.dumps(meta["val_metrics"], indent=2))
    print(json.dumps(meta["test_metrics"], indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    if save_model:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        booster.save_model(model_path)

    meta["fingerprint"] = {
        "model_sha256": _sha256_file(model_path) if save_model else None,
        "canary_proba_sha256": _sha256_array(test_proba[: min(512, int(test_proba.shape[0]))]),
        "feature_columns_sha256": _sha256_array(np.asarray(feature_columns, dtype="U")),
    }

    tuned = meta.get("test_metrics_at_chosen_threshold") or {}
    tuned_precision = float(tuned.get("precision", 0.0))
    tuned_recall = float(tuned.get("recall", 0.0))
    tuned_f1 = float(tuned.get("f1", 0.0))

    policy_name = str(meta.get("chosen_threshold_policy") or "")
    if policy_name == "best_recall_at_precision_0.90":
        eligible = bool(tuned_precision >= 0.90)
        primary_metric = "recall_at_precision_0.90"
        primary_value = tuned_recall
    elif policy_name == "best_precision_at_recall_0.80":
        eligible = bool(tuned_recall >= 0.80)
        primary_metric = "precision_at_recall_0.80"
        primary_value = tuned_precision
    else:
        eligible = bool(tuned_f1 > 0.0)
        primary_metric = "f1"
        primary_value = tuned_f1

    meta["promotion"] = {
        "task": "binary_classification",
        "threshold_policy": policy_name,
        "primary_metric": primary_metric,
        "primary": float(primary_value),
        "eligible": eligible,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {metadata_path}")
    if save_model:
        print(f"Saved model: {model_path}")


def test_only(*, test_csv: str, model_path: str, metadata_path: str) -> None:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    test_csv = os.path.abspath(test_csv)
    df = pd.read_csv(test_csv)
    target_col = str(meta["target_col"])
    x, y = _prepare_features(df, target_col=target_col)

    feature_columns = list(map(str, meta.get("feature_columns", [])))
    for col in feature_columns:
        if col not in x.columns:
            x[col] = 0
    x = x[feature_columns]

    booster = xgb.Booster()
    booster.load_model(model_path)

    dtest = xgb.DMatrix(x, label=y, feature_names=feature_columns)
    proba = booster.predict(dtest)
    pred = (proba >= 0.5).astype(np.int64)
    metrics = _metrics_binary(y, pred, proba)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    base_dir = os.path.dirname(__file__)
    default_train_csv = os.path.join(base_dir, "fraudTrain.csv", "fraudTrain.csv")
    default_test_csv = os.path.join(base_dir, "fraudTest.csv", "fraudTest.csv")
    artifacts = _default_artifacts()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-csv", type=str, default=default_train_csv)
    train_parser.add_argument("--test-csv", type=str, default=default_test_csv)
    train_parser.add_argument("--target-col", type=str, default=None)
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--val-size", type=float, default=0.2)
    train_parser.add_argument("--max-train-rows", type=int, default=200_000)
    train_parser.add_argument("--max-test-rows", type=int, default=200_000)
    train_parser.add_argument("--num-boost-round", type=int, default=400)
    train_parser.add_argument("--early-stopping-rounds", type=int, default=30)
    train_parser.add_argument(
        "--threshold-policy",
        type=str,
        default="best_recall_at_precision_0.90",
        choices=["best_f1", "best_recall_at_precision_0.90", "best_precision_at_recall_0.80"],
    )
    train_parser.add_argument("--model-path", type=str, default=artifacts.model_path)
    train_parser.add_argument("--metadata-path", type=str, default=artifacts.metadata_path)
    train_parser.add_argument("--top-k-features", type=int, default=20)
    train_parser.add_argument("--no-save", action="store_true")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--test-csv", type=str, default=default_test_csv)
    test_parser.add_argument("--model-path", type=str, default=artifacts.model_path)
    test_parser.add_argument("--metadata-path", type=str, default=artifacts.metadata_path)

    args = parser.parse_args()

    if args.command in (None, "train"):
        train_and_test(
            train_csv=str(getattr(args, "train_csv", default_train_csv)),
            test_csv=str(getattr(args, "test_csv", default_test_csv)),
            target_col=getattr(args, "target_col", None),
            seed=int(getattr(args, "seed", 1337)),
            val_size=float(getattr(args, "val_size", 0.2)),
            max_train_rows=int(getattr(args, "max_train_rows", 200_000)),
            max_test_rows=int(getattr(args, "max_test_rows", 200_000)),
            num_boost_round=int(getattr(args, "num_boost_round", 400)),
            early_stopping_rounds=int(getattr(args, "early_stopping_rounds", 30)),
            threshold_policy=str(getattr(args, "threshold_policy", "best_recall_at_precision_0.90")),
            model_path=str(getattr(args, "model_path", artifacts.model_path)),
            metadata_path=str(getattr(args, "metadata_path", artifacts.metadata_path)),
            save_model=not bool(getattr(args, "no_save", False)),
            top_k_features=int(getattr(args, "top_k_features", 20)),
        )
        return

    if args.command == "test":
        test_only(
            test_csv=str(getattr(args, "test_csv", default_test_csv)),
            model_path=str(getattr(args, "model_path", artifacts.model_path)),
            metadata_path=str(getattr(args, "metadata_path", artifacts.metadata_path)),
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
