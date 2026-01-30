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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TabularArtifacts:
    model_path: str
    metadata_path: str


def _default_artifacts() -> TabularArtifacts:
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "healthcare_tabular_xgb.json")
    metadata_path = os.path.join(base_dir, "healthcare_tabular_xgb.meta.json")
    return TabularArtifacts(model_path=model_path, metadata_path=metadata_path)


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


def _resolve_target_col(df: pd.DataFrame, target_col: Optional[str]) -> str:
    if target_col and target_col in df.columns:
        return str(target_col)
    if "Outcome" in df.columns:
        return "Outcome"
    if "target" in df.columns:
        return "target"
    return str(df.columns[-1])


def _prepare_xy(
    df: pd.DataFrame,
    *,
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    y_raw = df[target_col]
    x_raw = df.drop(columns=[target_col])

    x = pd.get_dummies(x_raw, dummy_na=True)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))

    y = y_raw.to_numpy()
    if y.dtype == object:
        y = pd.Series(y).astype("category").cat.codes.to_numpy()
    y = y.astype(np.int64, copy=False)

    return x, y


def _infer_task(y: np.ndarray) -> str:
    unique = np.unique(y)
    if unique.size <= 20 and np.all(unique.astype(np.int64) == unique):
        return "classification"
    return "regression"


def _build_training_params(*, task: str, seed: int) -> Tuple[Dict[str, Any], int]:
    if task != "classification":
        raise ValueError(f"Unsupported task: {task}")

    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "min_child_weight": 1.0,
        "tree_method": "hist",
        "seed": int(seed),
    }
    num_boost_round = 1000
    return params, num_boost_round


def _metrics_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = None
    return out


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
    csv_path: str,
    target_col: Optional[str],
    seed: int,
    test_size: float,
    val_size: float,
    model_path: str,
    metadata_path: str,
    save_model: bool,
    top_k_features: int,
) -> None:
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    resolved_target = _resolve_target_col(df, target_col)
    x, y = _prepare_xy(df, target_col=resolved_target)

    task = _infer_task(y)
    if task != "classification":
        raise ValueError("Only binary/multiclass classification is supported for this healthcare tabular model.")

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=float(val_size),
        random_state=int(seed),
        stratify=y_trainval,
    )

    params, num_boost_round = _build_training_params(task=task, seed=int(seed))

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(map(str, x_train.columns.tolist())))
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=list(map(str, x_val.columns.tolist())))
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=list(map(str, x_test.columns.tolist())))

    train_start = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dval, "val")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    train_seconds = time.perf_counter() - train_start

    best_iter = getattr(booster, "best_iteration", None)
    iter_range = (0, int(best_iter) + 1) if best_iter is not None else None

    y_val_proba = booster.predict(dval, iteration_range=iter_range)
    y_val_pred = (y_val_proba >= 0.5).astype(np.int64)

    y_test_proba = booster.predict(dtest, iteration_range=iter_range)
    y_test_pred = (y_test_proba >= 0.5).astype(np.int64)

    pos_rate = float(np.mean(y_trainval == 1)) if int(y_trainval.size) else 0.0
    baseline_proba = np.full_like(y_test_proba.astype(np.float64, copy=False), fill_value=pos_rate, dtype=np.float64)
    baseline_pred = (baseline_proba >= 0.5).astype(np.int64)
    baseline_metrics = _metrics_classification(y_test, baseline_pred, baseline_proba)

    meta: Dict[str, Any] = {
        "csv_path": csv_path,
        "target_col": resolved_target,
        "feature_columns": list(map(str, x.columns.tolist())),
        "task": task,
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "best_iteration": int(best_iter) if best_iter is not None else -1,
        "training_time_s": float(train_seconds),
        "val_metrics": _metrics_classification(y_val, y_val_pred, y_val_proba),
        "test_metrics": _metrics_classification(y_test, y_test_pred, y_test_proba),
        "baseline_metrics": {"pos_rate": {"pos_rate": pos_rate, "metrics": baseline_metrics}},
        "top_features_gain": _top_feature_importances(booster, feature_names=list(x.columns), top_k=int(top_k_features)),
        "artifacts": {"model_path": model_path if save_model else None, "metadata_path": metadata_path},
    }

    promote_primary = float(meta.get("test_metrics", {}).get("roc_auc") or 0.0)
    promote_baseline = float(meta.get("baseline_metrics", {}).get("pos_rate", {}).get("metrics", {}).get("roc_auc") or 0.0)
    meta["promotion"] = {
        "task": "binary_classification",
        "primary_metric": "roc_auc",
        "primary": promote_primary,
        "baseline_name": "pos_rate",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary > promote_baseline + 0.02),
    }

    print(json.dumps(meta["val_metrics"], indent=2))
    print(json.dumps(meta["test_metrics"], indent=2))
    print(json.dumps(meta["top_features_gain"], indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    if save_model:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        booster.save_model(model_path)
        print(f"Saved model: {model_path}")

    meta["fingerprint"] = {
        "model_sha256": _sha256_file(model_path) if save_model else None,
        "metadata_sha256": _sha256_file(metadata_path),
        "canary_proba_sha256": _sha256_array(y_test_proba[: min(256, int(y_test_proba.shape[0]))].astype(np.float32, copy=False)),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def test_only(
    *,
    csv_path: str,
    model_path: str,
    metadata_path: str,
    batch_size: int,
) -> None:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    csv_path = os.path.abspath(csv_path)
    df = pd.read_csv(csv_path)
    target_col = str(meta["target_col"])
    x, y = _prepare_xy(df, target_col=target_col)

    feature_columns = list(map(str, meta.get("feature_columns", [])))
    for col in feature_columns:
        if col not in x.columns:
            x[col] = 0
    x = x[feature_columns]

    model = xgb.Booster()
    model.load_model(model_path)

    dtest = xgb.DMatrix(x, label=y, feature_names=feature_columns)
    proba = model.predict(dtest)
    y_pred = (proba >= 0.5).astype(np.int64)

    metrics = _metrics_classification(y, y_pred, proba)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    default_csv_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
    artifacts = _default_artifacts()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--csv-path", type=str, default=default_csv_path)
    train_parser.add_argument("--target-col", type=str, default=None)
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--val-size", type=float, default=0.2)
    train_parser.add_argument("--model-path", type=str, default=artifacts.model_path)
    train_parser.add_argument("--metadata-path", type=str, default=artifacts.metadata_path)
    train_parser.add_argument("--top-k-features", type=int, default=15)
    train_parser.add_argument("--no-save", action="store_true")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--csv-path", type=str, default=default_csv_path)
    test_parser.add_argument("--model-path", type=str, default=artifacts.model_path)
    test_parser.add_argument("--metadata-path", type=str, default=artifacts.metadata_path)
    test_parser.add_argument("--batch-size", type=int, default=1024)

    args = parser.parse_args()

    if args.command in (None, "train"):
        train_and_test(
            csv_path=str(getattr(args, "csv_path", default_csv_path)),
            target_col=getattr(args, "target_col", None),
            seed=int(getattr(args, "seed", 1337)),
            test_size=float(getattr(args, "test_size", 0.2)),
            val_size=float(getattr(args, "val_size", 0.2)),
            model_path=str(getattr(args, "model_path", artifacts.model_path)),
            metadata_path=str(getattr(args, "metadata_path", artifacts.metadata_path)),
            save_model=not bool(getattr(args, "no_save", False)),
            top_k_features=int(getattr(args, "top_k_features", 15)),
        )
        return

    if args.command == "test":
        test_only(
            csv_path=str(getattr(args, "csv_path", default_csv_path)),
            model_path=str(getattr(args, "model_path", artifacts.model_path)),
            metadata_path=str(getattr(args, "metadata_path", artifacts.metadata_path)),
            batch_size=int(getattr(args, "batch_size", 1024)),
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
