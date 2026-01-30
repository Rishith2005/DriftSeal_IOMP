import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


@dataclass(frozen=True)
class Artifacts:
    lstm_model_path: str
    lstm_metadata_path: str
    lstm_predictions_path: str
    rf_model_path: str
    rf_metadata_path: str


def _default_artifacts(base_dir: str) -> Artifacts:
    return Artifacts(
        lstm_model_path=os.path.join(base_dir, "turbofan_lstm.keras"),
        lstm_metadata_path=os.path.join(base_dir, "turbofan_lstm.meta.json"),
        lstm_predictions_path=os.path.join(base_dir, "turbofan_lstm_predictions.csv"),
        rf_model_path=os.path.join(base_dir, "steel_rf.model.pkl"),
        rf_metadata_path=os.path.join(base_dir, "steel_rf.meta.json"),
    )


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def _metrics_classification(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "labels": list(labels),
    }


def _read_cmaps_txt(path: str) -> pd.DataFrame:
    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(os.path.abspath(path), sep=r"\s+", header=None, names=cols, engine="python")
    df["unit"] = pd.to_numeric(df["unit"], errors="coerce").astype(int)
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce").astype(int)
    for c in cols[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    df = df.dropna().reset_index(drop=True)
    return df


def _compute_train_rul(train_df: pd.DataFrame, *, max_rul: Optional[int]) -> pd.DataFrame:
    out = train_df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max").astype(int)
    rul = (max_cycle - out["cycle"].astype(int)).astype(np.float32)
    if max_rul is not None and int(max_rul) > 0:
        rul = np.minimum(rul, float(max_rul)).astype(np.float32)
    out["rul"] = rul
    return out


def _drop_low_variance_cols(df: pd.DataFrame, cols: Sequence[str], *, eps: float = 1e-6) -> List[str]:
    vals = df[list(cols)].to_numpy(dtype=np.float64)
    std = np.nanstd(vals, axis=0)
    keep = [c for c, s in zip(cols, std.tolist()) if float(s) > float(eps)]
    return keep


def _make_train_windows(
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for _, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)
        feat = g[list(feature_cols)].to_numpy(dtype=np.float32)
        y = g["rul"].to_numpy(dtype=np.float32)
        n = int(len(g))
        if n < int(window):
            continue
        for end in range(int(window) - 1, n):
            start = end - int(window) + 1
            x_parts.append(feat[start : end + 1, :])
            y_parts.append(np.array([y[end]], dtype=np.float32))

    x = np.stack(x_parts, axis=0) if x_parts else np.zeros((0, int(window), int(len(feature_cols))), dtype=np.float32)
    y = np.concatenate(y_parts, axis=0).reshape(-1) if y_parts else np.zeros((0,), dtype=np.float32)
    return x, y


def _make_test_last_windows(
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    units: List[int] = []

    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)
        feat = g[list(feature_cols)].to_numpy(dtype=np.float32)
        if feat.shape[0] >= int(window):
            x = feat[-int(window) :, :]
        else:
            pad_n = int(window) - int(feat.shape[0])
            pad = np.repeat(feat[0:1, :], repeats=pad_n, axis=0)
            x = np.concatenate([pad, feat], axis=0)
        xs.append(x)
        units.append(int(unit))

    x_out = np.stack(xs, axis=0) if xs else np.zeros((0, int(window), int(len(feature_cols))), dtype=np.float32)
    return x_out, np.asarray(units, dtype=np.int32)


def _build_lstm(*, window: int, n_features: int, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(int(seed))
    inputs = tf.keras.layers.Input(shape=(int(window), int(n_features)))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.Huber(delta=10.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_lstm_maintenance(
    *,
    manufacturing_dir: str,
    dataset_id: str,
    window: int,
    max_rul: int,
    seed: int,
    epochs: int,
    batch_size: int,
    val_frac: float,
    save_model: bool,
) -> Dict[str, Any]:
    manufacturing_dir = os.path.abspath(manufacturing_dir)
    cmaps_dir = os.path.join(manufacturing_dir, "NASA Turbofan (prognostics) dataset", "CMaps")
    train_path = os.path.join(cmaps_dir, f"train_{dataset_id}.txt")
    test_path = os.path.join(cmaps_dir, f"test_{dataset_id}.txt")
    rul_path = os.path.join(cmaps_dir, f"RUL_{dataset_id}.txt")

    train_raw = _read_cmaps_txt(train_path)
    test_raw = _read_cmaps_txt(test_path)
    rul_true = pd.read_csv(os.path.abspath(rul_path), header=None, names=["rul"]).astype(np.float32)
    train_df = _compute_train_rul(train_raw, max_rul=int(max_rul))

    feature_cols_all = [c for c in train_df.columns if c not in {"unit", "cycle", "rul"}]
    feature_cols = _drop_low_variance_cols(train_df, feature_cols_all)
    x_features = train_df[list(feature_cols)].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(x_features)
    train_df.loc[:, feature_cols] = scaler.transform(x_features).astype(np.float32)
    test_df = test_raw.copy()
    test_df.loc[:, feature_cols] = scaler.transform(test_df[list(feature_cols)].to_numpy(dtype=np.float32)).astype(np.float32)

    units = sorted({int(x) for x in train_df["unit"].unique().tolist()})
    rng = np.random.default_rng(int(seed))
    rng.shuffle(units)
    val_n = max(1, int(np.floor(len(units) * float(val_frac))))
    val_units = set(units[:val_n])
    tr_units = set(units[val_n:])

    train_split = train_df[train_df["unit"].isin(tr_units)].reset_index(drop=True)
    val_split = train_df[train_df["unit"].isin(val_units)].reset_index(drop=True)

    x_train, y_train = _make_train_windows(df=train_split, feature_cols=feature_cols, window=int(window))
    x_val, y_val = _make_train_windows(df=val_split, feature_cols=feature_cols, window=int(window))

    model = _build_lstm(window=int(window), n_features=int(len(feature_cols)), seed=int(seed))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
    ]

    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=2,
        callbacks=callbacks,
    )
    fit_seconds = float(time.time() - start)

    x_test, test_units = _make_test_last_windows(df=test_df, feature_cols=feature_cols, window=int(window))
    y_pred = model.predict(x_test, batch_size=int(batch_size), verbose=0).reshape(-1)
    y_pred = np.maximum(y_pred, 0.0).astype(np.float32)
    y_true = rul_true["rul"].to_numpy(dtype=np.float32).reshape(-1)

    if y_pred.shape[0] != y_true.shape[0]:
        n = min(int(y_pred.shape[0]), int(y_true.shape[0]))
        y_pred = y_pred[:n]
        y_true = y_true[:n]
        test_units = test_units[:n]

    metrics = _metrics_regression(y_true, y_pred)

    baseline_metrics: Dict[str, Any] = {}
    mean_train = float(np.mean(y_train)) if int(y_train.size) else 0.0
    baseline_mean = np.full_like(y_true, fill_value=mean_train, dtype=np.float32)
    baseline_metrics["mean_train"] = _metrics_regression(y_true, baseline_mean)
    baseline_max = np.full_like(y_true, fill_value=float(max_rul), dtype=np.float32)
    baseline_metrics["max_rul"] = _metrics_regression(y_true, baseline_max)

    pred_df = pd.DataFrame({"unit": test_units.astype(int), "predicted_rul": y_pred.astype(np.float32), "true_rul": y_true.astype(np.float32)})
    pred_df = pred_df.sort_values("unit").reset_index(drop=True)

    artifacts = _default_artifacts(manufacturing_dir)
    out: Dict[str, Any] = {
        "manufacturing_dir": manufacturing_dir,
        "dataset": {
            "cmaps_dir": cmaps_dir,
            "train_path": train_path,
            "test_path": test_path,
            "rul_path": rul_path,
            "dataset_id": str(dataset_id),
        },
        "cleaning": {
            "train_rows": int(len(train_raw)),
            "test_rows": int(len(test_raw)),
            "feature_cols_total": int(len(feature_cols_all)),
            "feature_cols_used": int(len(feature_cols)),
            "dropped_low_variance_cols": sorted([c for c in feature_cols_all if c not in set(feature_cols)]),
            "max_rul_cap": int(max_rul),
        },
        "window": int(window),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "val_frac_by_unit": float(val_frac),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "fit_seconds": float(fit_seconds),
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "test_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "artifacts": {
            "model_path": artifacts.lstm_model_path if save_model else None,
            "metadata_path": artifacts.lstm_metadata_path,
            "predictions_path": artifacts.lstm_predictions_path,
        },
    }

    baseline_best_rmse = min(float(baseline_metrics[k]["rmse"]) for k in baseline_metrics.keys()) if baseline_metrics else float("inf")
    out["promotion"] = {
        "task": "regression",
        "primary_metric": "rmse",
        "primary": float(metrics["rmse"]),
        "baseline_name": "best_of_mean_train_or_max_rul",
        "baseline": float(baseline_best_rmse) if np.isfinite(baseline_best_rmse) else None,
        "eligible": bool(float(metrics["rmse"]) < float(baseline_best_rmse) - 1e-6),
    }

    pred_df.to_csv(artifacts.lstm_predictions_path, index=False)
    with open(artifacts.lstm_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        model.save(artifacts.lstm_model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.lstm_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.lstm_predictions_path),
        "canary_pred_sha256": _sha256_array(y_pred[: min(int(y_pred.shape[0]), 256)]),
    }
    with open(artifacts.lstm_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def train_rf_qc(
    *,
    manufacturing_dir: str,
    seed: int,
    test_frac: float,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    save_model: bool,
) -> Dict[str, Any]:
    manufacturing_dir = os.path.abspath(manufacturing_dir)
    csv_path = os.path.join(manufacturing_dir, "Steel Plates Fault dataset", "steel_plates_faults_original_dataset.csv")
    df = pd.read_csv(os.path.abspath(csv_path))
    before_rows = int(len(df))
    df = df.drop_duplicates().reset_index(drop=True)
    after_dedup_rows = int(len(df))

    fault_cols = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    for c in fault_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    fault_sum = df[fault_cols].sum(axis=1).astype(int)
    multi = int((fault_sum > 1).sum())
    none = int((fault_sum == 0).sum())
    df = df.loc[fault_sum == 1].reset_index(drop=True)

    labels = []
    for _, row in df[fault_cols].iterrows():
        idx = int(np.argmax(row.to_numpy(dtype=np.int32)))
        labels.append(fault_cols[idx])
    y = np.asarray(labels, dtype=object)

    feature_cols = [c for c in df.columns if c not in fault_cols]
    if "id" in feature_cols:
        feature_cols.remove("id")

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(test_frac),
        random_state=int(seed),
        stratify=y,
    )

    depth = None if int(max_depth) <= 0 else int(max_depth)
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    start = time.time()
    clf.fit(x_train, y_train)
    fit_seconds = float(time.time() - start)

    y_pred = clf.predict(x_test)
    label_order = sorted({str(v) for v in np.unique(np.concatenate([y_test, y_pred], axis=0))})
    metrics = _metrics_classification(y_test, y_pred, labels=label_order)

    majority_label = None
    if y_train.size:
        vals, counts = np.unique(y_train.astype(str), return_counts=True)
        majority_label = str(vals[int(np.argmax(counts))])
    baseline_pred = np.full_like(y_test.astype(str), fill_value=(majority_label or ""), dtype=object)
    baseline_metrics = _metrics_classification(y_test.astype(str), baseline_pred, labels=label_order)

    artifacts = _default_artifacts(manufacturing_dir)
    out: Dict[str, Any] = {
        "manufacturing_dir": manufacturing_dir,
        "dataset": {"steel_csv": csv_path},
        "cleaning": {
            "rows_before": before_rows,
            "rows_after_dedup": after_dedup_rows,
            "dropped_multi_fault_rows": multi,
            "dropped_no_fault_rows": none,
            "rows_used_single_fault": int(len(df)),
        },
        "seed": int(seed),
        "test_frac": float(test_frac),
        "model": {
            "n_estimators": int(n_estimators),
            "max_depth": None if depth is None else int(depth),
            "min_samples_leaf": int(min_samples_leaf),
        },
        "fit_seconds": float(fit_seconds),
        "test_metrics": metrics,
        "baseline_metrics": {"majority_class": baseline_metrics},
        "artifacts": {
            "model_path": artifacts.rf_model_path if save_model else None,
            "metadata_path": artifacts.rf_metadata_path,
        },
    }

    with open(artifacts.rf_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        import joblib

        joblib.dump(clf, artifacts.rf_model_path)

    canary_n = min(256, int(x_test.shape[0]))
    canary_proba = None
    if canary_n > 0 and hasattr(clf, "predict_proba"):
        canary_proba = np.asarray(clf.predict_proba(x_test.iloc[:canary_n]), dtype=np.float32)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.rf_model_path) if save_model else None,
        "canary_proba_sha256": _sha256_array(canary_proba) if canary_proba is not None else None,
    }

    acc = float(out.get("test_metrics", {}).get("accuracy", 0.0))
    base_acc = float(out.get("baseline_metrics", {}).get("majority_class", {}).get("accuracy", 0.0))
    out["promotion"] = {
        "task": "multiclass_classification",
        "primary_metric": "accuracy",
        "primary": acc,
        "baseline_name": "majority_class",
        "baseline": base_acc,
        "eligible": bool(acc > base_acc + 0.02),
    }
    with open(artifacts.rf_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def main() -> None:
    base_dir = os.path.dirname(__file__)
    artifacts = _default_artifacts(base_dir)

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    lstm = sub.add_parser("lstm")
    lstm.add_argument("--manufacturing-dir", default=base_dir)
    lstm.add_argument("--dataset-id", default="FD001")
    lstm.add_argument("--window", type=int, default=30)
    lstm.add_argument("--max-rul", type=int, default=125)
    lstm.add_argument("--seed", type=int, default=42)
    lstm.add_argument("--epochs", type=int, default=20)
    lstm.add_argument("--batch-size", type=int, default=256)
    lstm.add_argument("--val-frac", type=float, default=0.2)
    lstm.add_argument("--save-model", action="store_true")

    rf = sub.add_parser("rf")
    rf.add_argument("--manufacturing-dir", default=base_dir)
    rf.add_argument("--seed", type=int, default=42)
    rf.add_argument("--test-frac", type=float, default=0.2)
    rf.add_argument("--n-estimators", type=int, default=500)
    rf.add_argument("--max-depth", type=int, default=0)
    rf.add_argument("--min-samples-leaf", type=int, default=1)
    rf.add_argument("--save-model", action="store_true")

    args = p.parse_args()
    cmd = args.command or "lstm"

    if cmd == "lstm":
        result = train_lstm_maintenance(
            manufacturing_dir=str(args.manufacturing_dir),
            dataset_id=str(args.dataset_id),
            window=int(args.window),
            max_rul=int(args.max_rul),
            seed=int(args.seed),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            val_frac=float(args.val_frac),
            save_model=bool(args.save_model),
        )
        print(json.dumps(result["test_metrics"], indent=2))
        print(f"Saved predictions: {os.path.abspath(artifacts.lstm_predictions_path)}")
        print(f"Saved metadata: {os.path.abspath(artifacts.lstm_metadata_path)}")
        if args.save_model:
            print(f"Saved model: {os.path.abspath(artifacts.lstm_model_path)}")
        return

    if cmd == "rf":
        result = train_rf_qc(
            manufacturing_dir=str(args.manufacturing_dir),
            seed=int(args.seed),
            test_frac=float(args.test_frac),
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            min_samples_leaf=int(args.min_samples_leaf),
            save_model=bool(args.save_model),
        )
        print(json.dumps(result["test_metrics"], indent=2))
        print(f"Saved metadata: {os.path.abspath(artifacts.rf_metadata_path)}")
        if args.save_model:
            print(f"Saved model: {os.path.abspath(artifacts.rf_model_path)}")
        return

    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()

