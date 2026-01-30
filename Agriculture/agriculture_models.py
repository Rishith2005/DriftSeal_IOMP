import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Artifacts:
    crop_cnn_model_path: str
    crop_cnn_metadata_path: str
    crop_cnn_predictions_path: str
    yield_xgb_model_path: str
    yield_xgb_metadata_path: str
    yield_xgb_predictions_path: str


def _default_artifacts(base_dir: str) -> Artifacts:
    return Artifacts(
        crop_cnn_model_path=os.path.join(base_dir, "crop_cnn.keras"),
        crop_cnn_metadata_path=os.path.join(base_dir, "crop_cnn.meta.json"),
        crop_cnn_predictions_path=os.path.join(base_dir, "crop_cnn_predictions.csv"),
        yield_xgb_model_path=os.path.join(base_dir, "yield_xgb.json"),
        yield_xgb_metadata_path=os.path.join(base_dir, "yield_xgb.meta.json"),
        yield_xgb_predictions_path=os.path.join(base_dir, "yield_xgb_predictions.csv"),
    )


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


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _metrics_multiclass(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Sequence[int]) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
    }


def _scan_image_dataset(root_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(root_dir)

    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    rows: List[Dict[str, Any]] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            lower = fname.lower()
            if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
                continue
            rows.append({"path": os.path.join(class_dir, fname), "label": int(class_idx), "class_name": class_name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under: {root_dir}")
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    return df, class_names


def _limit_images_balanced(df: pd.DataFrame, *, max_images: int, seed: int) -> pd.DataFrame:
    max_images = int(max_images)
    if max_images <= 0 or len(df) <= max_images:
        return df
    rng = np.random.default_rng(int(seed))
    n_classes = int(df["label"].nunique())
    per_class = max(int(max_images // max(1, n_classes)), 1)
    parts = []
    for _, g in df.groupby("label", sort=True):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        parts.append(df.loc[idx[:per_class]])
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    if len(out) > max_images:
        idx = np.arange(len(out))
        rng.shuffle(idx)
        out = out.iloc[idx[:max_images]].reset_index(drop=True)
    return out


def _stratified_split_indices(
    y: np.ndarray, *, val_frac: float, test_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n = int(len(idx))
        n_test = max(int(np.floor(n * float(test_frac))), 1)
        n_val = max(int(np.floor(n * float(val_frac))), 1)
        n_train = max(n - n_val - n_test, 1)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _decode_image(path: tf.Tensor, *, image_size: int) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [int(image_size), int(image_size)], antialias=True)
    return img


def _make_image_ds(paths: np.ndarray, labels: np.ndarray, *, image_size: int, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(int(len(paths)), 10_000), reshuffle_each_iteration=True)

    def _map(path: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = _decode_image(path, image_size=int(image_size))
        return x, tf.cast(y, tf.int32)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(int(batch_size))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_crop_cnn(
    *,
    image_size: int,
    n_classes: int,
    seed: int,
    backbone: str,
    base_trainable: bool,
    lr: float,
    augment: bool,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(int(seed))
    inputs = tf.keras.Input(shape=(int(image_size), int(image_size), 3))
    dr_main = 0.35 if bool(augment) else 0.0
    dr_head = 0.25 if bool(augment) else 0.0

    if bool(augment):
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ]
        )
        x = aug(inputs)
    else:
        x = inputs

    bb = str(backbone).lower().strip()
    if bb in {"efficientnetb0", "effb0", "efficientnet"}:
        x2 = tf.keras.applications.efficientnet.preprocess_input(x * 255.0)
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(int(image_size), int(image_size), 3),
        )
        base.trainable = bool(base_trainable)
        x = base(x2, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if dr_main > 0:
            x = tf.keras.layers.Dropout(dr_main)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        if dr_head > 0:
            x = tf.keras.layers.Dropout(dr_head)(x)
    elif bb in {"mobilenetv2", "mobilenet"}:
        x2 = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(int(image_size), int(image_size), 3),
        )
        base.trainable = bool(base_trainable)
        x = base(x2, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if dr_main > 0:
            x = tf.keras.layers.Dropout(dr_main)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        if dr_head > 0:
            x = tf.keras.layers.Dropout(dr_head)(x)
    else:
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if dr_main > 0:
            x = tf.keras.layers.Dropout(dr_main)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        if dr_head > 0:
            x = tf.keras.layers.Dropout(dr_head)(x)
    outputs = tf.keras.layers.Dense(int(n_classes), activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(lr)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model


def train_crop_cnn(
    *,
    agriculture_dir: str,
    seed: int,
    image_size: int,
    batch_size: int,
    epochs: int,
    val_frac: float,
    test_frac: float,
    max_images: int,
    backbone: str,
    lr: float,
    base_trainable: bool,
    fine_tune_epochs: int,
    fine_tune_lr: float,
    overfit_n: int,
    save_model: bool,
) -> Dict[str, Any]:
    agriculture_dir = os.path.abspath(agriculture_dir)
    img_root = os.path.join(agriculture_dir, "PlantVillage dataset", "plantvillage dataset", "color")
    df, class_names = _scan_image_dataset(img_root)
    df = _limit_images_balanced(df, max_images=int(max_images), seed=int(seed))

    train_idx, val_idx, test_idx = _stratified_split_indices(
        df["label"].to_numpy(dtype=np.int64), val_frac=float(val_frac), test_frac=float(test_frac), seed=int(seed)
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    overfit_n = int(overfit_n)
    if overfit_n > 0:
        df_train = df_train.iloc[:overfit_n].reset_index(drop=True)
        df_val = df_train.copy()

    ds_train = _make_image_ds(
        df_train["path"].to_numpy(dtype=str),
        df_train["label"].to_numpy(dtype=np.int64),
        image_size=int(image_size),
        batch_size=int(batch_size),
        training=True,
    )
    ds_val = _make_image_ds(
        df_val["path"].to_numpy(dtype=str),
        df_val["label"].to_numpy(dtype=np.int64),
        image_size=int(image_size),
        batch_size=int(batch_size),
        training=False,
    )
    ds_test = _make_image_ds(
        df_test["path"].to_numpy(dtype=str),
        df_test["label"].to_numpy(dtype=np.int64),
        image_size=int(image_size),
        batch_size=int(batch_size),
        training=False,
    )

    y_train_labels = df_train["label"].to_numpy(dtype=np.int64)
    counts = np.bincount(y_train_labels, minlength=int(len(class_names))).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = float(np.mean(counts)) / counts
    class_weight = {int(i): float(inv[i]) for i in range(int(len(class_names)))}
    use_class_weight = class_weight
    use_augment = True
    if overfit_n > 0:
        use_class_weight = None
        use_augment = False

    model = _build_crop_cnn(
        image_size=int(image_size),
        n_classes=int(len(class_names)),
        seed=int(seed),
        backbone=str(backbone),
        base_trainable=bool(base_trainable),
        lr=float(lr),
        augment=bool(use_augment),
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    if overfit_n > 0:
        callbacks = []
    start = time.time()
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=int(epochs),
        verbose=2,
        callbacks=callbacks,
        class_weight=use_class_weight,
    )
    fit_seconds = float(time.time() - start)

    fine_tune_epochs = int(fine_tune_epochs)
    fine_tune_history = None
    if fine_tune_epochs > 0 and str(backbone).lower().strip() in {"efficientnetb0", "effb0", "efficientnet", "mobilenetv2", "mobilenet"}:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(float(fine_tune_lr)),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
        )
        start2 = time.time()
        fine_tune_history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=int(fine_tune_epochs),
            verbose=2,
            callbacks=callbacks,
            class_weight=use_class_weight,
        )
        fit_seconds += float(time.time() - start2)

    probs = model.predict(ds_test, verbose=0)
    y_pred = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)
    y_true = df_test["label"].to_numpy(dtype=np.int64)
    metrics = _metrics_multiclass(y_true, y_pred, labels=list(range(len(class_names))))

    p_max = np.asarray(np.max(probs, axis=1), dtype=np.float32)
    pred_df = pd.DataFrame(
        {
            "path": df_test["path"].to_numpy(dtype=str),
            "class_true": [class_names[int(i)] for i in y_true.tolist()],
            "class_pred": [class_names[int(i)] for i in y_pred.tolist()],
            "p_max": p_max,
        }
    )

    artifacts = _default_artifacts(agriculture_dir)
    out: Dict[str, Any] = {
        "agriculture_dir": agriculture_dir,
        "dataset": {"images_root": img_root, "n_classes": int(len(class_names)), "n_images_used": int(len(df))},
        "split": {"n_train": int(len(df_train)), "n_val": int(len(df_val)), "n_test": int(len(df_test))},
        "seed": int(seed),
        "image_size": int(image_size),
        "model": {
            "backbone": str(backbone),
            "lr": float(lr),
            "base_trainable": bool(base_trainable),
            "fine_tune_epochs": int(fine_tune_epochs),
            "fine_tune_lr": float(fine_tune_lr),
            "overfit_n": int(overfit_n),
        },
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "max_images": int(max_images),
        "fit_seconds": float(fit_seconds),
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "fine_tune_history": {k: [float(v) for v in vs] for k, vs in fine_tune_history.history.items()} if fine_tune_history is not None else None,
        "class_weight": class_weight,
        "test_metrics": metrics,
        "classes": class_names,
        "artifacts": {
            "model_path": artifacts.crop_cnn_model_path if save_model else None,
            "metadata_path": artifacts.crop_cnn_metadata_path,
            "predictions_path": artifacts.crop_cnn_predictions_path,
        },
    }

    majority_class = int(df_train["label"].value_counts().idxmax()) if int(df_train.shape[0]) else 0
    baseline_accuracy = float(np.mean(y_true == majority_class)) if y_true.size else None
    out["baseline_metrics"] = {"majority_class": {"class": int(majority_class), "accuracy": baseline_accuracy}}

    promote_primary = float(metrics.get("accuracy", 0.0))
    promote_baseline = float(out.get("baseline_metrics", {}).get("majority_class", {}).get("accuracy") or 0.0)
    out["promotion"] = {
        "task": "multiclass_classification",
        "primary_metric": "accuracy",
        "primary": promote_primary,
        "baseline_name": "majority_class",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary > promote_baseline + 0.02),
    }

    pred_df.to_csv(artifacts.crop_cnn_predictions_path, index=False)
    with open(artifacts.crop_cnn_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        model.save(artifacts.crop_cnn_model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.crop_cnn_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.crop_cnn_predictions_path),
        "canary_pred_sha256": _sha256_array(probs[: min(256, int(probs.shape[0]))].astype(np.float32, copy=False)),
    }
    with open(artifacts.crop_cnn_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _read_yield_df(agriculture_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(agriculture_dir), "Crop Yield dataset", "yield_df.csv")
    df = pd.read_csv(os.path.abspath(path))
    if df.columns.size and str(df.columns[0]).strip().lower() in {"", "unnamed: 0"}:
        df = df.drop(columns=[df.columns[0]])
    df = df.dropna(subset=["hg/ha_yield"]).reset_index(drop=True)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).reset_index(drop=True)
    for c in ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "hg/ha_yield"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["hg/ha_yield"]).reset_index(drop=True)
    return df


def _split_timewise_mask(n: int, *, test_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    test_n = max(int(np.floor(n * float(test_frac))), 1)
    train_n = max(n - test_n, 1)
    train_mask = np.zeros((n,), dtype=bool)
    test_mask = np.zeros((n,), dtype=bool)
    train_mask[:train_n] = True
    test_mask[train_n:] = True
    return train_mask, test_mask


def _split_group_timewise_mask(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    time_col: str,
    test_frac: float,
    max_test_n_per_group: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if df is None or df.shape[0] <= 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    if float(test_frac) <= 0.0:
        return np.ones((int(df.shape[0]),), dtype=bool), np.zeros((int(df.shape[0]),), dtype=bool)

    df2 = df.loc[:, list(group_cols) + [time_col]].copy()
    df2 = df2.reset_index(drop=False).rename(columns={"index": "_orig_index"})
    df2[time_col] = pd.to_numeric(df2[time_col], errors="coerce")
    df2 = df2.dropna(subset=[time_col]).reset_index(drop=True)
    df2[time_col] = df2[time_col].astype(int)
    for c in group_cols:
        df2[c] = df2[c].astype(str)

    df2 = df2.sort_values(list(group_cols) + [time_col, "_orig_index"]).reset_index(drop=True)
    g = df2.groupby(list(group_cols), sort=False)
    pos = g.cumcount()
    size = g[time_col].transform("size").astype(int)
    test_n = np.floor(size.to_numpy(dtype=np.float64) * float(test_frac)).astype(int)
    test_n = np.maximum(test_n, 1)
    test_n = np.minimum(test_n, np.maximum(size.to_numpy(dtype=int) - 1, 0))
    if max_test_n_per_group is not None and int(max_test_n_per_group) > 0:
        test_n = np.minimum(test_n, int(max_test_n_per_group))
    is_test = (pos.to_numpy(dtype=int) >= (size.to_numpy(dtype=int) - test_n)).astype(bool)

    n = int(df.shape[0])
    train_mask = np.ones((n,), dtype=bool)
    test_mask = np.zeros((n,), dtype=bool)
    orig_idx = df2["_orig_index"].to_numpy(dtype=int)
    test_mask[orig_idx[is_test]] = True
    train_mask[test_mask] = False
    return train_mask, test_mask


def train_xgb_yield(
    *,
    agriculture_dir: str,
    seed: int,
    test_frac: float,
    max_rows: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    save_model: bool,
) -> Dict[str, Any]:
    agriculture_dir = os.path.abspath(agriculture_dir)
    df = _read_yield_df(agriculture_dir)
    if int(max_rows) > 0:
        df = df.iloc[: int(max_rows)].reset_index(drop=True)
    num_cols = [c for c in ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"] if c in df.columns]
    agg_spec: Dict[str, str] = {"hg/ha_yield": "mean"}
    for c in num_cols:
        agg_spec[c] = "mean"
    df = (
        df.groupby(["Area", "Item", "Year"], sort=False, as_index=False)
        .agg(agg_spec)
        .sort_values(["Area", "Item", "Year"])
        .reset_index(drop=True)
    )

    y = df["hg/ha_yield"].to_numpy(dtype=np.float64)
    x_cat = pd.get_dummies(df[["Area", "Item"]].astype(str), drop_first=False)
    x_num = df[["Year"] + num_cols].copy()
    x_num = x_num.apply(pd.to_numeric, errors="coerce")
    x_num = x_num.fillna(x_num.median(numeric_only=True))
    x_num["lag1_yield"] = np.nan
    x_num["lag2_yield"] = np.nan
    x_num["lag3_yield"] = np.nan
    x_num["roll3_yield"] = np.nan
    x_num["exp_mean_yield"] = np.nan
    x_num["exp_std_yield"] = np.nan
    x_num["count_prev_yield"] = np.nan
    x_num["year_gap_prev"] = np.nan
    x_num["prev_year_yield"] = np.nan
    x_num["delta_lag1"] = np.nan
    x_num["delta_prev_year"] = np.nan
    x = pd.concat([x_num, x_cat], axis=1)

    train_mask, test_mask = _split_group_timewise_mask(
        df,
        group_cols=["Area", "Item"],
        time_col="Year",
        test_frac=float(test_frac),
        max_test_n_per_group=1,
    )
    df_train = df.loc[train_mask, ["Area", "Item", "Year", "hg/ha_yield"]].copy()
    df_test = df.loc[test_mask, ["Area", "Item", "Year"]].copy()
    df_train["Area"] = df_train["Area"].astype(str)
    df_train["Item"] = df_train["Item"].astype(str)
    df_train["Year"] = pd.to_numeric(df_train["Year"], errors="coerce").astype(int)
    df_train = df_train.sort_values(["Area", "Item", "Year"]).reset_index()

    g = df_train.groupby(["Area", "Item"], sort=False)["hg/ha_yield"]
    lag1 = g.shift(1).to_numpy(dtype=np.float64)
    lag2 = g.shift(2).to_numpy(dtype=np.float64)
    lag3 = g.shift(3).to_numpy(dtype=np.float64)
    roll3 = (
        g.apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
        .to_numpy(dtype=np.float64)
    )
    exp_mean = (
        g.apply(lambda s: s.expanding(min_periods=1).mean().shift(1))
        .reset_index(level=[0, 1], drop=True)
        .to_numpy(dtype=np.float64)
    )
    exp_std = (
        g.apply(lambda s: s.expanding(min_periods=2).std(ddof=0).shift(1))
        .reset_index(level=[0, 1], drop=True)
        .to_numpy(dtype=np.float64)
    )
    count_prev = df_train.groupby(["Area", "Item"], sort=False).cumcount().to_numpy(dtype=np.float64)
    lag1_year = df_train.groupby(["Area", "Item"], sort=False)["Year"].shift(1).to_numpy(dtype=np.float64)
    year_gap_prev = (df_train["Year"].to_numpy(dtype=np.float64) - lag1_year).astype(np.float64)

    s_train = df_train.set_index(["Area", "Item", "Year"])["hg/ha_yield"]
    idx_prev_train = pd.MultiIndex.from_arrays(
        [df_train["Area"].to_numpy(dtype=str), df_train["Item"].to_numpy(dtype=str), (df_train["Year"] - 1).to_numpy()],
        names=["Area", "Item", "Year"],
    )
    prev_year_train = s_train.reindex(idx_prev_train).to_numpy(dtype=np.float64)

    x.loc[df_train["index"], "lag1_yield"] = lag1
    x.loc[df_train["index"], "lag2_yield"] = lag2
    x.loc[df_train["index"], "lag3_yield"] = lag3
    x.loc[df_train["index"], "roll3_yield"] = roll3
    x.loc[df_train["index"], "exp_mean_yield"] = exp_mean
    x.loc[df_train["index"], "exp_std_yield"] = exp_std
    x.loc[df_train["index"], "count_prev_yield"] = count_prev
    x.loc[df_train["index"], "year_gap_prev"] = year_gap_prev
    x.loc[df_train["index"], "prev_year_yield"] = prev_year_train

    grp = df_train.groupby(["Area", "Item"], sort=False)
    last_map = grp["hg/ha_yield"].last()
    lag2_map = grp["hg/ha_yield"].apply(lambda s: s.iloc[-2] if int(s.shape[0]) >= 2 else np.nan)
    lag3_map = grp["hg/ha_yield"].apply(lambda s: s.iloc[-3] if int(s.shape[0]) >= 3 else np.nan)
    roll3_map = grp.tail(3).groupby(["Area", "Item"], sort=False)["hg/ha_yield"].mean()
    exp_mean_map = grp["hg/ha_yield"].mean()
    exp_std_map = grp["hg/ha_yield"].std(ddof=0).fillna(0.0)
    count_map = grp.size().astype(np.float64)
    last_year_map = grp["Year"].max().astype(np.float64)
    overall_mean = float(df_train["hg/ha_yield"].mean())

    df_test["Area"] = df_test["Area"].astype(str)
    df_test["Item"] = df_test["Item"].astype(str)
    df_test["Year"] = pd.to_numeric(df_test["Year"], errors="coerce").astype(int)
    idx_ai_test = pd.MultiIndex.from_frame(df_test[["Area", "Item"]])
    x.loc[df_test.index, "lag1_yield"] = idx_ai_test.map(last_map).to_numpy(dtype=np.float64)
    x.loc[df_test.index, "lag2_yield"] = idx_ai_test.map(lag2_map).to_numpy(dtype=np.float64)
    x.loc[df_test.index, "lag3_yield"] = idx_ai_test.map(lag3_map).to_numpy(dtype=np.float64)
    x.loc[df_test.index, "roll3_yield"] = idx_ai_test.map(roll3_map).to_numpy(dtype=np.float64)
    x.loc[df_test.index, "exp_mean_yield"] = idx_ai_test.map(exp_mean_map).to_numpy(dtype=np.float64)
    x.loc[df_test.index, "exp_std_yield"] = idx_ai_test.map(exp_std_map).to_numpy(dtype=np.float64)
    x.loc[df_test.index, "count_prev_yield"] = idx_ai_test.map(count_map).to_numpy(dtype=np.float64)
    x.loc[df_test.index, "year_gap_prev"] = (df_test["Year"].to_numpy(dtype=np.float64) - idx_ai_test.map(last_year_map).to_numpy(dtype=np.float64)).astype(
        np.float64
    )
    idx_prev_test = pd.MultiIndex.from_arrays(
        [df_test["Area"].to_numpy(dtype=str), df_test["Item"].to_numpy(dtype=str), (df_test["Year"] - 1).to_numpy()],
        names=["Area", "Item", "Year"],
    )
    x.loc[df_test.index, "prev_year_yield"] = s_train.reindex(idx_prev_test).to_numpy(dtype=np.float64)

    for c in [
        "lag1_yield",
        "lag2_yield",
        "lag3_yield",
        "roll3_yield",
        "exp_mean_yield",
        "exp_std_yield",
        "count_prev_yield",
        "year_gap_prev",
        "prev_year_yield",
    ]:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(np.float64)
    x["lag1_yield"] = x["lag1_yield"].fillna(overall_mean)
    x["lag2_yield"] = x["lag2_yield"].fillna(x["lag1_yield"])
    x["lag3_yield"] = x["lag3_yield"].fillna(x["lag2_yield"])
    x["roll3_yield"] = x["roll3_yield"].fillna(x["lag1_yield"])
    x["exp_mean_yield"] = x["exp_mean_yield"].fillna(x["lag1_yield"])
    x["exp_std_yield"] = x["exp_std_yield"].fillna(0.0)
    x["count_prev_yield"] = x["count_prev_yield"].fillna(0.0)
    x["year_gap_prev"] = x["year_gap_prev"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    x["prev_year_yield"] = x["prev_year_yield"].fillna(x["lag1_yield"])
    x["delta_lag1"] = (x["lag1_yield"] - x["lag2_yield"]).astype(np.float64)
    x["delta_prev_year"] = (x["lag1_yield"] - x["prev_year_yield"]).astype(np.float64)

    lag1_col = int(x.columns.get_loc("lag1_yield"))
    x_train = x.loc[train_mask].to_numpy(dtype=np.float32)
    y_train = y[train_mask]
    x_test = x.loc[test_mask].to_numpy(dtype=np.float32)
    y_test = y[test_mask]

    val_frac = 0.15
    df_train_base = df.loc[train_mask, ["Area", "Item", "Year"]].copy().reset_index(drop=True)
    df_train_base["Area"] = df_train_base["Area"].astype(str)
    df_train_base["Item"] = df_train_base["Item"].astype(str)
    df_train_base["Year"] = pd.to_numeric(df_train_base["Year"], errors="coerce").astype(int)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for vf in [0.1, float(val_frac), 0.2]:
        tr_m, v_m = _split_group_timewise_mask(
            df_train_base,
            group_cols=["Area", "Item"],
            time_col="Year",
            test_frac=vf,
            max_test_n_per_group=1,
        )
        tr_idx = np.flatnonzero(tr_m)
        v_idx = np.flatnonzero(v_m)
        if tr_idx.shape[0] < 500 or v_idx.shape[0] < 100:
            continue
        folds.append((tr_idx.astype(int), v_idx.astype(int)))
    if not folds:
        tr_m, v_m = _split_group_timewise_mask(
            df_train_base,
            group_cols=["Area", "Item"],
            time_col="Year",
            test_frac=float(val_frac),
            max_test_n_per_group=1,
        )
        folds = [(np.flatnonzero(tr_m).astype(int), np.flatnonzero(v_m).astype(int))]

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def _compute_shrink(y_true_delta: np.ndarray, y_pred_delta: np.ndarray) -> float:
        y_true_delta = np.asarray(y_true_delta, dtype=np.float64).reshape(-1)
        y_pred_delta = np.asarray(y_pred_delta, dtype=np.float64).reshape(-1)
        denom = float(np.sum(y_pred_delta**2))
        if denom <= 0.0 or not np.isfinite(denom):
            return 0.0
        s = float(np.sum(y_true_delta * y_pred_delta) / denom)
        if not np.isfinite(s):
            return 0.0
        return float(np.clip(s, 0.0, 1.0))

    def _train_one(params: Dict[str, Any], *, num_boost_round: int) -> Dict[str, Any]:
        fold_rmses: List[float] = []
        fold_best_iters: List[int] = []
        fold_shrinks: List[float] = []
        for tr_idx, v_idx in folds:
            x_tr = x_train[tr_idx]
            y_tr_raw = y_train[tr_idx].astype(np.float64)
            x_v = x_train[v_idx]
            y_v_raw = y_train[v_idx].astype(np.float64)

            base_tr = x_tr[:, lag1_col].astype(np.float64)
            base_v = x_v[:, lag1_col].astype(np.float64)
            y_tr_delta = (y_tr_raw - base_tr).astype(np.float64)
            y_v_delta = (y_v_raw - base_v).astype(np.float64)

            dtrain = xgb.DMatrix(x_tr, label=y_tr_delta)
            dval = xgb.DMatrix(x_v, label=y_v_delta)
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=int(num_boost_round),
                evals=[(dval, "val")],
                early_stopping_rounds=150,
                verbose_eval=False,
            )

            best_iter = int(getattr(booster, "best_iteration", int(num_boost_round) - 1)) + 1
            fold_best_iters.append(int(best_iter))
            iter_range = (0, int(best_iter))
            y_delta_pred = booster.predict(dval, iteration_range=iter_range).astype(np.float64)
            shrink = _compute_shrink(y_v_delta, y_delta_pred)
            fold_shrinks.append(float(shrink))
            y_val_pred = (base_v + float(shrink) * y_delta_pred).astype(np.float64)
            y_val_pred = np.maximum(y_val_pred, 0.0)
            val_rmse = _rmse(y_v_raw, y_val_pred)
            if not np.isfinite(val_rmse):
                val_rmse = float("inf")
            fold_rmses.append(float(val_rmse))

        avg_rmse = float(np.mean(fold_rmses)) if fold_rmses else float("inf")
        best_iter_med = int(np.median(np.asarray(fold_best_iters, dtype=np.float64))) if fold_best_iters else int(num_boost_round)
        shrink_med = float(np.median(np.asarray(fold_shrinks, dtype=np.float64))) if fold_shrinks else 0.0
        return {
            "params": dict(params),
            "val_rmse": float(avg_rmse),
            "best_iter": int(best_iter_med),
            "shrink": float(shrink_med),
        }

    max_rounds = int(n_estimators) if int(n_estimators) > 0 else 3000
    base_cand = {
        "max_depth": int(max_depth),
        "eta": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
    }

    def _clamp(v: float, lo: float, hi: float) -> float:
        return float(min(max(float(v), float(lo)), float(hi)))

    def _clamp_int(v: int, lo: int, hi: int) -> int:
        return int(min(max(int(v), int(lo)), int(hi)))

    candidates: List[Dict[str, Any]] = [
        dict(base_cand),
        {**base_cand, "min_child_weight": 5.0},
        {**base_cand, "min_child_weight": 10.0},
        {
            **base_cand,
            "max_depth": _clamp_int(int(max_depth) - 2, 2, 16),
            "eta": _clamp(float(learning_rate) * 1.5, 0.005, 0.5),
            "min_child_weight": 1.0,
        },
        {
            **base_cand,
            "max_depth": _clamp_int(int(max_depth) + 2, 2, 16),
            "eta": _clamp(float(learning_rate) * 0.6, 0.005, 0.5),
            "subsample": _clamp(float(subsample) - 0.05, 0.5, 1.0),
            "colsample_bytree": _clamp(float(colsample_bytree) - 0.05, 0.5, 1.0),
            "min_child_weight": 5.0,
        },
        {
            **base_cand,
            "eta": _clamp(float(learning_rate) * 0.5, 0.005, 0.5),
            "subsample": _clamp(float(subsample) + 0.05, 0.5, 1.0),
            "colsample_bytree": _clamp(float(colsample_bytree) + 0.05, 0.5, 1.0),
            "lambda": 2.0,
        },
        {
            **base_cand,
            "eta": _clamp(float(learning_rate) * 0.8, 0.005, 0.5),
            "alpha": 0.5,
            "lambda": 1.5,
        },
        {
            **base_cand,
            "objective": "reg:pseudohubererror",
            "huber_slope": 1.0,
        },
        {
            **base_cand,
            "objective": "reg:pseudohubererror",
            "huber_slope": 5.0,
            "min_child_weight": 5.0,
        },
        {
            **base_cand,
            "max_depth": _clamp_int(int(max_depth) + 3, 2, 16),
            "eta": _clamp(float(learning_rate) * 0.5, 0.005, 0.5),
            "min_child_weight": 10.0,
            "lambda": 2.0,
        },
    ]
    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": int(seed),
        "tree_method": "hist",
    }
    trials: List[Dict[str, Any]] = []
    for cand in candidates:
        params = {**base_params, **cand}
        trials.append(_train_one(params, num_boost_round=max_rounds))
    best = min(trials, key=lambda d: d["val_rmse"])
    best_params = dict(best["params"])
    best_rounds = int(best["best_iter"])

    hold_tr_idx = folds[0][0]
    hold_v_idx = folds[0][1]
    x_hold_tr = x_train[hold_tr_idx]
    y_hold_tr_raw = y_train[hold_tr_idx].astype(np.float64)
    x_hold_v = x_train[hold_v_idx]
    y_hold_v_raw = y_train[hold_v_idx].astype(np.float64)
    base_hold_tr = x_hold_tr[:, lag1_col].astype(np.float64)
    base_hold_v = x_hold_v[:, lag1_col].astype(np.float64)
    y_hold_tr_delta = (y_hold_tr_raw - base_hold_tr).astype(np.float64)
    y_hold_v_delta = (y_hold_v_raw - base_hold_v).astype(np.float64)
    dhold_tr = xgb.DMatrix(x_hold_tr, label=y_hold_tr_delta)
    dhold_v = xgb.DMatrix(x_hold_v, label=y_hold_v_delta)
    hold_booster = xgb.train(
        best_params,
        dhold_tr,
        num_boost_round=int(max_rounds),
        evals=[(dhold_v, "val")],
        early_stopping_rounds=150,
        verbose_eval=False,
    )
    best_rounds = int(getattr(hold_booster, "best_iteration", int(max_rounds) - 1)) + 1
    iter_range = (0, int(best_rounds))
    hold_delta_pred = hold_booster.predict(dhold_v, iteration_range=iter_range).astype(np.float64)
    final_shrink = _compute_shrink(y_hold_v_delta, hold_delta_pred)

    start = time.time()
    base_train = x_train[:, lag1_col].astype(np.float64)
    y_train_delta = (y_train.astype(np.float64) - base_train).astype(np.float64)
    dtrain_full = xgb.DMatrix(x_train, label=y_train_delta)
    final_booster = xgb.train(best_params, dtrain_full, num_boost_round=int(best_rounds), verbose_eval=False)
    fit_seconds = float(time.time() - start)

    base_test = x_test[:, lag1_col].astype(np.float64)
    y_delta_test = final_booster.predict(xgb.DMatrix(x_test), iteration_range=iter_range).astype(np.float64)
    y_pred = (base_test + float(final_shrink) * y_delta_test).astype(np.float64)
    y_pred = np.maximum(y_pred, 0.0)
    metrics = _metrics_regression(y_test, y_pred)
    baseline_metrics = _metrics_regression(y_test, base_test)

    pred_df = pd.DataFrame(
        {
            "Area": df.loc[test_mask, "Area"].to_numpy(dtype=str),
            "Item": df.loc[test_mask, "Item"].to_numpy(dtype=str),
            "Year": df.loc[test_mask, "Year"].to_numpy(dtype=int),
            "actual": y_test.astype(np.float64),
            "baseline_lag1_yield": base_test.astype(np.float64),
            "predicted": y_pred.astype(np.float64),
        }
    )

    artifacts = _default_artifacts(agriculture_dir)
    out: Dict[str, Any] = {
        "agriculture_dir": agriculture_dir,
        "dataset": {"yield_df_csv": os.path.join(agriculture_dir, "Crop Yield dataset", "yield_df.csv")},
        "seed": int(seed),
        "test_frac": float(test_frac),
        "max_rows": int(max_rows),
        "features": {"count": int(x.shape[1]), "columns": list(x.columns)},
        "n_train": int(np.sum(train_mask)),
        "n_test": int(np.sum(test_mask)),
        "model": {
            "n_estimators": int(best_rounds),
            "max_depth": int(best_params.get("max_depth", max_depth)),
            "learning_rate": float(best_params.get("eta", learning_rate)),
            "subsample": float(best_params.get("subsample", subsample)),
            "colsample_bytree": float(best_params.get("colsample_bytree", colsample_bytree)),
            "min_child_weight": float(best_params.get("min_child_weight", 1.0)),
            "reg_lambda": float(best_params.get("lambda", 1.0)),
            "reg_alpha": float(best_params.get("alpha", 0.0)),
            "y_transform": "identity",
        },
        "fit_seconds": float(fit_seconds),
        "tuning": {
            "val_frac": float(val_frac),
            "best_val_rmse": float(best["val_rmse"]),
            "trials": [{"val_rmse": float(t["val_rmse"]), "shrink": float(t.get("shrink", 0.0))} for t in trials],
        },
        "test_metrics": metrics,
        "baseline_metrics": {"lag1_yield": baseline_metrics},
        "artifacts": {
            "model_path": artifacts.yield_xgb_model_path if save_model else None,
            "metadata_path": artifacts.yield_xgb_metadata_path,
            "predictions_path": artifacts.yield_xgb_predictions_path,
        },
    }

    promote_primary = float(metrics.get("rmse", 0.0))
    promote_baseline = float(baseline_metrics.get("rmse", 0.0))
    out["promotion"] = {
        "task": "regression",
        "primary_metric": "rmse",
        "primary": promote_primary,
        "baseline_name": "lag1_yield",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary < promote_baseline * 0.99),
    }

    pred_df.to_csv(artifacts.yield_xgb_predictions_path, index=False)
    with open(artifacts.yield_xgb_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        final_booster.save_model(artifacts.yield_xgb_model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.yield_xgb_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.yield_xgb_predictions_path),
        "canary_pred_sha256": _sha256_array(y_pred[: min(256, int(y_pred.shape[0]))].astype(np.float32, copy=False)),
    }
    with open(artifacts.yield_xgb_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    base_dir = os.path.dirname(__file__)
    artifacts = _default_artifacts(base_dir)

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    cnn = sub.add_parser("cnn_crop")
    cnn.add_argument("--agriculture-dir", default=base_dir)
    cnn.add_argument("--seed", type=int, default=42)
    cnn.add_argument("--image-size", type=int, default=160)
    cnn.add_argument("--batch-size", type=int, default=32)
    cnn.add_argument("--epochs", type=int, default=8)
    cnn.add_argument("--val-frac", type=float, default=0.15)
    cnn.add_argument("--test-frac", type=float, default=0.15)
    cnn.add_argument("--max-images", type=int, default=5000)
    cnn.add_argument("--backbone", default="efficientnetb0")
    cnn.add_argument("--lr", type=float, default=3e-4)
    cnn.add_argument("--base-trainable", action="store_true")
    cnn.add_argument("--fine-tune-epochs", type=int, default=2)
    cnn.add_argument("--fine-tune-lr", type=float, default=1e-5)
    cnn.add_argument("--overfit-n", type=int, default=0)
    cnn.add_argument("--save-model", action="store_true")

    yld = sub.add_parser("xgb_yield")
    yld.add_argument("--agriculture-dir", default=base_dir)
    yld.add_argument("--seed", type=int, default=42)
    yld.add_argument("--test-frac", type=float, default=0.15)
    yld.add_argument("--max-rows", type=int, default=0)
    yld.add_argument("--n-estimators", type=int, default=900)
    yld.add_argument("--max-depth", type=int, default=8)
    yld.add_argument("--learning-rate", type=float, default=0.05)
    yld.add_argument("--subsample", type=float, default=0.9)
    yld.add_argument("--colsample-bytree", type=float, default=0.9)
    yld.add_argument("--save-model", action="store_true")

    args = p.parse_args()
    cmd = args.command or "cnn_crop"

    def _arg(name: str, default: Any) -> Any:
        return getattr(args, name, default)

    if cmd == "cnn_crop":
        result = train_crop_cnn(
            agriculture_dir=str(_arg("agriculture_dir", base_dir)),
            seed=int(_arg("seed", 42)),
            image_size=int(_arg("image_size", 160)),
            batch_size=int(_arg("batch_size", 32)),
            epochs=int(_arg("epochs", 8)),
            val_frac=float(_arg("val_frac", 0.15)),
            test_frac=float(_arg("test_frac", 0.15)),
            max_images=int(_arg("max_images", 5000)),
            backbone=str(_arg("backbone", "efficientnetb0")),
            lr=float(_arg("lr", 3e-4)),
            base_trainable=bool(_arg("base_trainable", False)),
            fine_tune_epochs=int(_arg("fine_tune_epochs", 2)),
            fine_tune_lr=float(_arg("fine_tune_lr", 1e-5)),
            overfit_n=int(_arg("overfit_n", 0)),
            save_model=bool(_arg("save_model", False)),
        )
        print(json.dumps(result["test_metrics"], indent=2))
        print(f"Saved predictions: {os.path.abspath(artifacts.crop_cnn_predictions_path)}")
        print(f"Saved metadata: {os.path.abspath(artifacts.crop_cnn_metadata_path)}")
        if bool(_arg("save_model", False)):
            print(f"Saved model: {os.path.abspath(artifacts.crop_cnn_model_path)}")
        return

    if cmd == "xgb_yield":
        result = train_xgb_yield(
            agriculture_dir=str(_arg("agriculture_dir", base_dir)),
            seed=int(_arg("seed", 42)),
            test_frac=float(_arg("test_frac", 0.15)),
            max_rows=int(_arg("max_rows", 0)),
            n_estimators=int(_arg("n_estimators", 900)),
            max_depth=int(_arg("max_depth", 8)),
            learning_rate=float(_arg("learning_rate", 0.05)),
            subsample=float(_arg("subsample", 0.9)),
            colsample_bytree=float(_arg("colsample_bytree", 0.9)),
            save_model=bool(_arg("save_model", False)),
        )
        print(json.dumps(result["test_metrics"], indent=2))
        print(f"Saved predictions: {os.path.abspath(artifacts.yield_xgb_predictions_path)}")
        print(f"Saved metadata: {os.path.abspath(artifacts.yield_xgb_metadata_path)}")
        if bool(_arg("save_model", False)):
            print(f"Saved model: {os.path.abspath(artifacts.yield_xgb_model_path)}")
        return

    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()

