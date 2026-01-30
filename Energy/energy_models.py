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
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Artifacts:
    load_lstm_model_path: str
    load_lstm_metadata_path: str
    load_lstm_predictions_path: str
    load_prophet_model_path: str
    load_prophet_metadata_path: str
    load_prophet_predictions_path: str
    faults_xgb_model_path: str
    faults_xgb_metadata_path: str
    faults_xgb_predictions_path: str


def _default_artifacts(base_dir: str) -> Artifacts:
    return Artifacts(
        load_lstm_model_path=os.path.join(base_dir, "load_lstm.keras"),
        load_lstm_metadata_path=os.path.join(base_dir, "load_lstm.meta.json"),
        load_lstm_predictions_path=os.path.join(base_dir, "load_lstm_predictions.csv"),
        load_prophet_model_path=os.path.join(base_dir, "load_prophet_style.model.pkl"),
        load_prophet_metadata_path=os.path.join(base_dir, "load_prophet_style.meta.json"),
        load_prophet_predictions_path=os.path.join(base_dir, "load_prophet_style_predictions.csv"),
        faults_xgb_model_path=os.path.join(base_dir, "faults_xgb.json"),
        faults_xgb_metadata_path=os.path.join(base_dir, "faults_xgb.meta.json"),
        faults_xgb_predictions_path=os.path.join(base_dir, "faults_xgb_predictions.csv"),
    )


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


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


def _metrics_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "pos_rate": float(np.mean(y_true.astype(np.float64))) if y_true.size else 0.0,
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            out["pr_auc"] = None
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def _split_timewise_index(n: int, *, val_frac: float, test_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    test_n = max(int(np.floor(n * float(test_frac))), 1)
    val_n = max(int(np.floor(n * float(val_frac))), 1)
    train_n = max(n - val_n - test_n, 1)
    train_end = train_n
    val_end = train_n + val_n
    idx = np.arange(n, dtype=np.int64)
    train_mask = np.zeros((n,), dtype=bool)
    val_mask = np.zeros((n,), dtype=bool)
    test_mask = np.zeros((n,), dtype=bool)
    train_mask[idx[:train_end]] = True
    val_mask[idx[train_end:val_end]] = True
    test_mask[idx[val_end:]] = True
    return train_mask, val_mask, test_mask


def _read_load_dataset(energy_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(energy_dir), "Electricity Load dataset", "continuous dataset.csv")
    df = pd.read_csv(os.path.abspath(path))
    cols = {c.lower(): c for c in df.columns}
    dt_col = cols.get("datetime")
    y_col = cols.get("nat_demand")
    if not dt_col or not y_col:
        raise KeyError(f"Expected datetime/nat_demand columns. Found: {list(df.columns)}")
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    out = out.dropna(subset=[dt_col, y_col]).reset_index(drop=True)
    out = out.sort_values(dt_col).reset_index(drop=True)
    dup_before = int(out.duplicated(subset=[dt_col]).sum())
    if dup_before:
        out = out.drop_duplicates(subset=[dt_col], keep="last").reset_index(drop=True)
    numeric_cols = [c for c in out.columns if c != dt_col]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[numeric_cols] = out[numeric_cols].interpolate(limit_direction="both")
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median(numeric_only=True))
    out = out.rename(columns={dt_col: "datetime", y_col: "target"})

    p01 = float(np.nanpercentile(out["target"].to_numpy(dtype=np.float64), 1))
    p99 = float(np.nanpercentile(out["target"].to_numpy(dtype=np.float64), 99))
    out["target"] = out["target"].clip(lower=p01, upper=p99).astype(np.float32)

    return out


def _time_features(dt: pd.Series) -> pd.DataFrame:
    hour = dt.dt.hour.astype(np.float32)
    dow = dt.dt.dayofweek.astype(np.float32)
    doy = dt.dt.dayofyear.astype(np.float32)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25).astype(np.float32),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25).astype(np.float32),
        }
    )


def _make_windows(
    *,
    features: np.ndarray,
    target: np.ndarray,
    datetimes: np.ndarray,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    datetimes = np.asarray(datetimes)
    window = int(window)
    horizon = int(horizon)
    n = int(target.shape[0])
    max_start = n - window - horizon + 1
    if max_start <= 0:
        return (
            np.zeros((0, window, int(features.shape[1])), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype="datetime64[ns]"),
        )
    xs = []
    ys = []
    dts = []
    for start in range(max_start):
        end = start + window
        t_idx = end + horizon - 1
        xs.append(features[start:end, :])
        ys.append(float(target[t_idx]))
        dts.append(datetimes[t_idx])
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32), np.asarray(dts, dtype="datetime64[ns]")


def _build_load_lstm(*, window: int, n_features: int, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(int(seed))
    inputs = tf.keras.Input(shape=(int(window), int(n_features)))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train_lstm_load(
    *,
    energy_dir: str,
    window: int,
    horizon: int,
    seed: int,
    epochs: int,
    batch_size: int,
    val_frac: float,
    test_frac: float,
    max_rows: int,
    save_model: bool,
    tune_trials: int = 0,
) -> Dict[str, Any]:
    energy_dir = os.path.abspath(energy_dir)
    df = _read_load_dataset(energy_dir)
    if int(max_rows) > 0:
        df = df.iloc[: int(max_rows)].reset_index(drop=True)

    base_cols = [c for c in df.columns if c not in {"datetime", "target"}]
    y_raw = df["target"].to_numpy(dtype=np.float32).reshape(-1)
    lag1 = np.roll(y_raw, 1)
    lag24 = np.roll(y_raw, 24)
    lag168 = np.roll(y_raw, 168)
    lag1[:1] = np.nan
    lag24[:24] = np.nan
    lag168[:168] = np.nan
    fill = float(np.nanmean(y_raw.astype(np.float64))) if np.isfinite(np.nanmean(y_raw.astype(np.float64))) else 0.0
    lag1 = np.where(np.isfinite(lag1), lag1, fill).astype(np.float32)
    lag24 = np.where(np.isfinite(lag24), lag24, lag1).astype(np.float32)
    lag168 = np.where(np.isfinite(lag168), lag168, lag24).astype(np.float32)
    roll24 = pd.Series(y_raw.astype(np.float64)).rolling(24, min_periods=1).mean().shift(1).fillna(float(lag1[0])).to_numpy(dtype=np.float32)
    roll168 = pd.Series(y_raw.astype(np.float64)).rolling(168, min_periods=1).mean().shift(1).fillna(float(lag1[0])).to_numpy(dtype=np.float32)
    lag_df = pd.DataFrame(
        {
            "target_lag1": lag1,
            "target_lag24": lag24,
            "target_lag168": lag168,
            "target_roll24": roll24,
            "target_roll168": roll168,
        }
    )

    feat_df = pd.concat([df[base_cols].astype(np.float32), lag_df, _time_features(df["datetime"])], axis=1)
    feat_cols = list(feat_df.columns)

    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(feat_df.to_numpy(dtype=np.float32)).astype(np.float32)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(df[["target"]].to_numpy(dtype=np.float32)).reshape(-1).astype(np.float32)

    x_win, y_win, dt_win = _make_windows(
        features=x_scaled,
        target=y_scaled,
        datetimes=df["datetime"].to_numpy(dtype="datetime64[ns]"),
        window=int(window),
        horizon=int(horizon),
    )
    train_mask, val_mask, test_mask = _split_timewise_index(int(x_win.shape[0]), val_frac=float(val_frac), test_frac=float(test_frac))

    x_train, y_train = x_win[train_mask], y_win[train_mask]
    x_val, y_val = x_win[val_mask], y_win[val_mask]
    x_test, y_test = x_win[test_mask], y_win[test_mask]
    dt_test = dt_win[test_mask]

    def _baseline_scaled_from_windows(y_scaled_full: np.ndarray, n_windows: int) -> np.ndarray:
        y_scaled_full = np.asarray(y_scaled_full, dtype=np.float32).reshape(-1)
        base_idx = (np.arange(int(n_windows), dtype=np.int64) + int(window) - 1).astype(np.int64)
        base_idx = np.clip(base_idx, 0, max(int(y_scaled_full.shape[0]) - 1, 0))
        return y_scaled_full[base_idx].astype(np.float32)

    base_scaled_all = _baseline_scaled_from_windows(y_scaled, int(x_win.shape[0]))
    base_train_s = base_scaled_all[train_mask]
    base_val_s = base_scaled_all[val_mask]
    base_test_s = base_scaled_all[test_mask]

    y_train_delta = (y_train - base_train_s).astype(np.float32)
    y_val_delta = (y_val - base_val_s).astype(np.float32)
    y_test_delta = (y_test - base_test_s).astype(np.float32)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
    ]

    tuning_trials: List[Dict[str, Any]] = []
    tuning_best: Optional[Dict[str, Any]] = None
    tuned_params: Optional[Dict[str, Any]] = None

    if int(tune_trials) > 0:
        rng = np.random.default_rng(int(seed))

        def _fit_one_delta(*, lr: float, units1: int, units2: int, dropout: float) -> Tuple[tf.keras.Model, Any, float, float]:
            tf.keras.utils.set_random_seed(int(seed))
            inputs = tf.keras.Input(shape=(int(window), int(x_win.shape[-1])))
            x = tf.keras.layers.LSTM(int(units1), return_sequences=True)(inputs)
            x = tf.keras.layers.Dropout(float(dropout))(x)
            x = tf.keras.layers.LSTM(int(units2))(x)
            x = tf.keras.layers.Dense(32, activation="relu")(x)
            outputs = tf.keras.layers.Dense(1)(x)
            m = tf.keras.Model(inputs=inputs, outputs=outputs)
            m.compile(optimizer=tf.keras.optimizers.Adam(float(lr)), loss="mse", metrics=["mae"])
            start = time.time()
            hist = m.fit(
                x_train,
                y_train_delta,
                validation_data=(x_val, y_val_delta),
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                callbacks=callbacks,
            )
            fit_s = float(time.time() - start)
            best_val = float(np.min(np.asarray(hist.history.get("val_loss", [np.inf]), dtype=np.float64)))
            return m, hist, fit_s, best_val

        best = None
        for _ in range(int(tune_trials)):
            lr_c = float(rng.choice([1e-3, 5e-4, 2e-4]))
            u1 = int(rng.choice([64, 96, 128]))
            u2 = int(rng.choice([32, 48, 64]))
            do = float(rng.choice([0.1, 0.2, 0.3]))
            m, hist, fit_s, best_val = _fit_one_delta(lr=lr_c, units1=u1, units2=u2, dropout=do)
            t = {"lr": lr_c, "units1": u1, "units2": u2, "dropout": do, "fit_seconds": fit_s, "best_val_loss": best_val}
            tuning_trials.append(t)
            if best is None or best_val < float(best["trial"]["best_val_loss"]):
                best = {"trial": t, "model": m, "history": hist}

        model = best["model"]
        history = best["history"]
        fit_seconds = float(best["trial"]["fit_seconds"])
        tuned_params = {k: best["trial"][k] for k in ("lr", "units1", "units2", "dropout")}
        tuning_best = best["trial"]
    else:
        model = _build_load_lstm(window=int(window), n_features=int(x_win.shape[-1]), seed=int(seed))
        start = time.time()
        history = model.fit(
            x_train,
            y_train_delta,
            validation_data=(x_val, y_val_delta),
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=2,
            callbacks=callbacks,
        )
        fit_seconds = float(time.time() - start)
        tuned_params = None

    y_delta_pred_test = model.predict(x_test, batch_size=int(batch_size), verbose=0).reshape(-1)
    y_pred_scaled = (base_test_s.astype(np.float32) + y_delta_pred_test.astype(np.float32)).reshape(-1)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    metrics = _metrics_regression(y_true, y_pred)

    base_last = scaler_y.inverse_transform(base_test_s.reshape(-1, 1)).reshape(-1).astype(np.float64)
    base_week_s = np.full_like(base_test_s, np.nan, dtype=np.float32)
    if int(window) >= 168:
        base_week_s = x_test[:, -168, 0].astype(np.float32)
    base_week = scaler_y.inverse_transform(base_week_s.reshape(-1, 1)).reshape(-1).astype(np.float64)
    base_week = np.where(np.isfinite(base_week), base_week, base_last)
    baseline_metrics = {
        "last_value": _metrics_regression(y_true, base_last),
        "seasonal_week": _metrics_regression(y_true, base_week),
    }

    t0 = time.perf_counter()
    _ = model.predict(x_test[: min(2048, int(x_test.shape[0]))], batch_size=int(batch_size), verbose=0)
    infer_s = time.perf_counter() - t0
    infer_n = int(min(2048, int(x_test.shape[0])))
    inference = {
        "samples": infer_n,
        "seconds": float(infer_s),
        "ms_per_window": float(1000.0 * infer_s / max(1, infer_n)),
        "windows_per_second": float(infer_n / max(1e-9, infer_s)),
    }

    pred_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(dt_test).astype("datetime64[ns]"),
            "actual": y_true.astype(np.float32),
            "baseline_last_value": base_last.astype(np.float32),
            "baseline_seasonal_week": base_week.astype(np.float32),
            "predicted": y_pred.astype(np.float32),
        }
    ).sort_values("datetime")

    artifacts = _default_artifacts(energy_dir)
    out: Dict[str, Any] = {
        "energy_dir": energy_dir,
        "dataset": {"load_csv": os.path.join(energy_dir, "Electricity Load dataset", "continuous dataset.csv")},
        "window": int(window),
        "horizon": int(horizon),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "max_rows": int(max_rows),
        "features": {"count": int(len(feat_cols)), "columns": feat_cols},
        "n_windows": int(x_win.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "fit_seconds": float(fit_seconds),
        "tuning": {"trials": tuning_trials, "best": tuning_best} if int(tune_trials) > 0 else None,
        "selected_hparams": tuned_params,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "test_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "inference": inference,
        "artifacts": {
            "model_path": artifacts.load_lstm_model_path if save_model else None,
            "metadata_path": artifacts.load_lstm_metadata_path,
            "predictions_path": artifacts.load_lstm_predictions_path,
        },
    }

    promote_primary = float(metrics.get("rmse", 0.0))
    base_last_rmse = float(baseline_metrics.get("last_value", {}).get("rmse") or float("inf"))
    base_week_rmse = float(baseline_metrics.get("seasonal_week", {}).get("rmse") or float("inf"))
    promote_baseline = float(min(base_last_rmse, base_week_rmse)) if np.isfinite(base_last_rmse) or np.isfinite(base_week_rmse) else 0.0
    out["promotion"] = {
        "task": "time_series_regression",
        "primary_metric": "rmse",
        "primary": promote_primary,
        "baseline_name": "best_of_last_value_and_seasonal_week",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary < promote_baseline * 0.99),
    }

    pred_df.to_csv(artifacts.load_lstm_predictions_path, index=False)
    with open(artifacts.load_lstm_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        model.save(artifacts.load_lstm_model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.load_lstm_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.load_lstm_predictions_path),
        "canary_pred_sha256": _sha256_array(y_pred[: min(256, int(y_pred.shape[0]))]),
    }
    with open(artifacts.load_lstm_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def _fourier_series(t: np.ndarray, *, period: float, order: int, prefix: str) -> pd.DataFrame:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    out: Dict[str, np.ndarray] = {}
    for k in range(1, int(order) + 1):
        out[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / float(period))
        out[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / float(period))
    return pd.DataFrame(out)


def train_prophet_style_load(
    *,
    energy_dir: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    daily_order: int,
    weekly_order: int,
    alpha: float,
    max_rows: int,
    save_model: bool,
    tune: bool = True,
) -> Dict[str, Any]:
    energy_dir = os.path.abspath(energy_dir)
    df = _read_load_dataset(energy_dir)
    if int(max_rows) > 0:
        df = df.iloc[: int(max_rows)].reset_index(drop=True)

    df = df.reset_index(drop=True)
    t = np.arange(int(len(df)), dtype=np.float64)
    t0 = float(t[0]) if t.size else 0.0
    t = t - t0

    y_series = df["target"].to_numpy(dtype=np.float64).reshape(-1)
    lag1 = np.roll(y_series, 1)
    lag24 = np.roll(y_series, 24)
    lag168 = np.roll(y_series, 168)
    lag1[:1] = np.nan
    lag24[:24] = np.nan
    lag168[:168] = np.nan
    roll24 = pd.Series(y_series).rolling(24, min_periods=1).mean().shift(1).to_numpy(dtype=np.float64)
    roll168 = pd.Series(y_series).rolling(168, min_periods=1).mean().shift(1).to_numpy(dtype=np.float64)
    fill = float(np.nanmean(y_series)) if np.isfinite(np.nanmean(y_series)) else 0.0
    lag1 = np.where(np.isfinite(lag1), lag1, fill)
    lag24 = np.where(np.isfinite(lag24), lag24, lag1)
    lag168 = np.where(np.isfinite(lag168), lag168, lag24)
    roll24 = np.where(np.isfinite(roll24), roll24, lag1)
    roll168 = np.where(np.isfinite(roll168), roll168, roll24)

    design = pd.DataFrame({"trend": t.astype(np.float64), "trend2": (t**2).astype(np.float64)})
    design = pd.concat(
        [
            design,
            _fourier_series(t, period=24.0, order=int(daily_order), prefix="day"),
            _fourier_series(t, period=24.0 * 7.0, order=int(weekly_order), prefix="week"),
            pd.DataFrame(
                {
                    "lag1": lag1.astype(np.float64),
                    "lag24": lag24.astype(np.float64),
                    "lag168": lag168.astype(np.float64),
                    "roll24_mean": roll24.astype(np.float64),
                    "roll168_mean": roll168.astype(np.float64),
                }
            ),
        ],
        axis=1,
    )
    for c in ("holiday", "school", "Holiday_ID"):
        if c in df.columns:
            design[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float64)

    y = df["target"].to_numpy(dtype=np.float64)
    train_mask, val_mask, test_mask = _split_timewise_index(int(len(df)), val_frac=float(val_frac), test_frac=float(test_frac))

    x_train = design[train_mask].to_numpy(dtype=np.float64)
    y_train = y[train_mask]
    x_val = design[val_mask].to_numpy(dtype=np.float64)
    y_val = y[val_mask]
    x_test = design[test_mask].to_numpy(dtype=np.float64)
    y_test = y[test_mask]
    dt_test = df.loc[test_mask, "datetime"].to_numpy(dtype="datetime64[ns]")

    def _fit_ridge(alpha_v: float, x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray) -> Tuple[Ridge, StandardScaler, float]:
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr)
        x_va_s = scaler.transform(x_va)
        m = Ridge(alpha=float(alpha_v), random_state=int(seed))
        m.fit(x_tr_s, y_tr)
        pred = m.predict(x_va_s).astype(np.float64)
        rmse = float(np.sqrt(mean_squared_error(y_va, pred))) if int(len(y_va)) else float("inf")
        return m, scaler, rmse

    best_cfg = {"daily_order": int(daily_order), "weekly_order": int(weekly_order), "alpha": float(alpha)}
    tune_trials: List[Dict[str, Any]] = []
    if bool(tune) and int(len(y_val)):
        for d_ord in [2, 3, 4, 5, int(daily_order)]:
            for w_ord in [2, 3, 4, 5, 6, int(weekly_order)]:
                design_try = pd.DataFrame({"trend": t.astype(np.float64), "trend2": (t**2).astype(np.float64)})
                design_try = pd.concat(
                    [
                        design_try,
                        _fourier_series(t, period=24.0, order=int(d_ord), prefix="day"),
                        _fourier_series(t, period=24.0 * 7.0, order=int(w_ord), prefix="week"),
                        pd.DataFrame(
                            {
                                "lag1": lag1.astype(np.float64),
                                "lag24": lag24.astype(np.float64),
                                "lag168": lag168.astype(np.float64),
                                "roll24_mean": roll24.astype(np.float64),
                                "roll168_mean": roll168.astype(np.float64),
                            }
                        ),
                    ],
                    axis=1,
                )
                for c in ("holiday", "school", "Holiday_ID"):
                    if c in df.columns:
                        design_try[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float64)

                x_tr = design_try[train_mask].to_numpy(dtype=np.float64)
                x_va = design_try[val_mask].to_numpy(dtype=np.float64)
                for a in [0.1, 1.0, 10.0, float(alpha)]:
                    m, sc, rmse = _fit_ridge(float(a), x_tr, y_train, x_va, y_val)
                    tune_trials.append({"daily_order": int(d_ord), "weekly_order": int(w_ord), "alpha": float(a), "val_rmse": float(rmse)})
                    if float(rmse) < float(best_cfg.get("val_rmse", float("inf"))):
                        best_cfg = {"daily_order": int(d_ord), "weekly_order": int(w_ord), "alpha": float(a), "val_rmse": float(rmse)}

        design = pd.DataFrame({"trend": t.astype(np.float64), "trend2": (t**2).astype(np.float64)})
        design = pd.concat(
            [
                design,
                _fourier_series(t, period=24.0, order=int(best_cfg["daily_order"]), prefix="day"),
                _fourier_series(t, period=24.0 * 7.0, order=int(best_cfg["weekly_order"]), prefix="week"),
                pd.DataFrame(
                    {
                        "lag1": lag1.astype(np.float64),
                        "lag24": lag24.astype(np.float64),
                        "lag168": lag168.astype(np.float64),
                        "roll24_mean": roll24.astype(np.float64),
                        "roll168_mean": roll168.astype(np.float64),
                    }
                ),
            ],
            axis=1,
        )
        for c in ("holiday", "school", "Holiday_ID"):
            if c in df.columns:
                design[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float64)
        x_train = design[train_mask | val_mask].to_numpy(dtype=np.float64)
        y_train = y[train_mask | val_mask]
        x_test = design[test_mask].to_numpy(dtype=np.float64)
    else:
        best_cfg = {"daily_order": int(daily_order), "weekly_order": int(weekly_order), "alpha": float(alpha)}

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = Ridge(alpha=float(best_cfg["alpha"]), random_state=int(seed))
    start = time.time()
    model.fit(x_train_s, y_train)
    fit_seconds = float(time.time() - start)

    y_pred = model.predict(x_test_s).astype(np.float64)
    metrics = _metrics_regression(y_test, y_pred)

    dt_to_target = dict(zip(pd.to_datetime(df["datetime"]).astype("datetime64[ns]").to_numpy(), df["target"].to_numpy(dtype=np.float64)))
    dt_test_pd = pd.to_datetime(dt_test).astype("datetime64[ns]")
    base_last = np.asarray([dt_to_target.get((pd.Timestamp(d) - pd.Timedelta(hours=1)).to_datetime64(), np.nan) for d in dt_test_pd], dtype=np.float64)
    base_week = np.asarray([dt_to_target.get((pd.Timestamp(d) - pd.Timedelta(days=7)).to_datetime64(), np.nan) for d in dt_test_pd], dtype=np.float64)
    base_last = np.where(np.isfinite(base_last), base_last, np.nanmean(y_test))
    base_week = np.where(np.isfinite(base_week), base_week, base_last)
    baseline_metrics = {
        "last_value": _metrics_regression(y_test, base_last),
        "seasonal_week": _metrics_regression(y_test, base_week),
    }

    pred_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(dt_test).astype("datetime64[ns]"),
            "actual": y_test.astype(np.float32),
            "baseline_last_value": base_last.astype(np.float32),
            "baseline_seasonal_week": base_week.astype(np.float32),
            "predicted": y_pred.astype(np.float32),
        }
    ).sort_values("datetime")

    artifacts = _default_artifacts(energy_dir)
    out: Dict[str, Any] = {
        "energy_dir": energy_dir,
        "dataset": {"load_csv": os.path.join(energy_dir, "Electricity Load dataset", "continuous dataset.csv")},
        "model": {
            "type": "prophet_style_additive_regression",
            "daily_fourier_order": int(best_cfg["daily_order"]),
            "weekly_fourier_order": int(best_cfg["weekly_order"]),
            "ridge_alpha": float(best_cfg["alpha"]),
        },
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "seed": int(seed),
        "max_rows": int(max_rows),
        "features": {"count": int(design.shape[1]), "columns": list(design.columns)},
        "fit_seconds": float(fit_seconds),
        "test_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "tuning": {"trials": tune_trials, "best": best_cfg} if bool(tune) else None,
        "artifacts": {
            "model_path": artifacts.load_prophet_model_path if save_model else None,
            "metadata_path": artifacts.load_prophet_metadata_path,
            "predictions_path": artifacts.load_prophet_predictions_path,
        },
    }

    promote_primary = float(metrics.get("rmse", 0.0))
    base_last_rmse = float(baseline_metrics.get("last_value", {}).get("rmse") or float("inf"))
    base_week_rmse = float(baseline_metrics.get("seasonal_week", {}).get("rmse") or float("inf"))
    promote_baseline = float(min(base_last_rmse, base_week_rmse)) if np.isfinite(base_last_rmse) or np.isfinite(base_week_rmse) else 0.0
    out["promotion"] = {
        "task": "time_series_regression",
        "primary_metric": "rmse",
        "primary": promote_primary,
        "baseline_name": "best_of_last_value_and_seasonal_week",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary < promote_baseline * 0.99),
    }

    pred_df.to_csv(artifacts.load_prophet_predictions_path, index=False)
    with open(artifacts.load_prophet_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        import joblib

        joblib.dump({"model": model, "scaler": scaler, "columns": list(design.columns)}, artifacts.load_prophet_model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.load_prophet_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.load_prophet_predictions_path),
        "canary_pred_sha256": _sha256_array(y_pred[: min(256, int(y_pred.shape[0]))]),
    }
    with open(artifacts.load_prophet_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def _read_wind_turbine(energy_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(energy_dir), "Wind Turbine SCADA dataset", "T1.csv")
    df = pd.read_csv(os.path.abspath(path))
    cols = {c.lower(): c for c in df.columns}
    dt_col = cols.get("date/time")
    if not dt_col:
        raise KeyError(f"Could not find Date/Time column. Found: {list(df.columns)}")
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], format="%d %m %Y %H:%M", errors="coerce")
    out = out.dropna(subset=[dt_col]).reset_index(drop=True)
    out = out.sort_values(dt_col).reset_index(drop=True)
    out = out.rename(columns={dt_col: "datetime"})
    for c in out.columns:
        if c == "datetime":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    num_cols = [c for c in out.columns if c != "datetime"]
    out[num_cols] = out[num_cols].interpolate(limit_direction="both")
    out[num_cols] = out[num_cols].fillna(out[num_cols].median(numeric_only=True))
    return out


def _make_fault_label(
    df: pd.DataFrame,
    *,
    strategy: str,
    rolling_window: int,
    quantile: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ap = df["LV ActivePower (kW)"].to_numpy(dtype=np.float64)
    tp = df["Theoretical_Power_Curve (KWh)"].to_numpy(dtype=np.float64)
    ws = df["Wind Speed (m/s)"].to_numpy(dtype=np.float64)
    valid = (tp > 50.0) & (ws > 3.0) & (ws < 25.0)
    residual = (tp - ap).astype(np.float64)

    strategy = str(strategy).lower().strip()
    q = float(quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("--label-quantile must be in (0,1)")

    residual_valid = residual[valid]
    global_thresh = float(np.nanpercentile(residual_valid, 95)) if residual_valid.size else 300.0

    if strategy == "static_p95":
        thresh_arr = np.full_like(residual, fill_value=float(global_thresh), dtype=np.float64)
        label_rule = "fault_if(tp-ap > p95_residual_valid) and tp>50 and 3<ws<25"
        meta_thresh: Any = float(global_thresh)
    elif strategy == "rolling_quantile":
        w = int(rolling_window)
        if w <= 10:
            raise ValueError("--label-window must be > 10")
        s = pd.Series(residual)
        rolling_q = s.rolling(window=w, min_periods=max(10, int(w // 4))).quantile(q)
        rolling_q = rolling_q.shift(1)
        thresh_arr = rolling_q.to_numpy(dtype=np.float64)
        thresh_arr = np.where(np.isfinite(thresh_arr), thresh_arr, float(global_thresh))
        label_rule = f"fault_if(tp-ap > rolling_quantile(residual,{q})_window={w}) and tp>50 and 3<ws<25"
        meta_thresh = {"global_p95": float(global_thresh), "rolling_window": int(w), "quantile": float(q)}
    else:
        raise ValueError("--label-strategy must be one of: static_p95 | rolling_quantile")

    y = ((residual > thresh_arr) & valid).astype(np.int32)
    meta = {"label_rule": label_rule, "threshold": meta_thresh, "pos_rate": float(np.mean(y))}
    return y, meta


def _pick_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    policy: str,
    min_precision: float,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    proba = np.asarray(proba, dtype=np.float64).reshape(-1)
    policy = str(policy).lower().strip()

    if y_true.size <= 0:
        return {"policy": policy, "threshold": 0.5, "precision": None, "recall": None, "f1": None}

    candidates = np.unique(np.quantile(proba, np.linspace(0.05, 0.95, 19))).tolist()
    candidates = sorted({float(x) for x in candidates} | {0.5})
    best = None
    for thr in candidates:
        y_pred = (proba >= float(thr)).astype(np.int32)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1v = float(f1_score(y_true, y_pred, zero_division=0))

        if policy == "best_f1":
            key = (f1v, prec, rec)
        elif policy == "best_recall_at_precision":
            if prec + 1e-12 < float(min_precision):
                continue
            key = (rec, prec, f1v)
        else:
            raise ValueError("--threshold-policy must be one of: best_f1 | best_recall_at_precision")

        cand = {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1v, "key": key}
        if best is None or cand["key"] > best["key"]:
            best = cand

    if best is None:
        thr = float(np.quantile(proba, 0.99))
        y_pred = (proba >= thr).astype(np.int32)
        return {
            "policy": policy,
            "threshold": thr,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "note": "No threshold met min_precision; used p99 proba.",
        }

    best.pop("key", None)
    best["policy"] = policy
    best["min_precision"] = float(min_precision) if policy == "best_recall_at_precision" else None
    return best


def train_xgb_faults(
    *,
    energy_dir: str,
    seed: int,
    val_frac: float,
    test_frac: float,
    max_rows: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    save_model: bool,
    threshold_policy: str,
    min_precision: float,
    label_strategy: str,
    label_window: int,
    label_quantile: float,
    robust_folds: int,
    min_test_positives: int,
) -> Dict[str, Any]:
    energy_dir = os.path.abspath(energy_dir)
    df = _read_wind_turbine(energy_dir)
    if int(max_rows) > 0:
        df = df.iloc[: int(max_rows)].reset_index(drop=True)

    y, label_meta = _make_fault_label(df, strategy=str(label_strategy), rolling_window=int(label_window), quantile=float(label_quantile))
    dt = df["datetime"].to_numpy(dtype="datetime64[ns]")
    ws = df["Wind Speed (m/s)"].to_numpy(dtype=np.float64)
    wd = df["Wind Direction (°)"].to_numpy(dtype=np.float64)
    wd_rad = np.deg2rad(wd)
    features = pd.DataFrame(
        {
            "wind_speed": ws.astype(np.float32),
            "wind_dir_sin": np.sin(wd_rad).astype(np.float32),
            "wind_dir_cos": np.cos(wd_rad).astype(np.float32),
            "theoretical_power": df["Theoretical_Power_Curve (KWh)"].to_numpy(dtype=np.float64).astype(np.float32),
            "active_power": df["LV ActivePower (kW)"].to_numpy(dtype=np.float64).astype(np.float32),
            "residual": (df["Theoretical_Power_Curve (KWh)"].to_numpy(dtype=np.float64) - df["LV ActivePower (kW)"].to_numpy(dtype=np.float64)).astype(np.float32),
        }
    )
    denom = np.maximum(np.abs(features["theoretical_power"].to_numpy(dtype=np.float32)), 1e-6).astype(np.float32)
    features["residual_frac"] = (features["residual"].to_numpy(dtype=np.float32) / denom).astype(np.float32)
    features = pd.concat([features, _time_features(pd.to_datetime(df["datetime"]))], axis=1)
    feat_cols = list(features.columns)

    train_mask, val_mask, test_mask = _split_timewise_index(int(len(df)), val_frac=float(val_frac), test_frac=float(test_frac))
    x_train = features.loc[train_mask].to_numpy(dtype=np.float32)
    y_train = y[train_mask]
    x_val = features.loc[val_mask].to_numpy(dtype=np.float32)
    y_val = y[val_mask]
    x_test = features.loc[test_mask].to_numpy(dtype=np.float32)
    y_test = y[test_mask]
    dt_test = dt[test_mask]

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = float(neg / max(1.0, pos))

    booster = xgb.XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    start = time.time()
    fit_kwargs: Dict[str, Any] = {}
    if int(x_val.shape[0]) > 0 and int(np.unique(y_val).size) >= 1:
        fit_kwargs = {"eval_set": [(x_val, y_val)], "verbose": False}
    booster.fit(x_train, y_train, **fit_kwargs)
    fit_seconds = float(time.time() - start)

    val_proba = booster.predict_proba(x_val)[:, 1].astype(np.float64) if int(x_val.shape[0]) > 0 else np.asarray([], dtype=np.float64)
    threshold_info = _pick_threshold(y_val, val_proba, policy=str(threshold_policy), min_precision=float(min_precision)) if val_proba.size else {
        "policy": str(threshold_policy),
        "threshold": 0.5,
        "precision": None,
        "recall": None,
        "f1": None,
        "note": "No validation slice available; used 0.5",
    }

    proba = booster.predict_proba(x_test)[:, 1].astype(np.float64)
    y_pred = (proba >= float(threshold_info["threshold"])).astype(np.int32)
    metrics = _metrics_classification(y_test, y_pred, proba)
    metrics["threshold"] = float(threshold_info["threshold"])

    pred_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(dt_test).astype("datetime64[ns]"),
            "y_true": y_test.astype(int),
            "y_pred": y_pred.astype(int),
            "p_fault": proba.astype(np.float32),
        }
    ).sort_values("datetime")

    artifacts = _default_artifacts(energy_dir)
    out: Dict[str, Any] = {
        "energy_dir": energy_dir,
        "dataset": {"wind_turbine_csv": os.path.join(energy_dir, "Wind Turbine SCADA dataset", "T1.csv")},
        "labeling": label_meta,
        "seed": int(seed),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "max_rows": int(max_rows),
        "features": {"count": int(len(feat_cols)), "columns": feat_cols},
        "model": {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "scale_pos_weight": float(scale_pos_weight),
        },
        "fit_seconds": float(fit_seconds),
        "test_metrics": metrics,
        "threshold": threshold_info,
        "baseline_metrics": {"always_negative": _metrics_classification(y_test, np.zeros_like(y_test, dtype=np.int32), np.zeros_like(proba, dtype=np.float64))},
        "artifacts": {
            "model_path": artifacts.faults_xgb_model_path if save_model else None,
            "metadata_path": artifacts.faults_xgb_metadata_path,
            "predictions_path": artifacts.faults_xgb_predictions_path,
        },
    }

    pred_df.to_csv(artifacts.faults_xgb_predictions_path, index=False)
    with open(artifacts.faults_xgb_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        booster.save_model(artifacts.faults_xgb_model_path)

    folds_n = max(0, int(robust_folds))
    test_n = max(int(np.floor(int(len(df)) * float(test_frac))), 1)
    val_n = max(int(np.floor(int(len(df)) * float(val_frac))), 1)
    fold_reports: List[Dict[str, Any]] = []
    if folds_n > 1:
        for i in range(int(folds_n)):
            test_start = int(len(df)) - (int(folds_n) - i) * int(test_n)
            test_end = min(int(test_start + test_n), int(len(df)))
            val_end = int(test_start)
            val_start = max(0, int(val_end - val_n))
            train_end = int(val_start)
            if test_start <= 0 or train_end <= 0 or test_end <= test_start:
                continue

            x_tr = features.iloc[:train_end].to_numpy(dtype=np.float32)
            y_tr = y[:train_end]
            x_va = features.iloc[val_start:val_end].to_numpy(dtype=np.float32)
            y_va = y[val_start:val_end]
            x_te = features.iloc[test_start:test_end].to_numpy(dtype=np.float32)
            y_te = y[test_start:test_end]

            pos = float(np.sum(y_tr == 1))
            neg = float(np.sum(y_tr == 0))
            spw = float(neg / max(1.0, pos))
            fold_model = xgb.XGBClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                subsample=float(subsample),
                colsample_bytree=float(colsample_bytree),
                reg_lambda=1.0,
                random_state=int(seed) + int(i),
                n_jobs=-1,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=spw,
            )

            fold_fit_kwargs: Dict[str, Any] = {}
            if int(x_va.shape[0]) > 0:
                fold_fit_kwargs = {"eval_set": [(x_va, y_va)], "verbose": False}
            fold_model.fit(x_tr, y_tr, **fold_fit_kwargs)

            va_proba = fold_model.predict_proba(x_va)[:, 1].astype(np.float64) if int(x_va.shape[0]) > 0 else np.asarray([], dtype=np.float64)
            fold_thr = _pick_threshold(y_va, va_proba, policy=str(threshold_policy), min_precision=float(min_precision)) if va_proba.size else {"threshold": 0.5}
            te_proba = fold_model.predict_proba(x_te)[:, 1].astype(np.float64)
            te_pred = (te_proba >= float(fold_thr.get("threshold", 0.5))).astype(np.int32)
            fold_metrics = _metrics_classification(y_te, te_pred, te_proba)
            fold_metrics["threshold"] = float(fold_thr.get("threshold", 0.5))
            fold_reports.append(
                {
                    "fold": int(i),
                    "train_end": int(train_end),
                    "val_range": [int(val_start), int(val_end)],
                    "test_range": [int(test_start), int(test_end)],
                    "test_pos": int(np.sum(y_te == 1)),
                    "test_size": int(y_te.size),
                    "test_metrics": fold_metrics,
                }
            )

    out["robustness"] = {"folds": fold_reports, "min_test_positives": int(min_test_positives)}

    pr_values = [float(fr["test_metrics"].get("pr_auc") or 0.0) for fr in fold_reports] if fold_reports else [float(metrics.get("pr_auc") or 0.0)]
    pos_values = [int(fr.get("test_pos", 0)) for fr in fold_reports] if fold_reports else [int(np.sum(y_test == 1))]
    baseline_pr_values = [float(fr["test_metrics"].get("pos_rate") or 0.0) for fr in fold_reports] if fold_reports else [float(metrics.get("pos_rate") or 0.0)]
    pr_mean = float(np.mean(pr_values)) if pr_values else 0.0
    base_pr_mean = float(np.mean(baseline_pr_values)) if baseline_pr_values else 0.0
    min_pos = int(np.min(pos_values)) if pos_values else 0

    out["promotion"] = {
        "task": "binary_classification",
        "primary_metric": "pr_auc_mean",
        "primary": pr_mean,
        "baseline_name": "pos_rate_mean",
        "baseline": base_pr_mean,
        "constraints": {"min_test_positives": int(min_test_positives)},
        "eligible": bool((min_pos >= int(min_test_positives)) and (pr_mean > base_pr_mean + 0.02)),
    }

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.faults_xgb_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.faults_xgb_predictions_path),
        "canary_proba_sha256": _sha256_array(proba[: min(int(proba.shape[0]), 256)]),
    }
    with open(artifacts.faults_xgb_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def main() -> None:
    base_dir = os.path.dirname(__file__)
    artifacts = _default_artifacts(base_dir)

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    lstm = sub.add_parser("lstm_load")
    lstm.add_argument("--energy-dir", default=base_dir)
    lstm.add_argument("--window", type=int, default=48)
    lstm.add_argument("--horizon", type=int, default=1)
    lstm.add_argument("--seed", type=int, default=42)
    lstm.add_argument("--epochs", type=int, default=15)
    lstm.add_argument("--batch-size", type=int, default=512)
    lstm.add_argument("--val-frac", type=float, default=0.15)
    lstm.add_argument("--test-frac", type=float, default=0.15)
    lstm.add_argument("--max-rows", type=int, default=0)
    lstm.add_argument("--tune-trials", type=int, default=0)
    lstm.add_argument("--save-model", action="store_true")

    prophet = sub.add_parser("prophet_load")
    prophet.add_argument("--energy-dir", default=base_dir)
    prophet.add_argument("--val-frac", type=float, default=0.15)
    prophet.add_argument("--test-frac", type=float, default=0.15)
    prophet.add_argument("--seed", type=int, default=42)
    prophet.add_argument("--daily-order", type=int, default=8)
    prophet.add_argument("--weekly-order", type=int, default=6)
    prophet.add_argument("--alpha", type=float, default=1.0)
    prophet.add_argument("--max-rows", type=int, default=0)
    prophet.add_argument("--no-tune", action="store_true")
    prophet.add_argument("--save-model", action="store_true")

    faults = sub.add_parser("xgb_faults")
    faults.add_argument("--energy-dir", default=base_dir)
    faults.add_argument("--seed", type=int, default=42)
    faults.add_argument("--val-frac", type=float, default=0.15)
    faults.add_argument("--test-frac", type=float, default=0.15)
    faults.add_argument("--max-rows", type=int, default=0)
    faults.add_argument("--n-estimators", type=int, default=600)
    faults.add_argument("--max-depth", type=int, default=6)
    faults.add_argument("--learning-rate", type=float, default=0.05)
    faults.add_argument("--subsample", type=float, default=0.9)
    faults.add_argument("--colsample-bytree", type=float, default=0.9)
    faults.add_argument("--save-model", action="store_true")
    faults.add_argument("--threshold-policy", default="best_recall_at_precision")
    faults.add_argument("--min-precision", type=float, default=0.9)
    faults.add_argument("--label-strategy", default="rolling_quantile")
    faults.add_argument("--label-window", type=int, default=1440)
    faults.add_argument("--label-quantile", type=float, default=0.995)
    faults.add_argument("--robust-eval-folds", type=int, default=3)
    faults.add_argument("--min-test-positives", type=int, default=50)

    args = p.parse_args()
    cmd = args.command or "lstm_load"

    def _arg(name: str, default: Any) -> Any:
        return getattr(args, name, default)

    if cmd == "lstm_load":
        result = train_lstm_load(
            energy_dir=str(_arg("energy_dir", base_dir)),
            window=int(_arg("window", 48)),
            horizon=int(_arg("horizon", 1)),
            seed=int(_arg("seed", 42)),
            epochs=int(_arg("epochs", 15)),
            batch_size=int(_arg("batch_size", 512)),
            val_frac=float(_arg("val_frac", 0.15)),
            test_frac=float(_arg("test_frac", 0.15)),
            max_rows=int(_arg("max_rows", 0)),
            tune_trials=int(_arg("tune_trials", 0)),
            save_model=bool(_arg("save_model", False)),
        )
        print(json.dumps(result["test_metrics"], indent=2))
        print(f"Saved predictions: {os.path.abspath(artifacts.load_lstm_predictions_path)}")
        print(f"Saved metadata: {os.path.abspath(artifacts.load_lstm_metadata_path)}")
        if bool(_arg("save_model", False)):
            print(f"Saved model: {os.path.abspath(artifacts.load_lstm_model_path)}")
        return

    if cmd == "prophet_load":
        result = train_prophet_style_load(
            energy_dir=str(_arg("energy_dir", base_dir)),
            val_frac=float(_arg("val_frac", 0.15)),
            test_frac=float(_arg("test_frac", 0.15)),
            seed=int(_arg("seed", 42)),
            daily_order=int(_arg("daily_order", 8)),
            weekly_order=int(_arg("weekly_order", 6)),
            alpha=float(_arg("alpha", 1.0)),
            max_rows=int(_arg("max_rows", 0)),
            tune=not bool(_arg("no_tune", False)),
            save_model=bool(_arg("save_model", False)),
        )
        print(json.dumps(result["test_metrics"], indent=2))
        print(f"Saved predictions: {os.path.abspath(artifacts.load_prophet_predictions_path)}")
        print(f"Saved metadata: {os.path.abspath(artifacts.load_prophet_metadata_path)}")
        if bool(_arg("save_model", False)):
            print(f"Saved model: {os.path.abspath(artifacts.load_prophet_model_path)}")
        return

    if cmd == "xgb_faults":
        result = train_xgb_faults(
            energy_dir=str(_arg("energy_dir", base_dir)),
            seed=int(_arg("seed", 42)),
            val_frac=float(_arg("val_frac", 0.15)),
            test_frac=float(_arg("test_frac", 0.15)),
            max_rows=int(_arg("max_rows", 0)),
            n_estimators=int(_arg("n_estimators", 600)),
            max_depth=int(_arg("max_depth", 6)),
            learning_rate=float(_arg("learning_rate", 0.05)),
            subsample=float(_arg("subsample", 0.9)),
            colsample_bytree=float(_arg("colsample_bytree", 0.9)),
            save_model=bool(_arg("save_model", False)),
            threshold_policy=str(_arg("threshold_policy", "best_recall_at_precision")),
            min_precision=float(_arg("min_precision", 0.9)),
            label_strategy=str(_arg("label_strategy", "rolling_quantile")),
            label_window=int(_arg("label_window", 1440)),
            label_quantile=float(_arg("label_quantile", 0.995)),
            robust_folds=int(_arg("robust_eval_folds", 3)),
            min_test_positives=int(_arg("min_test_positives", 50)),
        )
        print(json.dumps(result["test_metrics"], indent=2))
        print(f"Saved predictions: {os.path.abspath(artifacts.faults_xgb_predictions_path)}")
        print(f"Saved metadata: {os.path.abspath(artifacts.faults_xgb_metadata_path)}")
        if bool(_arg("save_model", False)):
            print(f"Saved model: {os.path.abspath(artifacts.faults_xgb_model_path)}")
        return

    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()

