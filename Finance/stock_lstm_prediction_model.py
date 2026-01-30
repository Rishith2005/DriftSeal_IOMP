import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class Artifacts:
    model_path: str
    metadata_path: str


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    import hashlib

    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _default_artifacts() -> Artifacts:
    base_dir = os.path.dirname(__file__)
    return Artifacts(
        model_path=os.path.join(base_dir, "stock_lstm_model.keras"),
        metadata_path=os.path.join(base_dir, "stock_lstm_model.meta.json"),
    )


def _resolve_target_col(df_cols: Sequence[str], target_col: Optional[str]) -> str:
    cols = set(map(str, df_cols))
    if target_col and str(target_col) in cols:
        return str(target_col)
    for cand in ("adjclose", "close", "Adj Close", "target", "Target"):
        if cand in cols:
            return cand
    raise ValueError("No usable target column found. Try --target-col explicitly.")


def _choose_usecols(df_cols: Sequence[str], *, date_col: str, ticker_col: Optional[str], target_col: str) -> List[str]:
    cols = list(map(str, df_cols))
    base = [date_col]
    if ticker_col and ticker_col in cols:
        base.append(ticker_col)
    if target_col not in cols:
        raise KeyError(f"Target column not found in CSV header: {target_col}")

    desired = [
        "open",
        "high",
        "low",
        "close",
        "adjclose",
        "volume",
        "feargreed",
        "RSIadjclose15",
        "MACDhistadjclose15",
        "emaadjclose5",
        "emaadjclose10",
        "emaadjclose15",
        "smaadjclose5",
        "smaadjclose10",
        "smaadjclose15",
        "laglow1",
        "laghigh1",
        "lagvolume1",
        "laglow2",
        "laghigh2",
        "lagvolume2",
        "laglow5",
        "laghigh5",
        "lagvolume5",
    ]

    usecols = []
    for c in base + desired:
        if c in cols and c not in usecols:
            usecols.append(c)

    if target_col not in usecols:
        usecols.append(target_col)

    return usecols


def _coerce_numeric_features(df: pd.DataFrame, *, exclude: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            out[str(col)] = s.astype(np.float32, copy=False)
        else:
            out[str(col)] = pd.to_numeric(s, errors="coerce").astype(np.float32)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().fillna(0.0)
    return out


def _split_sizes(n: int, *, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    train_n = int(np.floor(n * float(train_frac)))
    val_n = int(np.floor(n * float(val_frac)))
    train_n = max(train_n, 0)
    val_n = max(val_n, 0)
    test_n = max(n - train_n - val_n, 0)
    if test_n <= 0 and n >= 3:
        test_n = 1
        if val_n > 0:
            val_n -= 1
        else:
            train_n = max(train_n - 1, 1)
    return train_n, val_n, test_n


def _build_windows(x: np.ndarray, y: np.ndarray, *, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    window = int(window)
    if n <= window:
        return np.empty((0, window, x.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    xs = []
    ys = []
    for end in range(window - 1, n):
        start = end - window + 1
        xs.append(x[start : end + 1])
        ys.append(y[end])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _build_windows_meta(x: np.ndarray, y: np.ndarray, meta: np.ndarray, *, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    window = int(window)
    if n <= window:
        return (
            np.empty((0, window, x.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=meta.dtype),
        )
    xs = []
    ys = []
    ms = []
    for end in range(window - 1, n):
        start = end - window + 1
        xs.append(x[start : end + 1])
        ys.append(y[end])
        ms.append(meta[end])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(ms)


def _build_model(*, window: int, n_features: int, seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(int(seed))
    inputs = tf.keras.layers.Input(shape=(int(window), int(n_features)))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True))(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def _add_engineered_features(features: pd.DataFrame, *, target_col: str, ticker: pd.Series) -> pd.DataFrame:
    out = features.copy()
    px = pd.to_numeric(out[target_col], errors="coerce")
    grp = ticker.astype("string")

    ret = (px / px.groupby(grp, sort=False).shift(1)).replace([np.inf, -np.inf], np.nan) - 1.0
    out["__ret_1"] = ret.astype(np.float32)
    out["__ret_5"] = (px / px.groupby(grp, sort=False).shift(5)).replace([np.inf, -np.inf], np.nan) - 1.0
    out["__ret_10"] = (px / px.groupby(grp, sort=False).shift(10)).replace([np.inf, -np.inf], np.nan) - 1.0

    log_px = np.log(np.maximum(px, 1e-8))
    roll5 = log_px.groupby(grp, sort=False).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    roll10 = log_px.groupby(grp, sort=False).rolling(10, min_periods=5).mean().reset_index(level=0, drop=True)
    out["__mom_5"] = (log_px - roll5).astype(np.float32)
    out["__mom_10"] = (log_px - roll10).astype(np.float32)

    vol5 = ret.groupby(grp, sort=False).rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    vol10 = ret.groupby(grp, sort=False).rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
    out["__vol_5"] = vol5.astype(np.float32)
    out["__vol_10"] = vol10.astype(np.float32)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.ffill().fillna(0.0)
    return out


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if y_true.size else None
    mae = float(mean_absolute_error(y_true, y_pred)) if y_true.size else None
    mape = None
    if y_true.size:
        denom = np.maximum(np.abs(y_true), 1e-8)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)))
    r2 = None
    if y_true.size >= 2:
        r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def train_and_test(
    *,
    csv_path: str,
    date_col: str,
    ticker_col: Optional[str],
    ticker_value: Optional[str],
    target_col: Optional[str],
    target_mode: str,
    horizon: int,
    window: int,
    train_frac: float,
    val_frac: float,
    seed: int,
    epochs: int,
    batch_size: int,
    predictions_csv: Optional[str],
    n_show: int,
    model_path: str,
    metadata_path: str,
    save_model: bool,
    x_scaler_kind: str,
) -> Dict[str, Any]:
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    resolved_target = _resolve_target_col(header.columns, target_col)
    usecols = _choose_usecols(header.columns, date_col=str(date_col), ticker_col=ticker_col, target_col=resolved_target)

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    if str(date_col) not in df.columns:
        raise KeyError(f"Date column not found: {date_col}")

    df[str(date_col)] = pd.to_datetime(df[str(date_col)], errors="coerce")
    df = df.dropna(subset=[str(date_col)])

    if ticker_col and str(ticker_col) in df.columns:
        df[str(ticker_col)] = df[str(ticker_col)].astype("string")
        if ticker_value is not None:
            df = df[df[str(ticker_col)] == str(ticker_value)]
        group_keys = list(df[str(ticker_col)].dropna().unique())
    else:
        ticker_col = None
        group_keys = ["__single__"]
        df["__single__"] = "__single__"
        ticker_col = "__single__"

    df = df.sort_values([str(ticker_col), str(date_col)], kind="mergesort").reset_index(drop=True)

    horizon = int(horizon)
    window = int(window)
    if horizon <= 0:
        raise ValueError("--horizon must be >= 1")
    if window <= 1:
        raise ValueError("--window must be >= 2")

    feature_exclude = [str(date_col), str(ticker_col)]
    features = _coerce_numeric_features(df, exclude=feature_exclude)

    if resolved_target not in features.columns:
        raise KeyError(f"Target column was not loaded as numeric: {resolved_target}")

    mode = str(target_mode).lower().strip()
    if mode not in {"price", "log_return"}:
        raise ValueError("--target-mode must be one of: price | log_return")

    px = pd.to_numeric(features[resolved_target], errors="coerce")
    px_prev = px.groupby(df[str(ticker_col)], sort=False).shift(1)
    log_ret = np.log(np.maximum(px, 1e-8) / np.maximum(px_prev, 1e-8))
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
    features = _add_engineered_features(features, target_col=resolved_target, ticker=df[str(ticker_col)])
    features["__log_return"] = log_ret.astype(np.float32)

    if mode == "price":
        y_next = px.groupby(df[str(ticker_col)], sort=False).shift(-horizon)
    else:
        px_fut = px.groupby(df[str(ticker_col)], sort=False).shift(-horizon)
        y_next = np.log(np.maximum(px_fut, 1e-8) / np.maximum(px, 1e-8))
        y_next = y_next.replace([np.inf, -np.inf], np.nan)

    valid_mask = y_next.notna()
    features = features.loc[valid_mask].reset_index(drop=True)
    y_next = y_next.loc[valid_mask].reset_index(drop=True).astype(np.float32)

    df_keys = df.loc[valid_mask, [str(ticker_col), str(date_col)]].reset_index(drop=True)

    train_rows = []
    for key in group_keys:
        gmask = df_keys[str(ticker_col)] == str(key)
        gcount = int(gmask.sum())
        train_n, _, _ = _split_sizes(gcount, train_frac=float(train_frac), val_frac=float(val_frac))
        if train_n > 0:
            train_rows.append(features.loc[gmask].iloc[:train_n])

    if not train_rows:
        raise ValueError("Not enough data to create a training split.")

    scaler_kind = str(x_scaler_kind).lower().strip()
    if scaler_kind not in {"minmax", "standard"}:
        raise ValueError("--x-scaler must be one of: minmax | standard")
    x_scaler = MinMaxScaler() if scaler_kind == "minmax" else StandardScaler()
    x_scaler.fit(pd.concat(train_rows, axis=0).to_numpy(dtype=np.float32, copy=False))

    y_scaler = MinMaxScaler() if scaler_kind == "minmax" else StandardScaler()
    train_y_parts = []
    for key in group_keys:
        gmask = df_keys[str(ticker_col)] == str(key)
        gcount = int(gmask.sum())
        train_n, _, _ = _split_sizes(gcount, train_frac=float(train_frac), val_frac=float(val_frac))
        if train_n > 0:
            train_y_parts.append(y_next.loc[gmask].iloc[:train_n])
    y_scaler.fit(pd.concat(train_y_parts, axis=0).to_numpy(dtype=np.float32, copy=False).reshape(-1, 1))

    x_scaled = x_scaler.transform(features.to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    y_scaled = y_scaler.transform(y_next.to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)).astype(np.float32, copy=False).reshape(-1)

    x_train_parts = []
    y_train_parts = []
    x_val_parts = []
    y_val_parts = []
    x_test_parts = []
    y_test_parts = []
    last_observed_parts = []
    last_return_parts = []
    test_dates_parts = []
    test_tickers_parts = []

    feature_names = list(features.columns)
    target_idx = feature_names.index(resolved_target)
    logret_idx = feature_names.index("__log_return")

    for key in group_keys:
        gmask = (df_keys[str(ticker_col)] == str(key)).to_numpy()
        xg = x_scaled[gmask]
        yg = y_scaled[gmask]
        last_obs = xg[:, target_idx]
        last_ret = xg[:, logret_idx]
        gdates = df_keys.loc[gmask, str(date_col)].to_numpy()
        gtickers = df_keys.loc[gmask, str(ticker_col)].to_numpy()

        n = int(xg.shape[0])
        train_n, val_n, test_n = _split_sizes(n, train_frac=float(train_frac), val_frac=float(val_frac))
        if min(train_n, val_n + test_n) <= 0:
            continue

        slices = {
            "train": slice(0, train_n),
            "val": slice(train_n, train_n + val_n),
            "test": slice(train_n + val_n, train_n + val_n + test_n),
        }

        for split, sl in slices.items():
            xs, ys = _build_windows(xg[sl], yg[sl], window=window)
            if xs.shape[0] <= 0:
                continue
            last_obs_split = last_obs[sl][window - 1 :]
            if split == "train":
                x_train_parts.append(xs)
                y_train_parts.append(ys)
            elif split == "val":
                x_val_parts.append(xs)
                y_val_parts.append(ys)
            else:
                x_test_parts.append(xs)
                y_test_parts.append(ys)
                last_observed_parts.append(last_obs_split.astype(np.float32))
                if mode == "log_return":
                    last_return_parts.append(last_ret[sl][window - 1 :].astype(np.float32))
                _, _, dates_end = _build_windows_meta(xg[sl], yg[sl], gdates[sl], window=window)
                test_dates_parts.append(dates_end)
                _, _, tickers_end = _build_windows_meta(xg[sl], yg[sl], gtickers[sl], window=window)
                test_tickers_parts.append(tickers_end)

    if not x_train_parts or not x_val_parts or not x_test_parts:
        raise ValueError("Not enough data after windowing; try reducing --window.")

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_val = np.concatenate(x_val_parts, axis=0)
    y_val = np.concatenate(y_val_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    last_observed_test = np.concatenate(last_observed_parts, axis=0).reshape(-1)
    last_return_test = np.concatenate(last_return_parts, axis=0).reshape(-1) if (mode == "log_return" and last_return_parts) else None
    test_dates = np.concatenate(test_dates_parts, axis=0)
    test_tickers = np.concatenate(test_tickers_parts, axis=0)

    model = _build_model(window=window, n_features=x_train.shape[-1], seed=int(seed))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
    ]

    if mode == "price":
        base_train_s = x_train[:, -1, int(target_idx)].reshape(-1)
        base_val_s = x_val[:, -1, int(target_idx)].reshape(-1)
        base_test_s = x_test[:, -1, int(target_idx)].reshape(-1)
        y_train_fit = (y_train.reshape(-1) - base_train_s).astype(np.float32)
        y_val_fit = (y_val.reshape(-1) - base_val_s).astype(np.float32)
        y_test_fit = (y_test.reshape(-1) - base_test_s).astype(np.float32)
    else:
        y_train_fit = y_train
        y_val_fit = y_val
        y_test_fit = y_test
        base_test_s = None

    start = time.time()
    history = model.fit(
        x_train,
        y_train_fit,
        validation_data=(x_val, y_val_fit),
        epochs=int(epochs),
        batch_size=int(batch_size),
        verbose=2,
        callbacks=callbacks,
    )
    fit_seconds = float(time.time() - start)

    y_pred_scaled = model.predict(x_test, batch_size=int(batch_size), verbose=0).reshape(-1)
    if mode == "price":
        y_pred_scaled = (base_test_s.astype(np.float32) + y_pred_scaled.astype(np.float32)).reshape(-1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    baseline_pred = None
    baseline_last_ret_pred = None
    baseline_metrics: Dict[str, Any] = {}
    if mode == "price":
        baseline_in = np.zeros((int(last_observed_test.size), int(x_train.shape[-1])), dtype=np.float32)
        baseline_in[:, int(target_idx)] = last_observed_test.astype(np.float32, copy=False)
        baseline_pred = x_scaler.inverse_transform(baseline_in)[:, int(target_idx)].reshape(-1)
        baseline_metrics["last_value"] = _metrics_regression(y_true, baseline_pred)
    else:
        baseline_zero = np.zeros_like(y_true, dtype=np.float64)
        baseline_metrics["zero_return"] = _metrics_regression(y_true, baseline_zero)
        if last_return_test is not None:
            baseline_in = np.zeros((int(last_return_test.size), int(x_train.shape[-1])), dtype=np.float32)
            baseline_in[:, int(logret_idx)] = last_return_test.astype(np.float32, copy=False)
            baseline_last_ret_pred = x_scaler.inverse_transform(baseline_in)[:, int(logret_idx)].reshape(-1)
            baseline_metrics["last_return"] = _metrics_regression(y_true, baseline_last_ret_pred)

    metrics_model = _metrics_regression(y_true, y_pred)

    direction_acc = None
    if y_true.size:
        base = baseline_pred if baseline_pred is not None else np.zeros_like(y_true)
        dir_true = np.sign(y_true - base)
        dir_pred = np.sign(y_pred - base)
        direction_acc = float(np.mean(dir_true == dir_pred))

    out: Dict[str, Any] = {
        "csv_path": csv_path,
        "date_col": str(date_col),
        "ticker_col": None if ticker_col == "__single__" else str(ticker_col),
        "ticker_value": None if ticker_value is None else str(ticker_value),
        "target_col": resolved_target,
        "target_mode": str(target_mode),
        "horizon": int(horizon),
        "window": int(window),
        "n_features": int(x_train.shape[-1]),
        "n_samples": {"train": int(x_train.shape[0]), "val": int(x_val.shape[0]), "test": int(x_test.shape[0])},
        "fit_seconds": fit_seconds,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "metrics": {"model": metrics_model, "baselines": baseline_metrics, "direction_accuracy": direction_acc},
        "x_scaler": {"kind": scaler_kind},
    }

    baseline_rmse = None
    if mode == "price":
        baseline_rmse = baseline_metrics.get("last_value", {}).get("rmse")
    else:
        baseline_rmse = baseline_metrics.get("zero_return", {}).get("rmse")
    margin = 1e-6
    eligible = False
    if (metrics_model.get("rmse") is not None) and (baseline_rmse is not None):
        eligible = bool(float(metrics_model["rmse"]) < float(baseline_rmse) - margin)
    out["promotion"] = {
        "task": "regression",
        "primary_metric": "rmse",
        "primary": float(metrics_model["rmse"]) if metrics_model.get("rmse") is not None else None,
        "baseline_name": "last_value" if mode == "price" else "zero_return",
        "baseline": float(baseline_rmse) if baseline_rmse is not None else None,
        "eligible": eligible,
    }

    predictions_path = None
    if predictions_csv:
        predictions_path = os.path.abspath(str(predictions_csv))
    elif save_model:
        predictions_path = os.path.join(os.path.dirname(metadata_path), "stock_lstm_predictions.csv")

    base_ref = baseline_pred.astype(np.float64) if baseline_pred is not None else np.zeros_like(y_true, dtype=np.float64)
    pred_cols = {
        "ticker": pd.Series(test_tickers).astype("string"),
        "date": pd.to_datetime(test_dates),
        "actual": y_true.astype(np.float64),
        "predicted": y_pred.astype(np.float64),
        "baseline_ref": base_ref.astype(np.float64),
    }
    if baseline_pred is not None:
        pred_cols["baseline_last_value"] = baseline_pred.astype(np.float64)
    if baseline_last_ret_pred is not None:
        pred_cols["baseline_last_return"] = baseline_last_ret_pred.astype(np.float64)
    pred_df = pd.DataFrame(pred_cols)
    pred_df["error"] = pred_df["predicted"] - pred_df["actual"]
    pred_df["abs_error"] = pred_df["error"].abs()
    pred_df["pct_error"] = (pred_df["abs_error"] / pred_df["actual"].abs().clip(lower=1e-8)).astype(np.float64)
    pred_df["direction_true"] = np.sign(pred_df["actual"] - pred_df["baseline_ref"]).astype(np.int8)
    pred_df["direction_pred"] = np.sign(pred_df["predicted"] - pred_df["baseline_ref"]).astype(np.int8)
    pred_df = pred_df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)

    out["predictions_path"] = predictions_path
    if predictions_path:
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        pred_df.to_csv(predictions_path, index=False)

    show_n = int(n_show)
    if show_n > 0:
        shown = pred_df.tail(show_n).copy()
        shown["date"] = shown["date"].dt.strftime("%Y-%m-%d")
        out["predictions_preview_tail"] = shown.to_dict(orient="records")

    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        out["fingerprint"] = {
            "model_sha256": _sha256_file(model_path),
            "canary_pred_sha256": _sha256_array(y_pred[: min(int(y_pred.shape[0]), 256)]),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    return out


def main() -> None:
    artifacts = _default_artifacts()
    default_csv = os.path.join(os.path.dirname(__file__), "infolimpioavanzadoTarget.csv", "infolimpioavanzadoTarget.csv")

    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=default_csv)
    p.add_argument("--date-col", default="date")
    p.add_argument("--ticker-col", default="ticker")
    p.add_argument("--ticker-value", default=None)
    p.add_argument("--target-col", default=None)
    p.add_argument("--target-mode", default="price")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--x-scaler", default="standard")
    p.add_argument("--predictions-csv", default=None)
    p.add_argument("--n-show", type=int, default=20)
    p.add_argument("--model-path", default=artifacts.model_path)
    p.add_argument("--metadata-path", default=artifacts.metadata_path)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    result = train_and_test(
        csv_path=str(args.csv),
        date_col=str(args.date_col),
        ticker_col=str(args.ticker_col) if args.ticker_col is not None else None,
        ticker_value=args.ticker_value,
        target_col=args.target_col,
        target_mode=str(getattr(args, "target_mode", "price")),
        horizon=int(args.horizon),
        window=int(args.window),
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        predictions_csv=args.predictions_csv,
        n_show=int(args.n_show),
        model_path=str(args.model_path),
        metadata_path=str(args.metadata_path),
        save_model=bool(args.save_model),
        x_scaler_kind=str(getattr(args, "x_scaler", "standard")),
    )

    print(json.dumps(result["metrics"], indent=2))
    if "predictions_preview_tail" in result:
        preview = pd.DataFrame(result["predictions_preview_tail"])
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(preview.to_string(index=False))
    if args.save_model:
        print(f"Saved model: {os.path.abspath(str(args.model_path))}")
        print(f"Saved metadata: {os.path.abspath(str(args.metadata_path))}")
        if result.get("predictions_path"):
            print(f"Saved predictions: {os.path.abspath(str(result['predictions_path']))}")


if __name__ == "__main__":
    main()
