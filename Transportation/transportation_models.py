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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Artifacts:
    cnn_model_path: str
    cnn_metadata_path: str
    cnn_predictions_path: str
    rl_metadata_path: str


def _default_artifacts(base_dir: str) -> Artifacts:
    return Artifacts(
        cnn_model_path=os.path.join(base_dir, "traffic_cnn.keras"),
        cnn_metadata_path=os.path.join(base_dir, "traffic_cnn.meta.json"),
        cnn_predictions_path=os.path.join(base_dir, "traffic_cnn_predictions.csv"),
        rl_metadata_path=os.path.join(base_dir, "routing_rl.meta.json"),
    )


def _read_csv_any(*paths: str) -> pd.DataFrame:
    for p in paths:
        fp = os.path.abspath(p)
        if os.path.isfile(fp):
            return pd.read_csv(fp)
    raise FileNotFoundError(f"CSV not found in any of: {paths}")


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


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _clean_traffic_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    required = ["DateTime", "Junction", "Vehicles"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column {c}. Found: {list(df.columns)}")

    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
    out["Junction"] = pd.to_numeric(out["Junction"], errors="coerce")
    out["Vehicles"] = pd.to_numeric(out["Vehicles"], errors="coerce")

    before_rows = int(len(out))
    out = out.dropna(subset=["DateTime", "Junction", "Vehicles"]).reset_index(drop=True)
    out["Junction"] = out["Junction"].astype(int)

    neg_before = int((out["Vehicles"] < 0).sum())
    out.loc[out["Vehicles"] < 0, "Vehicles"] = np.nan

    out = out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)

    filled_parts: List[pd.DataFrame] = []
    filled_total = 0
    for j, g in out.groupby("Junction", sort=True):
        g = g.sort_values("DateTime")
        dt_min = g["DateTime"].min()
        dt_max = g["DateTime"].max()
        full_idx = pd.date_range(dt_min, dt_max, freq="h")
        g2 = g.set_index("DateTime")[["Vehicles"]].reindex(full_idx)
        missing = int(g2["Vehicles"].isna().sum())
        filled_total += missing
        g2["Vehicles"] = g2["Vehicles"].interpolate(limit_direction="both")
        g2["Vehicles"] = g2["Vehicles"].fillna(g2["Vehicles"].median())
        g2 = g2.reset_index().rename(columns={"index": "DateTime"})
        g2["Junction"] = int(j)
        filled_parts.append(g2)

    out = pd.concat(filled_parts, ignore_index=True)
    out["Vehicles"] = out["Vehicles"].astype(np.float32)

    p01 = float(np.nanpercentile(out["Vehicles"].to_numpy(), 1))
    p99 = float(np.nanpercentile(out["Vehicles"].to_numpy(), 99))
    out["Vehicles"] = out["Vehicles"].clip(lower=p01, upper=p99)

    meta = {
        "rows_before": before_rows,
        "rows_after": int(len(out)),
        "negatives_set_to_nan": neg_before,
        "filled_missing_points": int(filled_total),
        "vehicles_clip_p01": p01,
        "vehicles_clip_p99": p99,
        "junctions": sorted([int(x) for x in out["Junction"].unique().tolist()]),
        "datetime_min": str(out["DateTime"].min()),
        "datetime_max": str(out["DateTime"].max()),
    }
    return out, meta


def _time_features(dt: pd.Series) -> pd.DataFrame:
    hour = dt.dt.hour.astype(np.float32)
    dow = dt.dt.dayofweek.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    return pd.DataFrame(
        {
            "hour_sin": hour_sin.astype(np.float32),
            "hour_cos": hour_cos.astype(np.float32),
            "dow_sin": dow_sin.astype(np.float32),
            "dow_cos": dow_cos.astype(np.float32),
        }
    )


def _make_windows(
    *,
    df: pd.DataFrame,
    window: int,
    horizon: int,
    junction_col: str,
    time_col: str,
    target_col: str,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    dt_parts: List[np.ndarray] = []
    j_parts: List[np.ndarray] = []

    for j, g in df.groupby(junction_col, sort=True):
        g = g.sort_values(time_col).reset_index(drop=True)
        dt = g[time_col]
        tfeat = _time_features(dt)
        vehicles = g[[target_col]].to_numpy(dtype=np.float32)
        vehicles_scaled = scaler.transform(vehicles).astype(np.float32)
        feat = np.concatenate([vehicles_scaled, tfeat.to_numpy(dtype=np.float32)], axis=1)

        n = int(len(g))
        max_start = n - int(window) - int(horizon) + 1
        if max_start <= 0:
            continue
        for start in range(max_start):
            end = start + int(window)
            target_idx = end + int(horizon) - 1
            x_parts.append(feat[start:end, :])
            y_parts.append(vehicles_scaled[target_idx, 0:1])
            dt_parts.append(np.array([dt.iloc[target_idx]], dtype="datetime64[ns]"))
            j_parts.append(np.array([int(j)], dtype=np.int32))

    x = np.stack(x_parts, axis=0) if x_parts else np.zeros((0, int(window), 5), dtype=np.float32)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.zeros((0, 1), dtype=np.float32)
    dts = np.concatenate(dt_parts, axis=0) if dt_parts else np.zeros((0,), dtype="datetime64[ns]")
    js = np.concatenate(j_parts, axis=0) if j_parts else np.zeros((0,), dtype=np.int32)
    return x, y, dts, js


def _split_timewise(dts: np.ndarray, *, val_frac: float, test_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dts = np.asarray(dts)
    if dts.size == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    order = np.argsort(dts)
    n = int(dts.size)
    test_n = max(int(np.floor(n * float(test_frac))), 1)
    val_n = max(int(np.floor(n * float(val_frac))), 1)
    train_n = max(n - val_n - test_n, 1)
    train_end = train_n
    val_end = train_n + val_n

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    train_mask = np.zeros((n,), dtype=bool)
    val_mask = np.zeros((n,), dtype=bool)
    test_mask = np.zeros((n,), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _build_cnn(*, window: int, n_features: int, seed: int, filters: int, dropout: float, lr: float) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(int(seed))
    inputs = tf.keras.Input(shape=(int(window), int(n_features)))
    x = tf.keras.layers.Conv1D(filters=int(filters), kernel_size=3, padding="causal", activation="relu")(inputs)
    x = tf.keras.layers.Dropout(float(dropout))(x)
    x = tf.keras.layers.Conv1D(filters=int(filters), kernel_size=3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(int(filters), activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss="mse", metrics=["mae"])
    return model


def train_cnn_traffic(
    *,
    transportation_dir: str,
    window: int,
    horizon: int,
    seed: int,
    epochs: int,
    batch_size: int,
    val_frac: float,
    test_frac: float,
    save_model: bool,
    tune_trials: int = 0,
) -> Dict[str, Any]:
    transportation_dir = os.path.abspath(transportation_dir)
    traffic = _read_csv_any(
        os.path.join(transportation_dir, "traffic.csv", "traffic.csv"),
        os.path.join(transportation_dir, "traffic.csv"),
    )
    traffic_clean, cleaning_meta = _clean_traffic_df(traffic)

    scaler = StandardScaler()
    scaler.fit(traffic_clean[["Vehicles"]].to_numpy(dtype=np.float32))

    x, y, dts, junctions = _make_windows(
        df=traffic_clean,
        window=int(window),
        horizon=int(horizon),
        junction_col="Junction",
        time_col="DateTime",
        target_col="Vehicles",
        scaler=scaler,
    )
    train_mask, val_mask, test_mask = _split_timewise(dts, val_frac=float(val_frac), test_frac=float(test_frac))

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]
    dt_test = dts[test_mask]
    j_test = junctions[test_mask]

    base_train_s = x_train[:, -1, 0].astype(np.float32)
    base_val_s = x_val[:, -1, 0].astype(np.float32)
    base_test_s = x_test[:, -1, 0].astype(np.float32)

    y_train_delta = (y_train.reshape(-1) - base_train_s.reshape(-1)).astype(np.float32)
    y_val_delta = (y_val.reshape(-1) - base_val_s.reshape(-1)).astype(np.float32)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
    ]

    trials: List[Dict[str, Any]] = []
    best = None
    if int(tune_trials) > 0:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(tune_trials)):
            cfg = {
                "filters": int(rng.choice([32, 64, 96])),
                "dropout": float(rng.choice([0.1, 0.2, 0.3])),
                "lr": float(rng.choice([1e-3, 5e-4, 2e-4])),
            }
            model = _build_cnn(window=int(window), n_features=int(x.shape[-1]), seed=int(seed), **cfg)
            start = time.time()
            history = model.fit(
                x_train,
                y_train_delta,
                validation_data=(x_val, y_val_delta),
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                callbacks=callbacks,
            )
            fit_seconds = float(time.time() - start)
            best_val = float(np.min(np.asarray(history.history.get("val_loss", [np.inf]), dtype=np.float64)))
            t = {**cfg, "fit_seconds": fit_seconds, "best_val_loss": best_val}
            trials.append(t)
            if best is None or best_val < float(best["best_val_loss"]):
                best = t
                best_model = model
                best_history = history
        model = best_model
        history = best_history
        fit_seconds = float(best["fit_seconds"])
        selected = {k: best[k] for k in ("filters", "dropout", "lr")}
    else:
        selected = {"filters": 64, "dropout": 0.2, "lr": 1e-3}
        model = _build_cnn(window=int(window), n_features=int(x.shape[-1]), seed=int(seed), **selected)
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

    y_pred_delta_scaled = model.predict(x_test, batch_size=int(batch_size), verbose=0).reshape(-1, 1)
    y_pred_scaled = (base_test_s.reshape(-1, 1) + y_pred_delta_scaled).astype(np.float32)
    y_true_scaled = y_test.reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y_true_scaled).reshape(-1)

    base_last_scaled = base_test_s.reshape(-1, 1).astype(np.float32)
    base_last = scaler.inverse_transform(base_last_scaled).reshape(-1)
    baselines: Dict[str, np.ndarray] = {"last_value": base_last}
    if int(window) >= 24:
        base_day_scaled = x_test[:, -24, 0].reshape(-1, 1).astype(np.float32)
        baselines["prev_day"] = scaler.inverse_transform(base_day_scaled).reshape(-1)
    if int(window) >= 168:
        base_week_scaled = x_test[:, -168, 0].reshape(-1, 1).astype(np.float32)
        baselines["prev_week"] = scaler.inverse_transform(base_week_scaled).reshape(-1)

    baseline_metrics: Dict[str, Any] = {}
    for name, pred in baselines.items():
        baseline_metrics[name] = {"overall": _metrics_regression(y_true, pred)}
        byj: Dict[str, Any] = {}
        for j in sorted(np.unique(j_test).tolist()):
            m = j_test == j
            if int(m.sum()) == 0:
                continue
            byj[str(int(j))] = _metrics_regression(y_true[m], pred[m])
        baseline_metrics[name]["by_junction"] = byj

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

    metrics = _metrics_regression(y_true, y_pred)

    by_junction: Dict[str, Any] = {}
    for j in sorted(np.unique(j_test).tolist()):
        m = j_test == j
        if int(m.sum()) == 0:
            continue
        by_junction[str(int(j))] = _metrics_regression(y_true[m], y_pred[m])

    artifacts = _default_artifacts(transportation_dir)
    pred_df = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(dt_test).astype("datetime64[ns]"),
            "Junction": j_test.astype(int),
            "actual_vehicles": y_true.astype(np.float32),
            "baseline_last_value": baselines["last_value"].astype(np.float32),
            "baseline_prev_day": baselines["prev_day"].astype(np.float32) if "prev_day" in baselines else np.full_like(y_true.astype(np.float32), np.nan),
            "baseline_prev_week": baselines["prev_week"].astype(np.float32) if "prev_week" in baselines else np.full_like(y_true.astype(np.float32), np.nan),
            "predicted_vehicles": y_pred.astype(np.float32),
        }
    ).sort_values(["Junction", "DateTime"])

    out: Dict[str, Any] = {
        "transportation_dir": transportation_dir,
        "dataset": {"traffic_csv": os.path.join(transportation_dir, "traffic.csv", "traffic.csv")},
        "cleaning": cleaning_meta,
        "window": int(window),
        "horizon": int(horizon),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "n_windows": int(x.shape[0]),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "fit_seconds": fit_seconds,
        "tuning": {"trials": trials, "best": best} if int(tune_trials) > 0 else None,
        "selected_hparams": selected,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "metrics": {"overall": metrics, "by_junction": by_junction},
        "baseline_metrics": baseline_metrics,
        "inference": inference,
        "artifacts": {
            "model_path": artifacts.cnn_model_path if save_model else None,
            "metadata_path": artifacts.cnn_metadata_path,
            "predictions_path": artifacts.cnn_predictions_path,
        },
    }

    promote_primary = float(metrics.get("rmse", 0.0))
    best_base_name = "none"
    best_base_rmse = None
    for name, m in baseline_metrics.items():
        rmse = float(m.get("overall", {}).get("rmse") or float("inf"))
        if best_base_rmse is None or rmse < best_base_rmse:
            best_base_rmse = rmse
            best_base_name = str(name)
    promote_baseline = float(best_base_rmse) if best_base_rmse is not None and np.isfinite(best_base_rmse) else 0.0
    out["promotion"] = {
        "task": "time_series_regression",
        "primary_metric": "rmse",
        "primary": promote_primary,
        "baseline_name": best_base_name,
        "baseline": promote_baseline,
        "eligible": bool(promote_primary < promote_baseline * 0.99),
    }

    pred_df.to_csv(artifacts.cnn_predictions_path, index=False)
    with open(artifacts.cnn_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        model.save(artifacts.cnn_model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.cnn_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.cnn_predictions_path),
        "canary_pred_sha256": _sha256_array(y_pred[: min(256, int(y_pred.shape[0]))]),
    }
    with open(artifacts.cnn_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def _build_graph_from_junctions(junctions: Sequence[int]) -> Dict[int, List[int]]:
    js = sorted({int(x) for x in junctions})
    if len(js) <= 1:
        return {int(js[0]): []} if js else {}
    graph: Dict[int, List[int]] = {j: [] for j in js}
    for i, j in enumerate(js):
        for k in js:
            if k == j:
                continue
            if abs(int(k) - int(j)) == 1:
                graph[j].append(k)
        if i + 2 < len(js):
            graph[j].append(js[i + 2])
        if i - 2 >= 0:
            graph[j].append(js[i - 2])
        graph[j] = sorted(set(graph[j]))
    return graph


def _normalize_series(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    v = np.asarray(values, dtype=np.float32)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(v), 0.0, 1.0
    return (v - vmin) / (vmax - vmin), vmin, vmax


def train_rl_routing(
    *,
    transportation_dir: str,
    episodes: int,
    max_steps: int,
    epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    alpha: float,
    gamma: float,
    congestion_alpha: float,
    reroute_penalty_prob: float,
    seed: int,
    tune_trials: int = 0,
    tune_episodes: int = 1000,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    transportation_dir = os.path.abspath(transportation_dir)
    traffic = _read_csv_any(
        os.path.join(transportation_dir, "traffic.csv", "traffic.csv"),
        os.path.join(transportation_dir, "traffic.csv"),
    )
    traffic_clean, cleaning_meta = _clean_traffic_df(traffic)
    traffic_clean["hour"] = traffic_clean["DateTime"].dt.hour.astype(int)

    junctions = sorted({int(x) for x in traffic_clean["Junction"].unique().tolist()})
    graph = _build_graph_from_junctions(junctions)
    if len(junctions) < 2:
        raise ValueError("Need at least two junctions for routing.")
    start_node = int(min(junctions))
    goal_node = int(max(junctions))

    pivot = traffic_clean.pivot_table(index="DateTime", columns="Junction", values="Vehicles", aggfunc="mean").sort_index()
    pivot = pivot.reindex(columns=junctions)
    pivot = pivot.interpolate(limit_direction="both").fillna(pivot.median())
    times = pivot.index.to_numpy()
    vehicles = pivot.to_numpy(dtype=np.float32)
    vehicles_norm, vmin, vmax = _normalize_series(vehicles)

    base_edge_time = 60.0
    q: Dict[Tuple[int, int, int], float] = {}

    def state_key(node: int, hour: int, cong_bin: int) -> Tuple[int, int, int]:
        return int(node), int(hour), int(cong_bin)

    def congestion_bin(v: float) -> int:
        if not np.isfinite(v):
            return 0
        return int(np.clip(np.floor(float(v) * 4.0), 0, 3))

    def choose_action(node: int, hour: int, cong: float, eps: float) -> int:
        actions = graph.get(int(node), [])
        if not actions:
            return int(node)
        if rng.random() < float(eps):
            return int(rng.choice(actions))
        sk = state_key(int(node), int(hour), congestion_bin(float(cong)))
        best_a = None
        best_q = None
        for a in actions:
            val = q.get((sk[0] * 100 + sk[1] * 10 + sk[2], int(a), goal_node), 0.0)
            if best_q is None or val > best_q:
                best_q = val
                best_a = int(a)
        return int(best_a) if best_a is not None else int(rng.choice(actions))

    def max_next_q(next_node: int, next_hour: int, next_cong: float) -> float:
        actions = graph.get(int(next_node), [])
        if not actions:
            return 0.0
        sk = state_key(int(next_node), int(next_hour), congestion_bin(float(next_cong)))
        best_q = None
        for a in actions:
            val = q.get((sk[0] * 100 + sk[1] * 10 + sk[2], int(a), goal_node), 0.0)
            if best_q is None or val > best_q:
                best_q = val
        return float(best_q if best_q is not None else 0.0)

    def train_q(*, episodes_n: int, eps_start: float, eps_decay: float, alpha_v: float, gamma_v: float, cong_alpha_v: float, reroute_prob_v: float) -> Tuple[Dict[Tuple[int, int, int], float], List[float], List[int]]:
        q_local: Dict[Tuple[int, int, int], float] = {}
        eps = float(eps_start)
        rets: List[float] = []
        steps_list: List[int] = []
        for _ in range(int(episodes_n)):
            t_idx = int(rng.integers(0, max(1, len(times) - 2)))
            node = int(start_node)
            total_reward = 0.0
            steps = 0
            extra_penalty_edges: Optional[Tuple[int, int]] = None
            if rng.random() < float(reroute_prob_v):
                a = int(rng.choice(graph[start_node]))
                extra_penalty_edges = (start_node, a)

            while steps < int(max_steps) and node != goal_node:
                hour = int(pd.Timestamp(times[t_idx]).hour)
                cong = float(vehicles_norm[t_idx, junctions.index(node)])
                actions = graph.get(int(node), [])
                if not actions:
                    break
                if rng.random() < float(eps):
                    next_node = int(rng.choice(actions))
                else:
                    sk = state_key(int(node), int(hour), congestion_bin(float(cong)))
                    sk_flat = sk[0] * 100 + sk[1] * 10 + sk[2]
                    best_a = None
                    best_q = None
                    for a in actions:
                        val = q_local.get((int(sk_flat), int(a), goal_node), 0.0)
                        if best_q is None or val > best_q:
                            best_q = val
                            best_a = int(a)
                    next_node = int(best_a if best_a is not None else int(rng.choice(actions)))

                next_t = min(t_idx + 1, len(times) - 1)
                next_cong = float(vehicles_norm[next_t, junctions.index(next_node)])
                travel_time = float(base_edge_time * (1.0 + float(cong_alpha_v) * next_cong))
                if extra_penalty_edges and (node, next_node) == extra_penalty_edges:
                    travel_time *= 2.0
                reward = -travel_time

                sk = state_key(node, hour, congestion_bin(cong))
                sk_flat = sk[0] * 100 + sk[1] * 10 + sk[2]
                q_key = (int(sk_flat), int(next_node), goal_node)
                old_q = float(q_local.get(q_key, 0.0))

                next_actions = graph.get(int(next_node), [])
                next_best = 0.0
                if next_actions:
                    sk2 = state_key(int(next_node), int(pd.Timestamp(times[next_t]).hour), congestion_bin(float(next_cong)))
                    sk2_flat = sk2[0] * 100 + sk2[1] * 10 + sk2[2]
                    next_best = max([float(q_local.get((int(sk2_flat), int(a), goal_node), 0.0)) for a in next_actions] + [0.0])

                target = float(reward + float(gamma_v) * next_best)
                q_local[q_key] = float(old_q + float(alpha_v) * (target - old_q))

                node = next_node
                t_idx = next_t
                total_reward += float(reward)
                steps += 1

            rets.append(float(total_reward))
            steps_list.append(int(steps))
            eps = max(float(min_epsilon), float(eps) * float(eps_decay))
        return q_local, rets, steps_list

    def _eval_policy_local(q_local: Dict[Tuple[int, int, int], float], *, n_episodes: int, reroute: bool, cong_alpha_v: float) -> Dict[str, float]:
        costs: List[float] = []
        succ = 0
        for _ in range(int(n_episodes)):
            t_idx = int(rng.integers(0, max(1, len(times) - 2)))
            node = int(start_node)
            steps = 0
            cost = 0.0
            extra_penalty_edges = None
            if reroute and rng.random() < 0.5:
                a = int(rng.choice(graph[start_node]))
                extra_penalty_edges = (start_node, a)

            while steps < int(max_steps) and node != goal_node:
                hour = int(pd.Timestamp(times[t_idx]).hour)
                cong = float(vehicles_norm[t_idx, junctions.index(node)])
                actions = graph.get(int(node), [])
                if not actions:
                    break
                sk = state_key(int(node), int(hour), congestion_bin(float(cong)))
                sk_flat = sk[0] * 100 + sk[1] * 10 + sk[2]
                best_a = None
                best_q = None
                for a2 in actions:
                    val = float(q_local.get((int(sk_flat), int(a2), goal_node), 0.0))
                    if best_q is None or val > best_q:
                        best_q = val
                        best_a = int(a2)
                next_node = int(best_a if best_a is not None else int(rng.choice(actions)))

                next_t = min(t_idx + 1, len(times) - 1)
                next_cong = float(vehicles_norm[next_t, junctions.index(next_node)])
                travel_time = float(base_edge_time * (1.0 + float(cong_alpha_v) * next_cong))
                if extra_penalty_edges and (node, next_node) == extra_penalty_edges:
                    travel_time *= 2.0
                cost += float(travel_time)
                node = next_node
                t_idx = next_t
                steps += 1
            if node == goal_node:
                succ += 1
            costs.append(float(cost))
        return {
            "avg_cost": float(np.mean(costs)) if costs else 0.0,
            "p50_cost": float(np.percentile(costs, 50)) if costs else 0.0,
            "success_rate": float(succ / max(1, int(n_episodes))),
        }

    tuning: Optional[Dict[str, Any]] = None
    if int(tune_trials) > 0:
        rng2 = np.random.default_rng(int(seed) + 1)
        best_t = None
        best_eval = None
        trials: List[Dict[str, Any]] = []
        for _ in range(int(tune_trials)):
            cfg = {
                "epsilon_decay": float(rng2.choice([0.995, 0.997, 0.999])),
                "alpha": float(rng2.choice([0.1, 0.2, 0.3])),
                "gamma": float(rng2.choice([0.9, 0.95, 0.98])),
                "congestion_alpha": float(rng2.choice([1.0, 1.5, 2.0])),
                "reroute_penalty_prob": float(rng2.choice([0.2, 0.3, 0.4])),
            }
            q_try, rets, steps_list = train_q(
                episodes_n=int(tune_episodes),
                eps_start=float(epsilon),
                eps_decay=float(cfg["epsilon_decay"]),
                alpha_v=float(cfg["alpha"]),
                gamma_v=float(cfg["gamma"]),
                cong_alpha_v=float(cfg["congestion_alpha"]),
                reroute_prob_v=float(cfg["reroute_penalty_prob"]),
            )

            ev = _eval_policy_local(q_try, n_episodes=200, reroute=False, cong_alpha_v=float(cfg["congestion_alpha"]))

            trials.append({**cfg, "avg_cost": float(ev["avg_cost"]), "success_rate": float(ev["success_rate"])})
            score = float(ev["avg_cost"])
            if best_eval is None or score < best_eval:
                best_eval = score
                best_t = cfg

        tuning = {"trials": trials, "best": best_t}
        q, episode_returns, episode_steps = train_q(
            episodes_n=int(episodes),
            eps_start=float(epsilon),
            eps_decay=float(best_t["epsilon_decay"]),
            alpha_v=float(best_t["alpha"]),
            gamma_v=float(best_t["gamma"]),
            cong_alpha_v=float(best_t["congestion_alpha"]),
            reroute_prob_v=float(best_t["reroute_penalty_prob"]),
        )
        alpha = float(best_t["alpha"])
        gamma = float(best_t["gamma"])
        congestion_alpha = float(best_t["congestion_alpha"])
        reroute_penalty_prob = float(best_t["reroute_penalty_prob"])
        epsilon_decay = float(best_t["epsilon_decay"])
    else:
        q, episode_returns, episode_steps = train_q(
            episodes_n=int(episodes),
            eps_start=float(epsilon),
            eps_decay=float(epsilon_decay),
            alpha_v=float(alpha),
            gamma_v=float(gamma),
            cong_alpha_v=float(congestion_alpha),
            reroute_prob_v=float(reroute_penalty_prob),
        )

    def eval_policy(n_episodes: int, reroute: bool) -> Dict[str, float]:
        costs: List[float] = []
        succ = 0
        for _ in range(int(n_episodes)):
            t_idx = int(rng.integers(0, max(1, len(times) - 2)))
            node = int(start_node)
            steps = 0
            cost = 0.0
            extra_penalty_edges = None
            if reroute and rng.random() < 0.5:
                a = int(rng.choice(graph[start_node]))
                extra_penalty_edges = (start_node, a)

            while steps < int(max_steps) and node != goal_node:
                hour = int(pd.Timestamp(times[t_idx]).hour)
                cong = float(vehicles_norm[t_idx, junctions.index(node)])
                action = choose_action(node, hour, cong, 0.0)
                next_node = int(action)
                next_t = min(t_idx + 1, len(times) - 1)
                next_cong = float(vehicles_norm[next_t, junctions.index(next_node)])
                travel_time = float(base_edge_time * (1.0 + float(congestion_alpha) * next_cong))
                if extra_penalty_edges and (node, next_node) == extra_penalty_edges:
                    travel_time *= 2.0
                cost += float(travel_time)
                node = next_node
                t_idx = next_t
                steps += 1
            if node == goal_node:
                succ += 1
            costs.append(float(cost))
        return {
            "avg_cost": float(np.mean(costs)) if costs else 0.0,
            "p50_cost": float(np.percentile(costs, 50)) if costs else 0.0,
            "success_rate": float(succ / max(1, int(n_episodes))),
        }

    eval_clean = eval_policy(200, reroute=False)
    eval_reroute = eval_policy(200, reroute=True)

    def eval_random(n_episodes: int) -> Dict[str, float]:
        costs: List[float] = []
        succ = 0
        for _ in range(int(n_episodes)):
            t_idx = int(rng.integers(0, max(1, len(times) - 2)))
            node = int(start_node)
            steps = 0
            cost = 0.0
            while steps < int(max_steps) and node != goal_node:
                actions = graph.get(int(node), [])
                if not actions:
                    break
                next_node = int(rng.choice(actions))
                next_t = min(t_idx + 1, len(times) - 1)
                next_cong = float(vehicles_norm[next_t, junctions.index(next_node)])
                cost += float(base_edge_time * (1.0 + float(congestion_alpha) * next_cong))
                node = next_node
                t_idx = next_t
                steps += 1
            if node == goal_node:
                succ += 1
            costs.append(float(cost))
        return {
            "avg_cost": float(np.mean(costs)) if costs else 0.0,
            "p50_cost": float(np.percentile(costs, 50)) if costs else 0.0,
            "success_rate": float(succ / max(1, int(n_episodes))),
        }

    baseline_random = eval_random(200)

    artifacts = _default_artifacts(transportation_dir)
    out: Dict[str, Any] = {
        "transportation_dir": transportation_dir,
        "cleaning": cleaning_meta,
        "junctions": junctions,
        "graph": {str(k): [int(x) for x in v] for k, v in graph.items()},
        "start_node": int(start_node),
        "goal_node": int(goal_node),
        "vehicles_min": float(vmin),
        "vehicles_max": float(vmax),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "epsilon_start": float(epsilon),
        "epsilon_decay": float(epsilon_decay),
        "min_epsilon": float(min_epsilon),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "congestion_alpha": float(congestion_alpha),
        "reroute_penalty_prob": float(reroute_penalty_prob),
        "training": {
            "avg_return_last_200": float(np.mean(episode_returns[-200:])) if episode_returns else 0.0,
            "avg_steps_last_200": float(np.mean(episode_steps[-200:])) if episode_steps else 0.0,
            "q_entries": int(len(q)),
        },
        "evaluation": {"no_reroute": eval_clean, "reroute": eval_reroute},
        "baseline": {"random_policy": baseline_random},
        "tuning": tuning,
        "artifacts": {"metadata_path": artifacts.rl_metadata_path},
    }

    promote_primary = float(out.get("evaluation", {}).get("reroute", {}).get("success_rate") or 0.0)
    promote_baseline = float(out.get("baseline", {}).get("random_policy", {}).get("success_rate") or 0.0)
    out["promotion"] = {
        "task": "reinforcement_learning",
        "primary_metric": "success_rate",
        "primary": promote_primary,
        "baseline_name": "random_policy",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary > promote_baseline + 0.05),
    }

    with open(artifacts.rl_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    out["fingerprint"] = {
        "metadata_sha256": _sha256_file(artifacts.rl_metadata_path),
        "q_table_sha256": _sha256_json(
            [[int(k[0]), int(k[1]), int(k[2]), float(v)] for k, v in sorted(q.items(), key=lambda kv: kv[0])]
        ),
        "q_entries": int(len(q)),
    }
    with open(artifacts.rl_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    base_dir = os.path.dirname(__file__)
    artifacts = _default_artifacts(base_dir)

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    cnn = sub.add_parser("cnn")
    cnn.add_argument("--transportation-dir", default=base_dir)
    cnn.add_argument("--window", type=int, default=24)
    cnn.add_argument("--horizon", type=int, default=1)
    cnn.add_argument("--seed", type=int, default=42)
    cnn.add_argument("--epochs", type=int, default=30)
    cnn.add_argument("--batch-size", type=int, default=256)
    cnn.add_argument("--val-frac", type=float, default=0.15)
    cnn.add_argument("--test-frac", type=float, default=0.15)
    cnn.add_argument("--tune-trials", type=int, default=0)
    cnn.add_argument("--save-model", action="store_true")

    rl = sub.add_parser("rl")
    rl.add_argument("--transportation-dir", default=base_dir)
    rl.add_argument("--episodes", type=int, default=5000)
    rl.add_argument("--max-steps", type=int, default=6)
    rl.add_argument("--epsilon", type=float, default=1.0)
    rl.add_argument("--epsilon-decay", type=float, default=0.999)
    rl.add_argument("--min-epsilon", type=float, default=0.05)
    rl.add_argument("--alpha", type=float, default=0.2)
    rl.add_argument("--gamma", type=float, default=0.95)
    rl.add_argument("--congestion-alpha", type=float, default=1.5)
    rl.add_argument("--reroute-penalty-prob", type=float, default=0.3)
    rl.add_argument("--seed", type=int, default=42)
    rl.add_argument("--tune-trials", type=int, default=0)
    rl.add_argument("--tune-episodes", type=int, default=1000)

    argv = sys.argv[1:]
    if not argv or argv[0] not in {"cnn", "rl"}:
        argv = ["cnn", *argv]
    args = p.parse_args(argv)
    cmd = str(args.command or "cnn")

    if cmd == "cnn":
        result = train_cnn_traffic(
            transportation_dir=str(args.transportation_dir),
            window=int(args.window),
            horizon=int(args.horizon),
            seed=int(args.seed),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            val_frac=float(args.val_frac),
            test_frac=float(args.test_frac),
            tune_trials=int(getattr(args, "tune_trials", 0)),
            save_model=bool(args.save_model),
        )
        print(json.dumps(result["metrics"], indent=2))
        print(f"Saved predictions: {os.path.abspath(artifacts.cnn_predictions_path)}")
        print(f"Saved metadata: {os.path.abspath(artifacts.cnn_metadata_path)}")
        if args.save_model:
            print(f"Saved model: {os.path.abspath(artifacts.cnn_model_path)}")
        return

    if cmd == "rl":
        result = train_rl_routing(
            transportation_dir=str(args.transportation_dir),
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            epsilon=float(args.epsilon),
            epsilon_decay=float(args.epsilon_decay),
            min_epsilon=float(args.min_epsilon),
            alpha=float(args.alpha),
            gamma=float(args.gamma),
            congestion_alpha=float(args.congestion_alpha),
            reroute_penalty_prob=float(args.reroute_penalty_prob),
            seed=int(args.seed),
            tune_trials=int(getattr(args, "tune_trials", 0)),
            tune_episodes=int(getattr(args, "tune_episodes", 1000)),
        )
        print(json.dumps(result["evaluation"], indent=2))
        print(f"Saved metadata: {os.path.abspath(artifacts.rl_metadata_path)}")
        return

    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
