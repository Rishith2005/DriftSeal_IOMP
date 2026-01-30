import argparse
import ast
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass(frozen=True)
class Artifacts:
    deep_cf_model_path: str
    deep_cf_metadata_path: str
    deep_cf_predictions_path: str
    spotify_gan_generator_path: str
    spotify_gan_metadata_path: str
    spotify_gan_samples_path: str
    nlp_transformer_metadata_path: str
    nlp_transformer_predictions_path: str


def _default_artifacts(base_dir: str) -> Artifacts:
    return Artifacts(
        deep_cf_model_path=os.path.join(base_dir, "deep_cf.keras"),
        deep_cf_metadata_path=os.path.join(base_dir, "deep_cf.meta.json"),
        deep_cf_predictions_path=os.path.join(base_dir, "deep_cf_predictions.csv"),
        spotify_gan_generator_path=os.path.join(base_dir, "spotify_gan_generator.keras"),
        spotify_gan_metadata_path=os.path.join(base_dir, "spotify_gan.meta.json"),
        spotify_gan_samples_path=os.path.join(base_dir, "spotify_gan_samples.csv"),
        nlp_transformer_metadata_path=os.path.join(base_dir, "nlp_transformer.meta.json"),
        nlp_transformer_predictions_path=os.path.join(base_dir, "nlp_transformer_predictions.csv"),
    )


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_movielens_ratings(entertainment_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(entertainment_dir), "MovieLens dataset", "rating.csv")
    df = pd.read_csv(os.path.abspath(path))
    for c in ["userId", "movieId", "rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(np.float32)
    df = df.drop_duplicates(subset=["userId", "movieId", "timestamp"]).reset_index(drop=True)
    df = df.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    return df


def _read_movielens_movies(entertainment_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(entertainment_dir), "MovieLens dataset", "movie.csv")
    df = pd.read_csv(os.path.abspath(path))
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
    df = df.dropna(subset=["movieId", "title", "genres"]).copy()
    df["movieId"] = df["movieId"].astype(int)
    df["title"] = df["title"].astype(str)
    df["genres"] = df["genres"].astype(str)
    df = df.drop_duplicates(subset=["movieId"]).reset_index(drop=True)
    return df


def _read_spotify_tracks(entertainment_dir: str) -> pd.DataFrame:
    path = os.path.join(os.path.abspath(entertainment_dir), "Spotify dataset", "data", "data.csv")
    df = pd.read_csv(os.path.abspath(path))
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


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


def _split_last_two_per_user(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df2 = df.reset_index(drop=True)
    g = df2.groupby("userId", sort=False)
    cnt = g.cumcount()
    size = g["movieId"].transform("size").astype(int)
    test_mask = (cnt == (size - 1)).to_numpy(dtype=bool)
    val_mask = (cnt == (size - 2)).to_numpy(dtype=bool) & (size.to_numpy(dtype=int) >= 3)
    train_mask = ~(test_mask | val_mask)
    return train_mask, val_mask, test_mask


def _build_ncf_model(
    *,
    n_users: int,
    n_items: int,
    embed_dim: int,
    mlp_layers: Sequence[int],
    lr: float,
    dropout: float,
    l2: float,
) -> tf.keras.Model:
    u_in = tf.keras.Input(shape=(), dtype=tf.int32, name="user")
    i_in = tf.keras.Input(shape=(), dtype=tf.int32, name="item")

    reg = tf.keras.regularizers.L2(float(l2)) if float(l2) > 0 else None
    u_emb = tf.keras.layers.Embedding(input_dim=int(n_users), output_dim=int(embed_dim), name="user_emb", embeddings_regularizer=reg)(u_in)
    i_emb = tf.keras.layers.Embedding(input_dim=int(n_items), output_dim=int(embed_dim), name="item_emb", embeddings_regularizer=reg)(i_in)
    u_bias = tf.keras.layers.Embedding(input_dim=int(n_users), output_dim=1, name="user_bias", embeddings_regularizer=reg)(u_in)
    i_bias = tf.keras.layers.Embedding(input_dim=int(n_items), output_dim=1, name="item_bias", embeddings_regularizer=reg)(i_in)
    u_vec = tf.keras.layers.Flatten()(u_emb)
    i_vec = tf.keras.layers.Flatten()(i_emb)
    u_b = tf.keras.layers.Flatten()(u_bias)
    i_b = tf.keras.layers.Flatten()(i_bias)

    gmf = tf.keras.layers.Multiply()([u_vec, i_vec])
    mlp = tf.keras.layers.Concatenate()([u_vec, i_vec])
    for h in mlp_layers:
        mlp = tf.keras.layers.Dense(int(h), activation="relu", kernel_regularizer=reg)(mlp)
        mlp = tf.keras.layers.Dropout(float(dropout))(mlp)

    x = tf.keras.layers.Concatenate()([gmf, mlp])
    out = tf.keras.layers.Dense(1, activation=None)(x)
    out = tf.keras.layers.Add()([out, tf.keras.layers.Reshape((1,))(u_b), tf.keras.layers.Reshape((1,))(i_b)])
    model = tf.keras.Model(inputs=[u_in, i_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def _sample_rank_metrics(
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    model: tf.keras.Model,
    k: int,
    n_neg: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    all_items = np.asarray(sorted(item_map.keys()), dtype=np.int64)
    user_seen: Dict[int, set] = {}
    for uid, mid in df_train[["userId", "movieId"]].itertuples(index=False, name=None):
        user_seen.setdefault(int(uid), set()).add(int(mid))

    hits = 0
    ndcgs = []
    for uid, true_mid in df_test[["userId", "movieId"]].itertuples(index=False, name=None):
        uid = int(uid)
        true_mid = int(true_mid)
        seen = user_seen.get(uid, set())
        candidates = [true_mid]
        tries = 0
        while len(candidates) < (1 + int(n_neg)) and tries < int(n_neg) * 50:
            tries += 1
            mid = int(rng.choice(all_items))
            if mid in seen or mid == true_mid:
                continue
            candidates.append(mid)
        if len(candidates) <= 1:
            continue

        u_idx = np.full((len(candidates),), int(user_map[uid]), dtype=np.int32)
        i_idx = np.asarray([int(item_map[m]) for m in candidates], dtype=np.int32)
        scores = model.predict({"user": u_idx, "item": i_idx}, verbose=0).reshape(-1).astype(np.float64)
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1
        if rank <= int(k):
            hits += 1
            ndcgs.append(1.0 / np.log2(rank + 1.0))
        else:
            ndcgs.append(0.0)

    denom = max(int(len(ndcgs)), 1)
    return {"hit_rate@k": float(hits / denom), "ndcg@k": float(float(np.mean(ndcgs)) if ndcgs else 0.0)}


def _sample_rank_metrics_popularity(
    *,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    k: int,
    n_neg: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    all_items = np.asarray(sorted(df_train["movieId"].unique().tolist()), dtype=np.int64)
    pop_counts = df_train.groupby("movieId", sort=False)["rating"].size().to_dict()
    pop_w = np.asarray([float(pop_counts.get(int(m), 1.0)) for m in all_items], dtype=np.float64)
    pop_w = np.power(pop_w, 0.75)
    pop_w = pop_w / max(float(pop_w.sum()), 1e-12)
    pop = df_train.groupby("movieId", sort=False)["rating"].size().to_dict()
    pop_w = np.asarray([float(pop.get(int(m), 1.0)) for m in all_items], dtype=np.float64)
    pop_w = np.power(pop_w, 0.75)
    pop_w = pop_w / max(float(pop_w.sum()), 1e-12)
    user_seen: Dict[int, set] = {}
    for uid, mid in df_train[["userId", "movieId"]].itertuples(index=False, name=None):
        user_seen.setdefault(int(uid), set()).add(int(mid))

    hits = 0
    ndcgs = []
    for uid, true_mid in df_test[["userId", "movieId"]].itertuples(index=False, name=None):
        uid = int(uid)
        true_mid = int(true_mid)
        seen = user_seen.get(uid, set())
        candidates = [true_mid]
        tries = 0
        while len(candidates) < (1 + int(n_neg)) and tries < int(n_neg) * 50:
            tries += 1
            mid = int(rng.choice(all_items, p=pop_w))
            if mid in seen or mid == true_mid:
                continue
            candidates.append(mid)
        if len(candidates) <= 1:
            continue

        scores = np.asarray([float(pop.get(int(m), 0)) for m in candidates], dtype=np.float64)
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1
        if rank <= int(k):
            hits += 1
            ndcgs.append(1.0 / np.log2(rank + 1.0))
        else:
            ndcgs.append(0.0)

    denom = max(int(len(ndcgs)), 1)
    return {"hit_rate@k": float(hits / denom), "ndcg@k": float(float(np.mean(ndcgs)) if ndcgs else 0.0)}


def train_deep_cf(
    *,
    entertainment_dir: str,
    seed: int,
    max_ratings: int,
    embed_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    k: int,
    n_neg: int,
    save_model: bool,
    tune_trials: int = 0,
    rank_finetune_steps: int = 0,
    rank_lr: float = 5e-5,
) -> Dict[str, Any]:
    _set_seed(int(seed))
    entertainment_dir = os.path.abspath(entertainment_dir)
    artifacts = _default_artifacts(entertainment_dir)

    df = _read_movielens_ratings(entertainment_dir)
    if int(max_ratings) > 0 and int(df.shape[0]) > int(max_ratings):
        df = df.iloc[: int(max_ratings)].reset_index(drop=True)

    counts = df.groupby("userId", sort=False)["movieId"].transform("size").astype(int)
    df = df.loc[counts >= 2].reset_index(drop=True)

    train_mask, val_mask, test_mask = _split_last_two_per_user(df)
    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)
    df_test = df.loc[test_mask].reset_index(drop=True)

    users = pd.Index(pd.concat([df_train["userId"], df_val["userId"], df_test["userId"]], axis=0).unique())
    items = pd.Index(pd.concat([df_train["movieId"], df_val["movieId"], df_test["movieId"]], axis=0).unique())
    user_map = {int(u): int(i) for i, u in enumerate(users.tolist())}
    item_map = {int(m): int(i) for i, m in enumerate(items.tolist())}

    def _map_ui(df_part: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = df_part["userId"].map(user_map).to_numpy(dtype=np.int32)
        it = df_part["movieId"].map(item_map).to_numpy(dtype=np.int32)
        y = df_part["rating"].to_numpy(dtype=np.float32)
        return u, it, y

    u_tr, i_tr, y_tr = _map_ui(df_train)
    u_val, i_val, y_val = _map_ui(df_val) if df_val.shape[0] else (np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32))
    u_te, i_te, y_te = _map_ui(df_test)

    user_mean = df_train.groupby("userId", sort=False)["rating"].mean().to_dict()
    global_mean = float(df_train["rating"].mean()) if int(df_train.shape[0]) else 3.0
    base_tr = np.asarray([float(user_mean.get(int(u), global_mean)) for u in df_train["userId"].to_numpy(dtype=int)], dtype=np.float32)
    base_val = np.asarray([float(user_mean.get(int(u), global_mean)) for u in df_val["userId"].to_numpy(dtype=int)], dtype=np.float32) if int(df_val.shape[0]) else np.zeros((0,), dtype=np.float32)
    base_te = np.asarray([float(user_mean.get(int(u), global_mean)) for u in df_test["userId"].to_numpy(dtype=int)], dtype=np.float32)

    y_tr_res = (y_tr.astype(np.float32) - base_tr).astype(np.float32)
    y_val_res = (y_val.astype(np.float32) - base_val).astype(np.float32) if int(df_val.shape[0]) else y_val

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, min_lr=1e-5),
    ]

    trials: List[Dict[str, Any]] = []
    best = None
    best_model = None
    best_history = None
    best_fit = 0.0
    best_rank_val = None

    def _fit_one(cfg: Dict[str, Any]) -> Tuple[tf.keras.Model, Any, float, float, Optional[Dict[str, float]]]:
        model = _build_ncf_model(
            n_users=int(len(user_map)),
            n_items=int(len(item_map)),
            embed_dim=int(cfg["embed_dim"]),
            mlp_layers=tuple(cfg["mlp_layers"]),
            lr=float(cfg["lr"]),
            dropout=float(cfg["dropout"]),
            l2=float(cfg["l2"]),
        )
        start = time.time()
        history = model.fit(
            x={"user": u_tr, "item": i_tr},
            y=y_tr_res,
            validation_data=({"user": u_val, "item": i_val}, y_val_res) if int(df_val.shape[0]) else None,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=0 if int(tune_trials) > 0 else 2,
            callbacks=callbacks if int(df_val.shape[0]) else None,
        )
        fit_seconds = float(time.time() - start)
        best_val = float(np.min(np.asarray(getattr(history, "history", {}).get("val_loss", [np.inf]), dtype=np.float64))) if int(df_val.shape[0]) else float("inf")
        rank_val = None
        if int(df_val.shape[0]):
            rank_val = _sample_rank_metrics(
                df_train=df_train,
                df_test=df_val,
                user_map=user_map,
                item_map=item_map,
                model=model,
                k=int(k),
                n_neg=min(int(n_neg), 80),
                seed=int(seed),
            )
        return model, history, fit_seconds, best_val, rank_val

    if int(tune_trials) > 0 and int(df_val.shape[0]):
        rng = np.random.default_rng(int(seed))
        for _ in range(int(tune_trials)):
            mlp = rng.choice([[64, 32, 16], [128, 64, 32], [64, 64, 32]])
            cfg = {
                "embed_dim": int(rng.choice([32, 48, 64, int(embed_dim)])),
                "lr": float(rng.choice([1e-3, 5e-4, 2e-4, float(lr)])),
                "dropout": float(rng.choice([0.05, 0.1, 0.2])),
                "l2": float(rng.choice([0.0, 1e-6, 1e-5])),
                "mlp_layers": [int(x) for x in list(mlp)],
            }
            model, history, fit_s, best_val, rank_val = _fit_one(cfg)
            t = {**cfg, "fit_seconds": fit_s, "best_val_loss": best_val, "val_rank": rank_val}
            trials.append(t)
            score = float(rank_val["ndcg@k"]) if rank_val is not None else -float(best_val)
            if best is None or score > float(best_rank_val if best_rank_val is not None else -1e18):
                best = t
                best_model = model
                best_history = history
                best_fit = fit_s
                best_rank_val = score

        model = best_model
        history = best_history
        fit_seconds = float(best_fit)
        selected_cfg = {k: best[k] for k in ("embed_dim", "lr", "dropout", "l2", "mlp_layers")}
    else:
        selected_cfg = {"embed_dim": int(embed_dim), "lr": float(lr), "dropout": 0.1, "l2": 0.0, "mlp_layers": [64, 32, 16]}
        model, history, fit_seconds, _, _ = _fit_one(selected_cfg)

    if int(rank_finetune_steps) > 0:
        rng = np.random.default_rng(int(seed) + 7)
        seen = df_train.groupby("userId", sort=False)["movieId"].apply(lambda s: set(map(int, s.tolist()))).to_dict()
        pop_counts = df_train.groupby("movieId", sort=False)["rating"].size().sort_values(ascending=False)
        pop_k = int(min(5000, int(pop_counts.shape[0])))
        pop_items = pop_counts.index.to_numpy(dtype=np.int64)[:pop_k]
        pop_w = pop_counts.to_numpy(dtype=np.float64)[:pop_k]
        pop_w = np.power(pop_w, 0.75)
        pop_w = pop_w / max(float(pop_w.sum()), 1e-12)

        opt = tf.keras.optimizers.Adam(learning_rate=float(rank_lr))

        u_tr_ids = df_train["userId"].to_numpy(dtype=int)
        i_tr_ids = df_train["movieId"].to_numpy(dtype=int)
        rank_batch = 256

        for _ in range(int(rank_finetune_steps)):
            idxs = rng.integers(0, max(1, len(u_tr_ids)), size=int(rank_batch), dtype=np.int64)
            u_batch = u_tr_ids[idxs]
            pos_batch = i_tr_ids[idxs]
            neg_batch = np.zeros((int(rank_batch),), dtype=np.int64)

            for j in range(int(rank_batch)):
                uid = int(u_batch[j])
                pos_mid = int(pos_batch[j])
                s = seen.get(uid, set())
                tries = 0
                neg_mid = None
                while tries < 50:
                    tries += 1
                    cand = int(rng.choice(pop_items, p=pop_w))
                    if cand == pos_mid or cand in s:
                        continue
                    neg_mid = cand
                    break
                if neg_mid is None:
                    neg_mid = int(rng.choice(pop_items, p=pop_w))
                neg_batch[j] = int(neg_mid)

            u_idx = tf.convert_to_tensor(np.asarray([int(user_map[int(u)]) for u in u_batch], dtype=np.int32), dtype=tf.int32)
            pos_i = tf.convert_to_tensor(np.asarray([int(item_map[int(m)]) for m in pos_batch], dtype=np.int32), dtype=tf.int32)
            neg_i = tf.convert_to_tensor(np.asarray([int(item_map[int(m)]) for m in neg_batch], dtype=np.int32), dtype=tf.int32)

            with tf.GradientTape() as tape:
                pos = tf.reshape(model({"user": u_idx, "item": pos_i}, training=True), (-1,))
                neg = tf.reshape(model({"user": u_idx, "item": neg_i}, training=True), (-1,))
                diff = pos - neg
                loss = tf.reduce_mean(tf.nn.softplus(-diff))
                if model.losses:
                    loss = loss + tf.add_n(model.losses)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

    y_pred_res = model.predict({"user": u_te, "item": i_te}, verbose=0).reshape(-1).astype(np.float64)
    y_pred = (base_te.astype(np.float64) + y_pred_res).astype(np.float64)
    y_pred = np.clip(y_pred, 0.5, 5.0)

    metrics = _metrics_regression(y_te.astype(np.float64), y_pred)

    base_user_mean = np.clip(base_te.astype(np.float64), 0.5, 5.0)
    baseline_metrics = {"user_mean": _metrics_regression(y_te.astype(np.float64), base_user_mean)}
    rank_metrics = _sample_rank_metrics(
        df_train=df_train,
        df_test=df_test,
        user_map=user_map,
        item_map=item_map,
        model=model,
        k=int(k),
        n_neg=int(n_neg),
        seed=int(seed),
    )
    rank_baseline_pop = _sample_rank_metrics_popularity(
        df_train=df_train,
        df_test=df_test,
        k=int(k),
        n_neg=int(n_neg),
        seed=int(seed),
    )

    pred_df = pd.DataFrame(
        {
            "userId": df_test["userId"].astype(int),
            "movieId": df_test["movieId"].astype(int),
            "rating_true": y_te.astype(np.float32),
            "rating_baseline_user_mean": base_user_mean.astype(np.float32),
            "rating_pred": y_pred.astype(np.float32),
        }
    )
    pred_df.to_csv(artifacts.deep_cf_predictions_path, index=False)

    out: Dict[str, Any] = {
        "entertainment_dir": entertainment_dir,
        "dataset": {"ratings_csv": os.path.join(entertainment_dir, "MovieLens dataset", "rating.csv")},
        "seed": int(seed),
        "cleaning": {
            "after_drop_invalid_rows": int(df.shape[0]),
            "n_users": int(df["userId"].nunique()),
            "n_items": int(df["movieId"].nunique()),
        },
        "split": {"n_train": int(df_train.shape[0]), "n_val": int(df_val.shape[0]), "n_test": int(df_test.shape[0])},
        "model": {
            "embed_dim": int(selected_cfg["embed_dim"]),
            "lr": float(selected_cfg["lr"]),
            "dropout": float(selected_cfg["dropout"]),
            "l2": float(selected_cfg["l2"]),
            "mlp_layers": list(map(int, selected_cfg["mlp_layers"])),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
        },
        "fit_seconds": float(fit_seconds),
        "tuning": {"trials": trials, "best": best} if int(tune_trials) > 0 else None,
        "history": {k: [float(x) for x in v] for k, v in getattr(history, "history", {}).items()},
        "test_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "test_ranking_metrics": {f"hit_rate@{int(k)}": float(rank_metrics["hit_rate@k"]), f"ndcg@{int(k)}": float(rank_metrics["ndcg@k"])},
        "baseline_ranking_metrics": {
            f"popularity_hit_rate@{int(k)}": float(rank_baseline_pop["hit_rate@k"]),
            f"popularity_ndcg@{int(k)}": float(rank_baseline_pop["ndcg@k"]),
        },
        "rank_finetune": {"steps": int(rank_finetune_steps), "lr": float(rank_lr)} if int(rank_finetune_steps) > 0 else None,
        "artifacts": {
            "model_path": artifacts.deep_cf_model_path if save_model else None,
            "metadata_path": artifacts.deep_cf_metadata_path,
            "predictions_path": artifacts.deep_cf_predictions_path,
        },
    }

    promote_primary = float(metrics.get("rmse", 0.0))
    promote_baseline = float(baseline_metrics.get("user_mean", {}).get("rmse", 0.0))
    out["promotion"] = {
        "task": "regression",
        "primary_metric": "rmse",
        "primary": promote_primary,
        "baseline_name": "user_mean",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary < promote_baseline * 0.99),
    }

    with open(artifacts.deep_cf_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if bool(save_model):
        model.save(artifacts.deep_cf_model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.deep_cf_model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.deep_cf_predictions_path),
        "canary_pred_sha256": _sha256_array(y_pred[: min(256, int(y_pred.shape[0]))]),
    }
    with open(artifacts.deep_cf_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


class _SpotifyGAN:
    def __init__(self, *, feature_dim: int, latent_dim: int, lr: float, corr_lambda: float) -> None:
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.corr_lambda = float(corr_lambda)

        self.gen = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.feature_dim, activation=None),
            ]
        )
        self.disc = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.feature_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.opt_g = tf.keras.optimizers.Adam(learning_rate=float(lr), beta_1=0.5)
        self.opt_d = tf.keras.optimizers.Adam(learning_rate=float(lr), beta_1=0.5)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function
    def train_step(self, real_x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        batch = tf.shape(real_x)[0]
        z = tf.random.normal((batch, self.latent_dim))
        with tf.GradientTape() as d_tape:
            fake_x = self.gen(z, training=True)
            y_real = tf.ones((batch, 1))
            y_fake = tf.zeros((batch, 1))
            p_real = self.disc(real_x, training=True)
            p_fake = self.disc(fake_x, training=True)
            d_loss = self.bce(y_real, p_real) + self.bce(y_fake, p_fake)
        d_grads = d_tape.gradient(d_loss, self.disc.trainable_variables)
        self.opt_d.apply_gradients(zip(d_grads, self.disc.trainable_variables))

        z2 = tf.random.normal((batch, self.latent_dim))
        with tf.GradientTape() as g_tape:
            fake_x2 = self.gen(z2, training=True)
            p_fake2 = self.disc(fake_x2, training=True)
            g_loss = self.bce(tf.ones((batch, 1)), p_fake2)

            if self.corr_lambda > 0.0:
                eps = tf.constant(1e-8, dtype=fake_x2.dtype)
                real = tf.cast(real_x, fake_x2.dtype)
                real = real - tf.reduce_mean(real, axis=0, keepdims=True)
                fake = fake_x2 - tf.reduce_mean(fake_x2, axis=0, keepdims=True)

                denom = tf.cast(tf.maximum(batch - 1, 1), fake_x2.dtype)
                cov_real = tf.matmul(real, real, transpose_a=True) / denom
                cov_fake = tf.matmul(fake, fake, transpose_a=True) / denom

                std_real = tf.sqrt(tf.maximum(tf.linalg.diag_part(cov_real), eps))
                std_fake = tf.sqrt(tf.maximum(tf.linalg.diag_part(cov_fake), eps))

                corr_real = cov_real / (tf.tensordot(std_real, std_real, axes=0) + eps)
                corr_fake = cov_fake / (tf.tensordot(std_fake, std_fake, axes=0) + eps)
                corr_real = tf.stop_gradient(corr_real)
                diff = corr_real - corr_fake
                corr_pen = tf.sqrt(tf.reduce_sum(tf.square(diff)))
                g_loss = g_loss + tf.cast(self.corr_lambda, fake_x2.dtype) * corr_pen
        g_grads = g_tape.gradient(g_loss, self.gen.trainable_variables)
        self.opt_g.apply_gradients(zip(g_grads, self.gen.trainable_variables))
        return d_loss, g_loss


def _spotify_numeric_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    numeric_cols = [
        "valence",
        "year",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "explicit",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "mode",
        "popularity",
        "speechiness",
        "tempo",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna().reset_index(drop=True)
    return x, cols


def train_spotify_gan(
    *,
    entertainment_dir: str,
    seed: int,
    max_rows: int,
    latent_dim: int,
    batch_size: int,
    steps: int,
    lr: float,
    n_samples: int,
    save_model: bool,
    corr_lambda: float = 0.0,
) -> Dict[str, Any]:
    _set_seed(int(seed))
    entertainment_dir = os.path.abspath(entertainment_dir)
    artifacts = _default_artifacts(entertainment_dir)

    df = _read_spotify_tracks(entertainment_dir)
    if int(max_rows) > 0 and int(df.shape[0]) > int(max_rows):
        df = df.iloc[: int(max_rows)].reset_index(drop=True)

    x_df, cols = _spotify_numeric_frame(df)
    scaler = StandardScaler()
    x = scaler.fit_transform(x_df.to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)

    gan = _SpotifyGAN(feature_dim=int(x.shape[1]), latent_dim=int(latent_dim), lr=float(lr), corr_lambda=float(corr_lambda))
    ds = tf.data.Dataset.from_tensor_slices(x).shuffle(min(int(x.shape[0]), 50_000), seed=int(seed), reshuffle_each_iteration=True).batch(
        int(batch_size), drop_remainder=True
    )
    it = iter(ds.repeat())

    d_losses = []
    g_losses = []
    start = time.time()
    for _ in range(int(steps)):
        real_x = next(it)
        d_loss, g_loss = gan.train_step(real_x)
        d_losses.append(float(d_loss.numpy()))
        g_losses.append(float(g_loss.numpy()))
    fit_seconds = float(time.time() - start)

    z = tf.random.normal((int(n_samples), int(latent_dim)))
    fake = gan.gen(z, training=False).numpy().astype(np.float32, copy=False)
    fake_inv = scaler.inverse_transform(fake.astype(np.float64)).astype(np.float64)
    real_inv = x_df.to_numpy(dtype=np.float64, copy=False)

    real_mean = real_inv.mean(axis=0)
    real_std = real_inv.std(axis=0, ddof=0)
    fake_mean = fake_inv.mean(axis=0)
    fake_std = fake_inv.std(axis=0, ddof=0)
    mean_abs_diff = float(np.mean(np.abs(real_mean - fake_mean)))
    std_abs_diff = float(np.mean(np.abs(real_std - fake_std)))
    rel_mean_abs_diff = float(np.mean(np.abs(real_mean - fake_mean) / (real_std + 1e-8)))
    rel_std_abs_diff = float(np.mean(np.abs(real_std - fake_std) / (real_std + 1e-8)))

    real_scaled = x.astype(np.float64, copy=False)
    fake_scaled = fake.astype(np.float64, copy=False)
    scaled_mean_abs_diff = float(np.mean(np.abs(real_scaled.mean(axis=0) - fake_scaled.mean(axis=0))))
    scaled_std_abs_diff = float(np.mean(np.abs(real_scaled.std(axis=0, ddof=0) - fake_scaled.std(axis=0, ddof=0))))

    real_corr = np.corrcoef(real_inv.T)
    fake_corr = np.corrcoef(fake_inv.T)
    corr_diff = float(np.linalg.norm(real_corr - fake_corr))

    rng = np.random.default_rng(int(seed) + 123)
    baseline_fake_inv = rng.normal(loc=real_mean.reshape(1, -1), scale=(real_std + 1e-8).reshape(1, -1), size=fake_inv.shape).astype(np.float64, copy=False)
    base_mean = baseline_fake_inv.mean(axis=0)
    base_std = baseline_fake_inv.std(axis=0, ddof=0)
    base_mean_abs_diff = float(np.mean(np.abs(real_mean - base_mean)))
    base_std_abs_diff = float(np.mean(np.abs(real_std - base_std)))
    base_rel_mean_abs_diff = float(np.mean(np.abs(real_mean - base_mean) / (real_std + 1e-8)))
    base_rel_std_abs_diff = float(np.mean(np.abs(real_std - base_std) / (real_std + 1e-8)))
    base_corr = np.corrcoef(baseline_fake_inv.T)
    base_corr_diff = float(np.linalg.norm(real_corr - base_corr))

    samples_df = pd.DataFrame(fake_inv, columns=cols)
    samples_df.to_csv(artifacts.spotify_gan_samples_path, index=False)

    out: Dict[str, Any] = {
        "entertainment_dir": entertainment_dir,
        "dataset": {"spotify_data_csv": os.path.join(entertainment_dir, "Spotify dataset", "data", "data.csv")},
        "seed": int(seed),
        "cleaning": {"after_drop_invalid_rows": int(x_df.shape[0]), "columns_used": cols},
        "model": {"latent_dim": int(latent_dim), "lr": float(lr), "batch_size": int(batch_size), "steps": int(steps), "corr_lambda": float(corr_lambda)},
        "fit_seconds": float(fit_seconds),
        "train_losses": {"disc": d_losses[-50:], "gen": g_losses[-50:]},
        "sample_metrics": {
            "mean_abs_diff": mean_abs_diff,
            "std_abs_diff": std_abs_diff,
            "rel_mean_abs_diff": rel_mean_abs_diff,
            "rel_std_abs_diff": rel_std_abs_diff,
            "scaled_mean_abs_diff": scaled_mean_abs_diff,
            "scaled_std_abs_diff": scaled_std_abs_diff,
            "corr_fro_norm_diff": corr_diff,
        },
        "baseline_metrics": {
            "gaussian_independent": {
                "mean_abs_diff": base_mean_abs_diff,
                "std_abs_diff": base_std_abs_diff,
                "rel_mean_abs_diff": base_rel_mean_abs_diff,
                "rel_std_abs_diff": base_rel_std_abs_diff,
                "corr_fro_norm_diff": base_corr_diff,
            }
        },
        "artifacts": {
            "generator_path": artifacts.spotify_gan_generator_path if save_model else None,
            "metadata_path": artifacts.spotify_gan_metadata_path,
            "samples_path": artifacts.spotify_gan_samples_path,
        },
    }

    promote_primary = float(out.get("sample_metrics", {}).get("corr_fro_norm_diff") or 0.0)
    promote_baseline = float(out.get("baseline_metrics", {}).get("gaussian_independent", {}).get("corr_fro_norm_diff") or 0.0)
    out["promotion"] = {
        "task": "tabular_generation",
        "primary_metric": "corr_fro_norm_diff",
        "primary": promote_primary,
        "baseline_name": "gaussian_independent",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary < promote_baseline * 0.98),
    }

    with open(artifacts.spotify_gan_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if bool(save_model):
        gan.gen.save(artifacts.spotify_gan_generator_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.spotify_gan_generator_path) if save_model else None,
        "samples_sha256": _sha256_file(artifacts.spotify_gan_samples_path),
        "canary_samples_sha256": _sha256_array(fake_inv[: min(256, int(fake_inv.shape[0]))].astype(np.float32, copy=False)),
    }
    with open(artifacts.spotify_gan_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


class _GenreTitleDataset(Dataset):
    def __init__(self, titles: Sequence[str], labels: np.ndarray, tokenizer: Any, max_length: int):
        self.titles = list(titles)
        self.labels = labels.astype(np.float32, copy=False)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return int(len(self.titles))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t = str(self.titles[int(idx)])
        enc = self.tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=int(self.max_length),
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.from_numpy(self.labels[int(idx)])
        return item


def _clean_movielens_title(title: str) -> str:
    t = str(title).strip()
    if t.endswith(")") and "(" in t:
        left = t.rfind("(")
        if left >= 0:
            yr = t[left + 1 : -1]
            if yr.isdigit() and len(yr) == 4:
                t = t[:left].strip()
    return t


def train_transformer_nlp(
    *,
    entertainment_dir: str,
    seed: int,
    model_name: str,
    max_movies: int,
    max_length: int,
    batch_size: int,
    epochs: int,
    lr: float,
    threshold: float,
    val_frac: float = 0.1,
    weight_decay: float = 0.01,
    threshold_mode: str = "per_label",
) -> Dict[str, Any]:
    _set_seed(int(seed))
    entertainment_dir = os.path.abspath(entertainment_dir)
    artifacts = _default_artifacts(entertainment_dir)

    movies = _read_movielens_movies(entertainment_dir)
    movies["title_clean"] = movies["title"].map(_clean_movielens_title)
    movies = movies.loc[movies["genres"].astype(str).str.lower().ne("(no genres listed)")].reset_index(drop=True)
    if int(max_movies) > 0 and int(movies.shape[0]) > int(max_movies):
        movies = movies.iloc[: int(max_movies)].reset_index(drop=True)

    genres = sorted({g for gs in movies["genres"].astype(str).tolist() for g in gs.split("|") if g.strip()})
    genre_to_idx = {g: int(i) for i, g in enumerate(genres)}

    y = np.zeros((int(movies.shape[0]), int(len(genres))), dtype=np.float32)
    for i, gs in enumerate(movies["genres"].astype(str).tolist()):
        for g in gs.split("|"):
            g = g.strip()
            if g in genre_to_idx:
                y[int(i), int(genre_to_idx[g])] = 1.0

    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(movies.shape[0]), dtype=np.int64)
    rng.shuffle(idx)
    n = int(idx.size)
    n_test = max(int(np.floor(float(n) * 0.2)), 1)
    n_val = max(int(np.floor(float(n) * float(val_frac))), 1)
    n_train = max(n - n_val - n_test, 1)
    tr_idx = idx[:n_train]
    va_idx = idx[n_train : n_train + n_val]
    te_idx = idx[n_train + n_val :]

    titles_tr = movies.loc[tr_idx, "title_clean"].astype(str).tolist()
    titles_va = movies.loc[va_idx, "title_clean"].astype(str).tolist()
    titles_te = movies.loc[te_idx, "title_clean"].astype(str).tolist()
    y_tr = y[tr_idx]
    y_va = y[va_idx]
    y_te = y[te_idx]

    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_name),
        num_labels=int(len(genres)),
        problem_type="multi_label_classification",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = _GenreTitleDataset(titles_tr, y_tr, tokenizer, int(max_length))
    val_ds = _GenreTitleDataset(titles_va, y_va, tokenizer, int(max_length))
    test_ds = _GenreTitleDataset(titles_te, y_te, tokenizer, int(max_length))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    pos = np.asarray(y_tr.sum(axis=0), dtype=np.float64)
    neg = float(len(y_tr)) - pos
    pos = np.maximum(pos, 1.0)
    neg = np.maximum(neg, 1.0)
    pos_weight = (neg / pos).astype(np.float32)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    total_steps = int(max(1, int(epochs) * max(1, len(train_loader))))
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    start = time.time()
    for _ in range(int(epochs)):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits
            loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
    fit_seconds = float(time.time() - start)

    def _predict_probs(loader: DataLoader) -> np.ndarray:
        model.eval()
        logits_all = []
        with torch.no_grad():
            for batch in loader:
                _ = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                logits_all.append(out.logits.cpu().numpy())
        logits = np.concatenate(logits_all, axis=0).astype(np.float64, copy=False)
        return 1.0 / (1.0 + np.exp(-logits))

    probs_va = _predict_probs(val_loader)
    probs_te = _predict_probs(test_loader)
    y_true_va = y_va.astype(np.int32, copy=False)
    y_true = y_te.astype(np.int32, copy=False)

    mode = str(threshold_mode).lower().strip()
    if mode == "per_label":
        grid = np.linspace(0.05, 0.95, 19, dtype=np.float64)
        thr = np.full((int(len(genres)),), float(threshold), dtype=np.float64)
        for j in range(int(len(genres))):
            best_f1 = -1.0
            best_t = float(threshold)
            yt = y_true_va[:, j]
            if int(yt.sum()) == 0:
                thr[j] = 0.99
                continue
            for t in grid:
                yp = (probs_va[:, j] >= float(t)).astype(np.int32)
                f1 = float(f1_score(yt, yp, zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            thr[j] = best_t
        y_pred = (probs_te >= thr.reshape(1, -1)).astype(np.int32)
        thresholds_used = thr.astype(np.float32).tolist()
    else:
        y_pred = (probs_te >= float(threshold)).astype(np.int32)
        thresholds_used = float(threshold)

    probs = probs_te

    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    subset_acc = float(np.mean(np.all(y_true == y_pred, axis=1)))
    avg_true_labels = float(np.mean(y_true.sum(axis=1)))
    avg_pred_labels = float(np.mean(y_pred.sum(axis=1)))

    train_prev = y_tr.mean(axis=0).astype(np.float64, copy=False)
    base_probs_va = np.broadcast_to(train_prev.reshape(1, -1), (int(y_true_va.shape[0]), int(train_prev.shape[0]))).copy()
    base_probs_te = np.broadcast_to(train_prev.reshape(1, -1), (int(y_true.shape[0]), int(train_prev.shape[0]))).copy()

    grid = np.linspace(0.05, 0.95, 19, dtype=np.float64)
    base_thr = np.full((int(len(genres)),), 0.5, dtype=np.float64)
    for j in range(int(len(genres))):
        best_f1 = -1.0
        best_t = 0.5
        yt = y_true_va[:, j]
        if int(yt.sum()) == 0:
            base_thr[j] = 0.99
            continue
        for t in grid:
            yp = (base_probs_va[:, j] >= float(t)).astype(np.int32)
            f1 = float(f1_score(yt, yp, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        base_thr[j] = best_t

    base_pred = (base_probs_te >= base_thr.reshape(1, -1)).astype(np.int32)
    baseline_metrics = {
        "micro_f1": float(f1_score(y_true, base_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, base_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, base_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, base_pred, average="macro", zero_division=0)),
        "subset_accuracy": float(np.mean(np.all(y_true == base_pred, axis=1))),
        "threshold": base_thr.astype(np.float32).tolist(),
        "threshold_mode": "per_label_prevalence",
    }

    pred_df = pd.DataFrame({"title": titles_te})
    for j, g in enumerate(genres):
        pred_df[f"true_{g}"] = y_true[:, j].astype(int)
        pred_df[f"pred_{g}"] = y_pred[:, j].astype(int)
        pred_df[f"prob_{g}"] = probs[:, j].astype(np.float32)
    pred_df.to_csv(artifacts.nlp_transformer_predictions_path, index=False)

    out: Dict[str, Any] = {
        "entertainment_dir": entertainment_dir,
        "dataset": {"movies_csv": os.path.join(entertainment_dir, "MovieLens dataset", "movie.csv")},
        "seed": int(seed),
        "cleaning": {"after_drop_no_genres": int(movies.shape[0]), "n_genres": int(len(genres))},
        "split": {"n_train": int(len(titles_tr)), "n_val": int(len(titles_va)), "n_test": int(len(titles_te))},
        "model": {
            "model_name": str(model_name),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "pos_weight": pos_weight.astype(np.float32).tolist(),
        },
        "fit_seconds": float(fit_seconds),
        "test_metrics": {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "subset_accuracy": subset_acc,
            "threshold": thresholds_used,
            "threshold_mode": str(threshold_mode),
        },
        "baseline_metrics": {"prevalence": baseline_metrics},
        "label_stats": {"avg_true_labels": avg_true_labels, "avg_pred_labels": avg_pred_labels},
        "artifacts": {"metadata_path": artifacts.nlp_transformer_metadata_path, "predictions_path": artifacts.nlp_transformer_predictions_path},
    }

    promote_primary = float(micro_f1)
    promote_baseline = float(baseline_metrics.get("micro_f1", 0.0))
    out["promotion"] = {
        "task": "multilabel_classification",
        "primary_metric": "micro_f1",
        "primary": promote_primary,
        "baseline_name": "prevalence",
        "baseline": promote_baseline,
        "eligible": bool(promote_primary > promote_baseline + 0.01),
    }

    with open(artifacts.nlp_transformer_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    out["fingerprint"] = {
        "predictions_sha256": _sha256_file(artifacts.nlp_transformer_predictions_path),
        "canary_pred_sha256": _sha256_array(probs_te[: min(256, int(probs_te.shape[0]))].astype(np.float32, copy=False)),
    }
    with open(artifacts.nlp_transformer_metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    base_dir = os.path.dirname(__file__)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    cf = sub.add_parser("deep_cf")
    cf.add_argument("--entertainment-dir", default=base_dir)
    cf.add_argument("--seed", type=int, default=42)
    cf.add_argument("--max-ratings", type=int, default=200_000)
    cf.add_argument("--embed-dim", type=int, default=32)
    cf.add_argument("--lr", type=float, default=1e-3)
    cf.add_argument("--batch-size", type=int, default=4096)
    cf.add_argument("--epochs", type=int, default=8)
    cf.add_argument("--k", type=int, default=10)
    cf.add_argument("--n-neg", type=int, default=99)
    cf.add_argument("--tune-trials", type=int, default=0)
    cf.add_argument("--rank-finetune-steps", type=int, default=0)
    cf.add_argument("--rank-lr", type=float, default=5e-5)
    cf.add_argument("--save-model", action="store_true")

    gan = sub.add_parser("gan_spotify")
    gan.add_argument("--entertainment-dir", default=base_dir)
    gan.add_argument("--seed", type=int, default=42)
    gan.add_argument("--max-rows", type=int, default=50_000)
    gan.add_argument("--latent-dim", type=int, default=32)
    gan.add_argument("--batch-size", type=int, default=512)
    gan.add_argument("--steps", type=int, default=800)
    gan.add_argument("--lr", type=float, default=2e-4)
    gan.add_argument("--n-samples", type=int, default=5000)
    gan.add_argument("--corr-lambda", type=float, default=0.0)
    gan.add_argument("--save-model", action="store_true")

    nlp = sub.add_parser("nlp_transformer")
    nlp.add_argument("--entertainment-dir", default=base_dir)
    nlp.add_argument("--seed", type=int, default=42)
    nlp.add_argument("--model-name", default="distilbert-base-uncased")
    nlp.add_argument("--max-movies", type=int, default=8000)
    nlp.add_argument("--max-length", type=int, default=32)
    nlp.add_argument("--batch-size", type=int, default=32)
    nlp.add_argument("--epochs", type=int, default=3)
    nlp.add_argument("--lr", type=float, default=2e-5)
    nlp.add_argument("--weight-decay", type=float, default=0.01)
    nlp.add_argument("--val-frac", type=float, default=0.1)
    nlp.add_argument("--threshold", type=float, default=0.5)
    nlp.add_argument("--threshold-mode", default="per_label")

    args = p.parse_args()
    cmd = args.command
    if not cmd:
        raise ValueError("Expected a command: deep_cf | gan_spotify | nlp_transformer")

    if cmd == "deep_cf":
        result = train_deep_cf(
            entertainment_dir=str(args.entertainment_dir),
            seed=int(args.seed),
            max_ratings=int(args.max_ratings),
            embed_dim=int(args.embed_dim),
            lr=float(args.lr),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            k=int(args.k),
            n_neg=int(args.n_neg),
            tune_trials=int(getattr(args, "tune_trials", 0)),
            rank_finetune_steps=int(getattr(args, "rank_finetune_steps", 0)),
            rank_lr=float(getattr(args, "rank_lr", 5e-5)),
            save_model=bool(args.save_model),
        )
        print(json.dumps({"test_metrics": result["test_metrics"], "ranking": result["test_ranking_metrics"]}, indent=2))
        return

    if cmd == "gan_spotify":
        result = train_spotify_gan(
            entertainment_dir=str(args.entertainment_dir),
            seed=int(args.seed),
            max_rows=int(args.max_rows),
            latent_dim=int(args.latent_dim),
            batch_size=int(args.batch_size),
            steps=int(args.steps),
            lr=float(args.lr),
            n_samples=int(args.n_samples),
            corr_lambda=float(getattr(args, "corr_lambda", 0.0)),
            save_model=bool(args.save_model),
        )
        print(json.dumps({"sample_metrics": result["sample_metrics"]}, indent=2))
        return

    if cmd == "nlp_transformer":
        result = train_transformer_nlp(
            entertainment_dir=str(args.entertainment_dir),
            seed=int(args.seed),
            model_name=str(args.model_name),
            max_movies=int(args.max_movies),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            threshold=float(args.threshold),
            val_frac=float(getattr(args, "val_frac", 0.1)),
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
            threshold_mode=str(getattr(args, "threshold_mode", "per_label")),
        )
        print(json.dumps({"test_metrics": result["test_metrics"]}, indent=2))
        return

    raise ValueError(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
