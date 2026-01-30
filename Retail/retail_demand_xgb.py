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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    model_path: str
    metadata_path: str
    predictions_path: str


def _default_artifacts() -> Artifacts:
    base_dir = os.path.dirname(__file__)
    return Artifacts(
        model_path=os.path.join(base_dir, "retail_demand_xgb.json"),
        metadata_path=os.path.join(base_dir, "retail_demand_xgb.meta.json"),
        predictions_path=os.path.join(base_dir, "retail_xgb_predictions.csv"),
    )


def _read_csv_any(*paths: str, **kwargs) -> pd.DataFrame:
    for p in paths:
        fp = os.path.abspath(p)
        if os.path.isfile(fp):
            return pd.read_csv(fp, **kwargs)
    raise FileNotFoundError(f"CSV not found in any of: {paths}")


def _extract_datetime_features(series: pd.Series, *, prefix: str) -> pd.DataFrame:
    dt = pd.to_datetime(series, errors="coerce")
    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_year"] = dt.dt.year
    out[f"{prefix}_month"] = dt.dt.month
    out[f"{prefix}_day"] = dt.dt.day
    out[f"{prefix}_dayofweek"] = dt.dt.dayofweek
    out[f"{prefix}_weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out[f"{prefix}_dayofyear"] = dt.dt.dayofyear
    return out


def _build_holiday_features(train_like: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    h = holidays.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["type"] = h["type"].astype("string")
    h["locale"] = h["locale"].astype("string")
    h["locale_name"] = h["locale_name"].astype("string")
    h["transferred"] = h["transferred"].astype("string")

    national = h[h["locale"] == "National"].copy()
    national = national[national["type"] != "Work Day"]
    national_flag = national[["date"]].drop_duplicates().assign(holiday_national=1)

    regional = h[h["locale"] == "Regional"].copy()
    regional = regional[regional["type"] != "Work Day"]
    regional = regional[["date", "locale_name"]].drop_duplicates().rename(columns={"locale_name": "state"})
    regional["holiday_regional"] = 1

    local = h[h["locale"] == "Local"].copy()
    local = local[local["type"] != "Work Day"]
    local = local[["date", "locale_name"]].drop_duplicates().rename(columns={"locale_name": "city"})
    local["holiday_local"] = 1

    transferred = h[h["transferred"] == "True"][["date"]].drop_duplicates().assign(holiday_transferred=1)

    df = train_like.copy()
    df = df.merge(national_flag, on="date", how="left")
    df = df.merge(regional, on=["date", "state"], how="left")
    df = df.merge(local, on=["date", "city"], how="left")
    df = df.merge(transferred, on="date", how="left")
    df[["holiday_national", "holiday_regional", "holiday_local", "holiday_transferred"]] = (
        df[["holiday_national", "holiday_regional", "holiday_local", "holiday_transferred"]].fillna(0).astype(np.int8)
    )
    return df


def _fit_category_maps(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: Tuple[str, ...]) -> Dict[str, list]:
    maps: Dict[str, list] = {}
    for c in cols:
        if c not in train_df.columns and c not in test_df.columns:
            continue
        s = pd.concat(
            [
                train_df[c].astype("string", copy=False) if c in train_df.columns else pd.Series([], dtype="string"),
                test_df[c].astype("string", copy=False) if c in test_df.columns else pd.Series([], dtype="string"),
            ],
            axis=0,
            ignore_index=True,
        ).fillna("missing")
        cats = pd.Index(s.unique()).sort_values()
        maps[c] = list(map(str, cats))
    return maps


def _apply_category_maps(df: pd.DataFrame, maps: Dict[str, list]) -> pd.DataFrame:
    out = df.copy()
    for c, cats in maps.items():
        if c not in out.columns:
            continue
        out[f"{c}_code"] = (
            pd.Categorical(out[c].astype("string").fillna("missing"), categories=cats)
            .codes.astype(np.int32)
        )
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape_all = float(np.mean(np.abs((y_true - y_pred) / denom))) if y_true.size else None
    nonzero = np.abs(y_true) > 1e-3
    mape_nonzero = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / np.maximum(np.abs(y_true[nonzero]), 1e-8)))) if nonzero.any() else None
    smape = None
    if y_true.size:
        smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape_all": mape_all,
        "mape_nonzero": mape_nonzero,
        "smape": smape,
        "r2": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else None,
    }


def _build_features(
    *,
    base: pd.DataFrame,
    stores: pd.DataFrame,
    transactions: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    df = base.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    stores["store_nbr"] = stores["store_nbr"].astype(int)
    df = df.merge(stores, on="store_nbr", how="left")
    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    oil = oil.copy()
    oil["date"] = pd.to_datetime(oil["date"], errors="coerce")
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    oil = oil.sort_values("date").ffill()
    df = df.merge(oil, on="date", how="left")
    df = _build_holiday_features(df, holidays)
    dt_feats = _extract_datetime_features(df["date"], prefix="date")
    df = pd.concat([df, dt_feats], axis=1)
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0).astype(np.int32)
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype(np.float32)
    df["dcoilwtico"] = pd.to_numeric(df["dcoilwtico"], errors="coerce").ffill().fillna(0.0).astype(np.float32)
    if "cluster" in df.columns:
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(-1).astype(np.int32)
    return df


def _train_val_split(df: pd.DataFrame, *, date_col: str, val_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last_date = df[date_col].max()
    cutoff = last_date - pd.Timedelta(days=int(val_days))
    train_df = df[df[date_col] <= cutoff]
    val_df = df[df[date_col] > cutoff]
    return train_df, val_df


def _select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list]:
    cols = [
        "store_nbr",
        "family_code",
        "onpromotion",
        "transactions",
        "dcoilwtico",
        "cluster",
        "city_code",
        "state_code",
        "type_code",
        "date_year",
        "date_month",
        "date_day",
        "date_dayofweek",
        "date_weekofyear",
        "date_dayofyear",
        "holiday_national",
        "holiday_regional",
        "holiday_local",
        "holiday_transferred",
    ]
    present = [c for c in cols if c in df.columns]
    X = df[present].astype(np.float32, copy=False)
    y = df["sales"].to_numpy(dtype=np.float32, copy=False) if "sales" in df.columns else None
    return X, y, present


def _xgb_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse"],
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "tree_method": "hist",
        "seed": int(seed),
    }
    start = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    seconds = float(time.time() - start)
    y_pred_val = booster.predict(dval)
    y_pred_val = np.maximum(y_pred_val, 0.0)
    metrics = _metrics_regression(y_val, y_pred_val)
    metrics["fit_seconds"] = seconds
    metrics["best_iteration"] = int(booster.best_iteration)
    return booster, metrics


def train_and_test(
    *,
    retail_dir: str,
    val_days: int,
    seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    save_model: bool,
) -> Dict[str, Any]:
    retail_dir = os.path.abspath(retail_dir)
    train = _read_csv_any(
        os.path.join(retail_dir, "train.csv", "train.csv"),
        os.path.join(retail_dir, "train.csv"),
        usecols=["id", "date", "store_nbr", "family", "sales", "onpromotion"],
    )
    test = _read_csv_any(
        os.path.join(retail_dir, "test.csv"),
        usecols=["id", "date", "store_nbr", "family", "onpromotion"],
    )
    stores = _read_csv_any(os.path.join(retail_dir, "stores.csv"))
    transactions = _read_csv_any(os.path.join(retail_dir, "transactions.csv", "transactions.csv"), os.path.join(retail_dir, "transactions.csv"))
    oil = _read_csv_any(os.path.join(retail_dir, "oil.csv"))
    holidays = _read_csv_any(os.path.join(retail_dir, "holidays_events.csv"))

    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    test["date"] = pd.to_datetime(test["date"], errors="coerce")
    stores["store_nbr"] = pd.to_numeric(stores["store_nbr"], errors="coerce").astype(int)
    transactions["store_nbr"] = pd.to_numeric(transactions["store_nbr"], errors="coerce").astype(int)

    train_feat = _build_features(base=train[["id", "date", "store_nbr", "family", "sales", "onpromotion"]], stores=stores, transactions=transactions, oil=oil, holidays=holidays)
    test_feat = _build_features(base=test[["id", "date", "store_nbr", "family", "onpromotion"]], stores=stores, transactions=transactions, oil=oil, holidays=holidays)

    cat_maps = _fit_category_maps(train_feat, test_feat, ("family", "city", "state", "type"))
    train_feat = _apply_category_maps(train_feat, cat_maps)
    test_feat = _apply_category_maps(test_feat, cat_maps)

    train_feat = train_feat.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    test_feat = test_feat.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    train_df, val_df = _train_val_split(train_feat, date_col="date", val_days=int(val_days))
    X_train, y_train, _ = _select_feature_columns(train_df)
    X_val, y_val, _ = _select_feature_columns(val_df)
    feat_cols = list(X_train.columns)

    booster, val_metrics = _xgb_train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=int(seed),
        num_boost_round=int(num_boost_round),
        early_stopping_rounds=int(early_stopping_rounds),
    )

    dtest = xgb.DMatrix(test_feat[feat_cols].astype(np.float32, copy=False))
    test_pred = booster.predict(dtest)
    test_pred = np.maximum(test_pred, 0.0)
    predictions = pd.DataFrame({"id": test_feat["id"].astype(int), "sales": test_pred.astype(np.float32)})
    predictions = predictions.sort_values("id").reset_index(drop=True)

    dval = xgb.DMatrix(X_val)
    val_pred = booster.predict(dval)
    val_pred = np.maximum(val_pred, 0.0)

    baseline_sales = train_feat.groupby(["store_nbr", "family"], sort=False)["sales"].shift(1)
    baseline_val = baseline_sales.loc[val_df.index].to_numpy(dtype=np.float32, copy=False)
    baseline_val = np.nan_to_num(baseline_val, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_val = np.maximum(baseline_val, 0.0)
    baseline_metrics = _metrics_regression(y_val, baseline_val)
    val_preview = val_df[["date", "store_nbr", "family", "sales"]].copy()
    val_preview["predicted_sales"] = val_pred.astype(np.float32)
    val_preview = val_preview.sort_values(["date", "store_nbr", "family"], kind="mergesort").tail(20)
    val_preview["date"] = pd.to_datetime(val_preview["date"]).dt.strftime("%Y-%m-%d")

    artifacts = _default_artifacts()
    out: Dict[str, Any] = {
        "retail_dir": retail_dir,
        "val_days": int(val_days),
        "seed": int(seed),
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(dtest.num_row()),
        "feature_columns": feat_cols,
        "val_metrics": val_metrics,
        "baseline_metrics": {"lag1_sales": baseline_metrics},
        "val_preview_tail": val_preview.to_dict(orient="records"),
        "artifacts": {
            "predictions_path": artifacts.predictions_path,
            "model_path": artifacts.model_path if save_model else None,
            "metadata_path": artifacts.metadata_path,
        },
    }

    os.makedirs(os.path.dirname(artifacts.predictions_path), exist_ok=True)
    predictions.to_csv(artifacts.predictions_path, index=False)

    os.makedirs(os.path.dirname(os.path.abspath(artifacts.metadata_path)), exist_ok=True)
    with open(artifacts.metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        booster.save_model(artifacts.model_path)

    out["fingerprint"] = {
        "model_sha256": _sha256_file(artifacts.model_path) if save_model else None,
        "predictions_sha256": _sha256_file(artifacts.predictions_path),
        "canary_pred_sha256": _sha256_array(predictions["sales"].to_numpy(dtype=np.float32, copy=False)[: min(512, int(predictions.shape[0]))]),
    }
    rmse = float(out.get("val_metrics", {}).get("rmse", float("inf")))
    base_rmse = float(out.get("baseline_metrics", {}).get("lag1_sales", {}).get("rmse", float("inf")))
    out["promotion"] = {
        "task": "regression",
        "primary_metric": "rmse",
        "primary": rmse,
        "baseline_name": "lag1_sales",
        "baseline": base_rmse,
        "eligible": bool(rmse < base_rmse * 0.99),
    }
    with open(artifacts.metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def main() -> None:
    artifacts = _default_artifacts()
    retail_dir = os.path.dirname(__file__)
    p = argparse.ArgumentParser()
    p.add_argument("--retail-dir", default=retail_dir)
    p.add_argument("--val-days", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-boost-round", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    result = train_and_test(
        retail_dir=str(args.retail_dir),
        val_days=int(args.val_days),
        seed=int(args.seed),
        num_boost_round=int(args.num_boost_round),
        early_stopping_rounds=int(args.early_stopping_rounds),
        save_model=bool(args.save_model),
    )

    print(json.dumps(result["val_metrics"], indent=2))
    preview = pd.DataFrame(result["val_preview_tail"])
    if not preview.empty:
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(preview.to_string(index=False))
    artifacts_out = result.get("artifacts") or {}
    print(f"Saved predictions: {os.path.abspath(str(artifacts_out.get('predictions_path')))}")
    print(f"Saved metadata: {os.path.abspath(str(artifacts_out.get('metadata_path')))}")
    if args.save_model:
        print(f"Saved model: {os.path.abspath(str(artifacts_out.get('model_path')))}")


if __name__ == "__main__":
    main()

