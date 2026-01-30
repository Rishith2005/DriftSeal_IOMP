import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def _sha256_file(path: str):
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


def _default_artifacts() -> Artifacts:
    base_dir = os.path.dirname(__file__)
    return Artifacts(
        model_path=os.path.join(base_dir, "healthcare_imaging_cnn.keras"),
        metadata_path=os.path.join(base_dir, "healthcare_imaging_cnn.meta.json"),
    )


@dataclass(frozen=True)
class DatasetSplits:
    train_dir: str
    val_dir: Optional[str]
    test_dir: Optional[str]


def _maybe_split_dirs(dataset_dir: str) -> DatasetSplits:
    dataset_dir = os.path.abspath(dataset_dir)
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    if os.path.isdir(train_dir):
        return DatasetSplits(
            train_dir=train_dir,
            val_dir=val_dir if os.path.isdir(val_dir) else None,
            test_dir=test_dir if os.path.isdir(test_dir) else None,
        )

    return DatasetSplits(train_dir=dataset_dir, val_dir=None, test_dir=None)


def _load_datasets(
    *,
    dataset_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    validation_split: float,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], list]:
    splits = _maybe_split_dirs(dataset_dir)

    if splits.val_dir is not None:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            splits.train_dir,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            splits.val_dir,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        if not (0.0 < validation_split < 1.0):
            raise ValueError("validation_split must be in (0, 1) when no val/ folder exists")

        root_dir = os.path.abspath(splits.train_dir)
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not class_names:
            raise ValueError(f"No class folders found under: {root_dir}")

        rows = []
        for class_idx, cname in enumerate(class_names):
            cdir = os.path.join(root_dir, cname)
            for fname in os.listdir(cdir):
                lower = fname.lower()
                if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
                    continue
                rows.append((os.path.join(cdir, fname), int(class_idx)))
        if not rows:
            raise ValueError(f"No images found under: {root_dir}")

        rng = np.random.default_rng(int(seed))
        paths = np.asarray([r[0] for r in rows], dtype=object)
        labels = np.asarray([r[1] for r in rows], dtype=np.int64)

        train_idx = []
        val_idx = []
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            rng.shuffle(idx)
            n = int(len(idx))
            n_val = max(int(np.floor(n * float(validation_split))), 1)
            n_val = min(n_val, max(n - 1, 0))
            val_idx.extend(idx[:n_val].tolist())
            train_idx.extend(idx[n_val:].tolist())

        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        def _decode(path: tf.Tensor) -> tf.Tensor:
            img_bytes = tf.io.read_file(path)
            img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, image_size, antialias=True)
            return img * 255.0

        def _make_ds(idxs: np.ndarray, *, training: bool) -> tf.data.Dataset:
            ds = tf.data.Dataset.from_tensor_slices((paths[idxs].astype(str), labels[idxs].astype(np.int64)))
            if training:
                ds = ds.shuffle(min(int(idxs.size), 10_000), seed=int(seed), reshuffle_each_iteration=True)

            def _map(p: tf.Tensor, y: tf.Tensor):
                return _decode(p), tf.cast(y, tf.int32)

            ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(int(batch_size))
            ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            return ds

        train_ds = _make_ds(train_idx, training=True)
        val_ds = _make_ds(val_idx, training=False)

    test_ds = None
    if splits.test_dir is not None:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            splits.test_dir,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        )

    class_names = list(getattr(train_ds, "class_names", class_names if 'class_names' in locals() else []))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


def _build_model(
    *,
    image_size: Tuple[int, int],
    num_classes: int,
    arch: str,
    seed: int,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.02),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augment",
    )

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = data_augmentation(x)

    if arch == "simple":
        x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.30)(x)
    elif arch == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size[0], image_size[1], 3),
        )
        base.trainable = False
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.30)(x)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if num_classes == 2:
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pred")(x)
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"healthcare_imaging_{arch}")
    return model


def train_and_test(
    *,
    dataset_dir: str,
    model_path: str,
    metadata_path: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    arch: str,
    seed: int,
    validation_split: float,
    save_model: bool,
    fine_tune_epochs: int = 2,
    fine_tune_lr: float = 1e-5,
    unfreeze_last_n: int = 40,
) -> None:
    image_hw = (int(image_size), int(image_size))
    train_ds, val_ds, test_ds, class_names = _load_datasets(
        dataset_dir=dataset_dir,
        image_size=image_hw,
        batch_size=int(batch_size),
        seed=int(seed),
        validation_split=float(validation_split),
    )

    num_classes = len(class_names) if class_names else 2
    model = _build_model(image_size=image_hw, num_classes=num_classes, arch=str(arch), seed=int(seed))

    if num_classes == 2:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=metrics)

    train_counts = np.zeros((int(num_classes),), dtype=np.int64)
    for _, yb in train_ds:
        yb = yb.numpy().reshape(-1)
        for v in yb.tolist():
            if 0 <= int(v) < int(num_classes):
                train_counts[int(v)] += 1
    train_counts = np.maximum(train_counts, 1)
    cw = (float(np.mean(train_counts)) / train_counts.astype(np.float64)).astype(np.float32)
    if int(num_classes) == 2:
        class_weight = {0: float(cw[0]), 1: float(cw[1])}
    else:
        class_weight = {int(i): float(cw[i]) for i in range(int(num_classes))}

    param_count = model.count_params()
    train_start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=int(epochs), verbose=1, class_weight=class_weight)
    train_seconds = time.perf_counter() - train_start

    fine_tune_epochs = int(fine_tune_epochs)
    if str(arch) == "mobilenetv2" and fine_tune_epochs > 0:
        base = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and layer.name.lower().startswith("mobilenet"):
                base = layer
                break
        if base is not None:
            base.trainable = True
            unfreeze_last_n = int(unfreeze_last_n)
            if unfreeze_last_n > 0:
                for l in base.layers[:-unfreeze_last_n]:
                    l.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(fine_tune_lr)), loss=loss, metrics=metrics)
            ft_start = time.perf_counter()
            _ = model.fit(train_ds, validation_data=val_ds, epochs=int(fine_tune_epochs), verbose=1, class_weight=class_weight)
            train_seconds += float(time.perf_counter() - ft_start)

    val_metrics = history.history
    last_val = {k: float(v[-1]) for k, v in val_metrics.items() if isinstance(v, list) and v}
    print(f"Classes: {class_names}")
    print(f"Parameters: {param_count}")
    print(f"Training time (s): {train_seconds:.3f}")
    print(f"Last epoch metrics: {last_val}")

    test_score_map = None
    sklearn_auc = None
    inference = None
    test_report = None
    canary = None
    if test_ds is not None:
        score = model.evaluate(test_ds, verbose=0)
        metric_names = list(model.metrics_names)
        test_score_map = {metric_names[i]: float(score[i]) for i in range(len(metric_names))}
        print(f"Test metrics: {test_score_map}")

        if int(num_classes) == 2:
            y_true = []
            y_prob = []
            for xb, yb in test_ds:
                p = model.predict(xb, verbose=0).reshape(-1)
                y_true.extend(yb.numpy().reshape(-1).astype(int).tolist())
                y_prob.extend(p.astype(float).tolist())
            if len(set(y_true)) >= 2:
                sklearn_auc = float(roc_auc_score(np.asarray(y_true, dtype=np.int32), np.asarray(y_prob, dtype=np.float64)))

            y_true_arr = np.asarray(y_true, dtype=np.int32)
            y_prob_arr = np.asarray(y_prob, dtype=np.float32)
            y_pred_arr = (y_prob_arr >= 0.5).astype(np.int32)
            test_report = _classification_report(y_true_arr, y_pred_arr, class_names=list(map(str, class_names)))
            canary = y_prob_arr[: min(512, int(y_prob_arr.shape[0]))]

        infer_samples = 0
        for batch_x, _ in test_ds.take(10):
            infer_samples += int(batch_x.shape[0])

        if infer_samples:
            for batch_x, _ in test_ds.take(1):
                model.predict(batch_x[:1], verbose=0)
                break

            infer_start = time.perf_counter()
            seen = 0
            for batch_x, _ in test_ds.take(10):
                _ = model.predict(batch_x, verbose=0)
                seen += int(batch_x.shape[0])
            infer_seconds = time.perf_counter() - infer_start

            ms_per_image = (infer_seconds / seen) * 1000.0 if seen else float("nan")
            images_per_second = (seen / infer_seconds) if infer_seconds > 0 else float("inf")
            print(f"Inference samples: {seen}")
            print(f"Inference time (s): {infer_seconds:.3f}")
            print(f"Latency (ms/image): {ms_per_image:.3f}")
            print(f"Throughput (images/s): {images_per_second:.2f}")
            inference = {
                "samples": int(seen),
                "seconds": float(infer_seconds),
                "ms_per_image": float(ms_per_image),
                "images_per_second": float(images_per_second),
            }

    out = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "arch": str(arch),
        "validation_split": float(validation_split),
        "classes": list(map(str, class_names)),
        "num_classes": int(num_classes),
        "parameters": int(param_count),
        "train_seconds": float(train_seconds),
        "last_epoch_metrics": last_val,
        "test_metrics": test_score_map,
        "test_report": test_report,
        "sklearn_auc": sklearn_auc,
        "class_weight": class_weight,
        "fine_tune": {"epochs": int(fine_tune_epochs), "lr": float(fine_tune_lr), "unfreeze_last_n": int(unfreeze_last_n)} if str(arch) == "mobilenetv2" else None,
        "inference": inference,
        "artifacts": {"model_path": model_path if save_model else None, "metadata_path": metadata_path},
    }

    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    if save_model:
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        model.save(model_path)
        print(f"Saved model: {model_path}")

    out["fingerprint"] = {
        "model_sha256": _sha256_file(model_path) if save_model else None,
        "metadata_sha256": _sha256_file(metadata_path),
        "canary_pred_sha256": _sha256_array(np.asarray(canary if canary is not None else np.zeros((0,), dtype=np.float32), dtype=np.float32)),
    }

    out["promotion"] = {
        "task": "binary_classification",
        "primary_metric": "sklearn_auc",
        "primary": float(out.get("sklearn_auc") or 0.0),
        "baseline_name": "random_auc",
        "baseline": 0.5,
        "eligible": bool(float(out.get("sklearn_auc") or 0.0) >= 0.85),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            mat[int(t), int(p)] += 1
    return mat


def _classification_report(y_true: np.ndarray, y_pred: np.ndarray, *, class_names: list) -> dict:
    num_classes = len(class_names)
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    per_class = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_precision = float(np.mean([per_class[n]["precision"] for n in class_names])) if class_names else 0.0
    macro_recall = float(np.mean([per_class[n]["recall"] for n in class_names])) if class_names else 0.0
    macro_f1 = float(np.mean([per_class[n]["f1"] for n in class_names])) if class_names else 0.0

    return {
        "accuracy": accuracy,
        "macro_avg": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _load_test_dataset(*, dataset_dir: str, image_size: Tuple[int, int], batch_size: int) -> Tuple[tf.data.Dataset, list]:
    splits = _maybe_split_dirs(dataset_dir)
    if splits.test_dir is None:
        raise FileNotFoundError(
            f"No test/ directory found under: {os.path.abspath(dataset_dir)}. Expected {os.path.join(dataset_dir, 'test')}"
        )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        splits.test_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = list(getattr(test_ds, "class_names", []))
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return test_ds, class_names


def test_only(
    *,
    dataset_dir: str,
    model_path: str,
    batch_size: int,
    image_size: int,
) -> None:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    inferred_hw: Optional[Tuple[int, int]] = None
    input_shape = model.input_shape
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]
    if isinstance(input_shape, tuple) and len(input_shape) >= 4:
        if input_shape[-1] in (1, 3):
            h, w = input_shape[1], input_shape[2]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                inferred_hw = (int(h), int(w))
        elif input_shape[1] in (1, 3):
            h, w = input_shape[2], input_shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                inferred_hw = (int(h), int(w))

    image_hw = inferred_hw if inferred_hw is not None else (int(image_size), int(image_size))
    test_ds, class_names = _load_test_dataset(dataset_dir=dataset_dir, image_size=image_hw, batch_size=int(batch_size))
    out_shape = model.output_shape
    is_binary = bool(isinstance(out_shape, (tuple, list)) and int(out_shape[-1]) == 1)

    y_true_all = []
    y_pred_all = []
    auc_metric = tf.keras.metrics.AUC(name="auc")

    for batch_x, batch_y in test_ds:
        probs = model(batch_x, training=False)
        y_true = batch_y.numpy().astype(np.int64)

        if is_binary:
            p = probs.numpy().reshape(-1)
            auc_metric.update_state(y_true.astype(np.float32), p.astype(np.float32))
            y_pred = (p >= 0.5).astype(np.int64)
        else:
            p = probs.numpy()
            y_pred = np.argmax(p, axis=-1).astype(np.int64)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all, axis=0) if y_true_all else np.zeros((0,), dtype=np.int64)
    y_pred_all = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.zeros((0,), dtype=np.int64)

    if not class_names:
        num_classes = int(np.max(y_true_all)) + 1 if y_true_all.size else 2
        class_names = [str(i) for i in range(num_classes)]

    report = _classification_report(y_true_all, y_pred_all, class_names=class_names)
    print(f"Classes: {class_names}")
    print(f"Test accuracy: {report['accuracy']:.6f}")
    if is_binary:
        print(f"Test AUC: {float(auc_metric.result().numpy()):.6f}")
    print(f"Confusion matrix: {report['confusion_matrix']}")
    print(f"Per-class metrics: {report['per_class']}")


def main() -> None:
    default_dataset_dir = os.path.join(os.path.dirname(__file__), "ChestXRay2017", "chest_xray")
    artifacts = _default_artifacts()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-dir", type=str, default=default_dataset_dir)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--arch", type=str, default="mobilenetv2", choices=["simple", "mobilenetv2"])
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--validation-split", type=float, default=0.2)
    train_parser.add_argument("--fine-tune-epochs", type=int, default=2)
    train_parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    train_parser.add_argument("--unfreeze-last-n", type=int, default=40)
    train_parser.add_argument("--model-path", type=str, default=artifacts.model_path)
    train_parser.add_argument("--metadata-path", type=str, default=artifacts.metadata_path)
    train_parser.add_argument("--no-save", action="store_true")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--dataset-dir", type=str, default=default_dataset_dir)
    test_parser.add_argument("--batch-size", type=int, default=32)
    test_parser.add_argument("--image-size", type=int, default=224)
    test_parser.add_argument("--model-path", type=str, default=artifacts.model_path)

    args = parser.parse_args()

    if args.command in (None, "train"):
        train_and_test(
            dataset_dir=str(getattr(args, "dataset_dir", default_dataset_dir)),
            model_path=str(getattr(args, "model_path", artifacts.model_path)),
            metadata_path=str(getattr(args, "metadata_path", artifacts.metadata_path)),
            epochs=int(getattr(args, "epochs", 10)),
            batch_size=int(getattr(args, "batch_size", 32)),
            image_size=int(getattr(args, "image_size", 224)),
            arch=str(getattr(args, "arch", "mobilenetv2")),
            seed=int(getattr(args, "seed", 1337)),
            validation_split=float(getattr(args, "validation_split", 0.2)),
            save_model=not bool(getattr(args, "no_save", False)),
            fine_tune_epochs=int(getattr(args, "fine_tune_epochs", 2)),
            fine_tune_lr=float(getattr(args, "fine_tune_lr", 1e-5)),
            unfreeze_last_n=int(getattr(args, "unfreeze_last_n", 40)),
        )
        return

    if args.command == "test":
        test_only(
            dataset_dir=str(getattr(args, "dataset_dir", default_dataset_dir)),
            model_path=str(getattr(args, "model_path", artifacts.model_path)),
            batch_size=int(getattr(args, "batch_size", 32)),
            image_size=int(getattr(args, "image_size", 224)),
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
