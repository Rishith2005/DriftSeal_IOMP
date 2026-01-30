import argparse
import hashlib
import json
import os
import time

import numpy as np
import tensorflow
from datasets import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1

default_model_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.keras")
default_metadata_path = os.path.join(os.path.dirname(__file__), "mnist_cnn.meta.json")


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

def load_mnist(cache_dir: str):
    ds = load_dataset("ylecun/mnist", cache_dir=cache_dir)
    train_ds = ds["train"]
    test_ds = ds["test"]

    input_train = np.stack([np.asarray(img, dtype=np.float32) for img in train_ds["image"]])
    target_train = np.asarray(train_ds["label"], dtype=np.int64)

    input_test = np.stack([np.asarray(img, dtype=np.float32) for img in test_ds["image"]])
    target_test = np.asarray(test_ds["label"], dtype=np.int64)

    return (input_train, target_train), (input_test, target_test)


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(no_classes, activation="softmax"))
    return model


def train_and_test(*, cache_dir: str, model_path: str, metadata_path: str, epochs: int, batch: int, save_model: bool):
    (input_train, target_train), (input_test, target_test) = load_mnist(cache_dir=cache_dir)

    target_train_raw = target_train.astype(np.int64, copy=True)
    target_test_raw = target_test.astype(np.int64, copy=True)

    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

    input_train = input_train.astype("float32")
    input_test = input_test.astype("float32")

    input_train = input_train / 255
    input_test = input_test / 255

    target_train = tensorflow.keras.utils.to_categorical(target_train, no_classes)
    target_test = tensorflow.keras.utils.to_categorical(target_test, no_classes)

    model = build_model(input_shape)

    model.compile(
        loss=tensorflow.keras.losses.categorical_crossentropy,
        optimizer=tensorflow.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    param_count = model.count_params()

    train_start = time.perf_counter()
    model.fit(
        input_train,
        target_train,
        batch_size=batch,
        epochs=epochs,
        verbose=verbosity,
        validation_split=validation_split,
    )
    train_seconds = time.perf_counter() - train_start

    score = model.evaluate(input_test, target_test, verbose=0)
    print(f"Test loss: {score[0]} / Test accuracy: {score[1]}")
    print(f"Parameters: {param_count}")
    print(f"Training time (s): {train_seconds:.3f}")

    infer_samples = min(int(input_test.shape[0]), 1000)
    infer_batch = input_test[:infer_samples]
    model.predict(infer_batch[:1], verbose=0)
    infer_start = time.perf_counter()
    model.predict(infer_batch, batch_size=batch, verbose=0)
    infer_seconds = time.perf_counter() - infer_start

    ms_per_image = (infer_seconds / infer_samples) * 1000 if infer_samples else float("nan")
    images_per_second = (infer_samples / infer_seconds) if infer_seconds > 0 else float("inf")
    print(f"Inference samples: {infer_samples}")
    print(f"Inference time (s): {infer_seconds:.3f}")
    print(f"Latency (ms/image): {ms_per_image:.3f}")
    print(f"Throughput (images/s): {images_per_second:.2f}")

    majority_class = int(np.bincount(target_train_raw, minlength=no_classes).argmax()) if target_train_raw.size else 0
    baseline_accuracy = float((target_test_raw == majority_class).mean()) if target_test_raw.size else None

    canary_n = min(256, int(input_test.shape[0]))
    canary_pred = model.predict(input_test[:canary_n], batch_size=batch, verbose=0) if canary_n > 0 else np.zeros((0, no_classes), dtype=np.float32)

    out = {
        "cache_dir": os.path.abspath(cache_dir),
        "seed": None,
        "epochs": int(epochs),
        "batch_size": int(batch),
        "image_size": [int(img_width), int(img_height)],
        "num_classes": int(no_classes),
        "validation_split": float(validation_split),
        "parameters": int(param_count),
        "train_seconds": float(train_seconds),
        "test_metrics": {"loss": float(score[0]), "accuracy": float(score[1])},
        "baseline_metrics": {"majority_class": {"class": int(majority_class), "accuracy": baseline_accuracy}},
        "inference": {
            "samples": int(infer_samples),
            "seconds": float(infer_seconds),
            "ms_per_image": float(ms_per_image),
            "images_per_second": float(images_per_second),
        },
        "artifacts": {"model_path": model_path if save_model else None, "metadata_path": metadata_path},
    }

    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    if save_model:
        model.save(model_path)
        print(f"Saved model: {model_path}")

    out["fingerprint"] = {
        "model_sha256": _sha256_file(model_path) if save_model else None,
        "canary_pred_sha256": _sha256_array(canary_pred.astype(np.float32, copy=False)),
    }

    acc = float(out.get("test_metrics", {}).get("accuracy", 0.0))
    base_acc = float(out.get("baseline_metrics", {}).get("majority_class", {}).get("accuracy") or 0.0)
    out["promotion"] = {
        "task": "multiclass_classification",
        "primary_metric": "accuracy",
        "primary": acc,
        "baseline_name": "majority_class",
        "baseline": base_acc,
        "eligible": bool(acc > base_acc + 0.02),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def preprocess_image_for_mnist(image, *, photo_mode: bool = True) -> np.ndarray:
    from PIL import Image, ImageFilter, ImageOps

    if image is None:
        raise ValueError("No image provided")

    if isinstance(image, dict) and "image" in image:
        image = image["image"]

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            image = Image.fromarray(image[..., :3].astype(np.uint8), mode="RGB")
        else:
            image = Image.fromarray(image.astype(np.uint8))

    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    def otsu_threshold(arr_u8: np.ndarray) -> int:
        hist = np.bincount(arr_u8.ravel(), minlength=256).astype(np.float64)
        total = float(arr_u8.size)
        sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))

        sum_b = 0.0
        w_b = 0.0
        max_var = -1.0
        threshold = 127

        for t in range(256):
            w_b += hist[t]
            if w_b <= 0.0:
                continue

            w_f = total - w_b
            if w_f <= 0.0:
                break

            sum_b += float(t) * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2

            if var_between > max_var:
                max_var = var_between
                threshold = t

        return int(threshold)

    def shift_image(img2d: np.ndarray, dy: int, dx: int) -> np.ndarray:
        out = np.zeros_like(img2d)

        y_src_start = max(0, -dy)
        y_src_end = min(img2d.shape[0], img2d.shape[0] - dy)
        y_dst_start = max(0, dy)
        y_dst_end = min(img2d.shape[0], img2d.shape[0] + dy)

        x_src_start = max(0, -dx)
        x_src_end = min(img2d.shape[1], img2d.shape[1] - dx)
        x_dst_start = max(0, dx)
        x_dst_end = min(img2d.shape[1], img2d.shape[1] + dx)

        if y_src_end <= y_src_start or x_src_end <= x_src_start:
            return out

        out[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = img2d[y_src_start:y_src_end, x_src_start:x_src_end]
        return out

    image = image.convert("L")

    if not photo_mode:
        image = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32)
        if float(arr.mean()) > 127.0:
            image = ImageOps.invert(image)
            arr = np.asarray(image, dtype=np.float32)

        arr = arr / 255.0
        arr = arr.reshape(1, img_width, img_height, 1)
        return arr

    image = ImageOps.autocontrast(image)
    if float(np.asarray(image, dtype=np.float32).mean()) > 127.0:
        image = ImageOps.invert(image)

    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    working = image.resize((256, 256), resample=Image.Resampling.BILINEAR)
    arr_u8 = np.asarray(working, dtype=np.uint8)

    threshold = otsu_threshold(arr_u8)
    mask = arr_u8 > threshold
    if int(mask.sum()) < 25:
        resized = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr.reshape(1, img_width, img_height, 1)
        return arr

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    pad = int(0.2 * max(y1 - y0, x1 - x0)) + 2
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(arr_u8.shape[0], y1 + pad)
    x1 = min(arr_u8.shape[1], x1 + pad)

    orig_u8 = np.asarray(image, dtype=np.uint8)
    scale = orig_u8.shape[0] / arr_u8.shape[0]
    y0o, y1o = int(round(y0 * scale)), int(round(y1 * scale))
    x0o, x1o = int(round(x0 * scale)), int(round(x1 * scale))

    cropped = image.crop((x0o, y0o, x1o, y1o))
    cw, ch = cropped.size
    if cw <= 0 or ch <= 0:
        resized = image.resize((img_width, img_height), resample=Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = arr.reshape(1, img_width, img_height, 1)
        return arr

    if cw > ch:
        new_w = 20
        new_h = max(1, int(round(20 * (ch / cw))))
    else:
        new_h = 20
        new_w = max(1, int(round(20 * (cw / ch))))

    resized_digit = cropped.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("L", (img_width, img_height), 0)
    canvas.paste(resized_digit, ((img_width - new_w) // 2, (img_height - new_h) // 2))

    canvas_arr = (np.asarray(canvas, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    density = float((canvas_arr > 0.2).mean())
    if density < 0.04:
        canvas = canvas.filter(ImageFilter.MaxFilter(size=3))
        canvas_arr = (np.asarray(canvas, dtype=np.float32) / 255.0).clip(0.0, 1.0)
    elif density > 0.30:
        canvas = canvas.filter(ImageFilter.MinFilter(size=3))
        canvas_arr = (np.asarray(canvas, dtype=np.float32) / 255.0).clip(0.0, 1.0)

    total = float(canvas_arr.sum())
    if total > 0.0:
        ys, xs = np.indices(canvas_arr.shape)
        cy = float((ys * canvas_arr).sum() / total)
        cx = float((xs * canvas_arr).sum() / total)
        dy = int(round((img_height / 2) - cy))
        dx = int(round((img_width / 2) - cx))
        canvas_arr = shift_image(canvas_arr, dy=dy, dx=dx)

    arr = canvas_arr.reshape(1, img_width, img_height, 1).astype(np.float32)
    return arr


def run_gradio_ui(model_path: str):
    import gradio as gr

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train first: python clean_model_fingerprint.py train"
        )

    model = tensorflow.keras.models.load_model(model_path)

    def predict(image, photo_mode):
        x = preprocess_image_for_mnist(image, photo_mode=bool(photo_mode))
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        processed = (x[0, :, :, 0] * 255.0).astype(np.uint8)
        prob_dict = {str(i): float(probs[i]) for i in range(no_classes)}
        return processed, f"{pred} (confidence {conf:.4f})", prob_dict

    sketchpad_cls = getattr(gr, "Sketchpad", None)

    with gr.Blocks() as demo:
        gr.Markdown("MNIST Digit Classifier")
        photo_mode_in = gr.Checkbox(value=True, label="Photo mode (crop + center + normalize)")

        with gr.Tabs():
            with gr.Tab("Upload"):
                upload_in = gr.Image(type="pil", label="Input image")
                upload_btn = gr.Button("Predict")

            with gr.Tab("Draw"):
                if sketchpad_cls is not None:
                    draw_in = sketchpad_cls(label="Draw a digit")
                else:
                    draw_in = gr.Image(type="pil", label="Draw a digit")
                draw_btn = gr.Button("Predict")

        processed_out = gr.Image(type="numpy", label="Processed 28x28")
        pred_out = gr.Textbox(label="Prediction")
        probs_out = gr.Label(label="Probabilities")

        upload_btn.click(
            fn=predict,
            inputs=[upload_in, photo_mode_in],
            outputs=[processed_out, pred_out, probs_out],
        )
        draw_btn.click(
            fn=predict,
            inputs=[draw_in, photo_mode_in],
            outputs=[processed_out, pred_out, probs_out],
        )

    demo.launch()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=no_epochs)
    train_parser.add_argument("--batch-size", type=int, default=batch_size)
    train_parser.add_argument("--model-path", type=str, default=default_model_path)
    train_parser.add_argument("--metadata-path", type=str, default=default_metadata_path)
    train_parser.add_argument("--no-save", action="store_true")

    ui_parser = subparsers.add_parser("ui")
    ui_parser.add_argument("--model-path", type=str, default=default_model_path)

    args = parser.parse_args()

    if args.command in (None, "train"):
        train_and_test(
            cache_dir=os.path.join(os.path.dirname(__file__), "data", "hf_datasets_cache"),
            model_path=getattr(args, "model_path", default_model_path),
            metadata_path=getattr(args, "metadata_path", default_metadata_path),
            epochs=int(getattr(args, "epochs", no_epochs)),
            batch=int(getattr(args, "batch_size", batch_size)),
            save_model=not bool(getattr(args, "no_save", False)),
        )
        return

    if args.command == "ui":
        run_gradio_ui(model_path=args.model_path)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
