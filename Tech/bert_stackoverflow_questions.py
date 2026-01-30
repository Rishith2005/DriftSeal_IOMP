import argparse
import hashlib
import html
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup


def _sha256_file(path: str) -> Optional[str]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir(path: str) -> Optional[str]:
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return None
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, path).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            with open(fp, "rb") as f:
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
    model_dir: str
    metadata_path: str
    label_map_path: str


def _default_artifacts(base_dir: str) -> Artifacts:
    return Artifacts(
        model_dir=os.path.join(base_dir, "bert_stackoverflow_model"),
        metadata_path=os.path.join(base_dir, "bert_stackoverflow.meta.json"),
        label_map_path=os.path.join(base_dir, "bert_stackoverflow.labels.json"),
    )


def _read_csv_any(*paths: str) -> pd.DataFrame:
    for p in paths:
        fp = os.path.abspath(p)
        if os.path.isfile(fp):
            return pd.read_csv(fp)
    raise FileNotFoundError(f"CSV not found in any of: {paths}")


def _resolve_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], str]:
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title")
    body_col = cols.get("body")
    if not title_col and not body_col:
        for cand in ("text", "question", "content"):
            if cand in cols:
                title_col = cols[cand]
                break
    label_col = cols.get("y") or cols.get("label") or cols.get("tag") or cols.get("tags") or cols.get("category")
    if not label_col:
        raise ValueError("Could not infer label column. Expected Y/label/tag/tags/category.")
    if not title_col and not body_col:
        raise ValueError("Could not infer text columns. Expected Title/Body or text/question/content.")
    return title_col or body_col, body_col if title_col else None, label_col


def _build_text(df: pd.DataFrame, *, title_col: str, body_col: Optional[str]) -> pd.Series:
    title = df[title_col].astype("string").fillna("")
    if body_col and body_col in df.columns:
        body = df[body_col].astype("string").fillna("")
        return (title + "\n\n" + body).str.strip()
    return title.str.strip()


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_BR_RE = re.compile(r"(?i)<br\\s*/?>")


def _clean_text_series(text: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    s = text.astype("string").fillna("")
    before_avg_len = float(s.str.len().mean() if len(s) else 0.0)
    before_has_tag = int(s.str.contains(r"<[^>]+>", regex=True).sum()) if len(s) else 0

    s = s.str.replace(_BR_RE, "\n", regex=True)
    s = s.map(lambda x: html.unescape(str(x)))
    s = s.str.replace(_TAG_RE, " ", regex=True)
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(_WS_RE, " ", regex=True)
    s = s.str.replace(_MULTI_NL_RE, "\n\n", regex=True)
    s = s.str.strip()

    after_avg_len = float(s.str.len().mean() if len(s) else 0.0)
    meta = {
        "before_avg_len": before_avg_len,
        "after_avg_len": after_avg_len,
        "rows_with_html_like_tags": before_has_tag,
    }
    return s, meta


def _clean_dataset(
    df: pd.DataFrame,
    *,
    title_col: str,
    body_col: Optional[str],
    label_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    out[title_col], title_meta = _clean_text_series(out[title_col])
    if body_col and body_col in out.columns:
        out[body_col], body_meta = _clean_text_series(out[body_col])
    else:
        body_meta = None

    out[label_col] = out[label_col].astype("string").fillna("missing").str.strip()
    empty_text = int((_build_text(out, title_col=title_col, body_col=body_col) == "").sum())
    before_n = int(len(out))
    if empty_text:
        keep = _build_text(out, title_col=title_col, body_col=body_col) != ""
        out = out.loc[keep].reset_index(drop=True)

    after_n = int(len(out))
    meta = {
        "rows_before": before_n,
        "rows_after": after_n,
        "rows_dropped_empty_text": int(before_n - after_n),
        "title_cleaning": title_meta,
        "body_cleaning": body_meta,
    }
    return out, meta


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TextDataset(Dataset):
    def __init__(
        self,
        *,
        texts: pd.Series,
        labels: Optional[np.ndarray],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self._texts = texts.astype("string").fillna("").tolist()
        self._labels = labels
        self._tokenizer = tokenizer
        self._max_length = int(max_length)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx = int(idx)
        enc = self._tokenizer(
            str(self._texts[idx]),
            truncation=True,
            max_length=self._max_length,
            return_tensors=None,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        if self._labels is not None:
            item["labels"] = torch.tensor(int(self._labels[idx]), dtype=torch.long)
        return item


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _eval_model(
    *,
    model: Any,
    loader: DataLoader,
    device: torch.device,
    class_weight: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            labels = batch.get("labels")
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            if labels is not None:
                loss = F.cross_entropy(logits, labels, weight=class_weight)
                total_loss += float(loss.detach().cpu().item())
            total_batches += 1

            labels = batch["labels"].detach().cpu().numpy().astype(np.int32).tolist()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int32).tolist()
            y_true.extend(labels)
            y_pred.extend(preds)

    avg_loss = float(total_loss / max(1, total_batches))
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    return {"loss": avg_loss, "accuracy": acc, "y_true": np.array(y_true, dtype=np.int32), "y_pred": np.array(y_pred, dtype=np.int32)}


def train_and_eval(
    *,
    tech_dir: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    save_model: bool,
    freeze_base: bool,
    max_train_samples: int,
    max_valid_samples: int,
    test_frac: float,
    log_every: int,
    num_workers: int,
    grad_accum_steps: int,
    fp16: bool,
    clean_data: bool,
    class_weighting: bool = True,
) -> Dict[str, Any]:
    _set_seed(int(seed))
    tech_dir = os.path.abspath(tech_dir)

    train_df = _read_csv_any(os.path.join(tech_dir, "train.csv", "train.csv"), os.path.join(tech_dir, "train.csv"))
    valid_df = _read_csv_any(os.path.join(tech_dir, "valid.csv", "valid.csv"), os.path.join(tech_dir, "valid.csv"))

    title_col, body_col, label_col = _resolve_columns(train_df)
    cleaning_meta: Dict[str, Any] = {"enabled": bool(clean_data)}
    if clean_data:
        train_df, train_meta = _clean_dataset(train_df, title_col=title_col, body_col=body_col, label_col=label_col)
        valid_df, valid_meta = _clean_dataset(valid_df, title_col=title_col, body_col=body_col, label_col=label_col)
        cleaning_meta["train"] = train_meta
        cleaning_meta["valid"] = valid_meta
    train_text = _build_text(train_df, title_col=title_col, body_col=body_col)
    valid_text_all = _build_text(valid_df, title_col=title_col, body_col=body_col)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col].astype("string").fillna("missing"))
    y_valid_all = le.transform(valid_df[label_col].astype("string").fillna("missing"))
    num_labels = int(len(le.classes_))

    max_train_samples = int(max_train_samples)
    max_valid_samples = int(max_valid_samples)
    if max_train_samples > 0:
        train_text = train_text.iloc[:max_train_samples]
        y_train = y_train[:max_train_samples]
    if max_valid_samples > 0:
        valid_text_all = valid_text_all.iloc[:max_valid_samples]
        y_valid_all = y_valid_all[:max_valid_samples]

    test_frac = float(test_frac)
    if not (0.0 < test_frac < 1.0):
        raise ValueError("--test-frac must be between 0 and 1")
    va_text, te_text, y_valid, y_test = train_test_split(
        valid_text_all,
        y_valid_all,
        test_size=test_frac,
        random_state=int(seed),
        stratify=y_valid_all,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_name))
    print(f"Tokenizing train={len(train_text)} valid={len(va_text)} test={len(te_text)} max_length={int(max_length)}")
    train_ds = _TextDataset(texts=train_text, labels=y_train.astype(np.int32), tokenizer=tokenizer, max_length=int(max_length))
    valid_ds = _TextDataset(texts=va_text, labels=y_valid.astype(np.int32), tokenizer=tokenizer, max_length=int(max_length))
    test_ds = _TextDataset(texts=te_text, labels=y_test.astype(np.int32), tokenizer=tokenizer, max_length=int(max_length))
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = bool(device.type == "cuda")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        collate_fn=collator,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(str(model_name), num_labels=num_labels)
    model.to(device)

    class_weight_t = None
    if bool(class_weighting) and int(num_labels) >= 2:
        counts = np.bincount(y_train.astype(np.int64), minlength=int(num_labels)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        w = (float(np.mean(counts)) / counts).astype(np.float32)
        class_weight_t = torch.tensor(w, dtype=torch.float32, device=device)

    if freeze_base:
        base_prefix = getattr(model, "base_model_prefix", None)
        if isinstance(base_prefix, str) and hasattr(model, base_prefix):
            base_model = getattr(model, base_prefix)
            for p in base_model.parameters():
                p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    grad_accum_steps = max(1, int(grad_accum_steps))
    steps_per_epoch = int(np.ceil(len(train_loader) / float(grad_accum_steps)))
    total_steps = int(steps_per_epoch * int(epochs))
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps))

    start = time.time()
    history: Dict[str, list[float]] = {"train_loss": [], "valid_loss": [], "valid_acc": []}

    log_every = max(1, int(log_every))
    amp_enabled = bool(fp16) and device.type == "cuda"
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device_type, enabled=amp_enabled)
    for epoch_idx in range(int(epochs)):
        print(f"Epoch {epoch_idx + 1}/{int(epochs)}")
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        optimizer_steps = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            labels = batch.get("labels")
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_enabled):
                outputs = model(**inputs)
                logits = outputs.logits
                if labels is None:
                    raise ValueError("Expected labels in training batch")
                loss = F.cross_entropy(logits, labels, weight=class_weight_t) / float(grad_accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += float(loss.detach().cpu().item()) * float(grad_accum_steps)
            epoch_batches += 1
            should_step = (epoch_batches % grad_accum_steps == 0) or (epoch_batches == len(train_loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps % log_every == 0:
                    avg = float(epoch_loss / max(1, epoch_batches))
                    print(f"  step={optimizer_steps}/{steps_per_epoch} train_loss={avg:.4f}")

        train_loss = float(epoch_loss / max(1, epoch_batches))
        valid_eval = _eval_model(model=model, loader=valid_loader, device=device, class_weight=class_weight_t)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(float(valid_eval["loss"]))
        history["valid_acc"].append(float(valid_eval["accuracy"]))
        print(f"  epoch_end train_loss={train_loss:.4f} valid_loss={float(valid_eval['loss']):.4f} valid_acc={float(valid_eval['accuracy']):.4f}")

    fit_seconds = float(time.time() - start)

    final_valid = _eval_model(model=model, loader=valid_loader, device=device, class_weight=class_weight_t)
    y_pred_va = final_valid["y_pred"]
    acc_va = float(accuracy_score(y_valid, y_pred_va))
    report_va = classification_report(y_valid, y_pred_va, target_names=list(le.classes_), output_dict=True, zero_division=0)
    cm_va = confusion_matrix(y_valid, y_pred_va).tolist()

    final_test = _eval_model(model=model, loader=test_loader, device=device, class_weight=class_weight_t)
    y_pred_te = final_test["y_pred"]
    acc_te = float(accuracy_score(y_test, y_pred_te))
    report_te = classification_report(y_test, y_pred_te, target_names=list(le.classes_), output_dict=True, zero_division=0)
    cm_te = confusion_matrix(y_test, y_pred_te).tolist()

    maj = int(np.bincount(y_train.astype(np.int64), minlength=int(num_labels)).argmax()) if y_train.size else 0
    base_va = np.full_like(y_valid, fill_value=maj, dtype=np.int32)
    base_te = np.full_like(y_test, fill_value=maj, dtype=np.int32)
    base_acc_va = float(accuracy_score(y_valid, base_va))
    base_rep_va = classification_report(y_valid, base_va, target_names=list(le.classes_), output_dict=True, zero_division=0)
    base_cm_va = confusion_matrix(y_valid, base_va).tolist()
    base_acc_te = float(accuracy_score(y_test, base_te))
    base_rep_te = classification_report(y_test, base_te, target_names=list(le.classes_), output_dict=True, zero_division=0)
    base_cm_te = confusion_matrix(y_test, base_te).tolist()

    artifacts = _default_artifacts(tech_dir)
    out: Dict[str, Any] = {
        "tech_dir": tech_dir,
        "model_name": str(model_name),
        "text_columns": {"title": title_col, "body": body_col},
        "label_column": str(label_col),
        "num_labels": num_labels,
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "freeze_base": bool(freeze_base),
        "class_weighting": bool(class_weighting),
        "max_train_samples": int(max_train_samples),
        "max_valid_samples": int(max_valid_samples),
        "log_every": int(log_every),
        "num_workers": int(num_workers),
        "grad_accum_steps": int(grad_accum_steps),
        "fp16": bool(fp16),
        "cleaning": cleaning_meta,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "split": {"n_val": int(len(va_text)), "n_test": int(len(te_text)), "test_frac": float(test_frac)},
        "fit_seconds": fit_seconds,
        "history": history,
        "valid_metrics": {
            "accuracy": acc_va,
            "macro_f1": float(report_va["macro avg"]["f1-score"]),
            "weighted_f1": float(report_va["weighted avg"]["f1-score"]),
            "macro_precision": float(report_va["macro avg"]["precision"]),
            "macro_recall": float(report_va["macro avg"]["recall"]),
            "confusion_matrix": cm_va,
        },
        "test_metrics": {
            "accuracy": acc_te,
            "macro_f1": float(report_te["macro avg"]["f1-score"]),
            "weighted_f1": float(report_te["weighted avg"]["f1-score"]),
            "macro_precision": float(report_te["macro avg"]["precision"]),
            "macro_recall": float(report_te["macro avg"]["recall"]),
            "confusion_matrix": cm_te,
        },
        "baseline_metrics": {
            "majority_class": {
                "class": int(maj),
                "valid": {
                    "accuracy": base_acc_va,
                    "macro_f1": float(base_rep_va["macro avg"]["f1-score"]),
                    "weighted_f1": float(base_rep_va["weighted avg"]["f1-score"]),
                    "confusion_matrix": base_cm_va,
                },
                "test": {
                    "accuracy": base_acc_te,
                    "macro_f1": float(base_rep_te["macro avg"]["f1-score"]),
                    "weighted_f1": float(base_rep_te["weighted avg"]["f1-score"]),
                    "confusion_matrix": base_cm_te,
                },
            }
        },
        "artifacts": {
            "model_dir": artifacts.model_dir if save_model else None,
            "metadata_path": artifacts.metadata_path,
            "label_map_path": artifacts.label_map_path,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(artifacts.metadata_path)), exist_ok=True)
    with open(artifacts.label_map_path, "w", encoding="utf-8") as f:
        json.dump({"classes": list(map(str, le.classes_))}, f, indent=2)
    with open(artifacts.metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if save_model:
        os.makedirs(artifacts.model_dir, exist_ok=True)
        model.save_pretrained(artifacts.model_dir)
        tokenizer.save_pretrained(artifacts.model_dir)

    out["fingerprint"] = {
        "model_dir_sha256": _sha256_dir(artifacts.model_dir) if save_model else None,
        "metadata_sha256": _sha256_file(artifacts.metadata_path),
        "label_map_sha256": _sha256_file(artifacts.label_map_path),
        "canary_pred_sha256": _sha256_array(np.asarray(y_pred_te[: min(512, int(y_pred_te.shape[0]))], dtype=np.int32)),
    }

    base_te_acc = float(out.get("baseline_metrics", {}).get("majority_class", {}).get("test", {}).get("accuracy", 0.0))
    out["promotion"] = {
        "task": "multiclass_classification",
        "primary_metric": "accuracy",
        "primary": float(acc_te),
        "baseline_name": "majority_class",
        "baseline": base_te_acc,
        "eligible": bool(float(acc_te) > base_te_acc + 0.02),
    }
    with open(artifacts.metadata_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def main() -> None:
    base_dir = os.path.dirname(__file__)
    p = argparse.ArgumentParser()
    p.add_argument("--tech-dir", default=base_dir)
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--freeze-base", action="store_true")
    p.add_argument("--train-full", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-valid-samples", type=int, default=0)
    p.add_argument("--test-frac", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    args = p.parse_args()

    result = train_and_eval(
        tech_dir=str(args.tech_dir),
        model_name=str(args.model_name),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        save_model=bool(args.save_model),
        freeze_base=(not bool(args.train_full)) or bool(args.freeze_base),
        max_train_samples=int(args.max_train_samples),
        max_valid_samples=int(args.max_valid_samples),
        test_frac=float(getattr(args, "test_frac", 0.5)),
        log_every=int(args.log_every),
        num_workers=int(args.num_workers),
        grad_accum_steps=int(args.grad_accum_steps),
        fp16=bool(args.fp16),
        clean_data=(not bool(args.no_clean)),
        class_weighting=(not bool(getattr(args, "no_class_weights", False))),
    )

    print(json.dumps(result["valid_metrics"], indent=2))


if __name__ == "__main__":
    main()
