#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetItem:
    split: str
    label: int
    augmentation: str
    source_path: Path
    processed_path: Path


@dataclass
class ForwardCache:
    x_down: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    p1: np.ndarray
    z2: np.ndarray
    a2: np.ndarray
    p2: np.ndarray
    z3: np.ndarray
    a3: np.ndarray
    gap: np.ndarray


def parse_args() -> argparse.Namespace:
    default_dataset = Path("/Users/alexeyguchko/lenta/Lenta_qualityset")
    default_out = Path("/Users/alexeyguchko/lenta/output/quality_cnn")

    parser = argparse.ArgumentParser(description="Train a small grayscale CNN quality classifier.")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["auto", "none"], default="auto")
    parser.add_argument("--reuse-processed", action="store_true")
    parser.add_argument("--model-out", type=Path, default=None)
    return parser.parse_args()


def list_class_images(dataset_dir: Path) -> dict[int, list[Path]]:
    class_paths: dict[int, list[Path]] = {}
    for label in (0, 1):
        class_dir = dataset_dir / str(label)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"No images found in {class_dir}")
        class_paths[label] = paths
    return class_paths


def split_paths(paths: list[Path], val_ratio: float, rng: np.random.Generator) -> tuple[list[Path], list[Path]]:
    indices = np.arange(len(paths))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(paths) * val_ratio)))
    val_indices = set(indices[:val_count].tolist())
    train_paths = [path for idx, path in enumerate(paths) if idx not in val_indices]
    val_paths = [path for idx, path in enumerate(paths) if idx in val_indices]
    return train_paths, val_paths


def preprocess_image(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def augmented_variants(image: np.ndarray, label: int, split: str) -> list[tuple[str, np.ndarray]]:
    variants = [("original", image)]
    if split == "train" and label == 1:
        variants.append(("flip_lr", cv2.flip(image, 1)))
        variants.append(("rotate_cw", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)))
    return variants


def prepare_dataset(
    dataset_dir: Path,
    processed_dir: Path,
    image_size: int,
    val_ratio: float,
    seed: int,
    reuse_processed: bool,
) -> list[DatasetItem]:
    manifest_path = processed_dir / "manifest.csv"
    if reuse_processed and manifest_path.exists():
        return read_manifest(manifest_path)

    protected_paths = {
        dataset_dir.resolve(),
        (dataset_dir / "0").resolve(),
        (dataset_dir / "1").resolve(),
    }
    if processed_dir.resolve() in protected_paths:
        raise ValueError(f"Refusing to use source dataset path as processed output: {processed_dir}")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    class_paths = list_class_images(dataset_dir)
    rng = np.random.default_rng(seed)
    items: list[DatasetItem] = []

    for label, paths in class_paths.items():
        train_paths, val_paths = split_paths(paths, val_ratio, rng)
        split_to_paths = {"train": train_paths, "val": val_paths}

        for split, split_paths_list in split_to_paths.items():
            for source_path in split_paths_list:
                image = preprocess_image(source_path, image_size)
                for augmentation, augmented in augmented_variants(image, label, split):
                    stem = source_path.stem
                    processed_name = f"{stem}__{augmentation}.png"
                    processed_path = processed_dir / split / str(label) / processed_name
                    processed_path.parent.mkdir(parents=True, exist_ok=True)
                    ok = cv2.imwrite(str(processed_path), augmented)
                    if not ok:
                        raise IOError(f"Could not write processed image: {processed_path}")
                    items.append(
                        DatasetItem(
                            split=split,
                            label=label,
                            augmentation=augmentation,
                            source_path=source_path,
                            processed_path=processed_path,
                        )
                    )

    write_manifest(manifest_path, items)
    return items


def write_manifest(path: Path, items: list[DatasetItem]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "augmentation", "source_path", "processed_path"],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "split": item.split,
                    "label": item.label,
                    "augmentation": item.augmentation,
                    "source_path": str(item.source_path),
                    "processed_path": str(item.processed_path),
                }
            )


def read_manifest(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            items.append(
                DatasetItem(
                    split=row["split"],
                    label=int(row["label"]),
                    augmentation=row["augmentation"],
                    source_path=Path(row["source_path"]),
                    processed_path=Path(row["processed_path"]),
                )
            )
    return items


def load_split(items: list[DatasetItem], split: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    split_items = [item for item in items if item.split == split]
    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []
    for item in split_items:
        image = cv2.imread(str(item.processed_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read processed image: {item.processed_path}")
        images.append(image[None])
        labels.append(item.label)
        paths.append(item.processed_path)
    return np.stack(images).astype(np.uint8), np.asarray(labels, dtype=np.float32), paths


def init_params(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": (rng.normal(0.0, np.sqrt(2.0 / 25.0), size=(8, 1, 5, 5))).astype(np.float32),
        "b1": np.zeros((8,), dtype=np.float32),
        "w2": (rng.normal(0.0, np.sqrt(2.0 / (8 * 9)), size=(16, 8, 3, 3))).astype(np.float32),
        "b2": np.zeros((16,), dtype=np.float32),
        "w3": (rng.normal(0.0, np.sqrt(2.0 / (16 * 9)), size=(24, 16, 3, 3))).astype(np.float32),
        "b3": np.zeros((24,), dtype=np.float32),
        "w_fc": (rng.normal(0.0, np.sqrt(2.0 / 24.0), size=(24, 1))).astype(np.float32),
        "b_fc": np.zeros((1,), dtype=np.float32),
    }


def avg_pool_factor(x: np.ndarray, factor: int) -> np.ndarray:
    n, c, h, w = x.shape
    if h % factor or w % factor:
        raise ValueError(f"Input shape {(h, w)} is not divisible by pool factor {factor}")
    return x.reshape(n, c, h // factor, factor, w // factor, factor).mean(axis=(3, 5))


def avg_pool_factor_backward(dout: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(dout, factor, axis=2), factor, axis=3) / float(factor * factor)


def conv2d_same_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, pad: int) -> np.ndarray:
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    kh, kw = weight.shape[2], weight.shape[3]
    windows = np.lib.stride_tricks.sliding_window_view(x_padded, (kh, kw), axis=(2, 3))
    return np.einsum("nchwkl,fckl->nfhw", windows, weight, optimize=True) + bias[None, :, None, None]


def conv2d_same_backward(
    dout: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    pad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    kh, kw = weight.shape[2], weight.shape[3]
    windows = np.lib.stride_tricks.sliding_window_view(x_padded, (kh, kw), axis=(2, 3))
    dweight = np.einsum("nfhw,nchwkl->fckl", dout, windows, optimize=True)
    dbias = dout.sum(axis=(0, 2, 3))

    dx_padded = np.zeros_like(x_padded)
    for row in range(kh):
        for col in range(kw):
            dx_padded[:, :, row : row + dout.shape[2], col : col + dout.shape[3]] += np.einsum(
                "nfhw,fc->nchw",
                dout,
                weight[:, :, row, col],
                optimize=True,
            )
    if pad == 0:
        return dx_padded, dweight, dbias
    return dx_padded[:, :, pad:-pad, pad:-pad], dweight, dbias


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
    x_down = avg_pool_factor(x, 4)
    z1 = conv2d_same_forward(x_down, params["w1"], params["b1"], pad=2)
    a1 = relu(z1)
    p1 = avg_pool_factor(a1, 2)

    z2 = conv2d_same_forward(p1, params["w2"], params["b2"], pad=1)
    a2 = relu(z2)
    p2 = avg_pool_factor(a2, 2)

    z3 = conv2d_same_forward(p2, params["w3"], params["b3"], pad=1)
    a3 = relu(z3)
    gap = a3.mean(axis=(2, 3))
    logits = gap @ params["w_fc"] + params["b_fc"]
    return logits.reshape(-1), ForwardCache(x_down, z1, a1, p1, z2, a2, p2, z3, a3, gap)


def predict_logits(params: dict[str, np.ndarray], x: np.ndarray, batch_size: int) -> np.ndarray:
    logits: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].astype(np.float32) / 255.0
        batch_logits, _ = forward(params, batch)
        logits.append(batch_logits)
    return np.concatenate(logits)


def binary_cross_entropy_with_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    logits64 = logits.astype(np.float64)
    labels64 = labels.astype(np.float64)
    weights64 = sample_weights.astype(np.float64)
    losses = np.maximum(logits64, 0.0) - logits64 * labels64 + np.log1p(np.exp(-np.abs(logits64)))
    normalizer = max(float(weights64.sum()), 1e-8)
    loss = float((losses * weights64).sum() / normalizer)
    probs = sigmoid(logits64)
    dlogits = ((probs - labels64) * weights64 / normalizer).astype(np.float32)
    return loss, dlogits


def l2_loss_and_grads(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    weight_decay: float,
) -> float:
    if weight_decay <= 0.0:
        return 0.0
    loss = 0.0
    for key, value in params.items():
        if not key.startswith("w"):
            continue
        loss += 0.5 * weight_decay * float(np.sum(value * value))
        grads[key] += weight_decay * value
    return loss


def backward(
    params: dict[str, np.ndarray],
    cache: ForwardCache,
    dlogits: np.ndarray,
    weight_decay: float,
) -> tuple[dict[str, np.ndarray], float]:
    grads = {key: np.zeros_like(value) for key, value in params.items()}

    dlogits_2d = dlogits[:, None]
    grads["w_fc"] = cache.gap.T @ dlogits_2d
    grads["b_fc"] = dlogits_2d.sum(axis=0)

    dgap = dlogits_2d @ params["w_fc"].T
    da3 = np.repeat(np.repeat(dgap[:, :, None, None], cache.a3.shape[2], axis=2), cache.a3.shape[3], axis=3)
    da3 = da3 / float(cache.a3.shape[2] * cache.a3.shape[3])

    dz3 = da3 * (cache.z3 > 0.0)
    dp2, grads["w3"], grads["b3"] = conv2d_same_backward(dz3, cache.p2, params["w3"], pad=1)

    da2 = avg_pool_factor_backward(dp2, 2)
    dz2 = da2 * (cache.z2 > 0.0)
    dp1, grads["w2"], grads["b2"] = conv2d_same_backward(dz2, cache.p1, params["w2"], pad=1)

    da1 = avg_pool_factor_backward(dp1, 2)
    dz1 = da1 * (cache.z1 > 0.0)
    _, grads["w1"], grads["b1"] = conv2d_same_backward(dz1, cache.x_down, params["w1"], pad=2)

    l2_loss = l2_loss_and_grads(params, grads, weight_decay)
    return grads, l2_loss


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, dict[str, np.ndarray] | int],
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    state["step"] = int(state["step"]) + 1
    step = int(state["step"])
    m = state["m"]
    v = state["v"]
    assert isinstance(m, dict)
    assert isinstance(v, dict)

    for key in params:
        m[key] = beta1 * m[key] + (1.0 - beta1) * grads[key]
        v[key] = beta2 * v[key] + (1.0 - beta2) * (grads[key] * grads[key])
        m_hat = m[key] / (1.0 - beta1**step)
        v_hat = v[key] / (1.0 - beta2**step)
        params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def make_adam_state(params: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray] | int]:
    return {
        "step": 0,
        "m": {key: np.zeros_like(value) for key, value in params.items()},
        "v": {key: np.zeros_like(value) for key, value in params.items()},
    }


def class_sample_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(labels, dtype=np.float32)
    positives = float(np.sum(labels == 1.0))
    negatives = float(np.sum(labels == 0.0))
    total = positives + negatives
    pos_weight = total / max(2.0 * positives, 1.0)
    neg_weight = total / max(2.0 * negatives, 1.0)
    return np.where(labels == 1.0, pos_weight, neg_weight).astype(np.float32)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    probs = sigmoid(logits)
    predictions = (probs >= threshold).astype(np.int32)
    labels_i = labels.astype(np.int32)
    tp = int(np.sum((predictions == 1) & (labels_i == 1)))
    tn = int(np.sum((predictions == 0) & (labels_i == 0)))
    fp = int(np.sum((predictions == 1) & (labels_i == 0)))
    fn = int(np.sum((predictions == 0) & (labels_i == 1)))

    accuracy = (tp + tn) / max(len(labels_i), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(probs, labels_i),
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return None

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum = float(ranks[positives].sum())
    return (rank_sum - pos_count * (pos_count + 1) / 2.0) / float(pos_count * neg_count)


def best_threshold_for_f1(logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, object]]:
    best_threshold = 0.5
    best_metrics = metrics_from_logits(logits, labels, threshold=0.5)
    for threshold in np.linspace(0.05, 0.95, 91):
        current = metrics_from_logits(logits, labels, threshold=float(threshold))
        if float(current["f1"]) > float(best_metrics["f1"]):
            best_threshold = float(threshold)
            best_metrics = current
    return best_threshold, best_metrics


def train(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    rng = np.random.default_rng(args.seed)
    state = make_adam_state(params)
    weights = class_sample_weights(train_labels, args.class_weight)
    best_params = {key: value.copy() for key, value in params.items()}
    best_val_loss = float("inf")
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        indices = np.arange(len(train_images))
        rng.shuffle(indices)
        train_losses: list[float] = []

        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            x_batch = train_images[batch_indices].astype(np.float32) / 255.0
            y_batch = train_labels[batch_indices]
            w_batch = weights[batch_indices]

            logits, cache = forward(params, x_batch)
            bce_loss, dlogits = binary_cross_entropy_with_logits(logits, y_batch, w_batch)
            grads, l2_loss = backward(params, cache, dlogits, args.weight_decay)
            adam_update(params, grads, state, args.learning_rate)
            train_losses.append(bce_loss + l2_loss)

        val_logits = predict_logits(params, val_images, args.batch_size)
        val_weights = np.ones_like(val_labels, dtype=np.float32)
        val_loss, _ = binary_cross_entropy_with_logits(val_logits, val_labels, val_weights)
        val_metrics = metrics_from_logits(val_logits, val_labels)
        train_logits = predict_logits(params, train_images, args.batch_size)
        train_metrics = metrics_from_logits(train_logits, train_labels)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "seconds": round(time.time() - started, 3),
        }
        history.append(row)
        print(
            "epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            "val_acc={val_accuracy:.3f} val_f1={val_f1:.3f} val_auc={val_roc_auc}".format(**row),
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {key: value.copy() for key, value in params.items()}

    final_logits = predict_logits(best_params, val_images, args.batch_size)
    threshold, threshold_metrics = best_threshold_for_f1(final_logits, val_labels)
    final_metrics = {
        "best_epoch_by_val_loss": int(min(history, key=lambda row: float(row["val_loss"]))["epoch"]),
        "val_loss": best_val_loss,
        "default_threshold_metrics": metrics_from_logits(final_logits, val_labels, threshold=0.5),
        "best_f1_threshold": threshold,
        "best_threshold_metrics": threshold_metrics,
    }
    return best_params, final_metrics, history


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def split_counts(items: list[DatasetItem]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in items:
        split_counts_for_label = counts.setdefault(item.split, {"0": 0, "1": 0})
        split_counts_for_label[str(item.label)] += 1
    return counts


def save_model(
    path: Path,
    params: dict[str, np.ndarray],
    image_size: int,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **params,
        image_size=np.asarray([image_size], dtype=np.int32),
        input_mode=np.asarray(["grayscale"]),
        downsample_factor=np.asarray([4], dtype=np.int32),
        decision_threshold=np.asarray([threshold], dtype=np.float32),
        model_name=np.asarray(["numpy_quality_cnn_v1"]),
    )


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = args.processed_dir or (args.out_dir / f"prepared_{args.image_size}_gray")
    model_out = args.model_out or (args.out_dir / "lenta_quality_cnn.npz")

    items = prepare_dataset(
        dataset_dir=args.dataset,
        processed_dir=processed_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        reuse_processed=args.reuse_processed,
    )
    train_images, train_labels, _ = load_split(items, "train")
    val_images, val_labels, _ = load_split(items, "val")

    rng = np.random.default_rng(args.seed)
    params = init_params(rng)
    trained_params, metrics, history = train(params, train_images, train_labels, val_images, val_labels, args)

    save_model(model_out, trained_params, args.image_size, float(metrics["best_f1_threshold"]))
    write_history(args.out_dir / "training_log.csv", history)

    metrics_payload = {
        "dataset": str(args.dataset),
        "processed_dir": str(processed_dir),
        "model_path": str(model_out),
        "image_size": args.image_size,
        "input_mode": "grayscale",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "class_weight": args.class_weight,
        "split_counts_after_augmentation": split_counts(items),
        "metrics": metrics,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics_payload["metrics"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
