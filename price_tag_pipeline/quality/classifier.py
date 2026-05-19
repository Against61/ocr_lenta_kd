from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CropQualityResult:
    label: int
    score: float
    passed: bool


class NoopCropQualityClassifier:
    enabled = False
    model_path = ""
    model_type = "none"
    input_size = ""
    threshold = ""
    pass_class = ""
    input_mode = ""
    normalization = ""

    def predict(self, crop: np.ndarray) -> CropQualityResult:
        return CropQualityResult(label=1, score=1.0, passed=True)


class OnnxCropQualityClassifier:
    enabled = True
    model_type = "onnx"

    def __init__(
        self,
        model_path: Path,
        input_size: int | None,
        threshold: float | None,
        pass_class: int,
        input_mode: str | None,
        normalization: str,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Quality classifier model not found: {model_path}")

        import onnxruntime as ort

        self.model_path = str(model_path)
        self.input_size = input_size or 224
        self.threshold = 0.5 if threshold is None else threshold
        self.pass_class = pass_class
        self.input_mode = input_mode or "grayscale"
        self.normalization = normalization
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, crop: np.ndarray) -> CropQualityResult:
        tensor = self._preprocess(crop)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode(output)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        image = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if self.input_mode == "grayscale":
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.astype(np.float32) / 255.0
            return image[None, None]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.normalization == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std
        return image.transpose(2, 0, 1)[None]

    def _decode(self, output: np.ndarray) -> CropQualityResult:
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return CropQualityResult(label=0, score=0.0, passed=False)

        if values.size == 1:
            prob_1 = scalar_to_probability(float(values[0]))
            label = 1 if prob_1 >= self.threshold else 0
            score = prob_1 if self.pass_class == 1 else 1.0 - prob_1
            return CropQualityResult(label=label, score=score, passed=score >= self.threshold)

        probs = logits_or_probs_to_probs(values)
        pass_class = min(max(self.pass_class, 0), len(probs) - 1)
        label = int(np.argmax(probs))
        score = float(probs[pass_class])
        return CropQualityResult(label=label, score=score, passed=score >= self.threshold)


class NumpyCnnCropQualityClassifier:
    enabled = True
    model_type = "npz"

    def __init__(
        self,
        model_path: Path,
        input_size: int | None,
        threshold: float | None,
        pass_class: int,
        input_mode: str | None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Quality classifier model not found: {model_path}")

        required = ("w1", "b1", "w2", "b2", "w3", "b3", "w_fc", "b_fc")
        with np.load(model_path, allow_pickle=False) as loaded:
            missing = [key for key in required if key not in loaded]
            if missing:
                raise ValueError(f"Numpy quality CNN is missing weights: {missing}")

            model_input_size = int(loaded["image_size"][0]) if "image_size" in loaded else 224
            model_input_mode = str(loaded["input_mode"][0]) if "input_mode" in loaded else "grayscale"
            model_threshold = float(loaded["decision_threshold"][0]) if "decision_threshold" in loaded else 0.5
            self.downsample_factor = int(loaded["downsample_factor"][0]) if "downsample_factor" in loaded else 4
            self.model_name = str(loaded["model_name"][0]) if "model_name" in loaded else "numpy_quality_cnn"
            self.params = {key: loaded[key].astype(np.float32) for key in required}

        resolved_input_mode = input_mode or model_input_mode
        if resolved_input_mode != "grayscale":
            raise ValueError("Numpy quality CNN supports only grayscale input mode")

        self.model_path = str(model_path)
        self.input_size = input_size or model_input_size
        self.threshold = model_threshold if threshold is None else threshold
        self.pass_class = pass_class
        self.input_mode = resolved_input_mode
        self.normalization = "none"

    def predict(self, crop: np.ndarray) -> CropQualityResult:
        tensor = self._preprocess(crop)
        prob_1 = float(sigmoid(np.asarray([self._forward_logit(tensor)], dtype=np.float32))[0])
        label = 1 if prob_1 >= self.threshold else 0
        score = prob_1 if self.pass_class == 1 else 1.0 - prob_1
        return CropQualityResult(label=label, score=score, passed=score >= self.threshold)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        image = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32) / 255.0
        return image[None, None]

    def _forward_logit(self, x: np.ndarray) -> float:
        x_down = avg_pool_factor(x, self.downsample_factor)
        z1 = conv2d_same_forward(x_down, self.params["w1"], self.params["b1"], pad=2)
        a1 = np.maximum(z1, 0.0)
        p1 = avg_pool_factor(a1, 2)

        z2 = conv2d_same_forward(p1, self.params["w2"], self.params["b2"], pad=1)
        a2 = np.maximum(z2, 0.0)
        p2 = avg_pool_factor(a2, 2)

        z3 = conv2d_same_forward(p2, self.params["w3"], self.params["b3"], pad=1)
        a3 = np.maximum(z3, 0.0)
        gap = a3.mean(axis=(2, 3))
        logits = gap @ self.params["w_fc"] + self.params["b_fc"]
        return float(logits.reshape(-1)[0])


def avg_pool_factor(x: np.ndarray, factor: int) -> np.ndarray:
    n, c, h, w = x.shape
    return x.reshape(n, c, h // factor, factor, w // factor, factor).mean(axis=(3, 5))


def conv2d_same_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, pad: int) -> np.ndarray:
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    kh, kw = weight.shape[2], weight.shape[3]
    windows = np.lib.stride_tricks.sliding_window_view(x_padded, (kh, kw), axis=(2, 3))
    return np.einsum("nchwkl,fckl->nfhw", windows, weight, optimize=True) + bias[None, :, None, None]


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def scalar_to_probability(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value
    return float(1.0 / (1.0 + np.exp(-value)))


def logits_or_probs_to_probs(values: np.ndarray) -> np.ndarray:
    if np.all(values >= 0.0) and np.all(values <= 1.0):
        total = float(values.sum())
        if total > 0:
            return values / total

    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    return exp_values / float(exp_values.sum())


def build_quality_classifier(
    args: argparse.Namespace,
) -> NoopCropQualityClassifier | OnnxCropQualityClassifier | NumpyCnnCropQualityClassifier:
    if args.quality_model is None:
        return NoopCropQualityClassifier()
    if args.quality_model.suffix.lower() == ".npz":
        return NumpyCnnCropQualityClassifier(
            model_path=args.quality_model,
            input_size=args.quality_input_size,
            threshold=args.quality_threshold,
            pass_class=args.quality_pass_class,
            input_mode=args.quality_input_mode,
        )
    return OnnxCropQualityClassifier(
        model_path=args.quality_model,
        input_size=args.quality_input_size,
        threshold=args.quality_threshold,
        pass_class=args.quality_pass_class,
        input_mode=args.quality_input_mode,
        normalization=args.quality_normalization,
    )
