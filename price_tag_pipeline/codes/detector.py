"""NCNN region detector for QR and barcode zones inside a price-tag crop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASS_NAMES = {
    0: "qr",
    1: "barcode",
    2: "matrix",
}

DEFAULT_CLASS_ROLES = {
    0: "qr",
    1: "barcode",
    2: "matrix",
}


@dataclass(frozen=True)
class CodeRegion:
    bbox: tuple[int, int, int, int]
    score: float
    class_id: int
    label: str
    role: str = ""


@dataclass(frozen=True)
class CodeRegionDetectorConfig:
    model_dir: Path
    param_path: Path | None = None
    bin_path: Path | None = None
    input_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    padding_ratio: float = 0.50
    resize_mode: str = "stretch"
    fallback_full_crop: bool = True
    class_names: dict[int, str] | None = None
    class_roles: dict[int, str] | None = None


class CodeRegionDetectorUnavailableError(RuntimeError):
    """Raised when the configured NCNN detector cannot be initialized."""


class NcnnCodeRegionDetector:
    def __init__(self, config: CodeRegionDetectorConfig) -> None:
        self.config = config
        self.param_path, self.bin_path = resolve_ncnn_model_files(
            model_dir=config.model_dir,
            param_path=config.param_path,
            bin_path=config.bin_path,
        )
        try:
            import ncnn
        except ImportError as exc:  # pragma: no cover - depends on runtime image.
            raise CodeRegionDetectorUnavailableError("Python package 'ncnn' is not installed") from exc

        self._ncnn = ncnn
        self._net = ncnn.Net()
        self._net.opt.num_threads = 1
        param_status = self._net.load_param(str(self.param_path))
        model_status = self._net.load_model(str(self.bin_path))
        if param_status != 0 or model_status != 0:
            raise CodeRegionDetectorUnavailableError(
                f"failed to load NCNN model: param_status={param_status}, model_status={model_status}"
            )

    @property
    def enabled(self) -> bool:
        return True

    def detect_file(self, image_path: Path) -> list[CodeRegion]:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        return self.detect(image)

    def detect(self, image: np.ndarray) -> list[CodeRegion]:
        if image.size == 0:
            return []

        input_image, transform = _prepare_input(image, self.config.input_size, self.config.resize_mode)
        mat = self._ncnn.Mat.from_pixels_resize(
            input_image,
            self._ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            input_image.shape[1],
            input_image.shape[0],
            self.config.input_size,
            self.config.input_size,
        )
        mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])

        extractor = self._net.create_extractor()
        extractor.input("in0", mat)
        _, output = extractor.extract("out0")
        raw_output = np.array(output)
        class_names = self.config.class_names or DEFAULT_CLASS_NAMES
        class_roles = self.config.class_roles or DEFAULT_CLASS_ROLES
        return decode_yolo_ncnn_output(
            raw_output,
            image_shape=image.shape[:2],
            input_size=self.config.input_size,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            transform=transform,
            class_names=class_names,
            class_roles=class_roles,
        )


def build_code_region_detector(config: CodeRegionDetectorConfig | None) -> NcnnCodeRegionDetector | None:
    if config is None:
        return None
    return NcnnCodeRegionDetector(config)


def resolve_ncnn_model_files(
    *,
    model_dir: Path,
    param_path: Path | None,
    bin_path: Path | None,
) -> tuple[Path, Path]:
    resolved_param = param_path
    resolved_bin = bin_path

    if resolved_param is None:
        for candidate in ("model-opt.param", "model.ncnn.param"):
            path = model_dir / candidate
            if path.exists():
                resolved_param = path
                break
    if resolved_bin is None:
        for candidate in ("model-opt.bin", "model.ncnn.bin"):
            path = model_dir / candidate
            if path.exists():
                resolved_bin = path
                break

    if resolved_param is None or not resolved_param.exists():
        raise CodeRegionDetectorUnavailableError(f"NCNN param file not found in {model_dir}")
    if resolved_bin is None or not resolved_bin.exists():
        raise CodeRegionDetectorUnavailableError(f"NCNN bin file not found in {model_dir}")
    return resolved_param, resolved_bin


def crop_region(
    image: np.ndarray,
    region: CodeRegion,
    *,
    padding_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, width, height = region.bbox
    pad_x = int(round(width * padding_ratio))
    pad_y = int(round(height * padding_ratio))
    image_height, image_width = image.shape[:2]

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + width + pad_x)
    y2 = min(image_height, y + height + pad_y)
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0], (0, 0, 0, 0)
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)


def decode_yolo_ncnn_output(
    raw_output: np.ndarray,
    *,
    image_shape: tuple[int, int],
    input_size: int,
    conf_threshold: float,
    iou_threshold: float,
    transform: tuple[str, float, int, int] | None = None,
    class_names: dict[int, str] | None = None,
    class_roles: dict[int, str] | None = None,
) -> list[CodeRegion]:
    output = np.squeeze(np.asarray(raw_output, dtype=np.float32))
    if output.ndim != 2:
        return []
    if output.shape[0] < output.shape[1] and output.shape[0] >= 5:
        output = output.T
    if output.shape[1] < 5:
        return []

    boxes_xywh = output[:, :4]
    scores = output[:, 4:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]

    candidates: list[CodeRegion] = []
    image_height, image_width = image_shape
    class_names = class_names or DEFAULT_CLASS_NAMES
    class_roles = class_roles or DEFAULT_CLASS_ROLES

    for box, confidence, class_id in zip(boxes_xywh, confidences, class_ids, strict=False):
        score = float(confidence)
        if score < conf_threshold:
            continue

        cx, cy, width, height = (float(value) for value in box)
        x1 = cx - width / 2.0
        y1 = cy - height / 2.0
        x2 = cx + width / 2.0
        y2 = cy + height / 2.0
        x1, y1, x2, y2 = _map_input_box_to_image(
            (x1, y1, x2, y2),
            image_width=image_width,
            image_height=image_height,
            input_size=input_size,
            transform=transform,
        )
        clipped = _clip_xyxy(x1, y1, x2, y2, image_width, image_height)
        if clipped is None:
            continue

        x_min, y_min, x_max, y_max = clipped
        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width <= 1 or box_height <= 1:
            continue

        resolved_class_id = int(class_id)
        candidates.append(
            CodeRegion(
                bbox=(x_min, y_min, box_width, box_height),
                score=score,
                class_id=resolved_class_id,
                label=class_names.get(resolved_class_id, f"class_{resolved_class_id}"),
                role=class_roles.get(resolved_class_id, ""),
            )
        )

    return _nms(candidates, iou_threshold)


def _prepare_input(
    image: np.ndarray,
    input_size: int,
    resize_mode: str,
) -> tuple[np.ndarray, tuple[str, float, int, int] | None]:
    if resize_mode == "letterbox":
        image_height, image_width = image.shape[:2]
        scale = min(input_size / image_width, input_size / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_x = (input_size - resized_width) // 2
        pad_y = (input_size - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, ("letterbox", scale, pad_x, pad_y)
    return image, None


def _map_input_box_to_image(
    box: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    input_size: int,
    transform: tuple[str, float, int, int] | None,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    if transform is not None and transform[0] == "letterbox":
        _, scale, pad_x, pad_y = transform
        return (
            (x1 - pad_x) / scale,
            (y1 - pad_y) / scale,
            (x2 - pad_x) / scale,
            (y2 - pad_y) / scale,
        )
    return (
        x1 * image_width / input_size,
        y1 * image_height / input_size,
        x2 * image_width / input_size,
        y2 * image_height / input_size,
    )


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    x_min = max(0, min(image_width, int(round(x1))))
    y_min = max(0, min(image_height, int(round(y1))))
    x_max = max(0, min(image_width, int(round(x2))))
    y_max = max(0, min(image_height, int(round(y2))))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _nms(regions: list[CodeRegion], iou_threshold: float) -> list[CodeRegion]:
    selected: list[CodeRegion] = []
    for region in sorted(regions, key=lambda item: item.score, reverse=True):
        if any(
            region.class_id == kept.class_id and _iou(region.bbox, kept.bbox) > iou_threshold
            for kept in selected
        ):
            continue
        selected.append(region)
    return selected


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union
