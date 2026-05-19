from __future__ import annotations

from pathlib import Path

import numpy as np

from price_tag_pipeline.core.geometry import clamp_box, nms_xyxy
from price_tag_pipeline.core.image import crop, letterbox, red_mask, red_ratio_from_mask, upper_white_ratio, white_ratio
from price_tag_pipeline.detector.types import Detection


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: Path,
        input_size: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        import onnxruntime as ort

        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame: np.ndarray) -> list[Detection]:
        image, ratio, pad_x, pad_y = letterbox(frame, self.input_size)
        tensor = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: tensor[None]})[0]
        predictions = output[0].T

        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        keep = scores >= self.conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        if len(boxes) == 0:
            return []

        h, w = frame.shape[:2]
        mask = red_mask(frame)
        xyxy = np.zeros_like(boxes, dtype=np.float32)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / ratio
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)

        detections: list[Detection] = []
        for idx in nms_xyxy(xyxy, scores, self.iou_threshold):
            x1, y1, x2, y2 = xyxy[idx]
            bw = max(1, int(round(x2 - x1)))
            bh = max(1, int(round(y2 - y1)))
            box = clamp_box((int(round(x1)), int(round(y1)), bw, bh), w, h)
            patch = crop(frame, box)
            detections.append(
                Detection(
                    bbox=box,
                    red_bbox=box,
                    score=float(scores[idx]),
                    white_ratio=white_ratio(patch),
                    upper_white_ratio=upper_white_ratio(patch),
                    red_ratio=red_ratio_from_mask(mask, box),
                )
            )
        return detections
