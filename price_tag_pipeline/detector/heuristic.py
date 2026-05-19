from __future__ import annotations

import cv2
import numpy as np

from price_tag_pipeline.core.geometry import box_iou, clamp_box
from price_tag_pipeline.core.image import crop, red_mask, red_ratio_from_mask, upper_white_ratio, white_ratio
from price_tag_pipeline.detector.types import Detection


class HeuristicPriceTagDetector:
    def predict(self, frame: np.ndarray) -> list[Detection]:
        return detect_price_tags(frame)


def expand_red_to_tag_box(red_box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = red_box
    tag_w = int(round(np.clip(max(w * 2.35, w + 42), 44, 170)))
    tag_h = int(round(np.clip(max(h * 2.10, h + 24), 30, 92)))

    # In normalized orientation the red price area is usually in the lower half.
    x1 = int(round(x - 0.34 * w))
    y1 = int(round(y - 0.78 * h))
    return clamp_box((x1, y1, tag_w, tag_h), width, height)


def detect_price_tags(frame: np.ndarray) -> list[Detection]:
    h, w = frame.shape[:2]
    mask = red_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Detection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 30 or area > 5000:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 5 or bh < 5:
            continue
        aspect = bw / float(bh)
        extent = area / float(max(1, bw * bh))
        if not (0.18 <= aspect <= 5.8) or extent < 0.18:
            continue

        tag_box = expand_red_to_tag_box((x, y, bw, bh), w, h)
        _, _, tag_w, tag_h = tag_box
        tag_aspect = tag_w / float(tag_h)
        if tag_aspect < 1.35 or tag_h > 96:
            continue

        tag_patch = crop(frame, tag_box)
        wr = white_ratio(tag_patch)
        uwr = upper_white_ratio(tag_patch)
        rr = red_ratio_from_mask(mask, tag_box)

        if wr < 0.14 or uwr < 0.18 or rr < 0.025 or rr > 0.62:
            continue

        score = float(np.clip(0.45 * wr + 0.35 * uwr + 1.25 * rr + 0.15 * min(1.0, extent), 0.0, 1.0))
        candidates.append(
            Detection(
                bbox=tag_box,
                red_bbox=(x, y, bw, bh),
                score=score,
                white_ratio=wr,
                upper_white_ratio=uwr,
                red_ratio=rr,
            )
        )

    return nms_detections(candidates)


def nms_detections(detections: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.score, reverse=True):
        cx, cy = det.center
        duplicate = False
        for existing in kept:
            ex, ey = existing.center
            center_dist = float(np.hypot(cx - ex, cy - ey))
            if box_iou(det.bbox, existing.bbox) > 0.22 or center_dist < 20:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept

