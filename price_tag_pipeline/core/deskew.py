from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from price_tag_pipeline.core.image import red_mask
from price_tag_pipeline.tracker.models import Track


@dataclass
class DeskewResult:
    image: np.ndarray
    angle: float = 0.0
    confidence: float = 0.0
    applied: bool = False
    method: str = "none"


def apply_deskew_to_tracks(
    tracks: list[Track],
    enabled: bool,
    max_angle: float,
    min_angle: float,
    min_confidence: float,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "enabled": enabled,
        "candidate_count": 0,
        "applied_count": 0,
        "skipped_count": 0,
        "methods": {},
    }
    if not enabled:
        return summary

    methods: dict[str, int] = {}
    for track in tracks:
        for candidate in track.crop_candidates:
            summary["candidate_count"] = int(summary["candidate_count"]) + 1
            result = deskew_crop(
                candidate.crop,
                max_angle=max_angle,
                min_angle=min_angle,
                min_confidence=min_confidence,
            )
            candidate.crop = result.image
            candidate.deskew_angle = result.angle
            candidate.deskew_confidence = result.confidence
            candidate.deskew_applied = result.applied
            candidate.deskew_method = result.method

            if result.applied:
                summary["applied_count"] = int(summary["applied_count"]) + 1
            else:
                summary["skipped_count"] = int(summary["skipped_count"]) + 1
            methods[result.method] = methods.get(result.method, 0) + 1

    summary["methods"] = methods
    return summary


def deskew_crop(
    crop: np.ndarray,
    max_angle: float,
    min_angle: float,
    min_confidence: float,
) -> DeskewResult:
    if crop.size == 0:
        return DeskewResult(image=crop, method="empty")

    estimate = estimate_red_region_angle(crop, max_angle)
    if estimate is None:
        estimate = estimate_hough_angle(crop, max_angle)
    if estimate is None:
        return DeskewResult(image=crop, method="no_estimate")

    angle, confidence, method = estimate
    if confidence < min_confidence:
        return DeskewResult(image=crop, angle=angle, confidence=confidence, method=f"{method}:low_confidence")
    if abs(angle) < min_angle:
        return DeskewResult(image=crop, angle=angle, confidence=confidence, method=f"{method}:small_angle")

    rotated = rotate_affine(crop, angle)
    return DeskewResult(
        image=rotated,
        angle=angle,
        confidence=confidence,
        applied=True,
        method=method,
    )


def estimate_red_region_angle(crop: np.ndarray, max_angle: float) -> tuple[float, float, str] | None:
    mask = red_mask(crop)
    points = cv2.findNonZero(mask)
    if points is None:
        return None

    crop_area = float(crop.shape[0] * crop.shape[1])
    red_pixels = float(len(points))
    if red_pixels < max(24.0, crop_area * 0.006):
        return None

    rect = cv2.minAreaRect(points)
    angle = normalize_min_area_angle(rect)
    if abs(angle) > max_angle:
        return None

    confidence = min(1.0, red_pixels / max(1.0, crop_area * 0.08))
    return angle, confidence, "red_region"


def estimate_hough_angle(crop: np.ndarray, max_angle: float) -> tuple[float, float, str] | None:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    h, w = crop.shape[:2]
    min_line_length = max(12, int(round(w * 0.28)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=20,
        minLineLength=min_line_length,
        maxLineGap=max(4, int(round(w * 0.06))),
    )
    if lines is None:
        return None

    weighted_angles: list[tuple[float, float]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(value) for value in line]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_line_length:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        angle = normalize_angle(angle)
        if abs(angle) <= max_angle:
            weighted_angles.append((angle, length))

    if not weighted_angles:
        return None

    total_weight = sum(weight for _, weight in weighted_angles)
    angle = sum(angle * weight for angle, weight in weighted_angles) / max(1.0, total_weight)
    confidence = min(1.0, total_weight / max(1.0, w * 1.5))
    return angle, confidence, "hough_lines"


def normalize_min_area_angle(rect) -> float:
    (_, _), (w, h), angle = rect
    if w < h:
        angle += 90.0
    return normalize_angle(float(angle))


def normalize_angle(angle: float) -> float:
    while angle <= -45.0:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    return angle


def rotate_affine(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
