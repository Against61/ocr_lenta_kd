from __future__ import annotations

import argparse

import cv2
import numpy as np


def normalize_orientation(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.no_rotate:
        return frame
    if args.rotate_cw:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if args.rotate_ccw:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def resize_to_width(frame: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w == width:
        return frame, 1.0
    scale = width / float(w)
    resized = cv2.resize(frame, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    ratio = min(size / float(w), size / float(h))
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, ratio, pad_x, pad_y


def crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return frame[y : y + h, x : x + w]


def laplacian_sharpness(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def red_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_1 = np.array([0, 35, 95], dtype=np.uint8)
    upper_1 = np.array([17, 255, 255], dtype=np.uint8)
    lower_2 = np.array([168, 35, 95], dtype=np.uint8)
    upper_2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_1, upper_1) | cv2.inRange(hsv, lower_2, upper_2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def white_ratio(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > 142) & (hsv[:, :, 1] < 95)
    return float(np.count_nonzero(white)) / float(white.size)


def upper_white_ratio(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    upper = frame[: max(1, int(round(frame.shape[0] * 0.62))), :]
    return white_ratio(upper)


def red_ratio_from_mask(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    roi = mask[y : y + h, x : x + w]
    return float(np.count_nonzero(roi)) / float(max(1, w * h))
