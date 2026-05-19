"""Barcode and QR-code extraction for selected price-tag crops."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency in normal pipeline installs.
    cv2 = None
    np = None

try:
    from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
except ImportError:  # pragma: no cover - OpenCV fallback still keeps the reader usable.
    Image = None
    ImageFilter = None
    ImageOps = None
    UnidentifiedImageError = OSError

try:
    from pyzbar.pyzbar import ZBarSymbol, decode as decode_zbar
except Exception:  # pragma: no cover - pyzbar often needs system zbar libraries.
    ZBarSymbol = None
    decode_zbar = None

try:
    import zxingcpp
except ImportError:  # pragma: no cover - optional extra decoder.
    zxingcpp = None


QR_OUTPUT_FIELDS = (
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
)

QR_FIELD_ALIASES = {
    "qr_code_barcode": ("qr_code_barcode", "barcode", "b"),
    "price1_qr": ("price1_qr", "price1", "p1"),
    "price2_qr": ("price2_qr", "price2", "p2"),
    "price3_qr": ("price3_qr", "price3", "p3"),
    "price4_qr": ("price4_qr", "price4", "p4"),
    "wholesale_level_1_count": (
        "wholesale_level_1_count",
        "wholesale_level_1_coun",
        "wholesaleLevel1Count",
        "wL1C",
    ),
    "wholesale_level_1_price": (
        "wholesale_level_1_price",
        "wholesaleLevel1Price",
        "wL1P",
    ),
    "wholesale_level_2_count": (
        "wholesale_level_2_count",
        "wholesaleLevel2Count",
        "wL2C",
    ),
    "wholesale_level_2_price": (
        "wholesale_level_2_price",
        "wholesaleLevel2Price",
        "wL2P",
    ),
    "action_price_qr": ("action_price_qr", "actionPrice", "aP"),
    "action_code_qr": ("action_code_qr", "actionCode", "aC"),
}

MAX_IMAGE_PIXELS = 50_000_000
MAX_CANDIDATE_PIXELS = 5_000_000
MAX_UPSCALE_FACTOR = 4.0
SCAN_WIDTHS = (350, 700, 1200)
OPENCV_WARP_SIZES = (250, 500, 800)
OPENCV_WARP_EXPAND_FACTORS = (1.0, 1.1, 1.25)

CROP_RATIOS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("original", (0.0, 0.0, 1.0, 1.0)),
    ("upper_half", (0.0, 0.0, 1.0, 0.5)),
    ("upper_third", (0.0, 0.0, 1.0, 0.33)),
    ("middle_60", (0.0, 0.2, 1.0, 0.8)),
    ("lower_half", (0.0, 0.5, 1.0, 1.0)),
    ("left_half", (0.0, 0.0, 0.5, 1.0)),
    ("right_half", (0.5, 0.0, 1.0, 1.0)),
    ("upper_left", (0.0, 0.0, 0.55, 0.55)),
    ("upper_right", (0.45, 0.0, 1.0, 0.55)),
    ("lower_left", (0.0, 0.45, 0.55, 1.0)),
    ("lower_right", (0.45, 0.45, 1.0, 1.0)),
    ("center", (0.2, 0.2, 0.8, 0.8)),
)


@dataclass
class CodeDetection:
    source_path: Path
    code_type: str
    value: str
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    parsed_fields: dict[str, str] = field(default_factory=dict)
    scan_source: str = ""
    detected: bool = True
    detection_source: str = ""
    region_bbox: tuple[int, int, int, int] | None = None
    region_class_id: int | None = None
    region_label: str = ""
    region_confidence: float = 0.0


@dataclass(frozen=True)
class QRScanResult:
    code: str | None
    codes: list[str]
    source: str | None
    detected: bool = False
    detection_source: str | None = None

    @property
    def found(self) -> bool:
        return self.code is not None


class QRReaderError(ValueError):
    """Raised when an image cannot be opened for QR scanning."""


def detect_codes(
    source_path: Path,
    *,
    preferred_types: tuple[str, ...] = ("qr", "barcode"),
) -> list[CodeDetection]:
    """Decode QR/DataMatrix/barcode payloads from a selected price-tag crop.

    The public contract intentionally stays small: the OCR stage gets a list of
    decoded payloads plus normalized fields, while this module handles crop
    retries, preprocessing and optional decoder backends internally.
    """

    for code_type in _normalize_preferred_types(preferred_types):
        if code_type == "qr":
            detections = _detect_qr_codes_from_path(source_path)
        elif code_type == "matrix":
            detections = _detect_matrix_codes(source_path)
        elif code_type == "barcode":
            detections = _detect_barcodes(source_path)
        else:
            continue
        if detections:
            return detections
    return []


def _detect_qr_codes_from_path(source_path: Path) -> list[CodeDetection]:
    try:
        scan_result = _decode_qr_from_path(source_path)
    except (OSError, QRReaderError):
        return []

    if scan_result.codes:
        return [
            CodeDetection(
                source_path=source_path,
                code_type="QR",
                value=code,
                confidence=1.0,
                parsed_fields=parse_qr_payload(code),
                scan_source=scan_result.source or "",
                detected=scan_result.detected,
                detection_source=scan_result.detection_source or "",
            )
            for code in scan_result.codes
        ]

    opencv_qr_detections = _detect_codes_with_opencv(source_path)
    if opencv_qr_detections:
        return opencv_qr_detections
    return []


def parse_qr_payload(value: str) -> dict[str, str]:
    payload = _parse_payload_to_dict(value)
    casefold_payload = {str(key).casefold(): item for key, item in payload.items()}
    normalized: dict[str, str] = {}
    for output_key, aliases in QR_FIELD_ALIASES.items():
        normalized[output_key] = ""
        for alias in aliases:
            payload_value = payload.get(alias, casefold_payload.get(alias.casefold()))
            if payload_value is not None:
                normalized[output_key] = str(payload_value).strip()
                break
    return normalized


def empty_qr_payload() -> dict[str, str]:
    return {key: "" for key in QR_OUTPUT_FIELDS}


def merge_qr_payloads(detections: list[CodeDetection]) -> dict[str, str]:
    merged = empty_qr_payload()
    for detection in detections:
        for key, value in detection.parsed_fields.items():
            if value and not merged.get(key):
                merged[key] = value
    return merged


def merge_barcode_payload(detections: list[CodeDetection]) -> dict[str, str]:
    for detection in detections:
        if detection.code_type in {"QR", "DATAMATRIX"}:
            continue
        value = _digits_only(detection.value)
        if value:
            return {"barcode": value}
    return {}


def _normalize_preferred_types(preferred_types: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for code_type in preferred_types:
        resolved = str(code_type).strip().casefold()
        if resolved in {"qr_code", "qrcode"}:
            resolved = "qr"
        elif resolved in {"data_matrix", "datamatrix", "dm"}:
            resolved = "matrix"
        elif resolved in {"bar", "linear", "linear_barcode"}:
            resolved = "barcode"
        if resolved not in {"qr", "matrix", "barcode"}:
            continue
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized or ["qr", "barcode"])


def _decode_qr_from_path(path: Path) -> QRScanResult:
    if Image is None or ImageOps is None:
        return QRScanResult(code=None, codes=[], source=None)

    try:
        with Image.open(path) as image:
            _validate_image_size(image)
            image.load()
            return _decode_qr_from_image(image)
    except QRReaderError:
        raise
    except (
        Image.DecompressionBombError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise QRReaderError("Payload is not a readable image") from exc


def _decode_qr_from_image(image) -> QRScanResult:
    _validate_image_size(image)

    try:
        base_image = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as exc:
        raise QRReaderError("Payload is not a readable image") from exc

    detection_source: str | None = None

    for source, candidate in _scan_candidates(base_image):
        for transform, prepared in _preprocess_candidates(candidate):
            scan_source = _format_source(source, transform)
            codes = _decode_qr_codes(prepared)
            if codes:
                return QRScanResult(
                    code=codes[0],
                    codes=codes,
                    source=scan_source,
                    detected=True,
                    detection_source=detection_source or scan_source,
                )

            inverted = ImageOps.invert(prepared)
            codes = _decode_qr_codes(inverted)
            if codes:
                scan_source = _format_source(scan_source, "inverted")
                return QRScanResult(
                    code=codes[0],
                    codes=codes,
                    source=scan_source,
                    detected=True,
                    detection_source=detection_source or scan_source,
                )

        opencv_codes, opencv_detected = _decode_with_opencv(candidate)
        if opencv_detected and detection_source is None:
            detection_source = _format_source(source, "opencv_detect")
        if opencv_codes:
            scan_source = _format_source(source, "opencv")
            return QRScanResult(
                code=opencv_codes[0],
                codes=opencv_codes,
                source=scan_source,
                detected=True,
                detection_source=detection_source or scan_source,
            )

        for transform, prepared in _opencv_warp_candidates(candidate):
            scan_source = _format_source(source, transform)
            detection_source = detection_source or scan_source

            for prep_transform, prep_candidate in _preprocess_candidates(prepared):
                prepared_source = _format_source(scan_source, prep_transform)
                codes = _decode_qr_codes(prep_candidate)
                if codes:
                    return QRScanResult(
                        code=codes[0],
                        codes=codes,
                        source=prepared_source,
                        detected=True,
                        detection_source=detection_source,
                    )

                inverted = ImageOps.invert(prep_candidate)
                codes = _decode_qr_codes(inverted)
                if codes:
                    prepared_source = _format_source(prepared_source, "inverted")
                    return QRScanResult(
                        code=codes[0],
                        codes=codes,
                        source=prepared_source,
                        detected=True,
                        detection_source=detection_source,
                    )

    return QRScanResult(
        code=None,
        codes=[],
        source=None,
        detected=detection_source is not None,
        detection_source=detection_source,
    )


def _scan_candidates(image) -> Iterable[tuple[str, Any]]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return

    seen_boxes: set[tuple[int, int, int, int]] = set()

    for name, ratios in CROP_RATIOS:
        box = _ratio_box(ratios, width, height)
        normalized_box = _normalize_box(box, width, height)
        if normalized_box in seen_boxes:
            continue
        seen_boxes.add(normalized_box)

        candidate = image.crop(normalized_box)
        if candidate.width <= 0 or candidate.height <= 0:
            continue

        for scale, scaled in _scale_candidates(candidate):
            source = _format_source(name, scale)
            yield source, scaled

            with_quiet_zone = _add_quiet_zone(scaled)
            if with_quiet_zone.size != scaled.size:
                yield _format_source(source, "quiet_zone"), with_quiet_zone


def _preprocess_candidates(image) -> Iterable[tuple[str | None, Any]]:
    yield None, image

    gray = ImageOps.grayscale(image)
    autocontrasted = ImageOps.autocontrast(gray)
    yield "gray_autocontrast", autocontrasted
    yield "gray_sharpen", autocontrasted.filter(ImageFilter.SHARPEN)
    yield "gray_otsu", _binarize_otsu(autocontrasted)


def _detect_codes_with_opencv(source_path: Path) -> list[CodeDetection]:
    if cv2 is None:
        return []

    image = cv2.imread(str(source_path))
    if image is None:
        return []

    for candidate_source, candidate in _opencv_decode_candidates(image):
        detections = _detect_codes_on_cv_image(source_path, candidate, scan_source=candidate_source)
        if detections:
            return detections
    return []


def _detect_barcodes(source_path: Path) -> list[CodeDetection]:
    detections = _detect_barcodes_with_image_decoders(source_path)
    if detections:
        return detections
    return _detect_barcodes_with_opencv(source_path)


def _detect_matrix_codes(source_path: Path) -> list[CodeDetection]:
    if Image is None or ImageOps is None:
        return []

    try:
        with Image.open(source_path) as image:
            _validate_image_size(image)
            image.load()
            base_image = ImageOps.exif_transpose(image).convert("RGB")
    except (
        Image.DecompressionBombError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ):
        return []

    for source, candidate in _scan_candidates(base_image):
        for transform, prepared in _preprocess_candidates(candidate):
            scan_source = _format_source(source, transform)
            codes = _decode_matrix_codes(prepared)
            if codes:
                return [
                    CodeDetection(
                        source_path=source_path,
                        code_type="DATAMATRIX",
                        value=code,
                        confidence=1.0,
                        parsed_fields=parse_qr_payload(code),
                        scan_source=scan_source,
                        detection_source=scan_source,
                    )
                    for code in codes
                ]

            inverted = ImageOps.invert(prepared)
            codes = _decode_matrix_codes(inverted)
            if codes:
                scan_source = _format_source(scan_source, "inverted")
                return [
                    CodeDetection(
                        source_path=source_path,
                        code_type="DATAMATRIX",
                        value=code,
                        confidence=1.0,
                        parsed_fields=parse_qr_payload(code),
                        scan_source=scan_source,
                        detection_source=scan_source,
                    )
                    for code in codes
                ]
    return []


def _detect_barcodes_with_image_decoders(source_path: Path) -> list[CodeDetection]:
    if Image is None or ImageOps is None:
        return []

    try:
        with Image.open(source_path) as image:
            _validate_image_size(image)
            image.load()
            base_image = ImageOps.exif_transpose(image).convert("RGB")
    except (
        Image.DecompressionBombError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ):
        return []

    for source, candidate in _scan_candidates(base_image):
        for transform, prepared in _preprocess_candidates(candidate):
            scan_source = _format_source(source, transform)
            codes = _decode_barcode_codes(prepared)
            if codes:
                return [
                    CodeDetection(
                        source_path=source_path,
                        code_type="BARCODE",
                        value=code,
                        confidence=1.0,
                        parsed_fields={},
                        scan_source=scan_source,
                        detection_source=scan_source,
                    )
                    for code in codes
                ]

            inverted = ImageOps.invert(prepared)
            codes = _decode_barcode_codes(inverted)
            if codes:
                scan_source = _format_source(scan_source, "inverted")
                return [
                    CodeDetection(
                        source_path=source_path,
                        code_type="BARCODE",
                        value=code,
                        confidence=1.0,
                        parsed_fields={},
                        scan_source=scan_source,
                        detection_source=scan_source,
                    )
                    for code in codes
                ]
    return []


def _detect_barcodes_with_opencv(source_path: Path) -> list[CodeDetection]:
    if cv2 is None or not hasattr(cv2, "barcode_BarcodeDetector"):
        return []

    image = cv2.imread(str(source_path))
    if image is None:
        return []

    for candidate_source, candidate in _opencv_decode_candidates(image):
        detections = _detect_barcodes_on_cv_image(source_path, candidate, scan_source=candidate_source)
        if detections:
            return detections
    return []


def _detect_barcodes_on_cv_image(
    source_path: Path,
    image,
    *,
    scan_source: str = "",
) -> list[CodeDetection]:
    detector = cv2.barcode_BarcodeDetector()

    try:
        ok, decoded_info, decoded_types, points = detector.detectAndDecodeWithType(image)
    except cv2.error:
        ok = False
        decoded_info = ()
        decoded_types = ()
        points = None

    detections: list[CodeDetection] = []
    if ok and decoded_info:
        for index, value in enumerate(decoded_info):
            if not value:
                continue
            code_format = ""
            if decoded_types is not None and index < len(decoded_types):
                code_format = str(decoded_types[index])
            bbox = _points_to_bbox(points[index]) if points is not None and index < len(points) else None
            detections.append(
                CodeDetection(
                    source_path=source_path,
                    code_type=code_format or "BARCODE",
                    value=str(value),
                    confidence=1.0,
                    bbox=bbox,
                    parsed_fields={},
                    scan_source=scan_source,
                    detection_source=_format_source(scan_source, "opencv_barcode"),
                )
            )
    if detections:
        return detections

    try:
        value, points, code_format = detector.detectAndDecode(image)
    except cv2.error:
        value = ""
        points = None
        code_format = None
    if not value:
        return []

    return [
        CodeDetection(
            source_path=source_path,
            code_type=str(code_format) if code_format else "BARCODE",
            value=str(value),
            confidence=1.0,
            bbox=_points_to_bbox(points) if points is not None else None,
            parsed_fields={},
            scan_source=scan_source,
            detection_source=_format_source(scan_source, "opencv_barcode"),
        )
    ]


def _detect_codes_on_cv_image(
    source_path: Path,
    image,
    *,
    scan_source: str = "",
) -> list[CodeDetection]:
    detector = cv2.QRCodeDetector()
    detections: list[CodeDetection] = []

    if hasattr(detector, "detectAndDecodeMulti"):
        try:
            ok, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
        except cv2.error:
            ok = False
            decoded_info = []
            points = None

        if ok and decoded_info:
            for index, value in enumerate(decoded_info):
                if not value:
                    continue
                bbox = _points_to_bbox(points[index]) if points is not None and index < len(points) else None
                detections.append(
                    CodeDetection(
                        source_path=source_path,
                        code_type="QR",
                        value=str(value),
                        confidence=1.0,
                        bbox=bbox,
                        parsed_fields=parse_qr_payload(str(value)),
                        scan_source=scan_source,
                        detection_source=_format_source(scan_source, "opencv"),
                    )
                )

    if detections:
        return detections

    try:
        value, points, _ = detector.detectAndDecode(image)
    except cv2.error:
        value = ""
        points = None
    if not value:
        return []

    return [
        CodeDetection(
            source_path=source_path,
            code_type="QR",
            value=str(value),
            confidence=1.0,
            bbox=_points_to_bbox(points) if points is not None else None,
            parsed_fields=parse_qr_payload(str(value)),
            scan_source=scan_source,
            detection_source=_format_source(scan_source, "opencv"),
        )
    ]


def _opencv_decode_candidates(image) -> Iterable[tuple[str, Any]]:
    yield "original", image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yield "gray", gray
    yield "gray_equalized", cv2.equalizeHist(gray)
    for scale in (2, 3, 4):
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
        yield f"x{scale}", resized
        resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        yield f"x{scale}:gray", resized_gray
        yield f"x{scale}:gray_equalized", cv2.equalizeHist(resized_gray)


def _decode_qr_codes(image) -> list[str]:
    return _deduplicate_codes(
        [
            *_decode_with_zbar(image),
            *_decode_with_zxing(image),
        ]
    )


def _decode_barcode_codes(image) -> list[str]:
    return _deduplicate_codes(
        [
            *_decode_barcodes_with_zbar(image),
            *_decode_barcodes_with_zxing(image),
        ]
    )


def _decode_matrix_codes(image) -> list[str]:
    return _deduplicate_codes([*_decode_matrix_with_zxing(image)])


def _decode_with_zbar(image) -> list[str]:
    if decode_zbar is None or ZBarSymbol is None:
        return []

    try:
        decoded_objects = decode_zbar(image, symbols=[ZBarSymbol.QRCODE])
    except Exception:
        return []

    codes: list[str] = []
    for decoded in decoded_objects:
        if not decoded.data:
            continue
        codes.append(decoded.data.decode("utf-8", errors="replace"))
    return codes


def _decode_barcodes_with_zbar(image) -> list[str]:
    if decode_zbar is None:
        return []

    try:
        decoded_objects = decode_zbar(image)
    except Exception:
        return []

    codes: list[str] = []
    for decoded in decoded_objects:
        if not decoded.data or str(getattr(decoded, "type", "")).upper() == "QRCODE":
            continue
        codes.append(decoded.data.decode("utf-8", errors="replace"))
    return codes


def _decode_with_zxing(image) -> list[str]:
    if zxingcpp is None:
        return []

    try:
        decoded_objects = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.QRCode,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
    except (RuntimeError, ValueError):
        return []

    return [
        decoded.text
        for decoded in decoded_objects
        if decoded.text and getattr(decoded, "error", None) is None
    ]


def _decode_barcodes_with_zxing(image) -> list[str]:
    if zxingcpp is None:
        return []

    try:
        decoded_objects = zxingcpp.read_barcodes(
            image,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
    except (RuntimeError, ValueError):
        return []

    qr_format = getattr(zxingcpp.BarcodeFormat, "QRCode", None)
    data_matrix_format = getattr(zxingcpp.BarcodeFormat, "DataMatrix", None)
    codes: list[str] = []
    for decoded in decoded_objects:
        if not decoded.text or getattr(decoded, "error", None) is not None:
            continue
        if qr_format is not None and getattr(decoded, "format", None) == qr_format:
            continue
        if data_matrix_format is not None and getattr(decoded, "format", None) == data_matrix_format:
            continue
        codes.append(decoded.text)
    return codes


def _decode_matrix_with_zxing(image) -> list[str]:
    if zxingcpp is None:
        return []

    data_matrix_format = getattr(zxingcpp.BarcodeFormat, "DataMatrix", None)
    if data_matrix_format is None:
        return []

    try:
        decoded_objects = zxingcpp.read_barcodes(
            image,
            formats=data_matrix_format,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
    except (RuntimeError, ValueError):
        return []

    return [
        decoded.text
        for decoded in decoded_objects
        if decoded.text and getattr(decoded, "error", None) is None
    ]


def _decode_with_opencv(image) -> tuple[list[str], bool]:
    if cv2 is None or np is None:
        return [], False

    detector = cv2.QRCodeDetector()
    cv_image = _to_cv_image(image)
    codes: list[str] = []
    detected = False

    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(cv_image)
    except cv2.error:
        ok = False
        decoded_info = []
        points = None

    if points is not None and len(points) > 0:
        detected = True

    if ok:
        codes.extend(code for code in decoded_info if code)

    try:
        code, points, _ = detector.detectAndDecode(cv_image)
    except cv2.error:
        code = ""
        points = None

    if points is not None and len(points) > 0:
        detected = True
    if code:
        codes.append(code)

    return _deduplicate_codes(codes), detected


def _opencv_warp_candidates(image) -> Iterable[tuple[str, Any]]:
    if cv2 is None or np is None:
        return

    detector = cv2.QRCodeDetector()
    cv_image = _to_cv_image(image)

    try:
        ok, points = detector.detect(cv_image)
    except cv2.error:
        ok = False
        points = None

    if not ok or points is None:
        return

    point_sets = np.asarray(points, dtype=np.float32).reshape(-1, 4, 2)
    for index, point_set in enumerate(point_sets):
        center = point_set.mean(axis=0)

        for expand in OPENCV_WARP_EXPAND_FACTORS:
            expanded = ((point_set - center) * expand) + center
            if _is_degenerate_quad(expanded):
                continue

            for size in OPENCV_WARP_SIZES:
                destination = np.array(
                    [
                        [0, 0],
                        [size - 1, 0],
                        [size - 1, size - 1],
                        [0, size - 1],
                    ],
                    dtype=np.float32,
                )

                try:
                    matrix = cv2.getPerspectiveTransform(expanded, destination)
                    warped = cv2.warpPerspective(
                        _to_rgb_array(image),
                        matrix,
                        (size, size),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),
                    )
                except cv2.error:
                    continue

                source = f"opencv_warp_{index}:x{expand:g}:size_{size}"
                yield source, Image.fromarray(warped)


def _parse_payload_to_dict(value: str) -> dict[str, Any]:
    text = value.strip()
    if not text:
        return {}

    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        parsed_json = None
    if isinstance(parsed_json, dict):
        return parsed_json

    parsed_url = urlparse(text)
    query = parsed_url.query if parsed_url.query else text
    pairs = dict(parse_qsl(query, keep_blank_values=True))
    if pairs:
        return pairs

    result: dict[str, str] = {}
    for separator in ("&", ";", "\n", "\r"):
        text = text.replace(separator, "\n")
    for chunk in text.splitlines():
        if not chunk.strip():
            continue
        if "=" in chunk:
            key, raw_value = chunk.split("=", 1)
        elif ":" in chunk:
            key, raw_value = chunk.split(":", 1)
        else:
            continue
        result[key.strip()] = raw_value.strip()
    return result


def _ratio_box(
    ratios: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = ratios
    return (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )


def _normalize_box(
    box: tuple[int, int, int, int],
    max_width: int,
    max_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(left, max_width))
    right = max(left, min(right, max_width))
    top = max(0, min(top, max_height))
    bottom = max(top, min(bottom, max_height))
    return left, top, right, bottom


def _resize_by_width(image, new_width: int):
    if image.width == new_width or image.width <= 0:
        return image

    new_height = max(1, round(image.height * (new_width / image.width)))
    return image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)


def _scale_candidates(image) -> Iterable[tuple[str | None, Any]]:
    yield None, image

    seen_sizes = {image.size}
    for width in SCAN_WIDTHS:
        if width > image.width and width / image.width > MAX_UPSCALE_FACTOR:
            continue

        resized = _resize_by_width(image, width)
        if resized.size in seen_sizes:
            continue
        if resized.width * resized.height > MAX_CANDIDATE_PIXELS:
            continue

        seen_sizes.add(resized.size)
        yield f"width_{width}", resized


def _add_quiet_zone(image):
    min_side = min(image.width, image.height)
    if min_side <= 0 or image.width * image.height > MAX_CANDIDATE_PIXELS:
        return image

    border = max(8, min_side // 24)
    return ImageOps.expand(image, border=border, fill="white")


def _binarize_otsu(image):
    gray = ImageOps.grayscale(image)
    threshold = _otsu_threshold(gray)
    return gray.point(lambda pixel: 255 if pixel > threshold else 0).convert("L")


def _otsu_threshold(gray) -> int:
    histogram = gray.histogram()
    total = sum(histogram)
    if total == 0:
        return 127

    sum_total = sum(level * count for level, count in enumerate(histogram))
    sum_background = 0
    weight_background = 0
    best_variance = -1.0
    best_threshold = 127

    for level, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += level * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = level

    return best_threshold


def _validate_image_size(image) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise QRReaderError("Image has invalid dimensions")
    if width * height > MAX_IMAGE_PIXELS:
        raise QRReaderError(f"Image is too large; max supported size is {MAX_IMAGE_PIXELS} pixels")


def _format_source(*parts: str | None) -> str:
    return ":".join(part for part in parts if part)


def _to_rgb_array(image):
    if np is None:
        raise RuntimeError("numpy is not available")
    return np.array(image.convert("RGB"))


def _to_cv_image(image):
    if cv2 is None:
        raise RuntimeError("opencv is not available")

    rgb_image = _to_rgb_array(image)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def _is_degenerate_quad(points) -> bool:
    if cv2 is None or np is None:
        return True

    area = abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))
    return area < 25


def _deduplicate_codes(codes: Iterable[str]) -> list[str]:
    unique_codes: list[str] = []
    seen: set[str] = set()

    for code in codes:
        if code in seen:
            continue
        seen.add(code)
        unique_codes.append(code)

    return unique_codes


def _digits_only(value: str) -> str:
    return "".join(char for char in str(value) if char.isdigit())


def _points_to_bbox(points) -> tuple[int, int, int, int] | None:
    if points is None:
        return None
    flat = points.reshape(-1, 2)
    if len(flat) == 0:
        return None
    x_min = int(flat[:, 0].min())
    y_min = int(flat[:, 1].min())
    x_max = int(flat[:, 0].max())
    y_max = int(flat[:, 1].max())
    return x_min, y_min, x_max - x_min, y_max - y_min
