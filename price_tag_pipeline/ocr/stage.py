from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2

from price_tag_pipeline.codes.detector import (
    CodeRegion,
    CodeRegionDetectorConfig,
    NcnnCodeRegionDetector,
    build_code_region_detector,
    crop_region,
)
from price_tag_pipeline.codes.reader import (
    CodeDetection,
    detect_codes,
    empty_qr_payload,
    merge_barcode_payload,
    merge_qr_payloads,
)
from price_tag_pipeline.ocr.extractor import (
    DEFAULT_OCR_MODEL,
    OcrConfig,
    OpenAICompatibleOcrClient,
    empty_price_tag_payload,
    extract_price_tag_text,
)
from price_tag_pipeline.ocr.schema import STRUCTURED_PRICE_TAG_FIELDS, StructuredPriceTagCsvRow


def run_ocr_stage(
    *,
    video_path: Path,
    output_csv_path: Path,
    raw_jsonl_path: Path,
    unique_rows: list[dict[str, Any]],
    fps: float,
    config: OcrConfig,
    limit: int = 0,
    code_detector_config: CodeRegionDetectorConfig | None = None,
    code_crop_dir: Path | None = None,
) -> dict[str, Any]:
    client = OpenAICompatibleOcrClient(config)
    model_ids = client.check_health(validate_model=False)
    resolved_config = config
    if model_ids and config.model not in model_ids:
        if config.model == DEFAULT_OCR_MODEL:
            resolved_config = replace(config, model=_select_default_model(model_ids))
            client = OpenAICompatibleOcrClient(resolved_config)
        else:
            client.check_health(validate_model=True)

    rows_to_process = [row for row in unique_rows if _row_ocr_crop_path(row)]
    if limit > 0:
        rows_to_process = rows_to_process[:limit]

    output_rows: list[dict[str, str]] = []
    raw_records: list[dict[str, Any]] = []
    ocr_errors = 0
    qr_detections = 0
    barcode_detections = 0
    code_regions = 0
    code_region_crops = 0
    code_full_crop_fallbacks = 0
    code_detector_error = ""

    code_region_detector: NcnnCodeRegionDetector | None = None
    if code_detector_config is not None:
        try:
            code_region_detector = build_code_region_detector(code_detector_config)
            if code_crop_dir is not None:
                code_crop_dir.mkdir(parents=True, exist_ok=True)
                _clear_generated_code_crops(code_crop_dir)
        except Exception as exc:
            code_detector_error = str(exc)

    for row in rows_to_process:
        crop_path = _row_ocr_crop_path(row)
        if crop_path is None:
            continue
        ocr_payload = empty_price_tag_payload()
        raw_response = ""
        error = ""
        try:
            ocr_result = extract_price_tag_text(crop_path, client=client)
            ocr_payload = ocr_result.parsed_response
            raw_response = ocr_result.raw_response
        except Exception as exc:  # Keep the batch moving and persist the failure in JSONL.
            ocr_errors += 1
            error = str(exc)

        code_scan = _detect_codes_for_crop(
            crop_path,
            row=row,
            code_region_detector=code_region_detector,
            code_crop_dir=code_crop_dir,
        )
        code_detections = code_scan.detections
        if code_detections:
            qr_detections += sum(1 for item in code_detections if item.code_type == "QR")
            barcode_detections += sum(1 for item in code_detections if item.code_type != "QR")
            barcode_payload = merge_barcode_payload(code_detections)
            if barcode_payload and not ocr_payload.get("barcode"):
                ocr_payload.update(barcode_payload)
            qr_payload = merge_qr_payloads(code_detections)
        else:
            qr_payload = empty_qr_payload()
        code_regions += len(code_scan.regions)
        code_region_crops += len(code_scan.region_crop_paths)
        if code_scan.fallback_used:
            code_full_crop_fallbacks += 1

        output_rows.append(
            _build_output_row(
                video_path=video_path,
                track_row=row,
                fps=fps,
                ocr_payload=ocr_payload,
                qr_payload=qr_payload,
            )
        )
        raw_records.append(
            {
                "track_id": row.get("track_id", ""),
                "crop_path": str(crop_path),
                "raw_response": raw_response,
                "parsed_response": ocr_payload,
                "ocr_error": error,
                "code_regions": [_code_region_to_record(item) for item in code_scan.regions],
                "code_region_crop_paths": [str(path) for path in code_scan.region_crop_paths],
                "code_detector_error": code_detector_error,
                "code_full_crop_fallback": code_scan.fallback_used,
                "codes": [_code_detection_to_record(item) for item in code_detections],
            }
        )

    _write_structured_csv(output_csv_path, output_rows)
    _write_jsonl(raw_jsonl_path, raw_records)

    return {
        "enabled": True,
        "base_url": resolved_config.base_url,
        "model": resolved_config.model,
        "configured_model": config.model,
        "disable_reasoning": resolved_config.disable_reasoning,
        "image_preprocess": resolved_config.image_preprocess,
        "available_models": model_ids,
        "rows_requested": len([row for row in unique_rows if _row_ocr_crop_path(row)]),
        "rows_processed": len(rows_to_process),
        "ocr_errors": ocr_errors,
        "qr_detections": qr_detections,
        "barcode_detections": barcode_detections,
        "code_region_detector_enabled": code_region_detector is not None,
        "code_region_detector_error": code_detector_error,
        "code_regions": code_regions,
        "code_region_crops": code_region_crops,
        "code_full_crop_fallbacks": code_full_crop_fallbacks,
        "code_crop_dir": str(code_crop_dir) if code_crop_dir is not None and code_region_detector is not None else "",
        "output_csv": str(output_csv_path),
        "raw_jsonl": str(raw_jsonl_path),
    }


@dataclass
class CodeScanResult:
    detections: list[CodeDetection]
    regions: list[CodeRegion]
    region_crop_paths: list[Path]
    fallback_used: bool = False


def _detect_codes_for_crop(
    crop_path: Path,
    *,
    row: dict[str, Any],
    code_region_detector: NcnnCodeRegionDetector | None,
    code_crop_dir: Path | None,
) -> CodeScanResult:
    if code_region_detector is None:
        return CodeScanResult(detections=detect_codes(crop_path), regions=[], region_crop_paths=[])

    image = cv2.imread(str(crop_path))
    if image is None:
        return CodeScanResult(detections=[], regions=[], region_crop_paths=[])

    regions = code_region_detector.detect(image)
    detections: list[CodeDetection] = []
    region_crop_paths: list[Path] = []

    for index, region in enumerate(regions, start=1):
        region_crop, padded_bbox = crop_region(
            image,
            region,
            padding_ratio=code_region_detector.config.padding_ratio,
        )
        if region_crop.size == 0:
            continue

        if code_crop_dir is None:
            continue

        region_path = code_crop_dir / _code_crop_filename(row, region, index)
        cv2.imwrite(str(region_path), region_crop)
        region_crop_paths.append(region_path)

        for detection in detect_codes(region_path, preferred_types=_preferred_code_types(region.role)):
            _attach_region_metadata(detection, region, padded_bbox)
            detections.append(detection)

    fallback_used = False
    if not detections and code_region_detector.config.fallback_full_crop:
        fallback_used = True
        detections = detect_codes(crop_path)
        for detection in detections:
            detection.scan_source = _join_source("full_crop_fallback", detection.scan_source)
            detection.detection_source = _join_source("full_crop_fallback", detection.detection_source)

    return CodeScanResult(
        detections=detections,
        regions=regions,
        region_crop_paths=region_crop_paths,
        fallback_used=fallback_used,
    )


def _code_crop_filename(row: dict[str, Any], region: CodeRegion, index: int) -> str:
    track_id = _to_int(row.get("track_id", "")) or 0
    frame = _to_int(row.get("best_frame", "")) or 0
    label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in region.label)
    return (
        f"track_{track_id:04d}_frame_{frame:04d}_code_{index:02d}_"
        f"class_{region.class_id}_{region.role or 'unknown'}_{label}_{region.score:.3f}.jpg"
    )


def _attach_region_metadata(
    detection: CodeDetection,
    region: CodeRegion,
    padded_bbox: tuple[int, int, int, int],
) -> None:
    prefix = f"ncnn_region:class_{region.class_id}:{region.role or 'unknown'}:{region.label}:conf_{region.score:.3f}"
    detection.region_bbox = padded_bbox
    detection.region_class_id = region.class_id
    detection.region_label = region.label
    detection.region_confidence = region.score
    detection.bbox = padded_bbox
    detection.scan_source = _join_source(prefix, detection.scan_source)
    detection.detection_source = _join_source(prefix, detection.detection_source)


def _join_source(*parts: str | None) -> str:
    return ":".join(part for part in parts if part)


def _preferred_code_types(region_role: str) -> tuple[str, ...]:
    role = str(region_role or "").casefold()
    if role == "barcode":
        return ("barcode", "qr", "matrix")
    if role == "matrix":
        return ("matrix", "qr", "barcode")
    return ("qr", "matrix", "barcode")


def _build_output_row(
    *,
    video_path: Path,
    track_row: dict[str, Any],
    fps: float,
    ocr_payload: dict[str, str],
    qr_payload: dict[str, str],
) -> dict[str, str]:
    x_min, y_min, x_max, y_max = _output_coordinates(track_row)
    return StructuredPriceTagCsvRow.from_payloads(
        video_path=video_path,
        track_row=track_row,
        frame_timestamp=_frame_timestamp_ms(track_row, fps),
        coordinates=(x_min, y_min, x_max, y_max),
        ocr_payload=ocr_payload,
        qr_payload=qr_payload,
    ).to_csv_row()


def _output_coordinates(track_row: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    x_min = _to_int(track_row.get("source_x_min", ""))
    y_min = _to_int(track_row.get("source_y_min", ""))
    x_max = _to_int(track_row.get("source_x_max", ""))
    y_max = _to_int(track_row.get("source_y_max", ""))
    if None not in (x_min, y_min, x_max, y_max):
        return x_min, y_min, x_max, y_max

    x_min = _to_int(track_row.get("best_full_x", ""))
    y_min = _to_int(track_row.get("best_full_y", ""))
    width = _to_int(track_row.get("best_full_w", ""))
    height = _to_int(track_row.get("best_full_h", ""))
    x_max = x_min + width if x_min is not None and width is not None else None
    y_max = y_min + height if y_min is not None and height is not None else None
    return x_min, y_min, x_max, y_max


def _row_ocr_crop_path(row: dict[str, Any]) -> Path | None:
    crop_path = row.get("ocr_crop_path") or (
        row.get("crop_path") if str(row.get("crop_saved", "0")) == "1" else ""
    )
    if not crop_path:
        return None
    return Path(str(crop_path))


def _frame_timestamp_ms(track_row: dict[str, Any], fps: float) -> str:
    frame = _to_int(track_row.get("best_frame", ""))
    if frame is not None and fps > 0:
        return str(int(round(frame / fps * 1000.0)))
    time_sec = _to_float(track_row.get("best_time_sec", ""))
    if time_sec is not None:
        return str(int(round(time_sec * 1000.0)))
    return ""


def _write_structured_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(STRUCTURED_PRICE_TAG_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clear_generated_code_crops(code_crop_dir: Path) -> int:
    cleared = 0
    for crop_path in code_crop_dir.glob("track_*_code_*.jpg"):
        if crop_path.is_file():
            crop_path.unlink()
            cleared += 1
    return cleared


def _code_region_to_record(region: CodeRegion) -> dict[str, Any]:
    return {
        "bbox": region.bbox,
        "score": region.score,
        "class_id": region.class_id,
        "label": region.label,
        "role": region.role,
    }


def _code_detection_to_record(detection: CodeDetection) -> dict[str, Any]:
    return {
        "source_path": str(detection.source_path),
        "code_type": detection.code_type,
        "value": detection.value,
        "confidence": detection.confidence,
        "bbox": detection.bbox,
        "parsed_fields": detection.parsed_fields,
        "scan_source": detection.scan_source,
        "detected": detection.detected,
        "detection_source": detection.detection_source,
        "region_bbox": detection.region_bbox,
        "region_class_id": detection.region_class_id,
        "region_label": detection.region_label,
        "region_confidence": detection.region_confidence,
    }


def _select_default_model(model_ids: list[str]) -> str:
    for model_id in model_ids:
        lower_model_id = model_id.lower()
        if "embedding" not in lower_model_id and "embed" not in lower_model_id:
            return model_id
    return model_ids[0]


def _to_int(value: Any) -> int | None:
    if value == "" or value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
