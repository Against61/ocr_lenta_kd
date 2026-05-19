from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from price_tag_pipeline.core.deskew import apply_deskew_to_tracks
from price_tag_pipeline.core.image import normalize_orientation, resize_to_width
from price_tag_pipeline.core.output import write_track_history, write_tracks_csv
from price_tag_pipeline.core.visualization import draw_overlay
from price_tag_pipeline.detector import build_detector
from price_tag_pipeline.codes.detector import CodeRegionDetectorConfig
from price_tag_pipeline.ocr import OcrConfig, run_ocr_stage
from price_tag_pipeline.quality import build_quality_classifier, select_track_crops
from price_tag_pipeline.sources.video import parse_debug_frames
from price_tag_pipeline.tracker import Track, assign_detections, mark_seen_counted, track_is_counted, track_is_valid


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.out_dir / "crops"
    ocr_fallback_crop_dir = args.out_dir / "ocr_fallback_crops"
    overlay_dir = args.out_dir / "overlays"
    crop_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    cleared_crop_files = 0

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    detector = build_detector(args)
    cleared_crop_files = _clear_generated_crops(crop_dir)
    cleared_ocr_fallback_crop_files = 0
    if args.run_ocr and args.ocr_include_quality_failed:
        ocr_fallback_crop_dir.mkdir(exist_ok=True)
        cleared_ocr_fallback_crop_files = _clear_generated_crops(ocr_fallback_crop_dir)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    rotation = _rotation_mode(args)
    debug_frames = parse_debug_frames(args.debug_frames)

    active_tracks: list[Track] = []
    finished_tracks: list[Track] = []
    next_track_id = 1
    counted_ids: set[int] = set()
    detection_rows: list[dict[str, object]] = []
    frame_count_rows: list[dict[str, object]] = []
    overlay_writer = None
    crossing_x = args.width * args.crossing_x_ratio
    analysis_shape: tuple[int, int] | None = None

    write_overlay_video = args.write_overlay_video and not args.no_overlay_video
    overlay_video_path = args.out_dir / "counting_overlay.mp4"

    frame_idx = 0
    processed_frames = 0
    while True:
        ok, original = cap.read()
        if not ok:
            break
        if args.frame_step > 1 and frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        full = normalize_orientation(original, args)
        small, scale = resize_to_width(full, args.width)
        if analysis_shape is None:
            analysis_shape = (small.shape[1], small.shape[0])

        detections = detector.predict(small)
        next_track_id, newly_finished = assign_detections(
            active_tracks=active_tracks,
            detections=detections,
            frame_idx=frame_idx,
            fps=fps,
            frame_small=small,
            frame_full=full,
            scale=scale,
            crossing_x=crossing_x,
            next_track_id=next_track_id,
            max_link_distance=args.max_link_distance,
            max_missed=args.max_missed,
            max_backward_step=args.max_backward_step,
            max_y_jump=args.max_y_jump,
            right_edge_retire_ratio=args.right_edge_retire_ratio,
            right_edge_ignore_ratio=args.right_edge_ignore_ratio,
            count_mode=args.count_mode,
            count_first_at_line=args.count_first_at_line,
            line_entry_margin=args.line_entry_margin,
            max_crop_candidates=args.max_crop_candidates,
        )
        finished_tracks.extend(newly_finished)

        for detection in detections:
            x, y, w, h = detection.bbox
            rx, ry, rw, rh = detection.red_bbox
            detection_rows.append(
                {
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / fps, 3) if fps else "",
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "red_x": rx,
                    "red_y": ry,
                    "red_w": rw,
                    "red_h": rh,
                    "score": round(detection.score, 4),
                    "white_ratio": round(detection.white_ratio, 4),
                    "upper_white_ratio": round(detection.upper_white_ratio, 4),
                    "red_ratio": round(detection.red_ratio, 4),
                }
            )

        visible_counted = 0
        for track in active_tracks:
            if args.count_mode == "seen" and track_is_valid(
                track,
                args.min_track_frames,
                args.min_delta_x,
                args.min_track_white_ratio,
                args.min_track_aspect,
                args.max_track_height,
            ):
                mark_seen_counted(track)

            if track_is_counted(
                track,
                args.min_track_frames,
                args.min_delta_x,
                args.min_track_white_ratio,
                args.min_track_aspect,
                args.max_track_height,
            ):
                counted_ids.add(track.track_id)
                visible_counted += 1

        frame_count_rows.append(
            {
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3) if fps else "",
                "detections": len(detections),
                "active_tracks": len(active_tracks),
                "visible_counted_tracks": visible_counted,
                "cumulative_unique_tracks": len(counted_ids),
            }
        )

        overlay = draw_overlay(
            small,
            active_tracks,
            crossing_x,
            args.count_mode,
            counted_ids,
            visible_counted,
            args.min_track_frames,
            args.min_delta_x,
            args.min_track_white_ratio,
            args.min_track_aspect,
            args.max_track_height,
        )

        if write_overlay_video:
            if overlay_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_fps = (fps / max(1, args.frame_step)) if fps else 20.0
                overlay_writer = cv2.VideoWriter(
                    str(overlay_video_path),
                    fourcc,
                    output_fps,
                    (overlay.shape[1], overlay.shape[0]),
                )
            overlay_writer.write(overlay)

        if frame_idx in debug_frames:
            cv2.imwrite(str(overlay_dir / f"overlay_{frame_idx:04d}.jpg"), overlay)

        processed_frames += 1
        frame_idx += 1

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()

    all_tracks = sorted(finished_tracks + active_tracks, key=lambda item: item.track_id)
    valid_tracks = [
        track
        for track in all_tracks
        if track_is_valid(
            track,
            args.min_track_frames,
            args.min_delta_x,
            args.min_track_white_ratio,
            args.min_track_aspect,
            args.max_track_height,
        )
    ]
    if args.count_mode == "seen":
        for track in valid_tracks:
            mark_seen_counted(track)

    counted_tracks = [
        track
        for track in all_tracks
        if track_is_counted(
            track,
            args.min_track_frames,
            args.min_delta_x,
            args.min_track_white_ratio,
            args.min_track_aspect,
            args.max_track_height,
        )
    ]

    deskew_summary = apply_deskew_to_tracks(
        valid_tracks,
        enabled=args.deskew_crops,
        max_angle=args.crop_deskew_max_angle,
        min_angle=args.crop_deskew_min_angle,
        min_confidence=args.crop_deskew_min_confidence,
    )
    quality_classifier = build_quality_classifier(args)
    quality_summary = select_track_crops(valid_tracks, quality_classifier)

    saved_crop_paths: set[Path] = set()
    for track in valid_tracks:
        best = track.best_observation or track.last_observation
        should_save_crop = track.best_crop is not None and (
            not quality_classifier.enabled or track.selected_quality_passed is True
        )
        if should_save_crop:
            crop_path = crop_dir / f"track_{track.track_id:04d}_frame_{best.frame:04d}.jpg"
            cv2.imwrite(str(crop_path), track.best_crop)
            saved_crop_paths.add(crop_path)

    _write_detection_rows(args.out_dir / "detections.csv", detection_rows)
    _write_frame_counts(args.out_dir / "frame_counts.csv", frame_count_rows)
    write_track_history(args.out_dir / "track_history.csv", all_tracks)
    all_track_rows = write_tracks_csv(
        args.out_dir / "all_valid_tracks.csv",
        valid_tracks,
        crop_dir,
        args.min_track_frames,
        args.min_delta_x,
        args.min_track_white_ratio,
        args.min_track_aspect,
        args.max_track_height,
        counted_only=False,
        prefiltered=True,
        saved_crop_paths=saved_crop_paths,
        source_width=source_width,
        source_height=source_height,
        rotation=rotation,
    )
    unique_rows = write_tracks_csv(
        args.out_dir / "unique_price_tags.csv",
        counted_tracks,
        crop_dir,
        args.min_track_frames,
        args.min_delta_x,
        args.min_track_white_ratio,
        args.min_track_aspect,
        args.max_track_height,
        counted_only=True,
        prefiltered=True,
        saved_crop_paths=saved_crop_paths,
        source_width=source_width,
        source_height=source_height,
        rotation=rotation,
    )

    structured_csv_path = args.out_dir / "structured_price_tags.csv"
    ocr_raw_jsonl_path = args.out_dir / "ocr_raw_responses.jsonl"
    code_crop_dir = args.out_dir / "code_crops"
    ocr_summary: dict[str, object] = {"enabled": False}
    if args.run_ocr:
        ocr_rows = _prepare_ocr_rows(
            unique_rows,
            counted_tracks,
            ocr_fallback_crop_dir,
            include_quality_failed=args.ocr_include_quality_failed,
        )
        ocr_config = OcrConfig(
            base_url=args.ocr_base_url,
            api_key=args.ocr_api_key,
            model=args.ocr_model,
            max_tokens=args.ocr_max_tokens,
            temperature=args.ocr_temperature,
            timeout=args.ocr_timeout,
            retries=args.ocr_retries,
            response_format=args.ocr_response_format,
            disable_reasoning=args.ocr_disable_reasoning,
            image_preprocess=args.ocr_image_preprocess,
        )
        code_detector_config = (
            CodeRegionDetectorConfig(
                model_dir=args.code_detector_dir,
                param_path=args.code_detector_param,
                bin_path=args.code_detector_bin,
                input_size=args.code_detector_input_size,
                conf_threshold=args.code_detector_conf_threshold,
                iou_threshold=args.code_detector_iou_threshold,
                padding_ratio=args.code_detector_padding_ratio,
                resize_mode=args.code_detector_resize_mode,
                fallback_full_crop=args.code_detector_fallback_full_crop,
                class_roles=_parse_code_detector_class_roles(args.code_detector_class_roles),
            )
            if args.code_detector_enabled
            else None
        )
        try:
            ocr_summary = run_ocr_stage(
                video_path=args.video,
                output_csv_path=structured_csv_path,
                raw_jsonl_path=ocr_raw_jsonl_path,
                unique_rows=ocr_rows,
                fps=fps,
                config=ocr_config,
                limit=args.ocr_limit,
                code_detector_config=code_detector_config,
                code_crop_dir=code_crop_dir,
            )
            ocr_summary["include_quality_failed"] = args.ocr_include_quality_failed
        except Exception as exc:
            ocr_summary = {
                "enabled": True,
                "base_url": args.ocr_base_url,
                "model": args.ocr_model,
                "image_preprocess": args.ocr_image_preprocess,
                "include_quality_failed": args.ocr_include_quality_failed,
                "error": str(exc),
                "output_csv": str(structured_csv_path),
                "raw_jsonl": str(ocr_raw_jsonl_path),
            }

    report = {
        "video": str(args.video),
        "detector": args.detector,
        "onnx_model": str(args.onnx_model) if args.detector == "onnx" else "",
        "onnx_input_size": args.onnx_input_size if args.detector == "onnx" else "",
        "onnx_conf_threshold": args.onnx_conf_threshold if args.detector == "onnx" else "",
        "onnx_iou_threshold": args.onnx_iou_threshold if args.detector == "onnx" else "",
        "fps": fps,
        "frame_count": frame_count,
        "source_width": source_width,
        "source_height": source_height,
        "processed_frames": processed_frames,
        "last_source_frame": frame_idx - 1,
        "frame_step": args.frame_step,
        "analysis_width": args.width,
        "analysis_height": analysis_shape[1] if analysis_shape else 0,
        "rotation": rotation,
        "count_mode": args.count_mode,
        "crossing_x": crossing_x,
        "crossing_x_ratio": args.crossing_x_ratio,
        "line_entry_margin": args.line_entry_margin,
        "count_first_at_line": args.count_first_at_line,
        "max_backward_step": args.max_backward_step,
        "max_y_jump": args.max_y_jump,
        "right_edge_retire_ratio": args.right_edge_retire_ratio,
        "right_edge_ignore_ratio": args.right_edge_ignore_ratio,
        "min_track_frames": args.min_track_frames,
        "min_delta_x": args.min_delta_x,
        "min_track_white_ratio": args.min_track_white_ratio,
        "min_track_aspect": args.min_track_aspect,
        "max_track_height": args.max_track_height,
        "max_crop_candidates": args.max_crop_candidates,
        "deskew_crops": args.deskew_crops,
        "crop_deskew_max_angle": args.crop_deskew_max_angle,
        "crop_deskew_min_angle": args.crop_deskew_min_angle,
        "crop_deskew_min_confidence": args.crop_deskew_min_confidence,
        "deskew_summary": deskew_summary,
        "quality_classifier_enabled": quality_classifier.enabled,
        "quality_model_type": getattr(quality_classifier, "model_type", ""),
        "quality_model": getattr(quality_classifier, "model_path", ""),
        "quality_input_size": getattr(quality_classifier, "input_size", ""),
        "quality_threshold": getattr(quality_classifier, "threshold", ""),
        "quality_pass_class": getattr(quality_classifier, "pass_class", ""),
        "quality_input_mode": getattr(quality_classifier, "input_mode", ""),
        "quality_normalization": getattr(quality_classifier, "normalization", ""),
        "quality_summary": quality_summary,
        "code_detector_enabled": args.code_detector_enabled,
        "code_detector_dir": str(args.code_detector_dir) if args.code_detector_enabled else "",
        "code_detector_param": str(args.code_detector_param) if args.code_detector_param else "",
        "code_detector_bin": str(args.code_detector_bin) if args.code_detector_bin else "",
        "code_detector_input_size": args.code_detector_input_size,
        "code_detector_conf_threshold": args.code_detector_conf_threshold,
        "code_detector_iou_threshold": args.code_detector_iou_threshold,
        "code_detector_padding_ratio": args.code_detector_padding_ratio,
        "code_detector_resize_mode": args.code_detector_resize_mode,
        "code_detector_class_roles": args.code_detector_class_roles,
        "code_detector_fallback_full_crop": args.code_detector_fallback_full_crop,
        "ocr_summary": ocr_summary,
        "crop_save_policy": "quality_passed_only" if quality_classifier.enabled else "all_selected",
        "cleared_crop_files": cleared_crop_files,
        "cleared_ocr_fallback_crop_files": cleared_ocr_fallback_crop_files,
        "saved_crops": len(saved_crop_paths),
        "raw_tracks": len(all_tracks),
        "valid_tracks": len(valid_tracks),
        "unique_counted_tracks": len(counted_tracks),
        "detection_rows": len(detection_rows),
        "outputs": {
            "unique_price_tags_csv": str(args.out_dir / "unique_price_tags.csv"),
            "all_valid_tracks_csv": str(args.out_dir / "all_valid_tracks.csv"),
            "track_history_csv": str(args.out_dir / "track_history.csv"),
            "frame_counts_csv": str(args.out_dir / "frame_counts.csv"),
            "detections_csv": str(args.out_dir / "detections.csv"),
            "structured_price_tags_csv": str(structured_csv_path) if args.run_ocr else "",
            "ocr_raw_jsonl": str(ocr_raw_jsonl_path) if args.run_ocr else "",
            "ocr_fallback_crops_dir": str(ocr_fallback_crop_dir) if args.run_ocr and args.ocr_include_quality_failed else "",
            "code_crops_dir": str(code_crop_dir) if args.run_ocr and args.code_detector_enabled else "",
            "crops_dir": str(crop_dir),
            "overlay_video": str(overlay_video_path) if write_overlay_video else "",
            "overlays_dir": str(overlay_dir),
        },
        "unique_track_ids": [row["track_id"] for row in unique_rows],
        "valid_track_ids": [row["track_id"] for row in all_track_rows],
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _rotation_mode(args: argparse.Namespace) -> str:
    if args.no_rotate:
        return "none"
    if args.rotate_cw:
        return "cw"
    if args.rotate_ccw:
        return "ccw"
    return "none"


def _clear_generated_crops(crop_dir: Path) -> int:
    cleared = 0
    for crop_path in crop_dir.glob("track_*_frame_*.jpg"):
        if crop_path.is_file():
            crop_path.unlink()
            cleared += 1
    return cleared


def _prepare_ocr_rows(
    unique_rows: list[dict[str, object]],
    counted_tracks: list[Track],
    ocr_fallback_crop_dir: Path,
    *,
    include_quality_failed: bool,
) -> list[dict[str, object]]:
    if not include_quality_failed:
        return unique_rows

    tracks_by_id = {track.track_id: track for track in counted_tracks}
    ocr_rows: list[dict[str, object]] = []
    for row in unique_rows:
        ocr_row = dict(row)
        if str(ocr_row.get("crop_saved", "0")) != "1":
            track_id = _safe_int(ocr_row.get("track_id"))
            track = tracks_by_id.get(track_id) if track_id is not None else None
            best = track.best_observation if track is not None else None
            if track is not None and best is not None and track.best_crop is not None:
                crop_path = ocr_fallback_crop_dir / f"track_{track.track_id:04d}_frame_{best.frame:04d}.jpg"
                cv2.imwrite(str(crop_path), track.best_crop)
                ocr_row["ocr_crop_path"] = str(crop_path)
        ocr_rows.append(ocr_row)
    return ocr_rows


def _safe_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _parse_code_detector_class_roles(value: object) -> dict[int, str]:
    roles: dict[int, str] = {}
    for chunk in str(value or "").split(","):
        if "=" not in chunk:
            continue
        key, raw_role = chunk.split("=", 1)
        try:
            class_id = int(key.strip())
        except ValueError:
            continue
        role = raw_role.strip().casefold()
        if role in {"qr_code", "qrcode"}:
            role = "qr"
        elif role in {"data_matrix", "datamatrix", "dm"}:
            role = "matrix"
        elif role in {"bar", "linear", "linear_barcode"}:
            role = "barcode"
        if role in {"qr", "barcode", "matrix"}:
            roles[class_id] = role
    return roles


def _write_detection_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame",
        "time_sec",
        "x",
        "y",
        "w",
        "h",
        "red_x",
        "red_y",
        "red_w",
        "red_h",
        "score",
        "white_ratio",
        "upper_white_ratio",
        "red_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_frame_counts(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame",
        "time_sec",
        "detections",
        "active_tracks",
        "visible_counted_tracks",
        "cumulative_unique_tracks",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
