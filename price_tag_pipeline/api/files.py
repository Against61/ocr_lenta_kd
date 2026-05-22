import argparse
import logging
import os
import shlex
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from price_tag_pipeline.core.cli import parse_args
from price_tag_pipeline.core.pipeline import run_pipeline

router = APIRouter(prefix="/video", tags=["video"])

VIDEO_CONTENT_TYPE_PREFIX = "video/"
CSV_MEDIA_TYPE = "text/csv"
DEFAULT_API_ARGS = ("--no-overlay-video",)
LOGGER = logging.getLogger("uvicorn.error")
DEFAULT_JOBS_DIR = Path("runs/api_jobs")
JOBS_DIR = Path(os.getenv("PRICE_TAG_API_JOBS_DIR", str(DEFAULT_JOBS_DIR)))
JOBS: dict[str, dict[str, object]] = {}
JOBS_LOCK = threading.Lock()


@router.post("/analyze", response_class=StreamingResponse)
def analyze_video(video: Annotated[UploadFile, File(...)]) -> StreamingResponse:
    validate_video_upload(video)

    try:
        with tempfile.TemporaryDirectory(prefix="price_tag_api_") as temp_dir:
            temp_path = Path(temp_dir)
            video_path = save_uploaded_video(video, temp_path)
            out_dir = temp_path / "pipeline_output"

            args = build_pipeline_args(video_path, out_dir)
            report = run_pipeline(args)
            log_pipeline_report(report)
            raise_for_ocr_failure_if_requested(args, report)
            csv_path = select_pipeline_csv(report)
            csv_content = csv_path.read_text(encoding="utf-8")
            filename = csv_path.name
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Не удалось обработать видео: {error}",
        ) from error

    response_headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return StreamingResponse(
        iter([csv_content]),
        media_type=CSV_MEDIA_TYPE,
        headers=response_headers,
    )


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_video_job(video: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    validate_video_upload(video)

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = save_uploaded_video(video, job_dir)
    out_dir = job_dir / "pipeline_output"

    _update_job(
        job_id,
        status="queued",
        created_at=time.time(),
        video_path=str(video_path),
        out_dir=str(out_dir),
    )

    worker = threading.Thread(
        target=_run_video_job,
        args=(job_id, video_path, out_dir),
        daemon=True,
        name=f"price-tag-job-{job_id[:8]}",
    )
    worker.start()
    return _get_job_response(job_id)


@router.get("/jobs/{job_id}")
def get_video_job(job_id: str) -> dict[str, object]:
    return _get_job_response(job_id)


@router.get("/jobs/{job_id}/csv", response_class=FileResponse)
def get_video_job_csv(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not completed yet: {job.get('status')}",
        )

    csv_path = Path(str(job.get("csv_path", "")))
    if not csv_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="CSV file is not available for this job.",
        )
    return FileResponse(csv_path, media_type=CSV_MEDIA_TYPE, filename=csv_path.name)


def validate_video_upload(video: UploadFile) -> None:
    if video.content_type is None or not video.content_type.startswith(VIDEO_CONTENT_TYPE_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Загруженный файл должен быть видео",
        )


def save_uploaded_video(upload_file: UploadFile, temp_dir: Path) -> Path:
    suffix = Path(upload_file.filename or "").suffix or ".mp4"
    video_path = temp_dir / f"uploaded_video{suffix}"
    try:
        with video_path.open("wb") as output:
            shutil.copyfileobj(upload_file.file, output)
    finally:
        upload_file.file.close()
    return video_path


def build_pipeline_args(video_path: Path, out_dir: Path) -> argparse.Namespace:
    extra_args = shlex.split(os.getenv("PRICE_TAG_API_PIPELINE_ARGS", ""))
    try:
        return parse_args([str(video_path), "--out-dir", str(out_dir), *DEFAULT_API_ARGS, *extra_args])
    except SystemExit as error:
        raise RuntimeError("Invalid PRICE_TAG_API_PIPELINE_ARGS configuration.") from error


def _run_video_job(job_id: str, video_path: Path, out_dir: Path) -> None:
    _update_job(job_id, status="running", started_at=time.time())
    try:
        args = build_pipeline_args(video_path, out_dir)
        report = run_pipeline(args)
        log_pipeline_report(report)
        raise_for_ocr_failure_if_requested(args, report)
        csv_path = select_pipeline_csv(report)
    except Exception as error:
        LOGGER.exception("Pipeline job %s failed", job_id)
        _update_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            error=str(error),
        )
        return

    _update_job(
        job_id,
        status="completed",
        finished_at=time.time(),
        report=report,
        csv_path=str(csv_path),
        filename=csv_path.name,
    )


def _update_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {"job_id": job_id})
        job.update(updates)


def _get_job(job_id: str) -> dict[str, object]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job was not found.",
            )
        return dict(job)


def _get_job_response(job_id: str) -> dict[str, object]:
    job = _get_job(job_id)
    response_keys = {
        "job_id",
        "status",
        "created_at",
        "started_at",
        "finished_at",
        "error",
        "filename",
        "report",
    }
    return {key: value for key, value in job.items() if key in response_keys}


def log_pipeline_report(report: dict[str, object]) -> None:
    ocr_summary = report.get("ocr_summary")
    LOGGER.info(
        "Pipeline finished: detector=%s, frames=%s/%s, detections=%s, raw_tracks=%s, "
        "valid_tracks=%s, unique_counted_tracks=%s, saved_crops=%s, run_ocr=%s, ocr_summary=%s",
        report.get("detector"),
        report.get("processed_frames"),
        report.get("frame_count"),
        report.get("detection_rows"),
        report.get("raw_tracks"),
        report.get("valid_tracks"),
        report.get("unique_counted_tracks"),
        report.get("saved_crops"),
        isinstance(ocr_summary, dict) and ocr_summary.get("enabled") is True,
        ocr_summary,
    )


def raise_for_ocr_failure_if_requested(args: argparse.Namespace, report: dict[str, object]) -> None:
    if not getattr(args, "run_ocr", False):
        return

    ocr_summary = report.get("ocr_summary")
    if not isinstance(ocr_summary, dict) or ocr_summary.get("enabled") is not True:
        raise RuntimeError("OCR was requested, but the OCR stage did not run.")

    if ocr_summary.get("error"):
        raise RuntimeError(f"OCR stage failed: {ocr_summary['error']}")

    rows_requested = _safe_int(ocr_summary.get("rows_requested"))
    rows_processed = _safe_int(ocr_summary.get("rows_processed"))
    ocr_errors = _safe_int(ocr_summary.get("ocr_errors"))
    if rows_requested == 0:
        diagnostics = (
            f"processed_frames={report.get('processed_frames')}, "
            f"detections={report.get('detection_rows')}, "
            f"raw_tracks={report.get('raw_tracks')}, "
            f"valid_tracks={report.get('valid_tracks')}, "
            f"unique_counted_tracks={report.get('unique_counted_tracks')}, "
            f"saved_crops={report.get('saved_crops')}"
        )
        raise RuntimeError(
            "OCR was requested, but no crop rows were selected for OCR. "
            "The detector/tracker did not produce usable price-tag crops. "
            f"Diagnostics: {diagnostics}."
        )

    if rows_processed > 0 and ocr_errors >= rows_processed:
        raise RuntimeError(
            f"OCR stage processed {rows_processed} crop(s), but every OCR request failed."
        )


def _safe_int(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def select_pipeline_csv(report: dict[str, object]) -> Path:
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("Pipeline report does not contain output paths.")

    structured_output_path = outputs.get("structured_price_tags_csv")
    if structured_output_path:
        structured_csv_path = Path(str(structured_output_path))
        if structured_csv_path.is_file():
            return structured_csv_path

        ocr_summary = report.get("ocr_summary")
        if isinstance(ocr_summary, dict) and ocr_summary.get("error"):
            raise RuntimeError(f"OCR failed: {ocr_summary['error']}")
        raise RuntimeError("OCR was enabled, but structured_price_tags.csv was not produced.")

    unique_output_path = outputs.get("unique_price_tags_csv")
    if unique_output_path:
        unique_csv_path = Path(str(unique_output_path))
        if unique_csv_path.is_file():
            return unique_csv_path

    raise RuntimeError("Pipeline did not produce a CSV file.")
