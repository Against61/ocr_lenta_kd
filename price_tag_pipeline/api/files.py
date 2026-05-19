import argparse
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from price_tag_pipeline.core.cli import parse_args
from price_tag_pipeline.core.pipeline import run_pipeline

router = APIRouter(prefix="/video", tags=["video"])

VIDEO_CONTENT_TYPE_PREFIX = "video/"
CSV_MEDIA_TYPE = "text/csv"
DEFAULT_API_ARGS = ("--no-overlay-video",)


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


def select_pipeline_csv(report: dict[str, object]) -> Path:
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("Pipeline report does not contain output paths.")

    for output_key in ("structured_price_tags_csv", "unique_price_tags_csv"):
        output_path = outputs.get(output_key)
        if not output_path:
            continue

        csv_path = Path(str(output_path))
        if csv_path.is_file():
            return csv_path

    raise RuntimeError("Pipeline did not produce a CSV file.")
