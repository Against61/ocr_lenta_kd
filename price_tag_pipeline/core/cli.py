from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_MODEL = PROJECT_ROOT / "models/openfoodfacts-price-tag-detection/weights/model_ir_8_opset_17.onnx"
DEFAULT_QUALITY_MODEL = PROJECT_ROOT / "models/lenta_quality_cnn.npz"
DEFAULT_CODE_DETECTOR_DIR = PROJECT_ROOT / "models/qr_detector"
DEFAULT_OCR_BASE_URL = "http://localhost:1234/v1"
DEFAULT_OCR_MODEL = "local-vlm"
DEFAULT_OCR_IMAGE_PREPROCESS = "none"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect, track, count, and crop shelf price tags in a moving video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--detector", choices=["heuristic", "onnx"], default="heuristic")
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--onnx-input-size", type=int, default=960)
    parser.add_argument("--onnx-conf-threshold", type=float, default=0.25)
    parser.add_argument("--onnx-iou-threshold", type=float, default=0.55)

    parser.add_argument("--width", type=int, default=720, help="Analysis width after orientation normalization.")
    parser.add_argument("--frame-step", type=int, default=1, help="Run detector on each Nth source frame.")
    parser.add_argument("--rotate-ccw", action="store_true", default=True)
    parser.add_argument("--rotate-cw", action="store_true")
    parser.add_argument("--no-rotate", action="store_true")

    parser.add_argument("--max-link-distance", type=float, default=46.0)
    parser.add_argument("--max-missed", type=int, default=8)
    parser.add_argument("--max-backward-step", type=float, default=8.0)
    parser.add_argument("--max-y-jump", type=float, default=32.0)
    parser.add_argument("--right-edge-retire-ratio", type=float, default=0.94)
    parser.add_argument("--right-edge-ignore-ratio", type=float, default=0.96)

    parser.add_argument("--min-track-frames", type=int, default=18)
    parser.add_argument("--min-delta-x", type=float, default=0.0)
    parser.add_argument("--min-track-white-ratio", type=float, default=0.19)
    parser.add_argument("--min-track-aspect", type=float, default=0.70)
    parser.add_argument("--max-track-height", type=int, default=0, help="Optional bbox height cap. 0 disables this filter.")
    parser.add_argument("--max-crop-candidates", type=int, default=24)

    parser.add_argument("--deskew-crops", action="store_true", default=True)
    parser.add_argument("--no-deskew-crops", action="store_false", dest="deskew_crops")
    parser.add_argument("--crop-deskew-max-angle", type=float, default=18.0)
    parser.add_argument("--crop-deskew-min-angle", type=float, default=1.0)
    parser.add_argument("--crop-deskew-min-confidence", type=float, default=0.10)

    parser.add_argument(
        "--quality-model",
        type=Path,
        default=DEFAULT_QUALITY_MODEL if DEFAULT_QUALITY_MODEL.exists() else None,
    )
    parser.add_argument("--no-quality-model", action="store_const", const=None, dest="quality_model")
    parser.add_argument("--quality-input-size", type=int, default=None)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--quality-pass-class", type=int, default=1)
    parser.add_argument("--quality-input-mode", choices=["grayscale", "rgb"], default=None)
    parser.add_argument("--quality-normalization", choices=["none", "imagenet"], default="none")

    parser.add_argument("--run-ocr", action="store_true", help="Run local LM Studio OCR on saved quality-passed crops.")
    parser.add_argument(
        "--ocr-base-url",
        default=_first_env("PRICE_TAG_OCR_BASE_URL", "LLAMA_CPP_BASE_URL", default=DEFAULT_OCR_BASE_URL),
    )
    parser.add_argument(
        "--ocr-api-key",
        default=_first_env("PRICE_TAG_OCR_API_KEY", "LLAMA_CPP_API_KEY", default="not-needed"),
    )
    parser.add_argument(
        "--ocr-model",
        default=_first_env("PRICE_TAG_OCR_MODEL", "LLAMA_CPP_MODEL", default=DEFAULT_OCR_MODEL),
    )
    parser.add_argument("--ocr-max-tokens", type=int, default=int(_first_env("PRICE_TAG_OCR_MAX_TOKENS", "LLAMA_CPP_MAX_TOKENS", default="900")))
    parser.add_argument("--ocr-temperature", type=float, default=float(_first_env("PRICE_TAG_OCR_TEMPERATURE", "LLAMA_CPP_TEMPERATURE", default="0.0")))
    parser.add_argument("--ocr-timeout", type=float, default=float(_first_env("PRICE_TAG_OCR_TIMEOUT", "LLAMA_CPP_TIMEOUT", default="180.0")))
    parser.add_argument("--ocr-retries", type=int, default=int(_first_env("PRICE_TAG_OCR_RETRIES", "LLAMA_CPP_RETRIES", default="1")))
    parser.add_argument("--ocr-response-format", action="store_true", default=_env_bool("PRICE_TAG_OCR_RESPONSE_FORMAT", _env_bool("LLAMA_CPP_RESPONSE_FORMAT", False)))
    parser.add_argument("--ocr-disable-reasoning", action="store_true", default=_env_bool("PRICE_TAG_OCR_DISABLE_REASONING", True))
    parser.add_argument("--ocr-enable-reasoning", action="store_false", dest="ocr_disable_reasoning")
    parser.add_argument(
        "--ocr-image-preprocess",
        choices=["none", "grayscale", "clahe", "vlm-board"],
        default=_first_env(
            "PRICE_TAG_OCR_IMAGE_PREPROCESS",
            "LLAMA_CPP_IMAGE_PREPROCESS",
            default=DEFAULT_OCR_IMAGE_PREPROCESS,
        ),
    )
    parser.add_argument("--ocr-include-quality-failed", action="store_true", default=True)
    parser.add_argument("--no-ocr-include-quality-failed", action="store_false", dest="ocr_include_quality_failed")
    parser.add_argument("--ocr-limit", type=int, default=0, help="Process only the first N saved crops. 0 means all.")

    parser.set_defaults(code_detector_enabled=DEFAULT_CODE_DETECTOR_DIR.exists())
    parser.add_argument("--code-detector", action="store_true", dest="code_detector_enabled")
    parser.add_argument("--no-code-detector", action="store_false", dest="code_detector_enabled")
    parser.add_argument("--code-detector-dir", type=Path, default=DEFAULT_CODE_DETECTOR_DIR)
    parser.add_argument("--code-detector-param", type=Path, default=None)
    parser.add_argument("--code-detector-bin", type=Path, default=None)
    parser.add_argument("--code-detector-input-size", type=int, default=640)
    parser.add_argument("--code-detector-conf-threshold", type=float, default=0.25)
    parser.add_argument("--code-detector-iou-threshold", type=float, default=0.45)
    parser.add_argument("--code-detector-padding-ratio", type=float, default=0.50)
    parser.add_argument("--code-detector-resize-mode", choices=["stretch", "letterbox"], default="stretch")
    parser.add_argument("--code-detector-class-roles", default="0=qr,1=barcode,2=matrix")
    parser.add_argument("--code-detector-fallback-full-crop", action="store_true", default=True)
    parser.add_argument("--no-code-detector-fallback-full-crop", action="store_false", dest="code_detector_fallback_full_crop")

    parser.add_argument("--count-mode", choices=["seen", "line"], default="seen")
    parser.add_argument("--crossing-x-ratio", type=float, default=0.12)
    parser.add_argument("--line-entry-margin", type=float, default=36.0)
    parser.add_argument("--count-first-at-line", action="store_true", default=True)
    parser.add_argument("--no-count-first-at-line", action="store_false", dest="count_first_at_line")

    parser.add_argument("--write-overlay-video", action="store_true", default=True)
    parser.add_argument("--no-overlay-video", action="store_true")
    parser.add_argument("--debug-frames", default="0,60,90,120,150,210,298")
    return parser.parse_args(argv)


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}
