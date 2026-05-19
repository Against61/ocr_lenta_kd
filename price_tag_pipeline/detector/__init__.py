"""Price tag detector implementations."""

from price_tag_pipeline.detector.factory import build_detector
from price_tag_pipeline.detector.types import Detection

__all__ = ["Detection", "build_detector"]

