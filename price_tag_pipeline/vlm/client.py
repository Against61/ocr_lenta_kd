"""External VLM service integration point.

The VLM stage receives selected, processed price-tag crops one by one and
returns structured JSON with product and price information. The concrete HTTP
request schema is intentionally left open until the service contract is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VlmPriceTagResult:
    crop_path: Path
    response_json: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class VlmClient:
    endpoint_url: str
    timeout_sec: float = 30.0

    def analyze_crop(self, crop_path: Path) -> VlmPriceTagResult:
        raise NotImplementedError("VLM request schema is not implemented yet.")
