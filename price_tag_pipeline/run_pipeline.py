#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from price_tag_pipeline.core.cli import parse_args
from price_tag_pipeline.core.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_pipeline(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
