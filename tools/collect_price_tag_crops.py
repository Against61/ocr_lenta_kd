#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_IMAGE_EXTENSIONS = ".jpg,.jpeg,.png,.webp,.bmp"
DEFAULT_CSV_NAME = "unique_price_tags.csv"
DEFAULT_CLASS_NAME = "unlabeled"

CSV_METADATA_COLUMNS = [
    "track_id",
    "counted",
    "counted_frame",
    "counted_time_sec",
    "first_frame",
    "first_time_sec",
    "last_frame",
    "last_time_sec",
    "frames_seen",
    "best_frame",
    "best_time_sec",
    "best_x",
    "best_y",
    "best_w",
    "best_h",
    "best_score",
    "best_sharpness",
]

MANIFEST_COLUMNS = [
    "label",
    "output_path",
    "copy_status",
    "source_path",
    "source_csv",
    "dedupe_mode",
    "dedupe_key",
    "content_sha1",
    *CSV_METADATA_COLUMNS,
]


@dataclass
class CropCandidate:
    source_path: Path
    source_csv: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect price-tag crop images from pipeline outputs into one folder "
            "for manual labeling or binary quality-classifier training."
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Input CSV files, crop image files, or directories to scan recursively.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--class-name",
        default=DEFAULT_CLASS_NAME,
        help="Destination subfolder and manifest label. Default: unlabeled.",
    )
    parser.add_argument(
        "--mode",
        choices=["csv", "crops", "both"],
        default="both",
        help="Collect from CSV crop_path rows, crops/ folders, or both. Default: both.",
    )
    parser.add_argument(
        "--csv-name",
        default=DEFAULT_CSV_NAME,
        help="CSV filename to search inside directories. Default: unique_price_tags.csv.",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_IMAGE_EXTENSIONS,
        help=f"Comma-separated image extensions. Default: {DEFAULT_IMAGE_EXTENSIONS}.",
    )
    parser.add_argument(
        "--dedupe",
        choices=["hash", "path", "none"],
        default="hash",
        help="Duplicate handling. Default: hash.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to put images into the output folder. Default: copy.",
    )
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument("--skipped-name", default="skipped.csv")
    parser.add_argument("--limit", type=int, default=0, help="Copy at most N images. 0 means no limit.")
    parser.add_argument("--min-size-bytes", type=int, default=1, help="Skip empty or too-small files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination files if names collide.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and print summary without writing files.")
    return parser.parse_args()


def parse_extensions(raw: str) -> set[str]:
    extensions = set()
    for item in raw.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        extensions.add(ext)
    return extensions


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def is_image(path: Path, extensions: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extensions


def iter_candidates(
    sources: list[Path],
    mode: str,
    csv_name: str,
    extensions: set[str],
    out_dir: Path,
) -> tuple[list[CropCandidate], list[dict[str, str]]]:
    candidates: list[CropCandidate] = []
    skipped: list[dict[str, str]] = []

    for source in sources:
        source = source.expanduser()
        if not source.exists():
            skipped.append(skip_row(source, "missing_source"))
            continue

        if source.is_file():
            if source.suffix.lower() == ".csv" and mode in {"csv", "both"}:
                candidates.extend(read_csv_candidates(source, skipped))
            elif is_image(source, extensions):
                candidates.append(CropCandidate(source_path=source))
            else:
                skipped.append(skip_row(source, "unsupported_file"))
            continue

        if source.is_dir():
            if mode in {"csv", "both"}:
                for csv_path in sorted(source.rglob(csv_name)):
                    if csv_path.is_file() and not is_under(csv_path, out_dir):
                        candidates.extend(read_csv_candidates(csv_path, skipped))

            if mode in {"crops", "both"}:
                for crops_dir in sorted(source.rglob("crops")):
                    if not crops_dir.is_dir() or is_under(crops_dir, out_dir):
                        continue
                    for image_path in sorted(crops_dir.rglob("*")):
                        if is_image(image_path, extensions) and not is_under(image_path, out_dir):
                            candidates.append(CropCandidate(source_path=image_path))

    return candidates, skipped


def read_csv_candidates(csv_path: Path, skipped: list[dict[str, str]]) -> list[CropCandidate]:
    candidates: list[CropCandidate] = []
    try:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if "crop_path" not in (reader.fieldnames or []):
                skipped.append(skip_row(csv_path, "csv_without_crop_path"))
                return candidates

            for row_idx, row in enumerate(reader, start=2):
                raw_crop_path = (row.get("crop_path") or "").strip()
                if not raw_crop_path:
                    skipped.append(skip_row(csv_path, "empty_crop_path", row=row_idx))
                    continue

                crop_path = Path(raw_crop_path).expanduser()
                if not crop_path.is_absolute():
                    crop_path = csv_path.parent / crop_path

                if not crop_path.exists():
                    skipped.append(skip_row(crop_path, "missing_crop", csv=csv_path, row=row_idx))
                    continue

                candidates.append(
                    CropCandidate(
                        source_path=crop_path,
                        source_csv=csv_path,
                        metadata={key: row.get(key, "") for key in CSV_METADATA_COLUMNS},
                    )
                )
    except OSError as exc:
        skipped.append(skip_row(csv_path, f"csv_read_error:{exc}"))
    return candidates


def process_candidates(
    candidates: list[CropCandidate],
    out_dir: Path,
    class_name: str,
    dedupe_mode: str,
    copy_mode: str,
    limit: int,
    min_size_bytes: int,
    overwrite: bool,
    dry_run: bool,
    initial_skipped: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    image_dir = out_dir / class_name
    manifest_rows: list[dict[str, str]] = []
    skipped = list(initial_skipped)
    seen_keys: set[str] = set()

    if not dry_run:
        image_dir.mkdir(parents=True, exist_ok=True)

    for index, candidate in enumerate(candidates, start=1):
        source_path = candidate.source_path
        if not source_path.is_file():
            skipped.append(skip_row(source_path, "not_a_file", csv=candidate.source_csv))
            continue

        try:
            if source_path.stat().st_size < min_size_bytes:
                skipped.append(skip_row(source_path, "too_small", csv=candidate.source_csv))
                continue
        except OSError as exc:
            skipped.append(skip_row(source_path, f"stat_error:{exc}", csv=candidate.source_csv))
            continue

        try:
            content_sha1 = sha1_file(source_path)
        except OSError as exc:
            skipped.append(skip_row(source_path, f"hash_error:{exc}", csv=candidate.source_csv))
            continue

        dedupe_key = build_dedupe_key(source_path, content_sha1, dedupe_mode, index)
        if dedupe_mode != "none" and dedupe_key in seen_keys:
            skipped.append(skip_row(source_path, "duplicate", csv=candidate.source_csv))
            continue
        seen_keys.add(dedupe_key)

        output_name = build_output_name(source_path, content_sha1, dedupe_key, dedupe_mode, index)
        output_path = image_dir / output_name
        copy_status = "dry_run" if dry_run else "copied"

        if output_path.exists() and not overwrite:
            copy_status = "existing"
        elif not dry_run:
            copy_image(source_path, output_path, copy_mode, overwrite)

        row = {
            "label": class_name,
            "output_path": str(output_path),
            "copy_status": copy_status,
            "source_path": str(source_path.resolve(strict=False)),
            "source_csv": str(candidate.source_csv.resolve(strict=False)) if candidate.source_csv else "",
            "dedupe_mode": dedupe_mode,
            "dedupe_key": dedupe_key,
            "content_sha1": content_sha1,
        }
        for key in CSV_METADATA_COLUMNS:
            row[key] = candidate.metadata.get(key, "")
        manifest_rows.append(row)

        if limit and len(manifest_rows) >= limit:
            break

    return manifest_rows, skipped


def build_dedupe_key(source_path: Path, content_sha1: str, dedupe_mode: str, index: int) -> str:
    if dedupe_mode == "hash":
        return content_sha1
    if dedupe_mode == "path":
        return sha1_text(str(source_path.resolve(strict=False)))
    return sha1_text(f"{source_path.resolve(strict=False)}:{index}")


def build_output_name(
    source_path: Path,
    content_sha1: str,
    dedupe_key: str,
    dedupe_mode: str,
    index: int,
) -> str:
    stem = safe_stem(source_path.stem)
    suffix = source_path.suffix.lower()
    if dedupe_mode == "none":
        prefix = f"{index:06d}_{sha1_text(str(source_path.resolve(strict=False)))[:10]}"
    elif dedupe_mode == "hash":
        prefix = content_sha1[:12]
    else:
        prefix = dedupe_key[:12]
    return f"{prefix}_{stem}{suffix}"


def safe_stem(raw: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return (stem or "crop")[:96]


def copy_image(source_path: Path, output_path: Path, copy_mode: str, overwrite: bool) -> None:
    if overwrite and output_path.exists():
        output_path.unlink()

    if copy_mode == "hardlink":
        try:
            os.link(source_path, output_path)
            return
        except OSError:
            pass

    shutil.copy2(source_path, output_path)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def skip_row(
    path: Path,
    reason: str,
    csv: Path | None = None,
    row: int | None = None,
) -> dict[str, str]:
    return {
        "path": str(path),
        "reason": reason,
        "source_csv": str(csv) if csv else "",
        "row": str(row) if row is not None else "",
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    extensions = parse_extensions(args.extensions)
    out_dir = args.out_dir.expanduser()

    candidates, initial_skipped = iter_candidates(
        sources=args.sources,
        mode=args.mode,
        csv_name=args.csv_name,
        extensions=extensions,
        out_dir=out_dir,
    )
    manifest_rows, skipped_rows = process_candidates(
        candidates=candidates,
        out_dir=out_dir,
        class_name=args.class_name,
        dedupe_mode=args.dedupe,
        copy_mode=args.copy_mode,
        limit=args.limit,
        min_size_bytes=args.min_size_bytes,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        initial_skipped=initial_skipped,
    )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(out_dir / args.manifest_name, manifest_rows, MANIFEST_COLUMNS)
        write_csv(out_dir / args.skipped_name, skipped_rows, ["path", "reason", "source_csv", "row"])

    summary = {
        "sources": [str(source) for source in args.sources],
        "out_dir": str(out_dir),
        "class_name": args.class_name,
        "mode": args.mode,
        "dedupe": args.dedupe,
        "dry_run": args.dry_run,
        "candidates_found": len(candidates),
        "images_collected": len(manifest_rows),
        "skipped": len(skipped_rows),
        "image_dir": str(out_dir / args.class_name),
        "manifest": str(out_dir / args.manifest_name),
        "skipped_csv": str(out_dir / args.skipped_name),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(manifest_rows) == 0:
        print("No crop images were collected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
