from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Hugging Face cache download progress.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename")
    parser.add_argument("--selector")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--idle-timeout", type=float, default=3600.0)
    args = parser.parse_args()
    if not args.filename and not args.selector:
        parser.error("either --filename or --selector is required")
    return args


def format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds <= 0 or seconds == float("inf"):
        return "--:--"
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_bar(progress: float | None, width: int = 30) -> str:
    if progress is None:
        return "[" + "?" * width + "]"
    bounded = max(0.0, min(1.0, progress))
    filled = int(round(bounded * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def resolve_model_artifact(
    repo_id: str,
    filename: str | None,
    selector: str | None,
    token: str | None,
) -> tuple[str, int | None]:
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    info = HfApi(token=token).model_info(repo_id)
    resolved_filename = filename
    matched_sibling = None

    if resolved_filename is None:
        normalized_selector = selector or ""
        suffix = f"-{normalized_selector}.gguf"
        candidates = [
            sibling for sibling in info.siblings
            if sibling.rfilename.endswith(suffix) or sibling.rfilename == normalized_selector
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not uniquely resolve selector '{normalized_selector}' in repo '{repo_id}'."
            )
        matched_sibling = candidates[0]
        resolved_filename = matched_sibling.rfilename
    else:
        for sibling in info.siblings:
            if sibling.rfilename == resolved_filename:
                matched_sibling = sibling
                break
        if matched_sibling is None:
            raise RuntimeError(f"File '{resolved_filename}' was not found in repo '{repo_id}'.")

    if matched_sibling is not None and matched_sibling.size is not None:
        return resolved_filename, int(matched_sibling.size)

    try:
        metadata = get_hf_file_metadata(
            hf_hub_url(repo_id=repo_id, filename=resolved_filename),
            token=token,
        )
    except Exception:
        return resolved_filename, None
    size = getattr(metadata, "size", None)
    return resolved_filename, int(size) if size is not None else None


def current_downloaded_bytes(cache_repo_dir: Path) -> int:
    if not cache_repo_dir.exists():
        return 0

    in_progress = sorted(
        cache_repo_dir.rglob("*.downloadInProgress"),
        key=lambda path: path.stat().st_size if path.exists() else 0,
        reverse=True,
    )
    if in_progress:
        return in_progress[0].stat().st_size

    complete_blobs = sorted(
        [path for path in cache_repo_dir.rglob("*") if path.is_file() and "." not in path.name],
        key=lambda path: path.stat().st_size if path.exists() else 0,
        reverse=True,
    )
    if complete_blobs:
        return complete_blobs[0].stat().st_size
    return 0


def print_progress(downloaded: int, total: int | None, start_time: float, prev_bytes: int, prev_time: float) -> tuple[int, float]:
    now = time.monotonic()
    elapsed = max(now - start_time, 1e-6)
    avg_speed = downloaded / elapsed
    delta_bytes = max(downloaded - prev_bytes, 0)
    delta_time = max(now - prev_time, 1e-6)
    speed = delta_bytes / delta_time if delta_bytes > 0 else avg_speed
    progress = downloaded / total if total else None
    remaining = (total - downloaded) if total is not None else None
    eta = (remaining / avg_speed) if (remaining is not None and avg_speed > 0) else None

    if total is not None:
        line = (
            f"{render_bar(progress)} "
            f"{progress * 100:6.2f}% "
            f"{format_bytes(downloaded)} / {format_bytes(total)} "
            f"speed {format_bytes(speed)}/s "
            f"eta {format_eta(eta)}"
        )
    else:
        line = f"{render_bar(None)} size unknown {format_bytes(downloaded)} downloaded speed {format_bytes(speed)}/s"
    print(line, flush=True)
    return downloaded, now


def main() -> int:
    args = parse_args()
    token = os.getenv("HF_TOKEN") or None
    resolved_filename, total_size = resolve_model_artifact(
        args.repo_id,
        args.filename,
        args.selector,
        token,
    )
    cache_repo_dir = Path(args.cache_dir) / "hub" / f"models--{args.repo_id.replace('/', '--')}"
    print(
        f"Monitoring download: repo={args.repo_id} file={resolved_filename} size={format_bytes(total_size) if total_size else 'unknown'}",
        flush=True,
    )

    start_time = time.monotonic()
    prev_time = start_time
    prev_bytes = 0
    last_growth = start_time

    while True:
        downloaded = current_downloaded_bytes(cache_repo_dir)
        if downloaded > prev_bytes:
            last_growth = time.monotonic()
        prev_bytes, prev_time = print_progress(downloaded, total_size, start_time, prev_bytes, prev_time)

        if total_size is not None and downloaded >= total_size:
            print("Download monitoring finished.", flush=True)
            return 0

        if time.monotonic() - last_growth > args.idle_timeout:
            print("No download growth detected within idle timeout; stopping monitor.", flush=True)
            return 0

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
