from __future__ import annotations


def parse_debug_frames(value: str) -> set[int]:
    frames = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        frames.add(int(item))
    return frames

