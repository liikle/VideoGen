"""Resolution, aspect-ratio and frame-count arithmetic shared by engines and post-processing."""

from __future__ import annotations

import math

from PIL import Image

_TARGETS = {
    ("1080p", "16:9"): (1920, 1080),
    ("1080p", "9:16"): (1080, 1920),
    ("1080p", "1:1"): (1080, 1080),
    ("720p", "16:9"): (1280, 720),
    ("720p", "9:16"): (720, 1280),
    ("720p", "1:1"): (720, 720),
}


def target_size(resolution: str, aspect: str) -> tuple[int, int]:
    """Final delivered (width, height) for a resolution preset and aspect ratio."""
    return _TARGETS[(resolution, aspect)]


def aspect_value(aspect: str) -> float:
    w, h = aspect.split(":")
    return int(w) / int(h)


def round_to(x: float, multiple: int) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def ceil_to(x: float, multiple: int) -> int:
    return max(multiple, int(math.ceil(x / multiple)) * multiple)


def snap_frames(n: int, step: int) -> int:
    """Smallest frame count of the form k*step + 1 that is >= n (video VAEs need this shape)."""
    n = max(int(n), 1)
    return int(math.ceil((n - 1) / step)) * step + 1


def frames_for(duration: float, fps: float, step: int) -> int:
    return snap_frames(max(int(round(duration * fps)), step + 1), step)


def fit_area(aspect: str, area: int, multiple: int) -> tuple[int, int]:
    """(width, height) with the given aspect ratio, about `area` pixels, both divisible by `multiple`."""
    ratio = aspect_value(aspect)
    width = math.sqrt(area * ratio)
    height = width / ratio
    return round_to(width, multiple), round_to(height, multiple)


def cover_size(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    """Size to scale (src_w, src_h) to so that it covers (dst_w, dst_h) while keeping its aspect.

    Dimensions are rounded up to even numbers, which video encoders need.
    """
    scale = max(dst_w / src_w, dst_h / src_h)
    w = max(dst_w, int(math.ceil(src_w * scale)))
    h = max(dst_h, int(math.ceil(src_h * scale)))
    return w + (w % 2), h + (h % 2)


def crop_to_aspect_size(src_w: int, src_h: int, aspect_ratio: float) -> tuple[int, int]:
    """Largest even-sized (w, h) inside (src_w, src_h) with the given aspect ratio."""
    if src_w / src_h > aspect_ratio:
        h = src_h
        w = int(round(h * aspect_ratio))
    else:
        w = src_w
        h = int(round(w / aspect_ratio))
    w, h = min(w, src_w), min(h, src_h)
    return w - (w % 2), h - (h % 2)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop an image to the target aspect ratio and resize it to exactly (width, height)."""
    image = image.convert("RGB")
    src_w, src_h = image.size
    target_ratio = width / height
    if src_w / src_h > target_ratio:
        new_w = int(round(src_h * target_ratio))
        left = (src_w - new_w) // 2
        box = (left, 0, left + new_w, src_h)
    else:
        new_h = int(round(src_w / target_ratio))
        top = (src_h - new_h) // 2
        box = (0, top, src_w, top + new_h)
    return image.crop(box).resize((width, height), Image.LANCZOS)


def plan_segments(total_frames: int, segment_frames: int, step: int) -> list[int]:
    """Split a long clip into chained segments.

    Every segment after the first starts from the previous segment's last frame,
    so it contributes (length - 1) new frames. Lengths are k*step + 1.
    """
    first = min(segment_frames, snap_frames(total_frames, step))
    segments = [first]
    remaining = total_frames - first
    while remaining > 0:
        length = min(segment_frames, snap_frames(remaining + 1, step))
        segments.append(length)
        remaining -= length - 1
    return segments
