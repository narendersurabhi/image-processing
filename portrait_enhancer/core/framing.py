"""Frame geometry: straighten, flip, and crop.

Applied as an output-stage transform on the rendered image, so it does not touch
the segmentation/mask pipeline. The crop rectangle is stored in **normalized**
coordinates (0..1) of the straightened canvas, which makes it resolution
independent: the same framing applies to the interactive proxy preview and the
full-resolution export.

This module is GUI-free so the geometry and the interactive crop-box math can be
unit tested without Qt.
"""

from __future__ import annotations

from PIL import Image, ImageOps

MAX_ANGLE = 45.0
MIN_CROP = 0.05  # smallest crop edge as a fraction of the canvas
DEFAULT_CROP = (0.0, 0.0, 1.0, 1.0)

# Interior plus the 8 resize handles, in normalized space.
HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def default_framing() -> dict:
    return {"angle": 0.0, "flip_h": False, "flip_v": False, "crop": list(DEFAULT_CROP)}


def normalize_framing(data) -> dict:
    """Coerce arbitrary input into a valid framing dict (clamped, typed)."""
    data = data or {}
    angle = _clamp(float(data.get("angle", 0.0) or 0.0), -MAX_ANGLE, MAX_ANGLE)
    flip_h = bool(data.get("flip_h", False))
    flip_v = bool(data.get("flip_v", False))
    crop = data.get("crop", DEFAULT_CROP)
    try:
        x, y, w, h = (float(v) for v in crop)
    except (TypeError, ValueError):
        x, y, w, h = DEFAULT_CROP
    w = _clamp(w, MIN_CROP, 1.0)
    h = _clamp(h, MIN_CROP, 1.0)
    x = _clamp(x, 0.0, 1.0 - w)
    y = _clamp(y, 0.0, 1.0 - h)
    return {"angle": angle, "flip_h": flip_h, "flip_v": flip_v, "crop": [x, y, w, h]}


def is_identity(framing) -> bool:
    f = normalize_framing(framing)
    x, y, w, h = f["crop"]
    return (
        abs(f["angle"]) < 1e-6
        and not f["flip_h"]
        and not f["flip_v"]
        and abs(x) < 1e-6
        and abs(y) < 1e-6
        and abs(w - 1.0) < 1e-6
        and abs(h - 1.0) < 1e-6
    )


def apply_framing(image: Image.Image, framing) -> Image.Image:
    """Apply flip -> straighten -> crop to a PIL image."""
    f = normalize_framing(framing)
    if is_identity(f):
        return image

    out = image
    if f["flip_h"]:
        out = ImageOps.mirror(out)
    if f["flip_v"]:
        out = ImageOps.flip(out)
    if abs(f["angle"]) >= 1e-6:
        # Positive angle straightens a clockwise-tilted horizon (counter-clockwise
        # rotation in PIL). Keep the canvas size; exposed corners fill black and the
        # crop box is expected to exclude them.
        out = out.rotate(f["angle"], resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    x, y, w, h = f["crop"]
    width, height = out.size
    x0 = int(round(x * width))
    y0 = int(round(y * height))
    x1 = int(round((x + w) * width))
    y1 = int(round((y + h) * height))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    if (x0, y0, x1, y1) == (0, 0, width, height):
        return out
    return out.crop((x0, y0, x1, y1))


def centered_crop_for_aspect(aspect: float | None, image_w: int, image_h: int) -> list[float]:
    """Largest centered crop with width/height ratio ``aspect`` (None = full)."""
    if not aspect or aspect <= 0 or image_w <= 0 or image_h <= 0:
        return list(DEFAULT_CROP)
    frame_aspect = image_w / image_h
    if aspect >= frame_aspect:
        # Crop is wider than the frame -> full width, reduced height.
        w = 1.0
        h = (frame_aspect / aspect)
    else:
        h = 1.0
        w = (aspect / frame_aspect)
    w = _clamp(w, MIN_CROP, 1.0)
    h = _clamp(h, MIN_CROP, 1.0)
    return [(1.0 - w) / 2.0, (1.0 - h) / 2.0, w, h]


# --- Interactive crop-box math (normalized space) -----------------------------

def _handle_points(crop) -> dict:
    x, y, w, h = crop
    cx, cy = x + w / 2.0, y + h / 2.0
    return {
        "nw": (x, y),
        "n": (cx, y),
        "ne": (x + w, y),
        "e": (x + w, cy),
        "se": (x + w, y + h),
        "s": (cx, y + h),
        "sw": (x, y + h),
        "w": (x, cy),
    }


def hit_test_handle(crop, px: float, py: float, tol: float) -> str | None:
    """Return the handle name nearest (px, py), 'move' if inside, else None."""
    best = None
    best_dist = tol
    for name, (hx, hy) in _handle_points(crop).items():
        dist = max(abs(px - hx), abs(py - hy))  # Chebyshev: square handle hit area
        if dist <= best_dist:
            best = name
            best_dist = dist
    if best is not None:
        return best
    x, y, w, h = crop
    if x <= px <= x + w and y <= py <= y + h:
        return "move"
    return None


def move_crop(crop, dx: float, dy: float) -> list[float]:
    x, y, w, h = crop
    x = _clamp(x + dx, 0.0, 1.0 - w)
    y = _clamp(y + dy, 0.0, 1.0 - h)
    return [x, y, w, h]


def resize_crop(crop, handle: str, dx: float, dy: float, *, min_size: float = MIN_CROP) -> list[float]:
    """Move the edges referenced by ``handle`` by (dx, dy), clamped to the canvas."""
    if handle == "move":
        return move_crop(crop, dx, dy)
    x, y, w, h = crop
    left, top, right, bottom = x, y, x + w, y + h
    if "w" in handle:
        left = _clamp(left + dx, 0.0, right - min_size)
    if "e" in handle:
        right = _clamp(right + dx, left + min_size, 1.0)
    if "n" in handle:
        top = _clamp(top + dy, 0.0, bottom - min_size)
    if "s" in handle:
        bottom = _clamp(bottom + dy, top + min_size, 1.0)
    return [left, top, right - left, bottom - top]
