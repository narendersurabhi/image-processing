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

import math

import numpy as np
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


def _apply_flip_rotate(image: Image.Image, f: dict, *, center=None) -> Image.Image:
    """Mirror/flip then straighten, with the canvas size unchanged (``expand=False``).

    ``center`` overrides the rotation pivot, in this image's own pixel coordinates,
    defaulting to PIL's own image-center pivot when omitted. A caller operating on a
    crop of a larger image can pass the crop-local position of the *full* image's
    center so the crop's rotated content lines up exactly with what rotating the full
    image about its own center and then cropping would have produced.
    """
    out = image
    if f["flip_h"]:
        out = ImageOps.mirror(out)
    if f["flip_v"]:
        out = ImageOps.flip(out)
    if abs(f["angle"]) >= 1e-6:
        # Positive angle straightens a clockwise-tilted horizon (counter-clockwise
        # rotation in PIL). Keep the canvas size; exposed corners fill black and the
        # crop box is expected to exclude them.
        rotate_kwargs = {"resample": Image.BICUBIC, "expand": False, "fillcolor": (0, 0, 0)}
        if center is not None:
            rotate_kwargs["center"] = center
        out = out.rotate(f["angle"], **rotate_kwargs)
    return out


def crop_box_px(f: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """The crop box of an already-normalized framing dict, in pixel coordinates of a
    ``width`` x ``height`` canvas (the rotated, pre-crop canvas -- same size as the
    source image, since rotate uses ``expand=False``). Single source of truth for the
    box math, shared by ``_apply_crop_box`` and callers that need this same box at a
    different resolution (e.g. translating a normalized display rect into full-image
    pixel coordinates for the hi-res tile path)."""
    x, y, w, h = f["crop"]
    x0 = int(round(x * width))
    y0 = int(round(y * height))
    x1 = int(round((x + w) * width))
    y1 = int(round((y + h) * height))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def _apply_crop_box(out: Image.Image, f: dict) -> Image.Image:
    width, height = out.size
    x0, y0, x1, y1 = crop_box_px(f, width, height)
    if (x0, y0, x1, y1) == (0, 0, width, height):
        return out
    return out.crop((x0, y0, x1, y1))


def apply_framing(image: Image.Image, framing) -> Image.Image:
    """Apply flip -> straighten -> crop to a PIL image."""
    f = normalize_framing(framing)
    if is_identity(f):
        return image
    out = _apply_flip_rotate(image, f)
    return _apply_crop_box(out, f)


def display_point_to_source(framing, image_w: int, image_h: int, dx: float, dy: float) -> tuple[float, float]:
    """Invert ``apply_framing`` for a single normalized point.

    Given a point in normalized **display** space (post flip/rotate/crop -- the space
    the live canvas and its pixmap are in), return the corresponding normalized point
    in **source** space (pre-framing, i.e. ``full_array``/``preview_array``'s own
    space). Pure point algebra: a rotation pivot is always the image's true center, so
    there is no pivot ambiguity the way there is for rotating a sub-array in
    ``render_display_tile``.
    """
    f = normalize_framing(framing)
    x, y, w, h = f["crop"]
    # Undo crop: re-expand the normalized point from crop-box-local to full-canvas-local.
    rx = x + float(dx) * w
    ry = y + float(dy) * h

    if abs(f["angle"]) >= 1e-6 and image_w > 0 and image_h > 0:
        # Undo rotate: +angle about the true (aspect-correct) center, in true pixel
        # units so a non-square image rotates correctly. Sign matches PIL's rotate()
        # convention -- pinned down empirically against apply_framing in the test suite.
        cx, cy = image_w / 2.0, image_h / 2.0
        px, py = rx * image_w - cx, ry * image_h - cy
        theta = math.radians(f["angle"])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        sx = cx + (px * cos_t - py * sin_t)
        sy = cy + (px * sin_t + py * cos_t)
        rx, ry = sx / image_w, sy / image_h

    # Undo flip: mirror back around the canvas midline.
    if f["flip_h"]:
        rx = 1.0 - rx
    if f["flip_v"]:
        ry = 1.0 - ry
    return rx, ry


def render_display_tile(
    processed_crop: Image.Image,
    framing,
    full_w: int,
    full_h: int,
    crop_origin: tuple[int, int],
    target_rect_px: tuple[int, int, int, int],
):
    """Produce the final display-oriented tile for ``target_rect_px`` from a crop that
    has already been run through the edit pipeline but is still in source
    (pre-framing) orientation, positioned at ``crop_origin`` within the full source
    image.

    ``target_rect_px`` is a pixel rect (x0, y0, x1, y1) in the rotated display canvas,
    which has the same dimensions as the full source image (``apply_framing`` rotates
    with ``expand=False``). This is the per-tile counterpart to ``apply_framing``:
    flip is a pure array reversal (no centering ambiguity, only bookkeeping of where
    the flipped crop now sits), and rotate uses PIL's ``center`` kwarg so a small crop
    pivots about the *full image's* center rather than its own -- making a windowed
    rotation match what rotating the full image and then cropping would have
    produced. Returns ``None`` if ``target_rect_px`` doesn't overlap the rotated crop
    at all (the caller's padding should normally prevent this).
    """
    f = normalize_framing(framing)
    ox0, oy0 = crop_origin
    cw, ch = processed_crop.size

    out = processed_crop
    if f["flip_h"]:
        out = ImageOps.mirror(out)
        ox0 = full_w - (ox0 + cw)
    if f["flip_v"]:
        out = ImageOps.flip(out)
        oy0 = full_h - (oy0 + ch)
    if abs(f["angle"]) >= 1e-6:
        center = (full_w / 2.0 - ox0, full_h / 2.0 - oy0)
        out = out.rotate(f["angle"], resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0), center=center)

    tx0, ty0, tx1, ty1 = target_rect_px
    lx0, ly0, lx1, ly1 = tx0 - ox0, ty0 - oy0, tx1 - ox0, ty1 - oy0
    width, height = out.size
    lx0c, ly0c = max(0, lx0), max(0, ly0)
    lx1c, ly1c = min(width, lx1), min(height, ly1)
    if lx1c <= lx0c or ly1c <= ly0c:
        return None
    tile = out.crop((lx0c, ly0c, lx1c, ly1c))
    pad_left, pad_top = lx0c - lx0, ly0c - ly0
    pad_right, pad_bottom = lx1 - lx1c, ly1 - ly1c
    if pad_left or pad_top or pad_right or pad_bottom:
        # Requested rect spilled past the rotated crop's covered area -- pad by edge
        # replication rather than show a gap; the caller's padding should make this rare.
        arr = np.array(tile)
        pad_width = ((max(0, pad_top), max(0, pad_bottom)), (max(0, pad_left), max(0, pad_right)))
        if arr.ndim == 3:
            pad_width = pad_width + ((0, 0),)
        tile = Image.fromarray(np.pad(arr, pad_width, mode="edge"))
    return tile


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


def resize_crop(
    crop,
    handle: str,
    dx: float,
    dy: float,
    *,
    min_size: float = MIN_CROP,
    aspect: float | None = None,
    canvas_aspect: float = 1.0,
) -> list[float]:
    """Move the edges referenced by ``handle`` by (dx, dy), clamped to the canvas.

    With ``aspect`` set (a target width:height ratio in true image pixels --
    ``canvas_aspect`` is the straightened source image's own width:height ratio, needed to
    convert that into the matching ratio for the normalized-fraction crop box, since a crop
    box that's square in true pixels isn't square in fraction-space unless the source image
    itself is square), the box resizes to preserve it: corner handles keep the opposite
    corner fixed; edge handles keep the opposite edge fixed and grow/shrink the perpendicular
    dimension symmetrically about its center -- standard crop-tool behavior.
    """
    if handle == "move":
        return move_crop(crop, dx, dy)
    x, y, w, h = crop
    left, top, right, bottom = x, y, x + w, y + h

    if not aspect or aspect <= 0:
        if "w" in handle:
            left = _clamp(left + dx, 0.0, right - min_size)
        if "e" in handle:
            right = _clamp(right + dx, left + min_size, 1.0)
        if "n" in handle:
            top = _clamp(top + dy, 0.0, bottom - min_size)
        if "s" in handle:
            bottom = _clamp(bottom + dy, top + min_size, 1.0)
        return [left, top, right - left, bottom - top]

    frac_aspect = aspect / max(canvas_aspect, 1e-6)  # target width_frac / height_frac

    if handle in ("n", "s"):
        if handle == "n":
            top = _clamp(top + dy, 0.0, bottom - min_size)
        else:
            bottom = _clamp(bottom + dy, top + min_size, 1.0)
        new_h = bottom - top
        new_w = _clamp(new_h * frac_aspect, min_size, 1.0)
        new_h = new_w / frac_aspect
        cx = (left + right) / 2.0
        left, right = cx - new_w / 2.0, cx + new_w / 2.0
        if left < 0.0:
            right -= left
            left = 0.0
        if right > 1.0:
            left -= right - 1.0
            right = 1.0
        left, right = max(0.0, left), min(1.0, right)
        new_h = (right - left) / frac_aspect
        if handle == "n":
            top = bottom - new_h
        else:
            bottom = top + new_h
        return [left, top, right - left, bottom - top]

    if handle in ("e", "w"):
        if handle == "w":
            left = _clamp(left + dx, 0.0, right - min_size)
        else:
            right = _clamp(right + dx, left + min_size, 1.0)
        new_w = right - left
        new_h = _clamp(new_w / frac_aspect, min_size, 1.0)
        new_w = new_h * frac_aspect
        cy = (top + bottom) / 2.0
        top, bottom = cy - new_h / 2.0, cy + new_h / 2.0
        if top < 0.0:
            bottom -= top
            top = 0.0
        if bottom > 1.0:
            top -= bottom - 1.0
            bottom = 1.0
        top, bottom = max(0.0, top), min(1.0, bottom)
        new_w = (bottom - top) * frac_aspect
        if handle == "w":
            left = right - new_w
        else:
            right = left + new_w
        return [left, top, right - left, bottom - top]

    # Corner handle: opposite corner is the anchor; whichever axis moved more (in true-pixel
    # terms) drives the resize, deriving the other dimension from the locked ratio.
    anchor_x = left if "e" in handle else right
    anchor_y = top if "s" in handle else bottom
    if "w" in handle:
        free_x = _clamp(left + dx, 0.0, anchor_x - min_size)
    else:
        free_x = _clamp(right + dx, anchor_x + min_size, 1.0)
    if "n" in handle:
        free_y = _clamp(top + dy, 0.0, anchor_y - min_size)
    else:
        free_y = _clamp(bottom + dy, anchor_y + min_size, 1.0)
    cand_w, cand_h = abs(free_x - anchor_x), abs(free_y - anchor_y)
    if cand_h <= 0 or cand_w / frac_aspect >= cand_h:
        new_w, new_h = cand_w, cand_w / frac_aspect
    else:
        new_h, new_w = cand_h, cand_h * frac_aspect

    # Available room is *toward the edge being dragged*, away from the fixed anchor: an "e"
    # handle grows rightward from the anchor, so the limit is the space to the canvas's right
    # edge (1 - anchor_x), not back toward the left edge.
    max_w = (1.0 - anchor_x) if "e" in handle else anchor_x
    max_h = (1.0 - anchor_y) if "s" in handle else anchor_y
    if new_w > max_w:
        new_w, new_h = max_w, max_w / frac_aspect
    if new_h > max_h:
        new_h, new_w = max_h, max_h * frac_aspect
    new_w = max(min_size, new_w)
    new_h = max(min_size, new_h)

    if "w" in handle:
        left, right = anchor_x - new_w, anchor_x
    else:
        left, right = anchor_x, anchor_x + new_w
    if "n" in handle:
        top, bottom = anchor_y - new_h, anchor_y
    else:
        top, bottom = anchor_y, anchor_y + new_h
    return [left, top, right - left, bottom - top]
