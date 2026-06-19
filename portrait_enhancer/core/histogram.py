"""Histogram computation for the edited preview.

Kept free of any GUI dependency so the binning logic can be unit tested without
Qt. The Qt widget consumes the plain dict returned by :func:`compute_histogram`.
"""

from __future__ import annotations

import numpy as np

LEVELS = 256
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _to_rgb_uint8(image) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB array from a PIL image or ndarray."""
    if hasattr(image, "convert"):  # PIL.Image
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        # Assume float images are in the 0..1 working range.
        arr = np.clip(arr.astype(np.float32) * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return arr


def _counts(channel: np.ndarray) -> np.ndarray:
    return np.bincount(channel.ravel(), minlength=LEVELS)[:LEVELS].astype(np.int64)


def compute_histogram(image) -> dict:
    """Compute per-channel and luminance histograms for ``image``.

    Returns a dict with:
        ``red``/``green``/``blue``/``luma``: length-256 int64 bin counts.
        ``total``: total pixel count.
        ``shadow_clip``/``highlight_clip``: per-key fractions (0..1) of pixels at
            pure black (bin 0) and pure white (bin 255) for each channel + luma.
    """
    arr = _to_rgb_uint8(image)
    total = int(arr.shape[0] * arr.shape[1])

    red = _counts(arr[:, :, 0])
    green = _counts(arr[:, :, 1])
    blue = _counts(arr[:, :, 2])
    luma_channel = (
        arr[:, :, 0] * _LUMA_WEIGHTS[0]
        + arr[:, :, 1] * _LUMA_WEIGHTS[1]
        + arr[:, :, 2] * _LUMA_WEIGHTS[2]
    ).round().clip(0, 255).astype(np.uint8)
    luma = _counts(luma_channel)

    denom = float(total) if total else 1.0
    channels = {"red": red, "green": green, "blue": blue, "luma": luma}
    shadow_clip = {name: float(counts[0]) / denom for name, counts in channels.items()}
    highlight_clip = {name: float(counts[LEVELS - 1]) / denom for name, counts in channels.items()}

    return {
        "levels": LEVELS,
        "red": red,
        "green": green,
        "blue": blue,
        "luma": luma,
        "total": total,
        "shadow_clip": shadow_clip,
        "highlight_clip": highlight_clip,
    }
