"""Non-destructive mask adjustment helpers."""

from __future__ import annotations

import cv2
import numpy as np

from portrait_enhancer.config import MASK_ORDER
from .utils import smooth_mask

MASK_ADJUSTMENT_LIMITS = {
    "strength": (0.0, 200.0, 100.0),
    "feather": (0.0, 40.0, 0.0),
    "expand": (-40.0, 40.0, 0.0),
}


def default_mask_adjustments(layers=MASK_ORDER) -> dict[str, dict[str, float]]:
    """Return default non-destructive mask settings for each selective layer."""
    return {
        layer: {key: default for key, (_mn, _mx, default) in MASK_ADJUSTMENT_LIMITS.items()}
        for layer in layers
    }


def normalize_mask_adjustments(payload, layers=MASK_ORDER) -> dict[str, dict[str, float]]:
    """Merge stored mask settings with defaults and clamp them to supported ranges."""
    defaults = default_mask_adjustments(layers)
    if not isinstance(payload, dict):
        return defaults

    normalized = {}
    for layer in layers:
        source = payload.get(layer, {})
        if not isinstance(source, dict):
            source = {}
        normalized[layer] = {}
        for key, (mn, mx, default) in MASK_ADJUSTMENT_LIMITS.items():
            try:
                value = float(source.get(key, default))
            except (TypeError, ValueError):
                value = default
            normalized[layer][key] = float(np.clip(value, mn, mx))
    return normalized


def adjust_mask(mask: np.ndarray, settings: dict | None, *, acceleration: str = "auto") -> np.ndarray:
    """Apply strength, feather, and expand/contract to one soft mask."""
    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 2 or arr.size == 0:
        return arr.astype(np.float32)

    cfg = normalize_mask_adjustments({"layer": settings or {}}, layers=("layer",))["layer"]
    expand = cfg["expand"]
    if abs(expand) >= 0.5:
        radius = _scaled_radius(abs(expand), arr.shape)
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            if expand > 0:
                arr = cv2.dilate(arr, kernel, iterations=1)
            else:
                arr = cv2.erode(arr, kernel, iterations=1)

    feather = cfg["feather"]
    if feather >= 0.5:
        sigma = _scaled_sigma(feather, arr.shape)
        if sigma > 0.0:
            arr = smooth_mask(arr, sigma=sigma, acceleration=acceleration)

    strength = cfg["strength"] / 100.0
    arr = np.clip(arr * strength, 0.0, 1.0)
    return arr.astype(np.float32)


def apply_mask_adjustments(
    masks: dict[str, np.ndarray] | None,
    adjustments: dict | None,
    *,
    acceleration: str = "auto",
) -> dict[str, np.ndarray] | None:
    """Return adjusted copies of masks without mutating the source masks."""
    if masks is None:
        return None

    normalized = normalize_mask_adjustments(adjustments)
    adjusted = {}
    for key, mask in masks.items():
        if key in normalized:
            adjusted[key] = adjust_mask(mask, normalized[key], acceleration=acceleration)
        else:
            adjusted[key] = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0).copy()
    return adjusted


def _scaled_radius(value: float, shape_hw: tuple[int, int]) -> int:
    h, w = shape_hw
    reference = max(1.0, float(min(h, w, 1800)))
    return int(np.clip(round(float(value) * reference / 1000.0), 1, 96))


def _scaled_sigma(value: float, shape_hw: tuple[int, int]) -> float:
    h, w = shape_hw
    reference = max(1.0, float(min(h, w, 1800)))
    return float(np.clip(float(value) * reference / 1000.0, 0.35, 72.0))
