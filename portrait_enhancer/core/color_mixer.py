"""HSL color mixer: per-hue-band hue / saturation / luminance adjustment.

Eight color bands (red…magenta) each carry hue, saturation, and luminance
parameters in [-100, 100]. Pixels are weighted into bands by a smooth raised-cosine
window over hue (bands overlap so adjustments blend), and gated by saturation so
near-gray pixels are left alone.

GUI-free so the band math can be unit tested without Qt.
"""

from __future__ import annotations

import numpy as np
import cv2

from .utils import display_to_working, to_float, to_uint8, working_to_display

# Band name -> center hue in degrees.
BANDS = (
    ("red", 0.0),
    ("orange", 30.0),
    ("yellow", 60.0),
    ("green", 120.0),
    ("aqua", 180.0),
    ("blue", 240.0),
    ("purple", 270.0),
    ("magenta", 300.0),
)
BAND_NAMES = tuple(name for name, _ in BANDS)
PARAMS = ("hue", "sat", "lum")

_FALLOFF = 45.0   # degrees of influence on each side of a band center
_HUE_SCALE = 0.30  # param 100 -> up to ±30° hue shift
_SAT_SCALE = 1.0   # param 100 -> up to ±100% saturation
_LUM_SCALE = 0.5   # param 100 -> up to ±50% value
_SAT_GATE = 0.15   # below this saturation a pixel has no meaningful hue


def default_color_mixer() -> dict:
    return {name: {p: 0 for p in PARAMS} for name in BAND_NAMES}


def normalize_color_mixer(data) -> dict:
    data = data or {}
    out = {}
    for name in BAND_NAMES:
        band = data.get(name, {}) if isinstance(data, dict) else {}
        values = {}
        for p in PARAMS:
            try:
                values[p] = int(max(-100, min(100, float(band.get(p, 0)))))
            except (TypeError, ValueError):
                values[p] = 0
        out[name] = values
    return out


def is_identity(data) -> bool:
    norm = normalize_color_mixer(data)
    return all(v == 0 for band in norm.values() for v in band.values())


def _hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(hue - center)
    return np.minimum(d, 360.0 - d)


def _band_weight(hue: np.ndarray, center: float) -> np.ndarray:
    d = _hue_distance(hue, center)
    inside = d < _FALLOFF
    w = np.zeros_like(hue, dtype=np.float32)
    w[inside] = 0.5 * (1.0 + np.cos(np.pi * d[inside] / _FALLOFF))
    return w


def apply_color_mixer(img: np.ndarray, adjustments, working_space: str = "srgb") -> np.ndarray:
    """Apply per-band HSL adjustments to a working-space float image."""
    norm = normalize_color_mixer(adjustments)
    if is_identity(norm):
        return img

    display = working_to_display(img, output_transform="srgb", working_space=working_space)
    hsv = cv2.cvtColor(to_uint8(display), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * 2.0          # OpenCV hue 0..179 -> degrees
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    sat_gate = np.clip(sat / _SAT_GATE, 0.0, 1.0)

    hue_shift = np.zeros_like(hue, dtype=np.float32)
    sat_delta = np.zeros_like(hue, dtype=np.float32)
    lum_delta = np.zeros_like(hue, dtype=np.float32)

    for name, center in BANDS:
        band = norm[name]
        hu, sa, lu = band["hue"], band["sat"], band["lum"]
        if hu == 0 and sa == 0 and lu == 0:
            continue
        weight = _band_weight(hue, center) * sat_gate
        if hu:
            hue_shift += weight * hu * _HUE_SCALE
        if sa:
            sat_delta += weight * (sa / 100.0)
        if lu:
            lum_delta += weight * (lu / 100.0)

    hue_out = (hue + hue_shift) % 360.0
    sat_out = np.clip(sat * (1.0 + sat_delta * _SAT_SCALE), 0.0, 1.0)
    val_out = np.clip(val * (1.0 + lum_delta * _LUM_SCALE), 0.0, 1.0)

    hsv_out = np.empty_like(hsv)
    hsv_out[:, :, 0] = np.clip(hue_out / 2.0, 0, 179)
    hsv_out[:, :, 1] = sat_out * 255.0
    hsv_out[:, :, 2] = val_out * 255.0
    out = to_float(cv2.cvtColor(hsv_out.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return display_to_working(out, working_space=working_space)
