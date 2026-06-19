"""White balance correction via per-channel gains in linear light.

True white balance is a diagonal (per-channel) scaling applied in **linear**
light, not a hue rotation. The existing Temperature/Tint controls are a stylistic
grade; this module neutralizes a color cast so a should-be-neutral pixel reads
gray.

GUI-free so the gain math and estimators can be unit tested without Qt.

Note on color space: gains are ratios in true linear light. The interactive
estimators receive **display/source** pixels (sRGB-encoded 0..1), so the UI passes
``working_space="srgb"`` when sampling. :func:`apply_white_balance_gains` is called
inside the render pipeline where the image is already in the working space, so it
passes the actual working space — both resolve to the same true-linear space.
"""

from __future__ import annotations

import math

import numpy as np

from .utils import clamp01, linear_to_srgb, srgb_to_linear

DEFAULT_GAINS = (1.0, 1.0, 1.0)
MIN_GAIN = 0.25
MAX_GAIN = 4.0
_EPS = 1e-4

# Temperature/Tint canonical model. Identity is anchored at the sRGB white point
# (D65 ~6500K) and tint 0, so a default image is unchanged.
NEUTRAL_K = 6500
MIN_K = 2000
MAX_K = 12000
TINT_MIN = -150
TINT_MAX = 150
_TINT_STRENGTH = 0.5  # green multiplier reaches 1 ± strength at |tint| = TINT_MAX

# Illuminant presets: name -> (kelvin, tint). Corrective values relative to D65.
PRESETS = (
    ("Tungsten", 3200, 0),
    ("Fluorescent", 4000, 18),
    ("Daylight", 5500, 0),
    ("Flash", 5500, 0),
    ("Cloudy", 6500, 0),
    ("Shade", 7500, 0),
)


def _to_linear(img: np.ndarray, working_space: str) -> np.ndarray:
    if working_space == "linear":
        return clamp01(img.astype(np.float32))
    return srgb_to_linear(img)


def _from_linear(linear: np.ndarray, working_space: str) -> np.ndarray:
    if working_space == "linear":
        return clamp01(linear)
    return linear_to_srgb(linear)


def normalize_gains(gains) -> list[float]:
    try:
        values = [float(v) for v in gains]
    except (TypeError, ValueError):
        return list(DEFAULT_GAINS)
    if len(values) != 3:
        return list(DEFAULT_GAINS)
    return [min(MAX_GAIN, max(MIN_GAIN, v)) for v in values]


def is_identity(gains) -> bool:
    return all(abs(v - 1.0) < 1e-4 for v in normalize_gains(gains))


def clamp_temp(temp_k) -> int:
    try:
        return int(round(max(MIN_K, min(MAX_K, float(temp_k)))))
    except (TypeError, ValueError):
        return NEUTRAL_K


def clamp_tint(tint) -> int:
    try:
        return int(round(max(TINT_MIN, min(TINT_MAX, float(tint)))))
    except (TypeError, ValueError):
        return 0


def _cct_to_srgb(temp_k: float) -> tuple[float, float, float]:
    """Tanner Helland's blackbody color approximation (sRGB 0..1)."""
    t = max(1000.0, min(40000.0, float(temp_k))) / 100.0
    if t <= 66:
        red = 255.0
    else:
        red = 329.698727446 * (t - 60) ** -0.1332047592
    if t <= 66:
        green = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        green = 288.1221695283 * (t - 60) ** -0.0755148492
    if t >= 66:
        blue = 255.0
    elif t <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(t - 10) - 305.0447927307
    return (
        min(255.0, max(0.0, red)) / 255.0,
        min(255.0, max(0.0, green)) / 255.0,
        min(255.0, max(0.0, blue)) / 255.0,
    )


def _cct_to_linear(temp_k: float) -> np.ndarray:
    srgb = np.asarray(_cct_to_srgb(temp_k), dtype=np.float32).reshape(1, 1, 3)
    return srgb_to_linear(srgb)[0, 0]


_REF_LINEAR = _cct_to_linear(NEUTRAL_K)


def _temp_gain_rb(temp_k: float) -> tuple[float, float]:
    """Red/blue gains (green normalized to 1) for a given temperature.

    gains = illuminant(D65) / illuminant(T), so a higher T (bluer light) produces
    a warmer correction (red up, blue down) — slider-right = warmer image.
    """
    illum = np.maximum(_cct_to_linear(temp_k), _EPS)
    gain = _REF_LINEAR / illum
    gain = gain / gain[1]
    return float(gain[0]), float(gain[2])


def _tint_green_mult(tint: float) -> float:
    tint = max(TINT_MIN, min(TINT_MAX, float(tint)))
    return 1.0 - (tint / TINT_MAX) * _TINT_STRENGTH


def kelvin_tint_to_gains(temp_k, tint=0.0) -> list[float]:
    """Per-channel gains for a (Kelvin, tint) white-balance setting."""
    gr, gb = _temp_gain_rb(clamp_temp(temp_k))
    return normalize_gains([gr, _tint_green_mult(clamp_tint(tint)), gb])


def gains_to_kelvin_tint(gains) -> tuple[int, int]:
    """Invert neutralizing gains to the nearest (Kelvin, tint) on the model.

    Temperature comes from the red/blue ratio (monotonic in T); tint captures the
    residual green relative to the red/blue level. Keeps the eyedropper and the
    Temp/Tint sliders in sync.
    """
    g = normalize_gains(gains)
    target_rb = g[0] / max(g[2], _EPS)
    lo, hi = float(MIN_K), float(MAX_K)
    for _ in range(40):  # bisection: gr/gb increases monotonically with T
        mid = (lo + hi) / 2.0
        gr, gb = _temp_gain_rb(mid)
        if gr / max(gb, _EPS) < target_rb:
            lo = mid
        else:
            hi = mid
    temp_k = (lo + hi) / 2.0
    gr, gb = _temp_gain_rb(temp_k)
    scale = math.sqrt(max(_EPS, (gr / max(g[0], _EPS)) * (gb / max(g[2], _EPS))))
    m_target = scale * g[1]
    tint = (1.0 - m_target) / _TINT_STRENGTH * TINT_MAX
    return clamp_temp(temp_k), clamp_tint(tint)


def neutral_sample_to_kelvin_tint(sample_rgb, working_space: str = "srgb") -> tuple[int, int]:
    return gains_to_kelvin_tint(gains_from_neutral_sample(sample_rgb, working_space))


def apply_white_balance_gains(img: np.ndarray, gains, working_space: str = "srgb") -> np.ndarray:
    """Multiply each channel by its gain in linear light, returning working space."""
    g = normalize_gains(gains)
    if is_identity(g):
        return img
    linear = _to_linear(img, working_space).astype(np.float32).copy()
    linear[:, :, 0] *= g[0]
    linear[:, :, 1] *= g[1]
    linear[:, :, 2] *= g[2]
    linear = clamp01(linear)
    return _from_linear(linear, working_space).astype(np.float32)


def _gains_from_linear_rgb(r: float, g: float, b: float) -> list[float]:
    r = max(float(r), _EPS)
    g = max(float(g), _EPS)
    b = max(float(b), _EPS)
    target = (r + g + b) / 3.0  # brightness-preserving neutral target
    return normalize_gains([target / r, target / g, target / b])


def gains_from_neutral_sample(sample_rgb, working_space: str = "srgb") -> list[float]:
    """Gains that neutralize a should-be-neutral sample (working-space RGB 0..1)."""
    arr = np.asarray(sample_rgb, dtype=np.float32).reshape(1, 1, 3)
    lin = _to_linear(arr, working_space)[0, 0]
    return _gains_from_linear_rgb(lin[0], lin[1], lin[2])


def gray_world_gains(image: np.ndarray, working_space: str = "srgb") -> list[float]:
    """Assume the scene averages to neutral; equalize the per-channel means."""
    lin = _to_linear(np.asarray(image, dtype=np.float32), working_space)
    means = lin.reshape(-1, 3).mean(axis=0)
    return _gains_from_linear_rgb(means[0], means[1], means[2])


def white_patch_gains(image: np.ndarray, working_space: str = "srgb", percentile: float = 99.0) -> list[float]:
    """Anchor to the brightest near-neutral region (per-channel high percentile)."""
    lin = _to_linear(np.asarray(image, dtype=np.float32), working_space)
    ref = np.percentile(lin.reshape(-1, 3), percentile, axis=0)
    ref = np.maximum(ref, _EPS)
    target = float(ref.max())
    return normalize_gains([target / ref[0], target / ref[1], target / ref[2]])
