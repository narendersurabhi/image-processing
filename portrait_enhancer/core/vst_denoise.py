"""Noise-model-aware luminance denoising via a variance-stabilizing transform (VST).

Real sensor noise on *linear* (scene-referred) values is signal-dependent: a Poisson-Gaussian
mix whose variance grows with brightness, ``var(x) = a*x + b`` (``a`` = shot/gain term, ``b`` =
read-noise variance). A fixed-strength smoother is therefore wrong everywhere at once -- it
over-smooths bright tones and under-smooths the shadows, exactly where photographic noise is
worst. The classical fix (Anscombe; generalized form Mäkitalo & Foi 2011) is to apply a
*generalized Anscombe transform* that maps the signal-dependent noise to ~unit, constant
variance, denoise there with a single uniform strength, then invert the transform.

This module realizes the "denoise in the linear/raw domain" direction of the RAW strategy
(docs/RAW_PROCESSING_STRATEGY.md §5.1/§5.2) for the scene-linear pipeline (Phase 3). It only
touches luminance -- chroma noise is blotchy, not shot-noise-shaped, and is handled in the
perceptual LAB path by the caller.

Self-contained (numpy + cv2 + the low-level ``bilateral_filter``) so it can't create an import
cycle with ``processing`` and can be unit tested in isolation.
"""

from __future__ import annotations

import math

import numpy as np

from .utils import bilateral_filter, clamp01

# Rec.709 luma weights (the working space is linear-sRGB / Rec.709 primaries).
_REC709 = (0.2126, 0.7152, 0.0722)

# Immerkjær (1996) Laplacian-of-Laplacian noise kernel -- its response cancels on smooth real
# edges, so the mean absolute response is a robust proxy for noise level. Same estimator the
# rest of the app uses, here run directly on the linear luminance.
_NOISE_KERNEL = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)

_EPS = 1e-6


def linear_luma(img: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of a linear HxWx3 image, as float32 HxW."""
    a = np.asarray(img, dtype=np.float32)
    return (a[:, :, 0] * _REC709[0] + a[:, :, 1] * _REC709[1] + a[:, :, 2] * _REC709[2]).astype(np.float32)


def estimate_linear_noise_sigma(luma: np.ndarray) -> float:
    """Immerkjær noise-std estimate on a linear luminance image, in linear [0,1] units."""
    import cv2

    lum = np.asarray(luma, dtype=np.float32)
    h, w = lum.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    conv = cv2.filter2D(lum, -1, _NOISE_KERNEL, borderType=cv2.BORDER_REFLECT)
    return float(np.sum(np.abs(conv)) * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2)))


def estimate_noise_params(luma: np.ndarray, read_fraction: float = 0.25) -> tuple[float, float]:
    """Estimate a shot-noise-dominant Poisson-Gaussian model ``var(x) = a*x + b`` for a linear
    luminance image, anchored to the measured global noise std at the image's mean brightness.

    Returns ``(a, b)``. ``(0.0, 0.0)`` means no measurable noise (skip denoising). ``read_fraction``
    splits a small constant read-noise floor off the measured variance; the rest is shot noise.
    """
    sigma = estimate_linear_noise_sigma(luma)
    if sigma <= 1e-4:
        return 0.0, 0.0
    var = sigma * sigma
    mu = float(np.clip(np.mean(luma), 1e-3, 1.0))
    b = (float(read_fraction) * sigma) ** 2  # constant read-noise variance floor
    a = max((var - b) / mu, 0.0)  # shot term: var(mu) = a*mu + b == measured var
    return a, b


def generalized_anscombe(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Forward generalized Anscombe transform. Maps ``var = a*x + b`` noise to ~unit variance."""
    a = max(float(a), _EPS)
    arg = np.maximum(a * np.asarray(x, dtype=np.float32) + 0.375 * a * a + b, 0.0)
    return ((2.0 / a) * np.sqrt(arg)).astype(np.float32)


def inverse_generalized_anscombe(y: np.ndarray, a: float, b: float) -> np.ndarray:
    """Algebraic inverse of :func:`generalized_anscombe` (exact inverse of the forward map)."""
    a = max(float(a), _EPS)
    y = np.asarray(y, dtype=np.float32)
    return ((a * y * y) / 4.0 - 0.375 * a - b / a).astype(np.float32)


def denoise_luma_linear(img_linear: np.ndarray, amount: float, acceleration: str = "auto") -> np.ndarray:
    """Denoise the luminance of a linear HxWx3 image in the variance-stabilized domain, leaving
    chroma untouched (applied as a luminance gain to preserve hue/saturation).

    ``amount`` in [0,1] scales the smoothing strength. Returns the image unchanged when no noise
    is measurable, so it's a safe no-op on clean images.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img_linear
    img = np.asarray(img_linear, dtype=np.float32)
    luma = linear_luma(img)
    a, b = estimate_noise_params(luma)
    if a <= 0.0 and b <= 0.0:
        return img

    # Stabilize -> noise is ~constant variance, so one uniform bilateral strength is correct
    # across shadows-to-highlights. sigma_color is in stabilized units where the noise std ~= 1.
    stabilized = generalized_anscombe(luma, a, b)
    sigma_color = 1.5 + amount * 3.0
    sigma_space = 1.0 + amount * 2.0
    smoothed = bilateral_filter(stabilized, sigma_color=sigma_color, sigma_space=sigma_space, acceleration=acceleration)
    luma_den = clamp01(inverse_generalized_anscombe(smoothed, a, b))

    gain = luma_den / np.maximum(luma, 1e-5)
    out = clamp01(img * gain[:, :, np.newaxis])
    return out.astype(np.float32)
