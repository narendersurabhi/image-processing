"""Interactive tone curve: control points -> smooth LUT applied to the image.

A curve is a list of ``(x, y)`` control points in [0, 1] (input -> output),
endpoints anchored at x=0 and x=1. The points are interpolated with a
monotone cubic (Fritsch-Carlson) spline so the curve is smooth and never
overshoots, then sampled into a 256-entry LUT.

GUI-free so the curve math and the interactive point editing can be unit tested
without Qt.
"""

from __future__ import annotations

import numpy as np

from .utils import clamp01

LUT_SIZE = 256
DEFAULT_CURVE = [(0.0, 0.0), (1.0, 1.0)]
_EPS = 1e-6
_LUMA = (0.2126, 0.7152, 0.0722)


def default_curve() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (1.0, 1.0)]


def normalize_curve(points) -> list[tuple[float, float]]:
    """Coerce input into a valid, sorted curve with anchored endpoints."""
    try:
        pts = [(float(x), float(y)) for x, y in points]
    except (TypeError, ValueError):
        return default_curve()
    pts = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in pts]
    pts.sort(key=lambda p: p[0])
    if len(pts) < 2:
        return default_curve()
    # Anchor endpoints at x=0 and x=1 (their y may move: black/white point).
    pts[0] = (0.0, pts[0][1])
    pts[-1] = (1.0, pts[-1][1])
    # Drop interior points that collide in x with a neighbor.
    out = [pts[0]]
    for x, y in pts[1:-1]:
        if x - out[-1][0] > 1e-3 and 1.0 - x > 1e-3:
            out.append((x, y))
    out.append(pts[-1])
    return out


def is_identity(points) -> bool:
    pts = normalize_curve(points)
    return pts == [(0.0, 0.0), (1.0, 1.0)]


def _pchip(xs: np.ndarray, ys: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    n = len(xs)
    if n == 1:
        return np.full_like(x_eval, ys[0])
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.empty(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    idx = np.clip(np.searchsorted(xs, x_eval, side="right") - 1, 0, n - 2)
    out = np.empty_like(x_eval)
    for k, i in enumerate(idx):
        t = (x_eval[k] - xs[i]) / h[i]
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        out[k] = h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]
    return out


def curve_to_lut(points, size: int = LUT_SIZE) -> np.ndarray:
    pts = normalize_curve(points)
    xs = np.asarray([p[0] for p in pts], dtype=np.float64)
    ys = np.asarray([p[1] for p in pts], dtype=np.float64)
    x_eval = np.linspace(0.0, 1.0, size)
    return np.clip(_pchip(xs, ys, x_eval), 0.0, 1.0).astype(np.float32)


def apply_curve(img: np.ndarray, points) -> np.ndarray:
    """Apply the tone curve to ``img`` (float 0..1), preserving chroma.

    Mirrors the band tone-curve handling: map luminance through the curve, scale
    RGB by the resulting gain, and blend a little direct per-channel mapping so the
    curve stays responsive while keeping chroma shifts restrained.
    """
    if is_identity(points):
        return img
    lut = curve_to_lut(points)
    img = clamp01(img.astype(np.float32))
    luma = np.clip(
        img[:, :, 0] * _LUMA[0] + img[:, :, 1] * _LUMA[1] + img[:, :, 2] * _LUMA[2],
        0.0,
        1.0,
    ).astype(np.float32)
    mapped = lut[(luma * 255).astype(np.uint8)]
    gain = mapped / np.maximum(luma, 1e-4)
    remapped = img * gain[:, :, np.newaxis]
    direct = lut[(img * 255).astype(np.uint8)]
    return clamp01(remapped * 0.85 + direct * 0.15)


# --- Interactive point editing (normalized space) -----------------------------

def nearest_point(points, x: float, y: float, tol: float) -> int | None:
    best, best_dist = None, tol
    for i, (px, py) in enumerate(points):
        dist = max(abs(px - x), abs(py - y))  # square hit area
        if dist <= best_dist:
            best, best_dist = i, dist
    return best


def add_point(points, x: float, y: float) -> tuple[list, int]:
    pts = [tuple(p) for p in points]
    x = min(1.0, max(0.0, float(x)))
    y = min(1.0, max(0.0, float(y)))
    insert = len(pts)
    for i, (px, _py) in enumerate(pts):
        if x < px:
            insert = i
            break
    pts.insert(insert, (x, y))
    return normalize_curve(pts), insert


def move_point(points, index: int, x: float, y: float) -> list:
    pts = [list(p) for p in points]
    if index < 0 or index >= len(pts):
        return [tuple(p) for p in pts]
    y = min(1.0, max(0.0, float(y)))
    if index == 0:
        pts[0] = [0.0, y]  # endpoints move only in y
    elif index == len(pts) - 1:
        pts[-1] = [1.0, y]
    else:
        lo = pts[index - 1][0] + 1e-3
        hi = pts[index + 1][0] - 1e-3
        pts[index] = [min(hi, max(lo, float(x))), y]
    return [tuple(p) for p in pts]


def remove_point(points, index: int) -> list:
    pts = [tuple(p) for p in points]
    if index <= 0 or index >= len(pts) - 1:
        return pts  # cannot remove the anchored endpoints
    del pts[index]
    return pts
