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
import os
from pathlib import Path

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


class WhiteBalanceEstimator:
    """Optional learned auto white balance: an ONNX model that predicts the scene illuminant,
    converted into neutralizing per-channel gains via the same math as the classical
    estimators. Unlike Gray World it isn't fooled by a dominant color (a red wall, a green
    lawn), because it reasons about scene content. Falls back silently (available=False) when
    the model file or onnxruntime is missing, so Gray World / White Patch keep working.

    Model I/O contract (FC4 / illuminant-estimation family): one image input, NCHW or NHWC,
    RGB in [0, 1]; one output that reduces to a 3-vector illuminant (RGB). The illuminant is
    treated as linear RGB; set PORTRAIT_WB_LINEARIZE_INPUT=1 to feed the model a linearized
    image if your model was trained on linear input.
    """

    def __init__(self, ref_size: int | None = None):
        self.available = False
        self.backend = "none"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._session = None
        self._input_name = None
        self._layout = "nchw"
        self._linearize_input = os.getenv("PORTRAIT_WB_LINEARIZE_INPUT", "") not in ("", "0", "false", "False")
        try:
            self.ref_size = int(ref_size if ref_size is not None else os.getenv("PORTRAIT_WB_REF_SIZE", "512"))
        except (TypeError, ValueError):
            self.ref_size = 512
        self.ref_size = max(64, self.ref_size)

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "white balance model not found"
            return

        providers = self._preferred_onnx_providers(ort)
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            try:
                self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                self.reason_unavailable = f"coreml unavailable; using cpu ({exc})"
            except Exception as cpu_exc:
                self.reason_unavailable = f"white balance init failed: {cpu_exc}"
                return
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._layout = self._infer_layout(inp.shape)
        if "CoreMLExecutionProvider" in set(self._session.get_providers()):
            self.execution_provider = "coreml"
        self.available = True
        self.backend = "onnx"

    def _infer(self, image: np.ndarray, working_space: str):
        """Run the model on a downscaled copy. Returns (small_srgb_HWC, raw_output) or None.
        small_srgb is always in the image's display space so image-to-image models can be
        compared input-vs-output regardless of what we feed the net."""
        if not self.available or self._session is None:
            return None
        arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None
        small_srgb = _resize_rgb(arr, self.ref_size)
        model_small = _to_linear(small_srgb, working_space) if self._linearize_input else small_srgb
        if self._layout == "nhwc":
            tensor = model_small[None, ...].astype(np.float32)
        else:
            tensor = np.transpose(model_small, (2, 0, 1))[None, ...].astype(np.float32)
        try:
            out = self._session.run(None, {self._input_name: tensor})[0]
        except Exception:
            return None
        return small_srgb, out

    def estimate_illuminant(self, image: np.ndarray, working_space: str = "srgb"):
        """Return the predicted scene illuminant as a 3-vector (illuminant-estimation models),
        or None. Image-to-image AWB models don't expose an illuminant -- use estimate_gains."""
        res = self._infer(image, working_space)
        if res is None:
            return None
        return self._to_illuminant(res[1])

    def estimate_gains(self, image: np.ndarray, working_space: str = "srgb"):
        """Neutralizing per-channel gains, supporting both model families:

        - illuminant estimation (output reduces to a 3-vector): gains neutralize the illuminant.
        - image-to-image AWB (e.g. Deep-WB; output is a corrected image): the diagonal gains
          that reproduce the model's average input->output color shift, computed in linear light.
        """
        res = self._infer(image, working_space)
        if res is None:
            return None
        small_srgb, out = res
        img_out = self._output_as_image(out, small_srgb.shape[:2])
        if img_out is not None:
            in_lin = srgb_to_linear(small_srgb).reshape(-1, 3).mean(axis=0)
            out_lin = srgb_to_linear(np.clip(img_out, 0.0, 1.0)).reshape(-1, 3).mean(axis=0)
            gains = [float(out_lin[i]) / max(float(in_lin[i]), _EPS) for i in range(3)]
            return normalize_gains(gains)
        illum = self._to_illuminant(out)
        if illum is None:
            return None
        return _gains_from_linear_rgb(illum[0], illum[1], illum[2])

    @staticmethod
    def _output_as_image(out, hw):
        """If the model output is a 3-channel image matching the input's spatial size, return
        it as HxWx3 (image-to-image AWB); otherwise None (illuminant-vector model)."""
        a = np.squeeze(np.asarray(out, dtype=np.float32))
        hw = (int(hw[0]), int(hw[1]))
        if a.ndim == 3:
            if a.shape[0] == 3 and a.shape[1:] == hw:
                return np.transpose(a, (1, 2, 0))
            if a.shape[2] == 3 and a.shape[:2] == hw:
                return a
        return None

    @staticmethod
    def _to_illuminant(out):
        a = np.squeeze(np.asarray(out, dtype=np.float32))
        if a.ndim == 1 and a.size == 3:
            v = a
        elif a.ndim >= 2 and 3 in a.shape:
            # Spatial illuminant/confidence map: average everything but the 3-channel axis.
            ax = next((i for i, n in enumerate(a.shape) if n == 3), None)
            if ax is None:
                return None
            v = a.mean(axis=tuple(i for i in range(a.ndim) if i != ax))
        elif a.size == 3:
            v = a.reshape(3)
        else:
            return None
        v = np.abs(np.asarray(v, dtype=np.float32))
        if v.size != 3 or not np.isfinite(v).all() or float(v.sum()) <= 0:
            return None
        return [float(v[0]), float(v[1]), float(v[2])]

    @staticmethod
    def _infer_layout(shape) -> str:
        try:
            if len(shape) == 4:
                if shape[1] == 3:
                    return "nchw"
                if shape[3] == 3:
                    return "nhwc"
        except Exception:
            pass
        return "nchw"

    def _resolve_model_path(self):
        candidates = []
        env_path = os.getenv("PORTRAIT_WB_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        for name in ("white_balance.onnx", "awb.onnx", "fc4.onnx"):
            candidates.append(models_dir / name)
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _preferred_onnx_providers(self, ort):
        available = set(ort.get_available_providers())
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers


def _resize_rgb(arr: np.ndarray, ref_size: int) -> np.ndarray:
    """Square-resize an HxWx3 float image to ref_size, without a hard cv2 import at module load."""
    import cv2

    return cv2.resize(arr, (int(ref_size), int(ref_size)), interpolation=cv2.INTER_AREA)
