"""Optional learned aesthetic scorer for Auto Crop candidates.

The crop generator in ``processing.py`` is responsible for safety: keep subjects, faces,
and eye/headroom inside the frame. This module is only a tie-breaker among already-safe
candidate rectangles. It is intentionally optional; when no model is installed, Auto Crop
uses the deterministic scorer unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


_aesthetic_crop_scorer = None
_aesthetic_crop_scorer_init = False


def get_aesthetic_crop_scorer():
    """Return the optional ONNX aesthetic scorer, or None when unavailable."""
    global _aesthetic_crop_scorer, _aesthetic_crop_scorer_init
    if not _aesthetic_crop_scorer_init:
        try:
            scorer = OnnxAestheticCropScorer()
            _aesthetic_crop_scorer = scorer if scorer.available else None
        except Exception:
            _aesthetic_crop_scorer = None
        _aesthetic_crop_scorer_init = True
    return _aesthetic_crop_scorer


class OnnxAestheticCropScorer:
    """NIMA/MUSIQ-style ONNX scorer for a single RGB crop.

    Supported outputs:
    - scalar score in [0, 1] or [1, 10]
    - NIMA-style ordinal distribution/logits, commonly 10 bins
    """

    def __init__(self):
        self.available = False
        self.backend = "none"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._session = None
        self._input_name = None
        self._layout = "nchw"
        self._height = _env_int("PORTRAIT_AESTHETIC_CROP_SIZE", 224, 64, 1024)
        self._width = self._height
        # "minus_one" ([-1, 1]) is the default, not "imagenet": every documented NIMA backbone
        # (MobileNet, NASNet, InceptionResNetV2 -- see scripts/convert_nima_to_onnx.py) uses
        # Keras's mode="tf" preprocessing, not ImageNet mean/std.
        self._normalization = os.getenv("PORTRAIT_AESTHETIC_CROP_NORMALIZE", "minus_one").strip().lower()

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "aesthetic crop model not found"
            return

        try:
            providers = self._preferred_onnx_providers(ort)
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            active = set(self._session.get_providers())
            if "CoreMLExecutionProvider" in active:
                self.execution_provider = "coreml"
            input_meta = self._session.get_inputs()[0]
            self._input_name = input_meta.name
            self._refresh_input_contract(input_meta.shape)
            self.available = True
            self.backend = f"aesthetic-crop-onnx:{self.execution_provider}"
        except Exception as exc:
            self.reason_unavailable = f"aesthetic crop init failed: {exc}"

    def __call__(self, rgb01: np.ndarray) -> float | None:
        if not self.available or self._session is None or self._input_name is None:
            return None
        arr = np.asarray(rgb01, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        arr = arr[:, :, :3]
        if float(np.nanmax(arr)) > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        try:
            tensor = self._prepare_input(arr)
            out = self._session.run(None, {self._input_name: tensor})[0]
        except Exception:
            return None
        return aesthetic_output_to_score(out)

    def _prepare_input(self, arr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        pil = pil.resize((self._width, self._height), resampling)
        data = np.asarray(pil, dtype=np.float32) / 255.0
        if self._normalization in {"imagenet", "torch"}:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            data = (data - mean) / std
        elif self._normalization in {"minus_one", "minus-one", "-1_1", "keras"}:
            data = data * 2.0 - 1.0
        elif self._normalization in {"none", "0_1", "zero_one"}:
            pass
        else:
            # Unknown value: keep the conservative [0, 1] tensor instead of failing Auto Crop.
            pass

        if self._layout == "nhwc":
            return data[None, ...].astype(np.float32)
        return np.transpose(data, (2, 0, 1))[None, ...].astype(np.float32)

    def _refresh_input_contract(self, shape) -> None:
        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            return
        if _dim_value(shape[1]) == 3:
            self._layout = "nchw"
            height, width = _dim_value(shape[2]), _dim_value(shape[3])
        elif _dim_value(shape[3]) == 3:
            self._layout = "nhwc"
            height, width = _dim_value(shape[1]), _dim_value(shape[2])
        else:
            height, width = None, None
        if height and height > 0:
            self._height = int(height)
        if width and width > 0:
            self._width = int(width)

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_AESTHETIC_CROP_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        for name in ("aesthetic_crop.onnx", "nima.onnx", "musiq.onnx"):
            candidates.append(models_dir / name)
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _preferred_onnx_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        providers = []
        use_coreml = os.getenv("PORTRAIT_AESTHETIC_CROP_USE_COREML", "").strip().lower() in {"1", "true", "yes"}
        if use_coreml and "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers


def score_crop_aesthetic(rgb01: np.ndarray, crop: list[float], scorer) -> float | None:
    """Score a normalized crop with a callable model scorer.

    Returns a normalized score in [0, 1], or None when scoring fails. The scorer may be an
    ``OnnxAestheticCropScorer`` or any test/dummy callable that accepts an RGB crop array.
    """
    if scorer is None:
        return None
    arr = np.asarray(rgb01, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0 or len(crop) != 4:
        return None
    x, y, cw, ch = crop
    x0 = max(0, int(np.floor(float(x) * w)))
    y0 = max(0, int(np.floor(float(y) * h)))
    x1 = min(w, int(np.ceil((float(x) + float(cw)) * w)))
    y1 = min(h, int(np.ceil((float(y) + float(ch)) * h)))
    if x1 <= x0 or y1 <= y0:
        return None
    crop_img = np.clip(arr[y0:y1, x0:x1, :3], 0.0, 1.0)
    try:
        raw_score = scorer(crop_img)
    except Exception:
        return None
    return _coerce_score(raw_score)


def aesthetic_output_to_score(output) -> float | None:
    """Normalize common aesthetic-model outputs to [0, 1]."""
    arr = np.asarray(output, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if arr.size == 1:
        return _coerce_score(float(arr[0]))

    if arr.size <= 20:
        probs = arr.astype(np.float64)
        total = float(probs.sum())
        if total <= 0.0 or np.any(probs < 0.0) or abs(total - 1.0) > 0.05:
            shifted = probs - float(np.max(probs))
            exp = np.exp(shifted)
            denom = float(exp.sum())
            if denom <= 0.0:
                return None
            probs = exp / denom
        else:
            probs = probs / total
        ratings = np.arange(1, probs.size + 1, dtype=np.float64)
        mean_rating = float(np.sum(probs * ratings))
        return float(np.clip((mean_rating - 1.0) / max(1.0, probs.size - 1.0), 0.0, 1.0))

    return _coerce_score(float(np.mean(arr)))


def _coerce_score(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    if score > 1.5:
        # NIMA-style scalar means are usually 1..10.
        score = (score - 1.0) / 9.0
    return float(np.clip(score, 0.0, 1.0))


def _dim_value(value):
    return value if isinstance(value, int) else None


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return int(np.clip(value, minimum, maximum))
