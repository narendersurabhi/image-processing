"""Optional generative face refinement backends."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .utils import clamp01, smooth_mask, to_uint8


def _infer_input_hw(shape) -> tuple[int, int]:
    dims = list(shape or [])
    if len(dims) >= 4:
        h = dims[-2]
        w = dims[-1]
    elif len(dims) == 3:
        h = dims[-2]
        w = dims[-1]
    else:
        return 512, 512
    if not isinstance(h, int) or h <= 0:
        h = 512
    if not isinstance(w, int) or w <= 0:
        w = 512
    return int(h), int(w)


def _mask_bbox(mask: np.ndarray, threshold: float = 0.08) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(np.clip(mask.astype(np.float32), 0.0, 1.0) > float(threshold))
    if coords.size == 0:
        return None
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1
    return int(x1), int(y1), int(x2), int(y2)


def _expand_box(box: tuple[int, int, int, int], shape_hw: tuple[int, int], pad_x: float = 0.32, pad_y: float = 0.38) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h, w = shape_hw
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    px = bw * pad_x
    py = bh * pad_y
    nx1 = max(0, int(round(x1 - px)))
    ny1 = max(0, int(round(y1 - py)))
    nx2 = min(w, int(round(x2 + px)))
    ny2 = min(h, int(round(y2 + py)))
    return nx1, ny1, nx2, ny2


class CodeFormerONNXRefiner:
    """Optional CodeFormer-compatible ONNX face refiner."""

    def __init__(self, model_path: str | None = None):
        self._explicit_model_path = model_path
        self._session = None
        self._input_name = None
        self._input_hw = (512, 512)
        self._weight_input_name = None
        self._weight_input_dtype = np.float32
        self._initialized = False
        self.available = False
        self.backend = "unavailable"
        self.execution_provider = "none"
        self.reason_unavailable = ""

    @property
    def backend_label(self) -> str:
        if self.available:
            return f"codeformer-onnx:{self.execution_provider}"
        return "codeformer-unavailable"

    def refine(self, img: np.ndarray, face_mask: np.ndarray, strength: float, fidelity: float = 0.65) -> np.ndarray:
        amount = float(np.clip(strength, 0.0, 1.0))
        if amount <= 0.0:
            return img
        if face_mask is None:
            return img
        if not self._ensure_session():
            return img

        face_mask = np.clip(np.asarray(face_mask, dtype=np.float32), 0.0, 1.0)
        bbox = _mask_bbox(face_mask, threshold=0.08)
        if bbox is None:
            return img
        x1, y1, x2, y2 = _expand_box(bbox, img.shape[:2])
        if (x2 - x1) < 24 or (y2 - y1) < 24:
            return img

        crop = np.clip(np.asarray(img[y1:y2, x1:x2], dtype=np.float32), 0.0, 1.0)
        restored = self._run_model(crop, fidelity=fidelity)
        if restored is None:
            return img

        local_mask = face_mask[y1:y2, x1:x2]
        support = smooth_mask(local_mask, sigma=max(1.0, min(crop.shape[:2]) * 0.035))
        if float(support.max()) <= 0.05:
            support = np.ones(crop.shape[:2], dtype=np.float32)
        blend = np.clip(support * amount, 0.0, 1.0)[:, :, np.newaxis]

        out = img.copy()
        out[y1:y2, x1:x2] = clamp01(crop * (1.0 - blend) + restored * blend)
        return out

    def _ensure_session(self) -> bool:
        if self._initialized:
            return self.available
        self._initialized = True

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return False

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "codeformer onnx model not found"
            return False

        providers = self._preferred_onnx_providers(ort)
        self._prepare_coreml_cache_dir(providers)
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            if "CoreMLExecutionProvider" in providers:
                try:
                    self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    self.execution_provider = "cpu"
                except Exception as cpu_exc:
                    self.reason_unavailable = f"codeformer init failed: {cpu_exc}"
                    return False
                else:
                    self.reason_unavailable = f"coreml unavailable; using cpu ({exc})"
            else:
                self.reason_unavailable = f"codeformer init failed: {exc}"
                return False

        active = list(self._session.get_providers())
        self.execution_provider = "coreml" if "CoreMLExecutionProvider" in active else "cpu"
        self._bind_model_inputs()
        self.available = True
        self.backend = "codeformer_onnx"
        return True

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        if self._explicit_model_path:
            candidates.append(Path(self._explicit_model_path))
        env_path = os.getenv("PORTRAIT_CODEFORMER_ONNX")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / "codeformer.onnx")
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _preferred_onnx_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _prepare_coreml_cache_dir(self, providers: list[str]) -> None:
        if "CoreMLExecutionProvider" not in providers:
            return
        cache_dir = Path(__file__).resolve().parents[2] / ".ort_coreml_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_str = str(cache_dir)
        os.environ["TMPDIR"] = cache_str
        os.environ["TMP"] = cache_str
        os.environ["TEMP"] = cache_str
        tempfile.tempdir = cache_str

    def _run_model(self, crop: np.ndarray, fidelity: float = 0.65) -> np.ndarray | None:
        if self._session is None or self._input_name is None:
            return None

        in_h, in_w = self._input_hw
        resized = cv2.resize(to_uint8(crop), (in_w, in_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        tensor = np.transpose(resized, (2, 0, 1))[None, ...]
        tensor = (tensor - 0.5) / 0.5
        feed = {self._input_name: tensor}
        if self._weight_input_name is not None:
            # CodeFormer exports often expose a scalar fidelity input `w`.
            # Lower values bias toward stronger restoration; higher values keep
            # the result closer to the original face patch.
            feed[self._weight_input_name] = np.array(float(np.clip(fidelity, 0.0, 1.0)), dtype=self._weight_input_dtype)

        try:
            output = self._session.run(None, feed)[0]
        except Exception as exc:
            self.reason_unavailable = f"codeformer inference failed: {exc}"
            self.available = False
            return None

        restored = self._output_to_image(output)
        if restored is None:
            return None
        return clamp01(cv2.resize(restored, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC))

    def _output_to_image(self, output: np.ndarray) -> np.ndarray | None:
        arr = np.asarray(output)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        arr = arr.astype(np.float32)
        if arr.min() < -0.1:
            arr = (arr + 1.0) * 0.5
        elif arr.max() > 1.5:
            arr = arr / 255.0
        return clamp01(arr)

    def _bind_model_inputs(self) -> None:
        self._weight_input_name = None
        self._weight_input_dtype = np.float32
        for input_meta in self._session.get_inputs():
            shape = list(input_meta.shape or [])
            if input_meta.name == "x" or (len(shape) >= 3 and 3 in shape):
                self._input_name = input_meta.name
                self._input_hw = _infer_input_hw(shape)
            elif len(shape) == 0 or shape == [1]:
                self._weight_input_name = input_meta.name
                self._weight_input_dtype = np.float64 if "double" in str(input_meta.type).lower() else np.float32
        if self._input_name is None:
            input_meta = self._session.get_inputs()[0]
            self._input_name = input_meta.name
            self._input_hw = _infer_input_hw(input_meta.shape)


_FACE_REFINER: CodeFormerONNXRefiner | None = None


def get_face_refiner() -> CodeFormerONNXRefiner:
    global _FACE_REFINER
    if _FACE_REFINER is None:
        _FACE_REFINER = CodeFormerONNXRefiner()
    return _FACE_REFINER
