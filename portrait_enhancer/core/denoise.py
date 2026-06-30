"""Optional learned image denoiser (DnCNN, ONNX).

A drop-in upgrade for the bilateral-filter noise reduction: when models/denoise.onnx is
present the noise-reduction slider routes through this CNN instead, which removes sensor
noise while preserving edges/detail better than a bilateral filter. Falls back silently
(available=False) when the model or onnxruntime is missing, so the bilateral path keeps
working unchanged.

The net is fully convolutional, so it runs on arbitrary sizes -- but a 20-layer CNN over a
full 24MP RAW would need many GB of activation memory, so inference is tiled with a halo
(the receptive field is ~41px; a 32px halo per side prevents seams). Mirrors the
CoreML->CPU provider pattern of MattingSegmenter / WhiteBalanceEstimator.
"""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


class MLDenoiser:
    def __init__(self, tile_size: int | None = None, halo: int | None = None):
        self.available = False
        self.backend = "none"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._session = None
        self._input_name = None

        try:
            self.tile_size = int(tile_size if tile_size is not None else os.getenv("PORTRAIT_DENOISE_TILE", "1024"))
        except (TypeError, ValueError):
            self.tile_size = 1024
        self.tile_size = max(128, self.tile_size)
        try:
            self.halo = int(halo if halo is not None else os.getenv("PORTRAIT_DENOISE_HALO", "32"))
        except (TypeError, ValueError):
            self.halo = 32
        self.halo = max(8, self.halo)

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "denoise model not found"
            return

        try:
            providers = self._preferred_onnx_providers(ort)
            self._prepare_coreml_cache_dir(providers)
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            if "CoreMLExecutionProvider" in set(self._session.get_providers()):
                self.execution_provider = "coreml"
            self._input_name = self._session.get_inputs()[0].name
            self.available = True
            self.backend = "dncnn"
        except Exception as exc:
            try:
                self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                self.execution_provider = "cpu"
                self.available = True
                self.backend = "dncnn"
                self.reason_unavailable = f"coreml unavailable; using cpu ({exc})"
            except Exception as cpu_exc:
                self.reason_unavailable = f"denoise init failed: {cpu_exc}"

    def denoise(self, rgb01: np.ndarray, amount: float = 1.0) -> np.ndarray | None:
        """Denoise an RGB image in [0,1]. `amount` (0..1) blends original->denoised so the
        slider controls strength. Returns None when unavailable (caller falls back)."""
        if not self.available or self._session is None:
            return None
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount <= 0.0:
            return rgb01
        arr = np.clip(np.asarray(rgb01, dtype=np.float32), 0.0, 1.0)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return None
        try:
            den = self._run_tiled(arr)
        except Exception:
            return None
        if den is None:
            return None
        return np.clip(arr * (1.0 - amount) + den * amount, 0.0, 1.0).astype(np.float32)

    def _run_tiled(self, arr: np.ndarray) -> np.ndarray | None:
        h, w = arr.shape[:2]
        ts, halo = self.tile_size, self.halo
        if max(h, w) <= ts:
            return self._infer(arr)
        out = np.empty_like(arr)
        for y0 in range(0, h, ts):
            for x0 in range(0, w, ts):
                y1, x1 = min(y0 + ts, h), min(x0 + ts, w)
                # Read tile with halo (clamped to image bounds), infer, keep inner region.
                ry0, rx0 = max(0, y0 - halo), max(0, x0 - halo)
                ry1, rx1 = min(h, y1 + halo), min(w, x1 + halo)
                patch = arr[ry0:ry1, rx0:rx1]
                den = self._infer(patch)
                if den is None:
                    return None
                out[y0:y1, x0:x1] = den[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]
        return out

    def _infer(self, patch: np.ndarray) -> np.ndarray | None:
        tensor = np.transpose(patch, (2, 0, 1))[None, ...].astype(np.float32)
        out = self._session.run(None, {self._input_name: tensor})[0]
        den = np.squeeze(np.asarray(out, dtype=np.float32))
        if den.ndim != 3:
            return None
        if den.shape[0] == 3:
            den = np.transpose(den, (1, 2, 0))
        if den.shape != patch.shape:
            return None
        return np.clip(den, 0.0, 1.0)

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_DENOISE_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        for name in ("denoise.onnx", "denoiser.onnx", "dncnn.onnx"):
            candidates.append(models_dir / name)
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
        os.environ.setdefault("TMPDIR", cache_str)
        tempfile.tempdir = cache_str


class DeepDenoiser:
    """Optional heavy real-noise denoiser (NAFNet-SIDD, ONNX) for the opt-in "Deep Denoise"
    action -- trained on real camera-noise pairs, so it cleans real sensor grain far better
    than the AWGN DnCNN, at ~5 min per 24MP image (hence a one-shot button, not a live
    slider). The exported model has a fixed 256x256 input, so inference tiles the image at
    256 with a halo; partial edge tiles are reflect-padded to 256. Same CoreML->CPU pattern.
    """

    TILE = 256

    def __init__(self, halo: int | None = None, use_coreml: bool | None = None):
        self.available = False
        self.backend = "none"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self.use_coreml = self._env_bool(
            "PORTRAIT_DEEP_DENOISE_USE_COREML",
            True,
        ) if use_coreml is None else bool(use_coreml)
        self._session = None
        self._input_name = None
        self._supports_dynamic_batch = False
        self._fixed_batch_size = 1
        self.batch_size = self._env_int("PORTRAIT_DEEP_DENOISE_BATCH", 2, 1, 16)
        self.workers = self._env_int("PORTRAIT_DEEP_DENOISE_WORKERS", 2, 1, 8)
        try:
            self.halo = int(halo if halo is not None else os.getenv("PORTRAIT_DEEP_DENOISE_HALO", "16"))
        except (TypeError, ValueError):
            self.halo = 16
        self.halo = max(0, min(self.halo, 64))

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "deep denoise model not found"
            return

        try:
            providers = self._preferred_onnx_providers(ort)
            self._prepare_coreml_cache_dir(providers)
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            if "CoreMLExecutionProvider" in set(self._session.get_providers()):
                self.execution_provider = "coreml"
            self._input_name = self._session.get_inputs()[0].name
            self._refresh_input_contract()
            self.available = True
            self.backend = "nafnet"
        except Exception as exc:
            try:
                self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                self.execution_provider = "cpu"
                self._refresh_input_contract()
                self.available = True
                self.backend = "nafnet"
                self.reason_unavailable = f"coreml unavailable; using cpu ({exc})"
            except Exception as cpu_exc:
                self.reason_unavailable = f"deep denoise init failed: {cpu_exc}"

    def denoise(self, rgb01: np.ndarray, amount: float = 1.0, detail_recovery: float = 1.0,
                progress=None) -> np.ndarray | None:
        """Denoise an RGB image in [0,1]. `progress(done, total)` is called per tile so the UI
        can show progress on the (slow) full-res pass. Tiles are independent after their halo
        is included, so the pass can use dynamic-batch ONNX inference or a small worker pool.

        `detail_recovery` (0..1) adds back the fine detail NAFNet removes at real edges -- the
        model is trained for heavy real noise and over-smooths low-noise images into a visibly
        soft/blurry result, so by default we restore edge/texture sharpness while keeping flat
        areas (skin, sky) clean. Returns None when unavailable."""
        if not self.available or self._session is None:
            return None
        amount = float(np.clip(amount, 0.0, 1.0))
        arr = np.clip(np.asarray(rgb01, dtype=np.float32), 0.0, 1.0)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return None
        if amount <= 0.0:
            return arr
        try:
            out = self._run_tiled(arr, progress=progress)
        except Exception:
            return None
        if out is None:
            return None
        if detail_recovery > 0.0:
            out = self._recover_detail(arr, out, detail_recovery)
        if amount < 1.0:
            return np.clip(arr * (1.0 - amount) + out * amount, 0.0, 1.0).astype(np.float32)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _recover_detail(original: np.ndarray, denoised: np.ndarray, strength: float) -> np.ndarray:
        """Add the removed high-frequency detail (original - denoised) back, but only at real
        edges -- the edge mask is keyed off the *denoised* (clean) image's gradients so sensor
        noise can't drive it. Flat regions stay denoised; hair/lashes/texture/edges stay sharp."""
        detail = original - denoised
        gray = cv2.cvtColor(
            (np.clip(denoised, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        # The denoised gradient is clean, but it cannot see fine strands/texture NAFNet has
        # already erased. Add a guarded original-gradient path so weak real detail can come back
        # without opening flat shadows/skin to full noise reintroduction.
        orig_gray = cv2.cvtColor(
            (np.clip(original, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY
        ).astype(np.float32) / 255.0
        ogx = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=3)
        ogy = cv2.Sobel(orig_gray, cv2.CV_32F, 0, 1, ksize=3)
        orig_grad = np.sqrt(ogx * ogx + ogy * ogy)
        clean_edge = np.clip((grad - 0.004) / 0.05, 0.0, 1.0)
        original_texture = np.clip((orig_grad - 0.003) / 0.04, 0.0, 1.0)
        detail_luma = np.mean(np.abs(detail), axis=2)
        detail_gate = np.clip((detail_luma - 0.001) / 0.015, 0.0, 1.0)
        edge = np.maximum(clean_edge, original_texture * detail_gate)
        edge = cv2.GaussianBlur(edge, (0, 0), 0.8)[:, :, np.newaxis]
        return np.clip(denoised + detail * edge * float(np.clip(strength, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)

    def _run_tiled(self, arr: np.ndarray, progress=None) -> np.ndarray | None:
        ts, halo = self.TILE, self.halo
        inner = ts - 2 * halo  # stride of placed (valid) region per tile
        if inner <= 0:
            inner = ts
            halo = 0
        h, w = arr.shape[:2]
        jobs = list(self._tile_jobs(h, w, inner, halo))
        total = len(jobs)
        out = np.empty_like(arr)
        if not jobs:
            return out
        if self.batch_size > 1 and (self._supports_dynamic_batch or self._fixed_batch_size > 1):
            return self._run_tiled_batched(arr, out, jobs, progress=progress)
        if self.workers > 1 and len(jobs) > 1:
            return self._run_tiled_parallel(arr, out, jobs, progress=progress)
        return self._run_tiled_serial(arr, out, jobs, progress=progress)

    def _tile_jobs(self, h: int, w: int, inner: int, halo: int):
        for y0 in range(0, h, inner):
            for x0 in range(0, w, inner):
                y1, x1 = min(y0 + inner, h), min(x0 + inner, w)
                ry0, rx0 = max(0, y0 - halo), max(0, x0 - halo)
                yield (y0, x0, y1, x1, ry0, rx0)

    def _read_tile(self, arr: np.ndarray, job) -> tuple[np.ndarray, int, int]:
        y0, x0, _y1, _x1, ry0, rx0 = job
        ts = self.TILE
        region = arr[ry0:ry0 + ts, rx0:rx0 + ts]
        rh, rw = region.shape[:2]
        if rh < ts or rw < ts:
            pad = ((0, ts - rh), (0, ts - rw), (0, 0))
            # Reflect padding needs at least two samples on each padded axis. Tiny images are
            # rare here, but edge padding keeps the denoise action robust for thumbnails/tests.
            mode = "reflect" if rh > 1 and rw > 1 else "edge"
            region = np.pad(region, pad, mode=mode)
        oy, ox = y0 - ry0, x0 - rx0  # offset of valid region inside the tile
        return region, oy, ox

    def _write_tile(self, out: np.ndarray, job, den: np.ndarray, oy: int, ox: int) -> bool:
        y0, x0, y1, x1, _ry0, _rx0 = job
        valid = den[oy:oy + (y1 - y0), ox:ox + (x1 - x0)]
        if valid.shape != out[y0:y1, x0:x1].shape:
            return False
        out[y0:y1, x0:x1] = valid
        return True

    def _run_tiled_serial(self, arr: np.ndarray, out: np.ndarray, jobs: list[tuple], progress=None) -> np.ndarray | None:
        total = len(jobs)
        for done, job in enumerate(jobs, start=1):
            tile, oy, ox = self._read_tile(arr, job)
            den = self._infer(tile)
            if den is None or not self._write_tile(out, job, den, oy, ox):
                return None
            if progress is not None:
                progress(done, total)
        return out

    def _run_tiled_parallel(self, arr: np.ndarray, out: np.ndarray, jobs: list[tuple], progress=None) -> np.ndarray | None:
        total = len(jobs)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._infer_tile_job, arr, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is None:
                    for pending in futures:
                        pending.cancel()
                    return None
                job, den, oy, ox = result
                if not self._write_tile(out, job, den, oy, ox):
                    for pending in futures:
                        pending.cancel()
                    return None
                done += 1
                if progress is not None:
                    progress(done, total)
        return out

    def _run_tiled_batched(self, arr: np.ndarray, out: np.ndarray, jobs: list[tuple], progress=None) -> np.ndarray | None:
        total = len(jobs)
        done = 0
        batch_size = self._effective_batch_size()
        for start in range(0, len(jobs), batch_size):
            batch_jobs = jobs[start:start + batch_size]
            tiles, offsets = [], []
            for job in batch_jobs:
                tile, oy, ox = self._read_tile(arr, job)
                tiles.append(tile)
                offsets.append((oy, ox))
            denoised = self._infer_batch(tiles)
            if denoised is None or len(denoised) != len(batch_jobs):
                return None
            for job, den, (oy, ox) in zip(batch_jobs, denoised, offsets):
                if not self._write_tile(out, job, den, oy, ox):
                    return None
                done += 1
                if progress is not None:
                    progress(done, total)
        return out

    def _infer_tile_job(self, arr: np.ndarray, job):
        tile, oy, ox = self._read_tile(arr, job)
        den = self._infer(tile)
        if den is None:
            return None
        return job, den, oy, ox

    def _infer(self, tile: np.ndarray) -> np.ndarray | None:
        tensor = np.transpose(tile, (2, 0, 1))[None, ...].astype(np.float32)
        out = self._session.run(None, {self._input_name: tensor})[0]
        den = np.squeeze(np.asarray(out, dtype=np.float32))
        if den.ndim != 3:
            return None
        if den.shape[0] == 3:
            den = np.transpose(den, (1, 2, 0))
        if den.shape != tile.shape:
            return None
        return np.clip(den, 0.0, 1.0)

    def _infer_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray] | None:
        if not tiles:
            return []
        requested = len(tiles)
        batch_tiles = list(tiles)
        fixed_batch = self._fixed_batch_size if self._fixed_batch_size > 1 and not self._supports_dynamic_batch else 0
        if fixed_batch:
            if requested > fixed_batch:
                return None
            # A fixed-batch ONNX cannot accept the smaller final batch. Pad with the last real
            # tile and discard the extra outputs after inference.
            while len(batch_tiles) < fixed_batch:
                batch_tiles.append(batch_tiles[-1])
        tensor = np.stack([np.transpose(tile, (2, 0, 1)) for tile in batch_tiles], axis=0).astype(np.float32)
        out = self._session.run(None, {self._input_name: tensor})[0]
        den = np.asarray(out, dtype=np.float32)
        if den.ndim != 4:
            return None
        if den.shape[1] == 3:
            den = np.transpose(den, (0, 2, 3, 1))
        expected = (len(batch_tiles), self.TILE, self.TILE, 3)
        if den.shape != expected:
            return None
        return [np.clip(den[i], 0.0, 1.0) for i in range(requested)]

    def _refresh_input_contract(self) -> None:
        self._supports_dynamic_batch = False
        self._fixed_batch_size = 1
        if self._session is None:
            self.batch_size = 1
            return
        try:
            batch_dim = self._session.get_inputs()[0].shape[0]
        except Exception:
            batch_dim = 1
        if batch_dim is None or isinstance(batch_dim, str):
            self._supports_dynamic_batch = True
            return
        try:
            self._fixed_batch_size = max(1, int(batch_dim))
        except (TypeError, ValueError):
            self._fixed_batch_size = 1
        if self._fixed_batch_size > 1:
            self.batch_size = self._fixed_batch_size
        else:
            self.batch_size = 1

    def _effective_batch_size(self) -> int:
        if self._fixed_batch_size > 1 and not self._supports_dynamic_batch:
            return self._fixed_batch_size
        return max(1, int(self.batch_size))

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return bool(default)
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_DEEP_DENOISE_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        for name in ("deep_denoise_b2.onnx", "deep_denoise.onnx", "nafnet.onnx"):
            candidates.append(models_dir / name)
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _preferred_onnx_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        providers = []
        if self.use_coreml and "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _prepare_coreml_cache_dir(self, providers: list[str]) -> None:
        if "CoreMLExecutionProvider" not in providers:
            return
        cache_dir = Path(__file__).resolve().parents[2] / ".ort_coreml_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_str = str(cache_dir)
        os.environ.setdefault("TMPDIR", cache_str)
        tempfile.tempdir = cache_str
