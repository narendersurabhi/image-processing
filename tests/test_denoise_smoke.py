import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np

    from portrait_enhancer.core.denoise import DeepDenoiser

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class _FakeOnnxSession:
    def __init__(self, delay: float = 0.0):
        self.delay = float(delay)
        self.calls = 0
        self.max_batch = 0
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()

    def run(self, _outputs, inputs):
        tensor = next(iter(inputs.values()))
        with self._lock:
            self.calls += 1
            self.max_batch = max(self.max_batch, int(tensor.shape[0]))
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay > 0:
                time.sleep(self.delay)
            return [tensor.copy()]
        finally:
            with self._lock:
                self._active -= 1


def _fake_deep_denoiser(*, workers=1, batch_size=1, dynamic_batch=False, fixed_batch=1, delay=0.0):
    denoiser = DeepDenoiser.__new__(DeepDenoiser)
    denoiser.available = True
    denoiser.backend = "nafnet"
    denoiser.reason_unavailable = ""
    denoiser.execution_provider = "fake"
    denoiser._session = _FakeOnnxSession(delay=delay)
    denoiser._input_name = "input"
    denoiser._supports_dynamic_batch = bool(dynamic_batch)
    denoiser._fixed_batch_size = int(fixed_batch)
    denoiser.batch_size = int(batch_size)
    denoiser.workers = int(workers)
    denoiser.halo = 16
    return denoiser


@unittest.skipUnless(HAS_DEPS, "numpy deps not installed")
class DeepDenoiserSmokeTests(unittest.TestCase):
    def _image(self):
        rng = np.random.default_rng(123)
        return rng.uniform(0.0, 1.0, (520, 520, 3)).astype(np.float32)

    def test_parallel_tile_path_preserves_image_and_uses_multiple_workers(self):
        denoiser = _fake_deep_denoiser(workers=4, batch_size=1, dynamic_batch=False, delay=0.02)
        progress = []

        out = denoiser.denoise(self._image(), progress=lambda done, total: progress.append((done, total)))

        self.assertIsNotNone(out)
        self.assertTrue(np.allclose(out, self._image()))
        self.assertGreater(denoiser._session.max_active, 1)
        self.assertEqual(progress[-1], (9, 9))

    def test_dynamic_batch_tile_path_groups_tiles_when_supported(self):
        denoiser = _fake_deep_denoiser(workers=4, batch_size=3, dynamic_batch=True)
        progress = []

        out = denoiser.denoise(self._image(), progress=lambda done, total: progress.append((done, total)))

        self.assertIsNotNone(out)
        self.assertTrue(np.allclose(out, self._image()))
        self.assertEqual(denoiser._session.max_batch, 3)
        self.assertEqual(denoiser._session.calls, 3)
        self.assertEqual(progress[-1], (9, 9))

    def test_fixed_batch_tile_path_pads_the_final_batch(self):
        denoiser = _fake_deep_denoiser(workers=4, batch_size=4, dynamic_batch=False, fixed_batch=4)
        progress = []

        out = denoiser.denoise(self._image(), progress=lambda done, total: progress.append((done, total)))

        self.assertIsNotNone(out)
        self.assertTrue(np.allclose(out, self._image()))
        self.assertEqual(denoiser._session.max_batch, 4)
        self.assertEqual(denoiser._session.calls, 3)
        self.assertEqual(progress[-1], (9, 9))

    def test_zero_amount_returns_without_inference(self):
        denoiser = _fake_deep_denoiser(workers=4, batch_size=3, dynamic_batch=True)
        img = self._image()

        out = denoiser.denoise(img, amount=0.0)

        self.assertTrue(np.array_equal(out, img))
        self.assertEqual(denoiser._session.calls, 0)

    def test_default_model_path_prefers_batch_two(self):
        denoiser = DeepDenoiser.__new__(DeepDenoiser)

        def exists(path):
            return Path(path).name in {"deep_denoise_b2.onnx", "deep_denoise.onnx", "nafnet.onnx"}

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(Path, "exists", exists), mock.patch.object(Path, "is_file", lambda _path: True):
                path = denoiser._resolve_model_path()

        self.assertEqual(path.name, "deep_denoise_b2.onnx")

    def test_deep_denoiser_coreml_provider_can_be_disabled(self):
        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        denoiser = DeepDenoiser.__new__(DeepDenoiser)
        denoiser.use_coreml = False
        self.assertEqual(denoiser._preferred_onnx_providers(FakeOrt), ["CPUExecutionProvider"])

        denoiser.use_coreml = True
        self.assertEqual(
            denoiser._preferred_onnx_providers(FakeOrt),
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )


if __name__ == "__main__":
    unittest.main()
