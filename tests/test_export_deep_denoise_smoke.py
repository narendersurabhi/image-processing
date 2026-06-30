import unittest
from unittest import mock

try:
    import numpy as np

    from portrait_enhancer.ui_qt import main_window
    from portrait_enhancer.ui_qt.main_window import PortraitEnhancerQtWindow

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message):
        self.messages.append(str(message))


class _FakeDenoiser:
    def __init__(self, result=None, *, available=True, provider="fake", reason=""):
        self._result = result
        self.available = bool(available)
        self.execution_provider = provider
        self.reason_unavailable = reason
        self.calls = 0

    def denoise(self, _array):
        self.calls += 1
        return self._result


@unittest.skipUnless(HAS_DEPS, "PySide6/numpy not installed")
class ExportDeepDenoiseSmokeTests(unittest.TestCase):
    def _window(self, denoiser):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        window.full_array = np.zeros((4, 4, 3), dtype=np.float32)
        window._deep_denoise_enabled = False
        window._deep_denoise_full = None
        window._deep_denoiser = denoiser
        status = _StatusBar()
        window.statusBar = lambda: status
        window._status = status
        return window

    def test_export_deep_denoise_retries_cpu_when_current_provider_fails(self):
        current = _FakeDenoiser(None, provider="coreml")
        cpu_result = np.full((4, 4, 3), 0.25, dtype=np.float32)
        cpu = _FakeDenoiser(cpu_result, provider="cpu")
        window = self._window(current)

        with mock.patch.object(main_window.QApplication, "processEvents", lambda: None):
            with mock.patch.object(main_window, "DeepDenoiser", lambda use_coreml=False: cpu):
                result = window._export_base_full_array(force_deep_denoise=True)

        self.assertTrue(np.array_equal(result, cpu_result))
        self.assertIs(window._deep_denoise_full, result)
        self.assertIs(window._deep_denoiser, cpu)
        self.assertEqual(current.calls, 1)
        self.assertEqual(cpu.calls, 1)

    def test_export_deep_denoise_failure_raises_instead_of_returning_original(self):
        current = _FakeDenoiser(None, provider="coreml")
        cpu = _FakeDenoiser(None, provider="cpu")
        window = self._window(current)

        with mock.patch.object(main_window.QApplication, "processEvents", lambda: None):
            with mock.patch.object(main_window, "DeepDenoiser", lambda use_coreml=False: cpu):
                with self.assertRaisesRegex(RuntimeError, "Deep Denoise was requested"):
                    window._export_base_full_array(force_deep_denoise=True)

        self.assertIsNone(window._deep_denoise_full)


if __name__ == "__main__":
    unittest.main()
