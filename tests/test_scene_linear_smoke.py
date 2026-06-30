import unittest

try:
    import numpy as np

    from portrait_enhancer.core import tone_curve as tc
    from portrait_enhancer.core.processing import process_global
    from portrait_enhancer.core.utils import (
        apply_tone_curve_preserve_chroma,
        linear_to_srgb,
        srgb_to_linear,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy deps not installed")
class SceneLinearSmokeTests(unittest.TestCase):
    """Guards the scene-linear pipeline: the default sRGB path must stay byte-identical, and
    the linear path must be the domain-correct 'convert -> apply display-referred -> convert
    back' transform with physically-correct exposure."""

    def _img(self):
        rng = np.random.default_rng(7)
        return rng.uniform(0.03, 0.97, (48, 48, 3)).astype(np.float32)

    def test_srgb_default_is_byte_identical(self):
        # The whole point of the working_space default: existing edits don't move a pixel.
        img = self._img()
        a = apply_tone_curve_preserve_chroma(img, 0, 30, 0, -20, 10)
        b = apply_tone_curve_preserve_chroma(img, 0, 30, 0, -20, 10, working_space="srgb")
        self.assertTrue(np.array_equal(a, b))

        pts = [(0.0, 0.0), (0.3, 0.22), (0.7, 0.8), (1.0, 1.0)]
        self.assertTrue(np.array_equal(tc.apply_curve(img, pts), tc.apply_curve(img, pts, working_space="srgb")))

    def test_process_global_srgb_unchanged_with_or_without_color_settings(self):
        img = self._img()
        p = {"exposure": 40, "shadows": 25, "highlights": -30, "blacks": 5}
        a = process_global(img.copy(), dict(p))  # color_settings=None -> srgb default
        b = process_global(img.copy(), dict(p), color_settings={"working_space": "srgb"})
        self.assertTrue(np.array_equal(a, b))

    def test_linear_tone_curve_equals_convert_apply_convert(self):
        # The linear branch must be exactly: linear -> sRGB -> apply -> linear.
        lin = srgb_to_linear(self._img())
        direct = apply_tone_curve_preserve_chroma(lin, 0, 30, 0, -20, 10, working_space="linear")
        manual = srgb_to_linear(apply_tone_curve_preserve_chroma(linear_to_srgb(lin), 0, 30, 0, -20, 10))
        self.assertTrue(np.allclose(direct, manual, atol=1e-5))

        pts = [(0.0, 0.0), (0.4, 0.3), (1.0, 1.0)]
        direct_c = tc.apply_curve(lin, pts, working_space="linear")
        manual_c = srgb_to_linear(tc.apply_curve(linear_to_srgb(lin), pts))
        self.assertTrue(np.allclose(direct_c, manual_c, atol=1e-5))

    def test_linear_is_domain_aware_not_naive(self):
        # Domain-aware (convert) must differ from applying display-referred control points
        # straight onto linear values -- proving the conversion actually happens.
        lin = srgb_to_linear(self._img())
        aware = apply_tone_curve_preserve_chroma(lin, 0, 40, 0, 0, 0, working_space="linear")
        naive = apply_tone_curve_preserve_chroma(lin, 0, 40, 0, 0, 0, working_space="srgb")
        self.assertGreater(float(np.mean(np.abs(aware - naive))), 1e-3)

    def test_process_global_linear_path_sane_and_differs(self):
        img = self._img()
        p = {"exposure": 50, "highlights": -40, "shadows": 20}
        srgb = process_global(img.copy(), dict(p), color_settings={"working_space": "srgb"})
        lin = process_global(srgb_to_linear(img), dict(p), color_settings={"working_space": "linear"})
        self.assertTrue(np.isfinite(lin).all())
        self.assertGreaterEqual(float(lin.min()), 0.0)
        self.assertLessEqual(float(lin.max()), 1.0)
        self.assertGreater(float(np.mean(np.abs(srgb - linear_to_srgb(lin)))), 1e-3)

    def test_exposure_runs_in_linear_light(self):
        # +1 EV on a flat patch in scene-linear is a true doubling of the linear value (all
        # other global ops are identity at these params), which display-referred sRGB can't be.
        gray = np.full((4, 4, 3), 0.5, np.float32)
        lin = process_global(srgb_to_linear(gray), {"exposure": 100}, color_settings={"working_space": "linear"})
        expected = np.clip(srgb_to_linear(gray) * 2.0, 0.0, 1.0)
        self.assertTrue(np.allclose(lin, expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
