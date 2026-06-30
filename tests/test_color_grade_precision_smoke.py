import unittest

try:
    import cv2
    import numpy as np

    from portrait_enhancer.core import utils
    from portrait_enhancer.core.utils import (
        adjust_color_balance_preserve_chroma,
        adjust_hsv_sat,
        adjust_warmth_preserve_hue,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/cv2 deps not installed")
class ColorGradePrecisionTests(unittest.TestCase):
    """The Global Color grades (Saturation/Vibrance/Warmth/Tint) now run their HSV/LAB math in
    float at the 8-bit value conventions instead of round-tripping through uint8. These guard
    that (a) the look is unchanged vs the old uint8 path and (b) the 256-level banding is gone.
    """

    def _img(self):
        rng = np.random.default_rng(0)
        return rng.uniform(0.05, 0.95, (64, 64, 3)).astype(np.float32)

    def test_saturation_matches_old_uint8_within_quantization(self):
        img = self._img()

        # The previous uint8 implementation, inline, as the reference.
        def old_sat(im, delta):
            hsv = cv2.cvtColor(utils.to_uint8(im), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + delta / 100.0), 0, 255)
            return utils.to_float(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))

        new = adjust_hsv_sat(img, 40)
        old = old_sat(img, 40)
        # Should differ only by the removed quantization (~1/255), not in look.
        self.assertLess(float(np.mean(np.abs(new - old))), 0.01)
        self.assertLess(float(np.max(np.abs(new - old))), 0.06)

    def test_no_8bit_banding_on_a_gradient(self):
        grad = np.repeat(np.linspace(0.1, 0.9, 2000, dtype=np.float32)[None, :, None], 3, axis=2)
        grad[:, :, 2] *= 0.6  # give it chroma so saturation has an effect
        out = adjust_hsv_sat(grad, 50)
        # uint8 round-trip would cap distinct outputs at ~256; float keeps them.
        self.assertGreater(np.unique(np.round(out[0, :, 0], 6)).size, 256)

    def test_identity_when_zero(self):
        img = self._img()
        self.assertTrue(np.array_equal(adjust_hsv_sat(img, 0), img))
        self.assertTrue(np.array_equal(adjust_warmth_preserve_hue(img, 0), img))
        self.assertTrue(np.array_equal(adjust_color_balance_preserve_chroma(img, 0, 0), img))

    def test_grades_stay_finite_and_in_range(self):
        img = self._img()
        for out in (
            adjust_hsv_sat(img, 70),
            adjust_warmth_preserve_hue(img, -80),
            adjust_color_balance_preserve_chroma(img, 50, -40),
        ):
            self.assertTrue(np.isfinite(out).all())
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
