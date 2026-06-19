import unittest

import numpy as np

from portrait_enhancer.core import color_mixer as cm


def _solid(rgb, size=8):
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :] = rgb
    return img


class NormalizationTests(unittest.TestCase):
    def test_default_is_identity(self):
        self.assertTrue(cm.is_identity(cm.default_color_mixer()))

    def test_normalize_clamps_and_fills(self):
        norm = cm.normalize_color_mixer({"red": {"sat": 500, "hue": "x"}})
        self.assertEqual(norm["red"]["sat"], 100)
        self.assertEqual(norm["red"]["hue"], 0)
        self.assertIn("magenta", norm)

    def test_normalize_handles_garbage(self):
        norm = cm.normalize_color_mixer("nope")
        self.assertTrue(cm.is_identity(norm))


class ApplyTests(unittest.TestCase):
    def test_identity_returns_same_object(self):
        img = _solid([0.8, 0.1, 0.1])
        self.assertIs(cm.apply_color_mixer(img, cm.default_color_mixer()), img)

    def test_red_saturation_down_desaturates_red(self):
        img = _solid([0.85, 0.12, 0.12])
        adj = cm.default_color_mixer()
        adj["red"]["sat"] = -100
        out = cm.apply_color_mixer(img, adj)
        # channels move closer together (less saturated)
        before_spread = float(img[0, 0].max() - img[0, 0].min())
        after_spread = float(out[0, 0].max() - out[0, 0].min())
        self.assertLess(after_spread, before_spread)

    def test_red_luminance_up_brightens_red(self):
        img = _solid([0.6, 0.1, 0.1])
        adj = cm.default_color_mixer()
        adj["red"]["lum"] = 100
        out = cm.apply_color_mixer(img, adj)
        self.assertGreater(float(out.mean()), float(img.mean()))

    def test_band_isolation_blue_adjust_leaves_red_untouched(self):
        img = _solid([0.85, 0.12, 0.12])  # red
        adj = cm.default_color_mixer()
        adj["blue"]["sat"] = -100
        out = cm.apply_color_mixer(img, adj)
        self.assertTrue(np.allclose(out, img, atol=0.02))

    def test_near_gray_is_unaffected(self):
        img = _solid([0.5, 0.5, 0.5])
        adj = cm.default_color_mixer()
        adj["red"]["sat"] = 100
        adj["red"]["lum"] = 100
        out = cm.apply_color_mixer(img, adj)
        self.assertTrue(np.allclose(out, img, atol=0.02))

    def test_output_in_range(self):
        rng = np.random.default_rng(0)
        img = rng.random((10, 7, 3), dtype=np.float32)
        adj = cm.default_color_mixer()
        adj["green"]["sat"] = 80
        adj["blue"]["lum"] = -60
        out = cm.apply_color_mixer(img, adj)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class WeightTests(unittest.TestCase):
    def test_weight_peaks_at_center(self):
        hue = np.array([0.0, 30.0, 45.0, 90.0], dtype=np.float32)
        w = cm._band_weight(hue, 0.0)
        self.assertAlmostEqual(float(w[0]), 1.0, places=5)
        self.assertEqual(float(w[3]), 0.0)  # beyond falloff

    def test_weight_is_circular(self):
        # hue 350 is close to red center 0 across the wrap
        w = cm._band_weight(np.array([350.0], dtype=np.float32), 0.0)
        self.assertGreater(float(w[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
