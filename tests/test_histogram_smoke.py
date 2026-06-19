import unittest

import numpy as np
from PIL import Image

from portrait_enhancer.core.histogram import LEVELS, compute_histogram


class HistogramComputationTests(unittest.TestCase):
    def test_uniform_black_image_clips_shadows(self):
        arr = np.zeros((10, 8, 3), dtype=np.uint8)
        hist = compute_histogram(arr)

        self.assertEqual(hist["total"], 80)
        self.assertEqual(int(hist["red"][0]), 80)
        self.assertEqual(int(hist["red"][1:].sum()), 0)
        self.assertAlmostEqual(hist["shadow_clip"]["red"], 1.0)
        self.assertAlmostEqual(hist["highlight_clip"]["red"], 0.0)

    def test_uniform_white_image_clips_highlights(self):
        arr = np.full((4, 4, 3), 255, dtype=np.uint8)
        hist = compute_histogram(arr)

        self.assertEqual(int(hist["blue"][LEVELS - 1]), 16)
        self.assertAlmostEqual(hist["highlight_clip"]["blue"], 1.0)
        self.assertAlmostEqual(hist["shadow_clip"]["blue"], 0.0)

    def test_counts_sum_to_total_per_channel(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(12, 9, 3), dtype=np.uint8)
        hist = compute_histogram(arr)

        for channel in ("red", "green", "blue", "luma"):
            self.assertEqual(int(np.asarray(hist[channel]).sum()), hist["total"])
            self.assertEqual(np.asarray(hist[channel]).shape[0], LEVELS)

    def test_float_array_is_scaled_from_unit_range(self):
        arr = np.ones((3, 3, 3), dtype=np.float32)  # 1.0 -> 255
        hist = compute_histogram(arr)
        self.assertEqual(int(hist["red"][LEVELS - 1]), 9)

    def test_accepts_pil_image(self):
        img = Image.new("RGB", (5, 5), (255, 0, 0))
        hist = compute_histogram(img)

        self.assertEqual(int(hist["red"][LEVELS - 1]), 25)
        self.assertEqual(int(hist["green"][0]), 25)
        self.assertEqual(int(hist["blue"][0]), 25)

    def test_grayscale_array_is_broadcast_to_rgb(self):
        arr = np.full((6, 6), 128, dtype=np.uint8)
        hist = compute_histogram(arr)
        self.assertEqual(int(hist["red"][128]), 36)
        self.assertEqual(int(hist["luma"][128]), 36)


if __name__ == "__main__":
    unittest.main()
