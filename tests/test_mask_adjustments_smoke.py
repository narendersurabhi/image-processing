import unittest

try:
    import numpy as np

    from portrait_enhancer.core.masks import (
        adjust_mask,
        apply_mask_adjustments,
        default_mask_adjustments,
        normalize_mask_adjustments,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/opencv not installed")
class MaskAdjustmentSmokeTests(unittest.TestCase):
    def test_normalize_mask_adjustments_clamps_and_fills_layers(self):
        normalized = normalize_mask_adjustments(
            {
                "skin": {"strength": 250, "feather": -5, "expand": -80},
            }
        )

        self.assertEqual(normalized["skin"]["strength"], 200.0)
        self.assertEqual(normalized["skin"]["feather"], 0.0)
        self.assertEqual(normalized["skin"]["expand"], -40.0)
        self.assertEqual(normalized["eyes"], default_mask_adjustments()["eyes"])

    def test_adjust_mask_strength_scales_soft_mask(self):
        mask = np.ones((16, 16), dtype=np.float32) * 0.8

        adjusted = adjust_mask(mask, {"strength": 50, "feather": 0, "expand": 0})

        self.assertAlmostEqual(float(adjusted.mean()), 0.4, places=3)

    def test_expand_and_contract_change_mask_coverage(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[52:76, 52:76] = 1.0

        expanded = adjust_mask(mask, {"strength": 100, "feather": 0, "expand": 30})
        contracted = adjust_mask(mask, {"strength": 100, "feather": 0, "expand": -30})

        self.assertGreater(float(expanded.mean()), float(mask.mean()))
        self.assertLess(float(contracted.mean()), float(mask.mean()))

    def test_apply_mask_adjustments_preserves_unconfigured_masks(self):
        masks = {
            "skin": np.ones((8, 8), dtype=np.float32),
            "brows": np.ones((8, 8), dtype=np.float32) * 0.25,
        }

        adjusted = apply_mask_adjustments(masks, {"skin": {"strength": 25}})

        self.assertAlmostEqual(float(adjusted["skin"].mean()), 0.25, places=3)
        self.assertAlmostEqual(float(adjusted["brows"].mean()), 0.25, places=3)
        self.assertIsNot(adjusted["brows"], masks["brows"])


if __name__ == "__main__":
    unittest.main()
