import unittest

import numpy as np

from portrait_enhancer.core import white_balance as wb


class GainMathTests(unittest.TestCase):
    def test_neutral_sample_returns_identity(self):
        gains = wb.gains_from_neutral_sample([0.5, 0.5, 0.5])
        self.assertTrue(wb.is_identity(gains))

    def test_normalize_rejects_garbage(self):
        self.assertEqual(wb.normalize_gains("nope"), list(wb.DEFAULT_GAINS))
        self.assertEqual(wb.normalize_gains([1.0, 2.0]), list(wb.DEFAULT_GAINS))

    def test_gains_clamped_to_range(self):
        gains = wb.gains_from_neutral_sample([0.001, 0.5, 0.5])
        self.assertTrue(all(wb.MIN_GAIN - 1e-9 <= g <= wb.MAX_GAIN + 1e-9 for g in gains))

    def test_blue_cast_sample_boosts_red_drops_blue(self):
        # Pixel that should be neutral but reads bluish -> red gain up, blue gain down.
        gains = wb.gains_from_neutral_sample([0.4, 0.5, 0.7])
        self.assertGreater(gains[0], 1.0)
        self.assertLess(gains[2], 1.0)


class ApplyTests(unittest.TestCase):
    def test_identity_returns_same_object(self):
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = wb.apply_white_balance_gains(img, [1.0, 1.0, 1.0])
        self.assertIs(out, img)

    def test_picking_neutral_makes_patch_gray(self):
        # Build a uniform bluish image; gains from a sample should neutralize it.
        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[:, :, 0] = 0.40
        img[:, :, 1] = 0.50
        img[:, :, 2] = 0.70
        gains = wb.gains_from_neutral_sample(img[0, 0], working_space="srgb")
        out = wb.apply_white_balance_gains(img, gains, working_space="srgb")
        px = out[0, 0]
        self.assertAlmostEqual(px[0], px[1], delta=0.02)
        self.assertAlmostEqual(px[1], px[2], delta=0.02)

    def test_apply_preserves_shape_and_range(self):
        rng = np.random.default_rng(0)
        img = rng.random((6, 5, 3), dtype=np.float32)
        out = wb.apply_white_balance_gains(img, [1.3, 1.0, 0.8])
        self.assertEqual(out.shape, img.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class KelvinModelTests(unittest.TestCase):
    def test_neutral_kelvin_is_identity(self):
        self.assertTrue(wb.is_identity(wb.kelvin_tint_to_gains(wb.NEUTRAL_K, 0)))

    def test_higher_kelvin_warms_image(self):
        g = wb.kelvin_tint_to_gains(9000, 0)
        self.assertGreater(g[0], 1.0)  # red up
        self.assertLess(g[2], 1.0)     # blue down

    def test_lower_kelvin_cools_image(self):
        g = wb.kelvin_tint_to_gains(3200, 0)
        self.assertLess(g[0], 1.0)
        self.assertGreater(g[2], 1.0)

    def test_tint_magenta_lowers_green(self):
        self.assertLess(wb.kelvin_tint_to_gains(wb.NEUTRAL_K, 100)[1], 1.0)
        self.assertGreater(wb.kelvin_tint_to_gains(wb.NEUTRAL_K, -100)[1], 1.0)

    def test_warming_a_neutral_image_makes_red_exceed_blue(self):
        img = np.full((6, 6, 3), 0.5, dtype=np.float32)
        out = wb.apply_white_balance_gains(img, wb.kelvin_tint_to_gains(8500, 0))
        self.assertGreater(float(out[0, 0, 0]), float(out[0, 0, 2]))

    def test_clamps_out_of_range(self):
        self.assertEqual(wb.clamp_temp(99999), wb.MAX_K)
        self.assertEqual(wb.clamp_temp(10), wb.MIN_K)
        self.assertEqual(wb.clamp_tint(9999), wb.TINT_MAX)


class InversionTests(unittest.TestCase):
    def test_round_trip_kelvin_tint(self):
        for temp, tint in ((3200, 0), (5500, 20), (7500, -30), (9000, 10)):
            gains = wb.kelvin_tint_to_gains(temp, tint)
            t2, ti2 = wb.gains_to_kelvin_tint(gains)
            self.assertAlmostEqual(t2, temp, delta=150)
            self.assertAlmostEqual(ti2, tint, delta=6)

    def test_neutral_gains_invert_to_neutral_setting(self):
        t, ti = wb.gains_to_kelvin_tint([1.0, 1.0, 1.0])
        self.assertAlmostEqual(t, wb.NEUTRAL_K, delta=150)
        self.assertAlmostEqual(ti, 0, delta=4)

    def test_pick_then_apply_neutralizes_with_synced_sliders(self):
        # Mild bluish cast (within the blackbody locus range): pick -> (kelvin, tint)
        # -> gains -> neutralizes the patch. A bluish cast needs a WARM correction.
        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[:, :, 0] = 0.45
        img[:, :, 1] = 0.50
        img[:, :, 2] = 0.58
        temp, tint = wb.neutral_sample_to_kelvin_tint(img[0, 0], working_space="srgb")
        self.assertGreater(temp, wb.NEUTRAL_K)  # bluish pixel -> warm/high-K correction
        out = wb.apply_white_balance_gains(img, wb.kelvin_tint_to_gains(temp, tint), working_space="srgb")
        px = out[0, 0]
        self.assertAlmostEqual(px[0], px[1], delta=0.03)
        self.assertAlmostEqual(px[1], px[2], delta=0.03)

    def test_warm_pixel_inverts_to_low_kelvin(self):
        # A reddish/warm pixel should map to a cool (low-K) correction.
        temp, _ = wb.neutral_sample_to_kelvin_tint([0.62, 0.50, 0.42], working_space="srgb")
        self.assertLess(temp, wb.NEUTRAL_K)


class PresetTests(unittest.TestCase):
    def test_presets_are_valid_and_in_range(self):
        self.assertTrue(wb.PRESETS)
        for name, temp, tint in wb.PRESETS:
            self.assertIsInstance(name, str)
            self.assertEqual(wb.clamp_temp(temp), temp)
            self.assertEqual(wb.clamp_tint(tint), tint)

    def test_tungsten_cools_shade_warms(self):
        tungsten = dict((n, (t, ti)) for n, t, ti in wb.PRESETS)["Tungsten"]
        shade = dict((n, (t, ti)) for n, t, ti in wb.PRESETS)["Shade"]
        self.assertLess(wb.kelvin_tint_to_gains(*tungsten)[0], 1.0)   # red down -> cooler
        self.assertGreater(wb.kelvin_tint_to_gains(*shade)[0], 1.0)   # red up -> warmer


class EstimatorTests(unittest.TestCase):
    def test_gray_world_neutralizes_uniform_cast(self):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        img[:, :, 0] = 0.30
        img[:, :, 1] = 0.50
        img[:, :, 2] = 0.60
        gains = wb.gray_world_gains(img)
        out = wb.apply_white_balance_gains(img, gains)
        means = out.reshape(-1, 3).mean(axis=0)
        self.assertAlmostEqual(means[0], means[1], delta=0.02)
        self.assertAlmostEqual(means[1], means[2], delta=0.02)

    def test_white_patch_neutralizes_bright_region(self):
        img = np.full((10, 10, 3), 0.2, dtype=np.float32)
        img[0, 0] = [0.6, 0.7, 0.9]  # brightest "white" reads blue
        gains = wb.white_patch_gains(img, percentile=99.0)
        self.assertGreater(gains[0], gains[2])


if __name__ == "__main__":
    unittest.main()
