import unittest

import numpy as np

from portrait_enhancer.core import tone_curve as tc


class CurveNormalizationTests(unittest.TestCase):
    def test_default_is_identity(self):
        self.assertTrue(tc.is_identity(tc.default_curve()))

    def test_normalize_sorts_and_anchors_endpoints(self):
        pts = tc.normalize_curve([(0.5, 0.6), (1.0, 0.9), (0.0, 0.1)])
        self.assertEqual(pts[0][0], 0.0)
        self.assertEqual(pts[-1][0], 1.0)
        self.assertEqual([p[0] for p in pts], sorted(p[0] for p in pts))

    def test_normalize_handles_garbage(self):
        self.assertEqual(tc.normalize_curve("nope"), tc.default_curve())

    def test_collisions_dropped(self):
        pts = tc.normalize_curve([(0.0, 0.0), (0.5, 0.4), (0.5005, 0.6), (1.0, 1.0)])
        xs = [p[0] for p in pts]
        self.assertEqual(len(xs), len(set(xs)))


class LutTests(unittest.TestCase):
    def test_identity_lut_is_linear(self):
        lut = tc.curve_to_lut(tc.default_curve())
        self.assertEqual(lut.shape[0], tc.LUT_SIZE)
        self.assertAlmostEqual(float(lut[0]), 0.0, places=5)
        self.assertAlmostEqual(float(lut[-1]), 1.0, places=5)
        self.assertTrue(np.allclose(lut, np.linspace(0, 1, tc.LUT_SIZE), atol=1e-5))

    def test_lut_is_monotonic_nondecreasing(self):
        lut = tc.curve_to_lut([(0.0, 0.0), (0.25, 0.1), (0.75, 0.9), (1.0, 1.0)])
        self.assertTrue(np.all(np.diff(lut) >= -1e-6))

    def test_lift_curve_brightens_midtones(self):
        lut = tc.curve_to_lut([(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)])
        mid = int(0.5 * (tc.LUT_SIZE - 1))
        self.assertGreater(float(lut[mid]), 0.5)


class ApplyTests(unittest.TestCase):
    def test_identity_returns_same_object(self):
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        self.assertIs(tc.apply_curve(img, tc.default_curve()), img)

    def test_lift_brightens_image(self):
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = tc.apply_curve(img, [(0.0, 0.0), (0.5, 0.75), (1.0, 1.0)])
        self.assertGreater(float(out.mean()), 0.5)

    def test_output_stays_in_range(self):
        rng = np.random.default_rng(0)
        img = rng.random((8, 6, 3), dtype=np.float32)
        out = tc.apply_curve(img, [(0.0, 0.05), (0.4, 0.2), (1.0, 0.95)])
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class PointEditingTests(unittest.TestCase):
    def test_add_point_inserts_sorted(self):
        pts, idx = tc.add_point(tc.default_curve(), 0.5, 0.6)
        self.assertEqual(len(pts), 3)
        self.assertEqual([p[0] for p in pts], sorted(p[0] for p in pts))
        self.assertAlmostEqual(pts[1][0], 0.5)

    def test_move_endpoint_keeps_x(self):
        pts = tc.move_point(tc.default_curve(), 0, 0.4, 0.2)
        self.assertEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[0][1], 0.2)

    def test_move_interior_clamps_between_neighbors(self):
        curve = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        moved = tc.move_point(curve, 1, 5.0, 0.7)  # x past the right neighbor
        self.assertLess(moved[1][0], 1.0)
        self.assertGreater(moved[1][0], 0.0)

    def test_remove_interior_only(self):
        curve = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        self.assertEqual(len(tc.remove_point(curve, 1)), 2)
        self.assertEqual(len(tc.remove_point(curve, 0)), 3)  # endpoint not removed
        self.assertEqual(len(tc.remove_point(curve, 2)), 3)

    def test_nearest_point_hit_and_miss(self):
        curve = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        self.assertEqual(tc.nearest_point(curve, 0.5, 0.5, 0.05), 1)
        self.assertIsNone(tc.nearest_point(curve, 0.5, 0.9, 0.05))


if __name__ == "__main__":
    unittest.main()
