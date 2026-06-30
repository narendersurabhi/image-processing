import unittest

import numpy as np
from PIL import Image

from portrait_enhancer.core import framing


class FramingNormalizationTests(unittest.TestCase):
    def test_default_is_identity(self):
        self.assertTrue(framing.is_identity(framing.default_framing()))

    def test_normalize_clamps_angle_and_crop(self):
        f = framing.normalize_framing({"angle": 200.0, "crop": [-0.5, -0.5, 5.0, 5.0]})
        self.assertEqual(f["angle"], framing.MAX_ANGLE)
        x, y, w, h = f["crop"]
        self.assertGreaterEqual(x, 0.0)
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(x + w, 1.0 + 1e-9)
        self.assertLessEqual(y + h, 1.0 + 1e-9)

    def test_normalize_handles_garbage_crop(self):
        f = framing.normalize_framing({"crop": "nonsense"})
        self.assertEqual(f["crop"], list(framing.DEFAULT_CROP))

    def test_non_identity_when_cropped(self):
        self.assertFalse(framing.is_identity({"crop": [0.1, 0.1, 0.5, 0.5]}))


class ApplyFramingTests(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (200, 100), (10, 20, 30))

    def test_identity_returns_same_object(self):
        out = framing.apply_framing(self.img, framing.default_framing())
        self.assertIs(out, self.img)

    def test_center_half_crop_dimensions(self):
        out = framing.apply_framing(self.img, {"crop": [0.25, 0.25, 0.5, 0.5]})
        self.assertEqual(out.size, (100, 50))

    def test_flip_horizontal_swaps_columns(self):
        img = Image.new("RGB", (2, 1))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((1, 0), (0, 0, 255))
        out = framing.apply_framing(img, {"flip_h": True})
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(out.getpixel((1, 0)), (255, 0, 0))

    def test_rotate_keeps_canvas_size(self):
        out = framing.apply_framing(self.img, {"angle": 10.0})
        self.assertEqual(out.size, self.img.size)


class AspectPresetTests(unittest.TestCase):
    def test_square_crop_on_wide_image_uses_full_height(self):
        crop = framing.centered_crop_for_aspect(1.0, 200, 100)
        x, y, w, h = crop
        self.assertAlmostEqual(h, 1.0)
        self.assertAlmostEqual(w, 0.5)
        self.assertAlmostEqual(x, 0.25)

    def test_none_aspect_is_full_frame(self):
        self.assertEqual(framing.centered_crop_for_aspect(None, 200, 100), list(framing.DEFAULT_CROP))


class CropInteractionTests(unittest.TestCase):
    def test_hit_test_corner_handle(self):
        crop = [0.2, 0.2, 0.6, 0.6]
        self.assertEqual(framing.hit_test_handle(crop, 0.2, 0.2, tol=0.03), "nw")

    def test_hit_test_interior_is_move(self):
        crop = [0.2, 0.2, 0.6, 0.6]
        self.assertEqual(framing.hit_test_handle(crop, 0.5, 0.5, tol=0.03), "move")

    def test_hit_test_outside_is_none(self):
        crop = [0.2, 0.2, 0.6, 0.6]
        self.assertIsNone(framing.hit_test_handle(crop, 0.95, 0.95, tol=0.03))

    def test_move_clamps_to_bounds(self):
        crop = [0.2, 0.2, 0.6, 0.6]
        moved = framing.move_crop(crop, 0.5, 0.5)
        self.assertAlmostEqual(moved[0], 0.4)
        self.assertAlmostEqual(moved[1], 0.4)

    def test_resize_east_handle_grows_width(self):
        crop = [0.2, 0.2, 0.4, 0.4]
        resized = framing.resize_crop(crop, "e", 0.1, 0.0)
        self.assertAlmostEqual(resized[2], 0.5)

    def test_resize_respects_min_size(self):
        crop = [0.2, 0.2, 0.4, 0.4]
        resized = framing.resize_crop(crop, "w", 1.0, 0.0)  # drag left edge past right
        self.assertGreaterEqual(resized[2], framing.MIN_CROP - 1e-9)

    def test_resize_locked_aspect_corner_keeps_opposite_corner_fixed(self):
        # Square canvas, square target aspect -- fraction-space ratio equals 1:1 too. Checks
        # the exact resulting size (not just squareness), since a box wrongly collapsed to
        # min_size on both axes is *also* square -- that exact false-positive shape is what
        # an earlier version of this code produced (inverted available-room-to-grow-into
        # check), and a same-on-both-axes assertion alone didn't catch it.
        crop = [0.2, 0.2, 0.4, 0.4]
        resized = framing.resize_crop(crop, "se", 0.2, 0.0, aspect=1.0, canvas_aspect=1.0)
        self.assertAlmostEqual(resized[0], 0.2)  # left (opposite corner) unchanged
        self.assertAlmostEqual(resized[1], 0.2)  # top (opposite corner) unchanged
        self.assertAlmostEqual(resized[2], 0.6)
        self.assertAlmostEqual(resized[3], 0.6)

    def test_resize_locked_aspect_corner_can_grow_to_near_full_canvas(self):
        # A large drag toward the far corner should be able to grow well past the box's
        # starting size, limited only by the canvas edge -- not silently capped back down
        # near the box's own anchor-side coordinate (the inverted-room-to-grow-into bug).
        crop = [0.1, 0.1, 0.2, 0.2]
        resized = framing.resize_crop(crop, "se", 0.9, 0.9, aspect=1.0, canvas_aspect=1.0)
        self.assertAlmostEqual(resized[0], 0.1)
        self.assertAlmostEqual(resized[1], 0.1)
        self.assertAlmostEqual(resized[2], 0.9, places=5)  # capped by the canvas edge (1.0 - 0.1)
        self.assertAlmostEqual(resized[3], 0.9, places=5)

    def test_resize_locked_aspect_accounts_for_canvas_aspect(self):
        # A 16:9 source canvas: a 1:1 true-pixel target is *not* 1:1 in fraction-space --
        # the fraction width must be narrower than the fraction height to compensate.
        crop = [0.4, 0.4, 0.2, 0.2]
        resized = framing.resize_crop(crop, "se", 0.1, 0.1, aspect=1.0, canvas_aspect=16.0 / 9.0)
        frac_ratio = resized[2] / resized[3]
        self.assertAlmostEqual(frac_ratio, 9.0 / 16.0, places=5)

    def test_resize_locked_aspect_edge_handle_grows_symmetrically(self):
        crop = [0.3, 0.3, 0.4, 0.2]
        resized = framing.resize_crop(crop, "s", 0.0, 0.1, aspect=2.0, canvas_aspect=1.0)
        self.assertAlmostEqual(resized[1], 0.3)  # top (opposite edge) unchanged
        # Width grew/shrank symmetrically about the original horizontal center (0.5).
        new_center = resized[0] + resized[2] / 2.0
        self.assertAlmostEqual(new_center, 0.5, places=5)
        self.assertAlmostEqual(resized[2] / resized[3], 2.0, places=5)

    def test_resize_locked_aspect_clamps_to_canvas_bounds(self):
        crop = [0.0, 0.4, 0.3, 0.2]
        # Drag west edge far past the canvas boundary -- box must not exceed [0, 1].
        resized = framing.resize_crop(crop, "w", -5.0, 0.0, aspect=1.0, canvas_aspect=1.0)
        self.assertGreaterEqual(resized[0], -1e-9)
        self.assertGreaterEqual(resized[1], -1e-9)
        self.assertLessEqual(resized[0] + resized[2], 1.0 + 1e-9)
        self.assertLessEqual(resized[1] + resized[3], 1.0 + 1e-9)


def _find_marker(img):
    """Locate the centroid of a pure-red marker painted onto an otherwise dark image."""
    arr = np.array(img)
    mask = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] < 50)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class DisplayPointToSourceTests(unittest.TestCase):
    def test_identity_framing_is_passthrough(self):
        sx, sy = framing.display_point_to_source(framing.default_framing(), 400, 300, 0.3, 0.7)
        self.assertAlmostEqual(sx, 0.3, places=6)
        self.assertAlmostEqual(sy, 0.7, places=6)

    def test_pure_crop_no_rotation_is_linear_remap(self):
        f = {"crop": [0.25, 0.1, 0.5, 0.6]}
        sx, sy = framing.display_point_to_source(f, 400, 300, 0.5, 0.5)
        self.assertAlmostEqual(sx, 0.25 + 0.5 * 0.5, places=6)
        self.assertAlmostEqual(sy, 0.1 + 0.5 * 0.6, places=6)

    def test_flip_horizontal_mirrors_back(self):
        f = {"flip_h": True}
        sx, sy = framing.display_point_to_source(f, 400, 300, 0.2, 0.4)
        self.assertAlmostEqual(sx, 0.8, places=6)
        self.assertAlmostEqual(sy, 0.4, places=6)

    def test_inverts_apply_framing_for_marked_point(self):
        # Empirical check (no assumed rotation sign): paint a marker at a known source
        # pixel, run it through apply_framing with a nonzero angle/flip/crop, locate where
        # the marker actually landed on screen, and confirm display_point_to_source maps
        # that screen location back to within ~1px of the original source pixel.
        width, height = 400, 300
        src_x, src_y = 220, 90
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[src_y - 1 : src_y + 2, src_x - 1 : src_x + 2] = (255, 0, 0)
        img = Image.fromarray(arr)

        f = {"angle": -20.0, "flip_h": False, "flip_v": False, "crop": [0.1, 0.15, 0.6, 0.5]}
        out = framing.apply_framing(img, f)
        marker = _find_marker(out)
        self.assertIsNotNone(marker, "marker should remain inside the crop for this fixture")
        disp_x, disp_y = marker
        out_w, out_h = out.size

        sx, sy = framing.display_point_to_source(f, width, height, disp_x / out_w, disp_y / out_h)
        self.assertAlmostEqual(sx * width, src_x, delta=1.5)
        self.assertAlmostEqual(sy * height, src_y, delta=1.5)


class RenderDisplayTileTests(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 400, 300
        rng = np.random.default_rng(0)
        self.full_arr = (rng.random((self.height, self.width, 3)) * 255).astype(np.uint8)
        self.full_img = Image.fromarray(self.full_arr)

    def _check(self, f, crop_box_px):
        """Crop a generous source window, render it as a display tile for crop_box_px (in
        the *cropped* output's own pixel space), and compare against directly cropping the
        ground-truth apply_framing(full_img) at the same box -- they must match exactly."""
        ground = framing.apply_framing(self.full_img, f)
        gw, gh = ground.size
        cx0, cy0, cx1, cy1 = crop_box_px
        cx1, cy1 = min(cx1, gw), min(cy1, gh)
        target = ground.crop((cx0, cy0, cx1, cy1))

        norm_f = framing.normalize_framing(f)
        corners = [(cx0 / gw, cy0 / gh), (cx1 / gw, cy0 / gh), (cx0 / gw, cy1 / gh), (cx1 / gw, cy1 / gh)]
        src_pts = [
            framing.display_point_to_source(f, self.width, self.height, dx, dy) for dx, dy in corners
        ]
        pad = 40
        sx0 = max(0, int(np.floor(min(p[0] for p in src_pts) * self.width - pad)))
        sy0 = max(0, int(np.floor(min(p[1] for p in src_pts) * self.height - pad)))
        sx1 = min(self.width, int(np.ceil(max(p[0] for p in src_pts) * self.width + pad)))
        sy1 = min(self.height, int(np.ceil(max(p[1] for p in src_pts) * self.height + pad)))
        crop = self.full_img.crop((sx0, sy0, sx1, sy1))

        origin_x = round(norm_f["crop"][0] * self.width)
        origin_y = round(norm_f["crop"][1] * self.height)
        target_rect_px = (origin_x + cx0, origin_y + cy0, origin_x + cx1, origin_y + cy1)

        tile = framing.render_display_tile(crop, f, self.width, self.height, (sx0, sy0), target_rect_px)
        self.assertIsNotNone(tile)
        np.testing.assert_array_equal(np.array(tile), np.array(target))

    def test_crop_only_no_rotation(self):
        self._check({"crop": [0.1, 0.1, 0.7, 0.7]}, (50, 40, 150, 130))

    def test_rotation_and_crop(self):
        self._check({"angle": 18.0, "crop": [0.05, 0.1, 0.7, 0.6]}, (40, 30, 160, 120))

    def test_rotation_flip_and_crop(self):
        self._check({"angle": -25.0, "flip_h": True, "crop": [0.1, 0.05, 0.6, 0.7]}, (30, 20, 140, 150))

    def test_flip_vertical_full_frame(self):
        self._check({"angle": 10.0, "flip_v": True, "crop": [0.0, 0.0, 1.0, 1.0]}, (10, 10, 200, 180))

    def test_both_flips_no_rotation(self):
        self._check({"flip_h": True, "flip_v": True, "crop": [0.2, 0.2, 0.5, 0.5]}, (0, 0, 100, 90))

    def test_rotation_and_both_flips(self):
        self._check(
            {"angle": 35.0, "flip_h": True, "flip_v": True, "crop": [0.15, 0.1, 0.6, 0.55]},
            (5, 5, 130, 110),
        )


if __name__ == "__main__":
    unittest.main()
