import unittest

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


if __name__ == "__main__":
    unittest.main()
