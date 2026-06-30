import unittest

try:
    import numpy as np

    from portrait_enhancer.core import culling

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class FakeBlinkSegmenter:
    def __init__(self, gap):
        self.gap = float(gap)

    def list_faces(self, _img):
        return [(20, 20, 50, 50)]

    def face_guides(self, _img, face_box=None):
        y = 20.0
        return {
            "left_eye_upper": (30.0, y),
            "left_eye_lower": (30.0, y + self.gap),
            "right_eye_upper": (45.0, y),
            "right_eye_lower": (45.0, y + self.gap),
        }


@unittest.skipUnless(HAS_DEPS, "numpy/opencv not installed")
class CullingSmokeTests(unittest.TestCase):
    def test_blink_analysis_classifies_open_closed_and_borderline_unknown(self):
        img = np.zeros((120, 120, 3), dtype=np.float32)

        self.assertEqual(culling.analyze_face_blinks(FakeBlinkSegmenter(gap=4.0), img), (0, 1))
        self.assertEqual(culling.analyze_face_blinks(FakeBlinkSegmenter(gap=0.5), img), (1, 1))
        self.assertEqual(culling.analyze_face_blinks(FakeBlinkSegmenter(gap=1.5), img), (0, 0))

    def test_pick_keeper_prefers_known_open_eyes_over_unknown_even_if_less_sharp(self):
        quality = {
            "open": {"blur": 50.0, "blinks": 0, "eyes_open": True},
            "unknown": {"blur": 500.0, "blinks": None, "eyes_open": None, "blink_checked": True},
        }

        self.assertEqual(culling.pick_keeper(["unknown", "open"], quality), "open")

    def test_pick_keeper_prefers_fewer_blinks_before_sharpness(self):
        quality = {
            "sharp_blink": {"blur": 500.0, "blinks": 1, "eyes_open": False},
            "soft_open": {"blur": 50.0, "blinks": 0, "eyes_open": True},
        }

        self.assertEqual(culling.pick_keeper(["sharp_blink", "soft_open"], quality), "soft_open")

    def test_analyze_faces_detailed_maps_guides_back_to_original_coordinates(self):
        img = np.zeros((120, 120, 3), dtype=np.float32)
        seg = FakeBlinkSegmenter(gap=4.0)

        details = culling.analyze_faces_detailed(seg, img)

        self.assertEqual(len(details), 1)
        entry = details[0]
        self.assertEqual(entry["box"], (20, 20, 50, 50))
        self.assertIsNotNone(entry["guides"])
        self.assertEqual(entry["open"], True)  # gap=4.0 is a confidently-open ratio

        # Crop offset (ox, oy) is (0, 0) here (x - mx = 20 - 20 = 0), and the fake's face is
        # upscaled by 512/50 before landmarking -- guide points must be divided back down by
        # that scale (and offset) to land in img's own coordinates.
        scale = 512.0 / 50.0
        left_upper = entry["guides"]["left_eye_upper"]
        self.assertAlmostEqual(left_upper[0], 30.0 / scale, places=4)
        self.assertAlmostEqual(left_upper[1], 20.0 / scale, places=4)

    def test_analyze_face_blinks_matches_aggregate_of_detailed_results(self):
        img = np.zeros((120, 120, 3), dtype=np.float32)
        for gap in (4.0, 0.5, 1.5):
            seg = FakeBlinkSegmenter(gap=gap)
            blinks, analyzed = culling.analyze_face_blinks(seg, img)
            details = culling.analyze_faces_detailed(seg, img)
            classified = [d for d in details if d["open"] is not None]
            self.assertEqual(analyzed, len(classified))
            self.assertEqual(blinks, sum(1 for d in classified if d["open"] is False))


if __name__ == "__main__":
    unittest.main()
