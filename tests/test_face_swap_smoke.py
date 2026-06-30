import math
import unittest

try:
    import numpy as np
    import cv2

    from portrait_enhancer.core import face_swap

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def _transform_point(p, theta, s, t):
    c, sn = math.cos(theta), math.sin(theta)
    x = s * (c * p[0] - sn * p[1]) + t[0]
    y = s * (sn * p[0] + c * p[1]) + t[1]
    return (x, y)


def _eye_guides(left_center, right_center, half_gap=3.0):
    return {
        "left_eye_upper": (left_center[0], left_center[1] - half_gap),
        "left_eye_lower": (left_center[0], left_center[1] + half_gap),
        "right_eye_upper": (right_center[0], right_center[1] - half_gap),
        "right_eye_lower": (right_center[0], right_center[1] + half_gap),
    }


def _make_face(eye_state, size=300):
    img = np.full((size, size, 3), 0.55, dtype=np.float32)
    img[:, :, 1] *= 0.85
    img[:, :, 2] *= 0.7
    left_c, right_c = (110, 150), (190, 150)
    for c in (left_c, right_c):
        if eye_state == "open":
            cv2.circle(img, c, 14, (0.9, 0.9, 0.95), -1)
            cv2.circle(img, c, 6, (0.05, 0.05, 0.05), -1)
        else:
            cv2.ellipse(img, c, (14, 3), 0, 0, 360, (0.45, 0.32, 0.25), -1)
    return img, _eye_guides(left_c, right_c)


@unittest.skipUnless(HAS_DEPS, "numpy/opencv not installed")
class FaceSwapSmokeTests(unittest.TestCase):
    def test_compute_eye_alignment_recovers_known_similarity_transform(self):
        d_left, d_right = (100.0, 200.0), (180.0, 205.0)
        theta, scale, translation = math.radians(8.0), 1.3, (50.0, 30.0)
        t_left = _transform_point(d_left, theta, scale, translation)
        t_right = _transform_point(d_right, theta, scale, translation)

        donor_guides = _eye_guides(d_left, d_right)
        target_guides = _eye_guides(t_left, t_right)

        M = face_swap.compute_eye_alignment(donor_guides, target_guides)
        self.assertIsNotNone(M)

        def apply(M, p):
            return (M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2])

        pred_left = apply(M, d_left)
        pred_right = apply(M, d_right)
        self.assertAlmostEqual(pred_left[0], t_left[0], places=2)
        self.assertAlmostEqual(pred_left[1], t_left[1], places=2)
        self.assertAlmostEqual(pred_right[0], t_right[0], places=2)
        self.assertAlmostEqual(pred_right[1], t_right[1], places=2)

    def test_compute_eye_alignment_returns_none_when_eye_missing(self):
        donor_guides = {"left_eye_upper": (1.0, 2.0), "left_eye_lower": (1.0, 4.0)}  # no right eye
        target_guides = _eye_guides((10, 10), (20, 10))
        self.assertIsNone(face_swap.compute_eye_alignment(donor_guides, target_guides))

    def test_match_face_by_position_accepts_near_and_rejects_far(self):
        target_box = (100, 100, 80, 80)
        near = {"box": (98, 102, 82, 78)}
        far = {"box": (500, 500, 80, 80)}

        self.assertEqual(face_swap.match_face_by_position(target_box, [near, far]), near)
        self.assertIsNone(face_swap.match_face_by_position(target_box, [far]))
        self.assertIsNone(face_swap.match_face_by_position(target_box, []))

    def test_swap_eyes_replaces_closed_eyes_with_open_donor_eyes(self):
        target_img, target_guides = _make_face("closed")
        donor_img, donor_guides = _make_face("open")
        target_box = (60, 60, 180, 200)

        result, sides = face_swap.swap_eyes(target_img, target_guides, target_box, donor_img, donor_guides)

        self.assertIsNotNone(result)
        self.assertEqual(set(sides), {"left", "right"})
        self.assertEqual(result.shape, target_img.shape)

        # Sclera ring (just off dead-center, so it isn't sampling the pupil) should now read
        # bright like the donor's open eye, not the target's closed-lid color.
        probe = (120, 150)
        before = float(target_img[probe[1], probe[0]].mean())
        after = float(result[probe[1], probe[0]].mean())
        self.assertGreater(after, before + 0.25)

        # A region far from both eyes should be essentially untouched.
        far_probe = (20, 20)
        self.assertLess(abs(float(target_img[far_probe[1], far_probe[0]].mean()) - float(result[far_probe[1], far_probe[0]].mean())), 0.01)

    def test_swap_eyes_returns_none_without_usable_guides(self):
        target_img, _ = _make_face("closed")
        donor_img, _ = _make_face("open")
        result, sides = face_swap.swap_eyes(target_img, {}, (60, 60, 180, 200), donor_img, {})
        self.assertIsNone(result)
        self.assertEqual(sides, [])


if __name__ == "__main__":
    unittest.main()
