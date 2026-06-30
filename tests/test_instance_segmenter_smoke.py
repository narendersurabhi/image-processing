import os
import unittest

try:
    import numpy as np
    from portrait_enhancer.core.segmentation import (
        FaceSegmenter,
        InstanceSegmenter,
        PersonInstanceSegmenter,
        _watershed_person_labels,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/opencv not installed")
class InstanceSegmenterFallbackTests(unittest.TestCase):
    """Milestone 1 contract: SAM is an optional upgrade. When unavailable (model absent OR
    force-disabled), InstanceSegmenter must report unavailable and never break the Person mask --
    _person_mask has to fall through to the watershed split exactly as before. conftest.py sets
    PORTRAIT_DISABLE_SAM so this holds deterministically whether or not the models are installed."""

    def setUp(self):
        # Pin the disable switch on for this test class so it's deterministic even if a future
        # change to conftest removes the session default; restore afterward.
        self._saved = os.environ.get("PORTRAIT_DISABLE_SAM")
        os.environ["PORTRAIT_DISABLE_SAM"] = "1"

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("PORTRAIT_DISABLE_SAM", None)
        else:
            os.environ["PORTRAIT_DISABLE_SAM"] = self._saved

    def test_unavailable_reports_reason(self):
        seg = InstanceSegmenter()
        # Disabled (or model/onnxruntime absent) -- either way, not available, with a reason.
        self.assertFalse(seg.available)
        self.assertTrue(seg.reason_unavailable)

    def test_instance_masks_returns_none_when_unavailable(self):
        seg = InstanceSegmenter()
        img = np.zeros((64, 96, 3), dtype=np.float32)
        faces = [(10, 10, 20, 24), (60, 12, 18, 22)]
        subjects = np.ones((64, 96), dtype=np.float32)
        self.assertIsNone(seg.instance_masks(img, faces, subjects))

    def test_click_mask_returns_none_when_unavailable(self):
        # AI Select's primitive: with SAM disabled/absent, a click must produce nothing rather
        # than raise -- the caller (main_window's _on_mask_click_finished) treats None as "show a
        # friendly status message, leave the mask untouched."
        seg = InstanceSegmenter()
        img = np.zeros((64, 96, 3), dtype=np.float32)
        self.assertIsNone(seg.click_mask(img, (48.0, 32.0)))

    def test_person_instance_segmenter_disabled_reports_reason(self):
        saved = os.environ.get("PORTRAIT_DISABLE_MASKDINO")
        os.environ["PORTRAIT_DISABLE_MASKDINO"] = "1"
        try:
            seg = PersonInstanceSegmenter()
            self.assertFalse(seg.available)
            self.assertIn("PORTRAIT_DISABLE_MASKDINO", seg.reason_unavailable)
            self.assertIsNone(seg.person_masks(np.zeros((16, 16, 3), dtype=np.float32), [(1, 1, 4, 4)]))
        finally:
            if saved is None:
                os.environ.pop("PORTRAIT_DISABLE_MASKDINO", None)
            else:
                os.environ["PORTRAIT_DISABLE_MASKDINO"] = saved

    def test_person_mask_falls_back_to_watershed(self):
        # With SAM unavailable, _person_mask must equal the watershed split for a clean 2-blob
        # subjects mask -- i.e. SAM being absent changes nothing.
        import cv2

        seg = FaceSegmenter(backend="heuristic")
        self.assertFalse(getattr(seg._instance, "available", False))

        h, w = 200, 320
        img = np.zeros((h, w, 3), dtype=np.float32)
        subjects = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(subjects, (90, 110), (60, 80), 0, 0, 360, 1.0, -1)
        cv2.ellipse(subjects, (230, 110), (60, 80), 0, 0, 360, 1.0, -1)
        faces = [(60, 50, 60, 60), (200, 50, 60, 60)]

        mask = seg._person_mask(img, faces, 0, subjects)
        self.assertEqual(seg.person_backend, "watershed")
        self.assertEqual(mask.shape, (h, w))
        # The watershed split assigns the left blob to face 0 and almost nothing on the right.
        self.assertGreater(float(mask[:, :w // 2].mean()), float(mask[:, w // 2:].mean()))


if __name__ == "__main__":
    unittest.main()
