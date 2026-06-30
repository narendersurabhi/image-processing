import unittest

try:
    import numpy as np
    from portrait_enhancer.ui_qt.main_window import PortraitEnhancerQtWindow, _masks_usable, MASK_ORDER

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "PySide6/numpy not installed")
class MasksUsableTests(unittest.TestCase):
    """Guards the fix for: a face profile injected from a saved collection override carries the
    per-face params but an EMPTY mask dict (masks aren't persisted). _restore_face_profile used
    to restore that empty dict and report success, blanking every layer's Mask View for that
    face until a fresh segmentation. _masks_usable is the gate that now forces re-segmentation."""

    def test_empty_dict_is_not_usable(self):
        # The exact shape _apply_settings_payload injects for non-active faces.
        self.assertFalse(_masks_usable({}))

    def test_none_is_not_usable(self):
        self.assertFalse(_masks_usable(None))

    def test_dict_without_layer_keys_is_not_usable(self):
        # Has entries, but none of them are real mask layers -- still nothing to display.
        self.assertFalse(_masks_usable({"_meta": 1, "note": "x"}))

    def test_dict_with_real_layer_arrays_is_usable(self):
        masks = {key: np.zeros((8, 8), dtype=np.float32) for key in MASK_ORDER}
        self.assertTrue(_masks_usable(masks))

    def test_partial_but_present_layer_is_usable(self):
        # Even one genuine layer key means there's something to restore/show.
        self.assertTrue(_masks_usable({"face": np.ones((4, 4), dtype=np.float32)}))

    def test_multi_face_profile_without_mask_face_tag_is_not_restored(self):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        window._detected_faces = [(0, 0, 10, 10), (20, 0, 10, 10)]
        window._sliders = {layer: {} for layer in MASK_ORDER}
        window._slider_value_labels = {layer: {} for layer in MASK_ORDER}
        window._layer_options = {layer: {} for layer in MASK_ORDER}
        window._mask_adjustments = {}
        window._mask_face_index = None
        masks = {key: np.ones((4, 4), dtype=np.float32) for key in MASK_ORDER}
        window._face_profiles = {
            0: {
                "selective_params": {layer: {} for layer in MASK_ORDER},
                "layer_options": {layer: {} for layer in MASK_ORDER},
                "layer_order": list(MASK_ORDER),
                "mask_adjustments": {},
                "preview_masks": masks,
                "full_masks": masks,
                "auto_preview_masks": masks,
                "auto_full_masks": masks,
            }
        }

        restored = window._restore_face_profile(0)

        self.assertFalse(restored)
        self.assertIsNone(getattr(window, "preview_masks", None))

    def test_profile_with_mismatched_mask_face_tag_is_not_restored(self):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        window._detected_faces = [(0, 0, 10, 10), (20, 0, 10, 10)]
        window._sliders = {layer: {} for layer in MASK_ORDER}
        window._slider_value_labels = {layer: {} for layer in MASK_ORDER}
        window._layer_options = {layer: {} for layer in MASK_ORDER}
        window._mask_adjustments = {}
        window._mask_face_index = None
        masks = {key: np.ones((4, 4), dtype=np.float32) for key in MASK_ORDER}
        window._face_profiles = {
            0: {
                "selective_params": {layer: {} for layer in MASK_ORDER},
                "layer_options": {layer: {} for layer in MASK_ORDER},
                "layer_order": list(MASK_ORDER),
                "mask_adjustments": {},
                "preview_masks": masks,
                "full_masks": masks,
                "auto_preview_masks": masks,
                "auto_full_masks": masks,
                "mask_face_index": 1,
            }
        }

        restored = window._restore_face_profile(0)

        self.assertFalse(restored)
        self.assertIsNone(getattr(window, "preview_masks", None))

    def test_profile_with_matching_mask_face_tag_restores(self):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        window._detected_faces = [(0, 0, 10, 10), (20, 0, 10, 10)]
        window._sliders = {layer: {} for layer in MASK_ORDER}
        window._slider_value_labels = {layer: {} for layer in MASK_ORDER}
        window._layer_options = {layer: {} for layer in MASK_ORDER}
        window._mask_adjustments = {}
        window._mask_face_index = None
        masks = {key: np.ones((4, 4), dtype=np.float32) for key in MASK_ORDER}
        window._face_profiles = {
            0: {
                "selective_params": {layer: {} for layer in MASK_ORDER},
                "layer_options": {layer: {} for layer in MASK_ORDER},
                "layer_order": list(MASK_ORDER),
                "mask_adjustments": {},
                "preview_masks": masks,
                "full_masks": masks,
                "auto_preview_masks": masks,
                "auto_full_masks": masks,
                "mask_face_index": 0,
            }
        }

        restored = window._restore_face_profile(0)

        self.assertTrue(restored)
        self.assertEqual(window._mask_face_index, 0)
        self.assertTrue(_masks_usable(window.preview_masks))

    def test_mask_review_stats_flags_empty_mask(self):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        window.preview_masks = {"eyes": np.zeros((24, 24), dtype=np.float32)}
        window._auto_preview_masks = {"eyes": np.zeros((24, 24), dtype=np.float32)}

        status, _detail, severity = window._mask_review_layer_stats("eyes", window.preview_masks["eyes"])

        self.assertEqual(status, "Empty")
        self.assertEqual(severity, "bad")

    def test_mask_review_stats_flags_edited_mask(self):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        mask = np.zeros((24, 24), dtype=np.float32)
        mask[6:18, 6:18] = 1.0
        window.preview_masks = {"hair": mask}
        window._auto_preview_masks = {"hair": np.zeros((24, 24), dtype=np.float32)}

        status, detail, severity = window._mask_review_layer_stats("hair", mask)

        self.assertEqual(status, "Edited")
        self.assertEqual(severity, "edited")
        self.assertIn("automatic baseline", detail)

    def test_mask_review_stats_flags_broad_hair_relative_to_face(self):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        face = np.zeros((48, 48), dtype=np.float32)
        face[18:30, 18:30] = 1.0
        hair = np.zeros((48, 48), dtype=np.float32)
        hair[4:44, 4:44] = 1.0
        window.preview_masks = {"face": face, "hair": hair}
        window._auto_preview_masks = {"hair": hair.copy()}

        status, detail, severity = window._mask_review_layer_stats("hair", hair)

        self.assertEqual(status, "Broad")
        self.assertEqual(severity, "warn")
        self.assertIn("face-relative", detail)


if __name__ == "__main__":
    unittest.main()
