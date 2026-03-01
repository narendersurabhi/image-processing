import unittest

try:
    import numpy as np
    from portrait_enhancer.core.segmentation import (
        _build_model_masks_from_output,
        _normalize_detector_faces,
        FacialHairSegmenter,
        FaceSegmenter,
        MASK_KEYS,
        ModelFaceSegmenter,
        _cleanup_region_mask,
        _expanded_face_crop,
        _mask_from_class_probs,
        _merge_model_views,
        _paste_crop_masks,
        _region_zones,
        _subjects_background_masks,
        _softmax_last_axis,
    )
    from portrait_enhancer.core.utils import refine_mask_edges

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/opencv not installed")
class SegmentationSmokeTests(unittest.TestCase):
    def test_segment_returns_all_masks(self):
        seg = FaceSegmenter(backend="heuristic")
        img = np.zeros((64, 64, 3), dtype=np.float32)
        faces = seg.list_faces(img)
        masks = seg.segment(img)

        self.assertIsInstance(faces, list)
        self.assertEqual(set(masks.keys()), set(MASK_KEYS))
        for key in MASK_KEYS:
            self.assertEqual(masks[key].shape, (64, 64))
            self.assertEqual(masks[key].dtype, np.float32)

    def test_cleanup_region_mask_removes_small_islands(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:36, 20:36] = 1.0
        mask[2:4, 2:4] = 1.0

        cleaned = _cleanup_region_mask(mask, min_area=32, close_size=1, open_size=1, keep_largest=1)

        self.assertGreater(cleaned[24:32, 24:32].mean(), 0.9)
        self.assertEqual(float(cleaned[2:4, 2:4].sum()), 0.0)

    def test_region_zones_focus_features_inside_face_box(self):
        zones = _region_zones((120, 120), (30, 20, 60, 80))

        self.assertGreater(zones["eyes"][45, 50], 0.0)
        self.assertGreater(zones["lips"][80, 60], 0.0)
        self.assertGreater(zones["hair"][18, 60], 0.0)
        self.assertLess(zones["lips"][20, 60], 0.1)

    def test_class_prob_masks_are_soft_and_aggregated(self):
        scores = np.array([[[0.0, 3.0, 1.0], [0.0, 0.5, 2.5]]], dtype=np.float32)
        probs = _softmax_last_axis(scores)
        mask = _mask_from_class_probs(probs, {1, 2})

        self.assertEqual(mask.shape, (1, 2))
        self.assertGreater(mask[0, 0], 0.9)
        self.assertGreater(mask[0, 1], 0.9)
        self.assertLessEqual(float(mask.max()), 1.0)

    def test_refine_mask_edges_preserves_shape_and_bounds(self):
        guide = np.zeros((32, 32, 3), dtype=np.float32)
        guide[:, 16:] = 1.0
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[:, 12:20] = 1.0

        refined = refine_mask_edges(mask, guide, radius=4, eps=1e-3)

        self.assertEqual(refined.shape, mask.shape)
        self.assertEqual(refined.dtype, np.float32)
        self.assertGreaterEqual(float(refined.min()), 0.0)
        self.assertLessEqual(float(refined.max()), 1.0)

    def test_expanded_face_crop_stays_in_bounds(self):
        crop = _expanded_face_crop((20, 15, 30, 40), (100, 120), x_scale=1.1, y_scale=1.25, y_up_scale=0.85)
        x1, y1, x2, y2 = crop

        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 120)
        self.assertLessEqual(y2, 100)
        self.assertGreater(x2 - x1, 30)
        self.assertGreater(y2 - y1, 40)

    def test_paste_crop_masks_places_crop_back_into_full_canvas(self):
        crop_masks = {"face": np.ones((10, 12), dtype=np.float32)}
        pasted = _paste_crop_masks(crop_masks, (5, 7, 17, 17), (32, 32), (10, 12))

        self.assertEqual(pasted["face"].shape, (32, 32))
        self.assertEqual(float(pasted["face"][7:17, 5:17].mean()), 1.0)
        self.assertEqual(float(pasted["face"][:5, :5].sum()), 0.0)

    def test_build_model_masks_from_output_uses_probs_when_available(self):
        probs = np.zeros((4, 4, 18), dtype=np.float32)
        probs[:, :, 1] = 0.8
        probs[:, :, 17] = 0.2

        masks = _build_model_masks_from_output(np.zeros((4, 4), dtype=np.uint8), probs)

        self.assertGreater(float(masks["skin"].mean()), float(masks["hair"].mean()))
        self.assertIn("brows", masks)
        self.assertIn("facial_hair", masks)

    def test_merge_model_views_prefers_crop_in_face_region(self):
        full_masks = {key: np.full((32, 32), 0.2, dtype=np.float32) for key in MASK_KEYS}
        crop_masks = {key: np.full((32, 32), 0.8, dtype=np.float32) for key in MASK_KEYS}

        merged = _merge_model_views(full_masks, crop_masks, face_box=(8, 8, 16, 16), shape_hw=(32, 32))

        self.assertGreater(float(merged["face"][16, 16]), float(merged["face"][2, 2]))
        self.assertGreater(float(merged["eyes"][14, 12]), 0.5)

    def test_normalize_detector_faces_clamps_and_sorts(self):
        detections = np.array(
            [
                [5.2, 6.1, 10.4, 12.3],
                [1.0, 1.0, 20.0, 18.0],
            ],
            dtype=np.float32,
        )

        faces = _normalize_detector_faces(detections, (32, 32))

        self.assertEqual(faces[0], (1, 1, 20, 18))
        self.assertEqual(faces[1], (5, 6, 11, 13))

    def test_subjects_background_masks_partition_scene(self):
        masks = {key: np.zeros((64, 64), dtype=np.float32) for key in MASK_KEYS}
        masks["face"][10:22, 20:32] = 1.0
        masks["skin"][12:24, 18:34] = 1.0
        masks["hair"][6:18, 18:34] = 1.0

        subjects, background = _subjects_background_masks((64, 64), [(18, 8, 16, 18)], masks)

        self.assertEqual(subjects.shape, (64, 64))
        self.assertEqual(background.shape, (64, 64))
        self.assertGreater(float(subjects[24:52, 14:38].mean()), 0.1)
        self.assertLess(float(background[24:52, 14:38].mean()), 0.9)

    def test_draw_region_indices_builds_mask_from_landmarks(self):
        seg = ModelFaceSegmenter.__new__(ModelFaceSegmenter)

        class LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        landmarks = [
            LM(0.10, 0.10),
            LM(0.80, 0.10),
            LM(0.80, 0.80),
            LM(0.10, 0.80),
        ]
        mask = seg._draw_region_indices(landmarks, (0, 1, 2, 3), 32, 32)

        self.assertEqual(mask.shape, (32, 32))
        self.assertGreater(float(mask[16, 16]), 0.9)

    def test_guides_from_landmarks_extracts_expression_points(self):
        seg = ModelFaceSegmenter.__new__(ModelFaceSegmenter)

        class LM:
            def __init__(self, x=0.5, y=0.5):
                self.x = x
                self.y = y

        landmarks = [LM() for _ in range(500)]
        landmarks[61] = LM(0.30, 0.62)
        landmarks[291] = LM(0.70, 0.61)
        landmarks[13] = LM(0.50, 0.58)
        landmarks[14] = LM(0.50, 0.66)
        landmarks[159] = LM(0.38, 0.38)
        landmarks[145] = LM(0.38, 0.43)
        landmarks[386] = LM(0.62, 0.38)
        landmarks[374] = LM(0.62, 0.43)
        landmarks[105] = LM(0.36, 0.30)
        landmarks[334] = LM(0.64, 0.30)

        guides = seg._guides_from_landmarks(landmarks, 100, 120)

        self.assertIn("mouth_left", guides)
        self.assertIn("mouth_right", guides)
        self.assertIn("left_eye_upper", guides)
        self.assertIn("right_eye_lower", guides)
        self.assertIn("left_brow", guides)
        self.assertIn("right_brow", guides)
        self.assertAlmostEqual(guides["mouth_left"][0], 30.0, places=3)
        self.assertAlmostEqual(guides["mouth_lower"][1], 79.2, places=3)

    def test_facial_hair_segmenter_decodes_yolov8_seg_outputs(self):
        seg = FacialHairSegmenter.__new__(FacialHairSegmenter)
        seg.execution_provider = "cpu"
        seg.backend = "onnx:cpu"
        seg._input_hw = (8, 8)

        preds = np.zeros((1, 7, 1), dtype=np.float32)
        preds[0, 0, 0] = 4.0
        preds[0, 1, 0] = 4.0
        preds[0, 2, 0] = 6.0
        preds[0, 3, 0] = 6.0
        preds[0, 4, 0] = 0.95
        preds[0, 5, 0] = 5.0
        preds[0, 6, 0] = -5.0

        protos = np.zeros((1, 2, 4, 4), dtype=np.float32)
        protos[0, 0, :, :] = 1.0

        mask = seg._to_mask([preds, protos])

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (4, 4))
        self.assertGreater(float(mask.mean()), 0.1)
        self.assertIn("yolov8-seg", seg.backend)


if __name__ == "__main__":
    unittest.main()
