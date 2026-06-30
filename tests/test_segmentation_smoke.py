import os
import unittest

try:
    import cv2
    import numpy as np
    from portrait_enhancer.core.segmentation import (
        _build_model_masks_from_output,
        _normalize_detector_faces,
        BackgroundRemovalSegmenter,
        FacialHairSegmenter,
        FaceSegmenter,
        MASK_KEYS,
        ModelFaceSegmenter,
        PersonInstanceSegmenter,
        _cleanup_region_mask,
        _apply_person_identity_gate_to_face_parts,
        _expanded_face_crop,
        _gate_sam_person_mask_to_watershed,
        _mask_from_class_probs,
        _merge_model_views,
        _paste_crop_masks,
        _refine_face_part_masks,
        _region_zones,
        _exclude_other_face_regions,
        _keep_anchor_connected_mask,
        _subjects_background_masks,
        _softmax_last_axis,
        _suppress_cross_person_spill,
        _validate_person_split,
        _validate_sam_face_mask,
        _watershed_person_labels,
        InstanceSegmenter,
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

    def test_refine_face_part_masks_tightens_eyes_and_lips_to_landmarks(self):
        h, w = 120, 160
        face_box = (50, 25, 60, 78)
        zones = _region_zones((h, w), face_box)
        masks = {key: np.zeros((h, w), dtype=np.float32) for key in MASK_KEYS}
        masks["face"] = zones["face"]
        masks["skin"] = zones["skin"]
        masks["eyes"] = zones["eyes"]  # deliberately broad parser/fallback blob
        masks["lips"] = zones["lips"]
        masks["hair"] = np.zeros((h, w), dtype=np.float32)
        masks["brows"] = np.zeros((h, w), dtype=np.float32)

        guides = {
            "left_eye_center": (70.0, 52.0),
            "left_eye_upper": (70.0, 50.0),
            "left_eye_lower": (70.0, 54.0),
            "right_eye_center": (91.0, 52.0),
            "right_eye_upper": (91.0, 50.0),
            "right_eye_lower": (91.0, 54.0),
            "mouth_left": (70.0, 82.0),
            "mouth_right": (91.0, 82.0),
            "mouth_upper": (80.0, 79.0),
            "mouth_lower": (80.0, 87.0),
            "mouth_center": (80.0, 82.5),
        }

        refined = _refine_face_part_masks(masks, np.zeros((h, w, 3), dtype=np.float32), face_box=face_box, guides=guides)

        self.assertGreater(float(refined["eyes"][52, 70]), 0.5)
        self.assertGreater(float(refined["eyes"][52, 91]), 0.5)
        self.assertLess(float(refined["eyes"][52, 54]), 0.05)
        self.assertGreater(float(refined["lips"][83, 80]), 0.5)
        self.assertLess(float(refined["lips"][72, 80]), 0.05)

    def test_refine_face_part_masks_recovers_sparse_skin_and_dark_hair(self):
        h, w = 120, 160
        face_box = (50, 25, 60, 78)
        zones = _region_zones((h, w), face_box)
        img = np.full((h, w, 3), 0.72, dtype=np.float32)
        img[zones["hair"] > 0.3] = 0.04
        masks = {key: np.zeros((h, w), dtype=np.float32) for key in MASK_KEYS}
        masks["face"] = zones["face"]
        masks["skin"][70:74, 76:84] = 1.0  # sparse parser skin
        masks["eyes"] = np.zeros((h, w), dtype=np.float32)
        masks["lips"] = np.zeros((h, w), dtype=np.float32)
        masks["hair"] = np.zeros((h, w), dtype=np.float32)
        masks["brows"] = np.zeros((h, w), dtype=np.float32)

        refined = _refine_face_part_masks(masks, img, face_box=face_box, guides=None)

        self.assertGreater(float(refined["skin"][60:82, 68:94].mean()), 0.25)
        self.assertGreater(float(refined["hair"][24:45, 58:102].mean()), 0.20)
        self.assertLess(float(refined["hair"][60:82, 68:94].mean()), 0.18)

    def test_landmark_masks_on_crop_maps_masks_and_guides_to_full_frame(self):
        seg = ModelFaceSegmenter.__new__(ModelFaceSegmenter)
        full_h, full_w = 120, 180
        face_box = (72, 46, 20, 24)
        image = np.zeros((full_h, full_w, 3), dtype=np.float32)
        calls = []

        def fake_landmark_masks(crop_img, face_box=None):
            calls.append((crop_img.shape[:2], face_box))
            ch, cw = crop_img.shape[:2]
            local_x, local_y, local_w, local_h = face_box
            face = np.zeros((ch, cw), dtype=np.float32)
            eyes = np.zeros((ch, cw), dtype=np.float32)
            lips = np.zeros((ch, cw), dtype=np.float32)
            x1 = int(local_x)
            y1 = int(local_y)
            x2 = min(cw, int(local_x + local_w))
            y2 = min(ch, int(local_y + local_h))
            face[y1:y2, x1:x2] = 1.0
            eyes[int(local_y + local_h * 0.35), int(local_x + local_w * 0.5)] = 1.0
            lips[int(local_y + local_h * 0.75), int(local_x + local_w * 0.5)] = 1.0
            guides = {
                "mouth_center": (float(local_x + local_w * 0.5), float(local_y + local_h * 0.75)),
                "left_eye_center": (float(local_x + local_w * 0.35), float(local_y + local_h * 0.35)),
            }
            return {"face": face, "eyes": eyes, "lips": lips}, guides

        seg._landmark_masks = fake_landmark_masks

        masks, guides = seg._landmark_masks_on_crop(image, face_box)

        self.assertIsNotNone(masks)
        self.assertIsNotNone(guides)
        self.assertGreater(calls[0][0][0], face_box[3])
        self.assertGreater(calls[0][0][1], face_box[2])
        self.assertGreater(float(masks["face"][50:68, 76:88].mean()), 0.6)
        self.assertAlmostEqual(guides["mouth_center"][0], face_box[0] + face_box[2] * 0.5, delta=1.0)
        self.assertAlmostEqual(guides["mouth_center"][1], face_box[1] + face_box[3] * 0.75, delta=1.0)

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

    def test_background_removal_segmenter_reports_disabled(self):
        saved = os.environ.get("PORTRAIT_DISABLE_RMBG")
        try:
            os.environ["PORTRAIT_DISABLE_RMBG"] = "1"
            seg = BackgroundRemovalSegmenter()
            self.assertFalse(seg.available)
            self.assertIn("PORTRAIT_DISABLE_RMBG", seg.reason_unavailable)
        finally:
            if saved is None:
                os.environ.pop("PORTRAIT_DISABLE_RMBG", None)
            else:
                os.environ["PORTRAIT_DISABLE_RMBG"] = saved

    def test_scene_subject_masks_prefers_rmbg_and_gates_with_person_instances(self):
        seg = FaceSegmenter.__new__(FaceSegmenter)

        class FakeRMBG:
            available = True
            backend = "rmbg2"

            def segment(self, _img):
                mask = np.zeros((40, 60), dtype=np.float32)
                mask[:, :30] = 1.0
                mask[:, 44:] = 1.0  # foreground clutter that person instances should remove
                return mask

        class FakeInstances:
            available = True

            def person_masks(self, _img, _faces, _subjects_hint):
                person = np.zeros((40, 60), dtype=np.float32)
                person[:, :30] = 1.0
                return [person]

        seg._background_removal = FakeRMBG()
        seg._matting = None
        seg._subjects = None
        seg._person_instances = FakeInstances()
        img = np.zeros((40, 60, 3), dtype=np.float32)
        masks = {key: np.zeros((40, 60), dtype=np.float32) for key in MASK_KEYS}
        masks["face"][8:18, 8:18] = 1.0
        masks["skin"][10:22, 8:20] = 1.0
        masks["hair"][4:10, 7:22] = 1.0

        subjects, background = seg._scene_subject_masks(img, [(8, 6, 14, 18)], masks)

        self.assertEqual(seg.subject_backend, "rmbg2")
        self.assertGreater(float(subjects[:, :30].mean()), 0.4)
        self.assertLess(float(subjects[:, 44:].mean()), 0.25)
        self.assertGreater(float(background[:, 44:].mean()), 0.6)

    def test_watershed_person_labels_returns_none_for_zero_or_one_face(self):
        mask = np.ones((64, 64), dtype=np.float32)
        self.assertIsNone(_watershed_person_labels((64, 64), [], mask))
        self.assertIsNone(_watershed_person_labels((64, 64), [(10, 10, 20, 20)], mask))

    def test_watershed_person_labels_splits_two_touching_blobs_at_the_neck(self):
        h, w = 200, 300
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (90, 100), (55, 80), 0, 0, 360, 1.0, -1)
        cv2.ellipse(mask, (210, 100), (55, 80), 0, 0, 360, 1.0, -1)
        cv2.rectangle(mask, (140, 95), (160, 105), 1.0, -1)  # touching neck
        faces = [(70, 40, 40, 50), (190, 40, 40, 50)]

        label_map = _watershed_person_labels((h, w), faces, mask)

        self.assertIsNotNone(label_map)
        self.assertEqual(sorted(set(label_map.flatten().tolist())), [0, 1, 2])
        area0 = int((label_map == 1).sum())
        area1 = int((label_map == 2).sum())
        self.assertGreater(area0, 1000)
        self.assertGreater(area1, 1000)
        self.assertLess(abs(area0 - area1), max(area0, area1) * 0.4)  # roughly symmetric scene

        total_area = float((mask > 0.5).sum())
        validated, confidence = _validate_person_split(label_map, faces, total_area)
        self.assertIsNotNone(validated)
        self.assertEqual(confidence, "normal")

    def test_validate_person_split_flags_low_confidence_for_close_faces(self):
        """Two face seeds close together over one continuous blob (e.g. a parent holding a
        child) still gets a real split -- not rejected outright -- but flagged low-confidence
        rather than presented as ground truth."""
        h, w = 200, 300
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (150, 110), (120, 85), 0, 0, 360, 1.0, -1)
        faces = [(60, 50, 50, 60), (75, 70, 25, 30)]

        label_map = _watershed_person_labels((h, w), faces, mask)
        total_area = float((mask > 0.5).sum())
        validated, confidence = _validate_person_split(label_map, faces, total_area)

        self.assertIsNotNone(validated)
        self.assertEqual(confidence, "low")

    def test_validate_person_split_rejects_degenerate_tiny_area_split(self):
        """A face seed landing on a near-isolated speck barely connected to the main blob
        gets implausibly little area -- the whole split should be abandoned (Person aliases
        to Subjects) rather than showing a near-empty person mask."""
        h, w = 200, 300
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (100, 110), (90, 80), 0, 0, 360, 1.0, -1)
        cv2.circle(mask, (280, 110), 6, 1.0, -1)
        faces = [(40, 60, 60, 70), (270, 100, 20, 20)]

        label_map = _watershed_person_labels((h, w), faces, mask)
        total_area = float((mask > 0.5).sum())
        validated, confidence = _validate_person_split(label_map, faces, total_area)

        self.assertIsNone(validated)
        self.assertEqual(confidence, "")

    def test_validate_sam_face_mask_accepts_plausible_candidate(self):
        h, w = 120, 120
        face_box = (30, 24, 50, 60)
        zones = _region_zones((h, w), face_box)
        sam = zones["face"].copy()
        hair = np.zeros((h, w), dtype=np.float32)

        candidate, reason = _validate_sam_face_mask(
            sam,
            base_face=zones["face"],
            hair_mask=hair,
            zones=zones,
            face_box=face_box,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(reason, "normal")

    def test_validate_sam_face_mask_rejects_incomplete_candidate(self):
        h, w = 120, 120
        face_box = (30, 24, 50, 60)
        zones = _region_zones((h, w), face_box)
        sam = np.zeros((h, w), dtype=np.float32)
        cv2.circle(sam, (55, 54), 5, 1.0, -1)

        candidate, reason = _validate_sam_face_mask(
            sam,
            base_face=zones["face"],
            hair_mask=None,
            zones=zones,
            face_box=face_box,
        )

        self.assertIsNone(candidate)
        self.assertEqual(reason, "sam-too-small")

    def test_validate_sam_face_mask_rejects_off_target_candidate(self):
        h, w = 120, 120
        face_box = (30, 24, 50, 60)
        zones = _region_zones((h, w), face_box)
        sam = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(sam, (96, 96), (24, 30), 0, 0, 360, 1.0, -1)

        candidate, reason = _validate_sam_face_mask(
            sam,
            base_face=zones["face"],
            hair_mask=None,
            zones=zones,
            face_box=face_box,
        )

        self.assertIsNone(candidate)
        self.assertIn(reason, {"sam-off-target", "sam-low-overlap"})

    def test_sam_face_refinement_falls_back_when_disabled(self):
        saved = os.environ.get("PORTRAIT_DISABLE_SAM_FACE")
        os.environ["PORTRAIT_DISABLE_SAM_FACE"] = "1"
        try:
            seg = FaceSegmenter.__new__(FaceSegmenter)
            seg.backend = "model"

            class FakeInstance:
                available = True

                def face_mask(self, *args, **kwargs):
                    raise AssertionError("disabled face SAM should not call the model")

            seg._instance = FakeInstance()
            img = np.zeros((64, 64, 3), dtype=np.float32)
            masks = {key: np.zeros((64, 64), dtype=np.float32) for key in MASK_KEYS}
            masks["face"][20:42, 20:42] = 1.0
            result = seg._refine_face_masks_with_sam(img, [(18, 16, 28, 32)], 0, masks, None)

            self.assertTrue(np.array_equal(result["face"], masks["face"]))
            self.assertEqual(seg.face_backend_label, "face=model")
        finally:
            if saved is None:
                os.environ.pop("PORTRAIT_DISABLE_SAM_FACE", None)
            else:
                os.environ["PORTRAIT_DISABLE_SAM_FACE"] = saved

    def test_face_backend_label_reports_sam_and_rejection(self):
        # SAM is available and runs, but the candidate it returns is implausible (too small
        # relative to the base face mask) -- _validate_sam_face_mask must reject it, and the
        # label must say so (face=model*) rather than silently looking identical to "never ran".
        saved = os.environ.get("PORTRAIT_DISABLE_SAM_FACE")
        os.environ.pop("PORTRAIT_DISABLE_SAM_FACE", None)
        try:
            h, w = 100, 100
            face_box = (28, 18, 44, 56)
            zones = _region_zones((h, w), face_box)

            tiny_sam = np.zeros((h, w), dtype=np.float32)
            cv2.circle(tiny_sam, (50, 46), 4, 1.0, -1)  # far too small to be a real face

            class FakeInstance:
                available = True

                def face_mask(self, *args, **kwargs):
                    return tiny_sam

            seg = FaceSegmenter.__new__(FaceSegmenter)
            seg.backend = "model"
            seg._instance = FakeInstance()

            masks = {key: np.zeros((h, w), dtype=np.float32) for key in MASK_KEYS}
            masks["face"] = zones["face"].copy()

            result = seg._refine_face_masks_with_sam(np.zeros((h, w, 3), dtype=np.float32), [face_box], 0, masks, None)

            self.assertEqual(seg.face_backend_label, "face=model*")
            self.assertIn("sam-too-small", seg.face_backend)
            # Rejected SAM must leave the base mask untouched, not partially blended.
            self.assertTrue(np.array_equal(result["face"], masks["face"]))
        finally:
            if saved is None:
                os.environ.pop("PORTRAIT_DISABLE_SAM_FACE", None)
            else:
                os.environ["PORTRAIT_DISABLE_SAM_FACE"] = saved

    def test_sam_face_refinement_preserves_semantic_parts(self):
        saved = os.environ.get("PORTRAIT_DISABLE_SAM_FACE")
        os.environ.pop("PORTRAIT_DISABLE_SAM_FACE", None)
        try:
            h, w = 100, 100
            face_box = (28, 18, 44, 56)
            zones = _region_zones((h, w), face_box)
            sam = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(sam, (50, 46), (20, 28), 0, 0, 360, 1.0, -1)

            class FakeInstance:
                available = True

                def face_mask(self, *args, **kwargs):
                    return sam

            seg = FaceSegmenter.__new__(FaceSegmenter)
            seg.backend = "model"
            seg._instance = FakeInstance()

            masks = {key: np.zeros((h, w), dtype=np.float32) for key in MASK_KEYS}
            masks["face"] = zones["face"].copy()
            masks["skin"] = zones["skin"].copy()
            masks["eyes"][35:40, 38:62] = 1.0
            masks["lips"][58:64, 42:58] = 1.0
            masks["brows"][30:34, 38:62] = 1.0
            masks["hair"] = np.zeros((h, w), dtype=np.float32)

            refined = seg._refine_face_masks_with_sam(np.zeros((h, w, 3), dtype=np.float32), [face_box], 0, masks, None)

            self.assertEqual(seg.face_backend_label, "face=sam")
            self.assertGreater(float(refined["eyes"].sum()), 0.0)
            self.assertGreater(float(refined["lips"].sum()), 0.0)
            self.assertLessEqual(float(refined["eyes"].max()), 1.0)
            self.assertLessEqual(float(refined["lips"].max()), 1.0)
            self.assertGreater(float(refined["face"].sum()), float(refined["eyes"].sum() + refined["lips"].sum()))
        finally:
            if saved is None:
                os.environ.pop("PORTRAIT_DISABLE_SAM_FACE", None)
            else:
                os.environ["PORTRAIT_DISABLE_SAM_FACE"] = saved

    def test_person_mask_aliases_subjects_when_split_unavailable(self):
        # _person_mask takes the image (SAM needs pixels), not just a shape -- with <2 faces it
        # aliases Subjects unchanged regardless of the image content.
        seg = FaceSegmenter(backend="heuristic")
        img = np.zeros((40, 40, 3), dtype=np.float32)
        subjects = np.full((40, 40), 0.7, dtype=np.float32)
        result = seg._person_mask(img, [], 0, subjects)
        self.assertTrue(np.allclose(result, subjects))
        result_one_face = seg._person_mask(img, [(5, 5, 10, 10)], 0, subjects)
        self.assertTrue(np.allclose(result_one_face, subjects))

    def test_suppress_cross_person_spill_removes_overlap(self):
        # Two independently-decoded person masks with a deliberately overlapping region --
        # person 0 confidently owns the left half, person 1 confidently owns the right half, but
        # person 1's raw mask (the "spill") also weakly claims a chunk of person 0's territory.
        # Reproduces the exact symptom reported: "person 2 spilling onto the other person".
        h, w = 40, 60
        mask0 = np.zeros((h, w), dtype=np.float32)
        mask0[:, :25] = 1.0
        mask1 = np.zeros((h, w), dtype=np.float32)
        mask1[:, 30:] = 1.0
        mask1[:, 15:30] = 0.6  # spill into person 0's confidently-owned left half

        out0, out1 = _suppress_cross_person_spill([mask0, mask1])

        # Person 0's own confident territory is untouched.
        self.assertTrue(np.array_equal(out0[:, :25], mask0[:, :25]))
        # The contested strip, where person 1's raw claim was weaker than person 0's, now belongs
        # entirely to person 0 -- person 1 no longer spills there.
        self.assertTrue(np.all(out1[:, 15:25] == 0.0))
        # Person 1 still owns its own uncontested core.
        self.assertTrue(np.array_equal(out1[:, 30:], mask1[:, 30:]))

    def test_suppress_cross_person_spill_passes_through_single_face(self):
        mask = np.full((10, 10), 0.8, dtype=np.float32)
        out = _suppress_cross_person_spill([mask])
        self.assertEqual(len(out), 1)
        self.assertTrue(np.array_equal(out[0], mask))

    def test_keep_anchor_connected_mask_removes_disconnected_hair_halo(self):
        mask = np.zeros((60, 90), dtype=np.float32)
        mask[18:48, 45:78] = 0.9  # selected person component
        mask[8:28, 8:30] = 0.65  # disconnected halo around another person's hair

        cleaned = _keep_anchor_connected_mask(mask, [(60, 30)])

        self.assertGreater(float(cleaned[24:42, 50:72].mean()), 0.8)
        self.assertEqual(float(cleaned[8:28, 8:30].max()), 0.0)

    def test_exclude_other_face_regions_removes_other_head_from_person_mask(self):
        h, w = 80, 120
        mask = np.ones((h, w), dtype=np.float32)
        faces = [(12, 14, 28, 34), (70, 16, 30, 36)]

        cleaned = _exclude_other_face_regions(mask, faces, 1, (h, w))

        self.assertEqual(float(cleaned[30, 26]), 0.0)
        self.assertEqual(float(cleaned[14, 26]), 0.0)  # hair/top-of-head area, above face center
        self.assertGreater(float(cleaned[36, 86]), 0.9)

    def test_gate_sam_person_mask_to_watershed_removes_other_person_hair(self):
        h, w = 80, 140
        sam = np.zeros((h, w), dtype=np.float32)
        sam[20:62, 80:124] = 0.95  # selected person
        sam[8:34, 12:45] = 0.75  # leaked other head/hair halo

        labels = np.zeros((h, w), dtype=np.int32)
        labels[8:58, 8:52] = 1
        labels[18:70, 76:130] = 2

        cleaned = _gate_sam_person_mask_to_watershed(sam, labels, 1, (h, w))

        self.assertEqual(float(cleaned[8:34, 12:45].max()), 0.0)
        self.assertGreater(float(cleaned[28:58, 86:118].mean()), 0.9)

    def test_person_mask_gates_sam_leak_with_watershed_baseline(self):
        h, w = 120, 180
        img = np.zeros((h, w, 3), dtype=np.float32)
        subjects = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(subjects, (48, 66), (32, 52), 0, 0, 360, 1.0, -1)
        cv2.ellipse(subjects, (128, 68), (34, 52), 0, 0, 360, 1.0, -1)
        faces = [(30, 20, 36, 38), (108, 22, 38, 40)]

        class FakeInstance:
            available = True

            def instance_masks(self, _img, _faces, _subjects):
                left = np.zeros((h, w), dtype=np.float32)
                right = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(left, (48, 66), (28, 45), 0, 0, 360, 0.95, -1)
                cv2.ellipse(right, (128, 68), (30, 45), 0, 0, 360, 0.95, -1)
                cv2.ellipse(right, (48, 28), (24, 20), 0, 0, 360, 0.7, -1)
                return [left, right]

        seg = FaceSegmenter(backend="heuristic")
        seg._instance = FakeInstance()

        mask = seg._person_mask(img, faces, 1, subjects)

        self.assertEqual(seg.person_backend_label, "person=sam+ws")
        self.assertEqual(float(mask[12:42, 24:66].max()), 0.0)
        self.assertGreater(float(mask[42:92, 106:152].mean()), 0.45)

    def test_person_mask_uses_raw_watershed_gate_when_validated_split_rejected(self):
        h, w = 160, 240
        img = np.zeros((h, w, 3), dtype=np.float32)
        subjects = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(subjects, (62, 78), (54, 68), 0, 0, 360, 1.0, -1)
        cv2.ellipse(subjects, (176, 74), (16, 22), 0, 0, 360, 1.0, -1)
        faces = [(34, 30, 56, 62), (162, 52, 28, 30)]

        raw_labels = _watershed_person_labels((h, w), faces, subjects)
        validated, _confidence = _validate_person_split(raw_labels, faces, float(np.sum(subjects > 0.5)))
        self.assertIsNone(validated)

        class FakeInstance:
            available = True

            def instance_masks(self, _img, _faces, _subjects):
                left = np.zeros((h, w), dtype=np.float32)
                right = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(left, (62, 78), (48, 60), 0, 0, 360, 0.95, -1)
                cv2.ellipse(right, (176, 74), (15, 20), 0, 0, 360, 0.95, -1)
                cv2.ellipse(right, (62, 38), (34, 26), 0, 0, 360, 0.7, -1)
                return [left, right]

        seg = FaceSegmenter(backend="heuristic")
        seg._instance = FakeInstance()

        mask = seg._person_mask(img, faces, 1, subjects)

        self.assertEqual(seg.person_backend_label, "person=sam+ws")
        self.assertEqual(float(mask[16:58, 28:96].max()), 0.0)
        self.assertGreater(float(mask[60:88, 164:188].mean()), 0.45)

    def test_person_instance_segmenter_assigns_faces_to_unique_person_masks(self):
        h, w = 80, 150
        left = np.zeros((h, w), dtype=np.float32)
        right = np.zeros((h, w), dtype=np.float32)
        left[8:68, 8:62] = 1.0
        right[10:70, 88:142] = 1.0
        faces = [(20, 18, 24, 28), (104, 20, 24, 28)]
        instances = [
            {"mask": right, "box": (86, 8, 144, 72), "score": 0.91},
            {"mask": left, "box": (6, 6, 64, 70), "score": 0.93},
        ]

        assigned = PersonInstanceSegmenter._assign_faces_to_instances(faces, instances, (h, w))

        self.assertIsNotNone(assigned)
        self.assertGreater(float(assigned[0][24:42, 24:44].mean()), 0.9)
        self.assertGreater(float(assigned[1][26:44, 106:126].mean()), 0.9)
        self.assertEqual(float(assigned[0][26:44, 106:126].max()), 0.0)
        self.assertEqual(float(assigned[1][24:42, 24:44].max()), 0.0)

    def test_person_mask_prefers_maskdino_person_instances(self):
        h, w = 100, 180
        img = np.zeros((h, w, 3), dtype=np.float32)
        subjects = np.ones((h, w), dtype=np.float32)
        faces = [(22, 18, 28, 32), (116, 20, 30, 34)]
        mask0 = np.zeros((h, w), dtype=np.float32)
        mask1 = np.zeros((h, w), dtype=np.float32)
        mask0[8:86, 8:74] = 1.0
        mask1[10:88, 98:166] = 1.0

        class FakePersonInstances:
            available = True

            def person_masks(self, _img, _faces, _subjects):
                return [mask0, mask1]

        class FailingSam:
            available = True

            def instance_masks(self, *_args, **_kwargs):
                raise AssertionError("Mask DINO should run before SAM")

        seg = FaceSegmenter(backend="heuristic")
        seg._person_instances = FakePersonInstances()
        seg._instance = FailingSam()

        result = seg._person_mask(img, faces, 1, subjects)

        self.assertEqual(seg.person_backend_label, "person=maskdino")
        self.assertGreater(float(result[30:70, 112:152].mean()), 0.9)
        self.assertEqual(float(result[20:70, 18:68].max()), 0.0)

    def test_person_identity_gate_clips_face_part_masks_to_selected_person(self):
        h, w = 80, 140
        other_slice = np.s_[12:42, 12:42]
        selected_slice = np.s_[14:48, 82:122]
        masks = {}
        for key in ("face", "skin", "eyes", "lips", "hair", "brows", "facial_hair"):
            mask = np.zeros((h, w), dtype=np.float32)
            mask[other_slice] = 0.9
            mask[selected_slice] = 0.8
            masks[key] = mask
        masks["subjects"] = np.ones((h, w), dtype=np.float32)
        masks["background"] = np.zeros((h, w), dtype=np.float32)

        person = np.zeros((h, w), dtype=np.float32)
        person[10:58, 76:128] = 1.0
        masks["person"] = person.copy()

        gated = _apply_person_identity_gate_to_face_parts(masks, person, person_backend="maskdino")

        self.assertIsNot(gated, masks)
        for key in ("face", "skin", "eyes", "lips", "hair", "brows", "facial_hair"):
            self.assertLess(float(gated[key][other_slice].max()), 0.01, key)
            self.assertGreater(float(gated[key][selected_slice].mean()), 0.7, key)
        self.assertTrue(np.array_equal(gated["subjects"], masks["subjects"]))
        self.assertTrue(np.array_equal(gated["background"], masks["background"]))
        self.assertTrue(np.array_equal(gated["person"], masks["person"]))

    def test_person_identity_gate_skips_subjects_alias(self):
        h, w = 50, 60
        masks = {"hair": np.ones((h, w), dtype=np.float32)}
        person = np.ones((h, w), dtype=np.float32)

        gated = _apply_person_identity_gate_to_face_parts(masks, person, person_backend="subjects")

        self.assertIs(gated, masks)

    def test_instance_masks_does_not_spill_across_overlapping_prompts(self):
        # End-to-end through InstanceSegmenter.instance_masks: fake the decoder to return masks
        # that deliberately overlap (simulating a real SAM call bleeding across two close/touching
        # people), and confirm the public method's output has zero mutual overlap afterward.
        h, w = 50, 80
        seg = InstanceSegmenter.__new__(InstanceSegmenter)
        seg.available = True
        seg._decoder = object()  # only checked for "is None"; never actually called below
        seg._emb = None
        seg._emb_sig = None

        def fake_embed(img):
            return np.zeros((1, 1, 1, 1), dtype=np.float32), 1.0

        seg.embed = fake_embed

        call = {"n": 0}

        def fake_decode_one(emb, scale, shape_hw, pos_points, neg_points):
            # First call -> left-biased mask with a spill to the right; second -> right-biased.
            mask = np.zeros((h, w), dtype=np.float32)
            if call["n"] == 0:
                mask[:, :35] = 1.0
                mask[:, 35:50] = 0.6  # spills toward the second person
            else:
                mask[:, 40:] = 1.0
            call["n"] += 1
            return mask

        seg._decode_one = fake_decode_one
        seg._body_points = lambda face_box, shape_hw, subjects_mask: [(face_box[0], face_box[1])]

        faces = [(5, 10, 20, 20), (60, 10, 15, 20)]
        subjects = np.ones((h, w), dtype=np.float32)
        masks = seg.instance_masks(np.zeros((h, w, 3), dtype=np.float32), faces, subjects)

        self.assertIsNotNone(masks)
        overlap = (masks[0] > 0.5) & (masks[1] > 0.5)
        self.assertEqual(int(overlap.sum()), 0)

    def test_instance_masks_removes_disconnected_other_hair_halo(self):
        h, w = 70, 110
        seg = InstanceSegmenter.__new__(InstanceSegmenter)
        seg.available = True
        seg._decoder = object()
        seg._emb = None
        seg._emb_sig = None
        seg.embed = lambda img: (np.zeros((1, 1, 1, 1), dtype=np.float32), 1.0)

        call = {"n": 0}

        def fake_decode_one(_emb, _scale, _shape_hw, _pos_points, _neg_points):
            mask = np.zeros((h, w), dtype=np.float32)
            if call["n"] == 0:
                mask[15:52, 8:42] = 0.9
            else:
                mask[16:55, 62:100] = 0.9
                mask[7:32, 9:40] = 0.65  # disconnected halo over face/person 0 hair
            call["n"] += 1
            return mask

        seg._decode_one = fake_decode_one
        seg._body_points = lambda face_box, shape_hw, subjects_mask: [(face_box[0] + face_box[2] * 0.5, face_box[1] + face_box[3] * 0.5)]
        faces = [(10, 10, 30, 34), (68, 14, 28, 34)]
        subjects = np.ones((h, w), dtype=np.float32)

        masks = seg.instance_masks(np.zeros((h, w, 3), dtype=np.float32), faces, subjects)

        self.assertIsNotNone(masks)
        self.assertLess(float(masks[1][10:30, 12:36].max()), 0.1)
        self.assertGreater(float(masks[1][24:48, 70:96].mean()), 0.7)

    def test_instance_masks_removes_connected_other_hair_halo_after_smoothing(self):
        h, w = 80, 130
        seg = InstanceSegmenter.__new__(InstanceSegmenter)
        seg.available = True
        seg._decoder = object()
        seg._emb = None
        seg._emb_sig = None
        seg.embed = lambda img: (np.zeros((1, 1, 1, 1), dtype=np.float32), 1.0)

        call = {"n": 0}

        def fake_decode_one(_emb, _scale, _shape_hw, _pos_points, _neg_points):
            mask = np.zeros((h, w), dtype=np.float32)
            if call["n"] == 0:
                mask[15:56, 8:45] = 0.9
            else:
                mask[18:62, 72:116] = 0.9
                mask[10:36, 10:44] = 0.55  # other person's hair/head halo
                mask[30:36, 44:74] = 0.35  # low bridge connecting halo to selected person
            call["n"] += 1
            return mask

        seg._decode_one = fake_decode_one
        seg._body_points = lambda face_box, shape_hw, subjects_mask: [(face_box[0] + face_box[2] * 0.5, face_box[1] + face_box[3] * 0.5)]
        faces = [(12, 14, 30, 36), (78, 16, 32, 38)]
        subjects = np.ones((h, w), dtype=np.float32)

        masks = seg.instance_masks(np.zeros((h, w, 3), dtype=np.float32), faces, subjects)

        self.assertIsNotNone(masks)
        self.assertLess(float(masks[1][12:34, 14:42].max()), 0.1)
        self.assertGreater(float(masks[1][28:54, 80:110].mean()), 0.7)

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

    def test_face_guides_can_use_landmarker_when_parsing_model_is_unavailable(self):
        seg = FaceSegmenter.__new__(FaceSegmenter)

        class DummyModel:
            available = False
            landmark_backend = "dummy_landmarker"

            def _landmark_masks(self, img_float, face_box=None):
                return None, {"left_eye_upper": (1.0, 2.0)}

        seg._model = DummyModel()

        guides = seg.face_guides(np.zeros((8, 8, 3), dtype=np.float32), face_box=(1, 1, 4, 4))

        self.assertEqual(guides, {"left_eye_upper": (1.0, 2.0)})

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
