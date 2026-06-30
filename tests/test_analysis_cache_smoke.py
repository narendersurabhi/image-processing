import os
import tempfile
import unittest

try:
    import numpy as np

    from portrait_enhancer.core.analysis_cache import (
        CACHE_VERSION,
        CACHE_KIND_ALL_FACES,
        CACHE_KIND_SINGLE_FACE,
        analysis_cache_record_path,
        image_signature,
        load_analysis,
        load_latest_compatible_analysis,
        save_analysis,
        segmenter_cache_signature,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy not installed")
class AnalysisCacheSmokeTests(unittest.TestCase):
    def test_analysis_cache_round_trips_masks_faces_and_guides(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = f"{tmp}/image.jpg"
            with open(source_path, "wb") as fh:
                fh.write(b"source")

            masks = {
                "face": np.ones((4, 5), dtype=np.float32),
                "background": np.zeros((4, 5), dtype=np.float32),
            }
            backend = {"backend": "heuristic", "detector": "haar"}

            written = save_analysis(
                source_path,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=(4, 5),
                masks=masks,
                faces=[(1, 2, 3, 4)],
                guides={"mouth_left": (1.5, 2.5)},
                face_index=0,
                backend_signature=backend,
                cache_root=tmp,
            )
            self.assertIsNotNone(written)
            self.assertTrue(written.exists())

            loaded = load_analysis(
                source_path,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=(4, 5),
                face_index=0,
                backend_signature=backend,
                cache_root=tmp,
            )

            self.assertEqual(loaded["faces"], [(1, 2, 3, 4)])
            self.assertEqual(loaded["guides"], {"mouth_left": [1.5, 2.5]})
            self.assertTrue(np.array_equal(loaded["masks"]["face"], masks["face"]))
            self.assertTrue(np.array_equal(loaded["masks"]["background"], masks["background"]))

    def test_analysis_records_are_independent_files_for_lock_free_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = f"{tmp}/image.jpg"
            with open(source_path, "wb") as fh:
                fh.write(b"source")
            signature = image_signature(source_path)
            backend = {"backend": "heuristic"}
            masks = {"face": np.ones((2, 2), dtype=np.float32)}

            path_a = save_analysis(
                source_path,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=(2, 2),
                masks=masks,
                face_index=0,
                backend_signature=backend,
                cache_root=tmp,
            )
            path_b = save_analysis(
                source_path,
                kind=CACHE_KIND_ALL_FACES,
                image_shape=(2, 2),
                masks=masks,
                backend_signature=backend,
                cache_root=tmp,
            )
            expected_a = analysis_cache_record_path(
                signature,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=[2, 2],
                face_index=0,
                backend_signature=backend,
                cache_root=tmp,
            )
            expected_b = analysis_cache_record_path(
                signature,
                kind=CACHE_KIND_ALL_FACES,
                image_shape=[2, 2],
                face_index=None,
                backend_signature=backend,
                cache_root=tmp,
            )

            self.assertEqual(path_a, expected_a)
            self.assertEqual(path_b, expected_b)
            self.assertNotEqual(path_a, path_b)
            self.assertTrue(path_a.exists())
            self.assertTrue(path_b.exists())

    def test_latest_compatible_analysis_filters_single_face_index_and_mask_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = f"{tmp}/image.jpg"
            with open(source_path, "wb") as fh:
                fh.write(b"source")

            face0 = save_analysis(
                source_path,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=(2, 2),
                masks={"face": np.full((2, 2), 0.25, dtype=np.float32), "skin": np.ones((2, 2), dtype=np.float32)},
                face_index=0,
                faces=[(0, 0, 1, 1), (1, 0, 1, 1)],
                cache_root=tmp,
            )
            face1 = save_analysis(
                source_path,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=(2, 2),
                masks={"face": np.full((2, 2), 0.75, dtype=np.float32)},
                face_index=1,
                faces=[(0, 0, 1, 1), (1, 0, 1, 1)],
                cache_root=tmp,
            )
            os.utime(face0, (1, 1))
            os.utime(face1, (2, 2))

            loaded = load_latest_compatible_analysis(
                source_path,
                target_shape=(2, 2),
                preferred_kinds=(CACHE_KIND_SINGLE_FACE,),
                face_index=0,
                required_mask_keys=("face",),
                cache_root=tmp,
            )

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["face_index"], 0)
            self.assertEqual(sorted(loaded["masks"].keys()), ["face"])
            self.assertAlmostEqual(float(loaded["masks"]["face"].mean()), 0.25)

            loaded_all = load_latest_compatible_analysis(
                source_path,
                target_shape=(2, 2),
                preferred_kinds=(CACHE_KIND_SINGLE_FACE,),
                face_index=0,
                cache_root=tmp,
            )
            self.assertEqual(sorted(loaded_all["masks"].keys()), ["face", "skin"])
            self.assertIsNone(
                load_latest_compatible_analysis(
                    source_path,
                    target_shape=(2, 2),
                    preferred_kinds=(CACHE_KIND_SINGLE_FACE,),
                    face_index=1,
                    required_mask_keys=("skin",),
                    cache_root=tmp,
                )
            )

    def test_segmenter_signature_tracks_face_sam_toggle(self):
        class FakeInstance:
            available = True
            backend = "sam"

        class FakePersonInstances:
            available = True
            backend = "maskdino"
            execution_provider = "cpu"

        class FakeSegmenter:
            backend = "model"
            backend_label = "model"
            detector_backend = "yunet"
            _model = None
            _background_removal = None
            _matting = None
            _subjects = None
            _facial_hair = None
            _person_instances = FakePersonInstances()
            _instance = FakeInstance()

        saved = os.environ.get("PORTRAIT_DISABLE_SAM_FACE")
        try:
            os.environ.pop("PORTRAIT_DISABLE_SAM_FACE", None)
            enabled = segmenter_cache_signature(FakeSegmenter())
            os.environ["PORTRAIT_DISABLE_SAM_FACE"] = "1"
            disabled = segmenter_cache_signature(FakeSegmenter())

            self.assertEqual(CACHE_VERSION, 14)
            self.assertTrue(enabled["person_instance_available"])
            self.assertEqual(enabled["person_instance_backend"], "maskdino")
            self.assertFalse(enabled["background_removal_available"])
            self.assertTrue(enabled["sam_face_enabled"])
            self.assertEqual(enabled["sam_face_backend"], "sam")
            self.assertFalse(disabled["sam_face_enabled"])
            self.assertEqual(disabled["sam_face_backend"], "off")
            self.assertNotEqual(enabled, disabled)
        finally:
            if saved is None:
                os.environ.pop("PORTRAIT_DISABLE_SAM_FACE", None)
            else:
                os.environ["PORTRAIT_DISABLE_SAM_FACE"] = saved


if __name__ == "__main__":
    unittest.main()
