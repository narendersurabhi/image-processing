import unittest
import tempfile
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    from PIL import Image
    import portrait_enhancer.batch_runner as batch_runner
    from portrait_enhancer.batch_runner import (
        _batch_config_key,
        _get_override_render_params,
        _has_real_person_split,
        _load_cached_masks_without_segmenter,
        _needs_multi_person_render,
        _override_needs_local_masks,
        _render_multi_person,
        _deep_denoise_wave_size,
        _deep_denoise_use_coreml,
        _process_one_image,
        _resolve_max_workers,
        _resolve_task_output_path,
        run_batch_job,
    )
    from portrait_enhancer.config import MASK_ORDER
    from portrait_enhancer.core.analysis_cache import CACHE_KIND_ALL_FACES, CACHE_KIND_SINGLE_FACE, save_analysis

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/opencv not installed")
class BatchRunnerSmokeTests(unittest.TestCase):
    def test_explicit_output_path_overrides_suffix_output_path(self):
        source = "/tmp/source/_MG_3987.CR2"
        explicit = "/tmp/out/custom_name.jpg"

        out_path = _resolve_task_output_path(
            source,
            "/tmp/out",
            suffix="_export",
            output_ext=".jpg",
            explicit_output_paths={source: explicit},
        )

        self.assertEqual(out_path, explicit)

    def test_deep_denoise_worker_logs_tile_progress(self):
        class FakeDenoiser:
            available = True
            execution_provider = "fake"
            reason_unavailable = ""
            workers = 1

            def denoise(self, arr, progress=None):
                if progress:
                    progress(1, 3)
                    progress(2, 3)
                    progress(3, 3)
                return arr

        image = np.zeros((8, 8, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = str(Path(tmp) / "batch_export_log.jsonl")
            task = {
                "path": str(Path(tmp) / "in.jpg"),
                "out_path": str(Path(tmp) / "out.jpg"),
                "render_params": {"global": {}},
                "color_settings": {},
                "mask_adjustments": {},
                "layer_options": {},
                "layer_order": MASK_ORDER,
                "framing": None,
                "override": {},
                "runtime_settings": {"acceleration_mode": "auto"},
                "disable_segmentation": True,
                "output_sharpening": "off",
                "output_quality": 92,
                "output_format": "jpeg",
                "resize_long_edge": 0,
                "keep_metadata": False,
                "deep_denoise": True,
                "deep_denoise_use_coreml": False,
                "cancel_path": "",
                "job_path": "",
                "job_id": "job_progress",
                "job_mode": "export_as",
                "progress_log_path": log_path,
                "log_fields": {},
            }

            with mock.patch.object(batch_runner, "_worker_deep_denoiser", FakeDenoiser()):
                with mock.patch.object(batch_runner, "_worker_deep_denoiser_use_coreml", False):
                    with mock.patch.object(batch_runner, "_read_image_file", return_value=image):
                        with mock.patch.object(batch_runner, "process_all_layers", return_value=Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))):
                            with mock.patch.object(batch_runner, "_save_export_image", return_value=None):
                                result = _process_one_image(task)

            self.assertEqual(result["status"], "success")
            records = [line for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(records), 3)
            self.assertIn('"status": "deep_denoise_progress"', records[-1])
            self.assertIn('"deep_denoise_done": 3', records[-1])
            self.assertIn('"deep_denoise_total": 3', records[-1])

    def test_override_needs_local_masks_checks_every_face_not_just_first(self):
        # Face 0 untouched, only face 1 has real selective settings -- segmentation must
        # still be triggered, even though _get_override_render_params only reflects face 0.
        override = {
            "selective_by_face": [
                {"face_index": 0, "params": {}},
                {"face_index": 1, "params": {"skin": {"smooth": 40}}},
            ]
        }
        self.assertTrue(_override_needs_local_masks(override))
        self.assertFalse(_override_needs_local_masks({"selective_by_face": [{"face_index": 0, "params": {}}]}))

    def test_override_render_params_uses_active_face_profile(self):
        override = {
            "active_face_index": 1,
            "global_params": {"exposure": 5},
            "selective_by_face": [
                {"face_index": 0, "params": {"face": {"exposure": 80}}},
                {"face_index": 1, "params": {"face": {"exposure": -40}}},
            ],
        }

        params = _get_override_render_params(override)

        self.assertEqual(params["global"], {"exposure": 5})
        self.assertEqual(params["face"], {"exposure": -40})

    def test_needs_multi_person_render_requires_two_faces_and_a_real_difference(self):
        same = {
            "selective_by_face": [
                {"face_index": 0, "params": {"person": {"exposure": 50}}},
                {"face_index": 1, "params": {"person": {"exposure": 50}}},
            ]
        }
        different = {
            "selective_by_face": [
                {"face_index": 0, "params": {"person": {"exposure": 50}}},
                {"face_index": 1, "params": {"person": {"exposure": -50}}},
            ]
        }
        self.assertFalse(_needs_multi_person_render(same, 2))
        self.assertTrue(_needs_multi_person_render(different, 2))
        self.assertFalse(_needs_multi_person_render(different, 1))  # only one face detected
        self.assertFalse(_needs_multi_person_render({"selective_by_face": [different["selective_by_face"][0]]}, 2))

    def test_has_real_person_split_detects_aliased_vs_genuine_split(self):
        subjects = np.ones((10, 10), dtype=np.float32)
        aliased = [{"subjects": subjects, "person": subjects.copy()}, {"subjects": subjects, "person": subjects.copy()}]
        self.assertFalse(_has_real_person_split(aliased))

        left = np.zeros((10, 10), dtype=np.float32)
        left[:, :5] = 1.0
        right = 1.0 - left
        genuine = [{"subjects": subjects, "person": left}, {"subjects": subjects, "person": right}]
        self.assertTrue(_has_real_person_split(genuine))

        self.assertFalse(_has_real_person_split([{"subjects": subjects, "person": subjects.copy()}]))

    def test_render_multi_person_applies_each_face_own_settings_to_its_own_region(self):
        h, w = 60, 100
        full = np.full((h, w, 3), 0.4, dtype=np.float32)

        mask_left = np.zeros((h, w), dtype=np.float32)
        mask_left[:, : w // 2] = 1.0
        mask_right = 1.0 - mask_left
        subjects = np.ones((h, w), dtype=np.float32)

        def make_masks(person_mask):
            masks = {key: np.zeros((h, w), dtype=np.float32) for key in MASK_ORDER}
            masks["subjects"] = subjects.copy()
            masks["person"] = person_mask.copy()
            return masks

        per_face_masks = [make_masks(mask_left), make_masks(mask_right)]
        per_face_guides = [None, None]
        override = {
            "global_params": {},
            "selective_by_face": [
                {"face_index": 0, "params": {"person": {"exposure": 100}}},
                {"face_index": 1, "params": {"person": {"exposure": -100}}},
            ],
        }

        result = _render_multi_person(
            full,
            per_face_masks,
            per_face_guides,
            override,
            mask_adjustments={},
            layer_options={},
            layer_order=MASK_ORDER,
            color_settings=None,
            runtime_settings={"acceleration_mode": "auto"},
        )
        arr = np.asarray(result, dtype=np.float32) / 255.0

        # Each person's own region reflects their own exposure setting, not the other's or
        # an average of the two -- this is the entire point of per-person compositing.
        left_mean = float(arr[:, : w // 2].mean())
        right_mean = float(arr[:, w // 2 :].mean())
        self.assertGreater(left_mean, 0.6)
        self.assertLess(right_mean, 0.4)

    def test_batch_config_key_distinguishes_export_options(self):
        plain = _batch_config_key("preset", "_enhanced", "jpeg", deep_denoise=False)
        denoised = _batch_config_key("preset", "_enhanced", "jpeg", deep_denoise=True)
        sharpened = _batch_config_key("preset", "_enhanced", "jpeg", output_sharpening="high")
        resized = _batch_config_key("preset", "_enhanced", "jpeg", resize_long_edge=2048)
        metadata = _batch_config_key("preset", "_enhanced", "jpeg", keep_metadata=True)
        quality = _batch_config_key("preset", "_enhanced", "jpeg", output_quality=80)

        self.assertNotEqual(plain, denoised)
        self.assertNotEqual(plain, sharpened)
        self.assertNotEqual(plain, resized)
        self.assertNotEqual(plain, metadata)
        self.assertNotEqual(plain, quality)
        self.assertIn('"deep_denoise": true', denoised)
        self.assertIn('"output_sharpening": "high"', sharpened)

    def test_loose_cache_reuse_filters_single_face_by_active_face(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                source_path = f"{tmp}/image.jpg"
                with open(source_path, "wb") as fh:
                    fh.write(b"source")
                face0 = save_analysis(
                    source_path,
                    kind=CACHE_KIND_SINGLE_FACE,
                    image_shape=(4, 4),
                    masks={"face": np.full((4, 4), 0.25, dtype=np.float32)},
                    face_index=0,
                    faces=[(0, 0, 2, 2), (2, 0, 2, 2)],
                )
                face1 = save_analysis(
                    source_path,
                    kind=CACHE_KIND_SINGLE_FACE,
                    image_shape=(4, 4),
                    masks={"face": np.full((4, 4), 0.75, dtype=np.float32)},
                    face_index=1,
                    faces=[(0, 0, 2, 2), (2, 0, 2, 2)],
                )
                os.utime(face0, (1, 1))
                os.utime(face1, (2, 2))

                masks, _guides, meta = _load_cached_masks_without_segmenter(
                    np.zeros((4, 4, 3), dtype=np.float32),
                    {
                        "path": source_path,
                        "render_params": {"face": {"exposure": 20}},
                        "override": {
                            "active_face_index": 0,
                            "selective_by_face": [
                                {"face_index": 0, "params": {"face": {"exposure": 20}}},
                            ],
                        },
                        "mask_adjustments": {},
                        "runtime_settings": {"acceleration_mode": "auto"},
                    },
                )
            finally:
                os.chdir(old_cwd)

        self.assertIsNotNone(masks)
        self.assertAlmostEqual(float(masks["face"].mean()), 0.25)
        self.assertEqual(meta["analysis_cache_kind"], CACHE_KIND_SINGLE_FACE)
        self.assertEqual(meta["analysis_cache_face_index"], 0)

    def test_loose_cache_reuse_declines_divergent_multi_person_override(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                source_path = f"{tmp}/image.jpg"
                with open(source_path, "wb") as fh:
                    fh.write(b"source")
                save_analysis(
                    source_path,
                    kind=CACHE_KIND_ALL_FACES,
                    image_shape=(4, 4),
                    masks={"person": np.ones((4, 4), dtype=np.float32)},
                    faces=[(0, 0, 2, 2), (2, 0, 2, 2)],
                )

                masks, guides, meta = _load_cached_masks_without_segmenter(
                    np.zeros((4, 4, 3), dtype=np.float32),
                    {
                        "path": source_path,
                        "render_params": {"person": {"exposure": 20}},
                        "override": {
                            "active_face_index": 0,
                            "selective_by_face": [
                                {"face_index": 0, "params": {"person": {"exposure": 20}}},
                                {"face_index": 1, "params": {"person": {"exposure": -20}}},
                            ],
                        },
                        "mask_adjustments": {},
                        "runtime_settings": {"acceleration_mode": "auto"},
                    },
                )
            finally:
                os.chdir(old_cwd)

        self.assertIsNone(masks)
        self.assertIsNone(guides)
        self.assertIsNone(meta)

    def test_background_deep_denoise_coreml_defaults_off_but_can_opt_in(self):
        from unittest import mock

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_deep_denoise_use_coreml({}))
            self.assertTrue(_deep_denoise_use_coreml({"deep_denoise_use_coreml": True}))
            self.assertFalse(_deep_denoise_use_coreml({"deep_denoise_use_coreml": "false"}))

        with mock.patch.dict("os.environ", {"PORTRAIT_EXPORT_DEEP_DENOISE_USE_COREML": "1"}, clear=True):
            self.assertTrue(_deep_denoise_use_coreml({}))

    def test_deep_denoise_worker_plan_uses_lower_static_cap(self):
        tasks = [{"disable_segmentation": True, "render_params": {}} for _ in range(6)]
        import portrait_enhancer.batch_runner as batch_runner

        original_available = batch_runner._available_memory_gb
        original_pressure = batch_runner._memory_pressure_level
        try:
            batch_runner._available_memory_gb = lambda: 64.0
            batch_runner._memory_pressure_level = lambda: "normal"
            workers, plan = _resolve_max_workers({"max_workers": 4}, tasks, deep_denoise=True)
        finally:
            batch_runner._available_memory_gb = original_available
            batch_runner._memory_pressure_level = original_pressure

        self.assertLessEqual(workers, 2)
        self.assertEqual(plan["static_cap"], 2)
        self.assertTrue(plan["deep_denoise"])

    def test_worker_plan_caps_to_available_memory(self):
        tasks = [{"disable_segmentation": True, "render_params": {}} for _ in range(6)]
        import portrait_enhancer.batch_runner as batch_runner

        original_available = batch_runner._available_memory_gb
        original_pressure = batch_runner._memory_pressure_level
        try:
            batch_runner._available_memory_gb = lambda: 5.0
            batch_runner._memory_pressure_level = lambda: "normal"
            workers, plan = _resolve_max_workers({"max_workers": 4}, tasks, deep_denoise=True)
        finally:
            batch_runner._available_memory_gb = original_available
            batch_runner._memory_pressure_level = original_pressure

        self.assertEqual(workers, 1)
        self.assertEqual(plan["memory_cap"], 1)
        self.assertEqual(plan["deep_denoise_tile_workers"], 1)

    def test_worker_plan_enables_low_memory_segmentation_when_memory_is_tight(self):
        tasks = [{"disable_segmentation": False, "render_params": {"background": {"exposure": -10}}}]
        import portrait_enhancer.batch_runner as batch_runner

        original_available = batch_runner._available_memory_gb
        original_pressure = batch_runner._memory_pressure_level
        try:
            batch_runner._available_memory_gb = lambda: 4.5
            batch_runner._memory_pressure_level = lambda: "normal"
            workers, plan = _resolve_max_workers({"max_workers": 4}, tasks, deep_denoise=False)
        finally:
            batch_runner._available_memory_gb = original_available
            batch_runner._memory_pressure_level = original_pressure

        self.assertEqual(workers, 1)
        self.assertTrue(plan["needs_segmentation"])
        self.assertTrue(plan["low_memory_segmentation"])

    def test_worker_plan_clamps_to_one_under_memory_pressure(self):
        tasks = [{"disable_segmentation": True, "render_params": {}} for _ in range(6)]
        import portrait_enhancer.batch_runner as batch_runner

        original_available = batch_runner._available_memory_gb
        original_pressure = batch_runner._memory_pressure_level
        try:
            batch_runner._available_memory_gb = lambda: 64.0
            batch_runner._memory_pressure_level = lambda: "warning"
            workers, plan = _resolve_max_workers(
                {"max_workers": 4, "deep_denoise_max_workers": 4},
                tasks,
                deep_denoise=True,
            )
        finally:
            batch_runner._available_memory_gb = original_available
            batch_runner._memory_pressure_level = original_pressure

        self.assertEqual(workers, 1)
        self.assertEqual(plan["memory_pressure"], "warning")
        self.assertEqual(plan["memory_cap"], 1)
        self.assertEqual(plan["deep_denoise_tile_workers"], 1)

    def test_deep_denoise_max_workers_can_be_explicitly_raised(self):
        tasks = [{"disable_segmentation": True, "render_params": {}} for _ in range(6)]
        import portrait_enhancer.batch_runner as batch_runner

        original_available = batch_runner._available_memory_gb
        original_pressure = batch_runner._memory_pressure_level
        try:
            batch_runner._available_memory_gb = lambda: 64.0
            batch_runner._memory_pressure_level = lambda: "normal"
            workers, plan = _resolve_max_workers(
                {"max_workers": 4, "deep_denoise_max_workers": 4},
                tasks,
                deep_denoise=True,
            )
        finally:
            batch_runner._available_memory_gb = original_available
            batch_runner._memory_pressure_level = original_pressure

        self.assertEqual(workers, 4)
        self.assertEqual(plan["static_cap"], 4)
        self.assertEqual(plan["deep_denoise_tile_workers"], 1)

    def test_deep_denoise_wave_size_rechecks_every_image_when_memory_capped(self):
        wave_size = _deep_denoise_wave_size(
            1,
            10,
            {"memory_cap": 1, "memory_pressure": "normal"},
        )

        self.assertEqual(wave_size, 1)

    def test_run_batch_job_honors_cancel_marker_before_processing(self):
        import json
        import tempfile
        from pathlib import Path

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.jpg"
            out_dir = root / "out"
            out_dir.mkdir()
            Image.new("RGB", (12, 12), (120, 80, 40)).save(src)
            cancel_path = out_dir / ".batch_jobs" / "job_cancel.cancel"
            cancel_path.parent.mkdir()
            cancel_path.write_text("{}", encoding="utf-8")

            run_batch_job(
                {
                    "job_id": "job_cancel",
                    "mode": "collection_export",
                    "output_dir": str(out_dir),
                    "suffix": "_enhanced",
                    "output_format": "jpeg",
                    "skip_completed": False,
                    "source_paths": [str(src)],
                    "image_overrides": {str(src.resolve()): {}},
                    "runtime_settings": {"acceleration_mode": "auto"},
                    "cancel_path": str(cancel_path),
                }
            )

            self.assertFalse((out_dir / "source_enhanced.jpg").exists())
            records = [
                json.loads(line)
                for line in (out_dir / "batch_export_log.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("canceled", {record.get("status") for record in records})

    def test_run_batch_job_supports_single_image_quick_export_resize(self):
        import tempfile
        from pathlib import Path

        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.jpg"
            out_dir = root / "out"
            out_dir.mkdir()
            Image.new("RGB", (40, 20), (120, 80, 40)).save(src)

            run_batch_job(
                {
                    "job_id": "job_quick",
                    "mode": "collection_export",
                    "quick_export": True,
                    "output_dir": str(out_dir),
                    "suffix": "_export",
                    "output_format": "jpeg",
                    "output_quality": 91,
                    "resize_long_edge": 10,
                    "skip_completed": False,
                    "source_paths": [str(src)],
                    "image_overrides": {str(src.resolve()): {}},
                    "runtime_settings": {"acceleration_mode": "auto"},
                }
            )

            out = out_dir / "source_export.jpg"
            self.assertTrue(out.exists())
            with Image.open(out) as im:
                self.assertEqual(im.size, (10, 5))


if __name__ == "__main__":
    unittest.main()
