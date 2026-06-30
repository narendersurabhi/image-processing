import unittest
from unittest import mock

try:
    import numpy as np

    from portrait_enhancer.core import raw_decode

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class _FakeRaw:
    """Stand-in for a rawpy RawPy object: exposes the as-shot attributes decode_raw captures
    and a postprocess that records the kwargs it was called with."""

    def __init__(self, recorder):
        self._recorder = recorder
        self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.daylight_whitebalance = [1.9, 1.0, 1.6, 1.0]
        self.rgb_xyz_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        self.black_level_per_channel = [0, 0, 0, 0]
        self.white_level = 16383
        self.raw_pattern = [[0, 1], [3, 2]]
        self.color_desc = b"RGBG"
        self.num_colors = 3

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **kwargs):
        self._recorder["kwargs"] = kwargs
        self._recorder.setdefault("calls", []).append(kwargs)
        # half_size returns a visibly smaller array so a test can tell preview vs full apart.
        size = 2 if kwargs.get("half_size") else 4
        return np.full((size, size, 3), 65535, dtype=np.uint16)


class _FakeColorSpace:
    sRGB = 1
    Adobe = 2
    ProPhoto = 4
    XYZ = 5
    raw = 0


class _FakeDemosaicAlgo:
    def __init__(self, name, supported=True):
        self.name = name
        self.isSupported = supported


class _FakeDemosaic:
    AHD = _FakeDemosaicAlgo("AHD", True)
    DCB = _FakeDemosaicAlgo("DCB", True)
    VNG = _FakeDemosaicAlgo("VNG", True)
    AAHD = _FakeDemosaicAlgo("AAHD", False)  # not compiled into this fake LibRaw build
    DHT = _FakeDemosaicAlgo("DHT", True)


class _FakeRawpy:
    def __init__(self, recorder):
        self._recorder = recorder
        self.ColorSpace = _FakeColorSpace
        self.DemosaicAlgorithm = _FakeDemosaic

    def imread(self, path):
        self._recorder["path"] = path
        return _FakeRaw(self._recorder)


@unittest.skipUnless(HAS_DEPS, "numpy deps not installed")
class RawDecodeSmokeTests(unittest.TestCase):
    def test_is_raw_path_matches_extensions_case_insensitively(self):
        self.assertTrue(raw_decode.is_raw_path("/x/photo.CR2"))
        self.assertTrue(raw_decode.is_raw_path("photo.nef"))
        self.assertTrue(raw_decode.is_raw_path("a.dng"))
        self.assertFalse(raw_decode.is_raw_path("photo.jpg"))
        self.assertFalse(raw_decode.is_raw_path("photo.png"))

    def test_normalize_applies_defaults_and_falls_back_on_bad_values(self):
        cfg = raw_decode.normalize_raw_settings(None)
        self.assertEqual(cfg["raw_white_balance"], "camera")
        self.assertEqual(cfg["raw_colorspace"], "srgb")
        self.assertFalse(cfg["raw_lut_enabled"])
        self.assertEqual(cfg["raw_lut_path"], "")
        # Default ON preserves LibRaw's auto-brightened import (user-chosen Phase 1 default).
        self.assertTrue(cfg["raw_auto_brightness"])
        # Phase 2 controls default to LibRaw-default behavior (no visible change).
        self.assertEqual(cfg["raw_highlight_mode"], "clip")
        self.assertEqual(cfg["raw_demosaic"], "auto")

        bad = raw_decode.normalize_raw_settings(
            {"raw_white_balance": "nonsense", "raw_colorspace": "weird", "raw_highlight_mode": "x", "raw_demosaic": "y"}
        )
        self.assertEqual(bad["raw_white_balance"], "camera")
        self.assertEqual(bad["raw_colorspace"], "srgb")
        self.assertEqual(bad["raw_highlight_mode"], "clip")
        self.assertEqual(bad["raw_demosaic"], "auto")

    def test_postprocess_kwargs_reflect_wb_and_auto_brightness(self):
        camera_on = raw_decode.build_postprocess_kwargs(
            raw_decode.normalize_raw_settings({"raw_white_balance": "camera", "raw_auto_brightness": True})
        )
        self.assertTrue(camera_on["use_camera_wb"])
        self.assertFalse(camera_on["use_auto_wb"])
        self.assertFalse(camera_on["no_auto_bright"])  # auto-brightness ON -> no_auto_bright False
        self.assertEqual(camera_on["output_bps"], 16)
        self.assertEqual(camera_on["highlight_mode"], 0)  # "clip" default

        blend = raw_decode.build_postprocess_kwargs(
            raw_decode.normalize_raw_settings({"raw_highlight_mode": "blend"})
        )
        self.assertEqual(blend["highlight_mode"], 2)
        rebuild = raw_decode.build_postprocess_kwargs(
            raw_decode.normalize_raw_settings({"raw_highlight_mode": "rebuild"})
        )
        self.assertEqual(rebuild["highlight_mode"], 5)

        auto_off = raw_decode.build_postprocess_kwargs(
            raw_decode.normalize_raw_settings({"raw_white_balance": "auto", "raw_auto_brightness": False})
        )
        self.assertTrue(auto_off["use_auto_wb"])
        self.assertFalse(auto_off["use_camera_wb"])
        self.assertTrue(auto_off["no_auto_bright"])  # deterministic baseline

    def test_decode_raw_returns_float_image_and_captures_metadata(self):
        recorder = {}
        fake = _FakeRawpy(recorder)
        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            img, meta = raw_decode.decode_raw("/tmp/shot.nef", {"raw_colorspace": "adobe"})

        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(img.shape, (4, 4, 3))
        self.assertTrue(np.all((img >= 0.0) & (img <= 1.0)))
        self.assertAlmostEqual(float(img.max()), 1.0, places=5)

        # Decode honored the requested output color space and used the Adobe enum value.
        self.assertEqual(recorder["kwargs"]["output_color"], _FakeColorSpace.Adobe)

        # As-shot camera metadata was captured and is JSON-serializable.
        self.assertTrue(meta["source_is_raw"])
        self.assertEqual(meta["raw_camera_whitebalance"], [2.0, 1.0, 1.5, 1.0])
        self.assertEqual(meta["raw_color_desc"], "RGBG")
        self.assertEqual(meta["raw_white_level"], 16383.0)
        self.assertIn("raw_decode:wb=camera:color=adobe", meta["input_profile_applied"])

    def test_decode_raw_selects_highlight_mode_and_demosaic(self):
        recorder = {}
        fake = _FakeRawpy(recorder)
        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            _img, meta = raw_decode.decode_raw(
                "/tmp/shot.nef", {"raw_highlight_mode": "rebuild", "raw_demosaic": "dcb"}
            )

        self.assertEqual(recorder["kwargs"]["highlight_mode"], 5)  # rebuild
        self.assertIs(recorder["kwargs"]["demosaic_algorithm"], _FakeDemosaic.DCB)
        self.assertNotIn("raw_demosaic_fallback", meta)
        self.assertIn("hl=rebuild:demosaic=dcb", meta["input_profile_applied"])

    def test_decode_raw_falls_back_on_unsupported_demosaic(self):
        recorder = {}
        fake = _FakeRawpy(recorder)
        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            _img, meta = raw_decode.decode_raw("/tmp/shot.nef", {"raw_demosaic": "aahd"})

        # AAHD isn't compiled into this build -> don't pass the kwarg, record the fallback.
        self.assertNotIn("demosaic_algorithm", recorder["kwargs"])
        self.assertIn("raw_demosaic_fallback", meta)

    def test_decode_raw_auto_demosaic_passes_no_algorithm(self):
        recorder = {}
        fake = _FakeRawpy(recorder)
        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            raw_decode.decode_raw("/tmp/shot.nef", {"raw_demosaic": "auto"})
        self.assertNotIn("demosaic_algorithm", recorder["kwargs"])  # LibRaw default preserved

    def test_decode_raw_survives_a_broken_lut_path(self):
        recorder = {}
        fake = _FakeRawpy(recorder)
        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            img, meta = raw_decode.decode_raw(
                "/tmp/shot.nef",
                {"raw_lut_enabled": True, "raw_lut_path": "/no/such/lut.cube"},
            )

        # A missing/broken LUT must not fail the decode -- the image still comes back and the
        # failure is recorded for the source-profile diagnostics.
        self.assertEqual(img.shape, (4, 4, 3))
        self.assertIn("raw_lut_error", meta)
        self.assertIn("lut=off", meta["input_profile_applied"])

    def test_decode_raw_raises_clearly_without_rawpy(self):
        with mock.patch.object(raw_decode, "HAS_RAWPY", False):
            with self.assertRaises(RuntimeError) as ctx:
                raw_decode.decode_raw("/tmp/shot.nef")
        self.assertIn("rawpy is not installed", str(ctx.exception))

    def test_on_preview_runs_a_half_size_pass_before_the_full_decode(self):
        recorder = {}
        fake = _FakeRawpy(recorder)
        preview_calls = []
        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            full_img, full_meta = raw_decode.decode_raw(
                "/tmp/shot.nef", on_preview=lambda arr, meta: preview_calls.append((arr, meta))
            )

        # Exactly one preview callback, with the smaller (half_size) array, before the full one.
        self.assertEqual(len(preview_calls), 1)
        preview_img, preview_meta = preview_calls[0]
        self.assertEqual(preview_img.shape, (2, 2, 3))
        self.assertEqual(full_img.shape, (4, 4, 3))

        # Both postprocess calls happened, half_size first.
        self.assertEqual(len(recorder["calls"]), 2)
        self.assertTrue(recorder["calls"][0].get("half_size"))
        self.assertNotIn("half_size", recorder["calls"][1])

        # Both passes share the same decode settings/audit shape, just flagged via raw_preview.
        self.assertTrue(preview_meta["raw_preview"])
        self.assertFalse(full_meta["raw_preview"])
        self.assertEqual(preview_meta["input_profile_applied"], full_meta["input_profile_applied"])

        # Only one file open for both passes -- no extra disk read for the preview.
        self.assertEqual(recorder["path"], "/tmp/shot.nef")

    def test_no_on_preview_means_a_single_postprocess_call(self):
        recorder = {}
        fake = _FakeRawpy(recorder)
        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            raw_decode.decode_raw("/tmp/shot.nef")
        self.assertEqual(len(recorder["calls"]), 1)
        self.assertNotIn("half_size", recorder["calls"][0])

    def test_on_preview_failure_does_not_break_the_full_decode(self):
        recorder = {}
        fake = _FakeRawpy(recorder)

        def boom(_arr, _meta):
            raise RuntimeError("preview callback blew up")

        with mock.patch.object(raw_decode, "rawpy", fake), mock.patch.object(raw_decode, "HAS_RAWPY", True):
            img, meta = raw_decode.decode_raw("/tmp/shot.nef", on_preview=boom)
        # The preview is best-effort; a broken callback must not prevent the real decode.
        self.assertEqual(img.shape, (4, 4, 3))
        self.assertFalse(meta["raw_preview"])


if __name__ == "__main__":
    unittest.main()
