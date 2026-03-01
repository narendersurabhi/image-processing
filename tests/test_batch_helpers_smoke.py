import unittest

try:
    from portrait_enhancer.ui.app import PortraitEnhancerV2

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "tkinter deps not installed")
class BatchHelpersSmokeTests(unittest.TestCase):
    def test_batch_output_path_and_key(self):
        app = PortraitEnhancerV2()
        app.withdraw()
        try:
            out = app._batch_output_path("/tmp/portrait_01.CR2", "/exports", suffix="_retouched", output_ext=".png")
            preset = {"global_params": {"exposure": 10}, "layer_order": ["skin", "hair", "face", "eyes", "lips"]}
            preset_hash = app._preset_hash(preset)
            key = app._batch_config_key(preset_hash, "_retouched", "png")
            self.assertEqual(out, "/exports/portrait_01_retouched.png")
            self.assertIn('"format": "png"', key)
            self.assertIn('"preset_hash"', key)
            self.assertEqual(len(preset_hash), 16)
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
