import unittest

try:
    from portrait_enhancer.ui.app import PortraitEnhancerV2

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "tkinter deps not installed")
class PresetStateSmokeTests(unittest.TestCase):
    def test_preset_round_trip_updates_slider_and_layer_state(self):
        app = PortraitEnhancerV2()
        app.withdraw()
        try:
            app._sliders["global"]["exposure"].set(25)
            app._sliders["skin"]["smooth"].set(40)
            app._layer_options["skin"]["opacity"] = 65.0
            app._layer_options["skin"]["blend_mode"] = "soft_light"
            app._layer_order = ["skin", "hair", "face", "eyes", "lips"]
            app._color_settings = {
                "input_profile": "ignore",
                "raw_white_balance": "auto",
                "raw_colorspace": "adobe",
                "raw_lut_enabled": True,
                "raw_lut_path": "/tmp/demo.cube",
                "working_space": "linear",
                "output_transform": "gamma22",
                "icc_policy": "none",
            }

            preset = app._serialize_preset_state()

            app._sliders["global"]["exposure"].set(0)
            app._sliders["skin"]["smooth"].set(0)
            app._layer_options = app._default_layer_options()
            app._layer_order = list(app._layer_order[::-1])
            app._color_settings = app._default_color_settings()

            app._apply_preset_state(preset)

            self.assertEqual(app._sliders["global"]["exposure"].get(), 25)
            self.assertEqual(app._sliders["skin"]["smooth"].get(), 40)
            self.assertEqual(app._layer_options["skin"]["opacity"], 65.0)
            self.assertEqual(app._layer_options["skin"]["blend_mode"], "soft_light")
            self.assertEqual(app._layer_order, ["skin", "hair", "face", "eyes", "lips"])
            self.assertEqual(app._color_settings["input_profile"], "ignore")
            self.assertEqual(app._color_settings["raw_white_balance"], "auto")
            self.assertEqual(app._color_settings["raw_colorspace"], "adobe")
            self.assertEqual(app._color_settings["raw_lut_enabled"], True)
            self.assertEqual(app._color_settings["raw_lut_path"], "/tmp/demo.cube")
            self.assertEqual(app._color_settings["working_space"], "linear")
            self.assertEqual(app._color_settings["output_transform"], "gamma22")
            self.assertEqual(app._color_settings["icc_policy"], "none")
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
