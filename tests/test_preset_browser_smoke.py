import os
import tempfile
import unittest
from pathlib import Path

try:
    from portrait_enhancer.ui.app import PortraitEnhancerV2

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "tkinter deps not installed")
class PresetBrowserSmokeTests(unittest.TestCase):
    def test_refresh_preset_browser_lists_local_presets(self):
        app = PortraitEnhancerV2()
        app.withdraw()
        prev_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                preset_dir = Path(tmpdir) / "presets"
                preset_dir.mkdir()
                (preset_dir / "demo.pepreset").write_text(
                    '{"meta":{"name":"Demo Portrait","category":"studio","tags":["skin","clean"]}}',
                    encoding="utf-8",
                )
                app._refresh_preset_browser()
                self.assertEqual(len(app._preset_library_entries), 1)
                self.assertEqual(app._preset_library_entries[0]["meta"]["category"], "studio")
                app._preset_category_var.set("studio")
                app._refresh_preset_browser()
                self.assertEqual(app._preset_listbox.size(), 1)
                app._preset_search_var.set("clean")
                app._refresh_preset_browser()
                self.assertEqual(app._preset_listbox.size(), 1)
        finally:
            os.chdir(prev_cwd)
            app.destroy()


if __name__ == "__main__":
    unittest.main()
