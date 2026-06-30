import unittest

try:
    from PySide6.QtWidgets import QApplication

    from portrait_enhancer.ui_qt.main_window import ImagePreviewLabel, PortraitEnhancerQtWindow

    _app = QApplication.instance() or QApplication([])
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "PySide6 not installed")
class HiresEnabledNowTests(unittest.TestCase):
    """_hires_enabled_now() gates the true full-res pixel zoom to only the plain edited
    image -- every mode it checks already shows something else (source-space edit tools, the
    crop-edit full frame, a mask/guide overlay, or a before/after split)."""

    def _window(self):
        window = PortraitEnhancerQtWindow.__new__(PortraitEnhancerQtWindow)
        window._mask_edit_enabled = False
        window._wb_pick_enabled = False
        window._click_mask_enabled = False
        window._denoise_point_enabled = False
        window._crop_edit_enabled = False
        window._show_mask = False
        window._show_sharpen_mask_preview = False
        window._show_expression_guides = False
        window._compare_mode = "off"
        return window

    def test_enabled_by_default(self):
        self.assertTrue(self._window()._hires_enabled_now())

    def test_each_disabling_flag_disables(self):
        flags = [
            "_mask_edit_enabled",
            "_wb_pick_enabled",
            "_click_mask_enabled",
            "_denoise_point_enabled",
            "_crop_edit_enabled",
            "_show_mask",
            "_show_sharpen_mask_preview",
            "_show_expression_guides",
        ]
        for flag in flags:
            window = self._window()
            setattr(window, flag, True)
            self.assertFalse(window._hires_enabled_now(), f"{flag} should disable hi-res zoom")

    def test_compare_mode_disables(self):
        window = self._window()
        window._compare_mode = "split"
        self.assertFalse(window._hires_enabled_now())


@unittest.skipUnless(HAS_DEPS, "PySide6 not installed")
class HiresTileCoversTests(unittest.TestCase):
    def test_no_cached_tile_does_not_cover(self):
        label = ImagePreviewLabel()
        self.assertFalse(label._hires_tile_covers((0.1, 0.1, 0.5, 0.5)))

    def test_covering_rect_returns_true(self):
        label = ImagePreviewLabel()
        label._hires_rect_norm = (0.0, 0.0, 1.0, 1.0)
        self.assertTrue(label._hires_tile_covers((0.2, 0.2, 0.8, 0.8)))

    def test_non_covering_rect_returns_false(self):
        label = ImagePreviewLabel()
        label._hires_rect_norm = (0.3, 0.3, 0.6, 0.6)
        self.assertFalse(label._hires_tile_covers((0.1, 0.1, 0.5, 0.5)))


@unittest.skipUnless(HAS_DEPS, "PySide6 not installed")
class HiresThresholdHysteresisTests(unittest.TestCase):
    """_paint_hires_tile's enter/exit thresholds: enters hi-res once a real full-res pixel
    maps to >=1 screen pixel, but once active stays active down to 0.75x to avoid flicker
    right at the boundary."""

    def _label(self, ratio, zoom, fit_scale, active=False, enabled=True):
        label = ImagePreviewLabel()
        label._full_to_proxy_ratio = ratio
        label._zoom = zoom
        label._fit_scale = lambda: fit_scale
        label._hires_active = active
        label._hires_enabled = enabled
        return label

    def test_disabled_short_circuits(self):
        label = self._label(ratio=2.0, zoom=2.0, fit_scale=1.0, enabled=False)
        label._paint_hires_tile((0, 0, 100, 100))
        self.assertFalse(label._hires_active)

    def test_ratio_at_or_below_one_never_activates(self):
        label = self._label(ratio=1.0, zoom=5.0, fit_scale=1.0)
        label._paint_hires_tile((0, 0, 100, 100))
        self.assertFalse(label._hires_active)

    def test_enters_active_at_1x_effective_ratio(self):
        # full_to_proxy_ratio=2 (proxy is half source res): entering needs
        # fit_scale*zoom/ratio >= 1.0, i.e. fit_scale*zoom >= 2.0.
        label = self._label(ratio=2.0, zoom=1.9, fit_scale=1.0)
        label._paint_hires_tile((0, 0, 100, 100))
        self.assertFalse(label._hires_active)
        label._zoom = 2.0
        label._paint_hires_tile((0, 0, 100, 100))
        self.assertTrue(label._hires_active)

    def test_hysteresis_keeps_active_down_to_075x_but_not_below(self):
        label = self._label(ratio=2.0, zoom=2.0, fit_scale=1.0, active=True)
        label._zoom = 1.5  # effective_ratio = 1.5 * 1.0 / 2.0 = 0.75 -- exactly the exit edge
        label._paint_hires_tile((0, 0, 100, 100))
        self.assertTrue(label._hires_active)
        label._zoom = 1.4  # effective_ratio = 0.70 -- below the exit edge
        label._paint_hires_tile((0, 0, 100, 100))
        self.assertFalse(label._hires_active)


if __name__ == "__main__":
    unittest.main()
