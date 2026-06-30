"""Preset/Template naming cleanup: verifies the new '_preset'/'preset' persisted keys work,
and that data saved before the rename (legacy '_template' override key, 'shoot_template' meta
kind) still loads correctly through the migration-aware read paths.

Qt requires real construction (Shiboken objects can't be __new__-bypassed like plain Python
classes), so this builds one real PortraitEnhancerQtWindow, shared read-only across the class --
the methods under test are pure data/logic and don't mutate window-level state.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    from portrait_enhancer.ui_qt.main_window import PortraitEnhancerQtWindow

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "PySide6 not installed")
class PresetMetaMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.win = PortraitEnhancerQtWindow()

    def test_read_preset_meta_prefers_new_key(self):
        payload = {"_preset": {"name": "New"}, "_template": {"name": "Legacy"}}
        self.assertEqual(PortraitEnhancerQtWindow._read_preset_meta(payload), {"name": "New"})

    def test_read_preset_meta_falls_back_to_legacy_key(self):
        # Data saved before the rename only has "_template" -- must still be readable.
        payload = {"_template": {"name": "Legacy Shoot Look", "signature": "abc"}}
        self.assertEqual(
            PortraitEnhancerQtWindow._read_preset_meta(payload),
            {"name": "Legacy Shoot Look", "signature": "abc"},
        )

    def test_read_preset_meta_handles_missing_or_non_dict(self):
        self.assertIsNone(PortraitEnhancerQtWindow._read_preset_meta({}))
        self.assertIsNone(PortraitEnhancerQtWindow._read_preset_meta(None))
        self.assertIsNone(PortraitEnhancerQtWindow._read_preset_meta("not a dict"))

    def test_normalize_preset_meta_migrates_legacy_kind_value(self):
        normalized = self.win._normalize_preset_meta({"name": "Old Preset", "kind": "shoot_template"})
        self.assertEqual(normalized["kind"], "preset")

    def test_normalize_preset_meta_defaults_to_preset_kind(self):
        normalized = self.win._normalize_preset_meta({"name": "No Kind Field"})
        self.assertEqual(normalized["kind"], "preset")

    def test_default_preset_meta_uses_new_kind(self):
        self.assertEqual(self.win._default_preset_meta()["kind"], "preset")

    def test_attach_preset_meta_writes_new_key_and_strips_legacy(self):
        # A payload that (hypothetically) still carries a stale legacy key from before a
        # re-save -- attaching fresh meta must end up with only the new key, never both.
        payload = {"global_params": {}, "_template": {"name": "Stale"}}
        meta = {"name": "Fresh", "signature": "sig123"}
        result = self.win._attach_preset_meta(payload, meta)
        self.assertEqual(result.get("_preset"), meta)
        self.assertNotIn("_template", result)

    def test_preset_state_for_override_reads_legacy_payload(self):
        # Simulate an old collection override saved before the rename: top-level payload
        # carries "_template", and its signature must match the payload-minus-meta signature
        # computed by _settings_payload_signature for "applied" state to be detected.
        base_payload = {"global_params": {"exposure": 5}, "framing": None, "mask_adjustments": {}}
        sig = self.win._settings_payload_signature(base_payload)
        legacy_payload = dict(base_payload)
        legacy_payload["_template"] = {"name": "Old Look", "signature": sig}

        state, name = self.win._preset_state_for_override(legacy_payload)
        self.assertEqual(state, "applied")
        self.assertEqual(name, "Old Look")

    def test_preset_state_for_override_detects_modified_on_legacy_payload(self):
        base_payload = {"global_params": {"exposure": 5}, "framing": None, "mask_adjustments": {}}
        legacy_payload = dict(base_payload)
        legacy_payload["_template"] = {"name": "Old Look", "signature": "stale-signature"}
        legacy_payload["global_params"] = {"exposure": 99}  # changed since the preset was applied

        state, name = self.win._preset_state_for_override(legacy_payload)
        self.assertEqual(state, "modified")
        self.assertEqual(name, "Old Look")

    def test_settings_payload_signature_strips_both_key_variants(self):
        payload_new = {"global_params": {"a": 1}, "_preset": {"name": "x"}}
        payload_legacy = {"global_params": {"a": 1}, "_template": {"name": "x"}}
        payload_bare = {"global_params": {"a": 1}}
        # The meta key (new or legacy) must not affect the signature -- otherwise re-saving
        # under the new key would wrongly look "modified" relative to old saved data.
        self.assertEqual(
            self.win._settings_payload_signature(payload_new), self.win._settings_payload_signature(payload_bare)
        )
        self.assertEqual(
            self.win._settings_payload_signature(payload_legacy), self.win._settings_payload_signature(payload_bare)
        )

    def test_preset_payload_meta_uses_preset_default_name(self):
        meta = self.win._preset_payload_meta({}, "")
        self.assertEqual(meta["name"], "Preset")


if __name__ == "__main__":
    unittest.main()
