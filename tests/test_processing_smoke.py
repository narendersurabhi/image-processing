import unittest
from unittest.mock import patch

try:
    import numpy as np
    from PIL.Image import Image as PILImage
    from portrait_enhancer.core.processing import (
        _apply_eye_whitening,
        _apply_face_refinement,
        _apply_expression_warp,
        _build_skin_protection_mask,
        _layer_composite_mask,
        _make_effect_masks,
        expression_warp_mode,
        process_background_layer,
        process_all_layers,
        process_face_layer,
        process_hair_layer,
        process_skin_layer,
        _apply_micro_contrast,
        _even_skin_chroma,
        _hair_strand_mask,
        _unify_face_tone,
        _edge_protection_mask,
    )
    from portrait_enhancer.core.utils import apply_tone_curve, apply_tone_curve_preserve_chroma, adjust_warmth_preserve_hue
    from portrait_enhancer.core.utils import adjust_color_balance_preserve_chroma
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/Pillow not installed")
class ProcessingSmokeTests(unittest.TestCase):
    def test_process_all_layers_returns_pil_image(self):
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        params = {
            "global": {"exposure": 0, "temperature": 0, "tint": 0},
            "face": {},
            "skin": {},
            "eyes": {},
            "lips": {},
            "hair": {},
        }
        layer_options = {
            "hair": {"enabled": True, "opacity": 100.0, "blend_mode": "normal"},
            "skin": {"enabled": True, "opacity": 80.0, "blend_mode": "soft_light"},
            "face": {"enabled": True, "opacity": 100.0, "blend_mode": "overlay"},
            "eyes": {"enabled": False, "opacity": 100.0, "blend_mode": "normal"},
            "lips": {"enabled": True, "opacity": 65.0, "blend_mode": "normal"},
        }
        result = process_all_layers(
            img,
            params,
            masks=None,
            layer_order=("skin", "face", "hair", "eyes", "lips"),
            layer_options=layer_options,
        )
        self.assertIsInstance(result, PILImage)

    def test_selective_layers_build_on_global_adjustments(self):
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        masks = {"face": np.ones((16, 16), dtype=np.float32)}
        params_global_only = {
            "global": {"temperature": 100},
            "face": {},
            "skin": {},
            "eyes": {},
            "lips": {},
            "hair": {},
        }
        params_global_plus_face = {
            "global": {"temperature": 100},
            "face": {"smooth": 20},
            "skin": {},
            "eyes": {},
            "lips": {},
            "hair": {},
        }

        global_only = np.asarray(process_all_layers(img, params_global_only, masks=None), dtype=np.uint8)
        global_plus_face = np.asarray(process_all_layers(img, params_global_plus_face, masks=masks), dtype=np.uint8)

        # Face processing should retain the global color-balance move instead of
        # rebuilding from the original neutral source. The selective render is
        # allowed to differ, but it should stay close to the globally adjusted
        # result rather than snapping back to the original flat gray input.
        self.assertGreaterEqual(float(np.abs(global_only.astype(np.float32) - img * 255.0).mean()), 0.5)
        self.assertLess(
            float(np.abs(global_plus_face.astype(np.float32) - global_only.astype(np.float32)).mean()),
            20.0,
        )

    def test_skin_layer_frequency_smoothing_changes_textured_input(self):
        y, x = np.mgrid[0:32, 0:32]
        base = np.full((32, 32, 3), 0.55, dtype=np.float32)
        texture = ((np.sin(x / 2.0) + np.cos(y / 3.0)) * 0.03).astype(np.float32)
        img = np.clip(base + texture[:, :, None], 0.0, 1.0)
        mask = np.ones((32, 32), dtype=np.float32)

        result = process_skin_layer(
            img,
            img,
            mask,
            {"smooth": 70, "blemish": 40},
        )

        self.assertEqual(result.shape, img.shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertGreater(float(np.abs(result - img).mean()), 0.001)

    def test_tone_curve_preserve_chroma_keeps_skin_ratio_more_stable(self):
        img = np.full((16, 16, 3), [0.72, 0.56, 0.48], dtype=np.float32)
        direct = apply_tone_curve(img, 0, 0, 0, -40, -30)
        stable = apply_tone_curve_preserve_chroma(img, 0, 0, 0, -40, -30)

        original_rg = float(img[:, :, 0].mean() / np.maximum(img[:, :, 1].mean(), 1e-6))
        direct_rg = float(direct[:, :, 0].mean() / np.maximum(direct[:, :, 1].mean(), 1e-6))
        stable_rg = float(stable[:, :, 0].mean() / np.maximum(stable[:, :, 1].mean(), 1e-6))

        self.assertLess(abs(stable_rg - original_rg), abs(direct_rg - original_rg))

    def test_edge_protection_mask_is_lower_on_strong_edges(self):
        img = np.zeros((24, 24, 3), dtype=np.float32)
        img[:, :12] = 0.2
        img[:, 12:] = 0.9

        guard = _edge_protection_mask(img, strength=3.0, blur_sigma=0.8)

        self.assertEqual(guard.shape, (24, 24))
        self.assertLess(float(guard[:, 11:13].mean()), float(guard[:, 4:8].mean()))

    def test_micro_contrast_changes_texture_and_preserves_bounds(self):
        y, x = np.mgrid[0:32, 0:32]
        base = np.full((32, 32, 3), 0.5, dtype=np.float32)
        texture = (np.sin(x / 2.5) * np.cos(y / 2.0) * 0.06).astype(np.float32)
        img = np.clip(base + texture[:, :, None], 0.0, 1.0)

        contrasted = _apply_micro_contrast(img, 40, detail_sigma=1.3, base_sigma=5.0)

        self.assertEqual(contrasted.shape, img.shape)
        self.assertGreaterEqual(float(contrasted.min()), 0.0)
        self.assertLessEqual(float(contrasted.max()), 1.0)
        self.assertGreater(float(np.abs(contrasted - img).mean()), 0.001)

    def test_hair_layer_responds_to_strand_texture(self):
        y, x = np.mgrid[0:32, 0:32]
        base = np.full((32, 32, 3), 0.28, dtype=np.float32)
        strands = ((np.sin(x * 1.8) * 0.5 + 0.5) * 0.10).astype(np.float32)
        img = np.clip(base + strands[:, :, None], 0.0, 1.0)
        mask = np.ones((32, 32), dtype=np.float32)

        strand_mask = _hair_strand_mask(img)
        result = process_hair_layer(
            img,
            img,
            mask,
            {"shine": 60, "clarity": 35, "highlights": 20},
        )

        self.assertGreater(float(strand_mask.mean()), 0.01)
        self.assertEqual(result.shape, img.shape)
        self.assertGreater(float(np.abs(result - img).mean()), 0.001)

    def test_adjust_warmth_preserve_hue_is_more_stable_than_channel_scaling(self):
        img = np.full((16, 16, 3), [0.70, 0.56, 0.47], dtype=np.float32)
        old = img.copy()
        warmth = 40 / 100.0
        old[:, :, 0] = np.clip(old[:, :, 0] * (1 + warmth * 0.2), 0.0, 1.0)
        old[:, :, 2] = np.clip(old[:, :, 2] * (1 - warmth * 0.15), 0.0, 1.0)
        new = adjust_warmth_preserve_hue(img, 40)

        original_rb = float(img[:, :, 0].mean() / np.maximum(img[:, :, 2].mean(), 1e-6))
        old_rb = float(old[:, :, 0].mean() / np.maximum(old[:, :, 2].mean(), 1e-6))
        new_rb = float(new[:, :, 0].mean() / np.maximum(new[:, :, 2].mean(), 1e-6))

        self.assertLess(abs(new_rb - original_rb), abs(old_rb - original_rb))

    def test_even_skin_chroma_reduces_color_blotchiness(self):
        y, x = np.mgrid[0:32, 0:32]
        img = np.full((32, 32, 3), [0.68, 0.54, 0.48], dtype=np.float32)
        blotch = (np.sin(x / 3.0) * np.cos(y / 4.0) * 0.06).astype(np.float32)
        img[:, :, 0] = np.clip(img[:, :, 0] + blotch, 0.0, 1.0)
        img[:, :, 2] = np.clip(img[:, :, 2] - blotch * 0.7, 0.0, 1.0)

        evened = _even_skin_chroma(img, 0.6)

        original_rb_std = float((img[:, :, 0] - img[:, :, 2]).std())
        evened_rb_std = float((evened[:, :, 0] - evened[:, :, 2]).std())

        self.assertLess(evened_rb_std, original_rb_std)

    def test_eye_whitening_brightens_and_neutralizes(self):
        img = np.full((16, 16, 3), [0.76, 0.74, 0.64], dtype=np.float32)
        guard = np.ones((16, 16), dtype=np.float32)

        whitened = _apply_eye_whitening(img, 0.8, edge_guard=guard)

        self.assertGreater(float(whitened.mean()), float(img.mean()))
        self.assertLess(
            float(np.abs(whitened[:, :, 0] - whitened[:, :, 2]).mean()),
            float(np.abs(img[:, :, 0] - img[:, :, 2]).mean()),
        )

    def test_unify_face_tone_reduces_low_frequency_variation(self):
        y, x = np.mgrid[0:32, 0:32]
        img = np.full((32, 32, 3), [0.62, 0.50, 0.45], dtype=np.float32)
        ramp = ((x / 31.0) * 0.12 + (np.sin(y / 5.0) * 0.03)).astype(np.float32)
        img = np.clip(img + ramp[:, :, None], 0.0, 1.0)

        unified = _unify_face_tone(img, 0.7)

        coarse_original = np.abs(img - img.mean(axis=(0, 1), keepdims=True)).mean()
        coarse_unified = np.abs(unified - unified.mean(axis=(0, 1), keepdims=True)).mean()

        self.assertLess(float(coarse_unified), float(coarse_original))

    def test_face_layer_uses_tone_unifier(self):
        y, x = np.mgrid[0:32, 0:32]
        img = np.full((32, 32, 3), [0.60, 0.49, 0.44], dtype=np.float32)
        ramp = ((x / 31.0) * 0.10).astype(np.float32)
        img = np.clip(img + ramp[:, :, None], 0.0, 1.0)
        mask = np.ones((32, 32), dtype=np.float32)

        result = process_face_layer(
            img,
            img,
            mask,
            {"smooth": 60, "highlights": -20, "shadows": 15},
        )

        self.assertEqual(result.shape, img.shape)
        self.assertGreater(float(np.abs(result - img).mean()), 0.001)

    def test_confidence_masks_reduce_effect_in_low_confidence_regions(self):
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[:, :16] = 1.0
        mask[:, 16:] = 0.2

        result = process_skin_layer(
            img,
            img,
            mask,
            {"clarity": 60, "smooth": 40, "blemish": 20},
        )

        high_delta = float(np.abs(result[:, :16] - img[:, :16]).mean())
        low_delta = float(np.abs(result[:, 16:] - img[:, 16:]).mean())

        self.assertGreater(high_delta, low_delta)

    def test_make_effect_masks_is_stricter_for_detail(self):
        mask = np.linspace(0.0, 1.0, 16, dtype=np.float32)[None, :].repeat(8, axis=0)
        effect_masks = _make_effect_masks(mask, "eyes")

        self.assertLess(float(effect_masks["strict"].mean()), float(effect_masks["broad"].mean()))

    def test_layer_composite_mask_is_more_conservative_for_overlay(self):
        mask = np.linspace(0.0, 1.0, 32, dtype=np.float32)[None, :].repeat(8, axis=0)

        normal = _layer_composite_mask(mask, "eyes", "normal", 1.0)
        overlay = _layer_composite_mask(mask, "eyes", "overlay", 1.0)
        soft_light = _layer_composite_mask(mask, "eyes", "soft_light", 1.0)

        self.assertLess(float(overlay.mean()), float(normal.mean()))
        self.assertLess(float(soft_light.mean()), float(normal.mean()))

    def test_face_refinement_noops_without_strength(self):
        img = np.full((16, 16, 3), 0.5, dtype=np.float32)
        masks = {"face": np.ones((16, 16), dtype=np.float32)}

        refined = _apply_face_refinement(img, masks, {"refine": 0})

        self.assertTrue(np.allclose(refined, img))

    def test_face_refinement_uses_refiner_backend(self):
        img = np.full((20, 20, 3), 0.4, dtype=np.float32)
        masks = {"face": np.ones((20, 20), dtype=np.float32)}

        class DummyRefiner:
            def refine(self, arr, face_mask, strength, fidelity=0.65):
                self.called = True
                self.strength = strength
                self.fidelity = fidelity
                boosted = arr.copy()
                boosted[:, :, 0] = np.clip(boosted[:, :, 0] + face_mask * 0.2 * strength, 0.0, 1.0)
                return boosted

        dummy = DummyRefiner()
        with patch("portrait_enhancer.core.processing.get_face_refiner", return_value=dummy):
            refined = _apply_face_refinement(img, masks, {"refine": 80, "refine_fidelity": 72})

        self.assertTrue(dummy.called)
        self.assertAlmostEqual(dummy.strength, 0.8)
        self.assertAlmostEqual(dummy.fidelity, 0.72)
        self.assertGreater(float(refined[:, :, 0].mean()), float(img[:, :, 0].mean()))

    def test_skin_protection_mask_covers_brows_and_facial_hair(self):
        img = np.full((64, 64, 3), [0.72, 0.60, 0.54], dtype=np.float32)
        img[14:19, 14:24] = 0.12
        img[14:19, 40:50] = 0.12
        img[34:39, 22:42] = 0.12
        img[40:55, 18:46] = 0.15
        masks = {
            "face": np.zeros((64, 64), dtype=np.float32),
            "skin": np.zeros((64, 64), dtype=np.float32),
            "brows": np.zeros((64, 64), dtype=np.float32),
        }
        masks["face"][8:58, 8:56] = 1.0
        masks["skin"][10:56, 10:54] = 1.0
        masks["brows"][13:19, 13:24] = 1.0
        masks["brows"][13:19, 40:51] = 1.0
        guides = {
            "left_brow": (19.0, 16.0),
            "right_brow": (45.0, 16.0),
            "left_eye_center": (19.0, 22.0),
            "right_eye_center": (45.0, 22.0),
            "mouth_left": (24.0, 35.0),
            "mouth_right": (40.0, 35.0),
            "mouth_upper": (32.0, 34.0),
            "mouth_lower": (32.0, 38.0),
            "mouth_center": (32.0, 36.0),
        }

        protection = _build_skin_protection_mask(img, masks, guides)

        self.assertGreater(float(protection[16:18, 16:22].mean()), 0.2)
        self.assertGreater(float(protection[35:38, 24:40].mean()), 0.2)
        self.assertGreater(float(protection[44:52, 22:42].mean()), 0.2)

    def test_process_all_layers_protects_facial_hair_from_skin_retouch(self):
        img = np.full((64, 64, 3), [0.72, 0.60, 0.54], dtype=np.float32)
        img[14:19, 14:24] = 0.12
        img[14:19, 40:50] = 0.12
        img[34:39, 22:42] = 0.12
        img[40:55, 18:46] = 0.15

        masks = {
            "face": np.zeros((64, 64), dtype=np.float32),
            "skin": np.zeros((64, 64), dtype=np.float32),
            "brows": np.zeros((64, 64), dtype=np.float32),
        }
        masks["face"][8:58, 8:56] = 1.0
        masks["skin"][10:56, 10:54] = 1.0
        masks["brows"][13:19, 13:24] = 1.0
        masks["brows"][13:19, 40:51] = 1.0
        guides = {
            "left_brow": (19.0, 16.0),
            "right_brow": (45.0, 16.0),
            "left_eye_center": (19.0, 22.0),
            "right_eye_center": (45.0, 22.0),
            "mouth_left": (24.0, 35.0),
            "mouth_right": (40.0, 35.0),
            "mouth_upper": (32.0, 34.0),
            "mouth_lower": (32.0, 38.0),
            "mouth_center": (32.0, 36.0),
        }
        params = {
            "global": {},
            "face": {},
            "skin": {"smooth": 90, "blemish": 40},
            "eyes": {},
            "lips": {},
            "hair": {},
            "subjects": {},
            "background": {},
        }

        result = np.asarray(process_all_layers(img, params, masks, geometry=guides), dtype=np.float32) / 255.0

        cheek_delta = float(np.abs(result[24:30, 14:22] - img[24:30, 14:22]).mean())
        brow_delta = float(np.abs(result[14:19, 14:24] - img[14:19, 14:24]).mean())
        beard_delta = float(np.abs(result[42:52, 22:42] - img[42:52, 22:42]).mean())

        self.assertGreater(cheek_delta, 0.01)
        self.assertLess(brow_delta, cheek_delta)
        self.assertLess(beard_delta, cheek_delta)

    def test_layer_composite_mask_respects_layer_confidence_profile(self):
        mask = np.full((8, 16), 0.35, dtype=np.float32)

        skin = _layer_composite_mask(mask, "skin", "normal", 1.0)
        eyes = _layer_composite_mask(mask, "eyes", "normal", 1.0)

        self.assertGreater(float(skin.mean()), float(eyes.mean()))

    def test_layer_composite_mask_opacity_curve_is_not_linear(self):
        mask = np.full((8, 16), 0.45, dtype=np.float32)

        half_opacity = _layer_composite_mask(mask, "skin", "normal", 0.5)
        linear_half = np.clip(_make_effect_masks(mask, "skin")["blend"] * 0.5, 0.0, 1.0)

        self.assertLess(float(half_opacity.mean()), float(linear_half.mean()))

    def test_global_color_balance_is_more_stable_than_direct_channel_scaling(self):
        img = np.full((16, 16, 3), [0.68, 0.55, 0.49], dtype=np.float32)
        old = img.copy()
        t = 35 / 100.0
        tint = 20 / 100.0
        old[:, :, 0] = np.clip(old[:, :, 0] * (1 + t * 0.4), 0.0, 1.0)
        old[:, :, 2] = np.clip(old[:, :, 2] * (1 - t * 0.4), 0.0, 1.0)
        old[:, :, 1] = np.clip(old[:, :, 1] * (1 + tint * 0.2), 0.0, 1.0)
        new = adjust_color_balance_preserve_chroma(img, temperature=35, tint=20)

        original_rg = float(img[:, :, 0].mean() / np.maximum(img[:, :, 1].mean(), 1e-6))
        old_rg = float(old[:, :, 0].mean() / np.maximum(old[:, :, 1].mean(), 1e-6))
        new_rg = float(new[:, :, 0].mean() / np.maximum(new[:, :, 1].mean(), 1e-6))

        self.assertLess(abs(new_rg - original_rg), abs(old_rg - original_rg))

    def test_expression_warp_changes_image_and_masks(self):
        img = np.zeros((48, 48, 3), dtype=np.float32)
        img[:, :, 0] = np.linspace(0.25, 0.75, 48, dtype=np.float32)[None, :]
        img[:, :, 1] = 0.45
        img[:, :, 2] = np.linspace(0.30, 0.60, 48, dtype=np.float32)[:, None]

        yy, xx = np.mgrid[0:48, 0:48]
        face = ((((xx - 24.0) / 14.0) ** 2 + ((yy - 24.0) / 18.0) ** 2) <= 1.0).astype(np.float32)
        lips = ((((xx - 24.0) / 8.0) ** 2 + ((yy - 32.0) / 4.0) ** 2) <= 1.0).astype(np.float32)
        left_eye = ((((xx - 18.0) / 4.0) ** 2 + ((yy - 20.0) / 2.5) ** 2) <= 1.0).astype(np.float32)
        right_eye = ((((xx - 30.0) / 4.0) ** 2 + ((yy - 20.0) / 2.5) ** 2) <= 1.0).astype(np.float32)
        masks = {
            "face": face,
            "lips": lips,
            "eyes": np.clip(left_eye + right_eye, 0.0, 1.0),
            "skin": np.clip(face - lips - left_eye - right_eye, 0.0, 1.0),
            "hair": np.zeros((48, 48), dtype=np.float32),
        }
        img[:, :, 0] = np.clip(img[:, :, 0] + lips * 0.20 + (left_eye + right_eye) * 0.10, 0.0, 1.0)
        img[:, :, 1] = np.clip(img[:, :, 1] + (left_eye + right_eye) * 0.16, 0.0, 1.0)
        img[:, :, 2] = np.clip(img[:, :, 2] - lips * 0.08, 0.0, 1.0)

        warped_img, warped_masks = _apply_expression_warp(
            img,
            masks,
            {"smile": 70, "eye_open": 60, "mouth_open": 35, "jaw_relax": 25},
        )

        self.assertEqual(warped_img.shape, img.shape)
        self.assertEqual(set(warped_masks.keys()), set(masks.keys()))
        self.assertGreater(float(np.abs(warped_img - img).mean()), 0.0005)
        self.assertGreater(float(np.abs(warped_masks["lips"] - masks["lips"]).mean()), 0.001)

    def test_expression_warp_mode_reports_landmark_vs_mask(self):
        self.assertEqual(expression_warp_mode({"smile": 0, "eye_open": 0, "brow_lift": 0}, None), "off")
        self.assertEqual(expression_warp_mode({"smile": 20}, None), "mask-guided")
        self.assertEqual(expression_warp_mode({"brow_lift": 20}, {"left_brow": (1.0, 2.0)}), "landmark-guided")
        self.assertEqual(expression_warp_mode({"mouth_open": 15}, None), "mask-guided")

    def test_background_layer_blur_and_tone_changes_masked_region(self):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[:, :, 0] = np.linspace(0.2, 0.8, 32, dtype=np.float32)[None, :]
        img[:, :, 1] = np.linspace(0.3, 0.7, 32, dtype=np.float32)[:, None]
        img[:, :, 2] = 0.45
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[:, 16:] = 1.0

        result = process_background_layer(
            img,
            img,
            mask,
            {"blur": 60, "exposure": -20, "dehaze": 40},
        )

        self.assertEqual(result.shape, img.shape)
        self.assertGreater(float(np.abs(result[:, 16:] - img[:, 16:]).mean()), 0.001)


if __name__ == "__main__":
    unittest.main()
