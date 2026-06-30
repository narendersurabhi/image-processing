import unittest
from unittest.mock import patch

try:
    import numpy as np
    from PIL.Image import Image as PILImage
    from portrait_enhancer.core.processing import (
        _apply_eye_whitening,
        _apply_face_refinement,
        _apply_expression_warp,
        _apply_luma_chroma_denoise,
        _apply_unsharp_mask,
        _build_skin_protection_mask,
        _layer_composite_mask,
        _make_effect_masks,
        _offset_expression_guides,
        _scale_expression_guides,
        apply_output_sharpening,
        StagePipelineCache,
        estimate_chroma_noise_sigma,
        estimate_noise_sigma,
        expression_warp_mode,
        process_background_layer,
        process_all_layers,
        process_face_layer,
        process_global,
        process_hair_layer,
        process_person_layer,
        process_skin_layer,
        suggest_global_auto_values,
        suggest_region_noise_red,
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

    def test_suggest_auto_crop_centers_on_off_center_subject(self):
        from portrait_enhancer.core.processing import suggest_auto_crop

        h, w = 200, 300
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[40:120, 20:80] = 1.0  # small subject in the left third, not frame-centered

        suggestion = suggest_auto_crop(img, mask, target_aspect=None)

        self.assertIn("crop", suggestion)
        x, y, cw, ch = suggestion["crop"]
        # The crop should be tighter than the full frame and shifted toward the subject
        # (left of frame center), not centered on the frame.
        self.assertLess(cw, 0.9)
        self.assertLess(x + cw / 2.0, 0.5)

    def test_suggest_auto_crop_fits_target_aspect_within_frame(self):
        from portrait_enhancer.core.processing import suggest_auto_crop

        h, w = 200, 300  # frame aspect 1.5
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[50:150, 100:200] = 1.0

        suggestion = suggest_auto_crop(img, mask, target_aspect=1.0)

        x, y, cw, ch = suggestion["crop"]
        self.assertAlmostEqual(cw * w / (ch * h), 1.0, places=2)
        self.assertGreaterEqual(x, -1e-6)
        self.assertLessEqual(x + cw, 1.0 + 1e-6)

    def test_suggest_auto_crop_uses_face_guides_for_headroom(self):
        from portrait_enhancer.core.processing import suggest_auto_crop

        h, w = 400, 300
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[70:360, 95:205] = 1.0
        faces = [(110, 70, 80, 95)]
        guides = {
            "left_eye_upper": (130.0, 112.0),
            "left_eye_lower": (130.0, 118.0),
            "right_eye_upper": (170.0, 112.0),
            "right_eye_lower": (170.0, 118.0),
        }

        suggestion = suggest_auto_crop(img, mask, target_aspect=4.0 / 5.0, faces=faces, guides=guides)

        self.assertIn("crop", suggestion)
        x, y, cw, ch = suggestion["crop"]
        self.assertAlmostEqual(cw * w / (ch * h), 4.0 / 5.0, places=2)
        eye_y = 115.0 / h
        rel_eye_y = (eye_y - y) / ch
        self.assertGreater(rel_eye_y, 0.18)
        self.assertLess(rel_eye_y, 0.48)

    def test_suggest_auto_crop_free_mode_chooses_stable_candidate_aspect(self):
        from portrait_enhancer.core.processing import suggest_auto_crop

        h, w = 400, 300
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[80:340, 90:210] = 1.0

        suggestion = suggest_auto_crop(img, mask, target_aspect=None)

        self.assertIn("crop", suggestion)
        _x, _y, cw, ch = suggestion["crop"]
        aspect = cw * w / (ch * h)
        self.assertTrue(any(abs(aspect - target) < 0.04 for target in (3.0 / 4.0, 4.0 / 5.0, 1.0, 2.0 / 3.0)))

    def test_suggest_auto_crop_runs_optional_aesthetic_scorer_on_safe_candidates(self):
        from portrait_enhancer.core.processing import suggest_auto_crop

        class FakeAestheticScorer:
            def __init__(self):
                self.calls = 0

            def __call__(self, crop):
                self.calls += 1
                return 0.75

        h, w = 240, 320
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[70:180, 110:190] = 1.0
        scorer = FakeAestheticScorer()

        suggestion = suggest_auto_crop(img, mask, target_aspect=None, aesthetic_scorer=scorer)

        self.assertIn("crop", suggestion)
        self.assertGreater(scorer.calls, 0)

    def test_aesthetic_output_to_score_handles_scalar_and_distribution_outputs(self):
        from portrait_enhancer.core.aesthetic_crop import aesthetic_output_to_score

        self.assertAlmostEqual(aesthetic_output_to_score(np.array([[7.0]], dtype=np.float32)), 6.0 / 9.0, places=3)

        distribution = np.zeros((1, 10), dtype=np.float32)
        distribution[0, 8] = 1.0
        self.assertAlmostEqual(aesthetic_output_to_score(distribution), 8.0 / 9.0, places=3)

    def test_suggest_auto_crop_skips_crop_when_subject_fills_frame(self):
        from portrait_enhancer.core.processing import suggest_auto_crop

        h, w = 100, 100
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)  # subject already fills the frame

        suggestion = suggest_auto_crop(img, mask, target_aspect=None)

        self.assertNotIn("crop", suggestion)

    def test_suggest_auto_crop_returns_empty_dict_without_subject_mask(self):
        from portrait_enhancer.core.processing import suggest_auto_crop

        img = np.full((50, 50, 3), 0.5, dtype=np.float32)
        self.assertEqual(suggest_auto_crop(img, None), {})

    def test_suggest_horizon_angle_detects_known_tilt_and_matches_framing_sign(self):
        from portrait_enhancer.core.processing import _suggest_horizon_angle
        from portrait_enhancer.core.framing import apply_framing, default_framing
        from PIL import Image

        def make_horizon_image(tilt_deg, w=400, h=300):
            base = np.ones((h, w, 3), dtype=np.float32) * 0.3
            for y in range(0, h, 30):
                base[y:y + 3, :, :] = 0.9
            pil = Image.fromarray((base * 255).astype(np.uint8))
            # Rotate clockwise by tilt_deg (PIL.rotate is CCW for positive angles).
            rotated = pil.rotate(-tilt_deg, resample=Image.BICUBIC, expand=False, fillcolor=(76, 76, 76))
            return np.asarray(rotated, dtype=np.float32) / 255.0

        img = make_horizon_image(6.0)
        angle = _suggest_horizon_angle(img, None)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 6.0, delta=0.5)

        # Applying the suggested angle through the real framing pipeline should straighten
        # the image -- a horizon search on the corrected image should find nothing left.
        framing = default_framing()
        framing["angle"] = angle
        corrected = apply_framing(Image.fromarray((img * 255).astype(np.uint8)), framing)
        residual = _suggest_horizon_angle(np.asarray(corrected, dtype=np.float32) / 255.0, None)
        self.assertIsNone(residual)

    def test_suggest_horizon_angle_stays_silent_without_a_clear_horizon(self):
        from portrait_enhancer.core.processing import _suggest_horizon_angle

        rng = np.random.default_rng(0)
        noise_img = rng.uniform(0.0, 1.0, (200, 200, 3)).astype(np.float32)
        self.assertIsNone(_suggest_horizon_angle(noise_img, None))

    def _noisy_image(self, seed=0, size=64):
        rng = np.random.default_rng(seed)
        base = np.full((size, size, 3), 0.5, dtype=np.float32)
        noise = rng.normal(0.0, 0.05, base.shape).astype(np.float32)
        return np.clip(base + noise, 0.0, 1.0)

    def test_chroma_boost_zero_matches_pre_existing_single_amount_behavior(self):
        # Backward compatibility: chroma_boost=0 (its default) must denoise chroma by exactly
        # `amount`, identical to this function's behavior before chroma_boost was added --
        # forces the bilateral path (fast_preview=True) so this isn't sensitive to whether an
        # optional ML denoiser model happens to be installed.
        img = self._noisy_image()
        baseline = _apply_luma_chroma_denoise(img, 0.6, working_space="srgb", fast_preview=True)
        explicit_zero = _apply_luma_chroma_denoise(img, 0.6, 0.0, working_space="srgb", fast_preview=True)
        self.assertTrue(np.array_equal(baseline, explicit_zero))

    def test_chroma_boost_raises_chroma_smoothing_above_luma(self):
        img = self._noisy_image()
        low_chroma = _apply_luma_chroma_denoise(img, 0.1, 0.1, working_space="srgb", fast_preview=True)
        boosted = _apply_luma_chroma_denoise(img, 0.1, 0.9, working_space="srgb", fast_preview=True)
        # A bigger chroma_boost should change the result more relative to the unfiltered
        # image -- the two outputs must differ when chroma_boost actually differs.
        self.assertFalse(np.array_equal(low_chroma, boosted))

    def test_chroma_only_denoise_runs_even_with_zero_luma_amount(self):
        img = self._noisy_image()
        result = _apply_luma_chroma_denoise(img, 0.0, 0.8, working_space="srgb", fast_preview=True)
        self.assertFalse(np.array_equal(result, img))

    def test_denoise_no_op_when_both_amounts_zero(self):
        img = self._noisy_image()
        result = _apply_luma_chroma_denoise(img, 0.0, 0.0, working_space="srgb", fast_preview=True)
        self.assertTrue(np.array_equal(result, img))

    def test_global_layer_reads_color_noise_red_param(self):
        img = self._noisy_image()
        only_luma = process_global(img, {"noise_red": 60}, runtime_settings={"fast_interactive_preview": True})
        with_color = process_global(
            img, {"noise_red": 60, "color_noise_red": 100}, runtime_settings={"fast_interactive_preview": True}
        )
        self.assertFalse(np.array_equal(only_luma, with_color))

    def test_unsharp_mask_radius_and_masking_change_the_result(self):
        img = self._noisy_image(seed=1)
        narrow = _apply_unsharp_mask(img, 1.0, working_space="srgb", radius=0.6, edge_threshold=0.04)
        wide = _apply_unsharp_mask(img, 1.0, working_space="srgb", radius=2.5, edge_threshold=0.04)
        self.assertFalse(np.array_equal(narrow, wide))

        low_masking = _apply_unsharp_mask(img, 1.0, working_space="srgb", radius=1.4, edge_threshold=0.0)
        high_masking = _apply_unsharp_mask(img, 1.0, working_space="srgb", radius=1.4, edge_threshold=0.4)
        self.assertFalse(np.array_equal(low_masking, high_masking))

    def test_global_layer_reads_sharpen_radius_and_masking_params(self):
        img = self._noisy_image(seed=1)
        default_sharp = process_global(img, {"sharpness": 50})
        wide_radius = process_global(img, {"sharpness": 50, "sharpen_radius": 300})
        self.assertFalse(np.array_equal(default_sharp, wide_radius))

    def test_process_all_layers_can_capture_sharpen_mask_preview_with_zero_sharpness(self):
        img = self._noisy_image(seed=1)
        debug_sink = {}
        result = process_all_layers(
            img,
            {"global": {"sharpness": 0, "sharpen_masking": 8}},
            masks=None,
            debug_sink=debug_sink,
        )

        self.assertIsInstance(result, PILImage)
        mask = debug_sink.get("sharpen_mask")
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, img.shape[:2])
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)
        self.assertGreater(float(mask.max()), float(mask.min()))

    def test_output_sharpening_off_is_identity(self):
        from PIL import Image

        pil = Image.fromarray((self._noisy_image(size=32) * 255).astype(np.uint8))
        self.assertIs(apply_output_sharpening(pil, "off"), pil)
        self.assertIs(apply_output_sharpening(pil, "not-a-real-level"), pil)

    def test_output_sharpening_levels_increase_in_strength(self):
        from PIL import Image

        arr = (self._noisy_image(size=200) * 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        low = np.asarray(apply_output_sharpening(pil, "low"), dtype=np.int32)
        standard = np.asarray(apply_output_sharpening(pil, "standard"), dtype=np.int32)
        high = np.asarray(apply_output_sharpening(pil, "high"), dtype=np.int32)
        base = arr.astype(np.int32)
        diff_low = np.abs(low - base).mean()
        diff_standard = np.abs(standard - base).mean()
        diff_high = np.abs(high - base).mean()
        self.assertLess(diff_low, diff_standard)
        self.assertLess(diff_standard, diff_high)

    def test_output_sharpening_preserves_image_size(self):
        from PIL import Image

        pil = Image.fromarray((self._noisy_image(size=150) * 255).astype(np.uint8))
        sharpened = apply_output_sharpening(pil, "standard")
        self.assertEqual(sharpened.size, pil.size)

    def test_output_sharpening_protects_flat_regions_more_than_edges(self):
        from PIL import Image

        arr = np.full((160, 160, 3), 128, dtype=np.uint8)
        arr[:, 80:] = 190
        rng = np.random.default_rng(7)
        arr[:, :70] = np.clip(arr[:, :70].astype(np.int16) + rng.integers(-3, 4, arr[:, :70].shape), 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)

        sharpened = np.asarray(apply_output_sharpening(pil, "high"), dtype=np.int16)
        base = arr.astype(np.int16)
        flat_delta = float(np.abs(sharpened[:, 10:60] - base[:, 10:60]).mean())
        edge_delta = float(np.abs(sharpened[:, 76:84] - base[:, 76:84]).mean())

        self.assertGreater(edge_delta, flat_delta * 2.0)

    def _half_masked(self, size=48):
        # mask=1 on the left half, 0 on the right half -- lets a test prove an edit is
        # confined to the masked region instead of leaking across the whole frame.
        img = self._noisy_image(size=size)
        mask = np.zeros((size, size), dtype=np.float32)
        mask[:, : size // 2] = 1.0
        return img, mask

    def test_skin_layer_noise_red_denoises_only_inside_its_mask(self):
        img, mask = self._half_masked()
        result = process_skin_layer(img, img, mask, {"noise_red": 80})
        # Mask edges are intentionally feathered (smooth_mask), so check well clear of the
        # boundary rather than the exact half-line.
        self.assertFalse(np.array_equal(result[:, :10], img[:, :10]))
        self.assertTrue(np.array_equal(result[:, -10:], img[:, -10:]))

    def test_skin_layer_without_noise_red_key_is_unaffected(self):
        # Absence of "noise_red" (e.g. an older saved preset) must default to 0 and skip the
        # new code path entirely, not raise.
        img, mask = self._half_masked()
        result = process_skin_layer(img, img, mask, {"smooth": 30})
        self.assertEqual(result.shape, img.shape)

    def test_background_layer_noise_red_denoises_only_inside_its_mask(self):
        img, mask = self._half_masked()
        result = process_background_layer(img, img, mask, {"noise_red": 80})
        self.assertFalse(np.array_equal(result[:, :10], img[:, :10]))
        self.assertTrue(np.array_equal(result[:, -10:], img[:, -10:]))

    def test_person_layer_noise_red_denoises_only_inside_its_mask(self):
        img, mask = self._half_masked()
        result = process_person_layer(img, img, mask, {"noise_red": 80})
        self.assertFalse(np.array_equal(result[:, :10], img[:, :10]))
        self.assertTrue(np.array_equal(result[:, -10:], img[:, -10:]))

    def _chroma_noisy_image(self, size=100, std=15.0, seed=0):
        import cv2

        rng = np.random.default_rng(seed)
        base_u8 = np.full((size, size, 3), 128, dtype=np.uint8)
        lab = cv2.cvtColor(base_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 1] += rng.normal(0, std, lab.shape[:2]).astype(np.float32)
        lab[:, :, 2] += rng.normal(0, std, lab.shape[:2]).astype(np.float32)
        rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        return rgb.astype(np.float32) / 255.0

    def test_estimate_noise_sigma_with_mask_isolates_a_noisy_region(self):
        clean = np.full((100, 100, 3), 0.5, dtype=np.float32)
        noisy = self._noisy_image(size=100)
        mixed = clean.copy()
        mixed[:, :50] = noisy[:, :50]
        mask_left = np.zeros((100, 100), dtype=np.float32)
        mask_left[:, :50] = 1.0
        mask_right = np.zeros((100, 100), dtype=np.float32)
        mask_right[:, 50:] = 1.0

        self.assertGreater(estimate_noise_sigma(mixed, mask_left), estimate_noise_sigma(mixed, mask_right))

    def test_estimate_noise_sigma_returns_zero_for_too_small_a_region(self):
        img = self._noisy_image(size=100)
        tiny_mask = np.zeros((100, 100), dtype=np.float32)
        tiny_mask[0, 0] = 1.0  # well under the 25-effective-pixel floor
        self.assertEqual(estimate_noise_sigma(img, tiny_mask), 0.0)

    def test_estimate_noise_sigma_returns_zero_for_mismatched_mask_shape(self):
        img = self._noisy_image(size=100)
        wrong_shape_mask = np.ones((50, 50), dtype=np.float32)
        self.assertEqual(estimate_noise_sigma(img, wrong_shape_mask), 0.0)

    def test_estimate_chroma_noise_sigma_detects_color_only_noise(self):
        clean = np.full((100, 100, 3), 0.5, dtype=np.float32)
        chroma_noisy = self._chroma_noisy_image()
        self.assertGreater(estimate_chroma_noise_sigma(chroma_noisy), estimate_chroma_noise_sigma(clean))

    def test_suggest_global_auto_values_includes_color_noise_red(self):
        clean = np.full((100, 100, 3), 0.5, dtype=np.float32)
        suggestion = suggest_global_auto_values(clean)
        self.assertIn("color_noise_red", suggestion)
        self.assertEqual(suggestion["color_noise_red"], 0)

    def test_suggest_global_auto_values_boosts_color_noise_red_for_chroma_heavy_noise(self):
        chroma_noisy = self._chroma_noisy_image()
        suggestion = suggest_global_auto_values(chroma_noisy)
        # The whole point of a separate Color NR suggestion: a chroma-noise-dominant image
        # should get a meaningfully higher color_noise_red than noise_red, not just a uniform
        # bump to both (that's what noise_red alone already does).
        self.assertGreater(suggestion["color_noise_red"], suggestion["noise_red"])

    def test_suggest_region_noise_red_returns_zero_without_a_mask(self):
        img = self._noisy_image(size=100)
        self.assertEqual(suggest_region_noise_red(img, None), 0)

    def test_suggest_region_noise_red_responds_to_a_real_masked_region(self):
        img = self._noisy_image(size=100, seed=2)
        mask = np.ones((100, 100), dtype=np.float32)
        clean = np.full((100, 100, 3), 0.5, dtype=np.float32)
        self.assertGreater(suggest_region_noise_red(img, mask), suggest_region_noise_red(clean, mask))


@unittest.skipUnless(HAS_DEPS, "numpy/Pillow not installed")
class CropSafeVignetteTests(unittest.TestCase):
    """Vignette is the only position-dependent effect in process_global -- crop_origin/
    full_shape let it be computed against the full image's center when processing a
    sub-crop (hi-res tile rendering), instead of the crop's own (wrong) center."""

    def _scene(self, h=200, w=300):
        img = np.full((h, w, 3), 0.5, dtype=np.float32)
        return img

    def test_default_args_are_byte_identical_to_current_behavior(self):
        img = self._scene()
        params = {"vignette": 60}
        baseline = process_global(img, params)
        explicit = process_global(img, params, crop_origin=(0, 0), full_shape=img.shape[:2])
        np.testing.assert_array_equal(baseline, explicit)

    def test_crop_matches_corresponding_region_of_full_image(self):
        h, w = 200, 300
        img = self._scene(h, w)
        params = {"vignette": 60}
        full_out = process_global(img, params)

        oy, ox, ch, cw = 50, 80, 90, 120
        crop = img[oy : oy + ch, ox : ox + cw].copy()
        crop_out = process_global(crop, params, crop_origin=(ox, oy), full_shape=(h, w))

        np.testing.assert_array_equal(crop_out, full_out[oy : oy + ch, ox : ox + cw])

    def test_crop_without_full_shape_uses_its_own_wrong_center(self):
        # Sanity check that the test above is actually exercising the fix: omitting
        # full_shape/crop_origin for an off-center crop must NOT match the full-image
        # region (the crop would vignette around its own center instead).
        h, w = 200, 300
        img = self._scene(h, w)
        params = {"vignette": 60}
        full_out = process_global(img, params)

        oy, ox, ch, cw = 50, 80, 90, 120
        crop = img[oy : oy + ch, ox : ox + cw].copy()
        crop_out_naive = process_global(crop, params)

        self.assertFalse(np.array_equal(crop_out_naive, full_out[oy : oy + ch, ox : ox + cw]))


@unittest.skipUnless(HAS_DEPS, "numpy/Pillow not installed")
class ExpressionGuideOffsetTests(unittest.TestCase):
    def test_offset_translates_points(self):
        guides = {"left_eye": (10.0, 20.0), "label": "face"}
        out = _offset_expression_guides(guides, -5.0, 3.0)
        self.assertEqual(out["left_eye"], (5.0, 23.0))
        self.assertEqual(out["label"], "face")

    def test_offset_handles_list_of_guides(self):
        guides = [{"left_eye": (10.0, 20.0)}, {"left_eye": (30.0, 40.0)}]
        out = _offset_expression_guides(guides, 1.0, 1.0)
        self.assertEqual(out[0]["left_eye"], (11.0, 21.0))
        self.assertEqual(out[1]["left_eye"], (31.0, 41.0))

    def test_offset_none_is_none(self):
        self.assertIsNone(_offset_expression_guides(None, 1.0, 1.0))

    def test_offset_then_scale_inverse_round_trips(self):
        # Mirrors the real usage shape (guides scaled from preview->full, then offset into
        # a crop's local frame) -- offsetting by the negative origin and back is a no-op.
        guides = {"nose": (123.0, 45.0)}
        shifted = _offset_expression_guides(guides, -100.0, -20.0)
        back = _offset_expression_guides(shifted, 100.0, 20.0)
        self.assertEqual(back["nose"], guides["nose"])


@unittest.skipUnless(HAS_DEPS, "numpy/Pillow not installed")
class StagePipelineCacheTests(unittest.TestCase):
    """The cached-graph pipeline (#3): a stage cache must (a) never change output vs the
    un-cached pipeline, and (b) skip recomputing upstream stages when only a downstream
    slider changes."""

    def _scene(self, size=96):
        rng = np.random.default_rng(7)
        img = (rng.random((size, size, 3), dtype=np.float32) * 0.6 + 0.2).astype(np.float32)
        masks = {
            "background": np.ones((size, size), dtype=np.float32),
            "skin": np.zeros((size, size), dtype=np.float32),
        }
        masks["skin"][size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
        return img, masks, ("background", "skin")

    def _params(self, skin_smooth=30, exposure=12):
        return {
            "global": {"exposure": exposure, "clarity": 20, "noise_red": 30, "sharpness": 25},
            "background": {"blur": 20, "noise_red": 25},
            "skin": {"smooth": skin_smooth, "blemish": 20, "noise_red": 15},
        }

    def test_cached_output_is_identical_to_uncached(self):
        img, masks, order = self._scene()
        params = self._params()
        ref = np.asarray(process_all_layers(img, params, masks, layer_order=order))
        cache = StagePipelineCache()
        got = np.asarray(
            process_all_layers(img, params, masks, layer_order=order, stage_cache=cache, inputs_token="t1")
        )
        self.assertTrue(np.array_equal(ref, got))

    def test_repeated_identical_render_is_all_hits_and_stable(self):
        img, masks, order = self._scene()
        params = self._params()
        cache = StagePipelineCache()
        first = np.asarray(
            process_all_layers(img, params, masks, layer_order=order, stage_cache=cache, inputs_token="t1")
        )
        second = np.asarray(
            process_all_layers(img, params, masks, layer_order=order, stage_cache=cache, inputs_token="t1")
        )
        self.assertTrue(np.array_equal(first, second))
        # Second render reused every stage rather than recomputing any.
        self.assertGreater(cache.hits, 0)

    def test_downstream_slider_change_reuses_upstream_stages(self):
        import portrait_enhancer.core.processing as P

        img, masks, order = self._scene()
        cache = StagePipelineCache()
        calls = {"global": 0, "face": 0}
        orig_global, orig_face = P.process_global, P._apply_face_refinement

        def spy_global(*a, **k):
            calls["global"] += 1
            return orig_global(*a, **k)

        def spy_face(*a, **k):
            calls["face"] += 1
            return orig_face(*a, **k)

        with patch.object(P, "process_global", spy_global), patch.object(P, "_apply_face_refinement", spy_face):
            process_all_layers(img, self._params(skin_smooth=30), masks, layer_order=order, stage_cache=cache, inputs_token="t1")
            self.assertEqual((calls["global"], calls["face"]), (1, 1))
            # Drag a SKIN slider -> upstream global/face must be cache hits, not recomputed.
            process_all_layers(img, self._params(skin_smooth=45), masks, layer_order=order, stage_cache=cache, inputs_token="t1")
            self.assertEqual((calls["global"], calls["face"]), (1, 1))
            # Change a GLOBAL slider -> upstream must recompute.
            process_all_layers(img, self._params(skin_smooth=45, exposure=25), masks, layer_order=order, stage_cache=cache, inputs_token="t1")
            self.assertEqual((calls["global"], calls["face"]), (2, 2))

    def test_inputs_token_bump_invalidates_every_stage(self):
        import portrait_enhancer.core.processing as P

        img, masks, order = self._scene()
        cache = StagePipelineCache()
        params = self._params()
        calls = {"global": 0}
        orig_global = P.process_global

        def spy_global(*a, **k):
            calls["global"] += 1
            return orig_global(*a, **k)

        with patch.object(P, "process_global", spy_global):
            process_all_layers(img, params, masks, layer_order=order, stage_cache=cache, inputs_token="t1")
            # A mask edit / new analysis bumps the token -> stale stages must not be served.
            process_all_layers(img, params, masks, layer_order=order, stage_cache=cache, inputs_token="t2")
            self.assertEqual(calls["global"], 2)

    def test_cache_respects_lru_limit(self):
        cache = StagePipelineCache(limit=3)
        for i in range(6):
            cache.put(f"k{i}", np.zeros((2, 2), dtype=np.float32))
        # Oldest keys evicted, newest retained.
        self.assertIsNone(cache.get("k0"))
        self.assertIsNotNone(cache.get("k5"))


if __name__ == "__main__":
    unittest.main()
