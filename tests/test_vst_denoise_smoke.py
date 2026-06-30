import unittest
from unittest import mock

try:
    import numpy as np

    from portrait_enhancer.core import processing
    from portrait_enhancer.core import vst_denoise as vst
    from portrait_enhancer.core.processing import process_global
    from portrait_enhancer.core.utils import srgb_to_linear

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class _FakeMlDenoiser:
    def __init__(self):
        self.available = True
        self.calls = 0

    def denoise(self, img, amount):
        self.calls += 1
        return img


@unittest.skipUnless(HAS_DEPS, "numpy deps not installed")
class VstDenoiseSmokeTests(unittest.TestCase):
    def test_generalized_anscombe_round_trips(self):
        x = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        for a, b in ((0.01, 1e-5), (0.05, 1e-4), (0.2, 0.0)):
            back = vst.inverse_generalized_anscombe(vst.generalized_anscombe(x, a, b), a, b)
            self.assertTrue(np.allclose(back, x, atol=1e-4), f"round-trip failed for a={a}, b={b}")

    def test_anscombe_stabilizes_signal_dependent_variance(self):
        # Poisson-Gaussian noise var(x) = a*x + b: raw noise is ~16x stronger in highlights than
        # shadows here; after the transform the per-tone noise variance should be ~constant.
        rng = np.random.default_rng(0)
        a, b, n = 0.01, 1e-5, 40000
        results = {}
        for level in (0.05, 0.8):
            clean = np.full(n, level, dtype=np.float32)
            std = np.sqrt(a * level + b)
            noisy = clean + rng.normal(0.0, std, n).astype(np.float32)
            raw_var = float(np.var(noisy - clean))
            stab_var = float(np.var(vst.generalized_anscombe(noisy, a, b) - vst.generalized_anscombe(clean, a, b)))
            results[level] = (raw_var, stab_var)

        raw_ratio = results[0.8][0] / results[0.05][0]
        stab_ratio = results[0.8][1] / results[0.05][1]
        self.assertGreater(raw_ratio, 5.0)  # raw noise is strongly signal-dependent
        self.assertLess(abs(stab_ratio - 1.0), 0.35)  # stabilized noise is ~uniform across tones

    def test_estimate_noise_params_recovers_magnitude(self):
        rng = np.random.default_rng(1)
        sigma = 0.02
        flat = np.full((128, 128), 0.4, dtype=np.float32) + rng.normal(0.0, sigma, (128, 128)).astype(np.float32)
        est = vst.estimate_linear_noise_sigma(flat)
        self.assertAlmostEqual(est, sigma, delta=0.006)
        a, b = vst.estimate_noise_params(flat)
        self.assertGreater(a, 0.0)

    def test_denoise_reduces_noise_on_a_gradient(self):
        rng = np.random.default_rng(2)
        ramp = np.linspace(0.1, 0.9, 96, dtype=np.float32)
        clean = np.repeat(ramp[None, :], 96, axis=0)
        clean3 = np.repeat(clean[:, :, None], 3, axis=2)
        noisy = np.clip(clean3 + rng.normal(0.0, 0.03, clean3.shape).astype(np.float32), 0.0, 1.0)
        out = vst.denoise_luma_linear(noisy, amount=0.7)
        self.assertLess(float(np.mean((out - clean3) ** 2)), float(np.mean((noisy - clean3) ** 2)))

    def test_clean_image_and_zero_amount_are_noops(self):
        rng = np.random.default_rng(3)
        clean = rng.uniform(0.2, 0.8, (32, 32, 3)).astype(np.float32)
        # Smooth (noise-free) image -> nothing measurable -> unchanged.
        smooth = np.repeat(np.linspace(0.2, 0.8, 32, dtype=np.float32)[None, :, None], 32, axis=0)
        smooth = np.repeat(smooth, 3, axis=2)
        self.assertTrue(np.array_equal(vst.denoise_luma_linear(smooth, 0.5), smooth))
        self.assertTrue(np.array_equal(vst.denoise_luma_linear(clean, 0.0), clean))

    def test_process_global_flag_is_noop_in_srgb_and_active_in_linear(self):
        rng = np.random.default_rng(4)
        img = np.clip(
            rng.uniform(0.1, 0.9, (48, 48, 3)).astype(np.float32)
            + rng.normal(0.0, 0.04, (48, 48, 3)).astype(np.float32),
            0.0, 1.0,
        )
        p = {"noise_red": 60}

        # In sRGB working space the flag must do nothing (the VST path requires linear).
        srgb_off = process_global(img.copy(), dict(p), color_settings={"working_space": "srgb"})
        srgb_on = process_global(
            img.copy(), dict(p), color_settings={"working_space": "srgb", "scene_linear_denoise": True}
        )
        self.assertTrue(np.array_equal(srgb_off, srgb_on))

        # In linear working space, enabling it changes the luma denoise result.
        lin_off = process_global(img.copy(), dict(p), color_settings={"working_space": "linear"})
        lin_on = process_global(
            img.copy(), dict(p), color_settings={"working_space": "linear", "scene_linear_denoise": True}
        )
        self.assertGreater(float(np.mean(np.abs(lin_off - lin_on))), 1e-4)


    def _noisy_srgb(self):
        rng = np.random.default_rng(9)
        base = rng.uniform(0.1, 0.9, (40, 40, 3)).astype(np.float32)
        return np.clip(base + rng.normal(0.0, 0.04, base.shape).astype(np.float32), 0.0, 1.0)

    def test_use_learned_denoise_gates_the_ml_denoiser(self):
        img = self._noisy_srgb()
        fake = _FakeMlDenoiser()
        with mock.patch.object(processing, "_get_ml_denoiser", return_value=fake):
            process_global(img.copy(), {"noise_red": 50}, color_settings={"use_learned_denoise": True})
            self.assertGreater(fake.calls, 0)  # learned denoiser used in sRGB mode by default
            fake.calls = 0
            process_global(img.copy(), {"noise_red": 50}, color_settings={"use_learned_denoise": False})
            self.assertEqual(fake.calls, 0)  # checkbox off -> classical path, ML never invoked

    def test_scene_linear_vst_bypasses_the_ml_denoiser(self):
        lin = srgb_to_linear(self._noisy_srgb())
        fake = _FakeMlDenoiser()
        with mock.patch.object(processing, "_get_ml_denoiser", return_value=fake):
            process_global(
                lin.copy(),
                {"noise_red": 50},
                color_settings={"working_space": "linear", "scene_linear_denoise": True, "use_learned_denoise": True},
            )
        # In scene-linear+VST mode the VST handles luma, so the learned denoiser is bypassed
        # even though use_learned_denoise is True (the mutually-exclusive engine behavior).
        self.assertEqual(fake.calls, 0)


if __name__ == "__main__":
    unittest.main()
