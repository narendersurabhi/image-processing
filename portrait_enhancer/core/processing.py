"""Image processing functions for global and selective portrait layers."""

import hashlib
import json
import math
import threading
from collections import OrderedDict

import numpy as np
import cv2
from PIL import Image, ImageEnhance

from portrait_enhancer.config import MASK_ORDER
from .aesthetic_crop import get_aesthetic_crop_scorer, score_crop_aesthetic
from .refine import get_face_refiner
from .white_balance import NEUTRAL_K, apply_white_balance_gains, kelvin_tint_to_gains
from .tone_curve import apply_curve
from .color_mixer import apply_color_mixer
from .vst_denoise import denoise_luma_linear
from .utils import (
    adjust_color_balance_preserve_chroma,
    adjust_hsv_hue,
    adjust_hsv_sat,
    adjust_warmth_preserve_hue,
    apply_tone_curve,
    apply_tone_curve_preserve_chroma,
    bilateral_filter,
    blend_with_mask,
    clamp01,
    display_to_working,
    gaussian_blur,
    smooth_mask,
    to_float,
    to_uint8,
    working_to_display,
)


def _working_space(color_settings: dict | None) -> str:
    return str((color_settings or {}).get("working_space", "srgb"))


def _scene_linear_denoise(color_settings: dict | None) -> bool:
    """True when noise-model-aware (VST) luma denoise should run: opt-in flag *and* a
    scene-linear working space (the VST is only meaningful on linear-light values)."""
    cs = color_settings or {}
    return bool(cs.get("scene_linear_denoise", False)) and _working_space(cs) == "linear"


def _use_learned_denoise(color_settings: dict | None) -> bool:
    """Whether the learned (DnCNN) luma denoiser may run. Defaults True (preserves the prior
    auto-on behavior); the scene-linear VST path bypasses it regardless."""
    return bool((color_settings or {}).get("use_learned_denoise", True))


def _output_transform(color_settings: dict | None) -> str:
    return str((color_settings or {}).get("output_transform", "srgb"))


def _acceleration_mode(runtime_settings: dict | None) -> str:
    return str((runtime_settings or {}).get("acceleration_mode", "auto"))


def _fast_interactive_preview(runtime_settings: dict | None) -> bool:
    return bool((runtime_settings or {}).get("fast_interactive_preview", False))


def _apply_sharpness(img: np.ndarray, factor: float, working_space: str) -> np.ndarray:
    display_img = working_to_display(img, output_transform="srgb", working_space=working_space)
    pil = Image.fromarray(to_uint8(display_img))
    pil = ImageEnhance.Sharpness(pil).enhance(factor)
    return display_to_working(to_float(np.array(pil)), working_space=working_space)


_ml_denoiser = None
_ml_denoiser_init = False
_AUTO_AESTHETIC_SCORER = object()


def _stage_param_digest(*objs) -> str:
    """Stable short digest of a stage's keying material (params, tokens, options). Uses the
    same json.dumps(sort_keys, separators) convention as the window's signature helpers so
    ordering never spuriously busts a cache entry."""
    data = json.dumps(objs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(data.encode("utf-8"), digest_size=16).hexdigest()


def _copy_stage_value(value):
    """Return an independent copy of a cached stage output so neither the cache nor the caller
    can mutate the other's array in place. Cheap (~a few ms for a preview-res frame) relative to
    recomputing the stage (tens to hundreds of ms), and it makes the cache robust to any
    downstream op that writes in place."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_copy_stage_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy_stage_value(item) for key, item in value.items()}
    return value


class StagePipelineCache:
    """Bounded LRU cache of intermediate pipeline-stage outputs (the cached-graph / pixelpipe
    idea from darktable & Lightroom). Keyed by content -- the cumulative digest of every input
    that affects a stage -- so changing one slider only misses from the first affected stage
    downstream; every upstream stage is a hit and is reused instead of recomputed.

    Owned by the window and passed into process_all_layers. Renders are serialized by the
    window's _render_in_flight guard, so single-threaded mutation is guaranteed in practice;
    a lock is still held around get/put as cheap insurance against a future concurrent caller."""

    def __init__(self, limit: int = 24):
        self._store: "OrderedDict[str, object]" = OrderedDict()
        self._limit = max(1, int(limit))
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key is None:
            return None
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, key, value):
        if key is None:
            return
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._limit:
                self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()


def _get_ml_denoiser():
    """Lazily build the optional learned denoiser once per process (mirrors how the batch
    runner builds one FaceSegmenter per worker). Returns None if the model/runtime is absent."""
    global _ml_denoiser, _ml_denoiser_init
    if not _ml_denoiser_init:
        try:
            from .denoise import MLDenoiser
            _ml_denoiser = MLDenoiser()
        except Exception:
            _ml_denoiser = None
        _ml_denoiser_init = True
    return _ml_denoiser


def _apply_luma_chroma_denoise(
    img: np.ndarray,
    amount: float,
    chroma_boost: float = 0.0,
    *,
    working_space: str,
    acceleration: str = "auto",
    fast_preview: bool = False,
    scene_linear_luma: bool = False,
    use_learned: bool = True,
) -> np.ndarray:
    """Noise reduction. Prefers the learned denoiser (DnCNN) when its model is installed;
    otherwise (or during fast interactive preview, to keep slider dragging snappy) falls
    back to edge-aware bilateral that smooths color (a/b) noise harder than luminance,
    matching how RAW converters split Color vs Luminance noise reduction.

    `chroma_boost` raises the *chroma* (a/b channel) strength above `amount` for stubborn
    color-noise blotches that the main slider alone doesn't fully clear. At 0 (the default),
    chroma gets exactly `amount` -- identical to this function's behavior before chroma_boost
    existed, so existing presets/projects render unchanged."""
    amount = float(np.clip(amount, 0.0, 1.0))
    chroma_amount = float(np.clip(max(amount, chroma_boost), 0.0, 1.0))
    if amount <= 0.0 and chroma_amount <= 0.0:
        return img

    # Noise-model-aware luminance denoise, in linear light, via a variance-stabilizing
    # transform (vst_denoise) -- handles signal-dependent shadow noise the fixed-strength
    # display-space path can't. Luma only; chroma still goes through the LAB path below, so we
    # zero out the luma amount and skip the all-channels ML denoiser for this image.
    luma_done = False
    if scene_linear_luma and working_space == "linear" and amount > 0.0:
        img = denoise_luma_linear(img, amount, acceleration=acceleration)
        amount = 0.0
        luma_done = True
        if chroma_amount <= 0.0:
            return img

    display_img = working_to_display(img, output_transform="srgb", working_space=working_space)

    if not luma_done and not fast_preview and use_learned:
        denoiser = _get_ml_denoiser()
        # The learned model denoises every channel together at one strength -- a faithful
        # substitute only when luma and chroma want roughly the same treatment. A
        # meaningfully stronger chroma_boost needs the bilateral path below instead, since
        # only that can treat luminance and chroma independently.
        if denoiser is not None and denoiser.available and (chroma_amount - amount) <= 0.15:
            ml_out = denoiser.denoise(display_img, amount)
            if ml_out is not None:
                return display_to_working(clamp01(ml_out), working_space=working_space)

    lab = cv2.cvtColor(to_uint8(display_img), cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    l_denoised = l_chan if amount <= 0.0 else bilateral_filter(
        l_chan, sigma_color=8.0 + amount * 12.0, sigma_space=1.0 + amount * 2.5, acceleration=acceleration
    )
    if chroma_amount <= 0.0:
        a_denoised, b_denoised = a_chan, b_chan
    else:
        a_denoised = bilateral_filter(a_chan, sigma_color=20.0 + chroma_amount * 40.0, sigma_space=3.0 + chroma_amount * 9.0, acceleration=acceleration)
        b_denoised = bilateral_filter(b_chan, sigma_color=20.0 + chroma_amount * 40.0, sigma_space=3.0 + chroma_amount * 9.0, acceleration=acceleration)

    out_lab = np.clip(np.stack([l_denoised, a_denoised, b_denoised], axis=-1), 0, 255).astype(np.uint8)
    out_rgb = to_float(cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB))
    return display_to_working(clamp01(out_rgb), working_space=working_space)


def _sharpen_edge_mask(l_chan: np.ndarray, edge_threshold: float, acceleration: str = "auto") -> np.ndarray:
    """The edge-detection gate behind the Sharpen Masking slider: 1.0 where a pixel's local
    luminance gradient clears `edge_threshold` (real edge -- sharpen it), 0.0 in flat areas
    (protect from re-amplifying denoised-out noise). Depends only on `l_chan` and the
    threshold, not on the unsharp amount/radius, so it can be previewed independently of
    whether sharpening itself is on."""
    gx = np.zeros_like(l_chan)
    gy = np.zeros_like(l_chan)
    gx[:, 1:-1] = (l_chan[:, 2:] - l_chan[:, :-2]) * 0.5
    gy[1:-1, :] = (l_chan[2:, :] - l_chan[:-2, :]) * 0.5
    edge_strength = np.sqrt(gx * gx + gy * gy) / 255.0
    return smooth_mask(np.clip((edge_strength - edge_threshold) * 8.0, 0.0, 1.0), sigma=0.8, acceleration=acceleration)


def compute_sharpen_mask_preview(img: np.ndarray, *, working_space: str, edge_threshold: float, acceleration: str = "auto") -> np.ndarray:
    """Standalone helper for the UI's Sharpen Masking preview overlay: the same edge mask
    `_apply_unsharp_mask` gates its sharpening with, computed on `img` regardless of whether
    sharpening amount is currently zero. Returns a float32 array in [0, 1], same H x W as img."""
    display_img = working_to_display(img, output_transform="srgb", working_space=working_space)
    l_chan = cv2.cvtColor(to_uint8(display_img), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0]
    return _sharpen_edge_mask(l_chan, edge_threshold, acceleration=acceleration)


def _apply_unsharp_mask(
    img: np.ndarray,
    amount: float,
    *,
    working_space: str,
    radius: float = 1.4,
    acceleration: str = "auto",
    edge_threshold: float = 0.04,
) -> np.ndarray:
    """Luminance-only unsharp mask, boosted only where an edge mask says there's real
    detail -- avoids re-amplifying noise that denoising just smoothed out of flat areas."""
    if amount <= 0.0:
        return img

    display_img = working_to_display(img, output_transform="srgb", working_space=working_space)
    lab = cv2.cvtColor(to_uint8(display_img), cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    blurred = gaussian_blur(l_chan, sigma=radius, acceleration=acceleration)
    detail = l_chan - blurred
    edge_mask = _sharpen_edge_mask(l_chan, edge_threshold, acceleration=acceleration)

    lab[:, :, 0] = np.clip(l_chan + detail * amount * edge_mask, 0, 255)
    out_rgb = to_float(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return display_to_working(clamp01(out_rgb), working_space=working_space)


OUTPUT_SHARPENING_LEVELS = {"off": 0.0, "low": 0.35, "standard": 0.6, "high": 0.9}


def apply_output_sharpening(pil_image: Image.Image, level: str = "standard") -> Image.Image:
    """Final calibrated sharpening pass for the exported image's *actual output* pixel size --
    distinct from (and applied after) the working-resolution Sharpness slider. A downsized
    export needs different sharpening than the same content viewed at full working
    resolution, since detail is denser per pixel after resampling; this is the standard
    "output/print sharpening" step real raw converters apply at export time, calibrated by
    the final long edge so the effect reads consistently across export sizes rather than
    needing to be re-tuned per image. `pil_image` is the fully composited, already-resized
    export result (display/sRGB, uint8) -- the last step before writing the file."""
    amount = OUTPUT_SHARPENING_LEVELS.get(level, 0.0)
    if amount <= 0.0:
        return pil_image
    arr = to_float(np.array(pil_image.convert("RGB")))
    long_edge = max(pil_image.size)
    # Calibrated against a ~2000px long edge (a common "web/social" export size) as the
    # reference radius; smaller exports get a tighter radius (detail is denser per pixel
    # after downsampling), larger/print-sized exports get a wider one, clamped to a sane
    # range so neither extreme produces visible halos.
    radius = float(np.clip(long_edge / 2000.0, 0.5, 1.6))
    lab = cv2.cvtColor(to_uint8(arr), cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    blurred = gaussian_blur(l_chan, sigma=radius, acceleration="auto")
    detail = l_chan - blurred

    gx = np.zeros_like(l_chan)
    gy = np.zeros_like(l_chan)
    gx[:, 1:-1] = (l_chan[:, 2:] - l_chan[:, :-2]) * 0.5
    gy[1:-1, :] = (l_chan[2:, :] - l_chan[:-2, :]) * 0.5
    edge_strength = np.sqrt(gx * gx + gy * gy) / 255.0

    abs_detail = np.abs(detail) / 255.0
    local_detail = gaussian_blur(abs_detail, sigma=max(0.7, radius * 0.75), acceleration="auto")
    edge_mask = np.clip((edge_strength - 0.010) / 0.060, 0.0, 1.0)
    texture_mask = np.clip((local_detail - 0.003) / 0.028, 0.0, 1.0)
    mask = np.maximum(edge_mask, texture_mask * 0.75)

    flat_guard = 1.0 - np.clip((0.020 - local_detail) / 0.020, 0.0, 1.0) * 0.75
    highlight_guard = 1.0 - np.clip((l_chan / 255.0 - 0.90) / 0.10, 0.0, 1.0) * 0.45
    shadow_guard = 1.0 - np.clip((0.05 - l_chan / 255.0) / 0.05, 0.0, 1.0) * 0.25
    mask = smooth_mask(mask * flat_guard * highlight_guard * shadow_guard, sigma=0.7, acceleration="auto")

    halo_limit = 10.0 + 18.0 * amount
    sharpened_l = l_chan + np.clip(detail * amount * mask, -halo_limit, halo_limit)
    lab[:, :, 0] = np.clip(sharpened_l, 0, 255)
    out = to_float(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return Image.fromarray(to_uint8(out))


_NOISE_ESTIMATION_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def estimate_noise_sigma(img: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fast, robust per-image (or per-region) noise estimate (Immerkjaer 1996) on luminance,
    0-255 scale.

    The Laplacian-of-Laplacian kernel cancels out on flat real edges (its response there
    is dominated by sensor noise), so a single average is a decent proxy for noise level
    without needing a flat-patch ROI or EXIF/ISO metadata.

    `mask` (optional, same H×W as img, 0-1) restricts the estimate to a masked region --
    e.g. Skin or Background -- instead of the whole frame, so a region with a meaningfully
    different actual noise level (a shadowed background, smoother midtone skin) gets its own
    reading rather than inheriting the whole-image average. Weighted by mask value and
    normalized by the mask's own effective pixel count, so a small or partial region isn't
    diluted by also summing over pixels outside it. Returns 0.0 if the region is too small
    (<25 effective pixels) to trust."""
    luma = to_uint8(_luma(img)).astype(np.float32)
    h, w = luma.shape
    if h < 5 or w < 5:
        return 0.0
    conv = cv2.filter2D(luma, -1, _NOISE_ESTIMATION_KERNEL, borderType=cv2.BORDER_REFLECT)
    if mask is None:
        return float(np.sum(np.abs(conv)) * math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2)))
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if m.shape != luma.shape:
        return 0.0
    effective_count = float(np.sum(m))
    if effective_count < 25.0:
        return 0.0
    return float(np.sum(np.abs(conv) * m) * math.sqrt(0.5 * math.pi) / (6.0 * effective_count))


def estimate_chroma_noise_sigma(img: np.ndarray) -> float:
    """Like estimate_noise_sigma, but measures noise in the chroma (LAB a/b) channels instead
    of luminance -- the same Laplacian-of-Laplacian estimator, applied to color rather than
    brightness. OpenCV's 8-bit LAB conversion keeps a/b on the same [0, 255] scale as L, so
    this is directly comparable to estimate_noise_sigma's luma reading: a meaningfully higher
    chroma sigma than luma sigma is the classic "color blotches in shadows" signature, as
    opposed to general sensor grain that affects both about equally."""
    rgb_u8 = to_uint8(img)
    h, w = rgb_u8.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    scale = math.sqrt(0.5 * math.pi) / (6.0 * (w - 2) * (h - 2))
    sigmas = []
    for channel in (lab[:, :, 1], lab[:, :, 2]):
        conv = cv2.filter2D(channel, -1, _NOISE_ESTIMATION_KERNEL, borderType=cv2.BORDER_REFLECT)
        sigmas.append(float(np.sum(np.abs(conv)) * scale))
    return sum(sigmas) / 2.0


def _solve_auto_tone(median: float, black_point: float, white_point: float) -> dict:
    """Shared math behind suggest_auto_tone/suggest_auto_tone_for_face: pick an exposure
    EV that puts `median` at a normal midtone, then -- against that exposure-adjusted
    histogram -- invert the tone curve's [0, 0.10] and [0.90, 1.0] segments (see
    apply_tone_curve) to find Blacks/Whites that stretch black_point/white_point toward
    true black/white."""
    ev = float(np.clip(math.log2(0.45 / max(median, 1e-3)), -2.0, 2.0))
    exposure = round(ev * 100.0)

    gain = 2.0 ** ev
    black_adj = float(np.clip(black_point * gain, 0.0, 1.0))
    white_adj = float(np.clip(white_point * gain, 0.0, 1.0))

    if black_adj < 0.099:
        blacks = -100.0 * black_adj / max(1e-4, 1.0 - 10.0 * black_adj)
    else:
        blacks = -100.0
    blacks = round(float(np.clip(blacks, -100.0, 0.0)))

    t = float(np.clip((white_adj - 0.90) / 0.10, 0.0, 1.0))
    whites = 40.0 * (1.0 - t) / max(1e-4, 2.0 - t)
    whites = round(float(np.clip(whites, 0.0, 100.0)))

    return {"exposure": exposure, "blacks": blacks, "whites": whites}


def suggest_auto_tone(img: np.ndarray) -> dict:
    """Suggest Exposure/Blacks/Whites (0-100 slider scale) via classic histogram auto-leveling:
    brighten/darken so the median luminance lands near a normal midtone, then stretch the
    near-black/near-white percentiles toward true black/white. Computed in pipeline order
    (exposure first, so blacks/whites are solved against the exposure-adjusted histogram)."""
    luma = _luma(img).reshape(-1)
    if luma.size == 0:
        return {"exposure": 0, "blacks": 0, "whites": 0}

    black_point = float(np.percentile(luma, 0.5))
    white_point = float(np.percentile(luma, 99.5))
    median = float(np.percentile(luma, 50.0))
    return _solve_auto_tone(median, black_point, white_point)


def suggest_auto_tone_for_face(img: np.ndarray, face_mask: np.ndarray) -> dict:
    """Like suggest_auto_tone, but the Exposure target is driven by the face region's own
    brightness ("expose for the face") instead of the whole frame's median -- useful for
    backlit/shadowed subjects where the background's brightness isn't representative.
    Blacks/Whites still stretch the whole frame's near-black/near-white percentiles, since
    contrast targets need the full tonal range, not just the (usually flat) face region."""
    luma = _luma(img)
    flat_luma = luma.reshape(-1)
    if flat_luma.size == 0:
        return {"exposure": 0, "blacks": 0, "whites": 0}

    black_point = float(np.percentile(flat_luma, 0.5))
    white_point = float(np.percentile(flat_luma, 99.5))

    mask = np.asarray(face_mask, dtype=np.float32)
    if mask.shape != luma.shape:
        mask = cv2.resize(mask, (luma.shape[1], luma.shape[0]), interpolation=cv2.INTER_LINEAR)
    face_luma = luma[mask > 0.5]
    median = float(np.percentile(face_luma, 50.0)) if face_luma.size >= 64 else float(np.percentile(flat_luma, 50.0))

    return _solve_auto_tone(median, black_point, white_point)


def suggest_global_auto_values(img: np.ndarray) -> dict:
    """Suggest starting Noise Reduc./Color NR/Sharpness slider values (0-100) from the
    image's own measured noise: clean images get little/no denoise and a normal
    capture-sharpening baseline; noisy images get more denoise and a reduced sharpening
    baseline so the default doesn't re-amplify grain. Meant as a fully user-adjustable
    starting point. Applied automatically once when an image opens -- unlike suggest_auto_tone,
    which is only applied on demand via the Auto / Auto Tone buttons."""
    sigma = estimate_noise_sigma(img)
    noise_red = float(np.clip((sigma - 1.5) * 6.0, 0.0, 70.0))
    sharpness = float(np.clip(25.0 - sigma * 2.0, 5.0, 25.0))
    # Color NR is a *boost above* noise_red's own (implicit) chroma treatment -- only suggest
    # one when chroma noise is measurably worse than luma noise (chroma sigma > luma sigma),
    # the signature of color blotches rather than ordinary grain that affects both channels
    # about equally. Capped lower than noise_red since it's a top-up, not the primary control.
    chroma_sigma = estimate_chroma_noise_sigma(img)
    chroma_excess = max(0.0, chroma_sigma - sigma)
    color_noise_red = float(np.clip(chroma_excess * 6.0, 0.0, 50.0))
    return {
        "noise_red": round(noise_red),
        "sharpness": round(sharpness),
        "color_noise_red": round(color_noise_red),
    }


def suggest_region_noise_red(img: np.ndarray, mask: np.ndarray | None) -> int:
    """Suggest a starting Noise Reduc. value (0-100) for a single masked region (Skin /
    Background / Person), using the same estimator and curve as the global suggestion but
    restricted to that region's own pixels via estimate_noise_sigma's mask support. Returns 0
    when there's no mask (region not present/detected in this image) rather than falling back
    to a whole-image guess -- a region that doesn't exist shouldn't get a denoise value."""
    if mask is None:
        return 0
    sigma = estimate_noise_sigma(img, mask)
    if sigma <= 0.0:
        return 0
    return int(round(float(np.clip((sigma - 1.5) * 6.0, 0.0, 70.0))))


def suggest_auto_subject(img: np.ndarray, subject_mask: np.ndarray, background_mask: np.ndarray | None = None) -> dict:
    """Suggest Subject/Background layer values for a one-click 'Auto Subject': expose for the
    subject and gently separate it from the background. Conservative by design -- every
    adjustment is damped/clamped and skipped when the image's own measurements say it won't
    help (don't darken an already-dark background, don't push an already-balanced subject).

    Returns {"subjects": {...}, "background": {...}} keyed by slider name; empty dicts mean
    "leave that layer alone". Region statistics only, so a mask scaled up from the preview is
    fine -- edge precision doesn't matter for medians.

    The analysis is done on a downscaled copy (region medians are scale-invariant), so it stays
    near-instant on large group photos instead of running a ~140ms full-resolution pass.
    """
    result: dict[str, dict] = {"subjects": {}, "background": {}}
    if subject_mask is None:
        return result

    sub_full = np.clip(np.asarray(subject_mask, dtype=np.float32), 0.0, 1.0)
    h, w = img.shape[:2]
    if sub_full.shape != (h, w):
        return result
    bg_full = None
    if background_mask is not None:
        bg_arr = np.clip(np.asarray(background_mask, dtype=np.float32), 0.0, 1.0)
        if bg_arr.shape == (h, w):
            bg_full = bg_arr

    # Downscale image + masks to a bounded analysis size; medians of a region don't change
    # meaningfully with resolution, so this is the same answer ~25x faster on a 26MP frame.
    max_dim = 1024
    if max(h, w) > max_dim:
        s = max_dim / float(max(h, w))
        size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
        small_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        sub = cv2.resize(sub_full, size, interpolation=cv2.INTER_AREA)
        bg = cv2.resize(bg_full, size, interpolation=cv2.INTER_AREA) if bg_full is not None else None
    else:
        small_img, sub, bg = img, sub_full, bg_full

    lum = _luma(small_img)
    sub_sel = sub > 0.5
    if float(sub_sel.mean()) < 0.02 or not sub_sel.any():
        return result  # no meaningful subject to expose for

    bg_sel = (bg > 0.5) if bg is not None else ~sub_sel

    sub_median = float(np.median(lum[sub_sel]))

    # Subject: expose toward a normal midtone (0.45), damped to 80% and clamped to +/-0.75
    # stop so a strongly back/under-exposed subject is lifted without blowing out.
    ev = math.log2(0.45 / max(sub_median, 1e-3))
    ev = float(np.clip(ev * 0.8, -0.75, 0.75))
    result["subjects"]["exposure"] = int(round(ev * 100))
    result["subjects"]["clarity"] = 10  # subtle presence

    # Background separation only when there is enough background to matter.
    if float(bg_sel.mean()) >= 0.08 and bg_sel.any():
        bg_median = float(np.median(lum[bg_sel]))
        gap = bg_median - sub_median  # > 0: background is brighter than the subject
        bg_ev = -float(np.clip(gap, 0.0, 0.4)) if gap > 0.05 else 0.0
        result["background"]["exposure"] = int(round(bg_ev * 100))
        result["background"]["saturation"] = -12  # ease background color competition

    return result


def _suggest_subject_crop(subject_mask, h: int, w: int, target_aspect: float | None) -> list[float] | None:
    """Legacy subject-box crop fallback.

    The public Auto Crop path now uses the scored candidate crop below. This simple bbox
    helper is kept for conservative fallback behavior when candidate scoring cannot produce
    a crop.
    """
    if subject_mask is None:
        return None
    mask = np.asarray(subject_mask, dtype=np.float32)
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    sel = mask > 0.5
    if not sel.any():
        return None

    ys, xs = np.where(sel)
    x0, x1 = float(xs.min()), float(xs.max()) + 1.0
    y0, y1 = float(ys.min()), float(ys.max()) + 1.0
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None

    margin_x, margin_y = bw * 0.12, bh * 0.12
    bx0, bx1 = max(0.0, x0 - margin_x), min(float(w), x1 + margin_x)
    by0, by1 = max(0.0, y0 - margin_y), min(float(h), y1 + margin_y)
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    bw, bh = bx1 - bx0, by1 - by0

    if target_aspect and target_aspect > 0:
        frame_aspect = w / h
        if target_aspect >= frame_aspect:
            crop_w, crop_h = float(w), float(w) / target_aspect
        else:
            crop_h, crop_w = float(h), float(h) * target_aspect
    else:
        crop_w, crop_h = bw, bh

    crop_w = min(crop_w, float(w))
    crop_h = min(crop_h, float(h))
    crop_x = float(np.clip(cx - crop_w / 2.0, 0.0, w - crop_w))
    crop_y = float(np.clip(cy - crop_h / 2.0, 0.0, h - crop_h))

    nx, ny, nw, nh = crop_x / w, crop_y / h, crop_w / w, crop_h / h
    if nw > 0.96 and nh > 0.96:
        return None  # already ~full frame -- nothing meaningful to crop
    return [nx, ny, nw, nh]


def _normalized_crop_to_pixels(crop: list[float], h: int, w: int) -> tuple[float, float, float, float]:
    x, y, cw, ch = crop
    return float(x) * w, float(y) * h, float(cw) * w, float(ch) * h


def _crop_from_center(cx: float, cy: float, crop_w: float, crop_h: float, frame_w: int, frame_h: int) -> list[float]:
    crop_w = float(np.clip(crop_w, 1.0, float(frame_w)))
    crop_h = float(np.clip(crop_h, 1.0, float(frame_h)))
    x = float(np.clip(cx - crop_w / 2.0, 0.0, float(frame_w) - crop_w))
    y = float(np.clip(cy - crop_h / 2.0, 0.0, float(frame_h) - crop_h))
    return [x / frame_w, y / frame_h, crop_w / frame_w, crop_h / frame_h]


def _max_crop_for_aspect(frame_w: int, frame_h: int, aspect: float) -> tuple[float, float]:
    frame_aspect = float(frame_w) / max(float(frame_h), 1e-6)
    if aspect >= frame_aspect:
        return float(frame_w), float(frame_w) / aspect
    return float(frame_h) * aspect, float(frame_h)


def _minimum_aspect_crop_for_bounds(bounds: tuple[float, float, float, float], aspect: float) -> tuple[float, float]:
    x0, y0, x1, y1 = bounds
    bw = max(1.0, float(x1 - x0))
    bh = max(1.0, float(y1 - y0))
    if bw / bh >= aspect:
        return bw, bw / aspect
    return bh * aspect, bh


def _crop_mask_coverage(mask: np.ndarray, crop: list[float], h: int, w: int) -> float:
    total = float(np.clip(mask, 0.0, 1.0).sum())
    if total <= 1e-6:
        return 0.0
    x, y, cw, ch = _normalized_crop_to_pixels(crop, h, w)
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(w, int(math.ceil(x + cw)))
    y1 = min(h, int(math.ceil(y + ch)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.clip(mask[y0:y1, x0:x1], 0.0, 1.0).sum() / total)


def _box_containment(box: tuple[float, float, float, float], crop: list[float], h: int, w: int) -> float:
    bx0, by0, bx1, by1 = box
    cx, cy, cw, ch = _normalized_crop_to_pixels(crop, h, w)
    cx1, cy1 = cx + cw, cy + ch
    ix0, iy0 = max(bx0, cx), max(by0, cy)
    ix1, iy1 = min(bx1, cx1), min(by1, cy1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (bx1 - bx0) * (by1 - by0))
    return float(np.clip(inter / area, 0.0, 1.0))


def _guide_points_for_crop(guides) -> list[dict]:
    if guides is None:
        return []
    if isinstance(guides, list):
        return [item for item in guides if isinstance(item, dict)]
    if isinstance(guides, dict):
        return [guides]
    return []


def _eye_anchor_from_guides(guides) -> tuple[float, float] | None:
    points = []
    for guide in _guide_points_for_crop(guides):
        for key in ("left_eye_upper", "left_eye_lower", "right_eye_upper", "right_eye_lower"):
            pt = _point_from_guides(guide, key)
            if pt is not None:
                points.append(pt)
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float32)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _face_boxes_for_crop(faces) -> list[tuple[float, float, float, float]]:
    boxes = []
    for face in faces or []:
        if not isinstance(face, (tuple, list)) or len(face) != 4:
            continue
        x, y, fw, fh = (float(v) for v in face)
        if fw <= 0 or fh <= 0:
            continue
        boxes.append((x, y, x + fw, y + fh))
    return boxes


def _auto_crop_aspect_candidates(frame_w: int, frame_h: int, target_aspect: float | None) -> list[float]:
    if target_aspect and target_aspect > 0:
        return [float(target_aspect)]
    frame_aspect = float(frame_w) / max(float(frame_h), 1e-6)
    portrait = frame_h >= frame_w
    common = [frame_aspect]
    common.extend([4.0 / 5.0, 1.0, 2.0 / 3.0] if portrait else [4.0 / 5.0, 1.0, 3.0 / 2.0, 16.0 / 9.0])
    out = []
    for aspect in common:
        if aspect <= 0:
            continue
        if all(abs(aspect - existing) > 0.02 for existing in out):
            out.append(float(aspect))
    return out


def _score_crop_candidate(
    crop: list[float],
    subject_mask: np.ndarray,
    subject_bounds: tuple[float, float, float, float],
    face_boxes: list[tuple[float, float, float, float]],
    eye_anchor: tuple[float, float] | None,
    h: int,
    w: int,
    *,
    fixed_aspect: bool,
) -> float:
    x, y, cw, ch = _normalized_crop_to_pixels(crop, h, w)
    area_frac = max(1e-6, float(crop[2] * crop[3]))
    score = 0.0

    subject_coverage = _crop_mask_coverage(subject_mask, crop, h, w)
    score += subject_coverage * 5.0
    if subject_coverage < 0.985:
        score -= (0.985 - subject_coverage) * 18.0

    face_targets = face_boxes
    if face_targets:
        containments = [_box_containment(box, crop, h, w) for box in face_targets]
        score += float(np.mean(containments)) * 3.0
        if min(containments) < 0.98:
            score -= (0.98 - min(containments)) * 12.0

    sx0, sy0, sx1, sy1 = subject_bounds
    subject_cx = (sx0 + sx1) * 0.5
    subject_cy = (sy0 + sy1) * 0.5
    rel_sx = (subject_cx - x) / max(cw, 1e-6)
    rel_sy = (subject_cy - y) / max(ch, 1e-6)
    score += max(0.0, 1.0 - abs(rel_sx - 0.5) / 0.5) * 0.7
    score += max(0.0, 1.0 - abs(rel_sy - 0.52) / 0.52) * 0.4

    if eye_anchor is not None:
        eye_x, eye_y = eye_anchor
        rel_eye_x = (eye_x - x) / max(cw, 1e-6)
        rel_eye_y = (eye_y - y) / max(ch, 1e-6)
        score += max(0.0, 1.0 - abs(rel_eye_y - 0.36) / 0.28) * 1.6
        score += max(0.0, 1.0 - abs(rel_eye_x - 0.5) / 0.42) * 0.8
        if rel_eye_y < 0.12 or rel_eye_y > 0.58 or rel_eye_x < 0.10 or rel_eye_x > 0.90:
            score -= 2.5
    elif face_targets:
        top = min(box[1] for box in face_targets)
        rel_top = (top - y) / max(ch, 1e-6)
        score += max(0.0, 1.0 - abs(rel_top - 0.16) / 0.24) * 0.8

    # Prefer meaningful crops over full-frame suggestions, but avoid brittle over-tight crops.
    sx_area = max(1.0, (sx1 - sx0) * (sy1 - sy0)) / max(1.0, float(w * h))
    lower_target = min(0.75, max(0.18, sx_area * 1.55))
    upper_target = 0.88 if fixed_aspect else 0.72
    if area_frac > upper_target:
        score -= (area_frac - upper_target) * 2.0
    if area_frac < lower_target:
        score -= (lower_target - area_frac) * 3.0

    return float(score)


def _suggest_scored_subject_crop(
    img: np.ndarray | None,
    subject_mask,
    h: int,
    w: int,
    target_aspect: float | None,
    *,
    faces=None,
    guides=None,
    aesthetic_scorer=None,
) -> list[float] | None:
    if subject_mask is None:
        return None
    mask = np.asarray(subject_mask, dtype=np.float32)
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)
    bbox = _mask_bbox(mask, threshold=0.5)
    if bbox is None:
        return None

    x0, y0, x1, y1 = (float(v) for v in bbox)
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None
    subject_area = (bw * bh) / max(1.0, float(w * h))
    if subject_area > 0.90 and not target_aspect:
        return None

    margin_x = max(8.0, bw * 0.10)
    margin_y_top = max(10.0, bh * 0.14)
    margin_y_bottom = max(8.0, bh * 0.10)
    bounds = (
        max(0.0, x0 - margin_x),
        max(0.0, y0 - margin_y_top),
        min(float(w), x1 + margin_x),
        min(float(h), y1 + margin_y_bottom),
    )
    bx0, by0, bx1, by1 = bounds
    subject_cx = (bx0 + bx1) * 0.5
    subject_cy = (by0 + by1) * 0.5
    eye_anchor = _eye_anchor_from_guides(guides)
    face_boxes = _face_boxes_for_crop(faces)

    # Faces are never optional in the candidate bounds: a crop that keeps the subject mask but
    # nicks a forehead/chin is worse than a looser crop.
    for face in face_boxes:
        fx0, fy0, fx1, fy1 = face
        fw, fh = fx1 - fx0, fy1 - fy0
        bx0 = min(bx0, max(0.0, fx0 - fw * 0.18))
        bx1 = max(bx1, min(float(w), fx1 + fw * 0.18))
        by0 = min(by0, max(0.0, fy0 - fh * 0.40))
        by1 = max(by1, min(float(h), fy1 + fh * 0.20))
    bounds = (bx0, by0, bx1, by1)

    candidates: list[list[float]] = []
    for aspect in _auto_crop_aspect_candidates(w, h, target_aspect):
        max_w, max_h = _max_crop_for_aspect(w, h, aspect)
        min_w, min_h = _minimum_aspect_crop_for_bounds(bounds, aspect)
        for scale in (1.00, 1.12, 1.28, 1.48):
            crop_w = min(max_w, min_w * scale)
            crop_h = min(max_h, min_h * scale)
            centers = [(subject_cx, subject_cy)]
            if eye_anchor is not None:
                eye_x, eye_y = eye_anchor
                centers.append((eye_x, eye_y + (0.36 - 0.50) * crop_h))
                centers.append((subject_cx, eye_y + (0.36 - 0.50) * crop_h))
            for cx, cy in centers:
                crop = _crop_from_center(cx, cy, crop_w, crop_h, w, h)
                if all(sum(abs(a - b) for a, b in zip(crop, existing)) > 1e-4 for existing in candidates):
                    candidates.append(crop)

    if not candidates:
        return _suggest_subject_crop(mask, h, w, target_aspect)

    fixed_aspect = bool(target_aspect and target_aspect > 0)
    scored_candidates = [
        (
            _score_crop_candidate(
                crop, mask, bounds, face_boxes, eye_anchor, h, w, fixed_aspect=fixed_aspect
            ),
            crop,
        )
        for crop in candidates
    ]
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best = scored_candidates[0][1]

    if aesthetic_scorer is not None and img is not None:
        reranked = []
        # Keep the model as a tie-breaker among safe candidates. Running only the top few
        # also keeps Auto Crop responsive even with a real ONNX scorer installed.
        for base_score, crop in scored_candidates[:8]:
            aesthetic_score = score_crop_aesthetic(img, crop, aesthetic_scorer)
            if aesthetic_score is None:
                continue
            combined_score = base_score + (aesthetic_score - 0.5) * 1.2
            reranked.append((combined_score, base_score, crop))
        if reranked:
            best = max(reranked, key=lambda item: (item[0], item[1]))[2]

    if best[2] > 0.96 and best[3] > 0.96:
        return None
    return best


def _suggest_horizon_angle(
    img: np.ndarray,
    subject_mask: np.ndarray | None,
    max_tilt: float = 12.0,
    min_lines: int = 4,
    max_angle_std: float = 2.5,
) -> float | None:
    """Classical (no model) horizon-straighten angle: Canny edges + Hough line detection,
    restricted to the non-subject region so shoulders/limbs can't be mistaken for a horizon.
    Deliberately conservative -- returns None (no suggestion) unless enough long,
    near-horizontal lines agree on the tilt; a single doorframe is not a horizon. Sign
    matches framing.py's convention (positive straightens a clockwise-tilted horizon)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(to_uint8(np.clip(img, 0.0, 1.0)), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    if subject_mask is not None:
        mask = np.asarray(subject_mask, dtype=np.float32)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        subj_u8 = (mask > 0.5).astype(np.uint8) * 255
        subj_u8 = cv2.dilate(subj_u8, np.ones((15, 15), np.uint8))
        edges[subj_u8 > 0] = 0

    min_len = max(20, int(w * 0.18))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return None

    angles = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dx < 0:
            dx, dy = -dx, -dy
        if dx < 1e-3:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) <= max_tilt:
            angles.append(angle)

    if len(angles) < min_lines:
        return None
    angles_arr = np.asarray(angles, dtype=np.float64)
    if float(np.std(angles_arr)) > max_angle_std:
        return None  # lines disagree too much to be a real horizon -- stay silent
    median_angle = float(np.median(angles_arr))
    if abs(median_angle) < 0.4:
        return None  # already straight; nothing meaningful to suggest
    return float(np.clip(median_angle, -max_tilt, max_tilt))


def suggest_auto_crop(
    img: np.ndarray,
    subject_mask: np.ndarray | None,
    target_aspect: float | None = None,
    *,
    faces=None,
    guides=None,
    aesthetic_scorer=_AUTO_AESTHETIC_SCORER,
) -> dict:
    """Suggest a model-informed crop and, when confident, a horizon-straighten angle.

    The crop is chosen by scoring candidate rectangles against model-derived subject masks,
    detected faces, and optional landmark/guide points. Returns only the keys it has something
    useful to say about -- {} means leave the framing alone.
    """
    result: dict = {}
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return result

    scorer = None
    if subject_mask is not None:
        scorer = get_aesthetic_crop_scorer() if aesthetic_scorer is _AUTO_AESTHETIC_SCORER else aesthetic_scorer

    crop = _suggest_scored_subject_crop(
        img,
        subject_mask,
        h,
        w,
        target_aspect,
        faces=faces,
        guides=guides,
        aesthetic_scorer=scorer,
    )
    if crop is not None:
        result["crop"] = crop

    angle = _suggest_horizon_angle(img, subject_mask)
    if angle is not None:
        result["angle"] = angle

    return result


def _frequency_smooth_skin(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    # Two-band split: preserve higher-frequency pore/edge structure while
    # smoothing only the broader low-frequency skin transitions.
    low_sigma = 1.5 + amount * 6.0
    detail_sigma = 0.8 + amount * 1.8
    low_band = gaussian_blur(img, sigma=low_sigma, acceleration=acceleration)
    detail_band = img - gaussian_blur(img, sigma=detail_sigma, acceleration=acceleration)

    detail_mix = 1.0 - amount * 0.45
    smoothed = low_band + detail_band * detail_mix

    # Recover stronger edges from a mild unsharp residual so skin stays soft
    # without collapsing facial boundaries.
    edge_residual = img - gaussian_blur(img, sigma=2.2 + amount * 1.8, acceleration=acceleration)
    edge_weight = 0.10 + amount * 0.12
    return clamp01(smoothed + edge_residual * edge_weight)


def _frequency_blemish_soften(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    base = gaussian_blur(img, sigma=2.0 + amount * 4.5, acceleration=acceleration)
    coarse = gaussian_blur(img, sigma=5.0 + amount * 7.0, acceleration=acceleration)
    fine_detail = img - base
    fine_strength = 1.0 - amount * 0.75
    return clamp01(coarse + fine_detail * fine_strength)


def _luma(img: np.ndarray) -> np.ndarray:
    return np.clip(
        img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722,
        0.0,
        1.0,
    ).astype(np.float32)


def _edge_protection_mask(
    img: np.ndarray,
    *,
    acceleration: str = "auto",
    strength: float = 2.5,
    blur_sigma: float = 1.2,
) -> np.ndarray:
    lum = _luma(img)
    gx = np.zeros_like(lum, dtype=np.float32)
    gy = np.zeros_like(lum, dtype=np.float32)
    gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) * 0.5
    gy[1:-1, :] = (lum[2:, :] - lum[:-2, :]) * 0.5
    edge = np.sqrt(gx * gx + gy * gy)
    if blur_sigma > 0:
        edge = gaussian_blur(edge, sigma=blur_sigma, acceleration=acceleration)
    return np.clip(1.0 - edge * float(strength), 0.0, 1.0).astype(np.float32)


def _mix_with_effect_mask(base: np.ndarray, effected: np.ndarray, effect_mask: np.ndarray) -> np.ndarray:
    mask3 = np.clip(effect_mask, 0.0, 1.0)[:, :, np.newaxis]
    return clamp01(base * (1.0 - mask3) + effected * mask3)


def _mask_bbox(mask: np.ndarray, threshold: float = 0.10) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(np.clip(mask.astype(np.float32), 0.0, 1.0) > float(threshold))
    if coords.size == 0:
        return None
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1
    return int(x1), int(y1), int(x2), int(y2)


def _gaussian_blob(shape_hw: tuple[int, int], cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    h, w = shape_hw
    if sx <= 0 or sy <= 0 or h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
    return field.astype(np.float32)


def _remap_array(arr: np.ndarray, dx: np.ndarray, dy: np.ndarray, interpolation: int) -> np.ndarray:
    h, w = arr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x - dx.astype(np.float32), 0.0, max(0, w - 1)).astype(np.float32)
    map_y = np.clip(grid_y - dy.astype(np.float32), 0.0, max(0, h - 1)).astype(np.float32)
    remapped = cv2.remap(arr.astype(np.float32), map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_REFLECT_101)
    return remapped.astype(np.float32)


def _point_from_guides(guides: dict | None, key: str) -> tuple[float, float] | None:
    if not guides or key not in guides:
        return None
    value = guides.get(key)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _scale_expression_guides(guides, scale_x: float, scale_y: float):
    if guides is None:
        return None
    if isinstance(guides, list):
        return [_scale_expression_guides(item, scale_x, scale_y) for item in guides]
    scaled = {}
    for key, value in guides.items():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            scaled[key] = (float(value[0]) * scale_x, float(value[1]) * scale_y)
        else:
            scaled[key] = value
    return scaled


def _offset_expression_guides(guides, dx: float, dy: float):
    """Translate landmark guide points by (dx, dy) pixels -- the crop-origin counterpart
    to ``_scale_expression_guides``, used to shift ``full_guides`` into a sub-crop's
    local coordinate frame before running the warp stage on that crop alone."""
    if guides is None:
        return None
    if isinstance(guides, list):
        return [_offset_expression_guides(item, dx, dy) for item in guides]
    offset = {}
    for key, value in guides.items():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            offset[key] = (float(value[0]) + dx, float(value[1]) + dy)
        else:
            offset[key] = value
    return offset


def expression_warp_mode(face_params: dict | None, geometry: dict | list[dict] | None) -> str:
    if not face_params:
        return "off"
    active = any(
        abs(float(face_params.get(key, 0.0))) > 1e-4
        for key in ("smile", "eye_open", "brow_lift", "mouth_open", "jaw_relax")
    )
    if not active:
        return "off"
    return "landmark-guided" if geometry else "mask-guided"


def _apply_expression_warp(
    img: np.ndarray,
    masks: dict | None,
    face_params: dict | None,
    geometry: dict | list[dict] | None = None,
) -> tuple[np.ndarray, dict | None]:
    if masks is None or not face_params:
        return img, masks

    smile = float(face_params.get("smile", 0.0)) / 100.0
    eye_open = float(face_params.get("eye_open", 0.0)) / 100.0
    brow_lift = float(face_params.get("brow_lift", 0.0)) / 100.0
    mouth_open = float(face_params.get("mouth_open", 0.0)) / 100.0
    jaw_relax = float(face_params.get("jaw_relax", 0.0)) / 100.0
    if abs(smile) < 1e-4 and abs(eye_open) < 1e-4 and abs(brow_lift) < 1e-4 and abs(mouth_open) < 1e-4 and abs(jaw_relax) < 1e-4:
        return img, masks

    face_mask = np.clip(np.asarray((masks or {}).get("face", 0.0), dtype=np.float32), 0.0, 1.0)
    face_box = _mask_bbox(face_mask, threshold=0.08)
    if face_box is None:
        return img, masks

    h, w = img.shape[:2]
    fx1, fy1, fx2, fy2 = face_box
    fw = max(1.0, float(fx2 - fx1))
    fh = max(1.0, float(fy2 - fy1))
    face_support = smooth_mask(face_mask, sigma=max(1.5, fw * 0.015))
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    if isinstance(geometry, list):
        geometries = [item for item in geometry if isinstance(item, dict)]
    elif isinstance(geometry, dict):
        geometries = [geometry]
    else:
        geometries = []

    lips_mask = np.clip(np.asarray((masks or {}).get("lips", 0.0), dtype=np.float32), 0.0, 1.0)
    lips_box = _mask_bbox(lips_mask, threshold=0.08)
    if lips_box is None:
        lips_box = (
            int(fx1 + fw * 0.28),
            int(fy1 + fh * 0.64),
            int(fx1 + fw * 0.72),
            int(fy1 + fh * 0.84),
        )
        lips_mask = _gaussian_blob((h, w), (lips_box[0] + lips_box[2]) * 0.5, (lips_box[1] + lips_box[3]) * 0.5, fw * 0.16, fh * 0.09)

    if abs(smile) >= 1e-4:
        guide = geometries[0] if geometries else None
        left_pt = _point_from_guides(guide, "mouth_left")
        right_pt = _point_from_guides(guide, "mouth_right")
        upper_pt = _point_from_guides(guide, "mouth_upper")
        lower_pt = _point_from_guides(guide, "mouth_lower")
        center_pt = _point_from_guides(guide, "mouth_center")

        if left_pt and right_pt and upper_pt and lower_pt:
            lip_w = max(1.0, abs(right_pt[0] - left_pt[0]))
            lip_h = max(1.0, abs(lower_pt[1] - upper_pt[1]))
            if center_pt is None:
                center_pt = (
                    (left_pt[0] + right_pt[0] + upper_pt[0] + lower_pt[0]) * 0.25,
                    (left_pt[1] + right_pt[1] + upper_pt[1] + lower_pt[1]) * 0.25,
                )
            left_corner = _gaussian_blob((h, w), left_pt[0], left_pt[1], max(2.0, lip_w * 0.34), max(2.0, lip_h * 1.8))
            right_corner = _gaussian_blob((h, w), right_pt[0], right_pt[1], max(2.0, lip_w * 0.34), max(2.0, lip_h * 1.8))
            mouth_center = _gaussian_blob((h, w), center_pt[0], center_pt[1], max(2.0, lip_w * 0.45), max(2.0, lip_h * 2.2))
        else:
            lx1, ly1, lx2, ly2 = lips_box
            lip_w = max(1.0, float(lx2 - lx1))
            lip_h = max(1.0, float(ly2 - ly1))
            left_corner = _gaussian_blob((h, w), lx1 + lip_w * 0.10, ly1 + lip_h * 0.45, fw * 0.11, fh * 0.10)
            right_corner = _gaussian_blob((h, w), lx2 - lip_w * 0.10, ly1 + lip_h * 0.45, fw * 0.11, fh * 0.10)
            mouth_center = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly1 + lip_h * 0.60, fw * 0.15, fh * 0.09)
        smile_support = smooth_mask(np.clip((lips_mask * 0.6 + mouth_center * 0.4) * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.006))
        corner_lift = fh * 0.110 * smile
        corner_out = fw * 0.040 * smile
        center_drop = fh * 0.030 * smile
        dy += (-corner_lift * left_corner) + (-corner_lift * right_corner) + (center_drop * mouth_center)
        dx += (-corner_out * left_corner) + (corner_out * right_corner)
        cheek_pull_left = _gaussian_blob((h, w), fx1 + fw * 0.18, fy1 + fh * 0.58, fw * 0.12, fh * 0.16)
        cheek_pull_right = _gaussian_blob((h, w), fx2 - fw * 0.18, fy1 + fh * 0.58, fw * 0.12, fh * 0.16)
        dx += (-fw * 0.015 * smile * cheek_pull_left) + (fw * 0.015 * smile * cheek_pull_right)
        dy *= np.clip(0.35 + smile_support * 0.65, 0.0, 1.0)
        dx *= np.clip(0.30 + smile_support * 0.70, 0.0, 1.0)

    if abs(mouth_open) >= 1e-4 or abs(jaw_relax) >= 1e-4:
        guide = geometries[0] if geometries else None
        upper_pt = _point_from_guides(guide, "mouth_upper")
        lower_pt = _point_from_guides(guide, "mouth_lower")
        center_pt = _point_from_guides(guide, "mouth_center")

        if upper_pt and lower_pt:
            mouth_w = max(2.0, fw * 0.18)
            mouth_h = max(2.0, abs(lower_pt[1] - upper_pt[1]) * 2.4)
            upper_blob = _gaussian_blob((h, w), upper_pt[0], upper_pt[1], mouth_w, mouth_h)
            lower_blob = _gaussian_blob((h, w), lower_pt[0], lower_pt[1], mouth_w, mouth_h * 1.1)
            center = center_pt if center_pt else ((upper_pt[0] + lower_pt[0]) * 0.5, (upper_pt[1] + lower_pt[1]) * 0.5)
            jaw_blob = _gaussian_blob((h, w), center[0], lower_pt[1] + fh * 0.11, fw * 0.28, fh * 0.18)
        else:
            lx1, ly1, lx2, ly2 = lips_box
            lip_w = max(1.0, float(lx2 - lx1))
            lip_h = max(1.0, float(ly2 - ly1))
            upper_blob = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly1 + lip_h * 0.32, lip_w * 0.55, lip_h * 0.90)
            lower_blob = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly1 + lip_h * 0.78, lip_w * 0.55, lip_h * 0.95)
            jaw_blob = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly2 + fh * 0.10, fw * 0.28, fh * 0.18)

        mouth_support = smooth_mask(np.clip((lips_mask * 0.7 + jaw_blob * 0.3) * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.008))
        open_push = fh * 0.060 * mouth_open
        jaw_push = fh * 0.050 * jaw_relax
        dy += (-open_push * 0.35 * upper_blob + open_push * lower_blob) * mouth_support
        dy += jaw_push * jaw_blob * mouth_support
        dx += jaw_relax * fw * 0.010 * (_gaussian_blob((h, w), fx1 + fw * 0.26, fy2 - fh * 0.10, fw * 0.10, fh * 0.14) * -1.0)
        dx += jaw_relax * fw * 0.010 * (_gaussian_blob((h, w), fx2 - fw * 0.26, fy2 - fh * 0.10, fw * 0.10, fh * 0.14))

    eyes_mask = np.clip(np.asarray((masks or {}).get("eyes", 0.0), dtype=np.float32), 0.0, 1.0)
    eyes_box = _mask_bbox(eyes_mask, threshold=0.08)
    if abs(eye_open) >= 1e-4:
        eye_boxes: list[tuple[int, int, int, int]] = []
        eye_guides = []
        guide = geometries[0] if geometries else None
        if guide:
            for prefix in ("left", "right"):
                upper_pt = _point_from_guides(guide, f"{prefix}_eye_upper")
                lower_pt = _point_from_guides(guide, f"{prefix}_eye_lower")
                center_pt = _point_from_guides(guide, f"{prefix}_eye_center")
                if upper_pt and lower_pt:
                    if center_pt is None:
                        center_pt = ((upper_pt[0] + lower_pt[0]) * 0.5, (upper_pt[1] + lower_pt[1]) * 0.5)
                    eye_guides.append((upper_pt, lower_pt, center_pt))
        if eye_guides:
            for upper_pt, lower_pt, center_pt in eye_guides:
                ew = max(2.0, fw * 0.14)
                eh = max(2.0, abs(lower_pt[1] - upper_pt[1]) * 2.0)
                upper = _gaussian_blob((h, w), upper_pt[0], upper_pt[1], ew, eh)
                lower = _gaussian_blob((h, w), lower_pt[0], lower_pt[1], ew, eh)
                eye_support = smooth_mask(np.clip((upper + lower) * face_support, 0.0, 1.0), sigma=max(0.6, ew * 0.05))
                eye_push = max(1.0, abs(lower_pt[1] - upper_pt[1]) * 1.8) * eye_open
                dy += (-eye_push * upper + eye_push * lower) * eye_support
        elif eyes_box is not None:
            ex1, ey1, ex2, ey2 = eyes_box
            mid_x = fx1 + fw * 0.5
            left_mask = eyes_mask.copy()
            left_mask[:, int(mid_x):] = 0.0
            right_mask = eyes_mask.copy()
            right_mask[:, :int(mid_x)] = 0.0
            left_box = _mask_bbox(left_mask, threshold=0.08)
            right_box = _mask_bbox(right_mask, threshold=0.08)
            eye_boxes = [b for b in (left_box, right_box) if b is not None]
        if not eye_boxes:
            eye_boxes = [
                (int(fx1 + fw * 0.18), int(fy1 + fh * 0.24), int(fx1 + fw * 0.42), int(fy1 + fh * 0.44)),
                (int(fx1 + fw * 0.58), int(fy1 + fh * 0.24), int(fx1 + fw * 0.82), int(fy1 + fh * 0.44)),
            ]
        if not eye_guides:
            for ex1, ey1, ex2, ey2 in eye_boxes:
                ew = max(1.0, float(ex2 - ex1))
                eh = max(1.0, float(ey2 - ey1))
                cx = (ex1 + ex2) * 0.5
                upper = _gaussian_blob((h, w), cx, ey1 + eh * 0.28, ew * 0.42, eh * 0.32)
                lower = _gaussian_blob((h, w), cx, ey1 + eh * 0.72, ew * 0.42, eh * 0.32)
                eye_support = smooth_mask(np.clip((upper + lower) * face_support, 0.0, 1.0), sigma=max(0.6, ew * 0.03))
                eye_push = eh * 0.32 * eye_open
                dy += (-eye_push * upper + eye_push * lower) * eye_support

    if abs(brow_lift) >= 1e-4:
        brow_guides = []
        guide = geometries[0] if geometries else None
        if guide:
            for brow_key, eye_key in (("left_brow", "left_eye_center"), ("right_brow", "right_eye_center")):
                brow_pt = _point_from_guides(guide, brow_key)
                eye_pt = _point_from_guides(guide, eye_key)
                if brow_pt and eye_pt:
                    brow_guides.append((brow_pt, eye_pt))
        if not brow_guides:
            brow_guides = [
                ((fx1 + fw * 0.32, fy1 + fh * 0.20), (fx1 + fw * 0.32, fy1 + fh * 0.34)),
                ((fx1 + fw * 0.68, fy1 + fh * 0.20), (fx1 + fw * 0.68, fy1 + fh * 0.34)),
            ]

        for brow_pt, eye_pt in brow_guides:
            span_y = max(2.0, abs(eye_pt[1] - brow_pt[1]))
            span_x = max(2.0, fw * 0.12)
            brow_blob = _gaussian_blob((h, w), brow_pt[0], brow_pt[1], span_x, span_y * 1.4)
            forehead_blob = _gaussian_blob((h, w), brow_pt[0], brow_pt[1] - span_y * 0.55, span_x * 1.1, span_y * 1.8)
            brow_support = smooth_mask(np.clip((brow_blob * 0.7 + forehead_blob * 0.3) * face_support, 0.0, 1.0), sigma=max(0.8, span_x * 0.08))
            lift = max(1.0, span_y * 0.9) * brow_lift
            dy += (-lift * brow_blob - lift * 0.4 * forehead_blob) * brow_support

    support = np.clip(face_support, 0.0, 1.0)
    dx *= support
    dy *= support
    max_dx = fw * 0.14
    max_dy = fh * 0.14
    dx = np.clip(dx, -max_dx, max_dx)
    dy = np.clip(dy, -max_dy, max_dy)

    warped_img = clamp01(_remap_array(img, dx, dy, cv2.INTER_LINEAR))
    warped_masks = {}
    for key, value in (masks or {}).items():
        warped = _remap_array(np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0), dx, dy, cv2.INTER_LINEAR)
        warped_masks[key] = np.clip(warped, 0.0, 1.0).astype(np.float32)
    return warped_img, warped_masks


def _apply_face_refinement(
    img: np.ndarray,
    masks: dict | None,
    face_params: dict | None,
    *,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    if masks is None or not face_params:
        return img
    if _fast_interactive_preview(runtime_settings):
        return img
    amount = float(face_params.get("refine", 0.0)) / 100.0
    if amount <= 0.0:
        return img
    fidelity = float(face_params.get("refine_fidelity", 65.0)) / 100.0
    face_mask = (masks or {}).get("face")
    if face_mask is None:
        return img
    working_space = _working_space(color_settings)
    display_img = working_to_display(img, output_transform="srgb", working_space=working_space)
    refined = get_face_refiner().refine(display_img, face_mask, amount, fidelity=fidelity)
    return display_to_working(refined, working_space=working_space)


def _build_skin_protection_mask(
    img: np.ndarray,
    masks: dict | None,
    geometry: dict | list[dict] | None = None,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    if not masks or "face" not in masks:
        return np.zeros(img.shape[:2], dtype=np.float32)

    face_mask = np.clip(np.asarray(masks.get("face", 0.0), dtype=np.float32), 0.0, 1.0)
    if float(face_mask.max()) <= 0.01:
        return np.zeros(img.shape[:2], dtype=np.float32)

    face_box = _mask_bbox(face_mask, threshold=0.08)
    if face_box is None:
        return np.zeros(img.shape[:2], dtype=np.float32)

    h, w = img.shape[:2]
    fx1, fy1, fx2, fy2 = face_box
    fw = max(1.0, float(fx2 - fx1))
    fh = max(1.0, float(fy2 - fy1))
    face_support = smooth_mask(face_mask, sigma=max(1.0, fw * 0.012), acceleration=acceleration)

    if isinstance(geometry, list):
        guides = next((item for item in geometry if isinstance(item, dict)), None)
    elif isinstance(geometry, dict):
        guides = geometry
    else:
        guides = None

    lum = _luma(img)
    dark = np.clip((0.52 - lum) / 0.36, 0.0, 1.0).astype(np.float32)
    texture = np.abs(lum - gaussian_blur(lum, sigma=1.4, acceleration=acceleration))
    texture = np.clip(texture * 5.0, 0.0, 1.0).astype(np.float32)
    hairlike = np.clip(dark * 0.8 + texture * 0.7, 0.0, 1.0).astype(np.float32)

    protection = np.zeros((h, w), dtype=np.float32)

    brow_mask = np.clip(np.asarray((masks or {}).get("brows", 0.0), dtype=np.float32), 0.0, 1.0)
    if float(brow_mask.max()) > 0.01:
        protection = np.maximum(
            protection,
            smooth_mask(np.clip(brow_mask * face_support, 0.0, 1.0), sigma=max(0.6, fw * 0.006), acceleration=acceleration),
        )

    facial_hair_mask = np.clip(np.asarray((masks or {}).get("facial_hair", 0.0), dtype=np.float32), 0.0, 1.0)
    if float(facial_hair_mask.max()) > 0.01:
        protection = np.maximum(
            protection,
            smooth_mask(np.clip(facial_hair_mask * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.008), acceleration=acceleration),
        )

    brow_guides = []
    if guides:
        for brow_key, eye_key in (("left_brow", "left_eye_center"), ("right_brow", "right_eye_center")):
            brow_pt = _point_from_guides(guides, brow_key)
            eye_pt = _point_from_guides(guides, eye_key)
            if brow_pt and eye_pt:
                brow_guides.append((brow_pt, eye_pt))
    if not brow_guides:
        brow_guides = [
            ((fx1 + fw * 0.32, fy1 + fh * 0.20), (fx1 + fw * 0.32, fy1 + fh * 0.34)),
            ((fx1 + fw * 0.68, fy1 + fh * 0.20), (fx1 + fw * 0.68, fy1 + fh * 0.34)),
        ]
    for brow_pt, eye_pt in brow_guides:
        span_y = max(2.0, abs(eye_pt[1] - brow_pt[1]))
        span_x = max(2.0, fw * 0.11)
        brow_blob = _gaussian_blob((h, w), brow_pt[0], brow_pt[1], span_x, span_y * 1.2)
        protection = np.maximum(protection, brow_blob * np.clip(hairlike * 0.85 + 0.15, 0.0, 1.0))

    mouth_center = _point_from_guides(guides, "mouth_center") if guides else None
    mouth_left = _point_from_guides(guides, "mouth_left") if guides else None
    mouth_right = _point_from_guides(guides, "mouth_right") if guides else None
    mouth_upper = _point_from_guides(guides, "mouth_upper") if guides else None
    mouth_lower = _point_from_guides(guides, "mouth_lower") if guides else None

    if mouth_center and mouth_upper and mouth_lower:
        lip_w = max(2.0, abs((mouth_right or mouth_center)[0] - (mouth_left or mouth_center)[0]))
        lip_h = max(2.0, abs(mouth_lower[1] - mouth_upper[1]))
        moustache_blob = _gaussian_blob(
            (h, w),
            mouth_center[0],
            mouth_upper[1] - lip_h * 0.65,
            max(2.0, lip_w * 0.55),
            max(2.0, lip_h * 0.95),
        )
        beard_blob = _gaussian_blob(
            (h, w),
            mouth_center[0],
            mouth_lower[1] + fh * 0.16,
            max(2.0, lip_w * 0.75),
            max(2.0, fh * 0.18),
        )
    else:
        moustache_blob = _gaussian_blob((h, w), fx1 + fw * 0.50, fy1 + fh * 0.63, fw * 0.19, fh * 0.06)
        beard_blob = _gaussian_blob((h, w), fx1 + fw * 0.50, fy1 + fh * 0.82, fw * 0.26, fh * 0.16)

    facial_hair = np.clip((moustache_blob * 0.9 + beard_blob) * hairlike * face_support, 0.0, 1.0)
    protection = np.maximum(protection, facial_hair)

    protection = smooth_mask(np.clip(protection * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.010), acceleration=acceleration)
    return np.clip(protection, 0.0, 1.0).astype(np.float32)


def _confidence_curve(mask: np.ndarray, power: float = 1.0, floor: float = 0.0) -> np.ndarray:
    base = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if floor != 0.0:
        base = np.clip((base - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    if power != 1.0:
        base = np.power(base, float(power))
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _make_effect_masks(mask: np.ndarray, layer: str, *, acceleration: str = "auto") -> dict[str, np.ndarray]:
    base = np.clip(mask.astype(np.float32), 0.0, 1.0)
    broad = smooth_mask(_confidence_curve(base, power=0.8), sigma=1.0, acceleration=acceleration)
    blend = smooth_mask(_confidence_curve(base, power=1.1), sigma=0.8, acceleration=acceleration)
    detail = smooth_mask(_confidence_curve(base, power=1.7, floor=0.10), sigma=0.6, acceleration=acceleration)
    strict = smooth_mask(_confidence_curve(base, power=2.4, floor=0.18), sigma=0.5, acceleration=acceleration)

    if layer == "skin":
        broad = smooth_mask(_confidence_curve(base, power=0.7), sigma=1.2, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=0.95), sigma=0.9, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.5, floor=0.08), sigma=0.7, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.0, floor=0.14), sigma=0.6, acceleration=acceleration)
    elif layer == "background":
        broad = smooth_mask(_confidence_curve(base, power=0.65), sigma=1.6, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=0.85), sigma=1.2, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.25, floor=0.04), sigma=0.9, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=1.7, floor=0.08), sigma=0.8, acceleration=acceleration)
    elif layer in {"subjects", "person"}:
        broad = smooth_mask(_confidence_curve(base, power=0.75), sigma=1.2, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=0.95), sigma=1.0, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.4, floor=0.06), sigma=0.8, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=1.9, floor=0.12), sigma=0.7, acceleration=acceleration)
    elif layer == "face":
        broad = smooth_mask(_confidence_curve(base, power=0.75), sigma=1.1, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=1.0), sigma=0.9, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.6, floor=0.10), sigma=0.7, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.1, floor=0.16), sigma=0.6, acceleration=acceleration)
    elif layer in {"eyes", "lips"}:
        broad = smooth_mask(_confidence_curve(base, power=1.2, floor=0.06), sigma=0.8, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=1.5, floor=0.10), sigma=0.7, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=2.1, floor=0.16), sigma=0.5, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.8, floor=0.22), sigma=0.4, acceleration=acceleration)
    elif layer == "hair":
        broad = smooth_mask(_confidence_curve(base, power=0.9, floor=0.03), sigma=1.0, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=1.1, floor=0.06), sigma=0.9, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.8, floor=0.12), sigma=0.6, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.2, floor=0.16), sigma=0.5, acceleration=acceleration)

    return {
        "blend": np.clip(blend, 0.0, 1.0).astype(np.float32),
        "broad": np.clip(broad, 0.0, 1.0).astype(np.float32),
        "detail": np.clip(detail, 0.0, 1.0).astype(np.float32),
        "strict": np.clip(strict, 0.0, 1.0).astype(np.float32),
    }


def _apply_micro_contrast(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
    detail_sigma: float = 1.6,
    base_sigma: float = 6.0,
    edge_strength: float = 2.2,
) -> np.ndarray:
    amount = float(amount) / 100.0
    if amount == 0.0:
        return img

    fine_base = gaussian_blur(img, sigma=detail_sigma, acceleration=acceleration)
    coarse_base = gaussian_blur(img, sigma=base_sigma, acceleration=acceleration)
    fine_detail = img - fine_base
    local_contrast = fine_base - coarse_base

    edge_guard = _edge_protection_mask(img, acceleration=acceleration, strength=edge_strength, blur_sigma=1.0)
    effect_mask = smooth_mask(np.clip(0.35 + edge_guard * 0.65, 0.0, 1.0), sigma=0.8, acceleration=acceleration)

    enhanced = clamp01(
        img
        + local_contrast * amount * 0.85
        + fine_detail * amount * 0.35
    )
    if amount < 0.0:
        enhanced = clamp01(
            img
            + local_contrast * amount * 0.65
            + fine_detail * amount * 0.20
        )
    return _mix_with_effect_mask(img, enhanced, effect_mask)


def _hair_strand_mask(
    img: np.ndarray,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    lum = _luma(img)
    fine = np.abs(lum - gaussian_blur(lum, sigma=1.1, acceleration=acceleration))
    coarse = np.abs(lum - gaussian_blur(lum, sigma=3.2, acceleration=acceleration))
    strands = np.clip(fine * 5.0 + coarse * 2.0, 0.0, 1.0)
    strands = smooth_mask(strands, sigma=0.8, acceleration=acceleration)
    return np.clip(strands, 0.0, 1.0).astype(np.float32)


def _apply_hair_shine(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    lum = _luma(img)
    strands = _hair_strand_mask(img, acceleration=acceleration)
    base_highlights = np.clip((lum - 0.35) / 0.45, 0.0, 1.0)
    directional = np.clip(gaussian_blur(lum, sigma=2.0, acceleration=acceleration) - gaussian_blur(lum, sigma=5.5, acceleration=acceleration), 0.0, 1.0)
    shine_mask = smooth_mask(np.clip(base_highlights * 0.55 + strands * 0.55 + directional * 0.35, 0.0, 1.0), sigma=1.0, acceleration=acceleration)

    boosted = clamp01(img + img * shine_mask[:, :, np.newaxis] * (0.10 + amount * 0.28))
    bloom = gaussian_blur(boosted * shine_mask[:, :, np.newaxis], sigma=2.4, acceleration=acceleration)
    blended = clamp01(boosted + bloom * amount * 0.18)
    return _mix_with_effect_mask(img, blended, shine_mask)


def _apply_hair_depth(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(amount) / 100.0
    if amount == 0.0:
        return img

    lum = _luma(img)
    strands = _hair_strand_mask(img, acceleration=acceleration)
    shadow_band = np.clip((0.62 - lum) / 0.42, 0.0, 1.0)
    depth_mask = smooth_mask(np.clip(shadow_band * 0.7 + strands * 0.4, 0.0, 1.0), sigma=1.0, acceleration=acceleration)
    darker = clamp01(img * (1.0 - depth_mask[:, :, np.newaxis] * amount * 0.18))
    lighter = clamp01(img + depth_mask[:, :, np.newaxis] * amount * 0.10)
    target = darker if amount > 0 else lighter
    return _mix_with_effect_mask(img, target, depth_mask)


def _even_skin_chroma(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    rgb8 = to_uint8(img)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    a_base = gaussian_blur(a_chan, sigma=2.0 + amount * 5.0, acceleration=acceleration)
    b_base = gaussian_blur(b_chan, sigma=2.0 + amount * 5.0, acceleration=acceleration)
    chroma_diff = np.sqrt((a_chan - a_base) ** 2 + (b_chan - b_base) ** 2)

    even_mask = np.clip(chroma_diff / 14.0, 0.0, 1.0)
    even_mask *= np.clip(1.0 - np.abs((l_chan / 255.0) - 0.58) * 1.6, 0.25, 1.0)
    even_mask = smooth_mask(even_mask.astype(np.float32), sigma=1.0, acceleration=acceleration)

    a_even = a_chan * (1.0 - even_mask * amount * 0.55) + a_base * (even_mask * amount * 0.55)
    b_even = b_chan * (1.0 - even_mask * amount * 0.55) + b_base * (even_mask * amount * 0.55)

    out_lab = lab.copy()
    out_lab[:, :, 1] = np.clip(a_even, 0, 255)
    out_lab[:, :, 2] = np.clip(b_even, 0, 255)
    out_rgb = to_float(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return clamp01(out_rgb)


def _apply_eye_whitening(
    img: np.ndarray,
    amount: float,
    *,
    edge_guard: np.ndarray,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    rgb8 = to_uint8(img)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    lum = _luma(img)
    sclera_mask = np.clip((lum - 0.48) / 0.30, 0.0, 1.0)
    sclera_mask *= np.clip(1.0 - np.abs((a_chan - 128.0) / 30.0), 0.2, 1.0)
    sclera_mask *= edge_guard
    sclera_mask = smooth_mask(sclera_mask.astype(np.float32), sigma=0.8, acceleration=acceleration)

    l_target = np.clip(l_chan + (255.0 - l_chan) * amount * 0.18 * sclera_mask, 0, 255)
    a_target = a_chan * (1.0 - amount * 0.35 * sclera_mask) + 128.0 * (amount * 0.35 * sclera_mask)
    b_target = b_chan * (1.0 - amount * 0.45 * sclera_mask) + 128.0 * (amount * 0.45 * sclera_mask)

    out_lab = lab.copy()
    out_lab[:, :, 0] = l_target
    out_lab[:, :, 1] = np.clip(a_target, 0, 255)
    out_lab[:, :, 2] = np.clip(b_target, 0, 255)
    out_rgb = to_float(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return clamp01(out_rgb)


def _unify_face_tone(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    rgb8 = to_uint8(img)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    l_base = gaussian_blur(l_chan, sigma=3.5 + amount * 8.0, acceleration=acceleration)
    a_base = gaussian_blur(a_chan, sigma=3.0 + amount * 6.0, acceleration=acceleration)
    b_base = gaussian_blur(b_chan, sigma=3.0 + amount * 6.0, acceleration=acceleration)

    lum = _luma(img)
    midtone_focus = np.clip(1.0 - np.abs(lum - 0.55) * 1.8, 0.2, 1.0)
    edge_guard = _edge_protection_mask(img, acceleration=acceleration, strength=2.8, blur_sigma=1.2)
    unify_mask = smooth_mask(np.clip(midtone_focus * edge_guard, 0.0, 1.0), sigma=1.1, acceleration=acceleration)

    l_out = l_chan * (1.0 - unify_mask * amount * 0.32) + l_base * (unify_mask * amount * 0.32)
    a_out = a_chan * (1.0 - unify_mask * amount * 0.24) + a_base * (unify_mask * amount * 0.24)
    b_out = b_chan * (1.0 - unify_mask * amount * 0.24) + b_base * (unify_mask * amount * 0.24)

    out_lab = lab.copy()
    out_lab[:, :, 0] = np.clip(l_out, 0, 255)
    out_lab[:, :, 1] = np.clip(a_out, 0, 255)
    out_lab[:, :, 2] = np.clip(b_out, 0, 255)
    out_rgb = to_float(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return clamp01(out_rgb)


def process_global(
    img: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
    crop_origin: tuple[int, int] = (0, 0),
    full_shape: tuple[int, int] | None = None,
    debug_sink: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    img = img.copy()

    # White balance correction first: neutralize a color cast in linear light
    # before any tonal/stylistic grade is applied. Temperature (Kelvin) + Tint are
    # the canonical setting; default 6500K/0 is identity.
    cs = color_settings or {}
    wb_gains = kelvin_tint_to_gains(cs.get("wb_temp_k", NEUTRAL_K), cs.get("wb_tint", 0))
    img = apply_white_balance_gains(img, wb_gains, working_space=working_space)

    ev = p.get("exposure", 0) / 100.0
    img = clamp01(img * (2**ev))

    img = adjust_color_balance_preserve_chroma(
        img,
        temperature=p.get("temperature", 0),
        tint=p.get("tint", 0),
        working_space=working_space,
    )

    img = apply_tone_curve_preserve_chroma(
        img,
        p.get("blacks", 0),
        p.get("shadows", 0),
        p.get("midtones", 0),
        p.get("highlights", 0),
        p.get("whites", 0),
        working_space=working_space,
    )

    # Interactive tone curve shapes tonality on top of the band sliders.
    tone_points = cs.get("tone_curve")
    if tone_points:
        img = apply_curve(img, tone_points, working_space=working_space)

    clarity = p.get("clarity", 0)
    if clarity != 0:
        img = _apply_micro_contrast(img, clarity, acceleration=acceleration, detail_sigma=1.8, base_sigma=9.0, edge_strength=2.4)

    img = adjust_hsv_sat(img, p.get("vibrance", 0) * 0.5 + p.get("saturation", 0) * 0.5, working_space=working_space)
    img = adjust_hsv_sat(img, p.get("vibrance", 0) * 0.5, working_space=working_space)

    # HSL color mixer: per-hue-band hue/saturation/luminance grading.
    color_mixer = cs.get("color_mixer")
    if color_mixer:
        img = apply_color_mixer(img, color_mixer, working_space=working_space)

    noise_red = p.get("noise_red", 0) / 100.0
    color_noise_red = p.get("color_noise_red", 0) / 100.0
    if noise_red > 0 or color_noise_red > 0:
        img = _apply_luma_chroma_denoise(
            img, noise_red, color_noise_red, working_space=working_space, acceleration=acceleration,
            fast_preview=_fast_interactive_preview(runtime_settings),
            scene_linear_luma=_scene_linear_denoise(color_settings),
            use_learned=_use_learned_denoise(color_settings),
        )

    sharpness = p.get("sharpness", 0) / 100.0
    sharpen_masking = p.get("sharpen_masking", 8) / 100.0 * 0.5
    if debug_sink is not None:
        # Captured on the same image state the real sharpen step would see, regardless of
        # whether sharpness is currently on -- so the masking threshold can be dialed in
        # before turning sharpening up.
        debug_sink["sharpen_mask"] = compute_sharpen_mask_preview(
            img, working_space=working_space, edge_threshold=sharpen_masking, acceleration=acceleration,
        )
    if sharpness > 0:
        sharpen_radius = p.get("sharpen_radius", 140) / 100.0
        img = _apply_unsharp_mask(
            img, sharpness * 2.0, working_space=working_space, radius=sharpen_radius,
            edge_threshold=sharpen_masking, acceleration=acceleration,
        )

    glow = p.get("glow", 0) / 100.0
    if glow > 0:
        bloom = gaussian_blur(img, sigma=20, acceleration=acceleration)
        img = clamp01(img + bloom * glow * 0.35)

    vignette = p.get("vignette", 0) / 100.0
    if vignette != 0:
        # Position-dependent -- the only effect in this pipeline that is. When ``img`` is
        # a sub-crop of a larger image (hi-res tile rendering), ``full_shape``/``crop_origin``
        # let the radial falloff be computed against the *full* image's center, so the crop
        # gets the same vignette it would have gotten as part of a full-image render.
        h, w = full_shape if full_shape is not None else img.shape[:2]
        ox, oy = crop_origin
        y_grid, x_grid = np.ogrid[:img.shape[0], :img.shape[1]]
        x_grid = x_grid + ox
        y_grid = y_grid + oy
        dist = np.sqrt(((x_grid - w / 2) / (w / 2)) ** 2 + ((y_grid - h / 2) / (h / 2)) ** 2)
        vmask = 1.0 - np.clip(dist * abs(vignette), 0, 1)
        if vignette < 0:
            vmask = 2.0 - vmask
        img = clamp01(img * vmask[:, :, np.newaxis])

    return img


def process_subjects_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "subjects", acceleration=acceleration)
    edited = original.copy()

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        edited = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.6, base_sigma=6.8, edge_strength=2.3)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_person_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    """Same controls/behavior as process_subjects_layer, scoped to one person's full-body
    mask (see FaceSegmenter._person_mask) instead of every detected person at once."""
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "person", acceleration=acceleration)
    edited = original.copy()

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    # Regional noise reduction -- denoise before clarity, same reasoning as the global layer:
    # sharpening/micro-contrast on top of un-denoised pixels would re-amplify the very grain
    # this is meant to remove.
    noise_red = p.get("noise_red", 0) / 100.0
    if noise_red > 0:
        edited = _apply_luma_chroma_denoise(
            edited, noise_red, working_space=working_space, acceleration=acceleration,
            fast_preview=_fast_interactive_preview(runtime_settings),
            scene_linear_luma=_scene_linear_denoise(color_settings),
            use_learned=_use_learned_denoise(color_settings),
        )

    clarity = p.get("clarity", 0)
    if clarity != 0:
        edited = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.6, base_sigma=6.8, edge_strength=2.3)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_background_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    fast_preview = _fast_interactive_preview(runtime_settings)
    effect_masks = _make_effect_masks(mask, "background", acceleration=acceleration)
    edited = original.copy()

    # Regional noise reduction, before Blur/Dehaze -- an out-of-focus background carries no
    # wanted detail, so it's a safe place to denoise independently of how much (if any) Blur
    # is also applied.
    noise_red = p.get("noise_red", 0) / 100.0
    if noise_red > 0:
        edited = _apply_luma_chroma_denoise(
            edited, noise_red, working_space=working_space, acceleration=acceleration, fast_preview=fast_preview,
            scene_linear_luma=_scene_linear_denoise(color_settings),
            use_learned=_use_learned_denoise(color_settings),
        )

    blur = p.get("blur", 0) / 100.0
    if blur > 0:
        if fast_preview:
            blur = min(blur, 0.22)
        sigma = 1.0 + blur * 12.0
        blurred = gaussian_blur(edited, sigma=sigma, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, blurred, effect_masks["broad"])

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        edited = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=2.0, base_sigma=10.0, edge_strength=1.8)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    dehaze = p.get("dehaze", 0) / 100.0
    if dehaze > 0:
        if fast_preview:
            dehaze = min(dehaze, 0.18)
        lum = _luma(edited)
        haze_mask = smooth_mask(np.clip(1.0 - lum * 1.15, 0.0, 1.0), sigma=2.2, acceleration=acceleration)
        dehazed = clamp01(edited + (edited - gaussian_blur(edited, sigma=7.0, acceleration=acceleration)) * dehaze * 0.85)
        edited = _mix_with_effect_mask(edited, dehazed, np.clip(haze_mask * effect_masks["detail"], 0.0, 1.0))

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_face_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    fast_preview = _fast_interactive_preview(runtime_settings)
    effect_masks = _make_effect_masks(mask, "face", acceleration=acceleration)
    edited = original.copy()

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    edited = apply_tone_curve_preserve_chroma(
        edited,
        0,
        0,
        0,
        p.get("highlights", 0),
        p.get("whites", 0) if "whites" in p else 0,
        working_space=working_space,
    )

    shadows = p.get("shadows", 0)
    if shadows != 0:
        edited = apply_tone_curve_preserve_chroma(edited, 0, shadows, 0, 0, 0, working_space=working_space)

    unify_amount = max(
        p.get("smooth", 0) / 100.0 * 0.55,
        abs(p.get("shadows", 0)) / 100.0 * 0.20,
        abs(p.get("highlights", 0)) / 100.0 * 0.18,
    )
    if unify_amount > 0:
        edited = _unify_face_tone(edited, unify_amount, acceleration=acceleration)

    clarity = p.get("clarity", 0)
    if clarity != 0:
        edited = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.5, base_sigma=7.0, edge_strength=2.6)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    sharpness = p.get("sharpness", 0) / 100.0
    if sharpness > 0:
        edited = _apply_sharpness(edited, 1 + sharpness * 4, working_space)

    smooth = p.get("smooth", 0) / 100.0
    if smooth > 0:
        if fast_preview:
            smooth = min(smooth, 0.18)
        smoothed = gaussian_blur(edited, sigma=smooth * 4, acceleration=acceleration)
        detail = edited - gaussian_blur(edited, sigma=1, acceleration=acceleration)
        edited = clamp01(smoothed + detail * (1 - smooth * 0.6))

    glow = p.get("glow", 0) / 100.0
    if glow > 0:
        if fast_preview:
            glow = min(glow, 0.10)
        bloom = gaussian_blur(edited, sigma=15, acceleration=acceleration)
        edited = clamp01(edited + bloom * glow * 0.3)

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_skin_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    fast_preview = _fast_interactive_preview(runtime_settings)
    effect_masks = _make_effect_masks(mask, "skin", acceleration=acceleration)
    edited = original.copy()

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    # Regional noise reduction, before Smooth/Blemish -- denoise targets fine sensor grain;
    # Smooth/Blemish work at a broader frequency-separation scale (texture/tone), so running
    # denoise first gives them a cleaner base instead of smoothing over visible grain.
    noise_red = p.get("noise_red", 0) / 100.0
    if noise_red > 0:
        edited = _apply_luma_chroma_denoise(
            edited, noise_red, working_space=working_space, acceleration=acceleration, fast_preview=fast_preview,
            scene_linear_luma=_scene_linear_denoise(color_settings),
            use_learned=_use_learned_denoise(color_settings),
        )

    smooth = p.get("smooth", 0) / 100.0
    if smooth > 0:
        if fast_preview:
            smooth = min(smooth, 0.30)
        edited = _frequency_smooth_skin(edited, smooth, acceleration=acceleration)

    blemish = p.get("blemish", 0) / 100.0
    if blemish > 0:
        if fast_preview:
            blemish = min(blemish, 0.25)
        edited = _frequency_blemish_soften(edited, blemish, acceleration=acceleration)

    clarity = p.get("clarity", 0)
    if clarity != 0:
        clarified = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.3, base_sigma=5.0, edge_strength=2.8)
        edited = _mix_with_effect_mask(edited, clarified, effect_masks["detail"])

    color_even_amount = max(smooth * 0.45, blemish * 0.35)
    if color_even_amount > 0:
        edited = _even_skin_chroma(edited, color_even_amount, acceleration=acceleration)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    glow = p.get("glow", 0) / 100.0
    if glow > 0:
        if fast_preview:
            glow = min(glow, 0.08)
        bloom = gaussian_blur(edited, sigma=12, acceleration=acceleration)
        edited = clamp01(edited + bloom * glow * 0.25)

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_eyes_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "eyes", acceleration=acceleration)
    edited = original.copy()
    edge_guard = _edge_protection_mask(edited, acceleration=acceleration, strength=3.0, blur_sigma=1.0)
    mid_luma = _luma(edited)
    iris_guard = np.clip(1.0 - np.abs(mid_luma - 0.42) * 3.0, 0.0, 1.0).astype(np.float32)

    brightness = p.get("brightness", 0) / 100.0
    if brightness != 0:
        edited = clamp01(edited * (2 ** (brightness * 1.5)))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        blurred = gaussian_blur(edited, sigma=3, acceleration=acceleration)
        clarity_mask = smooth_mask(np.clip(edge_guard * (0.55 + iris_guard * 0.45) * effect_masks["detail"], 0.0, 1.0), sigma=1.0, acceleration=acceleration)
        clarified = clamp01(edited + (edited - blurred) * (clarity / 100.0))
        edited = _mix_with_effect_mask(edited, clarified, clarity_mask)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    white_amt = p.get("whites", 0) / 100.0
    if white_amt > 0:
        edited = _apply_eye_whitening(edited, white_amt, edge_guard=edge_guard * effect_masks["strict"], acceleration=acceleration)

    sharpen = p.get("sharpen", 0) / 100.0
    if sharpen > 0:
        sharpened = _apply_sharpness(edited, 1 + sharpen * 5, working_space)
        sharpen_mask = smooth_mask(np.clip(edge_guard * (0.35 + iris_guard * 0.65) * effect_masks["strict"], 0.0, 1.0), sigma=0.8, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, sharpened, sharpen_mask)

    iris_pop = p.get("iris_pop", 0) / 100.0
    if iris_pop > 0:
        lum = _luma(edited)[:, :, np.newaxis]
        mid_mask = np.clip(1.0 - np.abs(lum - 0.35) * 4, 0, 1)
        iris_mask = mid_mask[:, :, 0] * edge_guard * effect_masks["detail"]
        iris_popped = clamp01(edited + (edited - 0.5) * mid_mask * iris_pop * 0.45)
        edited = _mix_with_effect_mask(edited, iris_popped, smooth_mask(iris_mask, sigma=1.0, acceleration=acceleration))

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_lips_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "lips", acceleration=acceleration)
    edited = original.copy()
    edge_guard = _edge_protection_mask(edited, acceleration=acceleration, strength=2.4, blur_sigma=1.0)

    brightness = p.get("brightness", 0) / 100.0
    if brightness != 0:
        edited = clamp01(edited * (2**brightness))

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)
    edited = adjust_hsv_hue(edited, p.get("hue_shift", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    smooth = p.get("smooth", 0) / 100.0
    if smooth > 0:
        edited_s = gaussian_blur(edited, sigma=smooth * 3, acceleration=acceleration)
        detail = edited - gaussian_blur(edited, sigma=0.5, acceleration=acceleration)
        smoothed = clamp01(edited_s + detail * (1 - smooth * 0.5))
        smooth_mask_local = smooth_mask(np.clip((0.75 + edge_guard * 0.25) * effect_masks["broad"], 0.0, 1.0), sigma=0.8, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, smoothed, smooth_mask_local)

    gloss = p.get("gloss", 0) / 100.0
    if gloss > 0:
        lum = _luma(edited)
        center_mask = np.clip(1.0 - np.abs(lum - 0.58) * 3.5, 0.0, 1.0)
        gloss_mask = smooth_mask(np.clip(center_mask * edge_guard * effect_masks["strict"], 0.0, 1.0), sigma=1.2, acceleration=acceleration)[:, :, np.newaxis]
        highlight = np.clip(lum[:, :, np.newaxis] - 0.5, 0, 1) * 2
        glossed = clamp01(edited + highlight * gloss_mask * gloss * 0.28)
        edited = _mix_with_effect_mask(edited, glossed, gloss_mask[:, :, 0])

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_hair_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "hair", acceleration=acceleration)
    edited = original.copy()
    strand_mask = _hair_strand_mask(edited, acceleration=acceleration)

    brightness = p.get("brightness", 0) / 100.0
    if brightness != 0:
        edited = clamp01(edited * (2**brightness))

    edited = apply_tone_curve_preserve_chroma(edited, 0, 0, 0, p.get("highlights", 0), 0, working_space=working_space)
    edited = _apply_hair_depth(edited, p.get("highlights", 0) * 0.35, acceleration=acceleration)
    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    shine = p.get("shine", 0) / 100.0
    if shine > 0:
        shined = _apply_hair_shine(edited, shine, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, shined, np.clip(effect_masks["detail"] * (0.35 + strand_mask * 0.65), 0.0, 1.0))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        clarified = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.1, base_sigma=4.8, edge_strength=1.7)
        edited = _mix_with_effect_mask(edited, clarified, np.clip(effect_masks["detail"] * (0.35 + strand_mask * 0.65), 0.0, 1.0))

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


LAYER_PROCESSORS = {
    "subjects": process_subjects_layer,
    "person": process_person_layer,
    "background": process_background_layer,
    "face": process_face_layer,
    "skin": process_skin_layer,
    "eyes": process_eyes_layer,
    "lips": process_lips_layer,
    "hair": process_hair_layer,
}


def _apply_blend_mode(base: np.ndarray, layer_img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "overlay":
        mixed = np.where(
            base <= 0.5,
            2.0 * base * layer_img,
            1.0 - 2.0 * (1.0 - base) * (1.0 - layer_img),
        )
        return clamp01(mixed)

    if mode == "soft_light":
        mixed = (1.0 - 2.0 * layer_img) * (base * base) + 2.0 * layer_img * base
        return clamp01(mixed)

    return layer_img


def _layer_composite_mask(
    mask: np.ndarray,
    layer: str,
    mode: str,
    opacity: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    effect_masks = _make_effect_masks(mask, layer, acceleration=acceleration)
    mode = str(mode or "normal").lower()

    if mode == "overlay":
        base_mask = effect_masks["detail"] if layer in {"face", "skin", "hair"} else effect_masks["strict"]
    elif mode == "soft_light":
        if layer in {"eyes", "lips"}:
            base_mask = effect_masks["detail"]
        else:
            base_mask = np.clip(effect_masks["blend"] * 0.7 + effect_masks["broad"] * 0.3, 0.0, 1.0)
    else:
        base_mask = effect_masks["blend"]

    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0.0:
        return np.zeros_like(mask, dtype=np.float32)
    if opacity >= 0.999:
        return np.clip(base_mask, 0.0, 1.0).astype(np.float32)

    opacity_power = 1.0 + (1.0 - opacity) * 1.6
    opacity_floor = max(0.0, 0.25 - opacity * 0.20)
    scaled = _confidence_curve(base_mask, power=opacity_power, floor=opacity_floor)
    return np.clip(scaled * opacity, 0.0, 1.0).astype(np.float32)


def process_all_layers(
    original: np.ndarray,
    all_params: dict,
    masks: dict,
    geometry: dict | list[dict] | None = None,
    layer_order=None,
    layer_options: dict | None = None,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
    stage_cache: "StagePipelineCache | None" = None,
    inputs_token=None,
    crop_origin: tuple[int, int] = (0, 0),
    full_shape: tuple[int, int] | None = None,
    debug_sink: dict | None = None,
):
    """Apply global then selective layers with per-layer blend configuration.

    When ``stage_cache`` is provided, each pipeline stage (warp -> global -> face refine ->
    each selective layer) is memoized by the cumulative digest of every input that affects it,
    so a slider change only recomputes from the first affected stage downstream -- editing a
    skin slider reuses the cached global/face/background/... stages instead of redoing them.
    ``inputs_token`` identifies the static, non-slider inputs (source image, mask revision,
    analysis signature, preview resolution); the caller bumps it whenever those change so stale
    stages are never served. With ``stage_cache=None`` (export, batch, tests) the path is
    byte-identical to the un-cached pipeline -- no keying overhead, no behavior change."""
    working_space = _working_space(color_settings)
    output_transform = _output_transform(color_settings)
    cached = stage_cache is not None

    def _staged(key, compute):
        """Return the cached stage output (a fresh copy) on hit, else compute, store a copy,
        and return the live result. Copy-on-both-sides isolates the cache from in-place writes."""
        if cached and key is not None:
            hit = stage_cache.get(key)
            if hit is not None:
                return _copy_stage_value(hit)
        value = compute()
        if cached and key is not None:
            stage_cache.put(key, _copy_stage_value(value))
        return value

    face_params = all_params.get("face", {})

    # Stage: warp. display->working transform + expression warp. Depends on source identity
    # (inputs_token), working space, the face params the warp consumes, and geometry.
    warp_key = _stage_param_digest("warp", inputs_token, working_space, face_params, geometry) if cached else None

    def _do_warp():
        working_original = display_to_working(original, working_space=working_space)
        working_masks = None if masks is None else {
            key: np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0) for key, value in masks.items()
        }
        return _apply_expression_warp(working_original, working_masks, face_params, geometry=geometry)

    working_original, working_masks = _staged(warp_key, _do_warp)

    # Stage: global. The single most expensive stage (WB, tone, clarity, denoise, sharpen).
    global_key = _stage_param_digest(
        "global", warp_key, all_params.get("global", {}), color_settings, runtime_settings
    ) if cached else None
    if debug_sink is not None:
        # A debug-sink request needs to observe this exact call's internals, so it bypasses
        # the stage cache rather than risk silently returning a stale/missing mask on a hit.
        result = process_global(
            working_original,
            all_params.get("global", {}),
            color_settings=color_settings,
            runtime_settings=runtime_settings,
            crop_origin=crop_origin,
            full_shape=full_shape,
            debug_sink=debug_sink,
        )
    else:
        result = _staged(global_key, lambda: process_global(
            working_original,
            all_params.get("global", {}),
            color_settings=color_settings,
            runtime_settings=runtime_settings,
            crop_origin=crop_origin,
            full_shape=full_shape,
        ))

    # Stage: face refinement (optional ML pass). Keyed on global + the face params it consumes,
    # so a non-face slider drag reuses the expensive refined result instead of re-running it.
    face_key = _stage_param_digest("face_refine", global_key, face_params) if cached else None
    result = _staged(face_key, lambda: _apply_face_refinement(
        result,
        working_masks,
        face_params,
        color_settings=color_settings,
        runtime_settings=runtime_settings,
    ))

    order = tuple(layer_order) if layer_order else MASK_ORDER
    options = layer_options or {}

    prev_key = face_key
    for layer in order:
        if layer not in LAYER_PROCESSORS:
            continue
        if not working_masks or layer not in working_masks:
            continue

        cfg = options.get(layer, {})
        if not cfg.get("enabled", True):
            continue

        opacity = float(np.clip(float(cfg.get("opacity", 100.0)) / 100.0, 0.0, 1.0))
        if opacity <= 0.0:
            continue

        # Cumulative key: this layer's composited output depends on the prior stage's output
        # (prev_key) plus this layer's own params and blend config. A layer earlier in the
        # order changing busts everything after it, exactly like the real data dependency.
        layer_key = _stage_param_digest(
            "layer", prev_key, layer, all_params.get(layer, {}), cfg
        ) if cached else None

        def _compose_layer(layer=layer, cfg=cfg, opacity=opacity, base=result):
            mask = working_masks[layer]
            if layer == "skin":
                protection = _build_skin_protection_mask(
                    base,
                    working_masks,
                    geometry=geometry,
                    acceleration=_acceleration_mode(runtime_settings),
                )
                mask = np.clip(mask.astype(np.float32) * (1.0 - protection), 0.0, 1.0)
            if mask is None or mask.max() <= 0.01:
                return base
            mode = str(cfg.get("blend_mode", "normal"))
            full_mask = _layer_composite_mask(
                mask,
                layer,
                mode,
                opacity,
                acceleration=_acceleration_mode(runtime_settings),
            )
            # Selective layers should build on top of the globally adjusted image,
            # not the original source, so global edits propagate into face/skin/etc.
            layer_img = LAYER_PROCESSORS[layer](
                base,
                base,
                mask.astype(np.float32),
                all_params.get(layer, {}),
                color_settings=color_settings,
                runtime_settings=runtime_settings,
            )
            mixed = _apply_blend_mode(base, layer_img, mode)
            return blend_with_mask(base, mixed, full_mask)

        result = _staged(layer_key, _compose_layer)
        prev_key = layer_key

    display_result = working_to_display(result, output_transform=output_transform, working_space=working_space)
    return Image.fromarray(to_uint8(display_result))
