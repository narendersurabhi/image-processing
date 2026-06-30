# RAW Image Processing — Steps, Best Practices & Implementation

How to turn a camera RAW file (the unprocessed sensor mosaic) into a finished image
**correctly**, what the modern research says about each stage, and how to do it **in this
app specifically** — which decode path runs today, where the gaps are, and what to build
next.

Grounded in the actual pipeline: `file:line` references point at real behavior. The
pipeline drifts, so verify a detail against current code before relying on it. Where this
doc describes a stage the app does **not** implement yet, it says so explicitly (§3, §7).

This sits alongside [COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) (which owns
profile/transfer policy and already flags RAW decode as follow-up work) and
[NOISE_SHARPENING_STRATEGY.md](NOISE_SHARPENING_STRATEGY.md) (region-aware detail, which is
the *downstream* consumer of a decoded RAW). Read those for color and detail policy; this
doc owns the **RAW-to-working-buffer** front end.

---

## 1. Why RAW is different from a JPEG/PNG

A RAW file is not an image. It is a near-direct dump of the sensor's analog-to-digital
readout plus the metadata needed to interpret it. Concretely it carries:

- a **single-channel mosaic** (CFA — color filter array), not RGB: most sensors are Bayer
  (RGGB/BGGR/…), Fujifilm is X-Trans (6×6), some are X3/Foveon (stacked). Each photosite
  measured *one* color through a colored filter.
- **black level and white (saturation) level** — the sensor's zero and clipping points,
  per channel, often non-zero black from thermal/electronic bias.
- **as-shot white balance multipliers** — the camera's chosen per-channel gains.
- a **camera color matrix** (and/or an embedded DNG `ColorMatrix`/DCP profile) mapping the
  sensor's idiosyncratic spectral response to a standard space (CIE XYZ).
- **linearization data** — some cameras store a non-linear encoding/LUT that must be undone
  to recover scene-linear values.
- bit depth of **12–16 bits** (vs. 8 for JPEG), so far more highlight/shadow latitude.

The single most important consequence: **RAW values are scene-linear** (proportional to
photons) *before* any tone curve. A JPEG has already had demosaic, white balance, a camera
tone curve, sharpening, and 8-bit quantization baked in irreversibly. Editing RAW means you
own all of those decisions — and you must do the physically-meaningful ones (exposure, white
balance, highlight recovery) in linear light, before the tone curve, or you fight artifacts
the whole way.

---

## 2. The canonical RAW pipeline (correct order)

The order is not arbitrary — each stage assumes the previous one's output domain. The
classic ISP (image signal processor) / RAW-developer order:

```
 1. Read + unpack       sensor mosaic, metadata, black/white levels, WB mult, color matrix
 2. Linearize           undo non-linear encoding -> true scene-linear
 3. Black/white level   subtract black, normalize to white -> [0,1] scene-linear
 4. Defect/dark correct  bad-pixel, dark-frame, optional flat-field/vignette
 5. White balance        per-channel multipliers, in linear (best done pre-demosaic)
 6. Highlight recovery   reconstruct or clip blown channels
 7. Demosaic (debayer)   CFA mosaic -> full RGB
 8. Denoise              most effective in raw/linear domain; modern nets do 7+8 jointly
 9. Color transform      camera-native RGB -> XYZ (color matrix) -> working RGB
10. Tone mapping         scene-linear -> display-referred (base/filmic/ACES curve), exposure
11. Creative/color       saturation, HSL, look LUTs, local contrast
12. Output encode        gamut map + transfer function + ICC tag
13. Output sharpen+resize resolution-aware final pass
```

Stages 1–9 are **"develop the RAW"** — turning sensor data into a clean working image in a
known color space. Stages 10–13 are the *same* editing pipeline this app already runs on any
image (and which the color and noise/sharpening strategy docs cover). The interesting,
RAW-specific engineering is stages 1–9.

Two ordering rules that are violated most often:

- **White balance belongs in linear light, ideally before demosaic.** Demosaic algorithms
  reason about color ratios at edges; balancing first makes the channels comparable and
  reduces color fringing. This app applies WB *after* decode in linear
  ([white_balance.py:176](../portrait_enhancer/core/white_balance.py#L176)) — fine for the
  corrective grade, but it is *also* relying on LibRaw to have applied the as-shot
  multipliers during decode.
- **Denoise before sharpen, and prefer the linear/raw domain.** Noise is closest to
  signal-independent (or analytically modelable) in the raw domain; once a tone curve and
  demosaic have run, noise is spatially correlated and harder to model. See
  [NOISE_SHARPENING_STRATEGY.md](NOISE_SHARPENING_STRATEGY.md) §1.

---

## 3. Current state in this repo

> **Status (Phases 1–3 shipped):** the three decode paths below have been unified into one
> shared decoder, [core/raw_decode.py](../portrait_enhancer/core/raw_decode.py) (`decode_raw`).
> Qt and batch now honor the same RAW WB/colorspace/LUT settings the data model already carried,
> capture as-shot camera metadata, and expose `raw_auto_brightness` (default ON). Phase 2 added
> a Qt **RAW Decode** inspector section surfacing WB, Color Space, **highlight mode**, and
> **demosaic algorithm**, each re-decoding the source on change. Phase 3 fixed the 8-bit
> tone-curve quantization and added a **scene-linear render path** (`working_space="linear"`,
> a Scene-linear toggle in the RAW Decode panel). Phase 4 added a **noise-model-aware luma
> denoise** (variance-stabilizing transform, [core/vst_denoise.py](../portrait_enhancer/core/vst_denoise.py))
> for the scene-linear path. The table and gaps below describe the *pre-Phase-1* state and are
> kept for context; gaps §3.1, §3.3 (Phase 1), the highlight/demosaic half of §3.4 (Phase 2),
> §3.2/§3.5 (Phase 3), and the classical/linear-domain half of §3.6 (Phase 4) are closed — see
> §6. The only remaining item is the *learned* raw denoiser (offline model training), which is
> out of scope for in-repo work.

The app decodes RAW through **rawpy** (LibRaw bindings, `rawpy>=0.18.1` in
[pyproject.toml](../pyproject.toml#L19)). LibRaw runs stages 1–9 internally and hands back a
finished RGB image; the app then runs its normal editing pipeline. Before Phase 1 there were
**three** decode paths and they did not agree:

| Path | File | What it does today |
|---|---|---|
| Tk app (full) | [ui/app.py:1491](../portrait_enhancer/ui/app.py#L1491) | `postprocess(output_bps=16, no_auto_bright=False)`; WB = camera **or** auto; output color = sRGB/Adobe/ProPhoto/XYZ/raw; optional RAW input `.cube` LUT applied right after decode; records `input_profile_applied` audit string |
| Qt app (forward UI) | [ui_qt/main_window.py:2222](../portrait_enhancer/ui_qt/main_window.py#L2222) | `postprocess(use_camera_wb=True, output_bps=16)` — **hardcoded**, no WB/colorspace/LUT/highlight options, returns empty metadata `{}` |
| Batch | [batch_runner.py:282](../portrait_enhancer/batch_runner.py#L282) | `postprocess(use_camera_wb=True, output_bps=16)` — **hardcoded**, same as Qt |

Supported RAW extensions are `.cr2 .nef .arw .dng .raw` across all three
([config.py], [ui/app.py:63](../portrait_enhancer/ui/app.py#L63),
[batch_runner.py:41](../portrait_enhancer/batch_runner.py#L41)). All paths normalize the
16-bit result to `float32` in `[0,1]` and feed it into the working pipeline.

What this means in practice — the **honest gaps**:

1. ~~**Qt and batch ignore the RAW color settings the Tk app exposes.**~~ **(Closed in Phase
   1.)** `raw_white_balance`, `raw_colorspace`, `raw_lut_*` existed in the color-settings
   schema but only the Tk decoder read them; Qt had the *least* capable decode despite being
   the forward UI. Now all paths route through `decode_raw`
   ([core/raw_decode.py](../portrait_enhancer/core/raw_decode.py)), which reads these settings;
   Qt threads its `_color_settings` into the load worker and batch passes `task["color_settings"]`.

2. **Decode is display-referred sRGB by default**, not scene-linear. LibRaw applies a tone
   curve and gamma during `postprocess`, so the float buffer the app receives is already
   *display-referred* — the very headroom RAW exists to preserve has been compressed before
   the app's exposure/tone math ever runs. The working space then defaults to `srgb`
   ([utils.py:162](../portrait_enhancer/core/utils.py#L162)), compounding this.

3. **`no_auto_bright=False`** lets LibRaw apply a non-deterministic auto-brightness stretch
   based on the histogram — two similar frames can get different exposure baselines, which
   undermines batch consistency. **(Addressed in Phase 1, default unchanged.)** This is now
   the `raw_auto_brightness` setting in `decode_raw`; it defaults **ON** (preserving the
   pleasant auto-brightened import) and can be turned **off** for a deterministic,
   batch-consistent baseline that the Exposure slider then owns.

4. **No control over highlight mode, demosaic algorithm, or output bit-precision of the
   tone curve.** LibRaw exposes `highlight_mode` (clip/unclip/blend/rebuild),
   `demosaic_algorithm` (LINEAR/VNG/PPG/AHD/DCB/DHT/AAHD/LMMSE), `output_bps`, custom WB
   multipliers, and DCP/camera-profile selection — none are surfaced.

5. **The tone-curve helper quantizes to 8-bit mid-pipeline.** `apply_tone_curve`
   ([utils.py:206](../portrait_enhancer/core/utils.py#L206)) indexes a 256-entry LUT via
   `(img*255).astype(uint8)` — for a 12–16-bit RAW source this throws away most of the tonal
   precision RAW provides, exactly where smooth gradients (skies, skin falloff) need it.

6. **Denoise is RGB-domain, post-demosaic.** The bilateral path
   ([utils.py:108](../portrait_enhancer/core/utils.py#L108)) and the ONNX denoisers
   (DnCNN-class `MLDenoiser`, NAFNet-SIDD `DeepDenoiser`,
   [denoise.py:26](../portrait_enhancer/core/denoise.py#L26),
   [denoise.py:159](../portrait_enhancer/core/denoise.py#L159)) all operate on the developed
   RGB image, not raw. That's a pragmatic choice (the models are trained on sRGB noise), but
   it forgoes the strongest place to denoise (§6.3).

None of this makes the current output *wrong* — LibRaw's defaults are good, and for
display-referred portrait editing they're acceptable. But the app is currently a thin
wrapper over LibRaw's "give me a nice JPEG-like RGB" mode, not a RAW developer that owns the
scene-linear pipeline.

---

## 4. Stage-by-stage: best practices

### 4.1 Read + unpack (stage 1)

- Use the as-shot metadata. LibRaw exposes `raw.camera_whitebalance`, `raw.color_desc`,
  `raw.rgb_xyz_matrix`, `raw.black_level_per_channel`, `raw.white_level`,
  `raw.raw_pattern` via the `rawpy` object **before** `postprocess`. Capturing these lets you
  reproduce or override any decode decision and feed a real `ColorContext`
  ([COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) §5.4) instead of `{}`.
- Honor EXIF orientation. (Non-RAW already does via `ImageOps.exif_transpose`,
  [ui_qt/main_window.py:2237](../portrait_enhancer/ui_qt/main_window.py#L2237); LibRaw's
  `user_flip` defaults to using the embedded orientation.)

### 4.2 Linearize + black/white level (stages 2–3)

LibRaw does this for you. The best practice you *can* control: decode to a **linear output**
(`output_color` + `gamma=(1,1)` / `output_bps=16`) when you intend to do exposure and tone
in the app, so you keep scene-linear headroom. Then linearize/clip only at boundaries
([COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) §5.2 — "clamp only at domain
boundaries").

### 4.3 White balance (stage 5)

- **As-shot first.** Camera WB (`use_camera_wb=True`) is the right default and is what all
  three paths use today. It comes from the camera's own measurement and is usually close.
- **Auto WB is a fallback, not an upgrade.** LibRaw's `use_auto_wb` is a gray-world average
  — easily fooled by a dominant color (the same failure mode this app's
  `gray_world_gains` has, [white_balance.py:204](../portrait_enhancer/core/white_balance.py#L204)).
  Prefer the app's content-aware AWB (`WhiteBalanceEstimator`, FC4/Deep-WB ONNX,
  [white_balance.py:220](../portrait_enhancer/core/white_balance.py#L220)) applied as a
  *corrective grade in linear* on top of as-shot decode.
- **Custom multipliers** (`user_wb=[r,g1,b,g2]`) let you push a specific neutral from a
  picked patch directly into the decode — the cleanest place to balance, because it happens
  pre-demosaic inside LibRaw.

### 4.4 Highlight recovery (stage 6)

When one channel clips before the others (very common — blue sky, skin in sun), a naive
clip turns the highlight magenta/cyan. LibRaw `highlight_mode`:

| Mode | Behavior | Use when |
|---|---|---|
| 0 clip | clip to white | fastest, fine if nothing important is blown |
| 1 unclip | leave as-is | diagnostic |
| 2 blend | blend clipped + unclipped | mild recovery, safe default |
| 3–9 rebuild | reconstruct from neighbors | strong recovery (skies, specular falloff) |

Best practice: default to **blend (2)** for portraits; offer **rebuild (5)** as a "recover
highlights" option. This is more faithful than recovering highlights *after* a display-referred
decode, where the information is already gone.

### 4.5 Demosaic (stage 7)

The CFA-to-RGB reconstruction quality ceiling. Classical algorithms, fast→best:

- **LINEAR/bilinear** — fast, zippering and false color at edges. Don't ship.
- **VNG / PPG** — decent, cheap.
- **AHD** (Adaptive Homogeneity-Directed, Hirakawa & Parks 2005) — long-time good default.
- **DCB**, **DHT**, **AAHD** — stronger edge/detail handling, more compute.
- **LMMSE**, **AMaZE** (RawTherapee) — among the best classical for detail vs. artifacts.

Best practice: **AHD or DCB as default**, AAHD/DHT as a "max detail" option. For X-Trans,
LibRaw uses a dedicated path (Markesteijn) — don't assume Bayer algorithms apply.

### 4.6 Color transform (stage 9)

Camera-native RGB is *not* any standard RGB. Convert through the **camera color matrix**
(`rgb_xyz_matrix`, or a DNG/DCP profile's forward matrix) to CIE XYZ, then to your working
profile (`linear-srgb` per [COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) §5.2).
LibRaw does a baseline matrix conversion when you set `output_color`; a true color-managed
pipeline uses the camera's DCP profile (with its HSV look table and dual-illuminant
interpolation) for accurate skin tones. The Tk app's `output_color` choice
([ui/app.py:1503](../portrait_enhancer/ui/app.py#L1503)) is the *output* primaries only —
it does not select a camera input profile.

### 4.7 Tone, color, output (stages 10–13)

These are the app's existing pipeline and are covered elsewhere:
[COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) (working space, output encode,
ICC) and [NOISE_SHARPENING_STRATEGY.md](NOISE_SHARPENING_STRATEGY.md) (denoise/sharpen order,
output sharpening). The one RAW-specific note: a RAW developer should apply a **base tone
curve** (filmic/ACES/camera) to map scene-linear into display-referred *before* the
perceptual grade — otherwise a linearly-decoded RAW looks flat and dark and the user fights
it with every slider.

---

## 5. Latest research by stage

The field has moved from hand-tuned ISP stages toward **learned, jointly-optimized** RAW
processing. The high-value results:

### 5.1 Joint demosaic + denoise (replaces stages 7–8)

Doing demosaic and denoise separately is provably suboptimal — each makes assumptions the
other breaks. The seminal result:

- **Gharbi et al., "Deep Joint Demosaicking and Denoising"** (SIGGRAPH Asia 2016) — a CNN
  that jointly demosaics and denoises from the raw mosaic, with a hard-case mining strategy
  for moiré/edges. Set the template the field still follows.
- **Kokkinos & Lefkimmiatis** (2018/2019) — iterative residual / majorization-minimization
  unrolled networks for joint demosaic-denoise.
- Transformer/UNet restorers — **Restormer** (Zamir et al., CVPR 2022), **NAFNet** (Chen et
  al., ECCV 2022), **SwinIR** (Liang et al., 2021) — now standard backbones; NAFNet-SIDD is
  literally what this app's `DeepDenoiser` runs ([denoise.py:159](../portrait_enhancer/core/denoise.py#L159)),
  just on RGB instead of raw.

### 5.2 Learned RAW noise models & low-light (improves stage 8)

- **Chen et al., "Learning to See in the Dark"** (SID, CVPR 2018) — end-to-end raw→RGB for
  extreme low light; the dataset everyone benchmarks on.
- **Brooks et al., "Unprocessing Images for Learned Raw Denoising"** (CVPR 2019) — invert the
  ISP to synthesize realistic raw training data from any sRGB image. Hugely practical:
  removes the need for paired raw captures.
- **Wei et al., "A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising"**
  (ELD, CVPR 2020) — models read/shot/row/quantization noise per-sensor; calibrate once,
  generalize. The principled answer to "what sigma do I denoise at" — note this app instead
  *measures* noise from pixels (Immerkjær estimator,
  [NOISE_SHARPENING_STRATEGY.md](NOISE_SHARPENING_STRATEGY.md) §5.1), which is the right call
  when you don't have per-sensor calibration.

### 5.3 End-to-end learned ISP (replaces stages 5–11)

- **Ignatov et al., "Replacing Mobile Camera ISP with a Single Deep Learning Model"**
  (PyNET, 2020) and the **AIM / Mobile AI (MAI) RAW-to-RGB challenges** (2020–2022) — one
  network maps raw Bayer straight to a finished RGB, learning WB+demosaic+denoise+tone+color
  jointly. State of the art for fixed-pipeline / mobile output; *less* suited to an editor
  where the user wants to intervene at each stage.
- **Karaimer & Brown, "A Software Platform for Manipulating the Camera Imaging Pipeline"**
  (ECCV 2016) — the reference decomposition of a real ISP into editable stages; the right
  mental model for an *editor* (keep stages separable) vs. a black-box learned ISP.

### 5.4 Learned white balance (improves stages 5 / 4.3)

- **Hu et al., "FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling"**
  (CVPR 2017) — the illuminant-estimation model family this app's `WhiteBalanceEstimator`
  already targets ([white_balance.py:220](../portrait_enhancer/core/white_balance.py#L220)).
- **Afifi & Brown, "Deep White-Balance Editing"** (CVPR 2020) — post-capture WB editing,
  also supported by the app's image-to-image AWB branch
  ([white_balance.py:306](../portrait_enhancer/core/white_balance.py#L306)).
- **Barron, "Fast Fourier Color Constancy"** (CVPR 2017) — efficient, robust classical-learned
  hybrid.

### 5.5 Local tone mapping (improves stage 10)

- **Mertens et al., "Exposure Fusion"** (2007/2009) — blends a virtual bracket; cheap, robust.
- **Paris et al., "Local Laplacian Filters"** (SIGGRAPH 2011) — edge-aware local contrast
  without halos; the principled basis for "Clarity"/"Dehaze"-type controls.
- **Gharbi et al., "Deep Bilateral Learning for Real-Time Image Enhancement"** (HDRnet,
  SIGGRAPH 2017) — learns a tone/color operator as a bilateral grid; real-time, edit-friendly.

**Takeaway for this app:** the highest-leverage research move is **raw-domain joint
demosaic+denoise** (§5.1–5.2) — the app already ships NAFNet, so the gap is *where* in the
pipeline it runs, not whether the model exists. The lowest-risk move is **unprocessing**
(§5.2) to train/fine-tune a raw denoiser without collecting paired raw data.

---

## 6. Recommendations for this app

Phased, mapped to the existing code. Each phase is independently shippable.

### Phase 1 — Make RAW decode consistent and deterministic ✅ Implemented

The cheapest, highest-value fixes; no new models. Shipped in
[core/raw_decode.py](../portrait_enhancer/core/raw_decode.py) with coverage in
[tests/test_raw_decode_smoke.py](../tests/test_raw_decode_smoke.py).

- ✅ **Unified the decode path.** All four `postprocess` call sites (Qt load, Qt
  thumbnail/culling, batch, plus the Tk app's logic that seeded the design) now route through
  one `decode_raw(path, color_settings) -> (array, metadata)`. Qt threads its `_color_settings`
  into `ImageLoadTask`; batch passes `task["color_settings"]`. Qt and batch inherit the
  WB/colorspace/LUT controls the data model already persisted. Closes gap §3.1 and the "single
  image and batch must call one API" rule
  ([COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) §2). *(The Tk app keeps its own
  inline decode for now — the user works in Qt; folding Tk into `decode_raw` is a trivial
  follow-up.)*
- ✅ **Exposure is now a setting.** `raw_auto_brightness` (→ `no_auto_bright`) is exposed in
  the decode config. Per product decision it defaults **ON** (preserves the current pleasant
  import); turning it off gives the deterministic, batch-consistent baseline. Addresses §3.3.
- ✅ **Captures RAW metadata.** `decode_raw` reads `camera_whitebalance`, `rgb_xyz_matrix`,
  `black_level_per_channel`, `white_level`, `raw_pattern`, `color_desc` (JSON-serializable)
  and returns them plus an `input_profile_applied` audit string and `source_is_raw=True`. Qt no
  longer returns `{}` for RAW. Feeds a future `ColorContext` and makes decode auditable.

### Phase 2 — Expose the decode controls that matter ✅ Implemented

A **RAW Decode** section in the advanced inspector (shown only for RAW sources), with four
combos wired through `decode_raw`; changing any of them re-decodes the source while preserving
all edits (via the project-state round-trip — `_reload_source_for_decode_settings`). Covered in
[tests/test_raw_decode_smoke.py](../tests/test_raw_decode_smoke.py).

- ✅ **Highlight mode** (§4.4): Clip / Blend / Rebuild (`raw_highlight_mode` → LibRaw
  `highlight_mode` 0/2/5). Defaults to **Clip** to preserve the pre-Phase-2 look (consistent
  with the Phase 1 auto-brightness decision); Blend/Rebuild are the opt-in quality wins for
  blown highlights.
- ✅ **Demosaic algorithm** (§4.5): Auto / AHD / DCB / VNG / AAHD / DHT (`raw_demosaic`).
  **Auto** (default) passes nothing → LibRaw's default (AHD), guaranteeing identical output;
  DCB/AAHD/DHT are the "max detail" options, with a graceful fallback (recording
  `raw_demosaic_fallback`) when the chosen algorithm isn't compiled into the installed LibRaw.
- ✅ **Qt parity:** the section also surfaces RAW **White Balance** and **Color Space** (wired
  in Phase 1 but previously UI-less), completing the advanced RAW decode panel
  ([COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) §8.2). *(A creative RAW-LUT file
  picker is the remaining parity item with the old Tk panel; the decode plumbing for it already
  exists in `decode_raw`.)*

### Phase 3 — Scene-linear RAW pipeline

The structural fix behind gaps §3.2 and §3.5.

- ✅ **Killed the 8-bit tone-curve quantization** (§3.5). Both `apply_tone_curve`
  ([utils.py:193](../portrait_enhancer/core/utils.py#L193)) and the interactive
  `apply_curve` ([tone_curve.py:93](../portrait_enhancer/core/tone_curve.py#L93)) now evaluate
  the curve on the actual float values via `np.interp` (the band curve directly on its control
  points; the interactive curve into a 4096-sample spline) instead of indexing a 256-entry LUT
  through `(v*255).astype(uint8)`. A 4000-level gradient now yields 4000 distinct output levels
  (was capped at ~256), so the smooth 12–16-bit RAW gradients in skies and skin falloff survive
  the tone stage. Benefits every image, not just RAW. Covered by
  [tests/test_tone_curve_smoke.py](../tests/test_tone_curve_smoke.py).
- ✅ **Scene-linear render pipeline** (§3.2). Implemented as the `working_space="linear"` render
  path (a re-render, not a re-decode — the source enters as display-referred sRGB and is
  linearized at the pipeline boundary, [processing.py:2284](../portrait_enhancer/core/processing.py#L2284)).
  In linear mode the physical operations run in linear light — exposure (`img * 2**ev`), white
  balance, blur, sharpen, and bloom — while the **perceptual** controls stay display-referred:
  the two tone curves are now domain-aware (`apply_tone_curve_preserve_chroma`,
  [utils.py:213](../portrait_enhancer/core/utils.py#L213); `apply_curve`,
  [tone_curve.py:93](../portrait_enhancer/core/tone_curve.py#L93)) and convert to sRGB, apply,
  and convert back, and HSL/saturation/color-balance already converted internally. The sRGB
  output transform is the base scene-linear→display curve. Exposed as a **Scene-linear
  (linear-light) processing** toggle in the Qt RAW Decode panel (default OFF — the default
  `working_space="srgb"` path is byte-identical to before, guarded by
  [tests/test_scene_linear_smoke.py](../tests/test_scene_linear_smoke.py)). A filmic/ACES base
  curve with super-white roll-off is a possible follow-up, but only matters once the decode
  preserves values above the sensor white level (Phase 2 highlight `rebuild` is the lever).

### Phase 4 — Raw-domain / joint denoise (research-grade)

- ✅ **Noise-model-aware luma denoise in linear light** (§5.1). New module
  [core/vst_denoise.py](../portrait_enhancer/core/vst_denoise.py) denoises luminance in the
  scene-linear domain through a **generalized Anscombe variance-stabilizing transform** (Mäkitalo
  & Foi 2011): it estimates a shot-noise-dominant Poisson-Gaussian model `var(x) = a*x + b` from
  the image, maps the signal-dependent noise to ~constant variance, denoises at one uniform
  strength, and inverts — so the shadow noise a fixed-strength filter mishandles is treated
  correctly across the tonal range. Wired into `_apply_luma_chroma_denoise` (luma only; chroma
  stays on the perceptual LAB path) and exposed as a **Noise-model-aware luma denoise (VST)**
  checkbox under the Scene-linear toggle (opt-in, enabled only in scene-linear mode; the sRGB
  path is byte-identical). Tests in
  [tests/test_vst_denoise_smoke.py](../tests/test_vst_denoise_smoke.py) verify the transform
  round-trips, that it flattens a 16× signal-dependent variance ratio to ~1, and that it reduces
  noise without touching the sRGB path.
- ✅ **Scene-linear + VST is now the RAW default; learned denoisers are the sRGB-mode engine.**
  Freshly-loaded RAW files default to `working_space="linear"` + `scene_linear_denoise=True`
  (non-RAW stays sRGB; saved projects keep their stored setting). The learned DnCNN luma
  denoiser is gated behind a `use_learned_denoise` setting (default True, preserving prior
  behavior) and a **Use learned denoiser (DnCNN, sRGB)** checkbox that is disabled while
  Scene-linear is on — in scene-linear+VST mode the VST handles luminance and the DnCNN is
  bypassed (the `luma_done` guard), so they're mutually-exclusive luma engines surfaced as such.
  Note both learned denoisers already operate on sRGB pixels internally — the live DnCNN on
  `working_to_display(...)` and the export NAFNet on the sRGB source array — so the export
  **Deep Denoise** checkbox is compatible with every mode and needs no gating.
- ⏳ **Learned raw denoise via unprocessing** (§5.2) — *out of scope for in-repo work.*
  Fine-tuning the NAFNet `DeepDenoiser` ([denoise.py:159](../portrait_enhancer/core/denoise.py#L159))
  on raw-synthesized noise is an offline training project (datasets + GPU hours) that can't be
  implemented or verified inside this repo. The classical VST path above is the shipped,
  testable realization of the same "denoise where the noise model is simplest" principle, and
  strength is still auto-seeded from the measured-noise estimator
  ([NOISE_SHARPENING_STRATEGY.md](NOISE_SHARPENING_STRATEGY.md) §5.1).

---

## 7. Engineering rules

- **One decode path.** Qt, Tk, and batch must call the same `decode_raw`. No hardcoded
  `postprocess(use_camera_wb=True)` in UI/batch code.
- **Decode metadata is never empty.** Every RAW decode returns the camera WB, color matrix,
  black/white levels, and the decode settings used — enough to reproduce or audit the result.
- **Physical operations in linear.** White balance, exposure, and highlight work happen in
  scene-/display-linear, never on the display-referred buffer
  ([COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md) §12).
- **Don't quantize a RAW mid-pipeline.** No `uint8` round-trips between decode and output for
  RAW sources; keep `float32` until the export/preview boundary.
- **Camera WB is the default; auto/learned WB is a corrective grade on top**, not a
  replacement for as-shot decode.
- **Decode settings are part of render identity.** `raw_white_balance`, `raw_colorspace`,
  highlight mode, demosaic algorithm, and RAW LUT must be in preview/analysis/batch cache
  keys — changing any of them changes pixels.
- **Apply decode settings *before* the decode, not just after.** A decode setting only takes
  effect when `decode_raw` runs, so on a project/collection load the saved decode keys must be
  merged into the color settings *before* the decode is snapshotted (`_preapply_raw_decode_settings`),
  or the image silently decodes with defaults while the UI shows the saved choice. Render-time
  keys (working space, scene-linear, learned-denoise) can still be applied post-decode.

---

## 8. Testing strategy

- **Decode determinism:** same RAW + same settings → byte-identical decoded array (requires
  `no_auto_bright=True`). Catches §3.3 regressions.
- **Three-path parity:** Tk, Qt, and batch produce the *same* decoded array for the same RAW
  and color settings. This is the guard that the unified `decode_raw` actually unified.
- **Metadata completeness:** decode returns non-empty camera WB + color matrix for a known
  fixture; assert specific values for a checked-in DNG.
- **Highlight mode behavior:** a synthetic clipped-channel fixture decodes without magenta
  highlights under blend/rebuild and *with* the artifact under clip — proves the control
  works.
- **Tone-curve precision:** a 16-bit gradient through `apply_tone_curve` shows no 8-bit
  banding (Phase 3). Catches §3.5.
- **Fixtures:** check in one small DNG per CFA family if licensing allows (Bayer + X-Trans);
  otherwise synthesize a mosaic from a known RGB image (the inverse of §5.2 unprocessing) so
  tests run without proprietary files. Keep these out of any path that touches real user data
  (the project's destructive-testing rule).

---

## 9. References

Research:

- Gharbi, Chaurasia, Paris, Durand. *Deep Joint Demosaicking and Denoising.* SIGGRAPH Asia 2016.
- Chen, Chen, Xu, Koltun. *Learning to See in the Dark.* CVPR 2018.
- Brooks, Mildenhall, Xue, Chen, Sharlet, Barron. *Unprocessing Images for Learned Raw Denoising.* CVPR 2019.
- Wei, Fu, Yang, Huang. *A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising* (ELD). CVPR 2020.
- Ignatov et al. *Replacing Mobile Camera ISP with a Single Deep Learning Model* (PyNET). CVPRW 2020; AIM/MAI RAW-to-RGB challenges 2020–2022.
- Karaimer, Brown. *A Software Platform for Manipulating the Camera Imaging Pipeline.* ECCV 2016.
- Hirakawa, Parks. *Adaptive Homogeneity-Directed Demosaicing Algorithm* (AHD). 2005.
- Zamir et al. *Restormer.* CVPR 2022. / Chen et al. *NAFNet.* ECCV 2022. / Liang et al. *SwinIR.* ICCVW 2021.
- Hu, Wang, Lin. *FC4: Fully Convolutional Color Constancy.* CVPR 2017. / Afifi, Brown. *Deep White-Balance Editing.* CVPR 2020. / Barron. *Fast Fourier Color Constancy.* CVPR 2017.
- Mertens, Kautz, Van Reeth. *Exposure Fusion.* 2007. / Paris, Hasinoff, Kautz. *Local Laplacian Filters.* SIGGRAPH 2011. / Gharbi et al. *Deep Bilateral Learning* (HDRnet). SIGGRAPH 2017.

Specs & tools:

- LibRaw documentation and `dcraw_process`/`postprocess` parameters: https://www.libraw.org/docs
- rawpy (LibRaw Python bindings): https://letmaik.github.io/rawpy/
- Adobe DNG Specification (color matrices, DCP profiles): https://helpx.adobe.com/camera-raw/digital-negative.html
- RawTherapee / darktable developer docs (AMaZE, demosaic and tone-curve references): https://rawpedia.rawtherapee.com/ , https://docs.darktable.org/
- Related internal docs: [COLOR_MANAGEMENT_STRATEGY.md](COLOR_MANAGEMENT_STRATEGY.md), [NOISE_SHARPENING_STRATEGY.md](NOISE_SHARPENING_STRATEGY.md).

---

*This guide describes the correct RAW pipeline and current research, mapped onto this app's
real decode paths. Where it names a function, flag, or `postprocess` argument, that exists in
the code or in LibRaw today; §3 and §7 flag the live gaps. Treat the phased recommendations as
a sequence, not a single change — Phase 1 (unify + determinism) is worth shipping on its own.*
