# Implementation Plan

Derived from [ROADMAP.md](../ROADMAP.md). Phases are ordered by ROI: a cheap,
purely-additive quick win first, then the highest-value core feature.

---

## Phase 1 — Histogram panel  *(quick win)*

**Goal:** Show a live RGB + luminance histogram of the current edited preview, with
shadow/highlight clipping indicators.

**Why first:** Nearly free to compute from the preview array that already exists,
purely additive (touches no existing edit math, masks, or geometry), and adds
immediate professional polish. Good warm-up before the larger crop work.

**Design**
- Pure computation lives in `portrait_enhancer/core/histogram.py` so it is unit
  testable without Qt: `compute_histogram(image) -> dict` returning per-channel
  256-bin counts (red/green/blue/luma), total pixel count, and clip fractions at
  pure black / pure white.
- `HistogramWidget(QWidget)` in the Qt window paints the precomputed bins with
  `QPainter` (additive RGB curves over a dark panel).
- Update on render completion in `_on_preview_render_finished`, driven by
  `self.preview_image`. Clear when no image is loaded.

**Touch points**
- New: `portrait_enhancer/core/histogram.py`
- New: `tests/test_histogram_smoke.py`
- Edit: `portrait_enhancer/ui_qt/main_window.py` (widget + preview-card placement + update hook)

**Acceptance**
- `compute_histogram` returns correct counts and clip fractions for known inputs.
- Histogram redraws after every preview render; clip markers light up on
  blown/crushed images. Existing Qt smoke test still passes.

**Status: ☑ done (this change)**

---

## Phase 2 — Crop & straighten  *(highest value)*

**Goal:** Non-destructive crop, straighten (angle), and flip, honored in preview,
export, and batch.

**Why:** The most conspicuous gap for a RAW editor. Cost is contained because
`process_all_layers` returns a finished `PIL.Image` at three call sites
(preview worker, export, batch); crop/straighten can be applied as an
**output-stage transform** on that image without touching the segmentation/mask
pipeline or per-layer math.

**Design**
- Geometry op (rotate-by-angle + rectangular crop + flip) as a pure function in a
  new `core` module, applied to the rendered `PIL.Image`.
- Apply *after* masks are generated and painted (masks stay in full-frame
  coordinates) to avoid re-running segmentation.
- Document state: add `crop_rect`, `straighten_angle`, `flip_h/flip_v` to
  `_serialize_project_state`; read in all three render paths.
- UI: crop overlay with aspect presets (free, original, 1:1, 4:5, 16:9, custom),
  straighten angle slider with horizon guide, flip buttons.

**Edge cases handled**
- Mask editing operates in source space, so the preview shows the **un-transformed**
  frame while mask-edit mode is active (`_display_framing`); crop-edit and
  mask-edit are mutually exclusive.
- Crop-edit mode shows the full straightened frame with a draggable crop-box
  overlay (rule-of-thirds + 8 handles); the crop is baked only outside that mode.
- Compare modes apply the same display framing to before/after before composition.
- Framing round-trips through `.peproj`, the undo/redo document history, and is
  reset on fresh image open and `reset_all`.

**Implemented as**
- New: `portrait_enhancer/core/framing.py` (pure geometry + crop-box math) and
  `tests/test_framing_smoke.py`.
- Edit: `main_window.py` — `HistogramWidget`-style `Geometry` section (crop toggle,
  aspect presets, straighten slider, flip H/V, reset), crop overlay + drag in
  `ImagePreviewLabel`, framing in project/snapshot/signature state, and framing
  baked into the preview display and export.
- Named "framing" to avoid colliding with the existing `geometry=` landmark-guides
  parameter of `process_all_layers`.

**Deliberately out of scope**
- Batch/preset framing: a crop rectangle is image-specific and cannot sensibly
  apply across a folder of differently-composed images, so framing is project-only
  (not in `.pepreset`, not in `batch_runner`).
- Aspect-locked handle dragging, perspective/keystone, and lens correction remain
  follow-ups (tracked in ROADMAP).

**Status: ☑ done (this change)** — full suite green except two pre-existing,
unrelated failures (`test_preset_state_smoke`, `test_batch_helpers_smoke`) that
also fail on `HEAD`.

---

## Phase 3 — White balance correction

**Goal:** True white-balance *neutralization* (remove a color cast), distinct from
the existing stylistic Temperature/Tint grade.

**Why this approach:** WB correction is a per-channel gain applied in **linear**
light, not a hue rotation. The current `adjust_color_balance_preserve_chroma`
(HSV/LAB hue move) is a stylistic grade and cannot fully clear a cast. The linear
infrastructure already exists (`srgb_to_linear` / `linear_to_srgb`, and a
`working_space` that can be `linear`). Keep the existing Temperature/Tint as the
stylistic grade; add WB correction as a separate first-stage operation.

**Design — eyedropper (primary path)**
1. User clicks a should-be-neutral pixel; sample a small neighborhood (e.g. 5×5)
   from the **source** working image to reduce noise.
2. Convert sample to linear: `p = (pr, pg, pb)`.
3. Brightness-preserving target `t = (pr + pg + pb) / 3`; gains `gain_c = t / p_c`.
4. Multiply the whole image by the gains in linear space → picked pixel becomes
   neutral at the same luminance.
- Guard denominators (clamp near-zero channels) and clamp gains to ~0.25–4×.

**Design — auto WB (same apply path)**
- Gray-world: `gain_c = mean_luma / mean(channel_c)` (safe default button).
- White-patch: `gain_c = max / channel_max`.

**Pipeline placement**
- Apply first in `process_global` (top, before/just after exposure), in linear:
  linearize → multiply by gains → return to working space. Order-independent
  because it samples the pre-grade source.
- Store the **gains** (`wb_gain = [gr, gg, gb]`) plus pick coords as a global
  param; serialize into `.peproj` / `.pepreset` with the other color settings.
- UI: reuse the existing canvas coordinate mapping in `ImagePreviewLabel`
  (rel_x/rel_y → callback) for a "pick WB" mode that samples `preview_array`;
  add a gray-world auto button and a reset.

**Out of scope (follow-up):** a true Kelvin Temperature + Tint model
(CCT → chromaticity → linear-RGB gains). Heavier and overlaps the existing
stylistic slider. RAW WB stays at decode time (existing `RAW WB` setting); this
phase targets non-RAW images and interactive override on top of RAW decode.

**Implemented as**
- New: `portrait_enhancer/core/white_balance.py` — `apply_white_balance_gains`,
  `gains_from_neutral_sample`, `gray_world_gains`, `white_patch_gains`,
  `normalize_gains` (clamped 0.25–4×) — plus `tests/test_white_balance_smoke.py`.
- Edit: `core/processing.py` — WB applied first in `process_global`, in linear,
  before exposure/tone.
- Edit: `main_window.py` — `White Balance` panel (Pick Neutral, Auto Gray, Auto
  White, Reset, gain readout); WB-pick mouse mode in `ImagePreviewLabel`; gains
  stored in `color_settings["wb_gain"]`.
- Gains live in `color_settings`, so project/preset save, undo/redo history, and
  the history signature serialize them **for free**.

**Color-space note (subtle but important):** gains are ratios in true linear
light. The UI samples `preview_array`, which is display/source sRGB, so estimators
always pass `working_space="srgb"`. `process_global` passes the actual working
space — both resolve to the same true-linear space, so the pick is consistent
regardless of the linear/sRGB working-space setting.

**Edge cases handled**
- WB-pick mode shows the **un-transformed** frame (`_display_framing`) so picks map
  directly to source pixels; mutually exclusive with crop-edit and mask-edit.
- Eyedropper samples a 5×5 neighborhood to reduce noise; gains clamped to 0.25–4×
  with zero-channel guards.

**Acceptance — verified**
- Picking a neutral patch on a bluish cast makes that patch read R≈G≈B with
  luminance preserved; gray-world neutralizes a uniform cast end-to-end through the
  render pipeline; gains round-trip through project save and undo/redo. (Confirmed
  via unit tests + a headless window integration check.)

**Status: ☑ done**

---

## Phase 3b — White balance adjustment UX

**Goal:** Make WB *adjustable*, not just pickable. The pick/auto path could only
neutralize; users almost always want to fine-tune ("a touch warmer").

**Design**
- Temperature + Tint sliders (−100..100) backed by the existing gain engine:
  `effective_gain = base_gain × offset_gains(temp, tint)`, computed in
  `process_global`. Temp warms (red up / blue down); Tint shifts green↔magenta.
- `base_gain` comes from pick/auto; the sliders are relative offsets on top. A
  fresh pick/auto resets Temp/Tint to 0 (new neutral base).
- `wb_temp` / `wb_tint` stored in `color_settings` (serialized + undo/redo free).
- Raw gain numbers hidden: the panel shows "White balance: neutral/adjusted" with
  the effective gain in a tooltip; the sliders are the primary control.
- Eyedropper **loupe**: a magnified 9×9 source-pixel grid with a center crosshair
  and live R,G,B readout follows the cursor in pick mode, so users can land on a
  genuinely neutral spot.

**Superseded by Phase 3c:** the relative-offset Temp/Tint model
(`offset_gains`/`effective_gains`) was replaced by the Kelvin model below. The 3b
UI scaffolding (two sliders, status line, eyedropper loupe) was repurposed.

**Status: ☑ done (superseded by 3c)**

---

## Phase 3c — True Kelvin model + illuminant presets

**Goal:** Make Temperature a real Kelvin value with a green/magenta Tint, plus
one-click illuminant presets — the standard pro mental model.

**Design**
- Canonical WB state is `(wb_temp_k, wb_tint)` in `color_settings`; the applied
  per-channel gains are derived: `kelvin_tint_to_gains(K, tint)`. Identity is
  anchored at the sRGB white point (D65 ≈ 6500K) / tint 0, so a default image is
  unchanged.
- Temperature → gains via a Tanner-Helland blackbody approximation, taken as
  `illuminant(D65) / illuminant(K)` so slider-right (higher K) = warmer image.
  Tint scales the green channel (magenta ↔ green).
- **Inversion** `gains_to_kelvin_tint`: temperature from the red/blue ratio
  (monotonic in K, found by bisection), tint from the residual green. The
  eyedropper and gray-world/white-patch auto compute neutralizing gains and invert
  them to `(K, tint)`, so the **sliders always reflect a pick/auto** — they stay in
  sync, which was the hard part flagged in 3b.
- Presets: Tungsten 3200 / Fluorescent 4000+18 / Daylight 5500 / Flash 5500 /
  Cloudy 6500 / Shade 7500, as corrective values relative to D65.

**Implemented as**
- `core/white_balance.py`: `kelvin_tint_to_gains`, `gains_to_kelvin_tint`,
  `neutral_sample_to_kelvin_tint`, `clamp_temp`/`clamp_tint`, `PRESETS`, and the
  Helland CCT→linear helper (+ tests: model direction, K↔gains round-trip,
  pick→neutralize, presets).
- `core/processing.py`: `process_global` applies `kelvin_tint_to_gains(K, tint)`.
- `main_window.py`: Kelvin Temp slider (2000–12000K), Tint slider (−150..150),
  preset combo (shows "Custom" when sliders are off-preset), pick/auto invert to
  `(K, tint)`; eyedropper loupe retained.

**Known limitation:** very strong synthetic casts can exceed the blackbody locus
range and clamp at 2000/12000K (real-world casts fall within range).

**Out of scope (follow-up):** mired-spaced slider travel for perceptual uniformity;
camera "As Shot" from RAW metadata.

**Status: ☑ done (this change)** — verified via unit tests + headless integration
(default 6500K = identity; Temp warms the render; Tungsten preset cools and syncs
sliders + combo; manual edit flips combo to "Custom"; pick inverts to a warm K and
neutralizes a bluish cast; serialize + undo/redo).

---

## Phase 4 — Interactive tone curve

**Goal:** A point-based tone curve for precise tonal shaping, on top of the band
sliders — the marquee precision-tone tool, and the natural sequel to the histogram.

**Design**
- Pure core `core/tone_curve.py`: a curve is `(x, y)` control points in [0, 1]
  with anchored endpoints, interpolated with a **monotone cubic** (Fritsch-Carlson)
  spline into a 256-entry LUT (smooth, no overshoot). `apply_curve` maps luminance
  through the LUT and scales RGB, preserving chroma like the band tone curve.
  Interactive helpers (`nearest_point`, `add_point`, `move_point`, `remove_point`)
  are pure and unit tested.
- Stored in `color_settings["tone_curve"]` (identity = `[[0,0],[1,1]]`), so project
  /preset save, undo/redo, and the history signature serialize it for free. Edits
  always reassign a new list so history snapshots stay independent.
- Applied in `process_global` after the band tone sliders.
- `ToneCurveWidget`: drag points, click to add, double-click to remove, with a
  faint luminance-histogram backdrop (reuses the Phase 1 histogram) and a
  rule-of-thirds grid. A `Tone Curve` panel hosts it plus a Reset.

**Status: ☑ done (this change)** — 15 core tests + headless integration (lift curve
brightens 127→183; serialize; undo restores identity / redo restores the lift;
add/move/remove point ops; histogram-backed widget paints; reset).

---

## Phase 5 — HSL color mixer

**Goal:** Per-hue-band hue / saturation / luminance grading — the last item in the
P0 "tone curve & color mixer" group, and the missing per-color control (only global
vibrance/saturation existed before).

**Design**
- Pure core `core/color_mixer.py`: 8 bands (red…magenta) at fixed hue centers; each
  pixel is weighted into bands by a smooth raised-cosine window over hue (bands
  overlap so adjustments blend) and gated by saturation so near-gray pixels are
  untouched. Hue shift, saturation, and luminance deltas accumulate per band, then
  apply in HSV. Helpers: `default_color_mixer`, `normalize_color_mixer`,
  `is_identity` (+ tests, including band isolation and gray-immunity).
- Stored in `color_settings["color_mixer"]`, so project/preset save, undo/redo, and
  the history signature serialize it for free. Each edit reassigns a freshly
  normalized dict so history snapshots stay independent.
- Applied in `process_global` after vibrance/saturation.
- UI: a `Color Mixer (HSL)` panel with a band selector combo + Hue/Saturation/
  Luminance sliders (per-band, remembered on switch) + Reset Color / Reset All.

**Status: ☑ done (this change)** — 11 core tests + headless integration (red
sat −100 fully desaturates a red image; band switch loads per-band values; blue
adjust leaves red untouched; gray immune; serialize + undo/redo; resets).

---

## Later phases

Tracked in [ROADMAP.md](../ROADMAP.md): per-channel RGB curves, manual
retouching/healing, local-adjustment masks (radial/linear/range), real denoise, and
output/workflow tooling.
