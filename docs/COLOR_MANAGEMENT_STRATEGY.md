# Color Management Strategy

A strategy and design document for implementing predictable color handling across an
image and video editing platform. The immediate target is this repository's portrait
editor, but the architecture is written so it can grow into timeline/video workflows
without reworking the core model.

This document is about color architecture, not a visual redesign. The goal is that a
pixel keeps a clear meaning from import, through edit math, through preview, through
export. Users should not need to understand ICC profiles to get correct output, but
advanced users should be able to choose working/output profiles deliberately.

---

## 1. Goals and non-goals

**Goals.**

- Preserve source color intent on import: embedded ICC for stills, RAW decode settings
  for camera files, and container/codec color metadata for video.
- Use a canonical internal model so filters, masks, previews, batch export, and future
  video frames agree.
- Do math in the right domain: linear RGB for exposure, compositing, blur, sharpen,
  white balance, relighting, and alpha operations; perceptual spaces only for perceptual
  controls.
- Export files with pixels and metadata that match: no sRGB-tagged wide-gamut pixels,
  no preserved source ICC after pixels have been converted away from that profile.
- Keep the common path simple: default import and export to sRGB SDR, with optional
  wide-gamut/HDR paths.
- Make preview honest enough for editing decisions, while allowing fast proxy paths.
- Keep projects and presets portable by serializing color decisions, not transient
  library objects.

**Non-goals.**

- Not building a full DI grading system in the first pass.
- Not making HDR the default for portrait editing. HDR should be a supported pipeline
  mode after SDR color management is correct.
- Not guaranteeing perfect display calibration. The platform can honor monitor profiles
  where the UI toolkit exposes them, but cannot calibrate user hardware.
- Not changing the existing adjustment UX just to expose implementation details.

---

## 2. Current state in this repo

The app already has useful color-management pieces, but they are split across code
paths and are not yet a single contract.

| Area | Current behavior | Design implication |
|---|---|---|
| Core processing | `core/utils.py` has sRGB transfer helpers and `working_space` can be `srgb` or `linear`. | Keep this, but make the working-space contract explicit and extensible. |
| White balance | `core/white_balance.py` applies gains in true linear light. | This is the right model and should become the reference pattern for other physical edits. |
| Tone/color controls | Tone curve, HSL mixer, LAB/HSV operations convert to display-ish sRGB and back. | Keep perceptual controls display-referred unless renamed/rebuilt as scene-linear controls. |
| RAW/Tk app | `portrait_enhancer/ui/app.py` has ImageCms input conversion, RAW color settings, RAW LUT, and ICC export policy. | Reuse the policy and normalize it into shared code. |
| Qt app | `_decode_image_file` captures ICC/EXIF/DPI but currently converts with Pillow to RGB without applying embedded ICC. Export re-embeds source ICC when metadata is kept. | Qt is the forward UI, so it needs the shared input transform and export policy. |
| Export | `process_all_layers` returns 8-bit Pillow images after `output_transform`. | Export needs an explicit output profile transform before quantization and metadata write. |
| Batch | `batch_runner.py` carries `color_settings`, but should use the same shared color engine as single export. | Single image and batch must call one API, not duplicate color decisions. |

The most important gap: the app has settings named like color management, but the
canonical pixel meaning is still mostly "sRGB float, optionally linearized for selected
math." That is a good starting point for SDR portrait editing, but not enough for a
platform-level color contract.

---

## 3. Product policy

The default product policy should be conservative:

1. **Default import target: sRGB SDR.**
   Non-RAW stills with embedded ICC are converted to sRGB on import. Untagged stills
   are assumed sRGB, with a visible warning only in an advanced diagnostics panel.

2. **Default edit space: scene-linear sRGB.**
   Keep UI sliders display-referred where they already are, but normalize physically
   meaningful operations to linear sRGB internally. The code can still expose an
   `srgb` preview mode for speed, but the target-quality render should be linear for
   math-heavy stages.

3. **Default preview: display-managed sRGB.**
   The preview renderer produces display-ready pixels for the UI. If monitor-profile
   support is unavailable, preview is sRGB. If it is available, convert from the
   document preview profile into the display profile at the final UI boundary.

4. **Default export: sRGB with embedded sRGB ICC.**
   JPEG/PNG/TIFF SDR exports should be encoded as sRGB and tagged as sRGB. Users can
   choose "preserve source" only when the output pixels are still in that source color
   encoding.

5. **Wide gamut is opt-in.**
   Display P3 and Adobe RGB export are useful, but they must be explicit output
   choices with correct profile embedding and gamut mapping.

6. **HDR/video is a separate mode.**
   Rec. 2100 PQ/HLG, Rec. 2020 primaries, and 10-bit output should not be hidden under
   generic "gamma" controls. They need explicit timeline/project color settings.

---

## 4. Core concepts

### 4.1 Color spaces are not just primaries

A color encoding has at least:

- primaries and white point, such as sRGB/D65, Display P3/D65, Rec. 2020/D65;
- transfer function, such as sRGB EOTF/OETF, gamma 2.2, PQ, HLG, or linear;
- range and encoding, such as full-range RGB or video-range YCbCr;
- optional ICC profile or video metadata describing the above.

Do not model "linear" as a complete color space by itself. It is a transfer
function paired with primaries and white point, for example `linear-srgb` or
`linear-rec2020`.

### 4.2 Pixel domains

The platform should distinguish these domains explicitly:

| Domain | Meaning | Used for |
|---|---|---|
| Source encoded | Pixels as loaded from file/container | Metadata capture, reversible decode diagnostics |
| Working linear RGB | Floating point, scene/display-linear depending on project mode | Exposure, compositing, WB, alpha, blur, sharpen, masks |
| Perceptual edit space | Derived transient space: Lab, Oklab/Oklch, HSV/HSL | Hue/saturation/luminance tools, color range masks |
| Preview encoded | Display-ready RGB for the canvas | UI rendering, histograms when explicitly display-referred |
| Export encoded | Final file/container encoding | JPEG/PNG/TIFF/video encoding and metadata |

### 4.3 Scene-referred vs display-referred

For the current portrait editor, SDR display-referred editing is acceptable: source
pixels are normalized for pleasant display, and controls operate on that image. For a
video platform, the model must allow scene-referred pipelines too:

- **Display-referred SDR:** good for JPEG/PNG, social export, existing portrait edits.
- **Scene-referred / log / RAW:** good for camera-original video, grading, HDR, and
  round-tripping to professional formats.

Recommendation: ship the still-image foundation as display-referred SDR with correct
profile transforms, but name internal types so a future scene-referred video pipeline can
coexist.

---

## 5. Proposed architecture

### 5.1 Shared color engine

Create a shared module, for example `portrait_enhancer/core/color_management.py`, that
owns all profile, transfer, and metadata policy decisions.

Suggested public API:

```python
@dataclass(frozen=True)
class ColorProfileRef:
    name: str                 # "srgb", "display-p3", "adobe-rgb", "rec709", ...
    icc_bytes: bytes | None = None
    primaries: str | None = None
    transfer: str | None = None
    white_point: str = "D65"


@dataclass(frozen=True)
class ColorPipelineConfig:
    input_policy: str         # "auto", "assign_srgb", "ignore"
    working_profile: str      # "linear-srgb" initially
    preview_profile: str      # "srgb" initially
    output_profile: str       # "srgb", "display-p3", "adobe-rgb"
    output_transfer: str      # "srgb", "gamma22", "linear", later "pq", "hlg"
    icc_policy: str           # "embed_output", "preserve_source_if_valid", "none"
    gamut_mapping: str        # "relative_colorimetric", "perceptual", "clip"
    bit_depth: int            # 8, 16, float32 for internal buffers


@dataclass
class DecodedImage:
    pixels: np.ndarray        # float32 RGB in working profile, range policy documented
    source_metadata: dict
    color_state: dict         # serializable audit trail
```

Core operations:

- `decode_still(path, config) -> DecodedImage`
- `decode_raw(path, config) -> DecodedImage`
- `to_working(pixels, source_profile, working_profile) -> np.ndarray`
- `to_preview(working_pixels, config, display_profile=None) -> np.ndarray`
- `to_export(working_pixels, config) -> tuple[np.ndarray, dict]`
- `build_export_metadata(source_metadata, color_state, config, ext) -> dict`
- `normalize_color_settings(payload) -> ColorPipelineConfig`

Qt, Tk, batch, and tests should call this module. UI classes should not implement
profile policy themselves.

### 5.2 Internal buffer format

Use `float32` RGB for render buffers. Avoid repeated uint8 conversions except at the UI
or export boundary.

Initial target:

- Working profile: `linear-srgb`
- Value range: nominal `[0, 1]`, with a later option for unclamped HDR/scene values
- Alpha/masks: linear scalar `[0, 1]`
- Preview: 8-bit sRGB for current Qt canvas
- Export: 8-bit JPEG/PNG initially; 16-bit TIFF/PNG as a follow-up

Important: clamp only at domain boundaries or when an operation requires bounded input.
Repeated mid-pipeline clamping makes highlights, saturated colors, and HDR expansion
harder to support later.

### 5.3 Transform backend

Use Pillow `ImageCms`/LittleCMS for ICC transforms in the near term because Pillow is
already a dependency. Wrap it behind the shared color engine so the backend can later
move to `lcms2`, OpenColorIO, or platform APIs without touching UI/render code.

Recommended implementation levels:

1. **Level 1: ICC in/out correctness.**
   Convert embedded still ICC to sRGB or configured output; embed output ICC.

2. **Level 2: named profile registry.**
   Built-in refs for sRGB, Display P3, Adobe RGB, Rec. 709, Rec. 2020. Store profile
   refs in project settings.

3. **Level 3: OCIO-style pipeline.**
   Add scene-linear, log, LUT input/output transforms, view transforms, and looks.
   This is when video grading starts to need OpenColorIO seriously.

### 5.4 Document color context

Every open image/project/timeline should have a `ColorContext`.

```python
@dataclass
class ColorContext:
    source_profile: ColorProfileRef | None
    working_profile: ColorProfileRef
    preview_profile: ColorProfileRef
    output_profile: ColorProfileRef
    display_profile: ColorProfileRef | None
    input_transform_applied: str
    output_transform_policy: str
    warnings: list[str]
```

This context is not just metadata. It is part of render identity and should be included
in cache keys for previews, analysis caches, LUT caches, and batch resume keys where
changing it changes pixels.

---

## 6. Image pipeline design

### 6.1 Import

For non-RAW stills:

1. Read pixels and metadata without losing ICC/EXIF/DPI.
2. Apply EXIF orientation before display/render.
3. If embedded ICC exists and input policy is `auto`, convert to working profile.
4. If no ICC exists, assign sRGB by default.
5. Record `input_transform_applied` in serializable metadata.

For RAW stills:

1. Decode with camera white balance by default.
2. Decode into a known output color space, initially sRGB or linear sRGB.
3. Store RAW settings in `color_state`: white balance source, output color space,
   highlight mode, demosaic quality, and any RAW LUT.
4. Treat camera profiles and "as shot" white balance as follow-up work.

Current code note: `_decode_image_file` in the Qt path should stop being the final
authority. It should delegate to the shared decoder so embedded ICC transforms are not
Tk-only.

### 6.2 Processing order

Recommended SDR order:

1. Decode/convert source to working RGB.
2. Apply white balance in linear light.
3. Apply exposure and physical tone operations in linear light.
4. Apply global perceptual tone/color controls as explicit display-referred operations.
5. Apply face refinement and selective layers in the same working profile.
6. Composite masks in linear light.
7. Convert to preview or export profile.
8. Quantize and write metadata.

Some existing controls are intentionally display-referred, such as HSL mixer and
LAB-based chroma operations. That is fine, but each function should make the conversion
explicit: `working -> display edit space -> working`.

### 6.3 LUTs

LUTs need declared input and output assumptions.

Minimum policy:

- RAW input LUTs: apply immediately after RAW decode, before portrait adjustments.
- Creative 3D LUTs: apply as a display-referred look after global tone and before
  output conversion.
- LUT metadata: store path, title, size, domain, intended input profile when known,
  and intensity.

Do not apply a random `.cube` in linear working space unless the user or LUT metadata
says it was authored for linear input. Most creative LUTs expect display-referred
Rec. 709/sRGB-like input.

### 6.4 Histograms and scopes

Histograms should declare their domain:

- RGB histogram: preview/output encoded by default, because that matches what users see.
- Luma histogram: offer both display luma and linear scene value in advanced mode.
- Clipping warnings: compute export clipping after output transform, not just working
  buffer clipping.
- Video scopes later: waveform, vectorscope, RGB parade should use the timeline/output
  color context.

---

## 7. Video pipeline design

Video adds frame sequences, codec metadata, limited/full range, chroma subsampling, and
timeline color spaces. The still-image pipeline should be shaped now so video can reuse
it.

### 7.1 Video import metadata

For each stream, capture:

- color primaries, transfer characteristic, matrix coefficients;
- full vs limited range;
- bit depth;
- pixel format and chroma subsampling;
- mastering display metadata and content light metadata for HDR when present;
- container vs codec disagreement warnings.

Common SDR default:

- Rec. 709 primaries
- Rec. 709/BT.1886-ish display behavior for broadcast video
- YCbCr to RGB conversion honoring matrix and range

Common HDR modes:

- Rec. 2100 PQ
- Rec. 2100 HLG
- Rec. 2020 primaries
- 10-bit or higher processing/export

### 7.2 Timeline color space

A video timeline should have one canonical timeline color space. Suggested first
choices:

- `timeline-sdr-linear-rec709`
- `timeline-sdr-linear-srgb` for image-first projects
- later: `timeline-hdr-linear-rec2020`, `timeline-acescg`, or OCIO-configured spaces

Each clip gets an input transform into the timeline space. Each viewer/export gets an
output/view transform from the timeline space.

### 7.3 Mixed-media projects

For mixed images and video:

- Convert stills into the timeline color space at import/render time.
- Do not assume still sRGB equals video Rec. 709 transfer behavior without an explicit
  transform.
- Make per-asset input overrides available: assign profile, convert profile, override
  range, override transfer.
- Cache decoded proxies with a color-context hash so changing timeline color invalidates
  only affected proxies.

### 7.4 Video export

Export should write both pixels and metadata:

- SDR web: Rec. 709/sRGB-compatible output, full or video range chosen by codec target.
- HDR: Rec. 2100 PQ/HLG, Rec. 2020 primaries, 10-bit HEVC/AV1/ProRes where supported.
- Intermediate/master: 10/12/16-bit codec, no unintended gamut clipping, explicit
  metadata.

Avoid presenting "gamma22" as a video export mode. Users need named targets such as
`Web SDR`, `Broadcast Rec.709`, `HDR10 PQ`, or `HLG`.

---

## 8. UI design

Most users should see intent-based choices, not implementation vocabulary.

### 8.1 Basic UI

Default visible controls:

- Import: Auto
- Export color: sRGB
- Embed color profile: On

Export presets:

- `Web / Social: sRGB JPEG/PNG`
- `Print / Lab: sRGB or Adobe RGB TIFF/JPEG`
- `Apple / Wide Gamut: Display P3`
- later: `HDR Video: Rec.2100 PQ`

### 8.2 Advanced UI

Advanced color panel:

- Input profile: auto, assign sRGB, ignore embedded, choose profile
- RAW decode: camera WB, auto WB, output color, highlight mode
- Working space: linear sRGB initially, later Display P3/Rec.2020/OCIO choices
- Preview transform/view: sRGB SDR initially
- Export profile: sRGB, Display P3, Adobe RGB, custom ICC
- Rendering intent: relative colorimetric, perceptual
- Gamut warning: on/off
- Black point compensation: on/off when backend supports it
- Diagnostics: source tag, applied transform, output tag, warnings

### 8.3 Warnings

Warnings should be specific and actionable:

- "This image has no embedded profile. Treating it as sRGB."
- "Source profile was converted to sRGB on import. Preserve Source ICC is unavailable."
- "Output profile is Display P3. Some web viewers may show muted color if they ignore ICC."
- "HDR export requires 10-bit output. Current format is 8-bit."

Do not block export for warnings unless pixels and metadata would be contradictory.

---

## 9. Data model and persistence

Projects and presets should store serializable color settings.

Recommended schema:

```json
{
  "color_management_version": 1,
  "input": {
    "policy": "auto",
    "assigned_profile": null,
    "raw_white_balance": "camera",
    "raw_output_profile": "srgb"
  },
  "working": {
    "profile": "linear-srgb",
    "bit_depth": "float32",
    "allow_hdr_values": false
  },
  "preview": {
    "view_transform": "srgb"
  },
  "export": {
    "profile": "srgb",
    "transfer": "srgb",
    "icc_policy": "embed_output",
    "rendering_intent": "relative_colorimetric",
    "bit_depth": 8
  },
  "looks": {
    "lut_path": "",
    "lut_intensity": 1.0,
    "lut_input_profile": "srgb"
  }
}
```

Migration from current settings:

| Current key | New meaning |
|---|---|
| `input_profile=auto` | `input.policy=auto` |
| `input_profile=ignore` | `input.policy=ignore_embedded` |
| `raw_white_balance` | `input.raw_white_balance` |
| `raw_colorspace` | `input.raw_output_profile` |
| `working_space=srgb` | keep as compatibility, render target should migrate to `linear-srgb` |
| `working_space=linear` | `working.profile=linear-srgb` |
| `output_transform=srgb` | `export.transfer=srgb`, `export.profile=srgb` unless overridden |
| `icc_policy=srgb` | `export.icc_policy=embed_output`, `export.profile=srgb` |
| `icc_policy=preserve_source` | `export.icc_policy=preserve_source_if_valid` |

Presets should include color settings only when they intentionally define a look or
output behavior. A portrait retouch preset probably should not silently change an export
profile unless it is marked as an export preset.

---

## 10. Implementation phases

### Phase 1: Make current SDR behavior correct

- Add shared `core/color_management.py`.
- Move default/normalize color settings out of UI classes and batch runner.
- Use the shared decoder in Qt and Tk.
- Convert embedded ICC to sRGB/working profile on import in Qt.
- Build export metadata from output color state, not raw source metadata.
- Embed sRGB ICC for default SDR exports.
- Add tests for source ICC conversion, untagged assumption, and export ICC policy.

Definition of done: a Display P3 or Adobe RGB tagged image imports to the same visual
appearance as color-managed viewers and exports as correctly tagged sRGB by default.

### Phase 2: Strengthen working-space contract

- Rename `working_space` internally to `working_profile` or normalize it at API
  boundaries.
- Make `linear-srgb` the target-quality render path for physical operations.
- Audit every function that calls `working_to_display`/`display_to_working`.
- Avoid uint8 round-trips in core operations where feasible.
- Include color context in preview/cache keys.
- Add golden-image tests around white balance, compositing, and export transforms.

Definition of done: preview and full-resolution export match within an agreed tolerance
for the same color context.

### Phase 3: Output profiles and gamut handling

- Add output profile choices: sRGB, Display P3, Adobe RGB, custom ICC.
- Add rendering intent/gamut mapping policy.
- Add gamut warning overlay or diagnostics.
- Add 16-bit TIFF/PNG export path for wide-gamut work.
- Validate that embedded ICC always matches output pixels.

Definition of done: wide-gamut exports open correctly in color-managed viewers and
fallback behavior is documented for unmanaged viewers.

### Phase 4: LUT and look management

- Add LUT intensity.
- Add LUT input/output assumptions.
- Validate `.cube` domain/size before use.
- Separate RAW technical LUTs from creative looks.
- Add preview thumbnails for LUTs under the active color context.

Definition of done: LUT application is deterministic and does not depend on hidden
working-space assumptions.

### Phase 5: Video/timeline color

- Add stream color metadata parser around the chosen video backend.
- Define timeline color settings.
- Convert YCbCr/video-range frames to timeline RGB correctly.
- Add named SDR export targets.
- Add HDR project mode only after 10-bit render/export is available.

Definition of done: imported Rec. 709 SDR video and sRGB stills preview/export
consistently in a mixed-media timeline.

---

## 11. Testing strategy

Color management needs tests that catch visual-looking but wrong metadata bugs.

### Unit tests

- sRGB transfer round-trip: encoded -> linear -> encoded.
- ICC policy normalization and migration.
- Untagged input defaults to assigned sRGB.
- Embedded non-sRGB input converts to sRGB by default.
- `preserve_source_if_valid` refuses to preserve source ICC after conversion.
- Output profile ICC bytes match output profile.
- White-balance gains operate in linear light.
- LUT application is identity-stable.

### Integration tests

- Load tagged sRGB, Display P3, Adobe RGB fixtures and export default sRGB.
- Compare mean/patch colors against reference transforms.
- Batch export produces the same pixels/ICC policy as single-image export.
- Project/preset round trip preserves color settings.
- Preview and export agree within tolerance after resizing and output sharpening.

### Fixture set

Create small synthetic fixtures:

- sRGB color chart with embedded sRGB ICC.
- Display P3 chart with saturated red/green patches.
- Adobe RGB chart.
- Untagged RGB chart.
- 16-bit TIFF gradient.
- RAW fixture if licensing/storage allows.
- Later: Rec. 709 full-range and limited-range video clips, HDR PQ clip.

### Manual QA

- Open exports in macOS Preview, Photoshop/Affinity, Chrome/Safari, and an unmanaged
  viewer to understand failure modes.
- Verify that a P3 source does not become oversaturated after default export.
- Verify that default JPEGs have embedded sRGB ICC.
- Verify that "preserve source" is disabled or changed when pixels were converted.

---

## 12. Engineering rules

- Every image/frame buffer crossing a module boundary must document its color profile
  and transfer function.
- Do not pass around naked `np.ndarray` for new pipeline APIs without a color context.
- Do not preserve an ICC profile unless the output pixels are actually in that profile.
- Do not use HSV/LAB/Oklab conversions on linear buffers without explicitly converting
  to the intended encoded/perceptual domain first.
- Do not use `gamma22` as a substitute for sRGB. sRGB has a specific transfer curve.
- Do not clamp mid-pipeline unless the operation requires it.
- Do not let preview-only shortcuts change exported pixels.
- Do not let batch use a different color path from single-image export.

---

## 13. Recommended decisions

1. Make **sRGB SDR with embedded sRGB ICC** the default import/export experience.
2. Make **linear sRGB float32** the target-quality internal working space for current
   portrait editing.
3. Centralize all profile and metadata policy in `core/color_management.py`.
4. Treat Display P3, Adobe RGB, Rec. 2020, PQ, and HLG as explicit named modes, not
   variants of the existing `output_transform` string.
5. Bring Qt to parity with the shared color engine first, because Qt is the forward UI.
6. Add video color management only after still-image SDR correctness is tested.

---

## 14. Standards and references

- International Color Consortium, ICC profile specification and ICC v4.4 resources:
  https://www.color.org/
- ICC specifications overview:
  https://www.color.org/icc_specs2.xalter
- ICC.1 v4 specification page:
  https://www.color.org/v4spec.xalter
- ITU-R BT.2100 HDR television recommendation:
  https://www.itu.int/rec/r-rec-bt.2100
- W3C CSS Color Module Level 4, useful for web-facing color spaces such as Display P3:
  https://www.w3.org/TR/css-color-4/
