# Portrait Enhancer — Roadmap

This roadmap tracks feature gaps relative to a complete RAW/portrait editor. The
portrait-specific pipeline (semantic layers, expression warp, frequency-separation
skin work, CodeFormer refine, color management, batch) is mature; the items below
are the general photo-editing fundamentals and manual-retouching tools that are
still missing.

Status legend: ☐ not started · ◐ partial · ☑ done

---

## P0 — Core editing gaps (highest leverage)

These are whole feature classes a general editor is expected to have. They unblock
the most common edits that currently have no workflow at all.

### Geometric / compositional tools ◐
Crop / straighten / flip shipped in the Qt app (`core/framing.py` + a `Geometry`
panel). See Phase 2 in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
- ☑ Crop with aspect-ratio presets (free, original, 1:1, 4:5, 5:4, 3:2, 2:3, 16:9)
- ☑ Rotate / straighten (angle slider) with an interactive crop-box + thirds overlay
- ☑ Flip horizontal / vertical
- ☑ Crop applied non-destructively, persisted in `.peproj`, undo/redo aware
- ☑ Crop honored on single-image export
- ☐ Draggable crop box with aspect-ratio **lock** (free-form drag only today)
- ☐ Batch/preset framing — intentionally deferred (a crop is image-specific)
- ☐ Perspective / keystone correction
- ☐ Lens correction (distortion, chromatic aberration, vignetting profiles)

### Interactive tone curve & color mixer ◐
- ☑ Point tone-curve widget (master luminance) — monotone-cubic curve with an
      interactive panel + histogram backdrop (`core/tone_curve.py`, Phase 4)
- ☐ Per-channel RGB curves
- ☑ HSL / color mixer: per-hue hue, saturation, luminance — 8-band weighted mixer
      with a band-selector panel (`core/color_mixer.py`, Phase 5)
- ☑ Histogram panel with shadow/highlight clipping warnings *(Phase 1 — done)*
- ☑ White-balance correction: eyedropper (with magnified loupe) + gray-world/
      white-patch auto + a true **Kelvin** Temperature / Tint model with illuminant
      **presets** (Tungsten / Fluorescent / Daylight / Flash / Cloudy / Shade).
      Pick/auto invert neutralizing gains to (Kelvin, tint) so the sliders stay in
      sync. Applied as per-channel gains in linear light, separate from the
      stylistic Temperature/Tint grade — `core/white_balance.py` + a
      `White Balance` panel (Phases 3 / 3b / 3c in
      [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md))
- ☐ Mired-spaced temperature slider travel; camera "As Shot" WB from RAW metadata

### Manual retouching / healing ☐
Everything is global or semantic-mask driven; there is no pixel-level repair.
- ☐ Spot / heal brush (content-aware blemish & object removal)
- ☐ Clone stamp
- ☐ Red-eye removal
- ☐ Teeth whitening as a discrete tool (eye-whiten exists; no lips/teeth equivalent)
- ☐ Stray-hair removal
- ☐ Dodge & burn brushes

---

## P1 — Local adjustment primitives

Masks today are AI segmentation plus freehand paint/erase of those same regions.
The standard local-adjustment primitives are missing.

- ☐ Radial gradient mask
- ☐ Linear gradient mask
- ☐ Luminosity / color-range mask
- ☐ Adjustment brush — paint an adjustment (e.g. exposure) rather than only editing
      a layer's coverage
- ☐ Brush hardness / flow and pressure-sensitive tablet support *(from README
      "Next Recommended Upgrades")*

---

## P2 — Detail & color refinement

- ☐ Real denoise with luma/chroma separation (current "Noise Reduc." is a plain
      Gaussian blur in `core/processing.py`)
- ☐ Texture slider (distinct from clarity)
- ☐ Output / print sharpening applied on export
- ☐ Split toning / color-grading wheels (shadows · midtones · highlights)
- ☐ Stronger LUT/profile management: validation previews, intensity mix, more
      LUT/profile formats *(from README "Next Recommended Upgrades")*

---

## P3 — Output & session workflow

- ☐ Export resize / long-edge constraint
- ☐ Watermarking on export
- ☐ Crop-on-export framing
- ☐ XMP / sidecar metadata write
- ☐ Full edit-history panel (today only mask edits have undo/redo)
- ☐ Snapshots / virtual copies
- ☐ Copy-paste settings between images
- ☐ CSV/HTML batch reports and richer retry filters beyond "latest failed batch"
      *(from README "Next Recommended Upgrades")*

---

## P4 — Performance & platform

- ☐ GPU acceleration for preview rendering and mask inference (current CUDA path
      covers resize and Gaussian-blur-heavy steps only) *(from README "Next
      Recommended Upgrades")*

---

## Notes

- The PySide6 (Qt) frontend is the forward path; the Tk app still has a deeper
  mask-edit workflow in places. New features should target Qt, with parity called
  out where the Tk app diverges.
- Anything that changes pixels geometrically (crop, rotate, straighten) must be
  threaded through both the interactive preview proxy and the full-resolution
  export render, and stored in `.peproj` / applied in batch.
