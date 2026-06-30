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

- ☑ Real denoise with luma/chroma separation — this line was stale: `core/processing.py`
      already does edge-aware luma/chroma-separated bilateral filtering, preferring an
      optional learned DnCNN denoiser when its model is installed (`core/denoise.py`).
      2026-06-23: added a dedicated **Color NR** slider (`color_noise_red`) so chroma can be
      boosted independently of the main Luminance NR slider, instead of one shared amount.
- ☑ Output / print sharpening applied on export — 2026-06-23: `apply_output_sharpening`
      (`core/processing.py`) applies a final pass calibrated to the export's actual pixel
      size (after resize), with an Off/Low/Standard/High selector in both the single-image
      Export dialog and Collection Export dialog (Quick Export and Export All reuse the
      last-chosen level). Wired into the collection batch runner (`batch_runner.py`) too.
- ☑ Sharpening Amount/Radius/Masking — 2026-06-23: added `sharpen_radius`/`sharpen_masking`
      sliders alongside the existing Sharpness (Amount) slider, wired into
      `_apply_unsharp_mask`'s previously-hardcoded `radius`/`edge_threshold`. Defaults match
      the old hardcoded values exactly, so existing presets/projects render unchanged.
- ☑ Per-region noise reduction — 2026-06-23: added a `noise_red` slider to Skin, Background,
      and Person (previously global-only), wired through the same `_apply_luma_chroma_denoise`
      and run before each layer's other detail work. See
      [docs/NOISE_SHARPENING_STRATEGY.md](docs/NOISE_SHARPENING_STRATEGY.md) for the
      region-by-region strategy. Eyes/Hair/Lips/Face/Subjects intentionally still rely on
      global NR + masked Sharpen/Clarity (those are sharpen-only regions anyway).
      Same day: extended the existing on-open auto-suggestion (`suggest_global_auto_values`,
      previously Luminance NR/Sharpness only) to also seed **Color NR** (boosted only when
      measured chroma noise exceeds luminance noise) and to seed Skin/Background/Person's
      Noise Reduc. from *that region's own mask* (`suggest_region_noise_red`,
      `estimate_noise_sigma`'s new optional mask param) rather than the whole-image number.
      The existing per-slider "Auto" buttons were also fixed to be region-aware for these
      three layers (previously every Auto button used the whole-image estimate regardless of
      layer). Still purely pixel-measured, no EXIF/ISO/camera metadata involved.
- ☐ Texture slider (distinct from clarity)
- ☐ Split toning / color-grading wheels (shadows · midtones · highlights)
- ☐ Stronger LUT/profile management: validation previews, intensity mix, more
      LUT/profile formats *(from README "Next Recommended Upgrades")*

---

## P3 — Output & session workflow

- ☑ Export resize / long-edge constraint *(Qt export dialog, Phase 7b)*
- ☐ Watermarking on export
- ☐ Crop-on-export framing
- ☐ XMP / sidecar metadata write
- ☐ Full edit-history panel (today only mask edits have undo/redo)
- ☐ Snapshots / virtual copies
- ☐ Copy-paste settings between images
- ☐ CSV/HTML batch reports and richer retry filters beyond "latest failed batch"
      *(from README "Next Recommended Upgrades")*

### Collections storage & saved previews ◐
Collections should be treated as a manifest plus derived artifact folders. The
source of truth is always the original image reference, saved edit/settings
payload, source-file signature, render/color settings, and analysis/mask
signature. Thumbnails, analysis records, and rendered previews are derived
artifacts that can be regenerated when their key changes.

Current state:
- ☑ `collections/collections.json` stores collection membership, active
      collection, per-image overrides, culling/quality metadata, and now
      `rendered_previews` records.
- ☑ Derived artifacts live outside the JSON manifest:
      `collections/thumbnails/`, `collections/analysis_cache/`, and
      `collections/rendered_previews/`.
- ☑ Rendered previews are saved per collection image and reused across sessions
      when the render key still matches. The key includes source path/mtime/size,
      preview dimensions, render-cache version, edit params, layer order/options,
      color/runtime settings, mask adjustments, and segmentation/model analysis
      signature.
- ☑ Filmstrip selection shows decoded source pixels first, then restores the
      saved rendered preview or renders once when the key changed.

Next implementation steps:
- ☐ Add explicit collection schema/version fields and migration helpers for
      `collections.json` so new collection metadata can evolve without ad hoc
      compatibility code.
- ☐ Move from one global `collections.json` to per-collection folders:
      `collections/<collection_id>/manifest.json`, `thumbnails/`,
      `rendered_previews/`, `analysis_cache/`, and optional `exports/`.
- ☐ Use stable collection/image IDs internally instead of source paths as primary
      keys; keep source paths as mutable metadata so moved files can be repaired
      without losing edits/previews.
- ☐ Add a collection integrity pass: detect missing source files, stale preview
      artifacts, orphaned cache files, and records pointing at missing artifacts;
      offer repair/prune actions in the UI.
- ☐ Add background rendered-preview generation for newly imported images and for
      images whose saved settings changed, with cancellation when a newer edit key
      supersedes the queued render.
- ☐ Add cache-hit/miss telemetry in the status/perf label so it is obvious when a
      collection image reused a saved preview versus rendered fresh.
- ☐ Add tests for render-key stability, invalidation on edit/source/model changes,
      collection JSON round-trip, and orphan-artifact pruning.

Longer-term target:
- ☐ Migrate the collection manifest to SQLite when collections become large:
      `images`, `image_settings`, `rendered_previews`, `analysis_refs`,
      `exports/jobs`, and `schema_migrations` tables.
- ☐ Use atomic transactions for edit-state changes and preview-record updates so a
      crash cannot leave the manifest pointing at a half-written artifact.
- ☐ Support two storage modes: referenced originals (current behavior) and managed
      imports that copy originals into the collection package for portable
      archives.

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
