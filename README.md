# Portrait Enhancer

A desktop RAW photo editor with real-time preview and selective portrait layers.

## Features
- CR2 / NEF / ARW / DNG RAW file support (via `rawpy` / LibRaw)
- JPEG / PNG / TIFF input support
- 16-bit RAW decode path (converted to float for non-destructive processing)
- Live preview updates while dragging sliders (debounced + queued render worker)
- Preview/full-resolution split: fast proxy preview, full-resolution export render
- Model-aware selective masks with automatic fallback
- Selective adjustment layers: Face, Skin, Eyes, Lips, Hair
- Layer stack controls: per-layer enable, opacity, blend mode, and ordering
- Interactive mask tools: paint, erase, feather, and reset-to-auto
- Non-destructive per-mask settings for strength, feather, and expand/contract
- Multi-face target picker (select which detected face drives selective masks)
- Per-face selective profiles (switching faces preserves that face's local adjustments and masks)
- Mask history controls: undo/redo for manual mask edits
- Project save/load (`.peproj`) for restoring image path, global settings, per-face profiles, and masks
- Portable preset save/load (`.pepreset`) for global/selective adjustment recipes
- Mask overlay preview per active layer
- Color-management controls for working space, output tone, and ICC embedding
- Optional CUDA acceleration for preview/render blur and resize operations with CPU fallback
- Compare modes: split, full before view, and side-by-side comparison
- Local preset browser with search, categories, and saved metadata backed by the `presets/` folder
- Metadata-aware export for non-RAW sources (preserves EXIF/ICC/DPI when supported by Pillow)
- Batch folder export using a `.pepreset`
- Full-resolution export at JPEG / PNG / TIFF
- Parallel PySide6 frontend for the migration off Tkinter

## Project Structure
- `pyproject.toml`: package metadata and GUI entrypoint
- `portrait_enhancer_v2.py`: compatibility entrypoint script
- `portrait_enhancer_qt.py`: PySide6 compatibility entrypoint script
- `portrait_enhancer/ui/app.py`: Tkinter application UI
- `portrait_enhancer/ui_qt/main_window.py`: PySide6 application UI
- `portrait_enhancer/core/processing.py`: image adjustment pipeline
- `portrait_enhancer/core/segmentation.py`: face/feature mask generation
- `portrait_enhancer/config.py`: constants, layer metadata, slider definitions

## Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv and install the app with runtime dependencies
uv sync

# Include the Qt frontend
uv sync --extra qt

# Include optional model backends too
uv sync --extra qt --extra model

# Or install everything, including test dependencies
uv sync --all-extras

# Run app (any option)
uv run portrait-enhancer
uv run --extra qt portrait-enhancer-qt
uv run python -m portrait_enhancer
uv run python portrait_enhancer_v2.py
uv run --extra qt python portrait_enhancer_qt.py

# Run tests
uv run --extra dev pytest
```

## Dependency Sets
1. `pyproject.toml`: canonical dependency metadata for `uv sync` and package installs.
2. Default dependencies: base runtime dependencies.
3. `qt` extra: PySide6 frontend dependency.
4. `model` extra: ONNX + MediaPipe model backend.
5. `dev` extra: local development and test dependencies.
6. `requirements.txt`, `requirements/base.txt`, `requirements/optional-model.txt`, `requirements/optional-qt.txt`, `requirements/dev.txt`, `requirements_v2.txt`, and `requirements_phase3_optional.txt` remain as compatibility wrappers for older pip-based instructions.

## Packaging
1. The project now ships package metadata in `pyproject.toml`.
2. GUI entrypoint: `portrait-enhancer`.
3. Optional extras:
   - `uv sync --extra model`
   - `uv sync --extra qt`
   - `uv sync --extra dev`
   - `uv sync --all-extras`

## Qt Frontend
1. The PySide6 frontend is available in parallel with the existing Tk app.
2. Current Qt scope:
    - open image
    - open/save `.pepreset` files
    - local preset browser backed by `./presets`
    - detached background batch export from a preset across a folder
    - `Batch Jobs` viewer for submitted background jobs
    - resumable `batch_export_log.jsonl` with skip-completed and retry-failed
    - face selection
    - live layer sliders for all layers
    - compare modes: `off`, `before`, `split`, `side_by_side`
    - live RGB + luminance histogram with shadow/highlight clipping markers
    - interactive tone curve (monotone-cubic, drag/add/remove points) with a histogram backdrop
    - HSL color mixer: per-hue-band hue/saturation/luminance (8 bands)
    - white balance: Kelvin Temperature/Tint sliders, illuminant presets (Tungsten/Daylight/Cloudy/Shade/…), neutral-pick eyedropper with magnified loupe, and gray-world/white-patch auto (per-channel gains in linear light)
    - frame geometry: crop (aspect presets + draggable crop box), straighten, flip — non-destructive, saved in the project
    - active-layer mask overlay
    - basic mask editing: paint, erase, brush size, reset-to-auto
    - per-mask strength, feather, and expand/contract controls for subject, background, skin, eyes, lips, and hair masks
    - project save/load (`.peproj`)
    - document undo/redo for slider, compare, face-target, and layer-reset changes
    - startup readiness check and manual `System Check` dialog for model/runtime status
    - guided built-in portrait recipes from the `Recipes` action
    - saved batch profiles inside the batch dialog for reusing preset/folder/output setups
    - preview rendering through the shared processing pipeline
    - export current result
3. The Tk app still has the deeper feature set today for some advanced mask-edit workflows.

## Project Files
1. Use `Save Project` to write a `.peproj` file.
2. A project stores:
   - source image path
   - global slider values
   - detected face list and active face selection
   - per-face selective sliders, layer settings, mask edits
3. Use `Open Project` to restore the editing session for the same source image.

## Preset Files
1. Use `Save Preset` to write a portable `.pepreset` file.
2. A preset stores:
   - global slider values
   - selective layer slider values
   - layer enable/opacity/blend settings
   - selective layer order
3. Presets do not store masks, face detections, or image paths.
4. Use `Open Preset` to apply the preset to the current editing target.
5. Use the `Preset Browser` panel to apply or save library presets in the local `presets/` directory.
6. The Qt preset browser writes sidecar thumbnail previews (`*.thumb.jpg`) for presets saved from the app and shows them in the browser preview pane.
6. Library presets can include `meta.name`, `meta.category`, `meta.tags`, and `meta.saved_at`.
7. Use the browser search field and category filter to narrow the list.

## Compare Modes
1. Use the `Compare` control in the top bar to switch between `off`, `split`, `before`, and `side_by_side`.
2. `split` shows source on the left and edited output on the right.
3. Drag on the canvas to move the split divider when `split` is active.
4. `before` shows the full unedited source preview.
5. `side_by_side` shows full-frame before/after panes together on the canvas.

## Color Management
1. `Input ICC`: `auto` converts embedded source ICC data to sRGB for non-RAW images; `ignore` treats pixels as already sRGB.
2. `RAW WB`: choose `camera` or `auto` for RAW decode white balance.
3. `RAW Color`: choose `srgb`, `adobe`, `prophoto`, `xyz`, or `raw` for RAW output color space.
4. `RAW LUT`: load a `.cube` 3D LUT and apply it after RAW decode.
5. `Working`: choose `srgb` or `linear` for the processing pipeline.
6. `Output Tone`: choose `srgb`, `gamma22`, `gamma18`, or `linear` for preview/export encoding.
7. `ICC Embed`: choose `srgb`, `preserve_source`, or `none`.
8. `preserve_source` embeds the original source ICC only when pixels were not converted away from that source profile.
9. Projects and presets both store color-management settings, including RAW decode and LUT settings.
10. RAW LUT support currently expects `.cube` 3D LUT files.

## GPU Acceleration
1. Use the `Performance` panel to choose `auto`, `cpu`, or `cuda`.
2. `auto` uses CUDA when OpenCV exposes a CUDA device; otherwise it falls back to CPU.
3. Acceleration currently targets preview/render resize and Gaussian-blur-heavy steps.
4. Segmentation/inference still uses the current CPU/runtime path unless the underlying model runtime handles acceleration separately.

## Batch Export
1. Use `Batch Export` and choose:
   - a `.pepreset` file
   - an input folder
   - an output folder
2. Set:
   - output suffix such as `_enhanced` or `_social`
   - output format: `jpeg`, `png`, or `tiff`
   - whether to skip files already completed in prior runs
3. The app writes a resumable log at `batch_export_log.jsonl` in the output folder.
4. Resume/skip behavior is keyed by preset content hash plus suffix/format, so editing a preset file changes the batch identity even if the filename stays the same.
5. Use `Retry Failed` to rerun only failed files from the latest failed batch in a selected log, using the matching preset.
6. In the PySide6 app, batch jobs are launched as a detached background process, so you can close the app after the job starts.
7. Use `Batch Jobs` in the Qt toolbar to inspect submitted jobs for an output folder, including status, counts, job file, batch log, and runner log.
8. The `Batch Jobs` dialog auto-refreshes every 2 seconds by default and can also be refreshed manually.
9. The Qt app writes the submitted job JSON into `.batch_jobs/` inside the output folder and captures worker stdout/stderr in `batch_runner_stdout.log`.
10. You can also run the worker directly with `portrait-enhancer-batch --job /abs/path/to/job.json` or `python -m portrait_enhancer.batch_runner --job /abs/path/to/job.json`.
11. Global preset settings are applied to the whole image.
12. Selective preset settings are applied to the union of all detected face-feature masks in each image.
13. Non-RAW source metadata is preserved on batch export when Pillow exposes it.

### Optional model backend (Phase 3)
```bash
# Optional runtime deps for model-based masks
uv sync --extra model
```

Provide a face parsing ONNX model using one of:
1. Place model at `models/face_parsing.onnx`
2. Or set env var: `PORTRAIT_FACE_PARSING_ONNX=/abs/path/to/face_parsing.onnx`

Optional stronger face detector:
1. Place YuNet detector model at `models/face_detection_yunet.onnx`
2. Or set env var: `PORTRAIT_FACE_DETECTOR_ONNX=/abs/path/to/face_detection_yunet.onnx`

Optional face refinement:
1. Place a CodeFormer-compatible model at `models/codeformer.onnx`
2. Or set env var: `PORTRAIT_CODEFORMER_ONNX=/abs/path/to/codeformer.onnx`
3. Use the `Face -> AI Refine` slider to blend the restored face patch into the edit
4. Use `Face -> AI Fidelity` to control how closely CodeFormer stays to the original face patch
5. In the Qt app, the perf line shows `refine=off`, `refine=codeformer-onnx:coreml`, `refine=codeformer-onnx:cpu`, or `refine=unavailable`

Optional dedicated facial-hair exclusion:
1. Place a facial-hair segmentation model at `models/facial_hair.onnx`
2. Or set env var: `PORTRAIT_FACIAL_HAIR_ONNX=/abs/path/to/facial_hair.onnx`
3. The app status line will show `f_hair=onnx:coreml`, `f_hair=onnx:cpu`, or `f_hair=fallback`

### macOS / Linux note
`rawpy` requires LibRaw.
- macOS: `brew install libraw`
- Ubuntu: `apt install libraw-dev`

## Masking Backends
The app uses a backend wrapper:
1. `model (onnx + landmarks)` when ONNX parsing and landmark refinement are both available.
2. `model (onnx only)` when ONNX parsing is available without landmark refinement.
3. `heuristic (fallback)` when model backend is unavailable or fails at runtime.

Face target detection uses:
1. OpenCV YuNet when a detector ONNX model is available.
2. Haar cascade fallback when the YuNet model is unavailable.

Backend status is shown in the app status bar during analysis.

## Rendering Model
1. The source image is kept at full resolution in memory.
2. A downscaled proxy image is created for interactive preview.
3. Segmentation runs on the proxy, then masks are upscaled for export.
4. Slider changes enqueue render jobs; stale preview jobs are dropped.

## Layer Stack and Mask Editing
1. Select a layer tab (Face/Skin/Eyes/Lips/Hair).
2. Tune layer mix controls: `Enabled`, `Opacity`, `Blend`.
3. Reorder selective layers with `Up` / `Down` (top of list applies first).
4. Use `Edit Mask` to paint/erase on the active layer mask.
5. Use `Feather` to soften edges, `Reset Mask` to restore auto-segmentation.
6. Use `Undo` / `Redo` to step through manual mask edits.

## Multi-Face Targeting
1. When multiple faces are detected, use the `Face Target` picker in the left panel.
2. The selected face index is used for selective masks and layer effects.
3. Each face keeps its own selective sliders, layer mix settings, and edited masks.
4. Status bar shows `faces=<n>, target=<index>` after analysis.

## Roadmap
See [ROADMAP.md](ROADMAP.md) for the full prioritized roadmap, including the larger
editing gaps (crop/straighten, interactive tone curve, HSL mixer, manual
retouching, and local-adjustment masks). Near-term refinements:
1. Add stronger LUT/profile management: validation previews, intensity mix, and support for more LUT/profile formats.
2. Add brush hardness/flow and pressure-sensitive tablet support for mask editing.
3. Add optional GPU acceleration for preview rendering and mask inference.
4. Add CSV/HTML batch reports and retry filters beyond “latest failed batch”.
