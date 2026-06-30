# SAM Face Mask Implementation Design

## Goal

Extend the Milestone 1 SAM instance-mask path from `Person` to the active
`Face` layer, while preserving the existing face-parsing and landmark stack for
semantic facial parts.

The target behavior is:

- `Face` gets a learned SAM boundary when SAM is available and plausible.
- `Skin`, `Eyes`, `Lips`, `Brows`, `Hair`, and `Facial hair` keep their semantic
  model/landmark behavior, but are constrained by the improved face boundary
  where that is safe.
- Any weak or incomplete SAM result falls back per face to the current model or
  heuristic mask, the same way `Person` falls back per person to watershed.
- The existing CPU-default SAM encoder, embedding memo, and
  `PORTRAIT_DISABLE_SAM` / `PORTRAIT_SAM_USE_COREML` switches remain the runtime
  foundation.

This is not a replacement for BiSeNet face parsing. SAM is promptable object
segmentation, not a face-part semantic parser. It can draw better boundaries,
but it does not know that hair, lips, eyes, brows, and skin are separate edit
layers. The safest design is to use SAM as a learned boundary/refinement layer
around the current semantic pipeline.

## Current Pipeline

Relevant code:

- `portrait_enhancer/core/segmentation.py`
- `ModelFaceSegmenter.segment_with_guides(...)`
- `FaceSegmenter.segment_with_guides(...)`
- `InstanceSegmenter`
- `_postprocess_model_masks(...)`
- `_region_zones(...)`
- `_person_mask(...)`
- `portrait_enhancer/core/analysis_cache.py`
- `portrait_enhancer/ui_qt/main_window.py`

Today, face-region masks are produced as follows:

1. `FaceSegmenter.list_faces()` detects face boxes through YuNet or Haar.
2. `ModelFaceSegmenter` runs the face-parsing ONNX model on the full image.
3. If a target `face_box` exists, it reruns parsing on an expanded crop and
   merges crop/full results through `_merge_model_views(...)`.
4. MediaPipe landmarks can add/repair `face`, `eyes`, and `lips`.
5. `_postprocess_model_masks(...)` cleans each semantic mask using geometric
   zones from `_region_zones(...)`, fills holes, keeps expected components, and
   uses `_zone_fallback(...)` when a feature would otherwise be blank.
6. `FaceSegmenter.segment_with_guides(...)` applies the active-face focus mask,
   derives `Subjects` / `Background`, and finally derives `Person`.

The weak point for `Face` is boundary precision and fallback geometry. The
current parser/landmark path is semantically useful, but the final outline still
leans on elliptical zones and edge refinement. SAM can improve that outline, but
must not be allowed to replace semantic facial-part masks blindly.

## Proposed Architecture

Add a SAM face-boundary refinement path inside `FaceSegmenter`, reusing the
existing `InstanceSegmenter` object.

High-level flow:

1. Run the existing model or heuristic segmentation exactly as today.
2. Before computing facial hair and before deriving scene subjects/person,
   request a SAM mask for the active face box.
3. Validate the SAM candidate against the existing face mask, face zone, face
   box, and optional landmarks.
4. If valid, use SAM as a boundary gate/refinement for `face` and lightly clamp
   related semantic masks.
5. If invalid or unavailable, keep the current masks unchanged.

The same `InstanceSegmenter.embed(...)` memo is reused. In multi-face export and
diagnostics, repeated `segment_with_guides(...)` calls on the same image reuse
the heavy ViT-B embedding and only rerun the lightweight decoder per face.

## InstanceSegmenter Changes

Generalize the SAM decoder so it can support both person point prompts and face
box/point prompts.

### New Internal Decoder API

Refactor:

```python
def _decode_one(self, emb, scale, shape_hw, pos_points, neg_points):
    ...
```

into:

```python
def _decode_prompt(
    self,
    emb,
    scale,
    shape_hw,
    points: list[tuple[float, float]],
    labels: list[int],
) -> np.ndarray | None:
    ...
```

`_decode_one(...)` remains as a compatibility wrapper for `Person`:

```python
def _decode_one(self, emb, scale, shape_hw, pos_points, neg_points):
    points = list(pos_points) + list(neg_points)
    labels = [1] * len(pos_points) + [0] * len(neg_points)
    return self._decode_prompt(emb, scale, shape_hw, points, labels)
```

The prompt API should continue to append the standard not-a-point marker:

```python
points.append((0.0, 0.0))
labels.append(-1)
```

For face boxes, use the standard SAM ONNX convention:

- label `2`: box top-left
- label `3`: box bottom-right
- label `1`: positive point
- label `0`: negative point
- label `-1`: padding marker

### New Face Prompt API

Add:

```python
def face_mask(
    self,
    img_float: np.ndarray,
    face_box: tuple[int, int, int, int],
    *,
    guides: dict | None = None,
    other_faces: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray | None:
    ...
```

Prompt construction:

- Box prompt: a conservative face box expansion, clamped to image bounds.
  Suggested first tuning:
  - x padding: `0.15 * fw`
  - top padding: `0.18 * fh`
  - bottom padding: `0.14 * fh`
- Positive points:
  - face-box center
  - cheek-left / cheek-right heuristics
  - chin heuristic
  - if `guides` are available, add mouth center and eye centers only as weak
    anchors inside the box
- Negative points:
  - just above the forehead band to discourage hair-heavy masks
  - below the chin to discourage neck/clothing expansion
  - left/right outside the face box
  - centers of other detected faces, when present

The decoder should return the best SAM candidate using the existing predicted-IoU
selection. It should not apply semantic gating itself. Validation and merging
belong in `FaceSegmenter`, where the current masks are available.

## FaceSegmenter Integration

Add face-mask backend state:

```python
self.face_backend = "model" | "heuristic" | "sam" | "model (sam-rejected)" | ...
self.face_backend_label = f"face={self.face_backend}"
```

Initialize it in `FaceSegmenter.__init__`, refresh it in `list_faces(...)`, and
surface it in the Qt status line next to `person=...`.

Add helper:

```python
def _refine_face_masks_with_sam(
    self,
    img_float: np.ndarray,
    faces: list[tuple[int, int, int, int]],
    face_index: int | None,
    masks: dict[str, np.ndarray],
    guides: dict | None,
) -> dict[str, np.ndarray]:
    ...
```

Call it in both model and heuristic paths:

1. After `masks, guides = self._model.segment_with_guides(...)`
2. Before `masks["facial_hair"] = self._facial_hair_mask(...)`
3. Before `_apply_focus_mask(...)`
4. Before `_scene_subject_masks(...)`
5. Before `_person_mask(...)`

This placement matters because:

- `Facial hair` uses `face`, `skin`, `hair`, and `brows` as support.
- `Subjects` fills holes using `face`, `skin`, and `hair`.
- `Person` may call the same SAM encoder afterward, so embedding memoization
  avoids a second encoder run.

## SAM Face Validation

Add a pure helper for unit testing:

```python
def _validate_sam_face_mask(
    sam_mask: np.ndarray | None,
    *,
    base_face: np.ndarray,
    hair_mask: np.ndarray | None,
    zones: dict[str, np.ndarray],
    face_box: tuple[int, int, int, int],
    guides: dict | None,
) -> tuple[np.ndarray | None, str]:
    ...
```

Return `(candidate, "normal")` when accepted. Return `(None, reason)` when
rejected. Suggested rejection reasons:

- `"sam-unavailable"`
- `"sam-empty"`
- `"sam-too-small"`
- `"sam-too-large"`
- `"sam-off-target"`
- `"sam-low-overlap"`
- `"sam-hair-heavy"`

Initial acceptance gates:

- Area:
  - `sam_area >= 0.35 * base_face_area` when `base_face_area > 0`
  - `sam_area <= 2.25 * face_box_area`
- Target coverage:
  - face-box center is inside SAM at `> 0.35`, or
  - at least two guide anchors are inside SAM when guides exist
- Overlap:
  - IoU with `zones["face"]` is at least `0.30`
  - IoU with `base_face` is at least `0.25` when `base_face` is non-empty
- Hair contamination:
  - if `hair_mask` exists, reject or clamp when more than roughly `45%` of the
    SAM area overlaps high-confidence hair

Keep these thresholds near the helper constants so real-image tuning is easy.

## Mask Merge Rules

When SAM is accepted, compute:

```python
sam_gate = _edge_aware_region_refine(sam_mask, img_float, "face")
sam_gate = smooth_mask(np.clip(sam_gate, 0.0, 1.0), sigma=max(0.8, w * 0.003))
```

Then merge conservatively:

```python
core_semantic = max(skin, eyes * 0.85, lips * 0.85, brows * 0.70)
face_refined = max(core_semantic, min(base_face, dilate(sam_gate)))
```

If the accepted SAM mask is larger than `base_face`, allow only limited growth:

```python
growth = np.clip(sam_gate * zones["face"] * (1.0 - hair * 0.75), 0.0, 1.0)
face_refined = max(face_refined, growth * 0.85)
```

This preserves semantic parts and prevents SAM from turning `Face` into
`Head + hair + neck`.

Clamp related masks:

- `skin`: `skin = min(skin, dilated(face_refined))`, but never synthesize new skin
  from SAM alone.
- `eyes`, `lips`, `brows`: keep existing model/landmark masks; only clamp to a
  slightly dilated `face_refined` so they cannot float outside the target face.
- `hair`: keep semantic hair, but reduce face bleed:
  `hair = hair * (1.0 - 0.25 * face_refined)` unless this causes a large hair
  area regression in validation.
- `facial_hair`: recompute after face refinement as described above.

The output should still pass through the existing final smoothing/clipping shape
contracts: every mask remains `float32`, full-image shape, range `[0, 1]`.

## Runtime Switches

Use existing switches:

- `PORTRAIT_DISABLE_SAM=1`: disable both Person and Face SAM paths.
- `PORTRAIT_SAM_USE_COREML=1`: opt into CoreML for the SAM encoder, still
  CPU-default otherwise.

Add one optional comparison switch:

- `PORTRAIT_DISABLE_SAM_FACE=1`: disable only SAM face refinement while leaving
  SAM Person masks enabled.

This is useful for real-image A/B checks without losing the Milestone 1 Person
work.

## Cache And Signature

Face masks change when this ships, so bump:

```python
CACHE_VERSION = 6
```

Update the cache-version comment in `analysis_cache.py`:

- v6: Face mask can now use SAM boundary refinement.

Extend `segmenter_cache_signature(...)` with face-specific SAM state, so a user
can toggle `PORTRAIT_DISABLE_SAM_FACE` without serving stale records:

```python
"sam_face_enabled": bool(...),
"sam_face_backend": "sam" if sam_face_enabled else "off",
"sam_face_disabled": bool(...),
```

Do not use the last per-image `face_backend_label` in the cache key. That label is
result state (`face=sam` vs `face=model*`), not configuration, and would make cache
records depend on image order. Also do not rely only on the existing
`instance_available` fields: those fields say SAM exists, but they do not
distinguish "SAM used for Person only" from "SAM used for Person and Face".

## UI And Documentation

Qt status line:

Current status includes:

```text
seg=... | detect=... | subjects=... | person=... | f_hair=... | faces=N | cache=...
```

Add:

```text
face=sam
face=model
face=model*
face=heuristic
```

Suggested meanings:

- `face=sam`: SAM refined the active Face layer.
- `face=model`: face parser/landmark path produced the Face layer.
- `face=model*`: SAM was available but rejected for this active face.
- `face=heuristic`: model unavailable and heuristic/SAM fallback path was used.

System Check:

- Either rename the current row to `SAM instance masks (Person/Face)`, or add a
  separate row `Face boundary refinement (SAM)`.
- If `PORTRAIT_DISABLE_SAM_FACE=1`, show `off` with detail
  `disabled via PORTRAIT_DISABLE_SAM_FACE`.
- If global `PORTRAIT_DISABLE_SAM=1`, both Person and Face SAM rows should show
  disabled.

Docs:

- `models/README.md`: extend the SAM section from "Optional Person Instance
  Masks" to "Optional SAM Instance Masks" and document both Person and Face use.
- `README.md`: update the backend-status paragraph to mention `face=...`.
- `scripts/download_sam_model.py`: no functional change required unless the
  final console text still says "per-person" only. Update wording to "person and
  face masks".

## Tests

Keep the suite fast by preserving `tests/conftest.py`:

```python
os.environ.setdefault("PORTRAIT_DISABLE_SAM", "1")
```

Add unit tests that do not require real SAM models:

1. `test_validate_sam_face_mask_accepts_plausible_candidate`
   - Synthetic face box, base face ellipse, SAM-like mask with strong overlap.
   - Expect accepted.

2. `test_validate_sam_face_mask_rejects_incomplete_candidate`
   - SAM mask covers only a small eye/forehead area.
   - Expect rejected as `"sam-too-small"`.

3. `test_validate_sam_face_mask_rejects_off_target_candidate`
   - SAM mask is plausible size but shifted away from the face center.
   - Expect rejected as `"sam-off-target"` or `"sam-low-overlap"`.

4. `test_sam_face_refinement_falls_back_when_disabled`
   - Set `PORTRAIT_DISABLE_SAM_FACE=1`.
   - Use a fake available `InstanceSegmenter`.
   - Verify `masks["face"]` remains unchanged.

5. `test_sam_face_refinement_preserves_semantic_parts`
   - Fake SAM mask tightens the face boundary.
   - Verify `eyes`, `lips`, and `brows` are not replaced by SAM, only clamped if
     outside the refined face.

6. `test_face_backend_label_reports_sam_and_rejection`
   - Fake accepted candidate gives `face=sam`.
   - Fake rejected candidate gives `face=model*` or equivalent.

7. Cache tests:
   - Update expected `CACHE_VERSION`.
   - Update signature-change tests to include `sam_face_enabled` /
     `sam_face_backend`.

Existing `Person` tests should continue to assert that disabling SAM falls back
to watershed/subjects unchanged.

## Real-Image Verification

Use the same verification set as Milestone 1:

- `_MG_3976`
- `_MG_3977`

Run these checks with SAM enabled:

1. Open each image and analyze every detected face target.
2. For every face:
   - `Face` mask is complete, not truncated.
   - `Face` excludes table/background/clothing.
   - `Face` does not absorb high-confidence hair.
   - `Skin`, `Eyes`, `Lips`, `Brows`, and `Hair` still appear in the right place.
   - `Person` still uses SAM or its existing per-person fallback behavior.
3. Flip `PORTRAIT_DISABLE_SAM_FACE=1` and confirm only the Face boundary changes,
   not the Person instance mask path.
4. Confirm the status label reports the active path:
   - `face=sam` where accepted
   - `face=model*` where rejected
   - `person=sam` / `person=watershed*` as before
5. Confirm cache behavior:
   - first run stores v6 records
   - second run hits cache
   - toggling `PORTRAIT_DISABLE_SAM_FACE` misses cache and recomputes

Definition of done:

- `_MG_3976` and `_MG_3977` have complete Face masks for all detected faces.
- No accepted SAM Face mask is visibly worse than the current parser/landmark
  mask.
- In any failure case, the current parser/landmark or heuristic mask reaches the
  user unchanged.
- Full test suite is green except any explicitly pre-existing unrelated failures.
- No generated diagnostics, temporary masks, or local cache artifacts are left in
  the commit.

## Implementation Order

1. Generalize `InstanceSegmenter` decoder prompts.
2. Add `InstanceSegmenter.face_mask(...)`.
3. Add `_validate_sam_face_mask(...)` and focused synthetic tests.
4. Add `_refine_face_masks_with_sam(...)` in `FaceSegmenter`.
5. Wire model and heuristic segmentation paths.
6. Add `face_backend` / `face_backend_label` and status output.
7. Add `PORTRAIT_DISABLE_SAM_FACE`.
8. Bump `CACHE_VERSION` to 6 and update cache signatures/tests.
9. Update `models/README.md`, `README.md`, System Check, and download-script
   wording.
10. Run smoke tests, then real-image verification on `_MG_3976` and `_MG_3977`.

## Risks And Mitigations

SAM returns head/hair instead of face.

- Mitigation: keep BiSeNet/landmark semantics, reject hair-heavy candidates, and
  allow only limited SAM-driven growth.

SAM returns a partial face.

- Mitigation: area and anchor coverage checks; per-face fallback to existing
  masks.

SAM improves the boundary but harms eyes/lips/skin.

- Mitigation: do not use SAM to synthesize facial parts. It only constrains
  existing semantic masks.

Performance regresses in multi-face images.

- Mitigation: reuse `InstanceSegmenter.embed(...)` memo across face targets and
  Person. Only the decoder runs per active face.

Cache serves stale masks after toggling face SAM.

- Mitigation: bump `CACHE_VERSION` and add face-SAM state to
  `segmenter_cache_signature(...)`.
