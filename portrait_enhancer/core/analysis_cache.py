"""Persistent cache for image-derived detection and segmentation data."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Bump whenever the set of mask keys OR the mask-generation logic itself changes -- ties
# cache validity to actual output, not just which models are loaded (segmenter_cache_signature
# has no way to know the *code* changed what masks it returns). v2: added "person". v3: fixed
# landmark misattribution across faces (a face the landmarker missed was silently reusing
# another face's landmarks) and added a zone-geometry fallback so a feature mask is never left
# completely blank when model confidence collapses for one face. v4: person-split watershed now
# seeds from a body capsule instead of a face-center dot, so per-person masks cover whole bodies
# instead of just heads. v5: Person mask can now come from learned SAM instance masks
# (InstanceSegmenter) instead of the watershed split -- different mask values. v6: Face masks can
# now use SAM boundary refinement, changing face/skin/eyes/lips/brows/hair values. v7: SAM
# Person instance masks are now cross-person/other-head suppressed and anchor-component filtered.
# v8: SAM Person masks are constrained to the selected watershed basin so a connected or
# low-confidence SAM region cannot keep another detected person's head/hair halo. v9: even
# rejected watershed labels can gate SAM, avoiding stale cross-person halos in cluttered scenes.
# v10: Person masks can come from Mask DINO person instance segmentation before SAM/watershed.
# v11: Face-part masks are identity-gated to the selected Person mask so Skin/Eyes/Lips/Hair
# cannot keep islands from a different detected person. v12: Skin/Eyes/Lips/Hair use face-width
# smoothing plus landmark/anatomy-aware part refinement. v13: landmarks retry on enlarged
# per-face crops, and Hair is suppressed more strongly over refined face/skin. v14: Subjects/
# Background can use RMBG-2.0/BiRefNet foreground matting with optional Mask DINO person gating.
CACHE_VERSION = 14
CACHE_KIND_SINGLE_FACE = "single_face"
CACHE_KIND_ALL_FACES = "all_faces"


def image_signature(path: str) -> dict[str, Any]:
    """Stable-enough file signature for invalidating derived data when a source changes."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def segmenter_cache_signature(segmenter: Any) -> dict[str, Any]:
    """Capture the model/detector setup that affects generated masks."""
    model = getattr(segmenter, "_model", None)
    background_removal = getattr(segmenter, "_background_removal", None)
    matting = getattr(segmenter, "_matting", None)
    subjects = getattr(segmenter, "_subjects", None)
    facial_hair = getattr(segmenter, "_facial_hair", None)
    person_instances = getattr(segmenter, "_person_instances", None)
    instance = getattr(segmenter, "_instance", None)
    sam_face_disabled = _env_truthy("PORTRAIT_DISABLE_SAM_FACE")
    sam_face_enabled = bool(getattr(instance, "available", False)) and not sam_face_disabled
    return {
        "backend": str(getattr(segmenter, "backend", "")),
        "backend_label": str(getattr(segmenter, "backend_label", "")),
        "detector": str(getattr(segmenter, "detector_backend", "")),
        "model_available": bool(getattr(model, "available", False)),
        "model_provider": str(getattr(model, "execution_provider", "")),
        "landmarks": str(getattr(model, "landmark_backend", "")),
        "background_removal_available": bool(getattr(background_removal, "available", False)),
        "background_removal_backend": str(getattr(background_removal, "backend", "")),
        "background_removal_device": str(getattr(background_removal, "execution_provider", "")),
        "background_removal_disabled": _env_truthy("PORTRAIT_DISABLE_RMBG"),
        "matting_available": bool(getattr(matting, "available", False)),
        "matting_backend": str(getattr(matting, "backend", "")),
        "subjects_available": bool(getattr(subjects, "available", False)),
        "subjects_backend": str(getattr(subjects, "backend", "")),
        "facial_hair_available": bool(getattr(facial_hair, "available", False)),
        "facial_hair_backend": str(getattr(facial_hair, "backend", "")),
        # Accuracy-first per-person instance backend. Include the effective install state so a
        # machine with Mask DINO never serves SAM/watershed-era Person masks from cache.
        "person_instance_available": bool(getattr(person_instances, "available", False)),
        "person_instance_backend": str(getattr(person_instances, "backend", "")),
        "person_instance_device": str(getattr(person_instances, "execution_provider", "")),
        "person_instance_disabled": _env_truthy("PORTRAIT_DISABLE_MASKDINO"),
        # SAM instance-mask backend -- a machine with SAM installed must not serve another's
        # watershed-era cached Person masks (and vice versa).
        "instance_available": bool(getattr(instance, "available", False)),
        "instance_backend": str(getattr(instance, "backend", "")),
        # Face SAM is independently switchable for A/B comparison; include the effective mode,
        # not the last per-image status label, so cache keys stay stable across images.
        "sam_face_enabled": sam_face_enabled,
        "sam_face_backend": "sam" if sam_face_enabled else "off",
        "sam_face_disabled": sam_face_disabled,
    }


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip() not in ("", "0", "false", "False", "no", "No")


def load_analysis(
    path: str,
    *,
    kind: str,
    image_shape: tuple[int, int],
    face_index: int | None = None,
    backend_signature: dict[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load a cached analysis record, returning None on misses or corrupt entries."""
    try:
        signature = image_signature(path)
    except OSError:
        return None
    shape = [int(image_shape[0]), int(image_shape[1])]
    record_path = analysis_cache_record_path(
        signature,
        kind=kind,
        image_shape=shape,
        face_index=face_index,
        backend_signature=backend_signature,
        cache_root=cache_root,
    )
    if not record_path.is_file():
        return None

    try:
        payload = _read_cache_file(record_path)
    except Exception:
        return None

    record = payload.get("record", {})
    if record.get("kind") != kind:
        return None
    if record.get("image_shape") != shape:
        return None
    if record.get("face_index") != face_index:
        return None
    if backend_signature is not None and record.get("backend_signature") != backend_signature:
        return None

    masks = {}
    for key, array_name in record.get("mask_arrays", {}).items():
        array = payload["arrays"].get(array_name)
        if array is None:
            return None
        masks[str(key)] = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0).copy()
    return {
        "faces": [tuple(int(v) for v in face) for face in record.get("faces", [])],
        "masks": masks,
        "guides": record.get("guides"),
        "backend_signature": record.get("backend_signature", {}),
        "created_at": record.get("created_at", ""),
    }


def load_latest_compatible_analysis(
    path: str,
    *,
    target_shape: tuple[int, int],
    preferred_kinds: tuple[str, ...] = (CACHE_KIND_ALL_FACES, CACHE_KIND_SINGLE_FACE),
    face_index: int | None = None,
    required_mask_keys: tuple[str, ...] | set[str] | list[str] | None = None,
    allow_resize: bool = True,
    cache_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load the newest compatible analysis for an image, resizing masks to target_shape.

    This intentionally ignores backend_signature and exact image_shape. It is a fallback for
    export: a preview-time cache hit is usually better than reloading every heavy model just to
    recreate masks that already exist. Exact load_analysis() remains the strict path. Callers
    can restrict single-face records by face_index and mask keys so a loose export cache hit
    never applies another face's masks or loads layers it will not render.
    """
    try:
        signature = image_signature(path)
    except OSError:
        return None
    cache_dir = analysis_cache_path(signature, cache_root=cache_root)
    if not cache_dir.is_dir():
        return None

    try:
        shape = (int(target_shape[0]), int(target_shape[1]))
    except (TypeError, ValueError, IndexError):
        return None
    required_keys = tuple(str(key) for key in (required_mask_keys or ()) if str(key))
    preferred = {kind: index for index, kind in enumerate(preferred_kinds)}
    candidates: list[tuple[int, int, float, Path, set[str] | None]] = []
    for record_path in cache_dir.glob("*.npz"):
        try:
            payload = _read_cache_file(record_path, include_arrays=False)
        except Exception:
            continue
        record = payload.get("record", {})
        kind = record.get("kind")
        if kind not in preferred:
            continue
        if kind == CACHE_KIND_SINGLE_FACE and face_index is not None:
            try:
                record_face_index = int(record.get("face_index"))
            except (TypeError, ValueError):
                continue
            if record_face_index != int(face_index):
                continue
        source_shape = record.get("image_shape")
        if not isinstance(source_shape, list) or len(source_shape) != 2:
            continue
        try:
            source_hw = (int(source_shape[0]), int(source_shape[1]))
        except (TypeError, ValueError):
            continue
        if not allow_resize and source_hw != shape:
            continue
        mask_arrays = record.get("mask_arrays")
        if not isinstance(mask_arrays, dict) or not mask_arrays:
            continue
        if required_keys:
            array_names = _array_names_for_mask_keys(mask_arrays, required_keys)
            if array_names is None:
                continue
        else:
            array_names = None
        shape_rank = 0 if source_hw == shape else 1
        candidates.append((preferred[kind], shape_rank, -record_path.stat().st_mtime, record_path, array_names))
    if not candidates:
        return None

    for _kind_rank, _shape_rank, _mtime, record_path, array_names in sorted(candidates):
        try:
            payload = _read_cache_file(record_path, array_names=array_names)
            result = _analysis_payload_to_result(
                payload,
                target_shape=target_shape,
                required_mask_keys=required_keys,
            )
        except Exception:
            result = None
        if result is not None:
            result["cache_path"] = str(record_path)
            return result
    return None


def save_analysis(
    path: str,
    *,
    kind: str,
    image_shape: tuple[int, int],
    masks: dict[str, np.ndarray] | None,
    faces: list[tuple[int, int, int, int]] | None = None,
    guides: Any = None,
    face_index: int | None = None,
    backend_signature: dict[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> Path | None:
    """Persist masks, guides, and face boxes for a source image."""
    if not masks:
        return None
    try:
        signature = image_signature(path)
    except OSError:
        return None

    shape = [int(image_shape[0]), int(image_shape[1])]
    record_path = analysis_cache_record_path(
        signature,
        kind=kind,
        image_shape=shape,
        face_index=face_index,
        backend_signature=backend_signature,
        cache_root=cache_root,
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)

    mask_arrays = {}
    arrays = {}
    for key, mask in masks.items():
        array_name = f"mask_{key}"
        mask_arrays[str(key)] = array_name
        arrays[array_name] = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)

    record = {
        "record_key": _record_key(kind, shape, face_index, backend_signature),
        "kind": str(kind),
        "image_shape": shape,
        "face_index": face_index,
        "backend_signature": dict(backend_signature or {}),
        "faces": [list(map(int, face)) for face in (faces or [])],
        "guides": _jsonable(guides),
        "mask_arrays": mask_arrays,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata = {
        "version": CACHE_VERSION,
        "source": signature,
        "record": record,
    }
    _write_cache_file(record_path, metadata, arrays)
    return record_path


def _analysis_payload_to_result(
    payload: dict[str, Any],
    *,
    target_shape: tuple[int, int],
    required_mask_keys: tuple[str, ...] | set[str] | list[str] | None = None,
) -> dict[str, Any] | None:
    record = payload.get("record", {})
    source_shape = record.get("image_shape")
    if not isinstance(source_shape, list) or len(source_shape) != 2:
        return None
    try:
        src_h, src_w = int(source_shape[0]), int(source_shape[1])
        dst_h, dst_w = int(target_shape[0]), int(target_shape[1])
    except (TypeError, ValueError):
        return None
    if min(src_h, src_w, dst_h, dst_w) <= 0:
        return None

    mask_arrays = record.get("mask_arrays", {})
    if not isinstance(mask_arrays, dict) or not mask_arrays:
        return None
    required_keys = tuple(str(key) for key in (required_mask_keys or ()) if str(key))
    if required_keys:
        selected_items = [(key, mask_arrays.get(key)) for key in required_keys]
        if any(array_name is None for _key, array_name in selected_items):
            return None
    else:
        selected_items = list(mask_arrays.items())

    masks = {}
    for key, array_name in selected_items:
        array = payload["arrays"].get(array_name)
        if array is None:
            return None
        mask = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
        if mask.shape != (dst_h, dst_w):
            mask = _resize_mask(mask, (dst_h, dst_w))
        masks[str(key)] = mask
    if not masks:
        return None

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return {
        "faces": [_scale_face(face, sx, sy) for face in record.get("faces", [])],
        "masks": masks,
        "guides": _scale_guides(record.get("guides"), sx, sy),
        "backend_signature": record.get("backend_signature", {}),
        "created_at": record.get("created_at", ""),
        "kind": record.get("kind", ""),
        "face_index": record.get("face_index"),
        "source_shape": (src_h, src_w),
        "resized": (src_h, src_w) != (dst_h, dst_w),
    }


def _array_names_for_mask_keys(mask_arrays: dict[str, Any], required_keys: tuple[str, ...]) -> set[str] | None:
    array_names = set()
    for key in required_keys:
        array_name = mask_arrays.get(str(key))
        if not array_name:
            return None
        array_names.add(str(array_name))
    return array_names


def _resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = shape_hw
    return np.clip(cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0).astype(np.float32)


def _scale_face(face: Any, sx: float, sy: float) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = face
        return (
            int(round(float(x) * sx)),
            int(round(float(y) * sy)),
            int(round(float(w) * sx)),
            int(round(float(h) * sy)),
        )
    except Exception:
        return (0, 0, 0, 0)


def _scale_guides(guides: Any, sx: float, sy: float) -> Any:
    if guides is None:
        return None
    if isinstance(guides, dict):
        scaled = {}
        for key, value in guides.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    scaled[key] = [float(value[0]) * sx, float(value[1]) * sy]
                    continue
                except Exception:
                    pass
            scaled[key] = _scale_guides(value, sx, sy)
        return scaled
    if isinstance(guides, list):
        return [_scale_guides(item, sx, sy) for item in guides]
    return guides


def analysis_cache_path(signature: dict[str, Any], *, cache_root: str | Path | None = None) -> Path:
    """Directory that contains all cache records for one source image signature."""
    root = Path(cache_root) if cache_root is not None else Path(os.getcwd()) / "collections" / "analysis_cache"
    key = _source_key(signature)
    return root / key[:2] / key


def analysis_cache_record_path(
    signature: dict[str, Any],
    *,
    kind: str,
    image_shape: list[int],
    face_index: int | None,
    backend_signature: dict[str, Any] | None,
    cache_root: str | Path | None = None,
) -> Path:
    """Lock-free record path for one source/backend/shape/face analysis."""
    record_key = _record_key(kind, image_shape, face_index, backend_signature)
    return analysis_cache_path(signature, cache_root=cache_root) / f"{record_key}.npz"


def _source_key(signature: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _record_key(
    kind: str,
    image_shape: list[int],
    face_index: int | None,
    backend_signature: dict[str, Any] | None,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "kind": str(kind),
        "image_shape": image_shape,
        "face_index": face_index,
        "backend_signature": backend_signature or {},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_cache_file(
    path: Path,
    *,
    include_arrays: bool = True,
    array_names: set[str] | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata"].item()
        if isinstance(metadata_raw, bytes):
            metadata_raw = metadata_raw.decode("utf-8")
        metadata = json.loads(str(metadata_raw))
        if int(metadata.get("version", 0)) != CACHE_VERSION:
            return {"record": {}, "arrays": {}}
        arrays = {}
        if include_arrays:
            selected = {str(name) for name in array_names} if array_names is not None else None
            arrays = {
                key: data[key].copy()
                for key in data.files
                if key != "metadata" and (selected is None or key in selected)
            }
    return {"record": metadata.get("record", {}), "arrays": arrays}


def _write_cache_file(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, metadata=np.array(json.dumps(metadata, sort_keys=True)), **arrays)
        if tmp_path.suffix != ".npz":
            generated = tmp_path.with_suffix(tmp_path.suffix + ".npz")
            if generated.exists():
                generated.replace(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
