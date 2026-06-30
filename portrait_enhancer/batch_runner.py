"""Detached batch runner for Portrait Enhancer."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from portrait_enhancer.config import MASK_ORDER
from portrait_enhancer.core.framing import apply_framing
from portrait_enhancer.core.denoise import DeepDenoiser
from portrait_enhancer.core.masks import apply_mask_adjustments
from portrait_enhancer.core.processing import process_all_layers, apply_output_sharpening
from portrait_enhancer.core.raw_decode import RAW_EXTS, decode_raw, is_raw_path
from portrait_enhancer.core.segmentation import FaceSegmenter
from portrait_enhancer.core.utils import blend_with_mask, to_float, to_uint8
from portrait_enhancer.core.analysis_cache import (
    CACHE_KIND_ALL_FACES,
    CACHE_KIND_SINGLE_FACE,
    load_analysis,
    load_latest_compatible_analysis,
    save_analysis,
    segmenter_cache_signature,
)

SUPPORTED_IMAGE_EXTS = tuple(RAW_EXTS) + (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
BATCH_OUTPUT_FORMATS = {
    "jpeg": ".jpg",
    "png": ".png",
    "tiff": ".tiff",
}


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def _append_batch_log(log_path, record):
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _cancel_marker_path(job: dict) -> str:
    cancel_path = str(job.get("cancel_path", "") or "")
    if cancel_path:
        return cancel_path
    job_path = str(job.get("_job_path", "") or "")
    if job_path:
        return f"{job_path}.cancel"
    output_dir = str(job.get("output_dir", "") or "")
    job_id = str(job.get("job_id", "") or "")
    if output_dir and job_id:
        return os.path.join(output_dir, ".batch_jobs", f"{job_id}.cancel")
    return ""


def _cancel_requested(job_or_task: dict) -> bool:
    cancel_path = str(job_or_task.get("cancel_path", "") or "")
    if cancel_path and os.path.exists(cancel_path):
        return True
    job_path = str(job_or_task.get("_job_path", "") or job_or_task.get("job_path", "") or "")
    if job_path:
        try:
            with open(job_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return bool(payload.get("cancel_requested", False))
        except Exception:
            return False
    return bool(job_or_task.get("cancel_requested", False))


def _append_cancel_record(log_path, job_id, job_mode, *, status="canceled", reason="user_canceled"):
    _append_batch_log(
        log_path,
        {
            "ts": _utc_timestamp(),
            "job_id": job_id,
            "job_mode": job_mode,
            "status": status,
            "reason": reason,
        },
    )


def _load_batch_log_records(log_path):
    records = []
    if not os.path.exists(log_path):
        return records
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def _load_batch_log_index(log_path):
    index = {}
    for record in _load_batch_log_records(log_path):
        if record.get("status") != "success":
            continue
        key = (
            record.get("batch_key"),
            os.path.abspath(record.get("source_path", "")),
            os.path.abspath(record.get("output_path", "")),
        )
        index[key] = record
    return index


def _batch_should_skip(completed_index, batch_key, source_path, output_path):
    key = (batch_key, os.path.abspath(source_path), os.path.abspath(output_path))
    return key in completed_index and os.path.exists(output_path)


def _preset_hash(preset):
    payload = json.dumps(preset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _batch_config_key(
    preset_hash,
    suffix,
    output_format,
    *,
    deep_denoise=False,
    output_sharpening="standard",
    output_quality=92,
    resize_long_edge=0,
    keep_metadata=False,
):
    try:
        quality = int(output_quality)
    except (TypeError, ValueError):
        quality = 92
    try:
        resize_limit = int(resize_long_edge)
    except (TypeError, ValueError):
        resize_limit = 0
    return json.dumps(
        {
            "preset_hash": preset_hash,
            "suffix": suffix,
            "format": output_format,
            "deep_denoise": bool(deep_denoise),
            "output_sharpening": str(output_sharpening or "standard"),
            "output_quality": quality,
            "resize_long_edge": resize_limit,
            "keep_metadata": bool(keep_metadata),
        },
        sort_keys=True,
    )


def _default_color_settings():
    return {
        "input_profile": "auto",
        "raw_white_balance": "camera",
        "raw_colorspace": "srgb",
        "raw_lut_enabled": False,
        "raw_lut_path": "",
        "raw_auto_brightness": True,
        "raw_highlight_mode": "clip",
        "raw_demosaic": "auto",
        "working_space": "srgb",
        "scene_linear_denoise": False,
        "use_learned_denoise": True,
        "output_transform": "srgb",
        "icc_policy": "srgb",
    }


def _get_preset_render_params(preset):
    params = {"global": dict(preset.get("global_params", {}))}
    selective = preset.get("selective_params", {})
    for layer in MASK_ORDER:
        params[layer] = dict(selective.get(layer, {}))
    return params


def _get_preset_color_settings(preset):
    merged = _default_color_settings()
    color_settings = preset.get("color_settings", {})
    if isinstance(color_settings, dict):
        merged.update(color_settings)
    return merged


def _face_index_from_entry(entry, fallback: int = 0) -> int:
    try:
        return max(0, int(entry.get("face_index", fallback)))
    except (AttributeError, TypeError, ValueError):
        return max(0, int(fallback))


def _override_active_face_index(override) -> int:
    try:
        return max(0, int((override or {}).get("active_face_index", 0)))
    except (TypeError, ValueError):
        return 0


def _override_has_explicit_active_face(override) -> bool:
    return isinstance(override, dict) and "active_face_index" in override


def _selective_params_for_face(override, face_index: int) -> dict:
    selective_by_face = override.get("selective_by_face") or []
    for fallback, entry in enumerate(selective_by_face):
        if _face_index_from_entry(entry, fallback) == face_index:
            return entry.get("params", {}) or {}
    if selective_by_face:
        return selective_by_face[0].get("params", {}) or {}
    return {}


def _get_override_render_params(override):
    """Like _get_preset_render_params, but for a per-image collection-override payload,
    whose selective settings are keyed per detected face rather than one flat layer dict.
    Single-pass export uses the active face's selective recipe, matching what the editor
    preview showed when the override was saved. Multi-person exports with genuinely different
    per-face settings take a separate path that renders each person with their own recipe."""
    params = {"global": dict(override.get("global_params", {}))}
    primary = _selective_params_for_face(override, _override_active_face_index(override))
    for layer in MASK_ORDER:
        params[layer] = dict(primary.get(layer, {}))
    return params


def _override_layer_order(override):
    layer_order = tuple([layer for layer in override.get("layer_order", MASK_ORDER) if layer in MASK_ORDER] or MASK_ORDER)
    for layer in MASK_ORDER:
        if layer not in layer_order:
            layer_order += (layer,)
    return layer_order


def _supported_image_paths(input_dir):
    paths = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        paths.append(path)
    return paths


def _batch_output_path(source_path, output_dir, suffix, output_ext):
    stem = Path(source_path).stem
    return os.path.join(output_dir, f"{stem}{suffix}{output_ext}")


def _resolve_task_output_path(source_path, output_dir, suffix, output_ext, explicit_output_paths=None):
    explicit = explicit_output_paths or {}
    out_path = explicit.get(os.path.abspath(source_path))
    if out_path:
        return os.path.abspath(out_path)
    return _batch_output_path(source_path, output_dir, suffix=suffix, output_ext=output_ext)


def _read_image_file(path: str, color_settings=None):
    if is_raw_path(path):
        # Shared decoder so batch honors the same RAW WB/colorspace/LUT/auto-brightness as
        # the Qt UI for the same color settings (the decode metadata is unused in batch).
        return decode_raw(path, color_settings)[0]

    pil = Image.open(path).convert("RGB")
    return np.asarray(pil, dtype=np.float32) / 255.0


def _segment_per_face(image_array, segmenter):
    """Run segmentation once per detected face and return the raw, unmerged per-face
    results -- the building block both _combine_face_masks (today's union-merge, for the
    common single-recipe render) and _render_multi_person (per-person compositing) are
    derived from, so no face ever pays for segmentation twice."""
    faces = segmenter.list_faces(image_array)
    if not faces:
        masks, guides = segmenter.segment_with_guides(image_array, face_index=0)
        return [masks], [guides], faces

    per_face_masks = []
    per_face_guides = []
    for idx in range(len(faces)):
        masks, guides = segmenter.segment_with_guides(image_array, face_index=idx)
        per_face_masks.append(masks)
        per_face_guides.append(guides)
    return per_face_masks, per_face_guides, faces


def _merge_face_masks(per_face_masks):
    """Union every face's masks together (np.maximum) -- correct for "this feature belongs
    to *a* face" layers (face/skin/eyes/lips/hair), and the long-standing batch convention
    for rendering one recipe across a whole multi-person photo."""
    combined = None
    for masks in per_face_masks:
        if combined is None:
            combined = {key: value.copy() for key, value in masks.items()}
        else:
            for key, value in masks.items():
                if key not in combined:
                    combined[key] = value.copy()
                else:
                    combined[key] = np.maximum(combined[key], value)
    return combined


def _combine_face_masks(image_array, segmenter):
    per_face_masks, per_face_guides, faces = _segment_per_face(image_array, segmenter)
    if not faces:
        return per_face_masks[0], per_face_guides[0], 0, faces
    combined = _merge_face_masks(per_face_masks)
    guide_list = [g for g in per_face_guides if g] or None
    return combined, guide_list, len(faces), faces


PER_PERSON_LAYERS = ("person", "face", "skin", "eyes", "lips", "hair")


def _mask_keys_for_render_params(render_params: dict | None) -> tuple[str, ...]:
    params = render_params or {}
    return tuple(layer for layer in MASK_ORDER if params.get(layer))


def _has_per_person_layer_params(render_params: dict | None) -> bool:
    params = render_params or {}
    return any(params.get(layer) for layer in PER_PERSON_LAYERS)


def _override_needs_local_masks(override) -> bool:
    """Like _needs_local_masks, but checks every saved face's params, not just the first --
    a collection image where only the second face was ever customized must still trigger
    segmentation, even though _get_override_render_params (which only reflects face 0)
    would look empty."""
    for entry in override.get("selective_by_face") or []:
        params = entry.get("params", {})
        if any(params.get(layer) for layer in MASK_ORDER):
            return True
    return False


def _needs_multi_person_render(override, n_faces) -> bool:
    """Whether this image's saved per-face settings actually differ from person to person --
    if everyone has the same (or no) per-person customization, the existing single-pass
    render already produces the correct result, so there's no reason to pay for N render
    passes plus re-segmenting on every cache hit."""
    if n_faces < 2:
        return False
    selective_by_face = override.get("selective_by_face") or []
    if len(selective_by_face) < 2:
        return False
    first = None
    for entry in selective_by_face:
        params = entry.get("params", {})
        snapshot = {layer: params.get(layer, {}) for layer in PER_PERSON_LAYERS}
        if first is None:
            first = snapshot
        elif snapshot != first:
            return True
    return False


def _override_per_person_profiles_differ(override) -> bool:
    selective_by_face = (override or {}).get("selective_by_face") or []
    if len(selective_by_face) < 2:
        return False
    first = None
    for entry in selective_by_face:
        params = entry.get("params", {}) or {}
        snapshot = {layer: params.get(layer, {}) for layer in PER_PERSON_LAYERS}
        if first is None:
            first = snapshot
        elif snapshot != first:
            return True
    return False


def _has_real_person_split(per_face_masks) -> bool:
    """Whether segmentation actually produced distinct per-person masks for this image, vs.
    Person aliasing to Subjects (a single face, or the watershed split was rejected as
    unreliable). Compared directly against the masks (np.array_equal) rather than trusting an
    internal confidence flag, since the masks are the one thing that actually determines
    whether per-person compositing means anything for this image."""
    if len(per_face_masks) < 2:
        return False
    subjects = per_face_masks[0].get("subjects")
    if subjects is None:
        return False
    return any(not np.array_equal(masks.get("person"), subjects) for masks in per_face_masks)


def _render_multi_person(
    full,
    per_face_masks,
    per_face_guides,
    override,
    mask_adjustments,
    layer_options,
    layer_order,
    color_settings,
    runtime_settings,
):
    """Render each detected person's own Face/Skin/Eyes/Lips/Hair/Person settings separately
    and composite them into one image using each person's own "person" mask, instead of
    collapsing everyone onto one face's settings. Scene-wide layers (background/subjects) and
    global come from the primary (first) face's saved params, matching the existing
    single-pass convention -- only the per-person layers vary pass to pass."""
    global_params = dict(override.get("global_params", {}))
    selective_by_face = override.get("selective_by_face") or []
    by_index = {entry.get("face_index", i): entry.get("params", {}) for i, entry in enumerate(selective_by_face)}
    scene_params = by_index.get(0, {})
    acceleration = runtime_settings.get("acceleration_mode", "auto")

    base = None
    for i, masks in enumerate(per_face_masks):
        face_params = by_index.get(i, {})
        params = {"global": global_params}
        for layer in ("background", "subjects"):
            params[layer] = dict(scene_params.get(layer, {}))
        for layer in PER_PERSON_LAYERS:
            params[layer] = dict(face_params.get(layer, {}))

        adjusted_masks = apply_mask_adjustments(masks, mask_adjustments, acceleration=acceleration)
        guides = per_face_guides[i] if i < len(per_face_guides) else None
        pass_result = process_all_layers(
            full,
            params,
            adjusted_masks,
            geometry=guides,
            layer_order=layer_order,
            layer_options=layer_options,
            color_settings=color_settings,
            runtime_settings=runtime_settings,
        )
        pass_array = to_float(np.asarray(pass_result, dtype=np.uint8))
        if base is None:
            base = pass_array
        else:
            person_mask = np.clip(np.asarray(adjusted_masks.get("person"), dtype=np.float32), 0.0, 1.0)
            base = blend_with_mask(base, pass_array, person_mask)

    return Image.fromarray(to_uint8(base))


DEFAULT_MAX_WORKERS = 4
DEEP_DENOISE_DEFAULT_MAX_WORKERS = 2
DEEP_DENOISE_WORKER_MEMORY_GB = 4.0
SEGMENTATION_WORKER_MEMORY_GB = 2.5
BASE_WORKER_MEMORY_GB = 1.0
RESERVED_SYSTEM_MEMORY_GB = 2.0
DEEP_DENOISE_TOTAL_TILE_WORKERS_CAP = 2
DEEP_DENOISE_WAVE_RECHECK = True
LOW_MEMORY_SEGMENTATION_MARGIN_GB = 3.0
LOW_MEMORY_SEGMENTATION_ENV = {
    "PORTRAIT_DISABLE_MASKDINO": "1",
    "PORTRAIT_DISABLE_RMBG": "1",
    "PORTRAIT_DISABLE_SAM": "1",
    "PORTRAIT_DISABLE_SAM_FACE": "1",
}

# Each worker process keeps its own FaceSegmenter, built lazily the first time a task
# on that worker actually needs one (and reused for any later task on the same worker).
# Tried sharing one segmenter across threads instead of processes so models load once
# total -- reverted: model inference releases the GIL, but the surrounding numpy/PIL
# mask + color-space work doesn't, so threads serialized on the GIL and a single image
# went from ~19s to ~270s. Processes still pay for N model copies in memory, but that's
# a one-time, bounded cost (DEFAULT_MAX_WORKERS=4) -- worth it for real parallelism.
_worker_segmenter = None
_worker_segmenter_low_memory = None
_worker_deep_denoiser = None
_worker_deep_denoiser_use_coreml = None


def _safe_int(value, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"1", "true", "yes", "on"}:
            return True
        if stripped in {"0", "false", "no", "off"}:
            return False
        return bool(default)
    return bool(value)


def _deep_denoise_use_coreml(job: dict) -> bool:
    if "deep_denoise_use_coreml" in job:
        return _safe_bool(job.get("deep_denoise_use_coreml"), False)
    env_value = os.getenv("PORTRAIT_EXPORT_DEEP_DENOISE_USE_COREML")
    return _safe_bool(env_value, False)


def _available_memory_gb():
    """Best-effort currently available memory estimate.

    This is intentionally conservative and optional: if probing fails, worker selection
    falls back to static caps. On macOS, "Pages free" alone is too pessimistic, so include
    inactive/speculative pages that the OS can usually reclaim under pressure.
    """
    try:
        if platform.system() == "Darwin":
            page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
            output = subprocess.check_output(["vm_stat"], text=True)
            pages = {}
            for line in output.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                digits = "".join(ch for ch in value if ch.isdigit())
                if digits:
                    pages[key.strip()] = int(digits)
            reclaimable = (
                pages.get("Pages free", 0)
                + pages.get("Pages inactive", 0)
                + pages.get("Pages speculative", 0)
                + pages.get("Pages purgeable", 0)
            )
            if reclaimable > 0:
                return reclaimable * page_size / (1024 ** 3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        return None
    return None


def _memory_pressure_level():
    """Return normal/warning/critical when the OS exposes pressure, else None."""
    if platform.system() != "Darwin":
        return None
    try:
        output = subprocess.check_output(
            ["memory_pressure"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except Exception:
        return None
    lower = output.lower()
    if "no memory pressure" in lower:
        return "normal"
    if "critical" in lower:
        return "critical"
    if "warn" in lower or "pressure" in lower:
        return "warning"
    return "normal"


def _task_needs_segmentation(task: dict) -> bool:
    if task.get("disable_segmentation", False):
        return False
    override = task.get("override")
    if override is not None:
        return _override_needs_local_masks(override)
    return _needs_local_masks(task.get("render_params", {}))


def _worker_memory_budget_gb(*, deep_denoise: bool, needs_segmentation: bool) -> float:
    budget = BASE_WORKER_MEMORY_GB
    if needs_segmentation:
        budget += SEGMENTATION_WORKER_MEMORY_GB
    if deep_denoise:
        budget += DEEP_DENOISE_WORKER_MEMORY_GB
    return budget


def _deep_denoise_static_cap(job: dict) -> int:
    default_cap = _safe_int(
        os.getenv("PORTRAIT_DEEP_DENOISE_MAX_WORKERS"),
        DEEP_DENOISE_DEFAULT_MAX_WORKERS,
        1,
    )
    return _safe_int(job.get("deep_denoise_max_workers"), default_cap, 1)


def _deep_denoise_requested_tile_workers() -> int:
    return _safe_int(os.getenv("PORTRAIT_DEEP_DENOISE_WORKERS"), 2, 1, 8)


def _deep_denoise_tile_workers_for_plan(max_workers: int, memory_cap: int | None, pressure_level: str | None) -> int:
    requested = _deep_denoise_requested_tile_workers()
    if pressure_level in {"warning", "critical"} or memory_cap == 1:
        return 1
    per_worker_cap = max(1, DEEP_DENOISE_TOTAL_TILE_WORKERS_CAP // max(1, max_workers))
    return max(1, min(requested, per_worker_cap))


def _resolve_max_workers(job: dict, tasks: list[dict], *, deep_denoise: bool) -> tuple[int, dict]:
    if not tasks:
        return 0, {}

    cpu_count = os.cpu_count() or DEFAULT_MAX_WORKERS
    requested = _safe_int(job.get("max_workers"), DEFAULT_MAX_WORKERS, 1)
    static_cap = _deep_denoise_static_cap(job) if deep_denoise else DEFAULT_MAX_WORKERS
    needs_segmentation = any(_task_needs_segmentation(task) for task in tasks)
    worker_budget = _worker_memory_budget_gb(
        deep_denoise=deep_denoise,
        needs_segmentation=needs_segmentation,
    )
    available_gb = _available_memory_gb()
    pressure_level = _memory_pressure_level()
    memory_cap = None
    if available_gb is not None:
        usable_gb = max(0.0, available_gb - RESERVED_SYSTEM_MEMORY_GB)
        memory_cap = max(1, int(usable_gb // worker_budget)) if worker_budget > 0 else len(tasks)
    if pressure_level in {"warning", "critical"}:
        memory_cap = 1

    max_workers = min(requested, static_cap, cpu_count, len(tasks))
    if memory_cap is not None:
        max_workers = min(max_workers, memory_cap)
    max_workers = max(1, max_workers)
    deep_denoise_tile_workers = None
    if deep_denoise:
        deep_denoise_tile_workers = _deep_denoise_tile_workers_for_plan(
            max_workers,
            memory_cap,
            pressure_level,
        )
    low_memory_segmentation = False
    if needs_segmentation:
        if pressure_level in {"warning", "critical"}:
            low_memory_segmentation = True
        elif available_gb is not None and available_gb <= worker_budget + LOW_MEMORY_SEGMENTATION_MARGIN_GB:
            low_memory_segmentation = True

    return max_workers, {
        "requested_workers": requested,
        "static_cap": static_cap,
        "cpu_count": cpu_count,
        "task_count": len(tasks),
        "deep_denoise": bool(deep_denoise),
        "needs_segmentation": bool(needs_segmentation),
        "available_memory_gb": round(available_gb, 2) if available_gb is not None else None,
        "memory_pressure": pressure_level,
        "estimated_worker_memory_gb": worker_budget,
        "memory_cap": memory_cap,
        "deep_denoise_tile_workers": deep_denoise_tile_workers,
        "low_memory_segmentation": low_memory_segmentation,
        "selected_workers": max_workers,
    }


def _lower_priority():
    """Run batch workers (and the dispatcher) at lower OS scheduling priority so the
    export doesn't starve the foreground app/UI of CPU while it runs."""
    try:
        os.nice(10)
    except Exception:
        pass


def _apply_low_memory_segmentation_env() -> None:
    for key, value in LOW_MEMORY_SEGMENTATION_ENV.items():
        os.environ[key] = value


def _needs_local_masks(render_params: dict) -> bool:
    """Whether any per-region layer (face/skin/eyes/lips/hair/subjects/background) has
    actual params -- if every one is empty, only the mask-free global adjustments apply,
    so the expensive face/subject segmentation can be skipped entirely."""
    return any(render_params.get(layer) for layer in MASK_ORDER)


def _resolve_masks_and_guides(full, task, segmenter, render_full=None):
    """Decide masks/guides for one collection-export image, going through the disk cache
    exactly as before for the common case. Returns either (masks, guides, None) for the
    existing single-pass render, or (None, None, image) when this image's saved per-face
    settings genuinely differ from person to person and a real person split exists, in which
    case the multi-person render has already happened."""
    override = task["override"]
    cache_signature = segmenter_cache_signature(segmenter)
    cached = load_analysis(
        task["path"],
        kind=CACHE_KIND_ALL_FACES,
        image_shape=full.shape[:2],
        backend_signature=cache_signature,
    )
    per_face_masks = per_face_guides = None
    if cached is not None:
        merged_masks, guides, faces = cached["masks"], cached.get("guides"), cached["faces"]
    else:
        per_face_masks, per_face_guides, faces = _segment_per_face(full, segmenter)
        merged_masks = _merge_face_masks(per_face_masks) if faces else per_face_masks[0]
        guides = ([g for g in per_face_guides if g] or None) if faces else per_face_guides[0]
        save_analysis(
            task["path"],
            kind=CACHE_KIND_ALL_FACES,
            image_shape=full.shape[:2],
            masks=merged_masks,
            faces=faces,
            guides=guides,
            backend_signature=segmenter_cache_signature(segmenter),
        )

    if _needs_multi_person_render(override, len(faces)):
        if per_face_masks is None:
            # Cache only ever stores the merged result -- this image's saved per-face
            # settings genuinely differ, so getting the unmerged per-face data back means
            # paying for segmentation again, even though the cache "hit".
            per_face_masks, per_face_guides, faces = _segment_per_face(full, segmenter)
        if faces and _has_real_person_split(per_face_masks):
            image = _render_multi_person(
                render_full if render_full is not None else full,
                per_face_masks,
                per_face_guides,
                override,
                task["mask_adjustments"],
                task["layer_options"],
                task["layer_order"],
                task["color_settings"],
                task["runtime_settings"],
            )
            return None, None, image

    masks = apply_mask_adjustments(
        merged_masks,
        task["mask_adjustments"],
        acceleration=task["runtime_settings"].get("acceleration_mode", "auto"),
    )
    return masks, guides, None


def _load_cached_masks_without_segmenter(full, task) -> tuple[dict | None, dict | list | None, dict | None]:
    override = task.get("override")
    render_params = task.get("render_params", {})
    required_mask_keys = _mask_keys_for_render_params(render_params)
    active_face_index = None
    active_per_person_layers = False
    preferred_kinds = (CACHE_KIND_ALL_FACES,)
    if override is not None and _override_has_explicit_active_face(override):
        active_face_index = _override_active_face_index(override)
        active_per_person_layers = _has_per_person_layer_params(render_params)
        preferred_kinds = (
            (CACHE_KIND_SINGLE_FACE, CACHE_KIND_ALL_FACES)
            if active_per_person_layers
            else (CACHE_KIND_ALL_FACES, CACHE_KIND_SINGLE_FACE)
        )

    cached = load_latest_compatible_analysis(
        task["path"],
        target_shape=full.shape[:2],
        preferred_kinds=preferred_kinds,
        face_index=active_face_index,
        required_mask_keys=required_mask_keys,
    )
    if cached is None:
        return None, None, None
    faces = cached.get("faces") or []
    kind = cached.get("kind", "")
    if override is not None:
        if _needs_multi_person_render(override, len(faces)):
            return None, None, None
        if kind == CACHE_KIND_SINGLE_FACE and active_face_index is None:
            return None, None, None
        if (
            kind == CACHE_KIND_ALL_FACES
            and active_face_index is not None
            and active_per_person_layers
            and (len(faces) != 1 or active_face_index != 0)
        ):
            return None, None, None
        if kind == CACHE_KIND_SINGLE_FACE and _override_per_person_profiles_differ(override) and not faces:
            return None, None, None

    masks = apply_mask_adjustments(
        cached["masks"],
        task["mask_adjustments"],
        acceleration=task["runtime_settings"].get("acceleration_mode", "auto"),
    )
    return masks, cached.get("guides"), {
        "analysis_cache_reused": True,
        "analysis_cache_kind": cached.get("kind", ""),
        "analysis_cache_face_index": cached.get("face_index"),
        "analysis_cache_mask_keys": sorted(cached.get("masks", {}).keys()),
        "analysis_cache_resized": bool(cached.get("resized")),
        "analysis_cache_source_shape": list(cached.get("source_shape") or ()),
    }


def _atomic_save_image(result: Image.Image, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}{out.suffix}")
    try:
        result.save(tmp)
        os.replace(tmp, out)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _resize_long_edge(image: Image.Image, limit: int | None) -> Image.Image:
    try:
        limit = int(limit or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= limit:
        return image
    scale = limit / float(longest)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _save_export_image(result: Image.Image, task: dict) -> None:
    out = Path(task["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp.{os.getpid()}{out.suffix}")
    fmt = str(task.get("output_format", "jpeg")).lower()
    save_kwargs = _export_metadata_kwargs(task.get("path", ""), out.suffix, bool(task.get("keep_metadata", False)))
    try:
        if fmt == "jpeg":
            result.save(tmp, "JPEG", quality=int(task.get("output_quality", 92)), subsampling=0, **save_kwargs)
        elif fmt == "png":
            result.save(tmp, "PNG", **save_kwargs)
        elif fmt == "tiff":
            result.save(tmp, "TIFF", **save_kwargs)
        else:
            result.save(tmp)
        os.replace(tmp, out)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _export_metadata_kwargs(source_path: str, out_ext: str, keep_metadata: bool) -> dict:
    if not keep_metadata:
        return {}
    if is_raw_path(source_path):
        return {}
    out_ext = str(out_ext or "").lower()
    kwargs = {}
    try:
        with Image.open(source_path) as im:
            exif = im.info.get("exif")
            if exif and out_ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                kwargs["exif"] = exif
            icc = im.info.get("icc_profile")
            if icc:
                kwargs["icc_profile"] = icc
            dpi = im.info.get("dpi")
            if dpi:
                kwargs["dpi"] = dpi
    except Exception:
        return {}
    return kwargs


def _process_one_image(task: dict) -> dict:
    global _worker_segmenter, _worker_segmenter_low_memory, _worker_deep_denoiser, _worker_deep_denoiser_use_coreml

    runtime_settings = task["runtime_settings"]
    disable_segmentation = task.get("disable_segmentation", False)
    deep_denoise = bool(task.get("deep_denoise", False))
    low_memory_segmentation = _safe_bool(task.get("low_memory_segmentation"), False)
    deep_denoise_retried_cpu = False
    analysis_cache_meta = None
    override = task.get("override")
    try:
        if _cancel_requested(task):
            return {"status": "canceled"}
        full = _read_image_file(task["path"], task.get("color_settings"))
        render_full = full
        if deep_denoise:
            use_coreml = _safe_bool(task.get("deep_denoise_use_coreml"), False)
            if _worker_deep_denoiser is None or _worker_deep_denoiser_use_coreml != use_coreml:
                _worker_deep_denoiser = DeepDenoiser(use_coreml=use_coreml)
                _worker_deep_denoiser_use_coreml = use_coreml
            if task.get("deep_denoise_workers") is not None:
                _worker_deep_denoiser.workers = _safe_int(
                    task.get("deep_denoise_workers"),
                    _worker_deep_denoiser.workers,
                    1,
                    8,
                )
            if not _worker_deep_denoiser.available:
                reason = _worker_deep_denoiser.reason_unavailable or "model unavailable"
                raise RuntimeError(f"Deep Denoise requested but unavailable: {reason}")

            def log_progress(done, total, *, provider=None):
                log_path = task.get("progress_log_path")
                if not log_path:
                    return
                _append_batch_log(
                    log_path,
                    {
                        "ts": _utc_timestamp(),
                        "job_id": task.get("job_id", ""),
                        "job_mode": task.get("job_mode", "batch"),
                        "status": "deep_denoise_progress",
                        "source_path": os.path.abspath(task["path"]),
                        "output_path": os.path.abspath(task["out_path"]),
                        "deep_denoise_done": int(done),
                        "deep_denoise_total": int(total),
                        "deep_denoise_provider": provider or getattr(_worker_deep_denoiser, "execution_provider", ""),
                    },
                )

            denoised = _worker_deep_denoiser.denoise(
                full,
                progress=lambda done, total: log_progress(done, total),
            )
            if denoised is None and use_coreml:
                cpu_denoiser = DeepDenoiser(use_coreml=False)
                if task.get("deep_denoise_workers") is not None:
                    cpu_denoiser.workers = _safe_int(
                        task.get("deep_denoise_workers"),
                        cpu_denoiser.workers,
                        1,
                        8,
                    )
                if not cpu_denoiser.available:
                    reason = cpu_denoiser.reason_unavailable or "model unavailable"
                    raise RuntimeError(f"Deep Denoise CoreML failed and CPU fallback is unavailable: {reason}")
                denoised = cpu_denoiser.denoise(
                    full,
                    progress=lambda done, total: log_progress(
                        done,
                        total,
                        provider=getattr(cpu_denoiser, "execution_provider", "CPUExecutionProvider"),
                    ),
                )
                if denoised is not None:
                    _worker_deep_denoiser = cpu_denoiser
                    _worker_deep_denoiser_use_coreml = False
                    deep_denoise_retried_cpu = True
            if denoised is None:
                raise RuntimeError("Deep Denoise requested but inference failed")
            render_full = denoised

        if override is not None:
            needs_masks = not disable_segmentation and _override_needs_local_masks(override)
        else:
            needs_masks = not disable_segmentation and _needs_local_masks(task["render_params"])

        result = None
        masks, guides = None, None
        if needs_masks:
            masks, guides, analysis_cache_meta = _load_cached_masks_without_segmenter(full, task)
            if masks is None:
                if low_memory_segmentation:
                    _apply_low_memory_segmentation_env()
                if _worker_segmenter is None or _worker_segmenter_low_memory != low_memory_segmentation:
                    _worker_segmenter = FaceSegmenter()
                    _worker_segmenter_low_memory = low_memory_segmentation
                if override is not None:
                    masks, guides, result = _resolve_masks_and_guides(
                        full, task, _worker_segmenter, render_full=render_full
                    )
                else:
                    cache_signature = segmenter_cache_signature(_worker_segmenter)
                    cached = load_analysis(
                        task["path"],
                        kind=CACHE_KIND_ALL_FACES,
                        image_shape=full.shape[:2],
                        backend_signature=cache_signature,
                    )
                    if cached is not None:
                        masks = cached["masks"]
                        guides = cached.get("guides")
                    else:
                        masks, guides, _face_count, faces = _combine_face_masks(full, _worker_segmenter)
                        save_analysis(
                            task["path"],
                            kind=CACHE_KIND_ALL_FACES,
                            image_shape=full.shape[:2],
                            masks=masks,
                            faces=faces,
                            guides=guides,
                            backend_signature=segmenter_cache_signature(_worker_segmenter),
                        )
                    masks = apply_mask_adjustments(
                        masks,
                        task["mask_adjustments"],
                        acceleration=runtime_settings.get("acceleration_mode", "auto"),
                    )
            else:
                result = None

        if result is None:
            result = process_all_layers(
                render_full,
                task["render_params"],
                masks,
                geometry=guides,
                layer_order=task["layer_order"],
                layer_options=task["layer_options"],
                color_settings=task["color_settings"],
                runtime_settings=runtime_settings,
            )
        if task["framing"] is not None:
            result = apply_framing(result, task["framing"])
        result = _resize_long_edge(result, task.get("resize_long_edge"))
        result = apply_output_sharpening(result, task.get("output_sharpening", "standard"))
        if _cancel_requested(task):
            return {"status": "canceled"}
        _save_export_image(result, task)
        result_payload = {"status": "success"}
        if deep_denoise_retried_cpu:
            result_payload["warning"] = "Deep Denoise CoreML inference failed; retried on CPU"
            result_payload["deep_denoise_coreml_retry"] = True
        if low_memory_segmentation:
            result_payload["low_memory_segmentation"] = True
        if analysis_cache_meta:
            result_payload.update(analysis_cache_meta)
        return result_payload
    except Exception as ex:
        return {"status": "error", "error": str(ex)}


def _task_with_worker_plan(task: dict, worker_plan: dict) -> dict:
    task = dict(task)
    tile_workers = worker_plan.get("deep_denoise_tile_workers")
    if tile_workers is not None:
        task["deep_denoise_workers"] = tile_workers
    if worker_plan.get("low_memory_segmentation"):
        task["low_memory_segmentation"] = True
    return task


def _deep_denoise_wave_size(max_workers: int, remaining_count: int, worker_plan: dict) -> int:
    if not DEEP_DENOISE_WAVE_RECHECK:
        return remaining_count
    if worker_plan.get("memory_pressure") in {"warning", "critical"} or worker_plan.get("memory_cap") == 1:
        return max(1, max_workers)
    if max_workers <= 1:
        return max(1, min(4, remaining_count))
    return max(1, max_workers)


def _run_task_wave(wave_tasks: list[dict], max_workers: int):
    logged_task_ids = set()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_lower_priority) as pool:
            futures = {pool.submit(_process_one_image, task): task for task in wave_tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                logged_task_ids.add(id(task))
                try:
                    result = future.result()
                except Exception as ex:
                    result = _retry_failed_task(task, f"worker failed: {ex}")
                    if result is None:
                        result = {"status": "error", "error": f"worker failed: {ex}"}
                yield task, result
    except Exception as ex:
        for task in wave_tasks:
            if id(task) not in logged_task_ids:
                result = _retry_failed_task(task, f"worker pool failed: {ex}")
                if result is None:
                    result = {"status": "error", "error": f"worker pool failed: {ex}"}
                yield task, result


def _retry_failed_task(task: dict, original_error: str) -> dict | None:
    result = _retry_deep_denoise_task_on_cpu(task, original_error)
    if result is not None:
        return result
    return _retry_segmentation_task_low_memory(task, original_error)


def _retry_deep_denoise_task_on_cpu(task: dict, original_error: str) -> dict | None:
    if (
        not task.get("deep_denoise")
        or not _safe_bool(task.get("deep_denoise_use_coreml"), False)
        or _cancel_requested(task)
    ):
        return None
    retry_task = dict(task)
    retry_task["deep_denoise_use_coreml"] = False
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=_lower_priority) as pool:
            future = pool.submit(_process_one_image, retry_task)
            result = future.result()
    except Exception as ex:
        return {"status": "error", "error": f"{original_error}; CPU retry failed: {ex}"}
    if result.get("status") == "success":
        result = dict(result)
        result["warning"] = f"{original_error}; retried Deep Denoise on CPU"
        result["deep_denoise_coreml_retry"] = True
        return result
    if result.get("status") == "error":
        result = dict(result)
        result["error"] = f"{original_error}; CPU retry failed: {result.get('error', 'unknown error')}"
    return result


def _retry_segmentation_task_low_memory(task: dict, original_error: str) -> dict | None:
    if _cancel_requested(task) or _safe_bool(task.get("low_memory_segmentation"), False):
        return None
    if not _task_needs_segmentation(task):
        return None
    retry_task = dict(task)
    retry_task["low_memory_segmentation"] = True
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=_lower_priority) as pool:
            future = pool.submit(_process_one_image, retry_task)
            result = future.result()
    except Exception as ex:
        return {"status": "error", "error": f"{original_error}; low-memory segmentation retry failed: {ex}"}
    if result.get("status") == "success":
        result = dict(result)
        result["warning"] = f"{original_error}; retried with low-memory segmentation"
        result["low_memory_segmentation"] = True
        result["segmentation_low_memory_retry"] = True
        return result
    if result.get("status") == "error":
        result = dict(result)
        result["error"] = (
            f"{original_error}; low-memory segmentation retry failed: "
            f"{result.get('error', 'unknown error')}"
        )
    return result


def run_batch_job(job: dict):
    output_dir = job["output_dir"]
    job_id = str(job.get("job_id", ""))
    job_mode = str(job.get("mode", "batch"))
    suffix = job.get("suffix", "_enhanced")
    output_format = job.get("output_format", "jpeg")
    skip_completed = bool(job.get("skip_completed", False))
    disable_segmentation = bool(job.get("disable_segmentation", False))
    output_sharpening = job.get("output_sharpening", "standard")
    output_quality = _safe_int(job.get("output_quality"), 92, 1, 100)
    resize_long_edge = _safe_int(job.get("resize_long_edge"), 0, 0) if job.get("resize_long_edge") else 0
    keep_metadata = bool(job.get("keep_metadata", False))
    deep_denoise = bool(job.get("deep_denoise", False))
    deep_denoise_use_coreml = _deep_denoise_use_coreml(job)
    runtime_settings = dict(job.get("runtime_settings", {"acceleration_mode": "auto"}))
    log_path = os.path.join(output_dir, "batch_export_log.jsonl")
    cancel_path = _cancel_marker_path(job)
    if cancel_path:
        job = dict(job)
        job["cancel_path"] = cancel_path

    # Collection export renders each image from its own saved per-image settings
    # (or plain defaults if it has none) instead of one preset applied to every image.
    use_image_overrides = job_mode in {"collection_export", "export_as"} and "image_overrides" in job
    image_overrides = {}
    preset_path = None
    preset_hash = None
    render_params = color_settings = mask_adjustments = layer_options = layer_order = None

    if use_image_overrides:
        image_overrides = job.get("image_overrides") or {}
    else:
        preset_path = job["preset_path"]
        with open(preset_path, "r", encoding="utf-8") as fh:
            preset = json.load(fh)
        render_params = _get_preset_render_params(preset)
        color_settings = _get_preset_color_settings(preset)
        mask_adjustments = preset.get("mask_adjustments", {})
        layer_options = preset.get("layer_options", {})
        layer_order = _override_layer_order(preset)
        preset_hash = _preset_hash(preset)

    output_ext = BATCH_OUTPUT_FORMATS[output_format]
    explicit_output_paths = {
        os.path.abspath(src): os.path.abspath(dst)
        for src, dst in (job.get("explicit_output_paths") or {}).items()
        if src and dst
    }
    fixed_batch_key = None if use_image_overrides else _batch_config_key(
        preset_hash,
        suffix,
        output_format,
        deep_denoise=deep_denoise,
        output_sharpening=output_sharpening,
        output_quality=output_quality,
        resize_long_edge=resize_long_edge,
        keep_metadata=keep_metadata,
    )
    completed_index = _load_batch_log_index(log_path) if skip_completed else {}

    source_paths = job.get("source_paths")
    if not source_paths:
        input_dir = job["input_dir"]
        source_paths = _supported_image_paths(input_dir)

    tasks = []
    for path in source_paths:
        out_path = _resolve_task_output_path(
            path,
            output_dir,
            suffix=suffix,
            output_ext=output_ext,
            explicit_output_paths=explicit_output_paths,
        )
        framing = None
        preset_name = os.path.basename(preset_path) if preset_path else "(per-image settings)"

        if use_image_overrides:
            override = image_overrides.get(os.path.abspath(path)) or {}
            render_params = _get_override_render_params(override)
            color_settings = _get_preset_color_settings(override)
            mask_adjustments = override.get("mask_adjustments", {})
            layer_options = override.get("layer_options", {})
            layer_order = _override_layer_order(override)
            framing = override.get("framing")
            settings_hash = _preset_hash(override)
            batch_key = _batch_config_key(
                settings_hash,
                suffix,
                output_format,
                deep_denoise=deep_denoise,
                output_sharpening=output_sharpening,
                output_quality=output_quality,
                resize_long_edge=resize_long_edge,
                keep_metadata=keep_metadata,
            )
        else:
            settings_hash = preset_hash
            batch_key = fixed_batch_key

        log_fields = {
            "source_path": os.path.abspath(path),
            "output_path": os.path.abspath(out_path),
            "batch_key": batch_key,
            "preset_hash": settings_hash,
            "preset_name": preset_name,
        }

        if skip_completed and _batch_should_skip(completed_index, batch_key, path, out_path):
            _append_batch_log(
                log_path,
                {
                    "ts": _utc_timestamp(),
                    "job_id": job_id,
                    "job_mode": job_mode,
                    "status": "skipped",
                    "reason": "already_completed",
                    "suffix": suffix,
                    "output_format": output_format,
                    **log_fields,
                },
            )
            continue

        tasks.append(
            {
                "path": path,
                "out_path": out_path,
                "render_params": render_params,
                "color_settings": color_settings,
                "mask_adjustments": mask_adjustments,
                "layer_options": layer_options,
                "layer_order": layer_order,
                "framing": framing if use_image_overrides else None,
                "override": override if use_image_overrides else None,
                "runtime_settings": runtime_settings,
                "disable_segmentation": disable_segmentation,
                "output_sharpening": output_sharpening,
                "output_quality": output_quality,
                "output_format": output_format,
                "resize_long_edge": resize_long_edge,
                "keep_metadata": keep_metadata,
                "deep_denoise": deep_denoise,
                "deep_denoise_use_coreml": deep_denoise_use_coreml,
                "cancel_path": cancel_path,
                "job_path": job.get("_job_path", ""),
                "job_id": job_id,
                "job_mode": job_mode,
                "progress_log_path": log_path,
                "log_fields": log_fields,
            }
        )

    if not tasks:
        return
    if _cancel_requested(job):
        _append_cancel_record(log_path, job_id, job_mode)
        return

    remaining = list(tasks)
    wave_index = 0
    while remaining:
        if _cancel_requested(job):
            _append_cancel_record(log_path, job_id, job_mode)
            return
        wave_index += 1
        max_workers, worker_plan = _resolve_max_workers(job, remaining, deep_denoise=deep_denoise)
        wave_size = (
            _deep_denoise_wave_size(max_workers, len(remaining), worker_plan)
            if deep_denoise
            else len(remaining)
        )
        wave_tasks = [
            _task_with_worker_plan(task, worker_plan)
            for task in remaining[:wave_size]
        ]
        _append_batch_log(
            log_path,
            {
                "ts": _utc_timestamp(),
                "job_id": job_id,
                "job_mode": job_mode,
                "status": "worker_plan",
                "suffix": suffix,
                "output_format": output_format,
                "wave_index": wave_index,
                "wave_size": len(wave_tasks),
                "remaining_tasks": len(remaining),
                "deep_denoise_use_coreml": deep_denoise_use_coreml,
                **worker_plan,
            },
        )

        for task, result in _run_task_wave(wave_tasks, max_workers):
            record = {
                "ts": _utc_timestamp(),
                "job_id": job_id,
                "job_mode": job_mode,
                "status": result["status"],
                "suffix": suffix,
                "output_format": output_format,
                **task["log_fields"],
            }
            if result["status"] == "error":
                record["error"] = result["error"]
            elif result["status"] == "canceled":
                record["reason"] = "user_canceled"
            if result.get("warning"):
                record["warning"] = result["warning"]
            if result.get("deep_denoise_coreml_retry"):
                record["deep_denoise_coreml_retry"] = True
            if result.get("low_memory_segmentation"):
                record["low_memory_segmentation"] = True
            if result.get("segmentation_low_memory_retry"):
                record["segmentation_low_memory_retry"] = True
            if result.get("analysis_cache_reused"):
                record["analysis_cache_reused"] = True
                record["analysis_cache_kind"] = result.get("analysis_cache_kind", "")
                record["analysis_cache_face_index"] = result.get("analysis_cache_face_index")
                record["analysis_cache_mask_keys"] = result.get("analysis_cache_mask_keys", [])
                record["analysis_cache_resized"] = bool(result.get("analysis_cache_resized", False))
                record["analysis_cache_source_shape"] = result.get("analysis_cache_source_shape", [])
            _append_batch_log(log_path, record)
        if _cancel_requested(job):
            _append_cancel_record(log_path, job_id, job_mode)
            return
        remaining = remaining[wave_size:]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portrait Enhancer background batch runner")
    parser.add_argument("--job", required=True, help="Path to batch job JSON")
    args = parser.parse_args(argv)

    _lower_priority()
    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    job["_job_path"] = args.job
    job.setdefault("cancel_path", f"{args.job}.cancel")
    run_batch_job(job)


if __name__ == "__main__":
    main()
