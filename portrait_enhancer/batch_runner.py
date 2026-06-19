"""Detached batch runner for Portrait Enhancer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rawpy

    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

from portrait_enhancer.config import MASK_ORDER
from portrait_enhancer.core.masks import apply_mask_adjustments
from portrait_enhancer.core.processing import process_all_layers
from portrait_enhancer.core.segmentation import FaceSegmenter

SUPPORTED_IMAGE_EXTS = (".cr2", ".nef", ".arw", ".dng", ".raw", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
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


def _batch_config_key(preset_hash, suffix, output_format):
    return json.dumps({"preset_hash": preset_hash, "suffix": suffix, "format": output_format}, sort_keys=True)


def _default_color_settings():
    return {
        "input_profile": "auto",
        "raw_white_balance": "camera",
        "raw_colorspace": "srgb",
        "raw_lut_enabled": False,
        "raw_lut_path": "",
        "working_space": "srgb",
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


def _read_image_file(path: str):
    ext = Path(path).suffix.lower()
    if ext in {".cr2", ".nef", ".arw", ".dng", ".raw"}:
        if not HAS_RAWPY:
            raise RuntimeError("rawpy is not installed. Install base dependencies first.")
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16)
        return rgb16.astype(np.float32) / 65535.0

    pil = Image.open(path).convert("RGB")
    return np.asarray(pil, dtype=np.float32) / 255.0


def _combine_face_masks(image_array, segmenter):
    faces = segmenter.list_faces(image_array)
    if not faces:
        masks, guides = segmenter.segment_with_guides(image_array, face_index=0)
        return masks, guides, 0

    combined = None
    guide_list = []
    for idx in range(len(faces)):
        masks, guides = segmenter.segment_with_guides(image_array, face_index=idx)
        if combined is None:
            combined = {key: value.copy() for key, value in masks.items()}
        else:
            for key, value in masks.items():
                if key not in combined:
                    combined[key] = value.copy()
                else:
                    combined[key] = np.maximum(combined[key], value)
        if guides:
            guide_list.append(guides)
    return combined, guide_list or None, len(faces)


def run_batch_job(job: dict):
    preset_path = job["preset_path"]
    output_dir = job["output_dir"]
    job_id = str(job.get("job_id", ""))
    job_mode = str(job.get("mode", "batch"))
    suffix = job.get("suffix", "_enhanced")
    output_format = job.get("output_format", "jpeg")
    skip_completed = bool(job.get("skip_completed", False))
    runtime_settings = dict(job.get("runtime_settings", {"acceleration_mode": "auto"}))
    log_path = os.path.join(output_dir, "batch_export_log.jsonl")

    with open(preset_path, "r", encoding="utf-8") as fh:
        preset = json.load(fh)

    render_params = _get_preset_render_params(preset)
    color_settings = _get_preset_color_settings(preset)
    mask_adjustments = preset.get("mask_adjustments", {})
    layer_options = preset.get("layer_options", {})
    layer_order = tuple([layer for layer in preset.get("layer_order", MASK_ORDER) if layer in MASK_ORDER] or MASK_ORDER)
    for layer in MASK_ORDER:
        if layer not in layer_order:
            layer_order += (layer,)

    output_ext = BATCH_OUTPUT_FORMATS[output_format]
    preset_hash = _preset_hash(preset)
    batch_key = _batch_config_key(preset_hash, suffix, output_format)
    completed_index = _load_batch_log_index(log_path) if skip_completed else {}
    segmenter = FaceSegmenter()

    source_paths = job.get("source_paths")
    if not source_paths:
        input_dir = job["input_dir"]
        source_paths = _supported_image_paths(input_dir)

    for path in source_paths:
        out_path = _batch_output_path(path, output_dir, suffix=suffix, output_ext=output_ext)
        if skip_completed and _batch_should_skip(completed_index, batch_key, path, out_path):
            _append_batch_log(
                log_path,
                {
                    "ts": _utc_timestamp(),
                    "job_id": job_id,
                    "job_mode": job_mode,
                    "status": "skipped",
                    "reason": "already_completed",
                    "source_path": os.path.abspath(path),
                    "output_path": os.path.abspath(out_path),
                    "batch_key": batch_key,
                    "preset_hash": preset_hash,
                    "preset_name": os.path.basename(preset_path),
                    "suffix": suffix,
                    "output_format": output_format,
                },
            )
            continue
        try:
            full = _read_image_file(path)
            masks, guides, _face_count = _combine_face_masks(full, segmenter)
            masks = apply_mask_adjustments(
                masks,
                mask_adjustments,
                acceleration=runtime_settings.get("acceleration_mode", "auto"),
            )
            result = process_all_layers(
                full,
                render_params,
                masks,
                geometry=guides,
                layer_order=layer_order,
                layer_options=layer_options,
                color_settings=color_settings,
                runtime_settings=runtime_settings,
            )
            result.save(out_path)
            _append_batch_log(
                log_path,
                {
                    "ts": _utc_timestamp(),
                    "job_id": job_id,
                    "job_mode": job_mode,
                    "status": "success",
                    "source_path": os.path.abspath(path),
                    "output_path": os.path.abspath(out_path),
                    "batch_key": batch_key,
                    "preset_hash": preset_hash,
                    "preset_name": os.path.basename(preset_path),
                    "suffix": suffix,
                    "output_format": output_format,
                },
            )
        except Exception as ex:
            _append_batch_log(
                log_path,
                {
                    "ts": _utc_timestamp(),
                    "job_id": job_id,
                    "job_mode": job_mode,
                    "status": "error",
                    "error": str(ex),
                    "source_path": os.path.abspath(path),
                    "output_path": os.path.abspath(out_path),
                    "batch_key": batch_key,
                    "preset_hash": preset_hash,
                    "preset_name": os.path.basename(preset_path),
                    "suffix": suffix,
                    "output_format": output_format,
                },
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Portrait Enhancer background batch runner")
    parser.add_argument("--job", required=True, help="Path to batch job JSON")
    args = parser.parse_args(argv)

    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    run_batch_job(job)


if __name__ == "__main__":
    main()
