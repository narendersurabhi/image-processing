"""Detached background-analysis runner.

Pre-computes and caches face/segmentation results for filmstrip-selected (but not yet
opened) images, in a separate OS process. A same-process QThreadPool thread can be given a
lower *OS scheduling priority* hint, but it still shares the GUI process's GIL -- two Python
threads doing CPU-bound work still serialize against each other regardless of OS priority,
which measured out to a real ~39% slowdown of active-image segmentation when a low-priority
preload thread was busy at the same time. A separate process has no such sharing: the OS
scheduler (helped by os.nice below) decides independently, and the GUI process's own thread
is never blocked waiting for this process's GIL turn.

Writes one JSON-lines progress record per image to `log_path` (status: cached | stored |
error) so the GUI can poll for completion -- mirrors batch_runner's batch_export_log.jsonl
pattern. Uses the same lock-free analysis_cache as the editor and batch export, so multiple
preload processes (or a preload process running alongside batch export) can safely share the
cache with no coordination.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import rawpy

    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

from portrait_enhancer.core.raw_decode import decode_raw, is_raw_path
from portrait_enhancer.core.analysis_cache import (
    CACHE_KIND_SINGLE_FACE,
    load_analysis,
    save_analysis,
    segmenter_cache_signature,
)
from portrait_enhancer.core.segmentation import FaceSegmenter
from portrait_enhancer.core.utils import resize_image


def _lower_priority():
    """Always yield to the foreground app -- this work is a convenience preload, never
    something a user is waiting on directly."""
    try:
        os.nice(10)
    except Exception:
        pass


def _append_log(log_path: str, record: dict) -> None:
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _read_image_file(path: str) -> np.ndarray:
    if is_raw_path(path):
        # Shared decoder (camera-WB defaults) so background analysis decodes RAW the same way
        # the interactive/batch paths do; the decode metadata is unused here.
        return decode_raw(path)[0]
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return np.asarray(pil, dtype=np.float32) / 255.0


def _build_preview_proxy(full: np.ndarray, max_dim: int = 1600):
    h, w = full.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale >= 0.999:
        return full.copy(), 1.0
    preview = resize_image(full, (max(1, int(w * scale)), max(1, int(h * scale))), acceleration="auto")
    return preview.astype(np.float32), scale


def run_preload_job(job: dict) -> None:
    _lower_priority()
    log_path = job["log_path"]
    items = job.get("items", [])

    # Built lazily on first use, then reused for every item in this job -- this process
    # already pays the ~900ms ONNX/mediapipe model-load cost once per launch, same tradeoff
    # already accepted for batch export jobs.
    segmenter = None

    for item in items:
        path = item["path"]
        face_index = max(0, int(item.get("face_index", 0)))
        try:
            full = _read_image_file(path)
            preview, _scale = _build_preview_proxy(full)
            if segmenter is None:
                segmenter = FaceSegmenter()
            signature = segmenter_cache_signature(segmenter)
            cached = load_analysis(
                path,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=preview.shape[:2],
                face_index=face_index,
                backend_signature=signature,
            )
            if cached is not None:
                _append_log(log_path, {"path": path, "status": "cached", "ts": time.time()})
                continue

            faces = segmenter.list_faces(preview)
            resolved_face_index = min(face_index, len(faces) - 1) if faces else 0
            masks, guides = segmenter.segment_with_guides(preview, face_index=resolved_face_index)
            save_analysis(
                path,
                kind=CACHE_KIND_SINGLE_FACE,
                image_shape=preview.shape[:2],
                masks=masks,
                faces=faces,
                guides=guides,
                face_index=resolved_face_index,
                backend_signature=segmenter_cache_signature(segmenter),
            )
            _append_log(log_path, {"path": path, "status": "stored", "ts": time.time()})
        except Exception as ex:
            _append_log(log_path, {"path": path, "status": "error", "error": str(ex), "ts": time.time()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Portrait Enhancer background analysis preload runner")
    parser.add_argument("--job", required=True, help="Path to preload job JSON")
    args = parser.parse_args(argv)

    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    run_preload_job(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
