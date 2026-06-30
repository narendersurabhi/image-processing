"""Import-time photo culling: blur, blink, and near-duplicate (burst) detection.

Blur scoring and burst grouping are classical CV (OpenCV + numpy), model-free, and cheap
enough to run during the import thumbnail pass. Blink detection is optional and only runs
when the caller supplies a face/landmark segmenter.

Nothing here deletes files -- callers categorize images (e.g. into a 'culled' list); the
review UI keeps the human in the loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .raw_decode import RAW_EXTS

BLINK_ANALYSIS_VERSION = 2


def _to_gray_u8(img: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """Normalize any HxW[x3] float([0,1]) or uint8 image to a small single-channel uint8.
    Downscaling keeps blur/hash cheap and, for blur, makes the score resolution-independent."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = arr.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / float(max(h, w))
        arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return arr


def blur_score(img: np.ndarray) -> float:
    """Sharpness via variance of the Laplacian on a size-normalized gray image. Higher is
    sharper; a soft/out-of-focus or motion-blurred frame scores low. The downscale to a fixed
    long edge makes the number comparable across images of different resolutions."""
    gray = _to_gray_u8(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def average_hash(img: np.ndarray, hash_size: int = 8) -> int:
    """64-bit average hash: downscale to hash_size x hash_size gray, threshold at the mean.
    Near-duplicate frames (same composition) produce hashes a small Hamming distance apart."""
    gray = _to_gray_u8(img, max_dim=max(64, hash_size * 4))
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    bits = (small > small.mean()).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming_distance(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def eye_open_ratio(upper_pt, lower_pt, eye_width: float) -> float:
    """Eye-aspect-ratio-style openness: vertical lid gap normalized by eye width (so it's scale
    invariant). ~0 when shut, larger when open. Mirrors the lid-gap signal the expression warp
    already derives from these same landmarks. Returns -1.0 when it can't be computed."""
    if upper_pt is None or lower_pt is None or eye_width <= 1e-6:
        return -1.0
    gap = abs(float(lower_pt[1]) - float(upper_pt[1]))
    return gap / float(eye_width)


# Eye lid-gap normalized by face-box width. Values below the closed threshold count as a
# blink; values above the open threshold count as confidently open; the middle band remains
# unknown so borderline landmarks do not create false-positive rejects.
EYES_CLOSED_RATIO = 0.025
EYES_OPEN_RATIO = 0.04


def _classify_eye_ratios(
    ratios: list[float],
    closed_ratio: float = EYES_CLOSED_RATIO,
    open_ratio: float = EYES_OPEN_RATIO,
) -> str | None:
    vals = [float(v) for v in ratios if v >= 0]
    if not vals:
        return None
    mean_ratio = float(sum(vals) / len(vals))
    if mean_ratio <= closed_ratio:
        return "closed"
    if mean_ratio >= open_ratio:
        return "open"
    return None


def analyze_faces_detailed(
    segmenter,
    img,
    max_faces: int = 10,
    closed_ratio: float = EYES_CLOSED_RATIO,
    open_ratio: float = EYES_OPEN_RATIO,
):
    """Per-face landmark + eye-state details, for callers that need more than the aggregate
    blink count (e.g. matching/swapping a specific face across burst frames -- see
    core/face_swap.py). `segmenter` is the app's FaceSegmenter (needs `list_faces` and the
    landmark-only `face_guides`). Each face is cropped and upscaled to ~512px before
    landmarking -- the mediapipe landmarker misses small faces in wide group shots otherwise.

    Returns a list of dicts, one per detected face (largest first):
    {"box": (x, y, w, h), "guides": dict|None, "open": True|False|None}. `box` and the points
    inside `guides` are both in `img`'s own pixel coordinates (the crop+upscale used for
    detection is undone before returning), so a caller can use them directly against the
    original-resolution image. `open` is None when eyes couldn't be confidently classified.
    """
    faces = segmenter.list_faces(img)
    if not faces:
        return []
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[:max_faces]
    h_img, w_img = img.shape[:2]
    results = []
    for (x, y, w, h) in faces:
        entry = {"box": (x, y, w, h), "guides": None, "open": None}
        mx, my = int(w * 0.4), int(h * 0.4)
        ox, oy = max(0, x - mx), max(0, y - my)
        crop = img[oy:min(h_img, y + h + my), ox:min(w_img, x + w + mx)]
        if crop.size == 0:
            results.append(entry)
            continue
        scale = 512.0 / max(1.0, float(w))
        if scale > 1.0:
            crop_rs = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                                 interpolation=cv2.INTER_CUBIC)
        else:
            crop_rs, scale = crop, 1.0
        cf = segmenter.list_faces(crop_rs)
        if not cf:
            results.append(entry)
            continue
        bf = max(cf, key=lambda b: b[2] * b[3])
        g = segmenter.face_guides(crop_rs, face_box=bf)
        if not g:
            results.append(entry)
            continue

        # Map crop-local (possibly upscaled) guide points back to img's own coordinates.
        guides_full = {}
        for key, pt in g.items():
            if isinstance(pt, (tuple, list)) and len(pt) == 2:
                guides_full[key] = (pt[0] / scale + ox, pt[1] / scale + oy)
        entry["guides"] = guides_full

        fw = float(bf[2])
        l = eye_open_ratio(g.get("left_eye_upper"), g.get("left_eye_lower"), fw)
        r = eye_open_ratio(g.get("right_eye_upper"), g.get("right_eye_lower"), fw)
        vals = [v for v in (l, r) if v >= 0]
        if vals:
            state = _classify_eye_ratios(vals, closed_ratio=closed_ratio, open_ratio=open_ratio)
            if state is not None:
                entry["open"] = (state == "open")
        results.append(entry)
    return results


def analyze_face_blinks(
    segmenter,
    img,
    max_faces: int = 10,
    closed_ratio: float = EYES_CLOSED_RATIO,
    open_ratio: float = EYES_OPEN_RATIO,
):
    """Count how many faces in `img` have closed eyes (blinking). Thin wrapper over
    analyze_faces_detailed for callers that only need the aggregate count.
    Returns (blinks, analyzed): analyzed=0 means no confidently classified faces (caller
    treats as unknown).
    """
    details = analyze_faces_detailed(
        segmenter, img, max_faces=max_faces, closed_ratio=closed_ratio, open_ratio=open_ratio
    )
    analyzed = [d for d in details if d["open"] is not None]
    blinks = sum(1 for d in analyzed if d["open"] is False)
    return blinks, len(analyzed)


def capture_time(path: str) -> float:
    """Best-effort capture time (epoch seconds) for ordering burst sequences: EXIF
    DateTimeOriginal when readable (JPEG/TIFF via Pillow), else the file mtime. Burst frames
    are written seconds apart, so even mtime is a usable ordering/proximity signal."""
    ext = Path(path).suffix.lower()
    if ext not in RAW_EXTS:
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            with Image.open(path) as im:
                exif = im.getexif()
            for tag_id, value in (exif or {}).items():
                if TAGS.get(tag_id) in ("DateTimeOriginal", "DateTime"):
                    import time as _time

                    return _time.mktime(_time.strptime(str(value), "%Y:%m:%d %H:%M:%S"))
        except Exception:
            pass
    try:
        return float(os.path.getmtime(path))
    except OSError:
        return 0.0


def group_bursts(items, hash_threshold: int = 8, time_window: float = 120.0):
    """Cluster near-duplicate burst frames. `items` is a list of dicts with at least
    'path', 'ahash', and 'time'. Frames are sorted by time, then single-link chained: a frame
    joins the current group if it's within `time_window` seconds of the previous frame AND its
    hash is within `hash_threshold` bits of it. Returns a list of groups (lists of paths),
    preserving order; singletons are length-1 groups.

    Hash distance is the primary signal; `time_window` is deliberately loose because the
    available timestamp is often a file mtime (copy time), not the true sub-second burst
    interval -- so time mainly serves to keep visually-similar frames from *different shooting
    sessions* (minutes/hours apart) out of the same group.
    """
    usable = [it for it in items if it.get("ahash") is not None]
    if not usable:
        return [[it["path"]] for it in items]
    usable.sort(key=lambda it: (it.get("time", 0.0), it["path"]))

    groups = []
    current = [usable[0]]
    for prev, cur in zip(usable, usable[1:]):
        dt = abs(float(cur.get("time", 0.0)) - float(prev.get("time", 0.0)))
        close_in_time = dt <= time_window
        similar = hamming_distance(cur["ahash"], prev["ahash"]) <= hash_threshold
        if close_in_time and similar:
            current.append(cur)
        else:
            groups.append(current)
            current = [cur]
    groups.append(current)
    return [[it["path"] for it in g] for g in groups]


def pick_keeper(group_paths, quality: dict):
    """Choose the best frame in a burst group: fewest blinkers first, then sharpest.
    `quality` maps path -> {'blur': float, 'blinks': int|None, 'eyes_open': bool|None}.
    When blink counts are known (group analyzed), the frame with the fewest closed eyes wins;
    otherwise it falls back to the eyes_open flag, then to unknown (neutral). Returns the
    chosen path."""
    if not group_paths:
        return None

    def key(p):
        q = quality.get(p, {})
        blinks = q.get("blinks")
        if blinks is not None:
            blink_rank = -float(blinks)  # fewer blinkers ranks higher
        else:
            # No per-face analysis: use the coarse flag (open > unknown > closed).
            blink_rank = {True: 0.0, None: -0.5, False: -1.0}.get(q.get("eyes_open"), -0.5)
        return (blink_rank, float(q.get("blur", 0.0)))

    return max(group_paths, key=key)
