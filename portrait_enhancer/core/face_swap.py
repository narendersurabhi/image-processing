"""Best-frame eye fix: replace a blinked eye region with the same region from an open-eyes
frame in the same burst.

No new ML model -- frames in a confirmed burst (culling.group_bursts) are near-identical
compositions captured a fraction of a second apart, so matching "the same face" across frames
by bounding-box position is reliable. A face-embedding model would be solving a harder problem
(matching identity across arbitrary photos) than the one we actually have here.

Geometry: a similarity transform (rotation + uniform scale + translation) mapping donor pixel
coordinates to target pixel coordinates, fit exactly from the two eye centers (2 point
correspondences = the 4 degrees of freedom of a similarity transform). The donor's eye region
is warped through that transform and seamlessly (Poisson) blended into the target so minor
lighting/color differences between frames don't show a seam.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

EYE_SIDES = ("left", "right")


def _face_center(box) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def match_face_by_position(target_box, candidate_faces, max_center_frac: float = 0.35):
    """Find the face in `candidate_faces` (list of per-face dicts with a "box" key, as
    returned by culling.analyze_faces_detailed) whose box center is closest to
    `target_box`'s center. Burst frames are the same composition, so a true match should be
    near-identical position; a match is only accepted when the offset is small relative to
    the target face's own width -- a large offset means this is a different person (or the
    subject moved enough that swapping would look wrong). Returns the matched face dict, or
    None.
    """
    if not candidate_faces:
        return None
    tx, ty = _face_center(target_box)
    tw = max(1.0, float(target_box[2]))
    best, best_dist = None, None
    for face in candidate_faces:
        cx, cy = _face_center(face["box"])
        dist = math.hypot(cx - tx, cy - ty)
        if best_dist is None or dist < best_dist:
            best, best_dist = face, dist
    if best is None or best_dist > tw * max_center_frac:
        return None
    return best


def _eye_center(guides: dict, side: str):
    upper, lower = guides.get(f"{side}_eye_upper"), guides.get(f"{side}_eye_lower")
    if upper is None or lower is None:
        return None
    return ((upper[0] + lower[0]) / 2.0, (upper[1] + lower[1]) / 2.0)


def compute_eye_alignment(donor_guides: dict, target_guides: dict) -> np.ndarray | None:
    """Similarity transform (2x3 float32 affine) mapping donor pixel coordinates to target
    pixel coordinates, fit exactly from both eye centers: rotation from the interocular
    angle difference, scale from the interocular distance ratio, translation so the left eye
    centers coincide (the right eye centers then coincide too, since 2 point correspondences
    exactly determine a similarity transform's 4 degrees of freedom). Returns None if either
    image is missing an eye center.
    """
    d_left, d_right = _eye_center(donor_guides, "left"), _eye_center(donor_guides, "right")
    t_left, t_right = _eye_center(target_guides, "left"), _eye_center(target_guides, "right")
    if not (d_left and d_right and t_left and t_right):
        return None

    dx_d, dy_d = d_right[0] - d_left[0], d_right[1] - d_left[1]
    dx_t, dy_t = t_right[0] - t_left[0], t_right[1] - t_left[1]
    interocular_d = math.hypot(dx_d, dy_d)
    if interocular_d < 1e-3:
        return None

    scale = math.hypot(dx_t, dy_t) / interocular_d
    theta = math.atan2(dy_t, dx_t) - math.atan2(dy_d, dx_d)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a, b = scale * cos_t, scale * sin_t
    tx = t_left[0] - (a * d_left[0] - b * d_left[1])
    ty = t_left[1] - (b * d_left[0] + a * d_left[1])
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float32)


def _to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    rgb = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def swap_eyes(
    target_img: np.ndarray,
    target_guides: dict,
    target_box,
    donor_img: np.ndarray,
    donor_guides: dict,
    sides=EYE_SIDES,
    patch_scale: float = 0.5,
) -> tuple[np.ndarray | None, list[str]]:
    """Composite the donor's eye region(s) onto a copy of `target_img`, aligned by the
    similarity transform from compute_eye_alignment and blended with Poisson (seamless)
    cloning so the join doesn't show. Both images are float32 RGB in [0, 1]; `target_box` is
    the target face's (x, y, w, h) (used to size the blend region). Returns (result, sides)
    where `sides` lists which eyes were actually swapped ("left"/"right"); result is None if
    neither side could be aligned (missing eye landmarks in either frame).
    """
    M = compute_eye_alignment(donor_guides, target_guides)
    if M is None:
        return None, []

    th, tw = target_img.shape[:2]
    face_w = max(8.0, float(target_box[2]))
    roi_half = face_w * patch_scale * 0.5

    target_bgr = _to_uint8_bgr(target_img)
    donor_bgr = _to_uint8_bgr(donor_img)
    out = target_bgr.copy()
    swapped = []

    for side in sides:
        t_center = _eye_center(target_guides, side)
        if t_center is None:
            continue
        tx, ty = t_center
        rx0 = int(round(tx - roi_half))
        ry0 = int(round(ty - roi_half))
        roi = int(round(roi_half * 2.0))
        rx0c = max(0, min(tw - roi, rx0))
        ry0c = max(0, min(th - roi, ry0))
        if roi <= 1 or rx0c + roi > tw or ry0c + roi > th:
            continue

        m_roi = M.copy()
        m_roi[0, 2] -= rx0c
        m_roi[1, 2] -= ry0c
        warped = cv2.warpAffine(
            donor_bgr, m_roi, (roi, roi), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        mask = np.zeros((roi, roi), dtype=np.uint8)
        cv2.ellipse(
            mask, (roi // 2, roi // 2), (max(1, int(roi * 0.42)), max(1, int(roi * 0.32))),
            0, 0, 360, 255, -1,
        )
        center_in_out = (rx0c + roi // 2, ry0c + roi // 2)
        try:
            out = cv2.seamlessClone(warped, out, mask, center_in_out, cv2.NORMAL_CLONE)
        except cv2.error:
            continue
        swapped.append(side)

    if not swapped:
        return None, []
    result_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return result_rgb, swapped
