"""Image processing functions for global and selective portrait layers."""

import numpy as np
import cv2
from PIL import Image, ImageEnhance

from portrait_enhancer.config import MASK_ORDER
from .refine import get_face_refiner
from .white_balance import NEUTRAL_K, apply_white_balance_gains, kelvin_tint_to_gains
from .tone_curve import apply_curve
from .color_mixer import apply_color_mixer
from .utils import (
    adjust_color_balance_preserve_chroma,
    adjust_hsv_hue,
    adjust_hsv_sat,
    adjust_warmth_preserve_hue,
    apply_tone_curve,
    apply_tone_curve_preserve_chroma,
    blend_with_mask,
    clamp01,
    display_to_working,
    gaussian_blur,
    smooth_mask,
    to_float,
    to_uint8,
    working_to_display,
)


def _working_space(color_settings: dict | None) -> str:
    return str((color_settings or {}).get("working_space", "srgb"))


def _output_transform(color_settings: dict | None) -> str:
    return str((color_settings or {}).get("output_transform", "srgb"))


def _acceleration_mode(runtime_settings: dict | None) -> str:
    return str((runtime_settings or {}).get("acceleration_mode", "auto"))


def _fast_interactive_preview(runtime_settings: dict | None) -> bool:
    return bool((runtime_settings or {}).get("fast_interactive_preview", False))


def _apply_sharpness(img: np.ndarray, factor: float, working_space: str) -> np.ndarray:
    display_img = working_to_display(img, output_transform="srgb", working_space=working_space)
    pil = Image.fromarray(to_uint8(display_img))
    pil = ImageEnhance.Sharpness(pil).enhance(factor)
    return display_to_working(to_float(np.array(pil)), working_space=working_space)


def _frequency_smooth_skin(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    # Two-band split: preserve higher-frequency pore/edge structure while
    # smoothing only the broader low-frequency skin transitions.
    low_sigma = 1.5 + amount * 6.0
    detail_sigma = 0.8 + amount * 1.8
    low_band = gaussian_blur(img, sigma=low_sigma, acceleration=acceleration)
    detail_band = img - gaussian_blur(img, sigma=detail_sigma, acceleration=acceleration)

    detail_mix = 1.0 - amount * 0.45
    smoothed = low_band + detail_band * detail_mix

    # Recover stronger edges from a mild unsharp residual so skin stays soft
    # without collapsing facial boundaries.
    edge_residual = img - gaussian_blur(img, sigma=2.2 + amount * 1.8, acceleration=acceleration)
    edge_weight = 0.10 + amount * 0.12
    return clamp01(smoothed + edge_residual * edge_weight)


def _frequency_blemish_soften(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    base = gaussian_blur(img, sigma=2.0 + amount * 4.5, acceleration=acceleration)
    coarse = gaussian_blur(img, sigma=5.0 + amount * 7.0, acceleration=acceleration)
    fine_detail = img - base
    fine_strength = 1.0 - amount * 0.75
    return clamp01(coarse + fine_detail * fine_strength)


def _luma(img: np.ndarray) -> np.ndarray:
    return np.clip(
        img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722,
        0.0,
        1.0,
    ).astype(np.float32)


def _edge_protection_mask(
    img: np.ndarray,
    *,
    acceleration: str = "auto",
    strength: float = 2.5,
    blur_sigma: float = 1.2,
) -> np.ndarray:
    lum = _luma(img)
    gx = np.zeros_like(lum, dtype=np.float32)
    gy = np.zeros_like(lum, dtype=np.float32)
    gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) * 0.5
    gy[1:-1, :] = (lum[2:, :] - lum[:-2, :]) * 0.5
    edge = np.sqrt(gx * gx + gy * gy)
    if blur_sigma > 0:
        edge = gaussian_blur(edge, sigma=blur_sigma, acceleration=acceleration)
    return np.clip(1.0 - edge * float(strength), 0.0, 1.0).astype(np.float32)


def _mix_with_effect_mask(base: np.ndarray, effected: np.ndarray, effect_mask: np.ndarray) -> np.ndarray:
    mask3 = np.clip(effect_mask, 0.0, 1.0)[:, :, np.newaxis]
    return clamp01(base * (1.0 - mask3) + effected * mask3)


def _mask_bbox(mask: np.ndarray, threshold: float = 0.10) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(np.clip(mask.astype(np.float32), 0.0, 1.0) > float(threshold))
    if coords.size == 0:
        return None
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1
    return int(x1), int(y1), int(x2), int(y2)


def _gaussian_blob(shape_hw: tuple[int, int], cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    h, w = shape_hw
    if sx <= 0 or sy <= 0 or h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    field = np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
    return field.astype(np.float32)


def _remap_array(arr: np.ndarray, dx: np.ndarray, dy: np.ndarray, interpolation: int) -> np.ndarray:
    h, w = arr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x - dx.astype(np.float32), 0.0, max(0, w - 1)).astype(np.float32)
    map_y = np.clip(grid_y - dy.astype(np.float32), 0.0, max(0, h - 1)).astype(np.float32)
    remapped = cv2.remap(arr.astype(np.float32), map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_REFLECT_101)
    return remapped.astype(np.float32)


def _point_from_guides(guides: dict | None, key: str) -> tuple[float, float] | None:
    if not guides or key not in guides:
        return None
    value = guides.get(key)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _scale_expression_guides(guides, scale_x: float, scale_y: float):
    if guides is None:
        return None
    if isinstance(guides, list):
        return [_scale_expression_guides(item, scale_x, scale_y) for item in guides]
    scaled = {}
    for key, value in guides.items():
        if isinstance(value, (tuple, list)) and len(value) == 2:
            scaled[key] = (float(value[0]) * scale_x, float(value[1]) * scale_y)
        else:
            scaled[key] = value
    return scaled


def expression_warp_mode(face_params: dict | None, geometry: dict | list[dict] | None) -> str:
    if not face_params:
        return "off"
    active = any(
        abs(float(face_params.get(key, 0.0))) > 1e-4
        for key in ("smile", "eye_open", "brow_lift", "mouth_open", "jaw_relax")
    )
    if not active:
        return "off"
    return "landmark-guided" if geometry else "mask-guided"


def _apply_expression_warp(
    img: np.ndarray,
    masks: dict | None,
    face_params: dict | None,
    geometry: dict | list[dict] | None = None,
) -> tuple[np.ndarray, dict | None]:
    if masks is None or not face_params:
        return img, masks

    smile = float(face_params.get("smile", 0.0)) / 100.0
    eye_open = float(face_params.get("eye_open", 0.0)) / 100.0
    brow_lift = float(face_params.get("brow_lift", 0.0)) / 100.0
    mouth_open = float(face_params.get("mouth_open", 0.0)) / 100.0
    jaw_relax = float(face_params.get("jaw_relax", 0.0)) / 100.0
    if abs(smile) < 1e-4 and abs(eye_open) < 1e-4 and abs(brow_lift) < 1e-4 and abs(mouth_open) < 1e-4 and abs(jaw_relax) < 1e-4:
        return img, masks

    face_mask = np.clip(np.asarray((masks or {}).get("face", 0.0), dtype=np.float32), 0.0, 1.0)
    face_box = _mask_bbox(face_mask, threshold=0.08)
    if face_box is None:
        return img, masks

    h, w = img.shape[:2]
    fx1, fy1, fx2, fy2 = face_box
    fw = max(1.0, float(fx2 - fx1))
    fh = max(1.0, float(fy2 - fy1))
    face_support = smooth_mask(face_mask, sigma=max(1.5, fw * 0.015))
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    if isinstance(geometry, list):
        geometries = [item for item in geometry if isinstance(item, dict)]
    elif isinstance(geometry, dict):
        geometries = [geometry]
    else:
        geometries = []

    lips_mask = np.clip(np.asarray((masks or {}).get("lips", 0.0), dtype=np.float32), 0.0, 1.0)
    lips_box = _mask_bbox(lips_mask, threshold=0.08)
    if lips_box is None:
        lips_box = (
            int(fx1 + fw * 0.28),
            int(fy1 + fh * 0.64),
            int(fx1 + fw * 0.72),
            int(fy1 + fh * 0.84),
        )
        lips_mask = _gaussian_blob((h, w), (lips_box[0] + lips_box[2]) * 0.5, (lips_box[1] + lips_box[3]) * 0.5, fw * 0.16, fh * 0.09)

    if abs(smile) >= 1e-4:
        guide = geometries[0] if geometries else None
        left_pt = _point_from_guides(guide, "mouth_left")
        right_pt = _point_from_guides(guide, "mouth_right")
        upper_pt = _point_from_guides(guide, "mouth_upper")
        lower_pt = _point_from_guides(guide, "mouth_lower")
        center_pt = _point_from_guides(guide, "mouth_center")

        if left_pt and right_pt and upper_pt and lower_pt:
            lip_w = max(1.0, abs(right_pt[0] - left_pt[0]))
            lip_h = max(1.0, abs(lower_pt[1] - upper_pt[1]))
            if center_pt is None:
                center_pt = (
                    (left_pt[0] + right_pt[0] + upper_pt[0] + lower_pt[0]) * 0.25,
                    (left_pt[1] + right_pt[1] + upper_pt[1] + lower_pt[1]) * 0.25,
                )
            left_corner = _gaussian_blob((h, w), left_pt[0], left_pt[1], max(2.0, lip_w * 0.34), max(2.0, lip_h * 1.8))
            right_corner = _gaussian_blob((h, w), right_pt[0], right_pt[1], max(2.0, lip_w * 0.34), max(2.0, lip_h * 1.8))
            mouth_center = _gaussian_blob((h, w), center_pt[0], center_pt[1], max(2.0, lip_w * 0.45), max(2.0, lip_h * 2.2))
        else:
            lx1, ly1, lx2, ly2 = lips_box
            lip_w = max(1.0, float(lx2 - lx1))
            lip_h = max(1.0, float(ly2 - ly1))
            left_corner = _gaussian_blob((h, w), lx1 + lip_w * 0.10, ly1 + lip_h * 0.45, fw * 0.11, fh * 0.10)
            right_corner = _gaussian_blob((h, w), lx2 - lip_w * 0.10, ly1 + lip_h * 0.45, fw * 0.11, fh * 0.10)
            mouth_center = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly1 + lip_h * 0.60, fw * 0.15, fh * 0.09)
        smile_support = smooth_mask(np.clip((lips_mask * 0.6 + mouth_center * 0.4) * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.006))
        corner_lift = fh * 0.110 * smile
        corner_out = fw * 0.040 * smile
        center_drop = fh * 0.030 * smile
        dy += (-corner_lift * left_corner) + (-corner_lift * right_corner) + (center_drop * mouth_center)
        dx += (-corner_out * left_corner) + (corner_out * right_corner)
        cheek_pull_left = _gaussian_blob((h, w), fx1 + fw * 0.18, fy1 + fh * 0.58, fw * 0.12, fh * 0.16)
        cheek_pull_right = _gaussian_blob((h, w), fx2 - fw * 0.18, fy1 + fh * 0.58, fw * 0.12, fh * 0.16)
        dx += (-fw * 0.015 * smile * cheek_pull_left) + (fw * 0.015 * smile * cheek_pull_right)
        dy *= np.clip(0.35 + smile_support * 0.65, 0.0, 1.0)
        dx *= np.clip(0.30 + smile_support * 0.70, 0.0, 1.0)

    if abs(mouth_open) >= 1e-4 or abs(jaw_relax) >= 1e-4:
        guide = geometries[0] if geometries else None
        upper_pt = _point_from_guides(guide, "mouth_upper")
        lower_pt = _point_from_guides(guide, "mouth_lower")
        center_pt = _point_from_guides(guide, "mouth_center")

        if upper_pt and lower_pt:
            mouth_w = max(2.0, fw * 0.18)
            mouth_h = max(2.0, abs(lower_pt[1] - upper_pt[1]) * 2.4)
            upper_blob = _gaussian_blob((h, w), upper_pt[0], upper_pt[1], mouth_w, mouth_h)
            lower_blob = _gaussian_blob((h, w), lower_pt[0], lower_pt[1], mouth_w, mouth_h * 1.1)
            center = center_pt if center_pt else ((upper_pt[0] + lower_pt[0]) * 0.5, (upper_pt[1] + lower_pt[1]) * 0.5)
            jaw_blob = _gaussian_blob((h, w), center[0], lower_pt[1] + fh * 0.11, fw * 0.28, fh * 0.18)
        else:
            lx1, ly1, lx2, ly2 = lips_box
            lip_w = max(1.0, float(lx2 - lx1))
            lip_h = max(1.0, float(ly2 - ly1))
            upper_blob = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly1 + lip_h * 0.32, lip_w * 0.55, lip_h * 0.90)
            lower_blob = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly1 + lip_h * 0.78, lip_w * 0.55, lip_h * 0.95)
            jaw_blob = _gaussian_blob((h, w), (lx1 + lx2) * 0.5, ly2 + fh * 0.10, fw * 0.28, fh * 0.18)

        mouth_support = smooth_mask(np.clip((lips_mask * 0.7 + jaw_blob * 0.3) * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.008))
        open_push = fh * 0.060 * mouth_open
        jaw_push = fh * 0.050 * jaw_relax
        dy += (-open_push * 0.35 * upper_blob + open_push * lower_blob) * mouth_support
        dy += jaw_push * jaw_blob * mouth_support
        dx += jaw_relax * fw * 0.010 * (_gaussian_blob((h, w), fx1 + fw * 0.26, fy2 - fh * 0.10, fw * 0.10, fh * 0.14) * -1.0)
        dx += jaw_relax * fw * 0.010 * (_gaussian_blob((h, w), fx2 - fw * 0.26, fy2 - fh * 0.10, fw * 0.10, fh * 0.14))

    eyes_mask = np.clip(np.asarray((masks or {}).get("eyes", 0.0), dtype=np.float32), 0.0, 1.0)
    eyes_box = _mask_bbox(eyes_mask, threshold=0.08)
    if abs(eye_open) >= 1e-4:
        eye_boxes: list[tuple[int, int, int, int]] = []
        eye_guides = []
        guide = geometries[0] if geometries else None
        if guide:
            for prefix in ("left", "right"):
                upper_pt = _point_from_guides(guide, f"{prefix}_eye_upper")
                lower_pt = _point_from_guides(guide, f"{prefix}_eye_lower")
                center_pt = _point_from_guides(guide, f"{prefix}_eye_center")
                if upper_pt and lower_pt:
                    if center_pt is None:
                        center_pt = ((upper_pt[0] + lower_pt[0]) * 0.5, (upper_pt[1] + lower_pt[1]) * 0.5)
                    eye_guides.append((upper_pt, lower_pt, center_pt))
        if eye_guides:
            for upper_pt, lower_pt, center_pt in eye_guides:
                ew = max(2.0, fw * 0.14)
                eh = max(2.0, abs(lower_pt[1] - upper_pt[1]) * 2.0)
                upper = _gaussian_blob((h, w), upper_pt[0], upper_pt[1], ew, eh)
                lower = _gaussian_blob((h, w), lower_pt[0], lower_pt[1], ew, eh)
                eye_support = smooth_mask(np.clip((upper + lower) * face_support, 0.0, 1.0), sigma=max(0.6, ew * 0.05))
                eye_push = max(1.0, abs(lower_pt[1] - upper_pt[1]) * 1.8) * eye_open
                dy += (-eye_push * upper + eye_push * lower) * eye_support
        elif eyes_box is not None:
            ex1, ey1, ex2, ey2 = eyes_box
            mid_x = fx1 + fw * 0.5
            left_mask = eyes_mask.copy()
            left_mask[:, int(mid_x):] = 0.0
            right_mask = eyes_mask.copy()
            right_mask[:, :int(mid_x)] = 0.0
            left_box = _mask_bbox(left_mask, threshold=0.08)
            right_box = _mask_bbox(right_mask, threshold=0.08)
            eye_boxes = [b for b in (left_box, right_box) if b is not None]
        if not eye_boxes:
            eye_boxes = [
                (int(fx1 + fw * 0.18), int(fy1 + fh * 0.24), int(fx1 + fw * 0.42), int(fy1 + fh * 0.44)),
                (int(fx1 + fw * 0.58), int(fy1 + fh * 0.24), int(fx1 + fw * 0.82), int(fy1 + fh * 0.44)),
            ]
        if not eye_guides:
            for ex1, ey1, ex2, ey2 in eye_boxes:
                ew = max(1.0, float(ex2 - ex1))
                eh = max(1.0, float(ey2 - ey1))
                cx = (ex1 + ex2) * 0.5
                upper = _gaussian_blob((h, w), cx, ey1 + eh * 0.28, ew * 0.42, eh * 0.32)
                lower = _gaussian_blob((h, w), cx, ey1 + eh * 0.72, ew * 0.42, eh * 0.32)
                eye_support = smooth_mask(np.clip((upper + lower) * face_support, 0.0, 1.0), sigma=max(0.6, ew * 0.03))
                eye_push = eh * 0.32 * eye_open
                dy += (-eye_push * upper + eye_push * lower) * eye_support

    if abs(brow_lift) >= 1e-4:
        brow_guides = []
        guide = geometries[0] if geometries else None
        if guide:
            for brow_key, eye_key in (("left_brow", "left_eye_center"), ("right_brow", "right_eye_center")):
                brow_pt = _point_from_guides(guide, brow_key)
                eye_pt = _point_from_guides(guide, eye_key)
                if brow_pt and eye_pt:
                    brow_guides.append((brow_pt, eye_pt))
        if not brow_guides:
            brow_guides = [
                ((fx1 + fw * 0.32, fy1 + fh * 0.20), (fx1 + fw * 0.32, fy1 + fh * 0.34)),
                ((fx1 + fw * 0.68, fy1 + fh * 0.20), (fx1 + fw * 0.68, fy1 + fh * 0.34)),
            ]

        for brow_pt, eye_pt in brow_guides:
            span_y = max(2.0, abs(eye_pt[1] - brow_pt[1]))
            span_x = max(2.0, fw * 0.12)
            brow_blob = _gaussian_blob((h, w), brow_pt[0], brow_pt[1], span_x, span_y * 1.4)
            forehead_blob = _gaussian_blob((h, w), brow_pt[0], brow_pt[1] - span_y * 0.55, span_x * 1.1, span_y * 1.8)
            brow_support = smooth_mask(np.clip((brow_blob * 0.7 + forehead_blob * 0.3) * face_support, 0.0, 1.0), sigma=max(0.8, span_x * 0.08))
            lift = max(1.0, span_y * 0.9) * brow_lift
            dy += (-lift * brow_blob - lift * 0.4 * forehead_blob) * brow_support

    support = np.clip(face_support, 0.0, 1.0)
    dx *= support
    dy *= support
    max_dx = fw * 0.14
    max_dy = fh * 0.14
    dx = np.clip(dx, -max_dx, max_dx)
    dy = np.clip(dy, -max_dy, max_dy)

    warped_img = clamp01(_remap_array(img, dx, dy, cv2.INTER_LINEAR))
    warped_masks = {}
    for key, value in (masks or {}).items():
        warped = _remap_array(np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0), dx, dy, cv2.INTER_LINEAR)
        warped_masks[key] = np.clip(warped, 0.0, 1.0).astype(np.float32)
    return warped_img, warped_masks


def _apply_face_refinement(
    img: np.ndarray,
    masks: dict | None,
    face_params: dict | None,
    *,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    if masks is None or not face_params:
        return img
    if _fast_interactive_preview(runtime_settings):
        return img
    amount = float(face_params.get("refine", 0.0)) / 100.0
    if amount <= 0.0:
        return img
    fidelity = float(face_params.get("refine_fidelity", 65.0)) / 100.0
    face_mask = (masks or {}).get("face")
    if face_mask is None:
        return img
    working_space = _working_space(color_settings)
    display_img = working_to_display(img, output_transform="srgb", working_space=working_space)
    refined = get_face_refiner().refine(display_img, face_mask, amount, fidelity=fidelity)
    return display_to_working(refined, working_space=working_space)


def _build_skin_protection_mask(
    img: np.ndarray,
    masks: dict | None,
    geometry: dict | list[dict] | None = None,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    if not masks or "face" not in masks:
        return np.zeros(img.shape[:2], dtype=np.float32)

    face_mask = np.clip(np.asarray(masks.get("face", 0.0), dtype=np.float32), 0.0, 1.0)
    if float(face_mask.max()) <= 0.01:
        return np.zeros(img.shape[:2], dtype=np.float32)

    face_box = _mask_bbox(face_mask, threshold=0.08)
    if face_box is None:
        return np.zeros(img.shape[:2], dtype=np.float32)

    h, w = img.shape[:2]
    fx1, fy1, fx2, fy2 = face_box
    fw = max(1.0, float(fx2 - fx1))
    fh = max(1.0, float(fy2 - fy1))
    face_support = smooth_mask(face_mask, sigma=max(1.0, fw * 0.012), acceleration=acceleration)

    if isinstance(geometry, list):
        guides = next((item for item in geometry if isinstance(item, dict)), None)
    elif isinstance(geometry, dict):
        guides = geometry
    else:
        guides = None

    lum = _luma(img)
    dark = np.clip((0.52 - lum) / 0.36, 0.0, 1.0).astype(np.float32)
    texture = np.abs(lum - gaussian_blur(lum, sigma=1.4, acceleration=acceleration))
    texture = np.clip(texture * 5.0, 0.0, 1.0).astype(np.float32)
    hairlike = np.clip(dark * 0.8 + texture * 0.7, 0.0, 1.0).astype(np.float32)

    protection = np.zeros((h, w), dtype=np.float32)

    brow_mask = np.clip(np.asarray((masks or {}).get("brows", 0.0), dtype=np.float32), 0.0, 1.0)
    if float(brow_mask.max()) > 0.01:
        protection = np.maximum(
            protection,
            smooth_mask(np.clip(brow_mask * face_support, 0.0, 1.0), sigma=max(0.6, fw * 0.006), acceleration=acceleration),
        )

    facial_hair_mask = np.clip(np.asarray((masks or {}).get("facial_hair", 0.0), dtype=np.float32), 0.0, 1.0)
    if float(facial_hair_mask.max()) > 0.01:
        protection = np.maximum(
            protection,
            smooth_mask(np.clip(facial_hair_mask * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.008), acceleration=acceleration),
        )

    brow_guides = []
    if guides:
        for brow_key, eye_key in (("left_brow", "left_eye_center"), ("right_brow", "right_eye_center")):
            brow_pt = _point_from_guides(guides, brow_key)
            eye_pt = _point_from_guides(guides, eye_key)
            if brow_pt and eye_pt:
                brow_guides.append((brow_pt, eye_pt))
    if not brow_guides:
        brow_guides = [
            ((fx1 + fw * 0.32, fy1 + fh * 0.20), (fx1 + fw * 0.32, fy1 + fh * 0.34)),
            ((fx1 + fw * 0.68, fy1 + fh * 0.20), (fx1 + fw * 0.68, fy1 + fh * 0.34)),
        ]
    for brow_pt, eye_pt in brow_guides:
        span_y = max(2.0, abs(eye_pt[1] - brow_pt[1]))
        span_x = max(2.0, fw * 0.11)
        brow_blob = _gaussian_blob((h, w), brow_pt[0], brow_pt[1], span_x, span_y * 1.2)
        protection = np.maximum(protection, brow_blob * np.clip(hairlike * 0.85 + 0.15, 0.0, 1.0))

    mouth_center = _point_from_guides(guides, "mouth_center") if guides else None
    mouth_left = _point_from_guides(guides, "mouth_left") if guides else None
    mouth_right = _point_from_guides(guides, "mouth_right") if guides else None
    mouth_upper = _point_from_guides(guides, "mouth_upper") if guides else None
    mouth_lower = _point_from_guides(guides, "mouth_lower") if guides else None

    if mouth_center and mouth_upper and mouth_lower:
        lip_w = max(2.0, abs((mouth_right or mouth_center)[0] - (mouth_left or mouth_center)[0]))
        lip_h = max(2.0, abs(mouth_lower[1] - mouth_upper[1]))
        moustache_blob = _gaussian_blob(
            (h, w),
            mouth_center[0],
            mouth_upper[1] - lip_h * 0.65,
            max(2.0, lip_w * 0.55),
            max(2.0, lip_h * 0.95),
        )
        beard_blob = _gaussian_blob(
            (h, w),
            mouth_center[0],
            mouth_lower[1] + fh * 0.16,
            max(2.0, lip_w * 0.75),
            max(2.0, fh * 0.18),
        )
    else:
        moustache_blob = _gaussian_blob((h, w), fx1 + fw * 0.50, fy1 + fh * 0.63, fw * 0.19, fh * 0.06)
        beard_blob = _gaussian_blob((h, w), fx1 + fw * 0.50, fy1 + fh * 0.82, fw * 0.26, fh * 0.16)

    facial_hair = np.clip((moustache_blob * 0.9 + beard_blob) * hairlike * face_support, 0.0, 1.0)
    protection = np.maximum(protection, facial_hair)

    protection = smooth_mask(np.clip(protection * face_support, 0.0, 1.0), sigma=max(0.8, fw * 0.010), acceleration=acceleration)
    return np.clip(protection, 0.0, 1.0).astype(np.float32)


def _confidence_curve(mask: np.ndarray, power: float = 1.0, floor: float = 0.0) -> np.ndarray:
    base = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if floor != 0.0:
        base = np.clip((base - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    if power != 1.0:
        base = np.power(base, float(power))
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _make_effect_masks(mask: np.ndarray, layer: str, *, acceleration: str = "auto") -> dict[str, np.ndarray]:
    base = np.clip(mask.astype(np.float32), 0.0, 1.0)
    broad = smooth_mask(_confidence_curve(base, power=0.8), sigma=1.0, acceleration=acceleration)
    blend = smooth_mask(_confidence_curve(base, power=1.1), sigma=0.8, acceleration=acceleration)
    detail = smooth_mask(_confidence_curve(base, power=1.7, floor=0.10), sigma=0.6, acceleration=acceleration)
    strict = smooth_mask(_confidence_curve(base, power=2.4, floor=0.18), sigma=0.5, acceleration=acceleration)

    if layer == "skin":
        broad = smooth_mask(_confidence_curve(base, power=0.7), sigma=1.2, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=0.95), sigma=0.9, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.5, floor=0.08), sigma=0.7, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.0, floor=0.14), sigma=0.6, acceleration=acceleration)
    elif layer == "background":
        broad = smooth_mask(_confidence_curve(base, power=0.65), sigma=1.6, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=0.85), sigma=1.2, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.25, floor=0.04), sigma=0.9, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=1.7, floor=0.08), sigma=0.8, acceleration=acceleration)
    elif layer == "subjects":
        broad = smooth_mask(_confidence_curve(base, power=0.75), sigma=1.2, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=0.95), sigma=1.0, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.4, floor=0.06), sigma=0.8, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=1.9, floor=0.12), sigma=0.7, acceleration=acceleration)
    elif layer == "face":
        broad = smooth_mask(_confidence_curve(base, power=0.75), sigma=1.1, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=1.0), sigma=0.9, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.6, floor=0.10), sigma=0.7, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.1, floor=0.16), sigma=0.6, acceleration=acceleration)
    elif layer in {"eyes", "lips"}:
        broad = smooth_mask(_confidence_curve(base, power=1.2, floor=0.06), sigma=0.8, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=1.5, floor=0.10), sigma=0.7, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=2.1, floor=0.16), sigma=0.5, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.8, floor=0.22), sigma=0.4, acceleration=acceleration)
    elif layer == "hair":
        broad = smooth_mask(_confidence_curve(base, power=0.9, floor=0.03), sigma=1.0, acceleration=acceleration)
        blend = smooth_mask(_confidence_curve(base, power=1.1, floor=0.06), sigma=0.9, acceleration=acceleration)
        detail = smooth_mask(_confidence_curve(base, power=1.8, floor=0.12), sigma=0.6, acceleration=acceleration)
        strict = smooth_mask(_confidence_curve(base, power=2.2, floor=0.16), sigma=0.5, acceleration=acceleration)

    return {
        "blend": np.clip(blend, 0.0, 1.0).astype(np.float32),
        "broad": np.clip(broad, 0.0, 1.0).astype(np.float32),
        "detail": np.clip(detail, 0.0, 1.0).astype(np.float32),
        "strict": np.clip(strict, 0.0, 1.0).astype(np.float32),
    }


def _apply_micro_contrast(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
    detail_sigma: float = 1.6,
    base_sigma: float = 6.0,
    edge_strength: float = 2.2,
) -> np.ndarray:
    amount = float(amount) / 100.0
    if amount == 0.0:
        return img

    fine_base = gaussian_blur(img, sigma=detail_sigma, acceleration=acceleration)
    coarse_base = gaussian_blur(img, sigma=base_sigma, acceleration=acceleration)
    fine_detail = img - fine_base
    local_contrast = fine_base - coarse_base

    edge_guard = _edge_protection_mask(img, acceleration=acceleration, strength=edge_strength, blur_sigma=1.0)
    effect_mask = smooth_mask(np.clip(0.35 + edge_guard * 0.65, 0.0, 1.0), sigma=0.8, acceleration=acceleration)

    enhanced = clamp01(
        img
        + local_contrast * amount * 0.85
        + fine_detail * amount * 0.35
    )
    if amount < 0.0:
        enhanced = clamp01(
            img
            + local_contrast * amount * 0.65
            + fine_detail * amount * 0.20
        )
    return _mix_with_effect_mask(img, enhanced, effect_mask)


def _hair_strand_mask(
    img: np.ndarray,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    lum = _luma(img)
    fine = np.abs(lum - gaussian_blur(lum, sigma=1.1, acceleration=acceleration))
    coarse = np.abs(lum - gaussian_blur(lum, sigma=3.2, acceleration=acceleration))
    strands = np.clip(fine * 5.0 + coarse * 2.0, 0.0, 1.0)
    strands = smooth_mask(strands, sigma=0.8, acceleration=acceleration)
    return np.clip(strands, 0.0, 1.0).astype(np.float32)


def _apply_hair_shine(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    lum = _luma(img)
    strands = _hair_strand_mask(img, acceleration=acceleration)
    base_highlights = np.clip((lum - 0.35) / 0.45, 0.0, 1.0)
    directional = np.clip(gaussian_blur(lum, sigma=2.0, acceleration=acceleration) - gaussian_blur(lum, sigma=5.5, acceleration=acceleration), 0.0, 1.0)
    shine_mask = smooth_mask(np.clip(base_highlights * 0.55 + strands * 0.55 + directional * 0.35, 0.0, 1.0), sigma=1.0, acceleration=acceleration)

    boosted = clamp01(img + img * shine_mask[:, :, np.newaxis] * (0.10 + amount * 0.28))
    bloom = gaussian_blur(boosted * shine_mask[:, :, np.newaxis], sigma=2.4, acceleration=acceleration)
    blended = clamp01(boosted + bloom * amount * 0.18)
    return _mix_with_effect_mask(img, blended, shine_mask)


def _apply_hair_depth(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(amount) / 100.0
    if amount == 0.0:
        return img

    lum = _luma(img)
    strands = _hair_strand_mask(img, acceleration=acceleration)
    shadow_band = np.clip((0.62 - lum) / 0.42, 0.0, 1.0)
    depth_mask = smooth_mask(np.clip(shadow_band * 0.7 + strands * 0.4, 0.0, 1.0), sigma=1.0, acceleration=acceleration)
    darker = clamp01(img * (1.0 - depth_mask[:, :, np.newaxis] * amount * 0.18))
    lighter = clamp01(img + depth_mask[:, :, np.newaxis] * amount * 0.10)
    target = darker if amount > 0 else lighter
    return _mix_with_effect_mask(img, target, depth_mask)


def _even_skin_chroma(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    rgb8 = to_uint8(img)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    a_base = gaussian_blur(a_chan, sigma=2.0 + amount * 5.0, acceleration=acceleration)
    b_base = gaussian_blur(b_chan, sigma=2.0 + amount * 5.0, acceleration=acceleration)
    chroma_diff = np.sqrt((a_chan - a_base) ** 2 + (b_chan - b_base) ** 2)

    even_mask = np.clip(chroma_diff / 14.0, 0.0, 1.0)
    even_mask *= np.clip(1.0 - np.abs((l_chan / 255.0) - 0.58) * 1.6, 0.25, 1.0)
    even_mask = smooth_mask(even_mask.astype(np.float32), sigma=1.0, acceleration=acceleration)

    a_even = a_chan * (1.0 - even_mask * amount * 0.55) + a_base * (even_mask * amount * 0.55)
    b_even = b_chan * (1.0 - even_mask * amount * 0.55) + b_base * (even_mask * amount * 0.55)

    out_lab = lab.copy()
    out_lab[:, :, 1] = np.clip(a_even, 0, 255)
    out_lab[:, :, 2] = np.clip(b_even, 0, 255)
    out_rgb = to_float(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return clamp01(out_rgb)


def _apply_eye_whitening(
    img: np.ndarray,
    amount: float,
    *,
    edge_guard: np.ndarray,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    rgb8 = to_uint8(img)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    lum = _luma(img)
    sclera_mask = np.clip((lum - 0.48) / 0.30, 0.0, 1.0)
    sclera_mask *= np.clip(1.0 - np.abs((a_chan - 128.0) / 30.0), 0.2, 1.0)
    sclera_mask *= edge_guard
    sclera_mask = smooth_mask(sclera_mask.astype(np.float32), sigma=0.8, acceleration=acceleration)

    l_target = np.clip(l_chan + (255.0 - l_chan) * amount * 0.18 * sclera_mask, 0, 255)
    a_target = a_chan * (1.0 - amount * 0.35 * sclera_mask) + 128.0 * (amount * 0.35 * sclera_mask)
    b_target = b_chan * (1.0 - amount * 0.45 * sclera_mask) + 128.0 * (amount * 0.45 * sclera_mask)

    out_lab = lab.copy()
    out_lab[:, :, 0] = l_target
    out_lab[:, :, 1] = np.clip(a_target, 0, 255)
    out_lab[:, :, 2] = np.clip(b_target, 0, 255)
    out_rgb = to_float(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return clamp01(out_rgb)


def _unify_face_tone(
    img: np.ndarray,
    amount: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0.0:
        return img

    rgb8 = to_uint8(img)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    l_base = gaussian_blur(l_chan, sigma=3.5 + amount * 8.0, acceleration=acceleration)
    a_base = gaussian_blur(a_chan, sigma=3.0 + amount * 6.0, acceleration=acceleration)
    b_base = gaussian_blur(b_chan, sigma=3.0 + amount * 6.0, acceleration=acceleration)

    lum = _luma(img)
    midtone_focus = np.clip(1.0 - np.abs(lum - 0.55) * 1.8, 0.2, 1.0)
    edge_guard = _edge_protection_mask(img, acceleration=acceleration, strength=2.8, blur_sigma=1.2)
    unify_mask = smooth_mask(np.clip(midtone_focus * edge_guard, 0.0, 1.0), sigma=1.1, acceleration=acceleration)

    l_out = l_chan * (1.0 - unify_mask * amount * 0.32) + l_base * (unify_mask * amount * 0.32)
    a_out = a_chan * (1.0 - unify_mask * amount * 0.24) + a_base * (unify_mask * amount * 0.24)
    b_out = b_chan * (1.0 - unify_mask * amount * 0.24) + b_base * (unify_mask * amount * 0.24)

    out_lab = lab.copy()
    out_lab[:, :, 0] = np.clip(l_out, 0, 255)
    out_lab[:, :, 1] = np.clip(a_out, 0, 255)
    out_lab[:, :, 2] = np.clip(b_out, 0, 255)
    out_rgb = to_float(cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return clamp01(out_rgb)


def process_global(
    img: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    img = img.copy()

    # White balance correction first: neutralize a color cast in linear light
    # before any tonal/stylistic grade is applied. Temperature (Kelvin) + Tint are
    # the canonical setting; default 6500K/0 is identity.
    cs = color_settings or {}
    wb_gains = kelvin_tint_to_gains(cs.get("wb_temp_k", NEUTRAL_K), cs.get("wb_tint", 0))
    img = apply_white_balance_gains(img, wb_gains, working_space=working_space)

    ev = p.get("exposure", 0) / 100.0
    img = clamp01(img * (2**ev))

    img = adjust_color_balance_preserve_chroma(
        img,
        temperature=p.get("temperature", 0),
        tint=p.get("tint", 0),
        working_space=working_space,
    )

    img = apply_tone_curve_preserve_chroma(
        img,
        p.get("blacks", 0),
        p.get("shadows", 0),
        p.get("midtones", 0),
        p.get("highlights", 0),
        p.get("whites", 0),
    )

    # Interactive tone curve shapes tonality on top of the band sliders.
    tone_points = cs.get("tone_curve")
    if tone_points:
        img = apply_curve(img, tone_points)

    clarity = p.get("clarity", 0)
    if clarity != 0:
        img = _apply_micro_contrast(img, clarity, acceleration=acceleration, detail_sigma=1.8, base_sigma=9.0, edge_strength=2.4)

    img = adjust_hsv_sat(img, p.get("vibrance", 0) * 0.5 + p.get("saturation", 0) * 0.5, working_space=working_space)
    img = adjust_hsv_sat(img, p.get("vibrance", 0) * 0.5, working_space=working_space)

    # HSL color mixer: per-hue-band hue/saturation/luminance grading.
    color_mixer = cs.get("color_mixer")
    if color_mixer:
        img = apply_color_mixer(img, color_mixer, working_space=working_space)

    sharpness = p.get("sharpness", 0) / 100.0
    if sharpness > 0:
        img = _apply_sharpness(img, 1 + sharpness * 3, working_space)

    noise_red = p.get("noise_red", 0) / 100.0
    if noise_red > 0:
        img = gaussian_blur(img, sigma=noise_red * 3.0, acceleration=acceleration)

    glow = p.get("glow", 0) / 100.0
    if glow > 0:
        bloom = gaussian_blur(img, sigma=20, acceleration=acceleration)
        img = clamp01(img + bloom * glow * 0.35)

    vignette = p.get("vignette", 0) / 100.0
    if vignette != 0:
        h, w = img.shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt(((x_grid - w / 2) / (w / 2)) ** 2 + ((y_grid - h / 2) / (h / 2)) ** 2)
        vmask = 1.0 - np.clip(dist * abs(vignette), 0, 1)
        if vignette < 0:
            vmask = 2.0 - vmask
        img = clamp01(img * vmask[:, :, np.newaxis])

    return img


def process_subjects_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "subjects", acceleration=acceleration)
    edited = original.copy()

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        edited = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.6, base_sigma=6.8, edge_strength=2.3)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_background_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    fast_preview = _fast_interactive_preview(runtime_settings)
    effect_masks = _make_effect_masks(mask, "background", acceleration=acceleration)
    edited = original.copy()

    blur = p.get("blur", 0) / 100.0
    if blur > 0:
        if fast_preview:
            blur = min(blur, 0.22)
        sigma = 1.0 + blur * 12.0
        blurred = gaussian_blur(edited, sigma=sigma, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, blurred, effect_masks["broad"])

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        edited = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=2.0, base_sigma=10.0, edge_strength=1.8)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    dehaze = p.get("dehaze", 0) / 100.0
    if dehaze > 0:
        if fast_preview:
            dehaze = min(dehaze, 0.18)
        lum = _luma(edited)
        haze_mask = smooth_mask(np.clip(1.0 - lum * 1.15, 0.0, 1.0), sigma=2.2, acceleration=acceleration)
        dehazed = clamp01(edited + (edited - gaussian_blur(edited, sigma=7.0, acceleration=acceleration)) * dehaze * 0.85)
        edited = _mix_with_effect_mask(edited, dehazed, np.clip(haze_mask * effect_masks["detail"], 0.0, 1.0))

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_face_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    fast_preview = _fast_interactive_preview(runtime_settings)
    effect_masks = _make_effect_masks(mask, "face", acceleration=acceleration)
    edited = original.copy()

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    edited = apply_tone_curve_preserve_chroma(
        edited,
        0,
        0,
        0,
        p.get("highlights", 0),
        p.get("whites", 0) if "whites" in p else 0,
    )

    shadows = p.get("shadows", 0)
    if shadows != 0:
        edited = apply_tone_curve_preserve_chroma(edited, 0, shadows, 0, 0, 0)

    unify_amount = max(
        p.get("smooth", 0) / 100.0 * 0.55,
        abs(p.get("shadows", 0)) / 100.0 * 0.20,
        abs(p.get("highlights", 0)) / 100.0 * 0.18,
    )
    if unify_amount > 0:
        edited = _unify_face_tone(edited, unify_amount, acceleration=acceleration)

    clarity = p.get("clarity", 0)
    if clarity != 0:
        edited = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.5, base_sigma=7.0, edge_strength=2.6)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    sharpness = p.get("sharpness", 0) / 100.0
    if sharpness > 0:
        edited = _apply_sharpness(edited, 1 + sharpness * 4, working_space)

    smooth = p.get("smooth", 0) / 100.0
    if smooth > 0:
        if fast_preview:
            smooth = min(smooth, 0.18)
        smoothed = gaussian_blur(edited, sigma=smooth * 4, acceleration=acceleration)
        detail = edited - gaussian_blur(edited, sigma=1, acceleration=acceleration)
        edited = clamp01(smoothed + detail * (1 - smooth * 0.6))

    glow = p.get("glow", 0) / 100.0
    if glow > 0:
        if fast_preview:
            glow = min(glow, 0.10)
        bloom = gaussian_blur(edited, sigma=15, acceleration=acceleration)
        edited = clamp01(edited + bloom * glow * 0.3)

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_skin_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    fast_preview = _fast_interactive_preview(runtime_settings)
    effect_masks = _make_effect_masks(mask, "skin", acceleration=acceleration)
    edited = original.copy()

    ev = p.get("exposure", 0) / 100.0
    if ev != 0:
        edited = clamp01(edited * (2**ev))

    smooth = p.get("smooth", 0) / 100.0
    if smooth > 0:
        if fast_preview:
            smooth = min(smooth, 0.30)
        edited = _frequency_smooth_skin(edited, smooth, acceleration=acceleration)

    blemish = p.get("blemish", 0) / 100.0
    if blemish > 0:
        if fast_preview:
            blemish = min(blemish, 0.25)
        edited = _frequency_blemish_soften(edited, blemish, acceleration=acceleration)

    clarity = p.get("clarity", 0)
    if clarity != 0:
        clarified = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.3, base_sigma=5.0, edge_strength=2.8)
        edited = _mix_with_effect_mask(edited, clarified, effect_masks["detail"])

    color_even_amount = max(smooth * 0.45, blemish * 0.35)
    if color_even_amount > 0:
        edited = _even_skin_chroma(edited, color_even_amount, acceleration=acceleration)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    glow = p.get("glow", 0) / 100.0
    if glow > 0:
        if fast_preview:
            glow = min(glow, 0.08)
        bloom = gaussian_blur(edited, sigma=12, acceleration=acceleration)
        edited = clamp01(edited + bloom * glow * 0.25)

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_eyes_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "eyes", acceleration=acceleration)
    edited = original.copy()
    edge_guard = _edge_protection_mask(edited, acceleration=acceleration, strength=3.0, blur_sigma=1.0)
    mid_luma = _luma(edited)
    iris_guard = np.clip(1.0 - np.abs(mid_luma - 0.42) * 3.0, 0.0, 1.0).astype(np.float32)

    brightness = p.get("brightness", 0) / 100.0
    if brightness != 0:
        edited = clamp01(edited * (2 ** (brightness * 1.5)))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        blurred = gaussian_blur(edited, sigma=3, acceleration=acceleration)
        clarity_mask = smooth_mask(np.clip(edge_guard * (0.55 + iris_guard * 0.45) * effect_masks["detail"], 0.0, 1.0), sigma=1.0, acceleration=acceleration)
        clarified = clamp01(edited + (edited - blurred) * (clarity / 100.0))
        edited = _mix_with_effect_mask(edited, clarified, clarity_mask)

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    white_amt = p.get("whites", 0) / 100.0
    if white_amt > 0:
        edited = _apply_eye_whitening(edited, white_amt, edge_guard=edge_guard * effect_masks["strict"], acceleration=acceleration)

    sharpen = p.get("sharpen", 0) / 100.0
    if sharpen > 0:
        sharpened = _apply_sharpness(edited, 1 + sharpen * 5, working_space)
        sharpen_mask = smooth_mask(np.clip(edge_guard * (0.35 + iris_guard * 0.65) * effect_masks["strict"], 0.0, 1.0), sigma=0.8, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, sharpened, sharpen_mask)

    iris_pop = p.get("iris_pop", 0) / 100.0
    if iris_pop > 0:
        lum = _luma(edited)[:, :, np.newaxis]
        mid_mask = np.clip(1.0 - np.abs(lum - 0.35) * 4, 0, 1)
        iris_mask = mid_mask[:, :, 0] * edge_guard * effect_masks["detail"]
        iris_popped = clamp01(edited + (edited - 0.5) * mid_mask * iris_pop * 0.45)
        edited = _mix_with_effect_mask(edited, iris_popped, smooth_mask(iris_mask, sigma=1.0, acceleration=acceleration))

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_lips_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "lips", acceleration=acceleration)
    edited = original.copy()
    edge_guard = _edge_protection_mask(edited, acceleration=acceleration, strength=2.4, blur_sigma=1.0)

    brightness = p.get("brightness", 0) / 100.0
    if brightness != 0:
        edited = clamp01(edited * (2**brightness))

    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)
    edited = adjust_hsv_hue(edited, p.get("hue_shift", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    smooth = p.get("smooth", 0) / 100.0
    if smooth > 0:
        edited_s = gaussian_blur(edited, sigma=smooth * 3, acceleration=acceleration)
        detail = edited - gaussian_blur(edited, sigma=0.5, acceleration=acceleration)
        smoothed = clamp01(edited_s + detail * (1 - smooth * 0.5))
        smooth_mask_local = smooth_mask(np.clip((0.75 + edge_guard * 0.25) * effect_masks["broad"], 0.0, 1.0), sigma=0.8, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, smoothed, smooth_mask_local)

    gloss = p.get("gloss", 0) / 100.0
    if gloss > 0:
        lum = _luma(edited)
        center_mask = np.clip(1.0 - np.abs(lum - 0.58) * 3.5, 0.0, 1.0)
        gloss_mask = smooth_mask(np.clip(center_mask * edge_guard * effect_masks["strict"], 0.0, 1.0), sigma=1.2, acceleration=acceleration)[:, :, np.newaxis]
        highlight = np.clip(lum[:, :, np.newaxis] - 0.5, 0, 1) * 2
        glossed = clamp01(edited + highlight * gloss_mask * gloss * 0.28)
        edited = _mix_with_effect_mask(edited, glossed, gloss_mask[:, :, 0])

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


def process_hair_layer(
    base: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    p: dict,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
) -> np.ndarray:
    working_space = _working_space(color_settings)
    acceleration = _acceleration_mode(runtime_settings)
    effect_masks = _make_effect_masks(mask, "hair", acceleration=acceleration)
    edited = original.copy()
    strand_mask = _hair_strand_mask(edited, acceleration=acceleration)

    brightness = p.get("brightness", 0) / 100.0
    if brightness != 0:
        edited = clamp01(edited * (2**brightness))

    edited = apply_tone_curve_preserve_chroma(edited, 0, 0, 0, p.get("highlights", 0), 0)
    edited = _apply_hair_depth(edited, p.get("highlights", 0) * 0.35, acceleration=acceleration)
    edited = adjust_hsv_sat(edited, p.get("saturation", 0), working_space=working_space)

    warmth = p.get("warmth", 0)
    if warmth != 0:
        edited = adjust_warmth_preserve_hue(edited, warmth, working_space=working_space)

    shine = p.get("shine", 0) / 100.0
    if shine > 0:
        shined = _apply_hair_shine(edited, shine, acceleration=acceleration)
        edited = _mix_with_effect_mask(edited, shined, np.clip(effect_masks["detail"] * (0.35 + strand_mask * 0.65), 0.0, 1.0))

    clarity = p.get("clarity", 0)
    if clarity != 0:
        clarified = _apply_micro_contrast(edited, clarity, acceleration=acceleration, detail_sigma=1.1, base_sigma=4.8, edge_strength=1.7)
        edited = _mix_with_effect_mask(edited, clarified, np.clip(effect_masks["detail"] * (0.35 + strand_mask * 0.65), 0.0, 1.0))

    return _mix_with_effect_mask(original, edited, effect_masks["blend"])


LAYER_PROCESSORS = {
    "subjects": process_subjects_layer,
    "background": process_background_layer,
    "face": process_face_layer,
    "skin": process_skin_layer,
    "eyes": process_eyes_layer,
    "lips": process_lips_layer,
    "hair": process_hair_layer,
}


def _apply_blend_mode(base: np.ndarray, layer_img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "overlay":
        mixed = np.where(
            base <= 0.5,
            2.0 * base * layer_img,
            1.0 - 2.0 * (1.0 - base) * (1.0 - layer_img),
        )
        return clamp01(mixed)

    if mode == "soft_light":
        mixed = (1.0 - 2.0 * layer_img) * (base * base) + 2.0 * layer_img * base
        return clamp01(mixed)

    return layer_img


def _layer_composite_mask(
    mask: np.ndarray,
    layer: str,
    mode: str,
    opacity: float,
    *,
    acceleration: str = "auto",
) -> np.ndarray:
    effect_masks = _make_effect_masks(mask, layer, acceleration=acceleration)
    mode = str(mode or "normal").lower()

    if mode == "overlay":
        base_mask = effect_masks["detail"] if layer in {"face", "skin", "hair"} else effect_masks["strict"]
    elif mode == "soft_light":
        if layer in {"eyes", "lips"}:
            base_mask = effect_masks["detail"]
        else:
            base_mask = np.clip(effect_masks["blend"] * 0.7 + effect_masks["broad"] * 0.3, 0.0, 1.0)
    else:
        base_mask = effect_masks["blend"]

    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0.0:
        return np.zeros_like(mask, dtype=np.float32)
    if opacity >= 0.999:
        return np.clip(base_mask, 0.0, 1.0).astype(np.float32)

    opacity_power = 1.0 + (1.0 - opacity) * 1.6
    opacity_floor = max(0.0, 0.25 - opacity * 0.20)
    scaled = _confidence_curve(base_mask, power=opacity_power, floor=opacity_floor)
    return np.clip(scaled * opacity, 0.0, 1.0).astype(np.float32)


def process_all_layers(
    original: np.ndarray,
    all_params: dict,
    masks: dict,
    geometry: dict | list[dict] | None = None,
    layer_order=None,
    layer_options: dict | None = None,
    color_settings: dict | None = None,
    runtime_settings: dict | None = None,
):
    """Apply global then selective layers with per-layer blend configuration."""
    working_space = _working_space(color_settings)
    output_transform = _output_transform(color_settings)
    working_original = display_to_working(original, working_space=working_space)
    working_masks = None if masks is None else {key: np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0) for key, value in masks.items()}
    working_original, working_masks = _apply_expression_warp(
        working_original,
        working_masks,
        all_params.get("face", {}),
        geometry=geometry,
    )
    result = process_global(
        working_original,
        all_params.get("global", {}),
        color_settings=color_settings,
        runtime_settings=runtime_settings,
    )
    result = _apply_face_refinement(
        result,
        working_masks,
        all_params.get("face", {}),
        color_settings=color_settings,
        runtime_settings=runtime_settings,
    )
    order = tuple(layer_order) if layer_order else MASK_ORDER
    options = layer_options or {}

    for layer in order:
        if layer not in LAYER_PROCESSORS:
            continue
        if not working_masks or layer not in working_masks:
            continue

        cfg = options.get(layer, {})
        if not cfg.get("enabled", True):
            continue

        opacity = float(np.clip(float(cfg.get("opacity", 100.0)) / 100.0, 0.0, 1.0))
        if opacity <= 0.0:
            continue

        mask = working_masks[layer]
        if layer == "skin":
            protection = _build_skin_protection_mask(
                result,
                working_masks,
                geometry=geometry,
                acceleration=_acceleration_mode(runtime_settings),
            )
            mask = np.clip(mask.astype(np.float32) * (1.0 - protection), 0.0, 1.0)
        if mask is None or mask.max() <= 0.01:
            continue

        mode = str(cfg.get("blend_mode", "normal"))
        full_mask = _layer_composite_mask(
            mask,
            layer,
            mode,
            opacity,
            acceleration=_acceleration_mode(runtime_settings),
        )
        # Selective layers should build on top of the globally adjusted image,
        # not the original source, so global edits propagate into face/skin/etc.
        layer_img = LAYER_PROCESSORS[layer](
            result,
            result,
            mask.astype(np.float32),
            all_params.get(layer, {}),
            color_settings=color_settings,
            runtime_settings=runtime_settings,
        )
        mixed = _apply_blend_mode(result, layer_img, mode)
        result = blend_with_mask(result, mixed, full_mask)

    display_result = working_to_display(result, output_transform=output_transform, working_space=working_space)
    return Image.fromarray(to_uint8(display_result))
