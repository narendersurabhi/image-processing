"""Core image utility functions."""

import numpy as np
import cv2

_CUDA_DEVICE_COUNT = None


def cuda_available():
    global _CUDA_DEVICE_COUNT
    if _CUDA_DEVICE_COUNT is None:
        try:
            _CUDA_DEVICE_COUNT = int(cv2.cuda.getCudaEnabledDeviceCount())
        except Exception:
            _CUDA_DEVICE_COUNT = 0
    return _CUDA_DEVICE_COUNT > 0


def resolve_acceleration_mode(requested="auto"):
    requested = str(requested or "auto").lower()
    if requested == "cuda" and cuda_available():
        return "cuda"
    if requested == "auto" and cuda_available():
        return "cuda"
    return "cpu"


def _odd_kernel_size_from_sigma(sigma: float) -> int:
    sigma = max(0.0, float(sigma))
    if sigma <= 0.0:
        return 1
    return max(3, int(round(sigma * 6)) | 1)


def resize_image(img, size, interpolation=cv2.INTER_AREA, acceleration="auto"):
    mode = resolve_acceleration_mode(acceleration)
    if mode == "cuda":
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(img.astype(np.float32, copy=False))
            result = cv2.cuda.resize(gpu, size, interpolation=interpolation).download()
            return result.astype(np.float32, copy=False)
        except Exception:
            pass
    return cv2.resize(img, size, interpolation=interpolation).astype(np.float32, copy=False)


def gaussian_blur(img, sigma, acceleration="auto"):
    sigma = float(sigma)
    if sigma <= 0.0:
        return img.astype(np.float32, copy=True)

    mode = resolve_acceleration_mode(acceleration)
    kernel = (_odd_kernel_size_from_sigma(sigma), _odd_kernel_size_from_sigma(sigma))
    source = img.astype(np.float32, copy=False)

    if mode == "cuda":
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(source)
            if source.ndim == 2:
                filt = cv2.cuda.createGaussianFilter(cv2.CV_32FC1, cv2.CV_32FC1, kernel, sigma)
            else:
                filt = cv2.cuda.createGaussianFilter(cv2.CV_32FC3, cv2.CV_32FC3, kernel, sigma)
            return filt.apply(gpu).download().astype(np.float32, copy=False)
        except Exception:
            pass

    return cv2.GaussianBlur(source, kernel, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT).astype(
        np.float32, copy=False
    )


def clamp01(a):
    return np.clip(a, 0.0, 1.0)


def srgb_to_linear(a):
    a = clamp01(a).astype(np.float32)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(a):
    a = clamp01(a).astype(np.float32)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1.0 / 2.4) - 0.055).astype(np.float32)


def gamma_to_linear(a, gamma):
    return np.power(clamp01(a).astype(np.float32), float(gamma)).astype(np.float32)


def linear_to_gamma(a, gamma):
    return np.power(clamp01(a).astype(np.float32), 1.0 / float(gamma)).astype(np.float32)


def working_to_display(img, output_transform="srgb", working_space="srgb"):
    if working_space == "linear":
        linear = clamp01(img)
    else:
        if output_transform == "srgb":
            return clamp01(img.astype(np.float32))
        linear = srgb_to_linear(img)

    if output_transform == "linear":
        return clamp01(linear)
    if output_transform == "gamma22":
        return linear_to_gamma(linear, 2.2)
    if output_transform == "gamma18":
        return linear_to_gamma(linear, 1.8)
    return linear_to_srgb(linear)


def display_to_working(img, working_space="srgb"):
    if working_space == "linear":
        return srgb_to_linear(img)
    return clamp01(img.astype(np.float32))


def to_uint8(a):
    return (clamp01(a) * 255).astype(np.uint8)


def to_float(a):
    return a.astype(np.float32) / 255.0


def apply_tone_curve(img, blacks, shadows, midtones, highlights, whites):
    x = np.array([0.0, 0.10, 0.30, 0.70, 0.90, 1.0])
    y = np.array(
        [
            max(0.0, blacks / 100.0),
            max(0.0, min(1.0, 0.10 + shadows / 200.0)),
            max(0.0, min(1.0, 0.30 + midtones / 200.0)),
            max(0.0, min(1.0, 0.70 + highlights / 200.0)),
            max(0.0, min(1.0, 0.90 + whites / 200.0)),
            min(1.0, 1.0 + whites / 400.0),
        ]
    )
    lut = np.interp(np.linspace(0, 1, 256), x, y)
    return lut[(img * 255).astype(np.uint8)].astype(np.float32)


def apply_tone_curve_preserve_chroma(img, blacks, shadows, midtones, highlights, whites):
    img = clamp01(img.astype(np.float32))
    luma = np.clip(
        img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722,
        0.0,
        1.0,
    ).astype(np.float32)
    mapped_luma = apply_tone_curve(luma, blacks, shadows, midtones, highlights, whites)
    gain = mapped_luma / np.maximum(luma, 1e-4)
    remapped = img * gain[:, :, np.newaxis]

    # Blend a small amount of direct per-channel mapping back in so the tone
    # curve still feels responsive while keeping chroma shifts under control.
    direct = apply_tone_curve(img, blacks, shadows, midtones, highlights, whites)
    return clamp01(remapped * 0.85 + direct * 0.15)


def adjust_hsv_sat(img_float, delta, working_space="srgb"):
    if delta == 0:
        return img_float
    display_img = working_to_display(img_float, output_transform="srgb", working_space=working_space)
    hsv = cv2.cvtColor(to_uint8(display_img), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + delta / 100.0), 0, 255)
    adjusted = to_float(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return display_to_working(adjusted, working_space=working_space)


def adjust_hsv_hue(img_float, delta, working_space="srgb"):
    if delta == 0:
        return img_float
    display_img = working_to_display(img_float, output_transform="srgb", working_space=working_space)
    hsv = cv2.cvtColor(to_uint8(display_img), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta * 0.9) % 180
    adjusted = to_float(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return display_to_working(adjusted, working_space=working_space)


def adjust_warmth_preserve_hue(img_float, delta, working_space="srgb"):
    if delta == 0:
        return img_float

    display_img = working_to_display(img_float, output_transform="srgb", working_space=working_space)
    hsv = cv2.cvtColor(to_uint8(display_img), cv2.COLOR_RGB2HSV).astype(np.float32)

    warmth = float(delta) / 100.0
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Focus the warmth move on red/orange/yellow ranges while preserving
    # underlying hue relationships more than direct RGB scaling.
    center = 18.0
    dist = np.minimum(np.abs(hue - center), 180.0 - np.abs(hue - center))
    hue_focus = np.clip(1.0 - dist / 48.0, 0.0, 1.0)
    sat_focus = np.clip(sat / 255.0, 0.15, 1.0)
    val_focus = np.clip(1.0 - np.abs((val / 255.0) - 0.58) * 1.6, 0.25, 1.0)
    weight = np.clip(hue_focus * sat_focus * val_focus, 0.0, 1.0)

    hue_shift = -6.0 * warmth
    sat_scale = 1.0 + warmth * 0.10
    val_scale = 1.0 + warmth * 0.05
    cool_sat_scale = 1.0 + warmth * 0.04

    hsv[:, :, 0] = (hue + hue_shift * weight) % 180.0
    hsv[:, :, 1] = np.clip(sat * (1.0 + (sat_scale - 1.0) * weight) * cool_sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(val * (1.0 + (val_scale - 1.0) * weight), 0, 255)

    adjusted = to_float(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return display_to_working(adjusted, working_space=working_space)


def adjust_color_balance_preserve_chroma(img_float, temperature=0, tint=0, working_space="srgb"):
    if temperature == 0 and tint == 0:
        return img_float

    # Temperature: hue-aware warm/cool move that avoids raw RGB scaling.
    adjusted = adjust_warmth_preserve_hue(img_float, temperature, working_space=working_space)
    if tint == 0:
        return adjusted

    display_img = working_to_display(adjusted, output_transform="srgb", working_space=working_space)
    lab = cv2.cvtColor(to_uint8(display_img), cv2.COLOR_RGB2LAB).astype(np.float32)
    tint_amt = float(tint) / 100.0

    l_chan = lab[:, :, 0] / 255.0
    sat_focus = np.clip(np.std(display_img, axis=2) * 4.0, 0.12, 1.0)
    luma_focus = np.clip(1.0 - np.abs(l_chan - 0.55) * 1.4, 0.25, 1.0)
    weight = np.clip(sat_focus * luma_focus, 0.0, 1.0)

    lab[:, :, 1] = np.clip(lab[:, :, 1] + tint_amt * 9.0 * weight, 0, 255)
    adjusted_rgb = to_float(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))
    return display_to_working(adjusted_rgb, working_space=working_space)


def smooth_mask(mask, sigma=4.0, acceleration="auto"):
    return gaussian_blur(mask.astype(np.float32), sigma=sigma, acceleration=acceleration)


def refine_mask_edges(mask, guide_img, radius=8, eps=1e-3):
    mask_f = clamp01(mask.astype(np.float32))
    guide = clamp01(guide_img.astype(np.float32))
    if guide.ndim == 2:
        guide_gray = guide
    else:
        guide_gray = cv2.cvtColor(to_uint8(guide), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            refined = cv2.ximgproc.guidedFilter(
                guide=(guide_gray * 255).astype(np.uint8),
                src=mask_f,
                radius=max(1, int(radius)),
                eps=float(eps),
            )
            return clamp01(np.nan_to_num(refined.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0))
    except Exception:
        pass

    # Fallback: bilateral filter still preserves stronger edges better than
    # a plain Gaussian blur when guided filtering is unavailable.
    try:
        refined = cv2.bilateralFilter(mask_f, d=max(3, int(radius) | 1), sigmaColor=0.1, sigmaSpace=max(3.0, radius))
        return clamp01(np.nan_to_num(refined.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0))
    except Exception:
        return mask_f


def blend_with_mask(orig, edited, mask):
    m = mask[:, :, np.newaxis]
    return clamp01(orig * (1 - m) + edited * m)
