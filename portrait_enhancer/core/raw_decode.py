"""Shared camera-RAW decode: one path for Qt, batch, and Tk.

A camera RAW file is a sensor mosaic, not an image -- LibRaw (via rawpy) runs demosaic,
white balance, and the camera color transform during ``postprocess`` and hands back a
finished RGB image. This module is the single place that owns those decode decisions, so
every caller decodes a given file *identically* and records what it did. Previously each UI
(Qt, batch, Tk) had its own ``postprocess`` call and they disagreed -- Qt/batch hardcoded
camera WB and ignored the RAW color settings the data model already carried.

See ``docs/RAW_PROCESSING_STRATEGY.md`` for the full pipeline and rationale. GUI-free so it
runs on Qt worker threads and in batch worker processes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .lut import apply_cube_lut, load_cube_lut

try:
    import rawpy

    HAS_RAWPY = True
except Exception:  # pragma: no cover - exercised only on hosts without rawpy
    rawpy = None
    HAS_RAWPY = False

# Camera RAW extensions LibRaw decodes. Kept broad on purpose -- gating to a handful (the old
# .cr2/.nef/.arw/.dng/.raw) silently excluded common modern formats, most notably Canon .cr3
# (every EOS R body). This is the single source of truth for "is this a RAW file" across the app.
RAW_EXTS = (
    ".cr2", ".cr3", ".crw",   # Canon
    ".nef", ".nrw",            # Nikon
    ".arw", ".srf", ".sr2",    # Sony
    ".raf",                     # Fujifilm
    ".orf",                     # Olympus / OM System
    ".rw2",                     # Panasonic
    ".pef",                     # Pentax
    ".srw",                     # Samsung
    ".rwl",                     # Leica
    ".3fr", ".fff",            # Hasselblad
    ".iiq",                     # Phase One
    ".x3f",                     # Sigma (Foveon)
    ".gpr",                     # GoPro
    ".dcr", ".kdc",            # Kodak
    ".mrw",                     # Minolta
    ".erf",                     # Epson
    ".mef",                     # Mamiya
    ".mos",                     # Leaf
    ".dng",                     # Adobe / generic
    ".raw",                     # generic
)

_VALID_WB = ("camera", "auto")
_RAW_COLORSPACE_MAP = {
    "srgb": "sRGB",
    "adobe": "Adobe",
    "prophoto": "ProPhoto",
    "xyz": "XYZ",
    "raw": "raw",
}

# Clipped-highlight handling -> LibRaw highlight_mode int. "clip" (0) is LibRaw's default and
# matches pre-Phase-2 behavior; "blend" (2) softens the channel-clip transition; "rebuild" (5)
# reconstructs blown channels from their neighbors (best for skies / specular falloff).
_HIGHLIGHT_MODES = {
    "clip": 0,
    "blend": 2,
    "rebuild": 5,
}

# CFA demosaic algorithm -> rawpy.DemosaicAlgorithm attribute name. "auto" passes nothing, so
# LibRaw uses its default (AHD) -- guaranteeing byte-identical output to the pre-Phase-2 decode.
# DCB/AAHD/DHT need GPL demosaic packs in the LibRaw build; decode_raw falls back to the LibRaw
# default (recording raw_demosaic_fallback) when the chosen one isn't compiled in.
_DEMOSAIC_ALGOS = {
    "auto": None,
    "ahd": "AHD",
    "dcb": "DCB",
    "vng": "VNG",
    "aahd": "AAHD",
    "dht": "DHT",
}

# .cube LUT cache keyed by (abspath, mtime) so repeated decodes of the same source don't
# reparse the file. Process-local, which is correct for both Qt (threads) and batch (worker
# processes); a stale cube on disk is picked up because mtime is part of the key.
_LUT_CACHE: dict[tuple[str, float], dict] = {}


def is_raw_path(path) -> bool:
    return Path(str(path)).suffix.lower() in RAW_EXTS


def normalize_raw_settings(color_settings) -> dict:
    """Coerce a (possibly partial) color-settings dict to the RAW decode keys this module
    understands, applying defaults. Unknown WB/colorspace values fall back to safe defaults."""
    cs = dict(color_settings or {})
    wb = str(cs.get("raw_white_balance", "camera")).lower()
    if wb not in _VALID_WB:
        wb = "camera"
    colorspace = str(cs.get("raw_colorspace", "srgb")).lower()
    if colorspace not in _RAW_COLORSPACE_MAP:
        colorspace = "srgb"
    highlight = str(cs.get("raw_highlight_mode", "clip")).lower()
    if highlight not in _HIGHLIGHT_MODES:
        highlight = "clip"
    demosaic = str(cs.get("raw_demosaic", "auto")).lower()
    if demosaic not in _DEMOSAIC_ALGOS:
        demosaic = "auto"
    return {
        "raw_white_balance": wb,
        "raw_colorspace": colorspace,
        # Both default to the LibRaw-default behavior so a decode is byte-identical to before
        # these controls existed unless the user opts into blend/rebuild or a specific demosaic.
        "raw_highlight_mode": highlight,
        "raw_demosaic": demosaic,
        "raw_lut_enabled": bool(cs.get("raw_lut_enabled", False)),
        "raw_lut_path": str(cs.get("raw_lut_path", "") or ""),
        # Default ON preserves LibRaw's pleasant auto-brightened import. Turn OFF for a
        # deterministic, batch-consistent baseline that the Exposure slider then owns.
        "raw_auto_brightness": bool(cs.get("raw_auto_brightness", True)),
    }


def build_postprocess_kwargs(cfg: dict) -> dict:
    """rawpy.postprocess kwargs for a normalized config, minus ``output_color`` (which needs
    the rawpy enum). Split out so the decode policy is unit-testable without rawpy installed."""
    kwargs = {
        "output_bps": 16,
        "no_auto_bright": not cfg["raw_auto_brightness"],
        "highlight_mode": _HIGHLIGHT_MODES[cfg["raw_highlight_mode"]],
    }
    if cfg["raw_white_balance"] == "auto":
        kwargs["use_auto_wb"] = True
        kwargs["use_camera_wb"] = False
    else:
        kwargs["use_camera_wb"] = True
        kwargs["use_auto_wb"] = False
    return kwargs


def _resolve_demosaic(cfg: dict, rawpy_module, metadata: dict):
    """Return the rawpy DemosaicAlgorithm enum for the config, or None to use the LibRaw
    default. Falls back to the default (recording raw_demosaic_fallback) when the requested
    algorithm isn't compiled into this LibRaw build."""
    name = _DEMOSAIC_ALGOS.get(cfg["raw_demosaic"])
    if not name:
        return None  # "auto" -> LibRaw default (AHD)
    algo = getattr(getattr(rawpy_module, "DemosaicAlgorithm", None), name, None)
    if algo is None:
        metadata["raw_demosaic_fallback"] = f"{cfg['raw_demosaic']} unavailable in this rawpy build"
        return None
    if not bool(getattr(algo, "isSupported", True)):
        metadata["raw_demosaic_fallback"] = f"{cfg['raw_demosaic']} not supported by this LibRaw build"
        return None
    return algo


def _get_raw_lut(path: str) -> dict:
    abspath = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(abspath)
    except OSError:
        mtime = 0.0
    key = (abspath, mtime)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = load_cube_lut(abspath)
        _LUT_CACHE[key] = lut
    return lut


def _extract_raw_metadata(raw) -> dict:
    """Capture as-shot decode metadata (JSON-serializable) so a decode is reproducible and
    auditable: camera WB multipliers, color matrix, black/white levels, CFA pattern. Best
    effort per field -- which rawpy attributes exist varies by file and rawpy version."""
    meta: dict = {}
    try:
        wb = getattr(raw, "camera_whitebalance", None)
        if wb is not None:
            meta["raw_camera_whitebalance"] = [float(x) for x in wb]
    except Exception:
        pass
    try:
        dwb = getattr(raw, "daylight_whitebalance", None)
        if dwb is not None:
            meta["raw_daylight_whitebalance"] = [float(x) for x in dwb]
    except Exception:
        pass
    try:
        mat = getattr(raw, "rgb_xyz_matrix", None)
        if mat is not None:
            meta["raw_rgb_xyz_matrix"] = np.asarray(mat, dtype=float).tolist()
    except Exception:
        pass
    try:
        black = getattr(raw, "black_level_per_channel", None)
        if black is not None:
            meta["raw_black_level_per_channel"] = [float(x) for x in black]
    except Exception:
        pass
    try:
        white = getattr(raw, "white_level", None)
        if white is not None:
            meta["raw_white_level"] = float(white)
    except Exception:
        pass
    try:
        pattern = getattr(raw, "raw_pattern", None)
        if pattern is not None:
            meta["raw_cfa_pattern"] = np.asarray(pattern).tolist()
    except Exception:
        pass
    try:
        desc = getattr(raw, "color_desc", None)
        if desc is not None:
            meta["raw_color_desc"] = desc.decode() if isinstance(desc, bytes) else str(desc)
    except Exception:
        pass
    try:
        num = getattr(raw, "num_colors", None)
        if num is not None:
            meta["raw_num_colors"] = int(num)
    except Exception:
        pass
    return meta


def _finish_decode(rgb16, cfg: dict, metadata: dict, *, preview: bool = False):
    """Shared tail of a decode pass: normalize to float32, apply the RAW LUT if configured,
    and stamp the audit metadata. Shared between the half-size preview pass and the final
    full-resolution pass so both get identical color treatment (LUT, audit string shape)."""
    full = rgb16.astype(np.float32) / 65535.0

    if cfg["raw_lut_enabled"] and cfg["raw_lut_path"]:
        # A broken/missing LUT must not make the image fail to open -- decode the RAW
        # anyway and record the failure for the source-profile diagnostics.
        try:
            lut = _get_raw_lut(cfg["raw_lut_path"])
            full = apply_cube_lut(full, lut)
            metadata["raw_lut_title"] = lut.get("title", os.path.basename(cfg["raw_lut_path"]))
        except Exception as ex:
            metadata["raw_lut_error"] = f"{type(ex).__name__}: {ex}"

    metadata["source_is_raw"] = True
    metadata["raw_preview"] = bool(preview)
    lut_on = bool(cfg["raw_lut_enabled"] and cfg["raw_lut_path"] and "raw_lut_error" not in metadata)
    metadata["input_profile_applied"] = (
        f"raw_decode:wb={cfg['raw_white_balance']}:color={cfg['raw_colorspace']}:"
        f"hl={cfg['raw_highlight_mode']}:demosaic={cfg['raw_demosaic']}:"
        f"auto_bright={'on' if cfg['raw_auto_brightness'] else 'off'}:"
        f"lut={'on' if lut_on else 'off'}"
    )
    return full, metadata


def decode_raw(path, color_settings=None, on_preview=None):
    """Decode a camera RAW file to float32 RGB in ``[0, 1]`` plus a decode-metadata dict.

    The metadata carries the captured as-shot camera data, ``source_is_raw=True``, an
    ``input_profile_applied`` audit string describing the decode, and (when a RAW LUT is
    applied) ``raw_lut_title``. Raises ``RuntimeError`` if rawpy is unavailable.

    If ``on_preview`` is given, a fast half-resolution pass runs first -- a real demosaic
    through the same color pipeline (WB, color space, LUT), just at half linear resolution,
    not the camera's embedded JPEG -- and ``on_preview(preview_array, preview_metadata)`` is
    called with it before the full-resolution pass continues. Both passes reuse the same
    opened RAW file (no extra disk read), so this only costs the half-size demosaic itself.
    A full-resolution RAW demosaic can take several seconds; the half-size pass is typically
    ~4x faster, letting a caller show a pixel-accurate preview well before that finishes. A
    failure in the preview pass is swallowed (best-effort) -- the full decode below is what
    the caller actually needs and must still succeed.
    """
    if not HAS_RAWPY:
        raise RuntimeError("rawpy is not installed. Install base dependencies first.")
    cfg = normalize_raw_settings(color_settings)
    metadata: dict = {}
    with rawpy.imread(str(path)) as raw:
        metadata.update(_extract_raw_metadata(raw))
        kwargs = build_postprocess_kwargs(cfg)
        cs_name = _RAW_COLORSPACE_MAP.get(cfg["raw_colorspace"], "sRGB")
        cs_value = getattr(rawpy.ColorSpace, cs_name, None)
        if cs_value is not None:
            kwargs["output_color"] = cs_value
        demosaic = _resolve_demosaic(cfg, rawpy, metadata)
        if demosaic is not None:
            kwargs["demosaic_algorithm"] = demosaic

        if on_preview is not None:
            try:
                preview_rgb16 = raw.postprocess(half_size=True, **kwargs)
                preview_array, preview_metadata = _finish_decode(
                    preview_rgb16, cfg, dict(metadata), preview=True
                )
                on_preview(preview_array, preview_metadata)
            except Exception:
                pass

        rgb16 = raw.postprocess(**kwargs)

    return _finish_decode(rgb16, cfg, metadata, preview=False)
