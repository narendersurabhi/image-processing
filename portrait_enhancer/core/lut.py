"""3D LUT parsing and application helpers."""

from __future__ import annotations

import os

import numpy as np


def load_cube_lut(path: str) -> dict:
    title = os.path.basename(path)
    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = []

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("TITLE"):
                parts = line.split('"')
                if len(parts) >= 2:
                    title = parts[1]
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if line.startswith("DOMAIN_MIN"):
                domain_min = np.array([float(v) for v in line.split()[1:4]], dtype=np.float32)
                continue
            if line.startswith("DOMAIN_MAX"):
                domain_max = np.array([float(v) for v in line.split()[1:4]], dtype=np.float32)
                continue
            if line.startswith("LUT_1D_SIZE"):
                raise ValueError("1D LUTs are not supported; expected a 3D .cube LUT")

            parts = line.split()
            if len(parts) == 3:
                values.append([float(v) for v in parts])

    if size is None:
        raise ValueError("Missing LUT_3D_SIZE in .cube file")

    expected = size * size * size
    if len(values) != expected:
        raise ValueError(f"Expected {expected} LUT rows, found {len(values)}")

    table = np.array(values, dtype=np.float32).reshape(size, size, size, 3)
    return {
        "title": title,
        "size": size,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "table": table,
        "path": os.path.abspath(path),
    }


def apply_cube_lut(img: np.ndarray, lut: dict) -> np.ndarray:
    table = lut["table"]
    size = int(lut["size"])
    domain_min = lut["domain_min"]
    domain_max = lut["domain_max"]

    denom = np.maximum(domain_max - domain_min, 1e-6)
    coords = np.clip((img.astype(np.float32) - domain_min) / denom, 0.0, 1.0) * (size - 1)

    x = coords[:, :, 0]
    y = coords[:, :, 1]
    z = coords[:, :, 2]

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, size - 1)
    y1 = np.clip(y0 + 1, 0, size - 1)
    z1 = np.clip(z0 + 1, 0, size - 1)

    xd = (x - x0)[..., None]
    yd = (y - y0)[..., None]
    zd = (z - z0)[..., None]

    c000 = table[x0, y0, z0]
    c001 = table[x0, y0, z1]
    c010 = table[x0, y1, z0]
    c011 = table[x0, y1, z1]
    c100 = table[x1, y0, z0]
    c101 = table[x1, y0, z1]
    c110 = table[x1, y1, z0]
    c111 = table[x1, y1, z1]

    c00 = c000 * (1.0 - xd) + c100 * xd
    c01 = c001 * (1.0 - xd) + c101 * xd
    c10 = c010 * (1.0 - xd) + c110 * xd
    c11 = c011 * (1.0 - xd) + c111 * xd
    c0 = c00 * (1.0 - yd) + c10 * yd
    c1 = c01 * (1.0 - yd) + c11 * yd
    out = c0 * (1.0 - zd) + c1 * zd
    return np.clip(out.astype(np.float32), 0.0, 1.0)
