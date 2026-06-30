#!/usr/bin/env python3
"""Download (optional) and validate an ML denoise ONNX model.

There's no single canonical denoiser ONNX, so this is validator-first: point --url at a
model you trust (image-in/image-out denoiser, e.g. DnCNN), or drop the file at
models/denoise.onnx yourself and run with --skip-download to validate. The shipped default
is DnCNN (cszn/KAIR) -- regenerate it with scripts/convert_dncnn_to_onnx.py.

It confirms the model loads and that, on a noisy dummy frame, its output is a same-shape
3-channel image that actually reduces noise -- the contract MLDenoiser expects.

Usage:
    python scripts/download_denoise_model.py --url <URL>      # download + validate
    python scripts/download_denoise_model.py --skip-download  # validate models/denoise.onnx
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "models" / "denoise.onnx"


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"Downloading:\n  {url}\n-> {out}")

    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            sys.stdout.write(f"\r  {done/1e6:6.1f} / {total_size/1e6:6.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    sys.stdout.write("\n")
    tmp.replace(out)


def _validate(path: Path, ref_size: int = 128) -> bool:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - optional deps
        print(f"  ! Cannot validate (missing dep): {exc}")
        print("    Install requirements/optional-model.txt into the app's environment first.")
        return False

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"  ! Failed to load as ONNX: {exc}")
        return False

    inputs = sess.get_inputs()
    if len(inputs) != 1:
        print(f"  ! Expected 1 input, model has {len(inputs)}.")
        return False

    shape = inputs[0].shape
    layout = "nhwc" if (len(shape) == 4 and shape[3] == 3 and shape[1] != 3) else "nchw"

    rng = np.random.default_rng(0)
    base = np.linspace(0.2, 0.8, ref_size, dtype=np.float32)[None, :, None]
    base = np.repeat(np.repeat(base, ref_size, 0), 3, 2)
    noisy = np.clip(base + rng.normal(0, 0.08, base.shape).astype(np.float32), 0, 1)
    if layout == "nhwc":
        tensor = noisy[None, ...].astype(np.float32)
    else:
        tensor = np.transpose(noisy, (2, 0, 1))[None, ...].astype(np.float32)

    try:
        out = sess.run(None, {inputs[0].name: tensor})[0]
    except Exception as exc:
        print(f"  ! Inference failed: {exc}")
        return False

    a = np.squeeze(np.asarray(out, dtype=np.float32))
    if a.ndim != 3:
        print(f"  ! Output is not a single image (shape {np.asarray(out).shape}).")
        return False
    if a.shape[0] == 3:
        a = np.transpose(a, (1, 2, 0))
    if a.shape != (ref_size, ref_size, 3):
        print(f"  ! Output shape {a.shape} != input {(ref_size, ref_size, 3)} (not image-to-image).")
        return False
    if not np.isfinite(a).all():
        print("  ! Output has non-finite values.")
        return False

    a = np.clip(a, 0, 1)
    noise_before = float(np.std(noisy - base))
    noise_after = float(np.std(a - base))
    reduction = 100.0 * (1.0 - noise_after / max(noise_before, 1e-6))
    if reduction < 20.0:
        print(f"  ! Model barely denoises (noise reduction only {reduction:.0f}%).")
        return False

    print(f"  OK: input '{inputs[0].name}' ({layout}), image-to-image, noise reduction {reduction:.0f}%")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download + validate an ML denoise ONNX model")
    parser.add_argument("--url", default="", help="URL of an image-to-image denoiser ONNX")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path (default models/denoise.onnx)")
    parser.add_argument("--skip-download", action="store_true", help="Only validate an existing file")
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser().resolve()

    if not args.skip_download:
        if not args.url:
            print("No --url given. Either pass a trusted denoiser ONNX URL, or place the file at")
            print(f"{out} and re-run with --skip-download. To build the default from DnCNN weights:")
            print("  python scripts/convert_dncnn_to_onnx.py")
            return 1
        try:
            _download(args.url, out)
        except Exception as exc:
            print(f"Download failed: {exc}")
            return 1
    elif not out.exists():
        print(f"No file at {out} to validate.")
        return 1

    print(f"Validating {out.name} ({out.stat().st_size/1e6:.1f} MB)...")
    if not _validate(out):
        print("Validation FAILED. Noise reduction will fall back to the bilateral filter.")
        return 2
    print("\nDone. The noise-reduction slider will now use this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
