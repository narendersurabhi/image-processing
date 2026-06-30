#!/usr/bin/env python3
"""Download and validate the optional MODNet matting model.

Fetches a MODNet ONNX to models/modnet.onnx and verifies it actually loads in
onnxruntime and produces a plausible alpha matte, so a truncated/wrong download is
caught here rather than silently falling back at runtime.

Usage:
    python scripts/download_matting_model.py                # default mirror
    python scripts/download_matting_model.py --url <URL>    # your own trusted MODNet ONNX
    python scripts/download_matting_model.py --out /abs/path/modnet.onnx
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "models" / "modnet.onnx"

# A commonly-mirrored MODNet photographic-portrait-matting ONNX. Override with --url if you
# prefer a different source you trust; the validation step below works regardless of origin.
DEFAULT_URL = "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx"


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"Downloading:\n  {url}\n-> {out}")

    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = done * 100 // total_size
            sys.stdout.write(f"\r  {done/1e6:6.1f} / {total_size/1e6:6.1f} MB ({pct:3d}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    sys.stdout.write("\n")
    tmp.replace(out)


def _validate(path: Path, ref_size: int = 512) -> bool:
    """Confirm the file is a working MODNet-style ONNX: single image input, single 2D alpha
    output, values in [0, 1] on a dummy frame."""
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"  ! Cannot validate (missing dep): {exc}")
        print("    Install requirements/optional-model.txt, then re-run to validate.")
        return False

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"  ! Failed to load as ONNX: {exc}")
        return False

    inputs = sess.get_inputs()
    if len(inputs) != 1:
        print(f"  ! Expected 1 input, model has {len(inputs)} (not a standard MODNet ONNX).")
        return False

    rh = rw = ref_size - (ref_size % 32)
    dummy = ((np.random.rand(1, 3, rh, rw).astype(np.float32)) - 0.5) / 0.5
    try:
        out = sess.run(None, {inputs[0].name: dummy})[0]
    except Exception as exc:
        print(f"  ! Inference failed: {exc}")
        return False

    alpha = np.squeeze(np.asarray(out, dtype=np.float32))
    if alpha.ndim != 2:
        print(f"  ! Output is not a 2D alpha matte (got shape {np.asarray(out).shape}).")
        return False
    lo, hi = float(alpha.min()), float(alpha.max())
    if not (-0.01 <= lo and hi <= 1.01):
        print(f"  ! Alpha out of [0,1] range (min={lo:.3f}, max={hi:.3f}).")
        return False

    print(f"  OK: input '{inputs[0].name}', alpha {alpha.shape}, range [{lo:.3f}, {hi:.3f}]")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download + validate the MODNet matting model")
    parser.add_argument("--url", default=DEFAULT_URL, help="MODNet ONNX URL")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path (default models/modnet.onnx)")
    parser.add_argument("--skip-download", action="store_true", help="Only validate an existing file")
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser().resolve()

    if not args.skip_download:
        try:
            _download(args.url, out)
        except Exception as exc:
            print(f"Download failed: {exc}")
            print("Tip: pass --url with a MODNet ONNX source you trust, or download manually to", out)
            return 1
    elif not out.exists():
        print(f"No file at {out} to validate.")
        return 1

    size_mb = out.stat().st_size / 1e6
    print(f"Validating {out.name} ({size_mb:.1f} MB)...")
    if not _validate(out):
        print("Validation FAILED. The app will keep using the selfie+guided-filter fallback.")
        return 2

    print("\nDone. The app will now use MODNet matting for Subjects/Background.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
