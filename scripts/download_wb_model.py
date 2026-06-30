#!/usr/bin/env python3
"""Download (optional) and validate an Auto WB (AI) illuminant-estimation ONNX model.

Unlike the matting model there is no single canonical white-balance ONNX, so this script is
validator-first: point --url at a model you trust (FC4 / illuminant-estimation family), or
drop the file at models/white_balance.onnx yourself and run with --skip-download to validate.

It confirms the model loads and that its output reduces to a finite, positive 3-vector
illuminant on a dummy frame -- the contract WhiteBalanceEstimator expects.

Usage:
    python scripts/download_wb_model.py --url <URL>     # download + validate
    python scripts/download_wb_model.py --skip-download  # validate models/white_balance.onnx
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "models" / "white_balance.onnx"


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


def _validate(path: Path, ref_size: int = 512) -> bool:
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

    # Match WhiteBalanceEstimator's layout inference: channel-first unless shape says NHWC.
    shape = inputs[0].shape
    layout = "nhwc" if (len(shape) == 4 and shape[3] == 3 and shape[1] != 3) else "nchw"
    if layout == "nhwc":
        dummy = np.random.rand(1, ref_size, ref_size, 3).astype(np.float32)
    else:
        dummy = np.random.rand(1, 3, ref_size, ref_size).astype(np.float32)

    try:
        out = sess.run(None, {inputs[0].name: dummy})[0]
    except Exception as exc:
        print(f"  ! Inference failed: {exc}")
        return False

    a = np.squeeze(np.asarray(out, dtype=np.float32))
    spatial = (ref_size, ref_size)
    # Image-to-image AWB model (e.g. Deep-WB): output is a corrected image, not a vector.
    if a.ndim == 3 and ((a.shape[0] == 3 and a.shape[1:] == spatial) or (a.shape[2] == 3 and a.shape[:2] == spatial)):
        if not np.isfinite(a).all():
            print("  ! Image output has non-finite values.")
            return False
        print(f"  OK: input '{inputs[0].name}' ({layout}), image-to-image AWB output {a.shape}")
        return True

    # Illuminant-estimation model: output reduces to a 3-vector.
    if a.ndim == 1 and a.size == 3:
        v = a
    elif a.ndim >= 2 and 3 in a.shape:
        ax = next((i for i, n in enumerate(a.shape) if n == 3), None)
        v = a.mean(axis=tuple(i for i in range(a.ndim) if i != ax)) if ax is not None else None
    elif a.size == 3:
        v = a.reshape(3)
    else:
        v = None

    if v is None or np.asarray(v).size != 3:
        print(f"  ! Output is neither a corrected image nor a 3-vector illuminant (shape {np.asarray(out).shape}).")
        return False
    v = np.abs(np.asarray(v, dtype=np.float32))
    if not np.isfinite(v).all() or float(v.sum()) <= 0:
        print(f"  ! Illuminant is not finite/positive: {v.tolist()}")
        return False

    print(f"  OK: input '{inputs[0].name}' ({layout}), illuminant ~ {[round(float(x),3) for x in v]}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download + validate an Auto WB (AI) ONNX model")
    parser.add_argument("--url", default="", help="URL of an illuminant-estimation ONNX (FC4-style)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path (default models/white_balance.onnx)")
    parser.add_argument("--skip-download", action="store_true", help="Only validate an existing file")
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser().resolve()

    if not args.skip_download:
        if not args.url:
            print("No --url given. Either pass a trusted MODNet-style WB ONNX URL, or place the")
            print(f"file at {out} and re-run with --skip-download to validate it.")
            print("\nThe model must take one RGB image (NCHW/NHWC, [0,1]) and output a 3-vector illuminant.")
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
        print("Validation FAILED. Auto WB (AI) will fall back to Gray World.")
        return 2
    print("\nDone. The 'Auto AI' white-balance button will now use this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
