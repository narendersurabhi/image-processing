#!/usr/bin/env python3
"""Download and validate the optional BRIA RMBG-2.0 background-removal model.

RMBG-2.0 is a gated Hugging Face model with non-commercial weight terms. This script requires an
explicit acknowledgement before downloading. Runtime inference is local-only; the app never
downloads this model implicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ID = "briaai/RMBG-2.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "models" / "RMBG-2.0"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download/validate BRIA RMBG-2.0 model assets.")
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face model repo.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output model folder.")
    parser.add_argument("--skip-download", action="store_true", help="Only validate an existing local folder.")
    parser.add_argument(
        "--i-understand-license",
        action="store_true",
        help="Required for download: acknowledge BRIA RMBG-2.0 weight/license terms.",
    )
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser().resolve()
    if not args.skip_download:
        if not args.i_understand_license:
            print("RMBG-2.0 weights are gated and licensed for non-commercial use unless separately licensed.")
            print("Review the model card/license, log in with huggingface-cli if needed, then rerun with:")
            print("  python scripts/download_rmbg_model.py --i-understand-license")
            return 2
        if not _download(args.repo_id, out):
            return 1
    elif not out.exists():
        print(f"No model folder at {out} to validate.")
        return 1

    if not _validate(out):
        print("Validation FAILED. The app will keep using MODNet/MediaPipe/heuristic fallback.")
        return 3

    print("\nDone. The app will now prefer RMBG-2.0 for Subjects/Background.")
    print("Disable with PORTRAIT_DISABLE_RMBG=1.")
    return 0


def _download(repo_id: str, out: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: pip install huggingface-hub")
        return False

    out.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading {repo_id} -> {out}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(out),
            local_dir_use_symlinks=False,
            ignore_patterns=("*.onnx", "*.tflite", "*.bin"),
        )
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("If this is an access error, accept the model terms on Hugging Face and run `huggingface-cli login`.")
        return False
    return True


def _validate(path: Path) -> bool:
    required = ("config.json",)
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        print(f"Missing expected model files in {path}: {', '.join(missing)}")
        return False

    try:
        import numpy as np
        from portrait_enhancer.core.segmentation import BackgroundRemovalSegmenter
    except Exception as exc:
        print(f"Cannot import validator deps: {exc}")
        return False

    old_env = {}
    import os

    for key, value in {
        "PORTRAIT_RMBG_MODEL": str(path),
        "PORTRAIT_RMBG_DEVICE": "cpu",
    }.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    old_disable = os.environ.pop("PORTRAIT_DISABLE_RMBG", None)
    try:
        seg = BackgroundRemovalSegmenter()
        if not seg.available:
            print(f"RMBG did not initialize: {seg.reason_unavailable}")
            print("Install runtime deps: pip install transformers kornia torchvision")
            return False
        test = np.zeros((96, 128, 3), dtype=np.float32)
        test[24:72, 40:88, :] = 1.0
        mask = seg.segment(test)
        if mask is None or mask.shape != test.shape[:2]:
            print("RMBG inference did not return a same-size mask.")
            return False
        lo, hi = float(mask.min()), float(mask.max())
        if not (-0.01 <= lo <= 1.01 and -0.01 <= hi <= 1.01):
            print(f"RMBG mask range is invalid: [{lo:.3f}, {hi:.3f}]")
            return False
        print(f"OK: {path} | backend={seg.backend} device={seg.execution_provider} range=[{lo:.3f}, {hi:.3f}]")
        return True
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_disable is not None:
            os.environ["PORTRAIT_DISABLE_RMBG"] = old_disable


if __name__ == "__main__":
    raise SystemExit(main())
