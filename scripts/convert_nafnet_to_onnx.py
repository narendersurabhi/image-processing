#!/usr/bin/env python3
"""Convert NAFNet-SIDD-width64 (Chen et al., ECCV 2022) to ONNX for the Deep Denoise action.

NAFNet is trained on SIDD *real* camera-noise pairs, so it cleans real sensor noise far
better than the AWGN-trained DnCNN -- but it's heavy (~116M params, ~5 min per 24MP image),
which is why it backs the opt-in "Deep Denoise" button rather than the live slider.

Produces models/deep_denoise_b2.onnx at a fixed batch of 2 and 256x256 input
(DeepDenoiser tiles the image at 256 with a halo). Use ``--batch-size 1`` for the old
single-tile model, or ``--dynamic-batch`` to export a model that can run a configurable
number of tiles per ONNX call. torch is only needed for this one-time export.

    pip install torch onnx onnxscript gdown          # build-time only
    python scripts/convert_nafnet_to_onnx.py
    python scripts/convert_nafnet_to_onnx.py --batch-size 1 --out models/deep_denoise.onnx
    python scripts/convert_nafnet_to_onnx.py --dynamic-batch --out models/deep_denoise_dynamic.onnx

Arch is fetched from megvii-research/NAFNet; weights from the official Google Drive release.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

ARCH_BASE = "https://raw.githubusercontent.com/megvii-research/NAFNet/main/basicsr/models/archs"
WEIGHTS_GDRIVE_ID = "14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR"  # NAFNet-SIDD-width64.pth
OUT = Path(__file__).resolve().parents[1] / "models" / "deep_denoise_b2.onnx"


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert NAFNet-SIDD-width64 to ONNX for Deep Denoise")
    parser.add_argument("--out", default=str(OUT), help="Output ONNX path")
    parser.add_argument("--batch-size", type=int, default=2, help="Fixed ONNX batch size for tile inference")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export symbolic batch dimension instead of a fixed batch size",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Use an existing NAFNet-SIDD-width64 .pth file instead of downloading it",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import torch
        import onnx
        import gdown
    except Exception as exc:
        print(f"Need torch + onnx + gdown for export: {exc}")
        print("  pip install torch onnx onnxscript gdown")
        return 1

    out_path = Path(args.out).expanduser().resolve()
    batch_size = max(1, int(args.batch_size))

    work = Path(tempfile.mkdtemp(prefix="nafnet_"))
    pkg = work / "basicsr" / "models" / "archs"
    pkg.mkdir(parents=True)
    for d in (work / "basicsr", work / "basicsr" / "models", pkg):
        (d / "__init__.py").write_text("")
    # arch_util imports basicsr.utils.get_root_logger -- stub it (we don't fetch all of basicsr).
    utils = types.ModuleType("basicsr.utils")
    utils.get_root_logger = lambda *a, **k: logging.getLogger("nafnet")
    sys.modules["basicsr.utils"] = utils

    print("Fetching architecture + weights...")
    for f in ("NAFNet_arch.py", "arch_util.py", "local_arch.py"):
        urllib.request.urlretrieve(f"{ARCH_BASE}/{f}", pkg / f)
    if args.weights:
        wpath = Path(args.weights).expanduser().resolve()
        if not wpath.exists():
            print(f"Weights file not found: {wpath}")
            return 1
    else:
        wpath = work / "NAFNet-SIDD-width64.pth"
        gdown.download(id=WEIGHTS_GDRIVE_ID, output=str(wpath), quiet=True)

    sys.path.insert(0, str(work))
    from basicsr.models.archs.NAFNet_arch import NAFNet  # type: ignore

    net = NAFNet(img_channel=3, width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
    sd = torch.load(str(wpath), map_location="cpu")
    sd = sd.get("params", sd)  # basicsr wraps weights under 'params'
    net.load_state_dict(sd, strict=True)
    net.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_label = "dynamic batch" if args.dynamic_batch else f"fixed batch {batch_size}"
    print(f"Exporting ONNX ({batch_label}, 256x256 tiles)...")
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "n"}, "output": {0: "n"}}
    torch.onnx.export(
        net,
        torch.rand(batch_size, 3, 256, 256),
        str(out_path),
        input_names=["input"], output_names=["output"], opset_version=18,
        dynamic_axes=dynamic_axes,
    )
    model = onnx.load(str(out_path))
    onnx.save(model, str(out_path), save_as_external_data=False)
    sidecar = out_path.with_suffix(out_path.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()

    print(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
